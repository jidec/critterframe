"""Pipeline diagnostics: open_report, Report, NullReport, PanelFanout, resolve_sample."""

import glob
import heapq
import logging
import math
import re
import time

import cv2
import numpy as np

from ..storage.jsonfiles import write_json
from ..project import paths
from ..recipes import DEFAULT_PART
from ..selectionhelpers import sample_occurrences
from . import grids

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE = 25
JPEG_QUALITY = 88
MAX_FAILURES = 200
RANKS = ("lowest", "highest")


def resolve_sample(occurrence_ids, visualize):
    """Turn a `visualize=` argument into the ids to build a grid from.

    Args:
        occurrence_ids: The item keys the activity will process, in order.
        visualize: False or None for none, True for the default sample of 25, an int for
            that many, or an iterable of ids. Ids not in `occurrence_ids` are ignored with
            a warning.

    Returns:
        A list of ids, or None if nothing is visualized.
    """
    if visualize is None or visualize is False:
        return None

    ids = [str(occurrence_id) for occurrence_id in occurrence_ids]
    count = sample_count(visualize)
    if count is not None:
        return sample_occurrences(ids, count)

    wanted = {str(occurrence_id) for occurrence_id in visualize}
    chosen = [occurrence_id for occurrence_id in ids if occurrence_id in wanted]
    missing = wanted - set(chosen)
    if missing:
        logger.warning(
            "visualize= named %d occurrence(s) this run isn't processing, ignoring them (e.g. %s)",
            len(missing),
            sorted(missing)[0],
        )
    return chosen


def sample_count(visualize):
    """Return how many items a `visualize=` argument asks for, or None when it names ids.

    Args:
        visualize: True, an int, or an iterable of ids.
    """
    if visualize is True:
        return DEFAULT_SAMPLE
    if isinstance(visualize, int) and not isinstance(visualize, bool):
        return visualize
    return None


class _Sheet:
    """Fitted cells for some items, and whether anything arrived since the last write."""

    def __init__(self):
        self.cells = {}  # item -> {stage: fitted cell}
        self.dirty = False

    def add(self, item, stage, cell):
        self.cells.setdefault(item, {})[stage] = cell
        self.dirty = True

    def clear(self):
        self.cells = {}
        self.dirty = False


class Report:
    """The diagnostics of one activity: a bounded grid, checkpoints, figures and a sidecar.

    Open one with `open_report`, call `begin()` with the items to process, route panels in
    through `sink()` or `collect()`, call `done()` after each item and `close()` at the
    end. Also a context manager, which closes on exit.

    Args:
        project_path: Project to write into.
        name: First part of every filename, e.g. a run name or `validate_masks__head`.
        identity_hash: Digest of what produced the output, e.g. a recipe hash.
        part: The part concerned; left out of filenames when None or the default part.
        visualize: True, an int, or ids, as in `resolve_sample`.
        visualize_every: Also write a checkpoint grid every N `done()` calls, sampled from
            that window's items.
        rank: None samples the grid deterministically. `"lowest"` or `"highest"` keeps the
            N items with the lowest or highest value passed to `done()`. Every item then
            draws its panels.
        identity: The JSON-serializable spec behind `identity_hash`, for the sidecar.
        cell: `(height, width)` of one grid cell.
        columns: Images per row, where the grid has no stage columns.
    """

    def __init__(
        self,
        project_path,
        name,
        identity_hash,
        part=None,
        visualize=True,
        visualize_every=None,
        rank=None,
        identity=None,
        cell=grids.DEFAULT_CELL,
        columns=grids.DEFAULT_COLUMNS,
    ):
        if rank is not None and rank not in RANKS:
            raise ValueError(f"rank must be one of {RANKS} or None, got {rank!r}")
        if sample_count(visualize) is None:
            visualize = [str(item) for item in visualize]

        self.project_path = project_path
        self.name = name
        self.identity_hash = identity_hash
        self.part = part
        self.visualize = visualize
        self.visualize_every = visualize_every or None
        self.rank = rank if sample_count(visualize) is not None else None
        self.identity = identity
        self.cell = cell
        self.columns = columns

        self._items = []
        self._eligible = None
        self._position = 0
        self._stages = []

        self._sample = []
        self._sample_set = set()
        self._final = _Sheet()

        self._pending = {}  # rank mode: item -> cells awaiting done()
        self._kept = []  # rank mode heap: (key, -seq, item, value)
        self._kept_cells = {}
        self._seq = 0

        self._window = _Sheet()
        self._window_order = []
        self._window_set = set()
        self._window_end = None

        self._failures = []
        self._failed = 0
        self._wrote = False
        self._started = None

    def __bool__(self):
        """Return True: a Report exists only when visualization is on."""
        return True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def identify(self, identity_hash, identity=None):
        """Set the identity once it is known, before anything is saved.

        Args:
            identity_hash: The digest to name files with.
            identity: The spec behind it, for the sidecar.
        """
        self.identity_hash = identity_hash
        if identity is not None:
            self.identity = identity
        return self

    # -- items ---------------------------------------------------------------

    def begin(self, items, eligible=None):
        """Declare the items the activity will process, which fixes the sample.

        Args:
            items: Item keys in processing order.
            eligible: The items that may appear in grids. Others still count toward
                `visualize_every`.

        Returns:
            The report.
        """
        self._items = [str(item) for item in items]
        self._eligible = None if eligible is None else {str(item) for item in eligible}
        self._position = 0
        self._started = time.monotonic()

        if self.rank is None:
            candidates = [item for item in self._items if self._is_eligible(item)]
            self._sample = resolve_sample(candidates, self.visualize) or []
            self._sample_set = set(self._sample)

        self._rotate(0)
        return self

    def wants(self, item):
        """Return whether a panel for this item would be kept."""
        item = str(item)
        if self.rank is not None:
            return self._is_eligible(item)
        return item in self._sample_set or item in self._window_set

    def sink(self, item):
        """Return the report as a panel sink when it wants the item, else None."""
        return self if self.wants(item) else None

    def collect(self, item, stage, image):
        """Take one display-ready panel for an item.

        A panel for an unwanted item is ignored, and one that can't be laid out is dropped
        with a warning.

        Args:
            item: The item key.
            stage: The column it belongs to, usually the operation's name.
            image: uint8 or boolean array.
        """
        item = str(item)
        in_final = item in self._sample_set
        in_window = item in self._window_set
        ranked = self.rank is not None and self._is_eligible(item)
        if not (in_final or in_window or ranked):
            return

        try:
            cell = grids.fit_cell(image, cell=self.cell)
        except Exception as exc:
            logger.warning("could not lay out '%s' panel for %s: %s", stage, item, exc)
            return

        if stage not in self._stages:
            self._stages.append(stage)
        if in_final:
            self._final.add(item, stage, cell)
        if ranked:
            self._pending.setdefault(item, {})[stage] = cell
        if in_window:
            self._window.add(item, stage, cell)

    panel = collect

    def done(self, item, rank_value=None):
        """Mark an item finished, and write a checkpoint at a window boundary.

        Args:
            item: The item key.
            rank_value: The value `rank` orders by. An item with none, or a non-finite one,
                is never kept.
        """
        item = str(item)
        if self.rank is not None:
            cells = self._pending.pop(item, None)
            if cells and rank_value is not None and math.isfinite(float(rank_value)):
                self._keep(item, float(rank_value), cells)

        self._position += 1
        if self.visualize_every and self._position % self.visualize_every == 0:
            self.save()
            self._save_window()
            self._rotate(self._position)

    def planned(self):
        """Return every item the report will want, or None if it may want any (rank mode)."""
        if self.rank is not None:
            return None
        wanted = set(self._sample)
        if self.visualize_every:
            for start in range(0, len(self._items), self.visualize_every):
                wanted.update(self._window_sample(start)[0])
        return wanted

    def failure(self, item, error):
        """Record a failed item in the sidecar.

        Args:
            item: The item key.
            error: The exception or message.
        """
        self._failed += 1
        if len(self._failures) < MAX_FAILURES:
            self._failures.append({"item": str(item), "error": str(error)})

    # -- output --------------------------------------------------------------

    def rows(self):
        """Return `(row images, row labels)` of the main grid as it stands."""
        entries = self._final_entries()
        return (
            [[cells.get(stage) for stage in self._stages] for _item, _label, cells in entries],
            [label for _item, label, _cells in entries],
        )

    def save(self):
        """Write the main grid if anything new was collected, and return its path or None."""
        if not self._final.dirty:
            return None
        self._final.dirty = False

        entries = self._final_entries()
        if not entries:
            logger.info(
                "no pipeline panels collected for '%s' part '%s' -- nothing in it draws one",
                self.name,
                self.part,
            )
            return None
        return self._write_grid(entries)

    def checkpoint(self, label):
        """Write what the main grid holds as `__<label>`, then empty it for the next round.

        Args:
            label: Filename suffix, e.g. `epoch0005`.

        Returns:
            The path written, or None if nothing was collected.
        """
        entries = self._final_entries()
        written = self._write_grid(entries, suffix=str(label)) if entries else None
        self._final.clear()
        self._pending = {}
        self._kept = []
        self._kept_cells = {}
        return written

    def figure(self, name, figure):
        """Write a whole-population figure as `__<name>.png`.

        Args:
            name: Filename suffix, e.g. `curves`.
            figure: A matplotlib Figure, or a uint8 or boolean image array.

        Returns:
            The path written, or None if writing failed.
        """
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
        dest = paths.pipeline_file_path(
            self.project_path,
            self.name,
            self.identity_hash,
            part=self._file_part,
            suffix=name,
            extension="png",
        )
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(figure, np.ndarray):
                image = figure.astype(np.uint8) * 255 if figure.dtype == bool else figure
                if not cv2.imwrite(str(dest), image):
                    raise ValueError("cv2 could not encode the image")
            else:
                from . import figures

                figures.save_figure(figure, dest)
        except Exception as exc:
            logger.warning("could not write figure '%s' for '%s': %s", name, self.name, exc)
            return None

        logger.info("pipeline figure -> %s", dest)
        self._wrote = True
        self._write_sidecar()
        return dest

    def close(self):
        """Write the main grid, the last checkpoint window and the sidecar."""
        self.save()
        self._save_window()
        if self._wrote or self._failed:
            self._write_sidecar()

    # -- internals -----------------------------------------------------------

    @property
    def _file_part(self):
        return None if self.part in (None, DEFAULT_PART) else self.part

    def _is_eligible(self, item):
        return self._eligible is None or item in self._eligible

    def _keep(self, item, value, cells):
        limit = sample_count(self.visualize)
        self._seq += 1
        key = -value if self.rank == "lowest" else value
        heapq.heappush(self._kept, (key, -self._seq, item, value))
        self._kept_cells[item] = cells
        if len(self._kept) > limit:
            _key, _seq, evicted, _value = heapq.heappop(self._kept)
            self._kept_cells.pop(evicted, None)
            if evicted == item:
                return
        self._final.dirty = True

    def _ordered_kept(self):
        if self.rank == "lowest":
            return sorted(self._kept, key=lambda entry: (entry[3], -entry[1]))
        return sorted(self._kept, key=lambda entry: (-entry[3], -entry[1]))

    def _final_entries(self):
        """Return `(item, label, cells)` for the main grid, in display order."""
        if self.rank is None:
            return [
                (item, item, self._final.cells[item]) for item in self._sample if self._final.cells.get(item)
            ]
        return [
            (item, f"{item} {value:.3g}", self._kept_cells[item])
            for _key, _seq, item, value in self._ordered_kept()
        ]

    def _rotate(self, start):
        self._window = _Sheet()
        self._window_order = []
        self._window_set = set()
        self._window_end = None
        if not self.visualize_every or start >= len(self._items):
            return

        sample, end = self._window_sample(start)
        self._window_order = sample
        self._window_set = set(sample)
        self._window_end = end

    def _window_sample(self, start):
        """Return `(sample, end)` for the checkpoint window starting at item position `start`."""
        end = min(start + self.visualize_every, len(self._items))
        ids = [item for item in self._items[start:end] if self._is_eligible(item)]
        count = sample_count(self.visualize)
        if count is None:
            wanted = set(self.visualize)
            return [item for item in ids if item in wanted], end
        return sample_occurrences(ids, count), end

    def _save_window(self):
        if not self._window.dirty or self._window_end is None:
            return None
        self._window.dirty = False
        entries = [
            (item, item, self._window.cells[item])
            for item in self._window_order
            if self._window.cells.get(item)
        ]
        if not entries:
            return None
        return self._write_grid(entries, suffix=f"at{self._window_end:08d}")

    def _write_grid(self, entries, suffix=None):
        heading = " / ".join(
            str(piece)
            for piece in (self.name, self.part, f"n={len(entries)}", self.identity_hash)
            if piece is not None
        )
        if suffix:
            heading = f"{heading} / {suffix}"
        if self.rank is not None:
            heading = f"{heading} / {self.rank} first"

        rows = [[cells.get(stage) for stage in self._stages] for _item, _label, cells in entries]
        labels = [label for _item, label, _cells in entries]
        if len(self._stages) == 1:
            grid = grids.image_grid(
                [row[0] for row in rows],
                labels=labels,
                title=f"{heading} / {self._stages[0]}",
                columns=self.columns,
                cell=self.cell,
            )
        else:
            grid = grids.comparison_grid(
                rows, column_titles=self._stages, row_labels=labels, title=heading, cell=self.cell
            )

        dest = paths.pipeline_file_path(
            self.project_path, self.name, self.identity_hash, part=self._file_part, suffix=suffix
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), grid, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        logger.info("pipeline grid -> %s", dest)

        self._wrote = True
        self._write_sidecar()
        return dest

    def _write_sidecar(self):
        directory = paths.pipeline_dir(self.project_path)
        stem = paths.pipeline_stem(self.name, self.identity_hash, part=self._file_part)
        sidecar = paths.pipeline_report_path(
            self.project_path, self.name, self.identity_hash, part=self._file_part
        )
        files = sorted(
            path.name
            for path in directory.glob(f"{glob.escape(stem)}*")
            if path != sidecar and (path.name.startswith(f"{stem}.") or path.name.startswith(f"{stem}__"))
        )

        if self.rank is None:
            shown = list(self._sample)
        else:
            shown = [{"item": item, "value": value} for _key, _seq, item, value in self._ordered_kept()]

        record = {
            "name": self.name,
            "part": self.part,
            "identity_hash": self.identity_hash,
            "identity": self.identity,
            "visualize": self.visualize,
            "visualize_every": self.visualize_every,
            "rank": self.rank,
            "shown": shown,
            "counts": {"items": len(self._items), "done": self._position, "failed": self._failed},
            "elapsed_s": (None if self._started is None else round(time.monotonic() - self._started, 1)),
            "failures": self._failures,
            "failures_truncated": self._failed > len(self._failures),
            "files": files,
        }
        write_json(sidecar, record)


class NullReport:
    """The report `open_report` returns when visualization is off: every method does nothing."""

    def __bool__(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def identify(self, identity_hash, identity=None):
        """Do nothing, and return the report."""
        return self

    def begin(self, items, eligible=None):
        """Do nothing, and return the report."""
        return self

    def wants(self, item):
        """Return False: nothing is wanted."""
        return False

    def sink(self, item):
        """Return None: no panel sink."""
        return None

    def collect(self, item, stage, image):
        """Discard the panel."""
        return None

    panel = collect

    def done(self, item, rank_value=None):
        """Do nothing."""
        return None

    def planned(self):
        """Return an empty set."""
        return set()

    def failure(self, item, error):
        """Do nothing."""
        return None

    def rows(self):
        """Return no rows and no labels."""
        return [], []

    def save(self):
        """Write nothing."""
        return None

    def checkpoint(self, label):
        """Write nothing."""
        return None

    def figure(self, name, figure):
        """Write nothing."""
        return None

    def close(self):
        """Do nothing."""
        return None


NULL_REPORT = NullReport()


class PanelFanout:
    """A panel sink that hands each panel to several reports.

    Args:
        reports: The reports to hand panels to.
    """

    def __init__(self, reports):
        self.reports = list(reports)

    def __bool__(self):
        return any(bool(report) for report in self.reports)

    def wants(self, occurrence_id):
        """Return whether any of the reports wants the occurrence."""
        return any(report.wants(occurrence_id) for report in self.reports)

    def collect(self, occurrence_id, stage, panel):
        """Hand the panel to every report that wants the occurrence."""
        for report in self.reports:
            if report.wants(occurrence_id):
                report.collect(occurrence_id, stage, panel)


def open_report(
    project_path,
    name,
    identity_hash,
    part=None,
    visualize=True,
    visualize_every=None,
    rank=None,
    identity=None,
):
    """Return the report for one activity, or a `NullReport` when `visualize` is off.

    Args:
        project_path: As in `Report`.
        name: As in `Report`.
        identity_hash: As in `Report`.
        part: As in `Report`.
        visualize: As in `Report`.
        visualize_every: As in `Report`.
        rank: As in `Report`.
        identity: As in `Report`.
    """
    if visualize is None or visualize is False:
        if visualize_every:
            logger.warning(
                "visualize_every=%d with visualize=%r: there's no report to "
                "checkpoint without visualize, so visualize_every has no effect",
                visualize_every,
                visualize,
            )
        return NULL_REPORT
    return Report(
        project_path,
        name,
        identity_hash,
        part=part,
        visualize=visualize,
        visualize_every=visualize_every,
        rank=rank,
        identity=identity,
    )


def panel_sink(report, occurrence_id):
    """Return the report as an occurrence's panel sink when it wants the occurrence, else None."""
    if report is None or not report.wants(occurrence_id):
        return None
    return report
