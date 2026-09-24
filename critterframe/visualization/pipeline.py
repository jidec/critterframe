"""
Every activity's diagnostics: resolve_sample, Report, NullReport, PanelFanout, open_report.

A report is one flat stem in `visualizations/pipeline/`: a bounded grid of item panels, optional
checkpoint grids, whole-population figures, and a `.report.json` sidecar. Items are any string key,
e.g. an occurrence id, a scope value, or an event.
"""

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
    """
    Turn a visualize= argument into the ids to build a grid from, or None if nothing is visualized.

    Pass the ids an activity will actually PROCESS: a grid can only show work that happened.

    - `occurrence_ids` -- candidate item keys, in processing order.
    - `visualize` -- False/None for none, True for a default-sized sample (25), an int for that
      many, or an iterable of exactly these ids (ids not in `occurrence_ids` are ignored with a
      warning).

    Returns a list of ids, or None.
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
        logger.warning("visualize= named %d occurrence(s) this run isn't "
                       "processing, ignoring them (e.g. %s)",
                       len(missing), sorted(missing)[0])
    return chosen


def sample_count(visualize):
    """
    The number of items a visualize= argument asks for, or None when it names explicit ids.

    - `visualize` -- True (the default sample size), an int, or an iterable of ids.
    """
    if visualize is True:
        return DEFAULT_SAMPLE
    if isinstance(visualize, int) and not isinstance(visualize, bool):
        return visualize
    return None


class _Sheet:
    """Fitted cells for some items, and whether anything arrived since the last write."""

    def __init__(self):
        self.cells = {}       # item -> {stage: fitted cell}
        self.dirty = False

    def add(self, item, stage, cell):
        self.cells.setdefault(item, {})[stage] = cell
        self.dirty = True

    def clear(self):
        self.cells = {}
        self.dirty = False


class Report:
    """
    The diagnostics of one activity: a bounded grid, checkpoints, figures, and a sidecar.

    Open one with `open_report`, call `begin()` with the items the activity will process, route
    panels in with `sink()`/`collect()`, call `done()` after each item, and `close()` at the end.
    Usable as a context manager, which closes on exit. Panels are fitted to their cell on arrival,
    so memory stays at sample × stages × one cell whatever the activity's size.

    - `project_path` -- project to write into.
    - `name` -- first part of every filename, e.g. a run name or `validate_masks__head`.
    - `identity_hash` -- digest of what produced this output (a recipe hash, an import hash, ...).
    - `part` -- the part concerned, or None. Omitted from filenames when None or the default part.
    - `visualize` -- True, an int, or explicit ids; see `resolve_sample`.
    - `visualize_every` -- write a checkpoint grid every N `done()` calls, each resampled from only
      the items of that window (`__at<N>`), in addition to the main grid.
    - `rank` -- None samples the main grid deterministically. `"lowest"`/`"highest"` instead keeps
      the N items with the lowest/highest value passed to `done()`, e.g. the worst IoUs. Every item
      then draws its panels, so use it where panels are cheap. Ignored for explicit ids.
    - `identity` -- the JSON-serializable spec behind `identity_hash`, recorded in the sidecar.
    - `cell`, `columns` -- grid layout, as in `grids.image_grid`.
    """

    def __init__(self, project_path, name, identity_hash, part=None, visualize=True,
                 visualize_every=None, rank=None, identity=None,
                 cell=grids.DEFAULT_CELL, columns=grids.DEFAULT_COLUMNS):
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

        self._pending = {}        # rank mode: item -> cells awaiting done()
        self._kept = []           # rank mode heap: (key, -seq, item, value)
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
        """True -- a Report exists only when visualization is on (see NullReport)."""
        return True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def identify(self, identity_hash, identity=None):
        """
        Set the identity once it's known, e.g. a hash over what an export actually wrote.

        Call before anything is saved, since every filename carries the hash.

        - `identity_hash` -- the digest to name files with.
        - `identity` -- the spec behind it, recorded in the sidecar.
        """
        self.identity_hash = identity_hash
        if identity is not None:
            self.identity = identity
        return self

    # -- items ---------------------------------------------------------------

    def begin(self, items, eligible=None):
        """
        Declare the items this activity will process, which fixes the sample.

        - `items` -- item keys in processing order; `done()` is expected once per item.
        - `eligible` -- optional subset that can appear in grids, e.g. the occurrences one part of
          a multi-part run still needs. Items outside it still count toward `visualize_every`.

        Returns the report.
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
        """True if a panel for this item would be kept anywhere."""
        item = str(item)
        if self.rank is not None:
            return self._is_eligible(item)
        return item in self._sample_set or item in self._window_set

    def sink(self, item):
        """This report as a Segment's panel_sink when it wants the item, else None."""
        return self if self.wants(item) else None

    def collect(self, item, stage, image):
        """
        Take one display-ready panel for an item; ignored if the item isn't wanted.

        A panel that can't be laid out is dropped with a warning, never raised.

        - `item` -- the item key.
        - `stage` -- the column it belongs to, usually the operation's name.
        - `image` -- uint8 or boolean array.
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
        """
        Mark an item finished: keep it if it ranks, and write checkpoints at window boundaries.

        - `item` -- the item key.
        - `rank_value` -- the value `rank` orders by. An item with none (or a non-finite one) is
          never kept.
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
        """
        Every item this report will want over its whole life, or None if it may want any (rank mode).

        For activities whose items finish out of order, e.g. concurrent downloads: hold on to
        what's needed for these, and offer panels when each item's turn comes.
        """
        if self.rank is not None:
            return None
        wanted = set(self._sample)
        if self.visualize_every:
            for start in range(0, len(self._items), self.visualize_every):
                wanted.update(self._window_sample(start)[0])
        return wanted

    def failure(self, item, error):
        """
        Record a failed item in the sidecar, e.g. a download with no image to show.

        - `item` -- the item key.
        - `error` -- the exception or message.
        """
        self._failed += 1
        if len(self._failures) < MAX_FAILURES:
            self._failures.append({"item": str(item), "error": str(error)})

    # -- output --------------------------------------------------------------

    def rows(self):
        """(row images, row labels) of the main grid as it stands, one row per item."""
        entries = self._final_entries()
        return ([[cells.get(stage) for stage in self._stages] for _item, _label, cells in entries],
                [label for _item, label, _cells in entries])

    def save(self):
        """
        Write the main grid if anything new was collected, and return its path (else None).

        Safe to call repeatedly; it always overwrites the same path.
        """
        if not self._final.dirty:
            return None
        self._final.dirty = False

        entries = self._final_entries()
        if not entries:
            logger.info("no pipeline panels collected for '%s' part '%s' -- nothing in it "
                        "draws one", self.name, self.part)
            return None
        return self._write_grid(entries)

    def checkpoint(self, label):
        """
        Write what the main grid holds now as `__<label>`, then empty it for the next round.

        For repeated passes over the same sample, e.g. validation predictions once per epoch.

        - `label` -- filename suffix, e.g. `epoch0005`.

        Returns the path written, or None if nothing was collected.
        """
        entries = self._final_entries()
        written = self._write_grid(entries, suffix=str(label)) if entries else None
        self._final.clear()
        self._pending = {}
        self._kept = []
        self._kept_cells = {}
        return written

    def figure(self, name, figure):
        """
        Write a whole-population figure as `__<name>.png`.

        Build an expensive figure only `if report:`, so a NullReport costs nothing.

        - `name` -- filename suffix, e.g. `curves`.
        - `figure` -- a matplotlib Figure (see `figures`) or a uint8/boolean image array.

        Returns the path written, or None if writing failed.
        """
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
        dest = paths.pipeline_file_path(self.project_path, self.name, self.identity_hash,
                                        part=self._file_part, suffix=name, extension="png")
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
        """Write the main grid, the trailing checkpoint window, and the sidecar."""
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
        """(item, label, cells) for the main grid, in display order."""
        if self.rank is None:
            return [(item, item, self._final.cells[item]) for item in self._sample
                    if self._final.cells.get(item)]
        return [(item, f"{item} {value:.3g}", self._kept_cells[item])
                for _key, _seq, item, value in self._ordered_kept()]

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
        """(sample, end) for the checkpoint window beginning at item position `start`."""
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
        entries = [(item, item, self._window.cells[item]) for item in self._window_order
                   if self._window.cells.get(item)]
        if not entries:
            return None
        return self._write_grid(entries, suffix=f"at{self._window_end:08d}")

    def _write_grid(self, entries, suffix=None):
        heading = " / ".join(str(piece) for piece in
                             (self.name, self.part, f"n={len(entries)}", self.identity_hash)
                             if piece is not None)
        if suffix:
            heading = f"{heading} / {suffix}"
        if self.rank is not None:
            heading = f"{heading} / {self.rank} first"

        rows = [[cells.get(stage) for stage in self._stages] for _item, _label, cells in entries]
        labels = [label for _item, label, _cells in entries]
        if len(self._stages) == 1:
            grid = grids.image_grid([row[0] for row in rows], labels=labels,
                                    title=f"{heading} / {self._stages[0]}",
                                    columns=self.columns, cell=self.cell)
        else:
            grid = grids.comparison_grid(rows, column_titles=self._stages, row_labels=labels,
                                         title=heading, cell=self.cell)

        dest = paths.pipeline_file_path(self.project_path, self.name, self.identity_hash,
                                        part=self._file_part, suffix=suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), grid, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        logger.info("pipeline grid -> %s", dest)

        self._wrote = True
        self._write_sidecar()
        return dest

    def _write_sidecar(self):
        directory = paths.pipeline_dir(self.project_path)
        stem = paths.pipeline_stem(self.name, self.identity_hash, part=self._file_part)
        sidecar = paths.pipeline_report_path(self.project_path, self.name, self.identity_hash,
                                             part=self._file_part)
        files = sorted(
            path.name for path in directory.glob(f"{glob.escape(stem)}*")
            if path != sidecar and (path.name.startswith(f"{stem}.")
                                    or path.name.startswith(f"{stem}__"))
        )

        if self.rank is None:
            shown = list(self._sample)
        else:
            shown = [{"item": item, "value": value}
                     for _key, _seq, item, value in self._ordered_kept()]

        record = {
            "name": self.name,
            "part": self.part,
            "identity_hash": self.identity_hash,
            "identity": self.identity,
            "visualize": self.visualize,
            "visualize_every": self.visualize_every,
            "rank": self.rank,
            "shown": shown,
            "counts": {"items": len(self._items), "done": self._position,
                       "failed": self._failed},
            "elapsed_s": (None if self._started is None
                          else round(time.monotonic() - self._started, 1)),
            "failures": self._failures,
            "failures_truncated": self._failed > len(self._failures),
            "files": files,
        }
        write_json(sidecar, record)


class NullReport:
    """
    What `open_report` returns when visualization is off: every method is a no-op.

    Lets call sites use a report unconditionally rather than branching on visualize=.
    """

    def __bool__(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def identify(self, identity_hash, identity=None):
        return self

    def begin(self, items, eligible=None):
        return self

    def wants(self, item):
        return False

    def sink(self, item):
        return None

    def collect(self, item, stage, image):
        return None

    panel = collect

    def done(self, item, rank_value=None):
        return None

    def planned(self):
        return set()

    def failure(self, item, error):
        return None

    def rows(self):
        return [], []

    def save(self):
        return None

    def checkpoint(self, label):
        return None

    def figure(self, name, figure):
        return None

    def close(self):
        return None


NULL_REPORT = NullReport()


class PanelFanout:
    """
    A panel sink that hands each panel to several reports at once.

    For a multi-output segmentation run, whose shared steps run once on a segment that then forks
    per part: those panels belong on every part's grid.
    """

    def __init__(self, reports):
        self.reports = list(reports)

    def __bool__(self):
        return any(bool(report) for report in self.reports)

    def wants(self, occurrence_id):
        return any(report.wants(occurrence_id) for report in self.reports)

    def collect(self, occurrence_id, stage, panel):
        for report in self.reports:
            if report.wants(occurrence_id):
                report.collect(occurrence_id, stage, panel)


def open_report(project_path, name, identity_hash, part=None, visualize=True,
                visualize_every=None, rank=None, identity=None):
    """
    The report for one activity, or a NullReport when visualize= is off.

    Every activity opens its diagnostics here, so `visualize=`/`visualize_every=` mean the same
    thing across the package.

    - `project_path`, `name`, `identity_hash`, `part`, `visualize`, `visualize_every`, `rank`,
      `identity` -- as in `Report`.

    Returns a Report or NullReport.
    """
    if visualize is None or visualize is False:
        if visualize_every:
            logger.warning(
                "visualize_every=%d with visualize=%r: there's no report to "
                "checkpoint without visualize, so visualize_every has no effect",
                visualize_every, visualize)
        return NULL_REPORT
    return Report(project_path, name, identity_hash, part=part, visualize=visualize,
                  visualize_every=visualize_every, rank=rank, identity=identity)


def panel_sink(report, occurrence_id):
    """
    What to give one occurrence's Segment as its panel sink: the report when it wants the
    occurrence, None otherwise.

    None keeps panel building off the path of the occurrences nobody is looking at.
    """
    if report is None or not report.wants(occurrence_id):
        return None
    return report
