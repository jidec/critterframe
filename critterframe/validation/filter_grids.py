"""
Image grids for filter calibration and audit: which labelled items a filter set catches, misses and costs.

Drawn from the labelled rows only and capped, so they stay bounded whatever the project's size.
"""

import logging

import numpy as np
import pandas as pd

from ..records.occurrences import ID_COL
from ..segments import iterate_segments
from ..selectionhelpers import sample_occurrences
from ..visualization import grids
from ..visualization.panels import ADDED_COLOR, REMOVED_COLOR, annotate, bordered

logger = logging.getLogger(__name__)

# One image cell, before its caption strip. Smaller than a pipeline grid's, since
# a row here is up to ROW_CELLS wide.
CELL = (160, 160)

# Items per row of a row-per-group grid, and in a grid that shows one group in
# full. Past either, a deterministic sample is drawn and the title says so.
ROW_CELLS = 10
FULL_GRID_CELLS = 40

# Items in one score strip. Past this the strip is thinned evenly by rank.
STRIP_CELLS = 40

GOOD_COLOR = ADDED_COLOR
BAD_COLOR = REMOVED_COLOR
BORDER = 5

# The comparators a swept threshold is returned with, and which side of an
# ascending strip each one removes.
_REMOVES_LOW = {">=": True, ">": True, "<=": False, "<": False}


def short_name(column):
    """An export column without its run and part, e.g. `mask_info__segment__score`."""
    return str(column).split("__", 2)[-1]


def thin_ranks(count, cap, keep=()):
    """
    Which of `count` ranked items to show when only `cap` fit: evenly spaced, plus `keep`.

    - `count` -- how many items there are, in rank order.
    - `cap` -- the most to show.
    - `keep` -- ranks that must be shown, e.g. the items either side of a cutoff.

    Returns sorted ranks, at most `cap` of them unless `keep` alone is longer.
    """
    keep = sorted({rank for rank in keep if 0 <= rank < count})
    if count <= cap:
        return list(range(count))
    spaced = np.linspace(0, count - 1, max(0, cap - len(keep))).round().astype(int)
    chosen = sorted(set(keep) | set(spaced.tolist()))
    # Rounding can land a spaced rank on a kept one; never more than the cap.
    while len(chosen) > max(cap, len(keep)):
        chosen.remove(next(rank for rank in chosen if rank not in keep))
    return chosen


def _cell(image, bad, caption):
    """One cutout fitted to a cell, framed by its label's colour, captioned underneath."""
    fitted = grids.fit_cell(image, cell=CELL)
    framed = bordered(fitted, BAD_COLOR if bad else GOOD_COLOR, BORDER)
    return np.vstack([framed, grids._text_strip(caption, CELL[1], grids.LABEL_HEIGHT)])


def _text_cell(lines):
    """A cell holding only text, the size of a captioned cell: a row heading or a cutoff marker."""
    cell = np.full((CELL[0] + grids.LABEL_HEIGHT, CELL[1], 3), grids.BACKGROUND, np.uint8)
    for line, text in enumerate(lines):
        annotate(cell, text, line=line, color=(255, 255, 255))
    return cell


def _wrap(text, width=22):
    """`text` broken into lines a text cell can hold."""
    text = str(text)
    return [text[start:start + width] for start in range(0, len(text), width)] or [""]


def _cutouts(project_path, part, occurrence_ids):
    """`{occurrence_id: masked cutout}` for the ids that have an image and a mask for `part`."""
    wanted = sorted({str(occurrence_id) for occurrence_id in occurrence_ids})
    if not wanted:
        return {}
    cells = {}
    try:
        for occurrence_id, segment in iterate_segments(
                project_path, part=part, occurrence_ids=wanted,
                progress="filter grids"):
            cells[occurrence_id] = grids.mask_cutout(segment)
    except Exception as exc:
        # No image store, no mask table: a project that can't show pictures
        # still gets its filters and its charts.
        logger.info("no images to draw filter grids from: %s", exc)
    return cells


def _showing(shown, total):
    return "" if shown >= total else f" (showing {shown} of {total})"


class _Plan:
    """What every grid will show, decided before any image is read so the store is read once."""

    def __init__(self, labelled, label_col, bad_labels):
        self.ids = labelled[ID_COL].astype(str).to_numpy()
        self.labels = dict(zip(self.ids, labelled[label_col].astype(str)))
        self.is_bad = dict(zip(self.ids, labelled[label_col].isin(bad_labels)))
        self.needed = set()

    def take(self, ids, cap):
        """Up to `cap` of `ids`, the same ones every run, and remembered as needing a cutout."""
        ids = [str(occurrence_id) for occurrence_id in ids]
        chosen = sample_occurrences(ids, cap) if len(ids) > cap else sorted(ids)
        self.needed.update(chosen)
        return chosen

    def bad_first(self, ids, cap):
        """Up to `cap` of `ids`, bad ones first, so the evidence against a group is what's visible."""
        ids = sorted(str(occurrence_id) for occurrence_id in ids)
        ordered = ([i for i in ids if self.is_bad[i]] + [i for i in ids if not self.is_bad[i]])[:cap]
        self.needed.update(ordered)
        return ordered


def draw_filter_grids(report, project_path, part, labelled, label_col, bad_labels,
                      filters, kept, passes, strips=None, categories=None):
    """
    Write the image grids of one calibration or audit into its report.

    - `report` -- the pipeline report to write into; nothing is drawn for a NullReport.
    - `project_path`, `part` -- where the cutouts come from: the part the labels describe.
    - `labelled` -- wide frame of the labelled rows, with `occurrence_id`.
    - `label_col`, `bad_labels` -- the label column, and the labels that count as bad.
    - `filters` -- the filter set, as `export_metrics(filters=...)`.
    - `kept` -- boolean mask over `labelled`: rows passing every filter.
    - `passes` -- `{column: boolean mask}`: rows passing each filter alone.
    - `strips` -- `{column: name}` for the continuous candidates to draw a score strip for.
    - `categories` -- `{column: (name, table, dropped)}` for categorical candidates: the
      per-category table of the sweep and the categories dropped.

    Writes `outcomes`, `misses`, `cost`, `only_here`, `<name>__strip` and
    `<name>__categories`, each only where it has something to show.
    """
    if not report or labelled.empty:
        return

    plan = _Plan(labelled, label_col, bad_labels)
    kept = np.asarray(kept, bool)
    bad = np.array([plan.is_bad[occurrence_id] for occurrence_id in plan.ids], bool)

    outcomes = {
        "bad, removed": plan.ids[bad & ~kept],
        "bad, kept": plan.ids[bad & kept],
        "good, removed": plan.ids[~bad & ~kept],
        "good, kept": plan.ids[~bad & kept],
    }
    outcome_rows = {name: plan.take(ids, ROW_CELLS) for name, ids in outcomes.items()}
    misses = plan.take(outcomes["bad, kept"], FULL_GRID_CELLS)
    cost = plan.take(outcomes["good, removed"], FULL_GRID_CELLS)

    only_here = {}
    for column, passed in passes.items():
        others = np.ones(len(plan.ids), bool)
        for other, other_passed in passes.items():
            if other != column:
                others &= np.asarray(other_passed, bool)
        alone = plan.ids[~np.asarray(passed, bool) & others]
        if len(alone):
            only_here[column] = (plan.bad_first(alone, ROW_CELLS), alone)

    strip_plans = {}
    for column, name in (strips or {}).items():
        rule = filters.get(column)
        if rule is None or callable(rule) or rule[0] not in _REMOVES_LOW:
            continue
        scores = pd.to_numeric(labelled[column], errors="coerce")
        order = scores.dropna().sort_values(kind="stable")
        if order.empty:
            continue
        ordered_ids = labelled.loc[order.index, ID_COL].astype(str).tolist()
        values = order.tolist()
        comparator, threshold = rule
        passing = [_passes(value, comparator, threshold) for value in values]
        removes_low = _REMOVES_LOW[comparator]
        # Ascending, so the cutoff falls after the last removed item (low side
        # removed) or after the last kept one (high side removed).
        boundary = sum(not ok for ok in passing) if removes_low else sum(passing)
        ranks = thin_ranks(len(values), STRIP_CELLS, keep=(boundary - 1, boundary))
        plan.needed.update(ordered_ids[rank] for rank in ranks)
        strip_plans[column] = (name, ordered_ids, values, ranks, boundary, comparator,
                               threshold, removes_low)

    category_plans = {}
    for column, (name, table, dropped) in (categories or {}).items():
        rows = []
        for row in table.itertuples(index=False):
            members = plan.ids[(labelled[column] == row.category).to_numpy()]
            rows.append((row, plan.bad_first(members, ROW_CELLS)))
        category_plans[column] = (name, rows, set(dropped))

    cutouts = _cutouts(project_path, part, plan.needed)
    if not cutouts:
        return

    def cell(occurrence_id, caption=None):
        if occurrence_id not in cutouts:
            return None
        return _cell(cutouts[occurrence_id], plan.is_bad[occurrence_id],
                     plan.labels[occurrence_id] if caption is None else caption)

    def cells(ids, caption=None):
        made = [cell(occurrence_id, None if caption is None else caption(occurrence_id))
                for occurrence_id in ids]
        return [image for image in made if image is not None]

    full_cell = (CELL[0] + grids.LABEL_HEIGHT, CELL[1])

    def rows_grid(rows, title, keep_empty=False):
        # A row is its heading cell plus its items. An itemless row is dropped,
        # unless the grid is one where "none" is itself the answer.
        if not any(len(row) > 1 for row in rows):
            return None
        if not keep_empty:
            rows = [row for row in rows if len(row) > 1]
        return grids.comparison_grid(rows, title=title, cell=full_cell)

    def write(name, grid):
        if grid is not None:
            report.figure(name, grid)

    # 1. outcomes, and the two rows that matter in full
    write("outcomes", rows_grid(
        [[_text_cell([name, f"n={len(outcomes[name])}"])] + cells(shown)
         for name, shown in outcome_rows.items()],
        f"labelled rows by outcome, {len(filters)} filter(s): red frame = bad label, "
        "green = good", keep_empty=True))

    removed_by = {
        occurrence_id: ", ".join(short_name(column) for column, passed in passes.items()
                                 if not np.asarray(passed, bool)[index])
        for index, occurrence_id in enumerate(plan.ids)}
    shown = cells(misses)
    if shown:
        write("misses", grids.image_grid(
            shown, columns=ROW_CELLS, cell=full_cell,
            title="bad labels the filters keep" + _showing(len(misses), len(outcomes["bad, kept"]))))
    shown = cells(cost, caption=lambda occurrence_id: removed_by[occurrence_id])
    if shown:
        write("cost", grids.image_grid(
            shown, columns=ROW_CELLS, cell=full_cell,
            title="good labels the filters remove, and by which"
                  + _showing(len(cost), len(outcomes["good, removed"]))))

    # 2. per-filter unique removals
    write("only_here", rows_grid(
        [[_text_cell(_wrap(short_name(column))
                     + [f"{sum(plan.is_bad[i] for i in alone)} bad, "
                        f"{sum(not plan.is_bad[i] for i in alone)} good"])] + cells(shown_ids)
         for column, (shown_ids, alone) in only_here.items()],
        "what each filter removes that no other filter does"))

    # 3. one strip per continuous candidate
    for column, (name, ordered_ids, values, ranks, boundary, comparator, threshold,
                 removes_low) in strip_plans.items():
        strip = []
        for rank in ranks:
            if rank == boundary:
                strip.append(_text_cell(
                    ["cutoff", f"keep {comparator} {threshold:g}",
                     "<- removed" if removes_low else "removed ->"]))
            image = cell(ordered_ids[rank],
                         f"{values[rank]:.3g} {plan.labels[ordered_ids[rank]]}")
            if image is not None:
                strip.append(image)
        if boundary >= len(values):
            strip.append(_text_cell(["cutoff", f"keep {comparator} {threshold:g}",
                                     "<- removed" if removes_low else "removed ->"]))
        if any(image is not None for image in strip):
            write(f"{name}__strip", grids.image_grid(
                strip, columns=ROW_CELLS, cell=full_cell,
                title=f"{name}, ascending{_showing(len(ranks), len(values))}"))

    # 4. one gallery per categorical candidate
    for column, (name, rows, dropped) in category_plans.items():
        gallery = []
        for row, members in rows:
            state = ("DROPPED" if row.category in dropped
                     else "" if row.enough_labels else "too few labels")
            heading = [f"{row.category}", f"{row.n_bad}/{row.n} bad"] + ([state] if state else [])
            gallery.append([_text_cell(heading)] + cells(members))
        write(f"{name}__categories", rows_grid(
            gallery, f"{name}: labelled members per category, worst first"))


def _passes(value, comparator, threshold):
    if comparator == ">=":
        return value >= threshold
    if comparator == ">":
        return value > threshold
    if comparator == "<=":
        return value <= threshold
    return value < threshold
