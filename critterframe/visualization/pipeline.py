"""
Pipeline visualization: one grid per run, showing what a recipe did to a sample
of the occurrences it processed.

The principle, which holds for every kind of run:

    Any operation in a recipe may contribute a visual panel, and a pipeline
    visualization summarizes those panels for a sample of the occurrences a run
    processed.

Nothing in that sentence mentions segmentation or metrics, and that's the point.
A segmentation run's panels happen to be detector boxes, raw model masks, and
cleaned masks; a colour-pattern metric run's happen to be the oriented segment,
the classified pixels, and the measurement overlay; a one-stage QC metric
contributes a single panel per occurrence. All three are the same object -- a
stage per column, an occurrence per row -- so `visualize=25` means exactly the
same thing wherever it's passed.

    run_segments(project_path, steps=[...], visualize=25)
    run_metrics(project_path, run_name="colour", metrics=[...], visualize=25)

each produce

    visualizations/pipeline/<run name>_<recipe hash>.jpg

one file, whatever the size of the project. The recipe hash is in the filename
so re-running an unchanged recipe overwrites its own grid while a changed one
writes a new grid beside the old -- the same identity rule the rest of the
package runs on, applied to pictures.

A run over 10,000 occurrences can't be inspected by looking at 10,000 debug
images, and nobody does. So there is no per-occurrence file mode: every form of
visualize= samples, and asking for a handful is the only way to ask.

The sample is drawn deterministically (see selectionhelpers.sample_occurrences),
so two grids from two versions of a recipe show the same specimens and can be
compared cell by cell. Sampling accepts a count or explicit ids now; the
intended later additions -- worst QC scores, failures, stratified by taxon --
are all "choose a different set of ids", which is why sampling is a function
returning ids rather than anything woven into the runs.
"""

import logging

import cv2

from ..project import paths
from ..selectionhelpers import sample_occurrences
from . import grids

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE = 25
JPEG_QUALITY = 88


def resolve_sample(occurrence_ids, visualize):
    """
    Turn a run's visualize= argument into the ids to build a grid from, or None
    if this run isn't making one.

    visualize -- False/None: no visualization.
                 True:       a default-sized sample (25).
                 int:        sample that many.
                 iterable:   exactly these occurrence ids, ignoring any this run
                             doesn't cover. Explicit ids are how you follow one
                             known-difficult specimen through a recipe.

    occurrence_ids should be the ids the run will actually PROCESS, not every id
    it was pointed at: a grid can only show work that happened, and a run whose
    occurrences were all already done has nothing to draw.
    """
    if visualize is None or visualize is False:
        return None

    ids = [str(occurrence_id) for occurrence_id in occurrence_ids]
    if visualize is True:
        return sample_occurrences(ids, DEFAULT_SAMPLE)
    if isinstance(visualize, int):
        return sample_occurrences(ids, visualize)

    wanted = {str(occurrence_id) for occurrence_id in visualize}
    chosen = [occurrence_id for occurrence_id in ids if occurrence_id in wanted]
    missing = wanted - set(chosen)
    if missing:
        logger.warning("visualize= named %d occurrence(s) this run isn't "
                       "processing, ignoring them (e.g. %s)",
                       len(missing), sorted(missing)[0])
    return chosen


class RunReport:
    """
    The panels of one recipe, on one part, for one sample -- and the grid they
    become.

    One report is one grid. It knows a recipe hash, a part, the occurrences it
    is collecting from, and the stages it has seen; it does not know about other
    parts, other recipes, or where its panels came from. A run producing several
    parts makes several reports (see PanelFanout for the shared-steps case).

    Panels are fitted to their final cell the moment they arrive, so a report
    holds a bounded amount of memory -- sample x stages x one cell -- rather
    than a full-resolution copy of every intermediate image of every sampled
    occurrence.

    project_path -- project to write the grid into.
    name         -- the run's name, and the first part of the filename.
    recipe_hash  -- identity of the recipe whose behaviour this shows.
    part         -- the part being produced or measured.
    sample       -- occurrence ids to collect from, in row order.
    """

    def __init__(self, project_path, name, recipe_hash, part, sample,
                 cell=grids.DEFAULT_CELL, columns=grids.DEFAULT_COLUMNS):
        self.project_path = project_path
        self.name = name
        self.recipe_hash = recipe_hash
        self.part = part
        self.sample = [str(occurrence_id) for occurrence_id in sample]
        self.cell = cell
        self.columns = columns

        self._sample_set = set(self.sample)
        self._panels = {}     # occurrence_id -> {stage: fitted panel}
        self._stages = []     # stage names, in the order first seen

    def __bool__(self):
        """Truthy only when there's actually something to collect."""
        return bool(self._sample_set)

    def wants(self, occurrence_id):
        """True if this occurrence is in the sample."""
        return str(occurrence_id) in self._sample_set

    def collect(self, occurrence_id, stage, panel):
        """
        Take one panel, from Segment.emit_panel.

        A panel that can't be laid out is dropped with a warning rather than
        raised: visualization is diagnostic, and losing a cell must never cost
        the run that produced it.
        """
        occurrence_id = str(occurrence_id)
        try:
            fitted = grids.fit_cell(panel, cell=self.cell)
        except Exception as exc:
            logger.warning("could not lay out '%s' panel for %s: %s",
                           stage, occurrence_id, exc)
            return

        self._panels.setdefault(occurrence_id, {})[stage] = fitted
        if stage not in self._stages:
            self._stages.append(stage)

    def rows(self):
        """(row images, row labels) in sample order, skipping occurrences with nothing."""
        rows = []
        labels = []
        for occurrence_id in self.sample:
            collected = self._panels.get(occurrence_id)
            if not collected:
                continue
            rows.append([collected.get(stage) for stage in self._stages])
            labels.append(occurrence_id)
        return rows, labels

    def save(self):
        """
        Write this report's grid, and return its path (None if nothing was
        collected).

        A single stage becomes an image grid -- the same view across specimens,
        which is what you scan for outliers. Several stages become a comparison
        grid -- a row per specimen, a column per stage, which is what shows
        WHERE in a recipe something went wrong. The layout follows the question
        the collected panels can answer.
        """
        rows, labels = self.rows()
        if not rows:
            logger.info("no pipeline panels collected for '%s' part '%s' -- no "
                        "operation in this recipe draws one", self.name, self.part)
            return None

        heading = f"{self.name} / {self.part} / n={len(rows)} / {self.recipe_hash}"
        if len(self._stages) == 1:
            grid = grids.image_grid([row[0] for row in rows], labels=labels,
                                    title=f"{heading} / {self._stages[0]}",
                                    columns=self.columns, cell=self.cell)
        else:
            grid = grids.comparison_grid(rows, column_titles=self._stages,
                                         row_labels=labels, title=heading,
                                         cell=self.cell)

        return self._write(grid)

    def _write(self, grid):
        """Write the grid, named for the run, the part, and the recipe."""
        from ..recipes import DEFAULT_PART

        directory = paths.pipeline_dir(self.project_path)
        directory.mkdir(parents=True, exist_ok=True)

        # The part is only in the filename when it isn't the default one, so a
        # plain whole-organism run gets the obvious `<name>_<hash>.jpg` and a
        # multi-part run still gets one distinct file per part.
        stem = self.name if self.part == DEFAULT_PART else f"{self.name}__{self.part}"
        dest = directory / f"{stem}_{self.recipe_hash}.jpg"

        cv2.imwrite(str(dest), grid, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        logger.info("pipeline grid -> %s", dest)
        return dest


class PanelFanout:
    """
    A panel sink that hands each panel to several reports at once.

    Exists for exactly one situation: a multi-output segmentation run, whose
    shared steps run ONCE on a segment that then forks per part. Those shared
    panels belong on every output part's grid -- the shared steps are literally
    in each part's recipe -- so the forking segment emits into this and each
    report gets its own copy.

    Kept separate so RunReport stays one recipe, one part, one sample. The
    multiplexing is a property of how a run is shaped, not of what a report is.
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


def run_report(project_path, name, recipe_hash, part, occurrence_ids, visualize):
    """
    The RunReport for one part of a run, or None if it wasn't asked to
    visualize anything.

    Runs call this instead of interpreting visualize= themselves, so 25 means
    the same thing to a segmentation run and a metric run.
    """
    sample = resolve_sample(occurrence_ids, visualize)
    if not sample:
        return None
    return RunReport(project_path, name, recipe_hash, part, sample)


def panel_sink(report, occurrence_id):
    """
    What to give one occurrence's Segment as its panel sink: the report when
    this occurrence is in the sample, None otherwise.

    None is what keeps panel building off the path of the thousands of
    occurrences nobody is looking at -- an operation's emit_panel() call costs
    nothing when there's no sink to send it to.
    """
    if report is None or not report.wants(occurrence_id):
        return None
    return report
