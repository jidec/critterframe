"""
Define every path and filename in critterframe project folders.

Creates nothing -- directories appear when a writer first needs them, so what a
project holds is an honest account of what has been done to it. Imports nothing
else in the package, so it holds no opinion about parts or recipes: where a name
varies by part, `part=None` omits it and the caller decides what counts as
default.

Every function returns a pathlib.Path, except product_filename, which returns
just the name, and the helpers that read and write a stored path string
(relative_to_project and the *_anywhere checks).
"""

import time
import uuid
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath

OCCURRENCES_FILE = "occurrences.parquet"
IMAGES_DIR = "images.lmdb"
MASKS_FILE = "masks.parquet"
REFERENCE_MASKS_FILE = "reference_masks.parquet"
MASK_SHARDS_DIR = "mask_shards"
FAILURES_FILE = "failures.parquet"
CALIBRATIONS_FILE = "calibrations.parquet"
RUNS_AND_METRICS_FILE = "runs_and_metrics.sqlite"
RUNS_LOG_FILE = "runs.jsonl"
RAW_IMPORTS_DIR = "raw_imports"
IMPORTS_LOG_FILE = "imports.jsonl"
IMPORT_SIDECAR_SUFFIX = ".import.json"
DEFINITIONS_DIR = "definitions"
SUBSETS_FILE = "subsets.toml"
RECIPES_FILE = "recipes.py"
VISUALIZATIONS_DIR = "visualizations"
PIPELINE_DIR = "pipeline"
REPORT_SIDECAR_SUFFIX = ".report.json"
PRODUCTS_DIR = "products"
MODELS_DIR = "models"
MODELS_REGISTRY_FILE = "registry.json"
EXPORTS_DIR = "exports"
EXPORTS_LOG_FILE = "exports.jsonl"
EXPORT_SIDECAR_SUFFIX = ".export.json"


def project_dir(project_path):
    """
    The project directory itself, as a Path.

    Every other function goes through this, so a project_path may be a string
    or a Path anywhere.
    """
    return Path(project_path)


def is_absolute_anywhere(stored):
    """
    Whether a stored path string is absolute on ANY platform.

    A path written on Windows read on Linux or macOS is not recognized as
    absolute by the native Path, and the reverse, so both flavours are asked.
    """
    return (PurePosixPath(stored).is_absolute()
            or PureWindowsPath(stored).is_absolute())


def file_name_anywhere(stored):
    """The last component of a stored path string, split on forward slashes and backslashes alike."""
    return PureWindowsPath(stored).name


def relative_to_project(project_path, target):
    """
    A path as a record should store it: relative to the project when it sits
    inside, absolute otherwise, posix-separated either way.

    A relative, `/`-separated path resolves the same on Windows, Linux and
    macOS once joined onto the project folder, so a copied project still finds
    what it recorded.

    - `target` -- the path to store.

    Returns a string.
    """
    absolute = Path(target).resolve()
    try:
        return absolute.relative_to(project_dir(project_path).resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def resolve_in_project(project_path, stored):
    """
    A stored path back as a Path: relative ones against the project, absolute
    ones as they are.

    Always this, never a bare Path(stored), which would resolve a relative one
    against the current working directory.

    - `stored` -- a path string from a record, or None.

    Returns a Path, or None.
    """
    if stored is None:
        return None
    if is_absolute_anywhere(stored):
        return Path(stored)
    return project_dir(project_path) / stored


def occurrences_path(project_path):
    """The central imported/normalized occurrence table."""
    return project_dir(project_path) / OCCURRENCES_FILE


def images_path(project_path):
    """The LMDB environment holding one original analysis image per occurrence."""
    return project_dir(project_path) / IMAGES_DIR


def masks_path(project_path, reference=False):
    """
    The mask table.

    - `reference` -- True returns the reference table instead of the canonical
      one. Identical schema; the two coexist, since validation compares them.
    """
    return project_dir(project_path) / (
        REFERENCE_MASKS_FILE if reference else MASKS_FILE
    )


def mask_shards_dir(project_path, part="", reference=False):
    """
    Staging area for a sharded run's mask writes, read back by
    records.masks.merge_mask_shards.

    - `part` -- narrows to one part's shards. "" is the root they share, for
      listing which parts have anything staged.
    """
    return project_dir(project_path) / MASK_SHARDS_DIR / \
        ("reference" if reference else "canonical") / part


def mask_shard_path(project_path, part, reference=False):
    """
    A fresh, never-before-used path for one flush of a sharded run's masks.

    Timestamp first so filenames sort in write order, which merge_mask_shards
    relies on; the uuid breaks ties within a nanosecond. A new name every call
    means no two writers contend for a file.
    """
    return (mask_shards_dir(project_path, part=part, reference=reference)
            / f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.parquet")


def failures_path(project_path):
    """
    The ledger of failed download/segmentation attempts, keyed by
    occurrence-part-stage -- what a rerun consults so a failure that hasn't
    changed isn't retried (see records.failures).
    """
    return project_dir(project_path) / FAILURES_FILE


def calibrations_path(project_path):
    """
    What is known about the imaging system rather than any organism, e.g. px/mm.

    One table for every calibration type, keyed by what an imaging setup holds
    constant: one occurrence, one session, one copy stand.
    """
    return project_dir(project_path) / CALIBRATIONS_FILE


def runs_and_metrics_path(project_path):
    """The sqlite database holding run records and the metric values they produced."""
    return project_dir(project_path) / RUNS_AND_METRICS_FILE


def runs_log_path(project_path):
    """
    The append-only, human-readable mirror of runs_and_metrics.sqlite's runs
    table -- one JSON line per finished run, readable without opening the
    database. See records.runs.finish_run.
    """
    return project_dir(project_path) / RUNS_LOG_FILE


def raw_imports_dir(project_path):
    """
    Immutable, byte-exact copies of every raw import this project has read --
    the recovery path if turning one into an import was ever wrong.
    """
    return project_dir(project_path) / RAW_IMPORTS_DIR


def raw_import_path(project_path, name_prefix, extension=".csv"):
    """
    Where one archived raw import lands: `<prefix>_<today>[_n]<extension>`.

    The `_n` suffix stops a same-day archive of genuinely different content
    clobbering the earlier one, so this reads the directory to find a free
    name. It still creates nothing. Content-identical raw imports never reach
    here at all -- see ingest._archive_raw_import, which reuses the existing
    file instead.

    - `name_prefix` -- import kind, usually carrying the source, e.g.
      "occurrences_antenna_199".
    """
    directory = raw_imports_dir(project_path)
    base = f"{name_prefix}_{date.today().isoformat()}"
    extension = extension or ".csv"

    dest = directory / f"{base}{extension}"
    n = 1
    while dest.exists():
        dest = directory / f"{base}_{n}{extension}"
        n += 1
    return dest


def imports_log_path(project_path):
    """
    The append-only log of every import this project has produced from a raw
    import -- one line per call to ingest_occurrences that actually did work.

    JSON Lines for the same reason exports_log_path is: it only ever grows,
    and a read-merge-rewrite of a growing file is what an appending writer
    should not be doing.
    """
    return raw_imports_dir(project_path) / IMPORTS_LOG_FILE


def import_sidecar_path(raw_path, import_hash):
    """
    The manifest for one import derived from a raw import: `<raw file
    stem>.<import hash>.import.json`, beside the raw file it describes.

    The hash is in the name, the same reasoning pipeline_stem gives for
    putting a recipe hash in a grid's filename: one raw import can become
    several imports under different decisions (a different drop=, a different
    cap), so two imports of one raw file leave two manifests to compare rather
    than one overwriting the other.

    - `raw_path` -- the archived raw import the manifest describes.
    """
    raw_path = Path(raw_path)
    return raw_path.with_name(f"{raw_path.stem}.{import_hash}.import.json")


def definitions_dir(project_path):
    """Project subsets/recipes/config -- the hand-edited part of a project."""
    return project_dir(project_path) / DEFINITIONS_DIR


def subsets_path(project_path):
    """Named selections of occurrences that receive a particular recipe."""
    return definitions_dir(project_path) / SUBSETS_FILE


def recipes_path(project_path):
    """
    Optional project-local module holding recipes a user runs repeatedly, so a
    project carries its own processing definitions. Nothing imports it for you.
    """
    return definitions_dir(project_path) / RECIPES_FILE


def visualizations_dir(project_path, subdir=""):
    """
    Diagnostic images and figure material.

    pipeline_dir and products_dir have fixed meanings and hold a run's output;
    anything else here is named by whoever wrote it (panels.save_panel).
    """
    return project_dir(project_path) / VISUALIZATIONS_DIR / subdir


def pipeline_dir(project_path):
    """
    Every activity's diagnostics: a sampled grid, checkpoints, figures, and a
    sidecar per report, all sharing one flat stem (see pipeline_stem).

    Bounded by design -- 10,000 occurrences still produce one grid per report,
    since "is this step working" is answered from a representative sample.
    """
    return visualizations_dir(project_path, PIPELINE_DIR)


def pipeline_stem(name, identity_hash, part=None):
    """
    The filename stem every file of one report shares: `<name>[__<part>]_<hash>`.

    The hash is in the name so two versions of a recipe (or of any other
    activity's configuration) leave two sets of files rather than one
    overwriting the other.

    - `name` -- the report's name, e.g. a run name or `validate_masks__head`.
    - `identity_hash` -- the recipe hash or other identity digest.
    - `part` -- None omits it. Pass it so each part of a multi-part run gets
      its own files.
    """
    stem = name if part is None else f"{name}__{part}"
    return f"{stem}_{identity_hash}"


def pipeline_file_path(project_path, name, identity_hash, part=None, suffix=None,
                       extension="jpg"):
    """
    One file of a report: `<name>[__<part>]_<hash>[__<suffix>].<ext>`.

    - `project_path` -- project whose pipeline directory to write into.
    - `name`, `identity_hash`, `part` -- as in `pipeline_stem`.
    - `suffix` -- None for the report's main grid; a checkpoint label
      (`at00012500`, `epoch0005`) or a figure name (`curves`) otherwise.
    - `extension` -- `jpg` for grids, `png` for figures.
    """
    stem = pipeline_stem(name, identity_hash, part=part)
    if suffix:
        stem = f"{stem}__{suffix}"
    return pipeline_dir(project_path) / f"{stem}.{extension}"


def pipeline_report_path(project_path, name, identity_hash, part=None):
    """
    A report's sidecar: `<name>[__<part>]_<hash>.report.json`, beside its files.

    - `project_path`, `name`, `identity_hash`, `part` -- as in `pipeline_file_path`.
    """
    stem = pipeline_stem(name, identity_hash, part=part)
    return pipeline_dir(project_path) / f"{stem}{REPORT_SIDECAR_SUFFIX}"


def products_dir(project_path, name=""):
    """
    Assets materialized for downstream use: one file per occurrence-part, in a
    folder per render.

    The opposite contract to pipeline_dir -- outputs, not diagnostics. The
    occurrence id is in each filename, so a directory listing joins back to an
    exported trait table from any language.
    """
    return visualizations_dir(project_path, PRODUCTS_DIR) / name


def product_filename(occurrence_id, part=None, extension="png"):
    """
    The filename one rendered occurrence-part gets:
    `<occurrence_id>[__<part>].<ext>`.

    Id first, part after a double underscore, so splitting a filename back into
    ids is one operation in any language.

    - `part` -- None writes `<occurrence_id>.<ext>`, for a single-part render.
    """
    stem = str(occurrence_id) if part is None else f"{occurrence_id}__{part}"
    return f"{stem}.{str(extension).lstrip('.')}"


def models_dir(project_path):
    """Named custom segmenters/checkpoints belonging to this project."""
    return project_dir(project_path) / MODELS_DIR


def models_registry_path(project_path):
    """
    What this project knows about the models it uses: name, checkpoint
    fingerprint, task, base model, training data (see records.models).

    JSON because it holds a few rows of deeply nested provenance and nothing
    hand-edits it, unlike subsets.toml.
    """
    return models_dir(project_path) / MODELS_REGISTRY_FILE


def exports_dir(project_path):
    """What this project has handed out: one manifest line per export written."""
    return project_dir(project_path) / EXPORTS_DIR


def exports_log_path(project_path):
    """
    The append-only log of every export this project has produced.

    JSON Lines rather than one JSON document, because it only ever grows and a
    read-merge-rewrite of a growing file is what an appending writer should not
    be doing.
    """
    return exports_dir(project_path) / EXPORTS_LOG_FILE


def export_sidecar_path(path):
    """
    The manifest that travels with one written export: `traits.csv` ->
    `traits.export.json`.

    Takes any path, not a project one, since an export is usually written
    outside the project it came from.

    - `path` -- the exported file the manifest describes.
    """
    return Path(path).with_suffix(EXPORT_SIDECAR_SUFFIX)


def default_export_path(project_path):
    """
    A fresh, never-before-used CSV path under this project's exports/ folder.

    Timestamp-plus-uuid, the same scheme mask_shard_path uses, so exporting
    repeatedly without naming a file never overwrites a previous export.
    """
    return (exports_dir(project_path)
            / f"traits_{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.csv")


def require_project(project_path):
    """
    Raise unless project_path holds an occurrence table, and return it as a
    Path.

    Called before anything that reads existing state, so a typo'd path fails
    loudly rather than reporting an empty project. Writers don't call it --
    ingest has to work on a directory that doesn't exist yet.
    """
    directory = project_dir(project_path)
    if not occurrences_path(directory).exists():
        raise FileNotFoundError(
            f"{directory} isn't a CritterFrame project -- no "
            f"{OCCURRENCES_FILE}. Run critterframe.ingest_occurrences() or "
            "ingest_images() to create one."
        )
    return directory
