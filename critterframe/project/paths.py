"""
Define every path and filename in critterframe project folders.

Creates nothing -- directories appear when a writer first needs them, so what a
project holds is an honest account of what has been done to it. Imports nothing
else in the package, so it holds no opinion about parts or recipes: where a name
varies by part, `part=None` omits it and the caller decides what counts as
default.

Every function returns a pathlib.Path, except product_filename, which returns
just the name.
"""

import time
import uuid
from datetime import date
from pathlib import Path

OCCURRENCES_FILE = "occurrences.parquet"
IMAGES_DIR = "images.lmdb"
MASKS_FILE = "masks.parquet"
REFERENCE_MASKS_FILE = "reference_masks.parquet"
MASK_SHARDS_DIR = "mask_shards"
CALIBRATIONS_FILE = "calibrations.parquet"
RUNS_AND_METRICS_FILE = "runs_and_metrics.sqlite"
IMPORTS_DIR = "imports"
DEFINITIONS_DIR = "definitions"
SUBSETS_FILE = "subsets.toml"
RECIPES_FILE = "recipes.py"
VISUALIZATIONS_DIR = "visualizations"
PIPELINE_DIR = "pipeline"
PRODUCTS_DIR = "products"
MODELS_DIR = "models"
MODELS_REGISTRY_FILE = "registry.json"


def project_dir(project_path):
    """
    The project directory itself, as a Path.

    Every other function goes through this, so a project_path may be a string
    or a Path anywhere.
    """
    return Path(project_path)


def occurrences_path(project_path):
    """The central imported/normalized occurrence table."""
    return project_dir(project_path) / OCCURRENCES_FILE


def images_path(project_path):
    """The LMDB environment holding one original analysis image per occurrence."""
    return project_dir(project_path) / IMAGES_DIR


def masks_path(project_path, reference=False):
    """
    The mask table.

    reference -- True returns the reference table instead of the canonical one.
                 Identical schema; the two coexist, since validation compares
                 them.
    """
    return project_dir(project_path) / (
        REFERENCE_MASKS_FILE if reference else MASKS_FILE
    )


def mask_shards_dir(project_path, part="", reference=False):
    """
    Staging area for a sharded run's mask writes, read back by
    records.masks.merge_mask_shards.

    part -- narrows to one part's shards. "" is the root they share, for
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


def imports_dir(project_path):
    """Immutable source imports -- the recovery path if an ingest was ever wrong."""
    return project_dir(project_path) / IMPORTS_DIR


def import_path(project_path, name_prefix, extension=".csv"):
    """
    Where one archived source file lands: `<prefix>_<today>[_n]<extension>`.

    The `_n` suffix stops a same-day re-import clobbering the earlier one, so
    this reads the directory to find a free name. It still creates nothing.

    name_prefix -- import kind, usually carrying the source, e.g.
                   "occurrences_antenna_199".
    """
    directory = imports_dir(project_path)
    base = f"{name_prefix}_{date.today().isoformat()}"
    extension = extension or ".csv"

    dest = directory / f"{base}{extension}"
    n = 1
    while dest.exists():
        dest = directory / f"{base}_{n}{extension}"
        n += 1
    return dest


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
    Sample-level summaries of how processing behaved: one sheet per run.

    Bounded by design -- 10,000 occurrences still produce one image per run,
    since "is this step working" is answered from a representative sample.
    """
    return visualizations_dir(project_path, PIPELINE_DIR)


def pipeline_grid_path(project_path, run_name, recipe_hash, part=None):
    """
    One run's QC grid: `<run>[__<part>]_<hash>.jpg`.

    The hash is in the name so two versions of a recipe leave two grids to
    compare rather than one overwriting the other.

    part -- None omits it, giving a whole-organism run the plain
            `<run>_<hash>.jpg`. Pass it so each part of a multi-part run gets
            its own file.
    """
    stem = run_name if part is None else f"{run_name}__{part}"
    return pipeline_dir(project_path) / f"{stem}_{recipe_hash}.jpg"


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

    part -- None writes `<occurrence_id>.<ext>`, for a single-part render.
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
