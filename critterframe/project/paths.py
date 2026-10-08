"""Every path and filename in a project folder; creates nothing."""

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
    """Return the project directory as a Path."""
    return Path(project_path)


def is_absolute_anywhere(stored):
    """Return whether a stored path string is absolute on Windows or on POSIX."""
    return PurePosixPath(stored).is_absolute() or PureWindowsPath(stored).is_absolute()


def file_name_anywhere(stored):
    """Return the last component of a stored path string, split on either kind of slash."""
    return PureWindowsPath(stored).name


def relative_to_project(project_path, target):
    """Return a path as a record should store it.

    Args:
        project_path: The project.
        target: The path to store.

    Returns:
        A forward-slashed string: relative to the project when inside it, absolute otherwise.
    """
    absolute = Path(target).resolve()
    try:
        return absolute.relative_to(project_dir(project_path).resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def resolve_in_project(project_path, stored):
    """Return a stored path string as a Path, relative ones resolved against the project.

    Args:
        project_path: The project.
        stored: A path string from a record, or None.

    Returns:
        A Path, or None.
    """
    if stored is None:
        return None
    if is_absolute_anywhere(stored):
        return Path(stored)
    return project_dir(project_path) / stored


def occurrences_path(project_path):
    """Return the path of the occurrence table."""
    return project_dir(project_path) / OCCURRENCES_FILE


def images_path(project_path):
    """Return the path of the LMDB image store."""
    return project_dir(project_path) / IMAGES_DIR


def masks_path(project_path, reference=False):
    """Return the path of the mask table.

    Args:
        project_path: The project.
        reference: Return the reference table instead of the canonical one.
    """
    return project_dir(project_path) / (REFERENCE_MASKS_FILE if reference else MASKS_FILE)


def mask_shards_dir(project_path, part="", reference=False):
    """Return the staging directory for a sharded run's mask writes.

    Args:
        project_path: The project.
        part: Part whose shards to return; `""` is the root all parts share.
        reference: Use the reference table's staging area.
    """
    return project_dir(project_path) / MASK_SHARDS_DIR / ("reference" if reference else "canonical") / part


def mask_shard_path(project_path, part, reference=False):
    """Return a fresh, never-used path for one flush of a sharded run's masks.

    Names sort in write order, which `merge_mask_shards` relies on.

    Args:
        project_path: The project.
        part: Part being written.
        reference: Use the reference table's staging area.
    """
    return (
        mask_shards_dir(project_path, part=part, reference=reference)
        / f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.parquet"
    )


def failures_path(project_path):
    """Return the path of the failures ledger."""
    return project_dir(project_path) / FAILURES_FILE


def calibrations_path(project_path):
    """Return the path of the calibrations table."""
    return project_dir(project_path) / CALIBRATIONS_FILE


def runs_and_metrics_path(project_path):
    """Return the path of the sqlite database of runs and metric values."""
    return project_dir(project_path) / RUNS_AND_METRICS_FILE


def runs_log_path(project_path):
    """Return the path of the JSON Lines mirror of the runs table."""
    return project_dir(project_path) / RUNS_LOG_FILE


def raw_imports_dir(project_path):
    """Return the directory of archived raw imports."""
    return project_dir(project_path) / RAW_IMPORTS_DIR


def raw_import_path(project_path, name_prefix, extension=".csv"):
    """Return where one archived raw import lands: `<prefix>_<today>[_n]<extension>`.

    Reads the directory to find a free name, so a same-day archive of different content
    gets `_n`.

    Args:
        project_path: The project.
        name_prefix: Import kind, usually with the source, e.g. `"occurrences_antenna_199"`.
        extension: File extension, with its dot.
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
    """Return the path of the append-only log of imports."""
    return raw_imports_dir(project_path) / IMPORTS_LOG_FILE


def import_sidecar_path(raw_path, import_hash):
    """Return the manifest path for one import: `<raw file stem>.<import hash>.import.json`.

    Args:
        raw_path: The archived raw import the manifest describes.
        import_hash: The import's hash.
    """
    raw_path = Path(raw_path)
    return raw_path.with_name(f"{raw_path.stem}.{import_hash}.import.json")


def definitions_dir(project_path):
    """Return the directory of hand-edited project definitions."""
    return project_dir(project_path) / DEFINITIONS_DIR


def subsets_path(project_path):
    """Return the path of `subsets.toml`."""
    return definitions_dir(project_path) / SUBSETS_FILE


def recipes_path(project_path):
    """Return the path of the optional project-local recipes module."""
    return definitions_dir(project_path) / RECIPES_FILE


def visualizations_dir(project_path, subdir=""):
    """Return the visualizations directory, or a subdirectory of it.

    Args:
        project_path: The project.
        subdir: Subdirectory name; `""` for the root.
    """
    return project_dir(project_path) / VISUALIZATIONS_DIR / subdir


def pipeline_dir(project_path):
    """Return the directory of pipeline diagnostics."""
    return visualizations_dir(project_path, PIPELINE_DIR)


def pipeline_stem(name, identity_hash, part=None):
    """Return the filename stem every file of one report shares: `<name>[__<part>]_<hash>`.

    Args:
        name: The report's name, e.g. a run name or `validate_masks__head`.
        identity_hash: The recipe hash or other identity digest.
        part: Part to qualify the stem with; None omits it.
    """
    stem = name if part is None else f"{name}__{part}"
    return f"{stem}_{identity_hash}"


def pipeline_file_path(project_path, name, identity_hash, part=None, suffix=None, extension="jpg"):
    """Return one file of a report: `<name>[__<part>]_<hash>[__<suffix>].<ext>`.

    Args:
        project_path: The project.
        name: As in `pipeline_stem`.
        identity_hash: As in `pipeline_stem`.
        part: As in `pipeline_stem`.
        suffix: A checkpoint label or figure name; None for the report's main grid.
        extension: `jpg` for grids, `png` for figures.
    """
    stem = pipeline_stem(name, identity_hash, part=part)
    if suffix:
        stem = f"{stem}__{suffix}"
    return pipeline_dir(project_path) / f"{stem}.{extension}"


def pipeline_report_path(project_path, name, identity_hash, part=None):
    """Return a report's sidecar path: `<name>[__<part>]_<hash>.report.json`.

    Args:
        project_path: The project.
        name: As in `pipeline_stem`.
        identity_hash: As in `pipeline_stem`.
        part: As in `pipeline_stem`.
    """
    stem = pipeline_stem(name, identity_hash, part=part)
    return pipeline_dir(project_path) / f"{stem}{REPORT_SIDECAR_SUFFIX}"


def products_dir(project_path, name=""):
    """Return the directory of rendered products, or one render's folder.

    Args:
        project_path: The project.
        name: A render's folder name; `""` for the root.
    """
    return visualizations_dir(project_path, PRODUCTS_DIR) / name


def product_filename(occurrence_id, part=None, extension="png", label=None):
    """Return a rendered occurrence-part's filename: `[<label>__]<occurrence_id>[__<part>].<ext>`.

    Args:
        occurrence_id: The occurrence.
        part: Part to qualify the name with; None for a single-part render.
        extension: Image format.
        label: Text to lead the name with, e.g. a species. Must be filename-safe and hold no
            double underscore.
    """
    stem = str(occurrence_id) if part is None else f"{occurrence_id}__{part}"
    if label is not None:
        stem = f"{label}__{stem}"
    return f"{stem}.{str(extension).lstrip('.')}"


def models_dir(project_path):
    """Return the directory of this project's model checkpoints."""
    return project_dir(project_path) / MODELS_DIR


def models_registry_path(project_path):
    """Return the path of the model registry."""
    return models_dir(project_path) / MODELS_REGISTRY_FILE


def exports_dir(project_path):
    """Return the directory of this project's exports."""
    return project_dir(project_path) / EXPORTS_DIR


def exports_log_path(project_path):
    """Return the path of the append-only log of exports."""
    return exports_dir(project_path) / EXPORTS_LOG_FILE


def export_sidecar_path(path):
    """Return the manifest path beside an exported file: `traits.csv` gives `traits.export.json`.

    Args:
        path: The exported file, inside or outside a project.
    """
    return Path(path).with_suffix(EXPORT_SIDECAR_SUFFIX)


def default_export_path(project_path):
    """Return a fresh, never-used CSV path under the project's exports folder."""
    return exports_dir(project_path) / f"traits_{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.csv"


def require_project(project_path):
    """Return the project directory as a Path.

    Args:
        project_path: The project.

    Raises:
        FileNotFoundError: If it holds no occurrence table.
    """
    directory = project_dir(project_path)
    if not occurrences_path(directory).exists():
        raise FileNotFoundError(
            f"{directory} isn't a CritterFrame project -- no "
            f"{OCCURRENCES_FILE}. Run critterframe.ingest_occurrences() or "
            "ingest_images() to create one."
        )
    return directory
