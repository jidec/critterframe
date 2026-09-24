"""
archive_project(): a deposit-ready copy of a project, for archiving alongside a publication.

Leaves out the images and raw source data (licensing and size; cite the source instead) and working files, and
reduces local paths to file names.
"""

import contextlib
import importlib.metadata
import json
import logging
import platform
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from . import paths

logger = logging.getLogger(__name__)

# Record keys whose value can be a path on the machine that made it. Reduced to
# the file name when absolute, so a deposit neither leaks a local user folder
# nor names a location no reader has.
PATH_KEYS = ("import_source_path", "path", "project", "source")

# Table columns holding a source file's path, e.g. ingest_images' source_path.
PATH_COLUMNS = ("source_path",)

# Installed distributions whose versions are recorded, when present.
DISTRIBUTIONS = (
    "numpy", "pandas", "pyarrow", "opencv-python", "opencv-python-headless",
    "opencv-contrib-python", "lmdb", "pycocotools", "scikit-learn", "matplotlib",
    "torch", "transformers", "segmentation-models-pytorch",
)

TABLES = (paths.OCCURRENCES_FILE, paths.MASKS_FILE, paths.REFERENCE_MASKS_FILE,
          paths.CALIBRATIONS_FILE)


def archive_project(project_path, dest, source_doi=None, scripts=None):
    """
    Write a deposit-ready copy of a project to `dest`.

    Copies the tables, the metric database, run and import logs, import and
    export manifests, exports, definitions and the model registry, plus
    `environment.json` and a short `README.md`. Leaves out the image store,
    raw source data, mask shards, the failure ledger, visualizations and
    model checkpoints. Absolute paths in copied records become file names.

    - `project_path` -- project to archive.
    - `dest` -- folder to write; must be new or empty, and outside the project.
    - `source_doi` -- DOI(s) of the source data left out, e.g. a GBIF
      download's; listed in the README to cite.
    - `scripts` -- a folder, or a list of files, of the code that drove the
      pipeline; copied into `code/`.

    Returns `dest` as a Path.
    """
    paths.require_project(project_path)
    project = paths.project_dir(project_path).resolve()
    dest = Path(dest).resolve()

    if dest == project or project in dest.parents:
        raise ValueError(f"dest {dest} is inside the project; archive it elsewhere")
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"dest {dest} already has files in it")
    dest.mkdir(parents=True, exist_ok=True)

    included = []
    for name in TABLES:
        if _copy_table(project / name, dest / name):
            included.append(name)
    if _copy_database(project / paths.RUNS_AND_METRICS_FILE,
                      dest / paths.RUNS_AND_METRICS_FILE):
        included.append(paths.RUNS_AND_METRICS_FILE)
    if _copy_json(project / paths.RUNS_LOG_FILE, dest / paths.RUNS_LOG_FILE):
        included.append(paths.RUNS_LOG_FILE)

    raw_imports = project / paths.RAW_IMPORTS_DIR
    for source in sorted(raw_imports.glob(f"*{paths.IMPORT_SIDECAR_SUFFIX}")) + [
            raw_imports / paths.IMPORTS_LOG_FILE]:
        _copy_json(source, dest / paths.RAW_IMPORTS_DIR / source.name)
    if (dest / paths.RAW_IMPORTS_DIR).exists():
        included.append(f"{paths.RAW_IMPORTS_DIR}/")

    exports = project / paths.EXPORTS_DIR
    for source in sorted(exports.glob("*")) if exports.is_dir() else []:
        target = dest / paths.EXPORTS_DIR / source.name
        if source.suffix in (".json", ".jsonl"):
            _copy_json(source, target)
        elif source.suffix == ".csv":
            _copy_csv(source, target)
    if (dest / paths.EXPORTS_DIR).exists():
        included.append(f"{paths.EXPORTS_DIR}/")

    definitions = project / paths.DEFINITIONS_DIR
    if definitions.is_dir():
        shutil.copytree(definitions, dest / paths.DEFINITIONS_DIR,
                        ignore=shutil.ignore_patterns("__pycache__"))
        included.append(f"{paths.DEFINITIONS_DIR}/")
    if _copy_json(project / paths.MODELS_DIR / paths.MODELS_REGISTRY_FILE,
                  dest / paths.MODELS_DIR / paths.MODELS_REGISTRY_FILE):
        included.append(f"{paths.MODELS_DIR}/{paths.MODELS_REGISTRY_FILE}")

    if scripts is not None:
        _copy_scripts(scripts, dest / "code")
        included.append("code/")

    environment = _environment()
    _write_text(dest / "environment.json", json.dumps(environment, indent=2) + "\n")
    _write_text(dest / "README.md",
                _readme(project.name, dest, included, environment, source_doi))

    logger.info("archived %s -> %s (%d item(s))", project, dest, len(included))
    return dest


def _redacted(value, key=None):
    """`value` with every absolute path under a PATH_KEYS key reduced to its file name."""
    if isinstance(value, dict):
        return {k: _redacted(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_redacted(v) for v in value]
    if key in PATH_KEYS and isinstance(value, str) and paths.is_absolute_anywhere(value):
        return paths.file_name_anywhere(value)
    return value


def _copy_json(source, target):
    """Copy a .json or .jsonl record file, redacted. Returns whether it existed."""
    if not source.is_file():
        return False
    text = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        lines = [json.dumps(_redacted(json.loads(line))) for line in text.splitlines()
                 if line.strip()]
        text = "\n".join(lines) + ("\n" if lines else "")
    else:
        text = json.dumps(_redacted(json.loads(text)), indent=2) + "\n"
    _write_text(target, text)
    return True


def _names_only(df):
    """df with every PATH_COLUMNS column reduced to file names."""
    for column in PATH_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(
                lambda value: paths.file_name_anywhere(value)
                if isinstance(value, str) else value)
    return df


def _copy_table(source, target):
    """Copy a parquet table, path columns reduced. Returns whether it existed."""
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    _names_only(pd.read_parquet(source)).to_parquet(target, index=False)
    return True


def _copy_csv(source, target):
    """Copy an export CSV, byte for byte unless it has a path column to reduce."""
    target.parent.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(source, nrows=0).columns
    if not set(PATH_COLUMNS) & set(header):
        shutil.copy2(source, target)
        return
    df = pd.read_csv(source, dtype=str, keep_default_na=False, na_values=[""])
    _names_only(df).to_csv(target, index=False)


def _copy_database(source, target):
    """
    Copy the metric database as one self-contained file. Returns whether it existed.

    Through sqlite's backup API, so anything still in the write-ahead log is
    included, then switched out of WAL mode, which a reader on read-only media
    couldn't open.
    """
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(source)) as src, \
            contextlib.closing(sqlite3.connect(target)) as dst:
        src.backup(dst)
        dst.execute("PRAGMA journal_mode=DELETE")
    return True


def _copy_scripts(scripts, target):
    """Copy a folder of scripts, or a list of files, into target."""
    if isinstance(scripts, (str, Path)) and Path(scripts).is_dir():
        shutil.copytree(scripts, target, ignore=shutil.ignore_patterns("__pycache__"))
        return
    files = [scripts] if isinstance(scripts, (str, Path)) else list(scripts)
    target.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, target / Path(file).name)


def _environment():
    """critterframe, Python and installed library versions."""
    from .. import __version__

    versions = {}
    for name in DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {"critterframe": __version__, "python": sys.version.split()[0],
            "platform": platform.platform(), "packages": versions}


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _counts(dest):
    """What the README reports: occurrences, masks per part, runs, metric values."""
    counts = {}
    occurrences = dest / paths.OCCURRENCES_FILE
    if occurrences.exists():
        counts["occurrences"] = pq.ParquetFile(occurrences).metadata.num_rows
    masks = dest / paths.MASKS_FILE
    if masks.exists():
        counts["masks per part"] = pd.read_parquet(masks, columns=["part"])[
            "part"].value_counts().sort_index().to_dict()
    database = dest / paths.RUNS_AND_METRICS_FILE
    if database.exists():
        with contextlib.closing(sqlite3.connect(database)) as connection:
            counts["runs"] = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            counts["metric values"] = connection.execute(
                "SELECT COUNT(*) FROM metrics").fetchone()[0]
    return counts


# One line per item a package can hold.
_DESCRIPTIONS = {
    paths.OCCURRENCES_FILE: "one row per occurrence; `image_url` and any license "
                            "columns identify each image",
    paths.MASKS_FILE: "one mask per occurrence-part (see Reading the files)",
    paths.REFERENCE_MASKS_FILE: "reference masks, same layout, for validation",
    paths.CALIBRATIONS_FILE: "scale and other calibrations, with provenance",
    paths.RUNS_AND_METRICS_FILE: "every run's recipe and every metric value",
    paths.RUNS_LOG_FILE: "one line per finished run",
    f"{paths.RAW_IMPORTS_DIR}/": "import manifests: every decision that turned "
                                 "raw source data into occurrences",
    f"{paths.EXPORTS_DIR}/": "exported tables, each with a `.export.json` manifest",
    f"{paths.DEFINITIONS_DIR}/": "named subsets and project recipes",
    f"{paths.MODELS_DIR}/{paths.MODELS_REGISTRY_FILE}": "registered models with "
                                                        "checkpoint fingerprints",
    "code/": "the scripts that ran the pipeline",
}


def _readme(name, dest, included, environment, source_doi):
    """The package's README.md."""
    dois = [source_doi] if isinstance(source_doi, str) else list(source_doi or [])
    lines = [
        f"# {name}",
        "",
        f"A CritterFrame project archived {datetime.now(timezone.utc).date()} with "
        f"critterframe {environment['critterframe']} (Python {environment['python']}; "
        "library versions in `environment.json`).",
        "",
        "## Contents",
        "",
        *[f"- `{item}` -- {_DESCRIPTIONS[item]}" for item in included],
        "- `environment.json` -- software versions",
        "",
        "## Not included",
        "",
        "- Images: licensing and size. Each is identified by `image_url` in "
        "`occurrences.parquet`.",
        "- Raw source data: its own license applies. "
        + ("Cite: " + ", ".join(dois) if dois else "See the import manifests for sources."),
        "- Working files: mask shards, the failure ledger, visualizations, model checkpoints.",
        "- Local paths in records are reduced to file names.",
        "",
        "## Reading the files",
        "",
        "Masks are COCO run-length encoded, in the coordinates of the original "
        "image; `rle_counts` holds pycocotools' compressed counts as bytes:",
        "",
        "```python",
        "from pycocotools import mask as mask_utils",
        "rle = {\"counts\": bytes(row[\"rle_counts\"]), "
        "\"size\": [int(row[\"rle_height\"]), int(row[\"rle_width\"])]}",
        "mask = mask_utils.decode(rle).astype(bool)",
        "```",
        "",
        "`runs_and_metrics.sqlite` holds tables `runs` (recipe and context as JSON), "
        "`metrics` (one row per value, `value_json` as JSON; unit `transform_info` marks "
        "what a transform recorded) and `current_recipes`. Each export CSV's "
        "`.export.json` says which runs, recipes and masks its columns came from.",
        "",
        "## Counts",
        "",
        *[f"- {key}: {value}" for key, value in _counts(dest).items()],
        "",
    ]
    return "\n".join(lines)
