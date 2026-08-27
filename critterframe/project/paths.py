"""
Project layout: every path in a CritterFrame project derives from one project
directory

A project is one self-contained analytical dataset -- occurrences, images,
masks, metrics, processing definitions, and provenance

    my_project/
        occurrences.parquet         central imported/normalized metadata
        images.lmdb/                original images, one per occurrence
        masks.parquet               canonical masks, one per occurrence-part
        reference_masks.parquet     human-vetted or otherwise trusted masks
        calibrations.parquet        px/mm and the like, keyed by what was calibrated
        runs_and_metrics.sqlite     run records + metric values
        imports/                    immutable source imports
        definitions/                subsets.toml, recipes.py
        visualizations/
            pipeline/               one sampled QC sheet per run
            products/               rendered assets, one file per occurrence-part
        models/                     registry.json + checkpoints trained here

Nothing here creates anything. A project comes into existence lazily: every
writer in the package makes the directory it needs on its way to writing, so
ingesting a table produces a project holding an occurrence table and nothing
else -- no empty mask file implying segmentation already ran, no visualizations
folder for diagnostics nobody asked for. What a project directory contains is
therefore an honest account of what has actually been done to it, which is also
what summarize reads.

Every function returns a pathlib.Path.
"""

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

    Every other function here goes through this, so a caller may pass a string
    or a Path anywhere a project_path is taken and get the same result.
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

    reference -- False (default) returns the canonical processing mask table
                 (one current mask per occurrence-part). True returns the
                 reference table instead, which has an identical schema and
                 is allowed to coexist with the canonical one -- validation
                 is comparison between the two, not a separate processing
                 system.
    """
    return project_dir(project_path) / (
        REFERENCE_MASKS_FILE if reference else MASKS_FILE
    )


def mask_shards_dir(project_path, part="", reference=False):
    """
    Staging area for a sharded run_segments() call's mask writes.

    A sharded run (shard=(index, total)) never writes masks.parquet
    directly -- upsert_table's whole-file read-merge-overwrite has no
    locking, so two concurrent workers rewriting it at once would silently
    lose each other's rows. It writes a new, uniquely-named file here on
    every flush instead (see records.masks.save_mask_shard) -- an operation
    safe for any number of concurrent writers, since no two of them ever
    touch the same file. Nothing here is meaningful on its own; see
    records.masks.merge_mask_shards, the one place that reads it back.

    part -- narrows to one part's shards, matching how a sharded run flushes
            each output part into its own subdirectory. "" (default) is the
            root both parts share, useful for listing which parts have
            anything staged.
    """
    return project_dir(project_path) / MASK_SHARDS_DIR / \
        ("reference" if reference else "canonical") / part


def calibrations_path(project_path):
    """
    What is known about the imaging system rather than about any organism:
    pixels per millimetre, and in time colour correction and whatever else a
    reference in the frame can establish.

    One table for all of them, keyed by what an imaging setup holds constant --
    one occurrence, one deployment session, one copy stand. A calibration isn't
    a property of an occurrence: several occurrences imaged in one sitting share
    one, and the sitting is the thing that was calibrated (see
    records.calibrations).
    """
    return project_dir(project_path) / CALIBRATIONS_FILE


def runs_and_metrics_path(project_path):
    """The sqlite database holding run records and the metric values they produced."""
    return project_dir(project_path) / RUNS_AND_METRICS_FILE


def imports_dir(project_path):
    """Immutable source imports -- the recovery path if an ingest was ever wrong."""
    return project_dir(project_path) / IMPORTS_DIR


def definitions_dir(project_path):
    """Project subsets/recipes/config -- the hand-edited part of a project."""
    return project_dir(project_path) / DEFINITIONS_DIR


def subsets_path(project_path):
    """Named selections of occurrences that receive a particular recipe."""
    return definitions_dir(project_path) / SUBSETS_FILE


def recipes_path(project_path):
    """
    Optional project-local Python module where a user keeps the recipes they
    run repeatedly, so a pipeline script can import them instead of respelling
    the operation list every time. Nothing in the package imports this for you
    -- it exists so a project carries its own processing definitions alongside
    its data.
    """
    return definitions_dir(project_path) / RECIPES_FILE


def visualizations_dir(project_path, subdir=""):
    """
    Diagnostic images and figure material.

    Two subfolders have a fixed meaning -- pipeline_dir and products_dir -- and
    that's where a run's output goes. Anything else here is a caller's own,
    named by whoever wrote it (visualization.panels.save_panel), which is where
    per-panel files land for a segment you built and pointed at a PanelFiles
    sink yourself.
    """
    return project_dir(project_path) / VISUALIZATIONS_DIR / subdir


def pipeline_dir(project_path):
    """
    Sample-level visual summaries of how processing BEHAVED: one sheet per run,
    a sampled handful of occurrences laid out so a person can judge a step at a
    glance.

    Bounded by design. A project of 10,000 occurrences produces one image per
    run here, not 10,000 -- the question "is this step working" is answered by
    looking at a representative sample, and a folder holding one file per
    occurrence per operation is a folder nobody opens.
    """
    return visualizations_dir(project_path, PIPELINE_DIR)


def products_dir(project_path, name=""):
    """
    Visual assets deliberately materialized for downstream use: one file per
    occurrence-part, in a folder per render.

    The opposite contract to pipeline_dir. These are outputs, not diagnostics --
    figure panels, per-specimen plates, images fed to another tool -- so they
    are one-file-per-thing with the occurrence id in the filename, which is what
    makes them joinable back to an exported trait table by anything that can
    read a directory listing (R very much included).

    Loose files rather than the image store on purpose: the store holds original
    analysis images byte-exactly and is addressed by occurrence id from Python.
    A render is derived, disposable, and usually wanted by something that isn't
    Python.
    """
    return visualizations_dir(project_path, PRODUCTS_DIR) / name


def models_dir(project_path):
    """Named custom segmenters/checkpoints belonging to this project."""
    return project_dir(project_path) / MODELS_DIR


def models_registry_path(project_path):
    """
    What is known about the models this project uses: name, checkpoint
    fingerprint, task, base model, the training data behind it (see
    records.models).

    JSON rather than parquet because it holds a handful of rows of deeply
    nested, per-framework provenance, and rather than TOML because nothing
    hand-edits it -- unlike subsets.toml, which exists precisely to be edited.
    Beside the checkpoints rather than in definitions/, since a registered
    model is a record OF the files here.
    """
    return models_dir(project_path) / MODELS_REGISTRY_FILE


def require_project(project_path):
    """
    Raise unless project_path holds an occurrence table, and return it as a
    Path.

    Called before anything that READS existing state, so a typo'd path fails
    loudly instead of quietly reporting an empty project. The occurrence table
    is the right thing to check for: it's what ingest writes first, and nothing
    else in the pipeline is meaningful without it.

    Writers deliberately don't call this -- ingest has to work on a directory
    that doesn't exist yet, which is what makes projects lazy.
    """
    directory = project_dir(project_path)
    if not occurrences_path(directory).exists():
        raise FileNotFoundError(
            f"{directory} isn't a CritterFrame project -- no "
            f"{OCCURRENCES_FILE}. Run critterframe.ingest_occurrences() or "
            "ingest_images() to create one."
        )
    return directory
