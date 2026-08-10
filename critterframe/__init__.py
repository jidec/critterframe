"""
CritterFrame: images to traits (and image frames to dataframes).

A pipeline is five calls:

    import critterframe as cf

    cf.ingest_occurrences(project_path, "my_export.csv")
    cf.download_images(project_path)

    cf.run_segments(project_path, steps=[cf.segment(cf.groundedsam2())])

    cf.run_metrics(
        project_path,
        run_name="traits",
        transforms=[cf.remove_appendages()],
        metrics=[cf.mean_lightness()],
    )

    cf.export_metrics(project_path, "traits.csv")

Everything a pipeline script needs is re-exported here, so scripts read as a
description of the processing rather than as a list of imports. The modules
underneath stay importable directly for anything deeper -- writing a new
operation, inspecting run records, building a training set.

The vocabulary, in the order it matters:

  Project     -- one self-contained analytical dataset: occurrences, images,
                 masks, metrics, definitions, provenance. A directory.
  Occurrence  -- ONE focal organism at a place and time, as one analysis image
                 plus metadata. One organism per image; images with several
                 must be separated upstream of ingest.
  Part        -- a named component of that organism: the whole organism by
                 default, or head, abdomen, forewing. Any number per occurrence.
  Mask        -- at most one current mask per occurrence-part, always in the
                 coordinates of the original image.
  Segment     -- image plus current mask, in flight. Never persisted.
  Metric      -- any derived value: a trait, a QC score, a human label, an
                 embedding, a cluster assignment, an outlier score.
  Transform   -- an operation that changes the working segment without
                 producing a value.
  Operation   -- one configured processing action: remove_appendages(),
                 segment(groundedsam2()), mean_lightness().
  Recipe      -- an immutable, hashable specification of ordered operations
                 and their inputs. Identity covers operation order, parameters,
                 versions, model checkpoints, and upstream dependencies.
  Run         -- one execution of a recipe over a set of occurrences.
  Subset      -- a named selection of occurrences receiving a particular recipe.
  Filter      -- a rule for selecting occurrences at export. Never deletes.

Two properties are worth knowing before you rely on anything else. Processing
is repeat-aware: work already done by an equivalent recipe is skipped, so runs
are interruptible and resumable and an expensive metric behaves like cached
derived data. And persisted masks are always in original image coordinates,
whatever crops and rotations produced them, so parts and whole organisms are
always comparable.
"""

__version__ = "0.1.0"

# --- project ---------------------------------------------------------------
from .project.subsets import define_subset, define_subsets, load_subsets
from .project.summarize import print_summary, summarize

# --- getting data in -------------------------------------------------------
from .calibrations.scale import (
    declare_scale,
    measure_scales,
    scale_for_occurrences,
    scale_from_target,
)
from .download import download_images
from .ingest import ingest_images, ingest_occurrences

# --- recipe vocabulary -----------------------------------------------------
from .recipes import DEFAULT_PART, Metric, Recipe, Segment, Segmentation, Transform

# --- segmentation ----------------------------------------------------------
from .segmentation.groundedsam import GroundedSAM2, groundedsam2, sam2
from .segmentation.manual import correct_mask, draw_mask
from .segmentation.run import run_segments, segment

# --- transforms ------------------------------------------------------------
from .transforms.appendages import remove_appendages
from .transforms.crop import crop, crop_to_mask, remove_background, resize, rotate
from .transforms.orient import orient

# --- metrics ---------------------------------------------------------------
from .metrics.annotation import annotate_flags, click_two_points
from .metrics.color import (
    black_fraction,
    hue_fraction,
    mean_color,
    mean_lightness,
    red_fraction,
    yellow_fraction,
)
from .metrics.dimensions import body_length, bounding_box, length, mask_area, max_width
from .metrics.outliers import ClusterMetric, GroupMetric, OutlierMetric, cluster, outlier
from .metrics.position import centroid, image_bounds, relative_position
from .metrics.quality import (
    bilateral_asymmetry,
    blur_variance,
    edge_fraction,
    mask_fraction,
)
from .metrics.run import run_metrics

# --- getting data out ------------------------------------------------------
from .export import export_metrics, export_units
from .selectionhelpers import occurrences_matching, sample_occurrences
from .visualization.grids import comparison_grid, image_grid
from .visualization.panels import PanelFiles
from .visualization.products import render_segments

# --- validation ------------------------------------------------------------
from .validation.filters import (
    get_validated_filters,
    suggest_threshold,
    sweep_thresholds,
)
from .validation.masks import validate_masks
from .validation.metrics import compare_metrics

__all__ = [
    "DEFAULT_PART",
    "GroundedSAM2",
    "Metric",
    "PanelFiles",
    "Recipe",
    "Segment",
    "Segmentation",
    "Transform",
    "ClusterMetric",
    "GroupMetric",
    "OutlierMetric",
    "annotate_flags",
    "bilateral_asymmetry",
    "black_fraction",
    "blur_variance",
    "body_length",
    "bounding_box",
    "centroid",
    "click_two_points",
    "cluster",
    "compare_metrics",
    "comparison_grid",
    "correct_mask",
    "crop",
    "crop_to_mask",
    "declare_scale",
    "define_subset",
    "define_subsets",
    "download_images",
    "draw_mask",
    "edge_fraction",
    "export_metrics",
    "export_units",
    "get_validated_filters",
    "groundedsam2",
    "hue_fraction",
    "image_bounds",
    "image_grid",
    "ingest_images",
    "ingest_occurrences",
    "length",
    "load_subsets",
    "mask_area",
    "mask_fraction",
    "max_width",
    "mean_color",
    "mean_lightness",
    "measure_scales",
    "occurrences_matching",
    "orient",
    "outlier",
    "print_summary",
    "red_fraction",
    "relative_position",
    "remove_appendages",
    "remove_background",
    "render_segments",
    "resize",
    "rotate",
    "run_metrics",
    "run_segments",
    "sam2",
    "sample_occurrences",
    "scale_for_occurrences",
    "scale_from_target",
    "segment",
    "suggest_threshold",
    "summarize",
    "sweep_thresholds",
    "validate_masks",
    "yellow_fraction",
]
