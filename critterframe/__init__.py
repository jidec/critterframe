"""CritterFrame: every function used to compose and inspect a pipeline, importable from the package itself."""

__version__ = "0.1.0"

# --- project ---------------------------------------------------------------
from .project.archive import archive_project
from .project.subsets import define_subset, define_subsets, grow_subset, load_subsets
from .project.summarize import describe_run, print_summary, summarize

# --- getting data in -------------------------------------------------------
from .calibrations.scale import (
    declare_scale,
    measure_scale_by_hand,
    measure_scales,
    scale_for_occurrences,
    scale_from_click,
    scale_from_target,
)
from .download import download_images
from .ingest import ingest_images, ingest_occurrences, load_imports

# --- recipe vocabulary -----------------------------------------------------
from .recipes import DEFAULT_PART

# --- segmentation ----------------------------------------------------------
from .records.masks import load_masks, merge_mask_shards, occurrence_ids_with_mask
from .segmentation.groundedsam import groundedsam2, sam2
from .segmentation.manual import correct_mask, draw_mask
from .segmentation.mask_import_export import export_masks, import_masks
from .segmentation.run import run_segments, segment

# --- transforms ------------------------------------------------------------
from .transforms.appendages import remove_appendages
from .transforms.crop import crop, crop_to_mask, remove_background, resize, rotate
from .transforms.erode import erode
from .transforms.islands import remove_islands
from .transforms.orient import orient

# --- metrics ---------------------------------------------------------------
from .metrics.annotation import (
    click_two_points,
    exclusive_label_annotation,
)
from .metrics.color_clusters import color_clusters
from .metrics.color_means import (
    background_color,
    mean_color,
    mean_lightness,
    white_balanced_color,
)
from .metrics.color_thresholds import (
    black_fraction,
    color_presence,
    color_threshold,
    hue_fraction,
    hue_thresholds,
    red_fraction,
    threshold_fractions,
    yellow_fraction,
)
from .metrics.derived import derived
from .metrics.label_score import label_score
from .metrics.embedding import embedding
from .metrics.inductive_color_thresholds import inductive_color_thresholds
from .metrics.islands import n_islands
from .metrics.mask_info import mask_info
from .metrics.dimensions import (
    body_length,
    bounding_box,
    elongation,
    jaggedness,
    length,
    mask_area,
    max_width,
)
from .metrics.outliers import cluster, outlier
from .metrics.position import centroid, image_bounds, relative_position
from .metrics.quality import (
    bilateral_asymmetry,
    blur_variance,
    edge_fraction,
    mask_fraction,
)
from .metrics.run import run_metrics

# --- training data and trained models --------------------------------------
from .records.models import list_models, load_model, register_model, unregister_model
from .training.datasets import export_training_data
from .training.splits import split_ids

# --- run history -------------------------------------------------------
from .records.metrics import load_metrics
from .records.runs import load_runs

# --- getting data out ------------------------------------------------------
from .export import (
    completed_ids,
    exemplars_per_group,
    export_metrics,
    export_units,
    load_exports,
    occurrences_matching,
)
from .selectionhelpers import (
    grow_sample,
    sample_occurrences,
    sample_per_group,
    shard_occurrences,
)
from .visualization.grids import comparison_grid, image_grid
from .visualization.products import render_segments

# --- validation ------------------------------------------------------------
from .validation.filters import (
    audit_filters,
    get_validated_filters,
    suggest_threshold,
    sweep_thresholds,
)
from .validation.masks import validate_masks
from .validation.metrics import compare_metrics

__all__ = [
    "DEFAULT_PART",
    "archive_project",
    "audit_filters",
    "background_color",
    "bilateral_asymmetry",
    "black_fraction",
    "blur_variance",
    "body_length",
    "bounding_box",
    "centroid",
    "click_two_points",
    "cluster",
    "color_clusters",
    "color_presence",
    "color_threshold",
    "compare_metrics",
    "comparison_grid",
    "completed_ids",
    "correct_mask",
    "crop",
    "crop_to_mask",
    "declare_scale",
    "define_subset",
    "define_subsets",
    "derived",
    "describe_run",
    "download_images",
    "draw_mask",
    "edge_fraction",
    "elongation",
    "embedding",
    "erode",
    "exclusive_label_annotation",
    "exemplars_per_group",
    "export_masks",
    "export_metrics",
    "export_training_data",
    "export_units",
    "get_validated_filters",
    "groundedsam2",
    "grow_sample",
    "grow_subset",
    "hue_fraction",
    "hue_thresholds",
    "image_bounds",
    "image_grid",
    "import_masks",
    "inductive_color_thresholds",
    "ingest_images",
    "ingest_occurrences",
    "jaggedness",
    "label_score",
    "length",
    "list_models",
    "load_exports",
    "load_imports",
    "load_masks",
    "load_metrics",
    "load_model",
    "load_runs",
    "load_subsets",
    "mask_area",
    "mask_fraction",
    "mask_info",
    "max_width",
    "mean_color",
    "mean_lightness",
    "measure_scale_by_hand",
    "measure_scales",
    "merge_mask_shards",
    "n_islands",
    "occurrence_ids_with_mask",
    "occurrences_matching",
    "orient",
    "outlier",
    "print_summary",
    "red_fraction",
    "register_model",
    "relative_position",
    "remove_appendages",
    "remove_background",
    "remove_islands",
    "render_segments",
    "resize",
    "rotate",
    "run_metrics",
    "run_segments",
    "sam2",
    "sample_occurrences",
    "sample_per_group",
    "scale_for_occurrences",
    "scale_from_click",
    "scale_from_target",
    "segment",
    "shard_occurrences",
    "split_ids",
    "suggest_threshold",
    "summarize",
    "sweep_thresholds",
    "threshold_fractions",
    "unregister_model",
    "validate_masks",
    "white_balanced_color",
    "yellow_fraction",
]
