"""CritterFrame: every function used to compose and inspect a pipeline, importable from the package itself."""

__version__ = "0.1.0"

# --- project ---------------------------------------------------------------
from .project.archive import archive_project
from .project.summarize import describe_run, print_summary, summarize
from .selection.subsets import define_subset, define_subsets, grow_subset, load_subsets, select_ids

# --- getting data in -------------------------------------------------------
from .calibrations.scale import (
    declare_scale,
    measure_scale_by_hand,
    measure_scales,
    scale_for_occurrences,
    scale_from_click,
    scale_from_target,
)
from .ingest.imports import load_imports
from .ingest.download import download_images
from .ingest.images import ingest_images
from .ingest.occurrences import ingest_occurrences

# --- recipe vocabulary -----------------------------------------------------
from .core.recipes import DEFAULT_PART

# --- segmentation ----------------------------------------------------------
from .records.masks import load_masks, merge_mask_shards
from .segmentation.groundedsam import groundedsam2, sam2
from .segmentation.manual import correct_mask, draw_mask
from .segmentation.mask_import_export import export_masks, import_masks
from .segmentation.run_segments import run_segments, segment

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
from .metrics.color.clusters import color_clusters
from .metrics.color.means import (
    background_color,
    mean_color,
    mean_lightness,
    white_balanced_color,
)
from .metrics.color.thresholds import (
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
from .metrics.color.inductive_thresholds import inductive_color_thresholds
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
from .metrics.clusters import cluster
from .metrics.outliers import outlier
from .metrics.position import centroid, image_bounds, relative_position
from .metrics.quality import (
    bilateral_asymmetry,
    blur_variance,
    edge_fraction,
    mask_fraction,
)
from .metrics.run_metrics import run_metrics

# --- training data and trained models --------------------------------------
from .records.models import list_models, load_model, register_model, unregister_model
from .training.datasets import export_training_data
from .training.splits import split_ids

# --- run history -------------------------------------------------------
from .records.metrics import load_metrics
from .records.runs import load_runs

# --- getting data out ------------------------------------------------------
from .export import export_metrics, export_units, load_exports
from .selection.queries import (
    exemplars_per_group,
    ids_completed,
    ids_matching,
    ids_passing,
    ids_with_image,
    ids_with_mask,
)
from .selection.algorithms import (
    grow_sample,
    sample_ids,
    sample_per_group,
    shard_ids,
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
    "ids_completed",
    "ids_matching",
    "ids_passing",
    "ids_with_image",
    "ids_with_mask",
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
    "sample_ids",
    "sample_per_group",
    "scale_for_occurrences",
    "scale_from_click",
    "scale_from_target",
    "segment",
    "select_ids",
    "shard_ids",
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
