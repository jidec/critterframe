# Guide

Which pipeline to start from, how to validate one, and what the extensions add.

## Example pipelines

Every script under `scripts/` is a pipeline written out flat: one explicit call per step, read top to
bottom as the record of what was done to a project. Each points at a `PROJECT_PATH` of its own, so change
that before running one.

The [Examples](examples/simplest_full_pipeline.md) tab shows them in full, built from the scripts themselves:

- **Start here:** [the simplest pipeline](examples/simplest_full_pipeline.md), every step once, end to end.
- **Reusable workflows**, one page per task most projects need at some point:
  [calibration](examples/calibration.md), [reference annotation](examples/reference_annotation.md),
  [a custom segmentation model](examples/custom_segmentation_model.md) and
  [validation](examples/validation.md).
- **[A full project](examples/full_project.md):** `scripts/author_pipelines/odonata_inat_obsorg/`, dragonfly
  bodies from iNaturalist observations published through GBIF, as eight numbered steps. Beside those steps
  the folder has `validate_groundedsam.py` (which GroundedSAM2 settings work best for step 1's organism
  segment) and `train_bioencoder.py` (train and register a BioEncoder embedding model on the organism
  segments).

`scripts/author_pipelines/` holds other whole projects as they were actually run, one folder each:

| Folder | What it shows |
| --- | --- |
| `mm2_moths/` | Light-trap insects preprocessed by Antenna. |
| `mothitor_moths/` | Light-trap insects from a Mothitor deployment, ingested through Antenna. |
| `carabidae_inat/` | Carabid beetles from iNaturalist, ingested from a GBIF archive. |
| `dragonfly_wings_museums_untested/` | Dragonfly wings from several museum collections: subsets and parts. Not yet run end to end. |
| `salamander_boxes_untested/` | Salamanders in experimental boxes: local images, a specialized segmenter, and position as the trait. Not yet run end to end. |

## Cross-Cutting Validation

Validation should be a part of every pipeline: critterframe supports validation at multiple levels with worked examples in `/scripts`

For reference set creation/annotation
- `correct_masks` or `manual_masks` to create a reference/ground truth mask set
- `exclusive_label_annotation` to give each item exactly one label from a vocabulary of your own, one keypress each, e.g. screening images as usable or not before any mask is drawn, or finished segments as good or why not (no valid input, wrong region, incomplete, overflow)
- Stratified sampling of reference sets across grouping columns (e.g. taxa, collections)

For segmentation:
- `validate_masks` to directly compare a segmentation model to a reference set

For traits:
- `compare_metrics` to assess agreement between an automatically computed trait and a human-measured one

For the validity/quality of final outputs (AKA noise filtering):
- `get_validated_filters` that given candidate columns (e.g. seg model confidence, transform reliability flags) and project-scoped negative image cases (e.g. blurry, cutoff) computes filtering columns & values that screen out invalid or low quality examples under different coverage-quality tradeoffs. Candidates from several runs go in one call; a categorical one such as a cluster assignment has its worst categories dropped within the same budget, i.e. picking clusters by their labels rather than by eye; a candidate that catches nothing is left out; what the chosen filters do together is logged, & on held-out labels too with `audit_subset=`
- `audit_filters` that scores any filter set, e.g. one written by hand, on labels it was not calibrated on: the share of bad rows before & after filtering and the share of rows kept, each with an interval
- `label_score` to fit a score directly to the labels (e.g. on embeddings) & threshold it like any other candidate
- Certain group-level metrics (e.g. outlier labels/scores, cluster assignments) are especially useful filtering candidates

Accuracy & validity are separate claims. Reference masks exist only where a correct mask can be drawn, so `validate_masks` says how accurate a segment is on a valid input. Invalid inputs still reach every later step, so what share of an export is bad needs a label that exists for every input: screen a random sample of finished segments, calibrate filters on one part of it & audit them on the rest.

For the whole pipeline:
- Coming soon - "sensitivity analysis" exports across defensible alternative metric parameters and filtering tradeoffs
- Coming soon - bootstrap uncertainty intervals for validation results

## Extensions

Extensions are typically for handling specific data sources, very specialized metrics, or trainable models
not general enough to bundle in core. By convention they mirror the package layout.

- **`antenna_lighttraps`** — light-trap camera monitoring. Scale calibration is scoped per trap night (`event_id`) rather than per occurrence.
- **`bioencoder`** — dataset preparation for training a metric-learning model, and a loader for
  checkpoints the BioEncoder package trained (by path or by its own YAML config), feeding core's
  `embedding()`. The training loop itself is deliberately left to the BioEncoder package.
- **`gbif_darwincore_inat`** — a GBIF Darwin Core Archive, one photo per occurrence. The way to pull
  iNaturalist observations: iNat photo sizes, cross-source deduplication, and `prioritize_inat` for mixed pulls.
- **`smp_segmenter`** — a trainable UNet++ segmenter (segmentation_models_pytorch) for refining or replacing
  the bundled zero-shot segmenter on one project's own masks. Unlike `bioencoder`'s scaffold, its
  `train()` is a real, working training loop, not a stub -- binary mask segmentation doesn't carry the same
  dataset-dependent backbone/loss judgment calls that make guessing at a metric-learning setup risky.
