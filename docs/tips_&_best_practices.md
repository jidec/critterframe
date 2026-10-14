# Tips & Best Practices

How to work well with `critterframe`, beyond what any one function's reference says.

## Validation

Validation should be a part of every pipeline: critterframe supports validation at multiple levels, with worked examples under [reference annotation](examples/reusable_workflows/reference_annotation.md) and [validation](examples/reusable_workflows/validation.md)

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
