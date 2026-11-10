# Concepts

## `critterframe` project folder structure

```
my_project/
    occurrences.parquet         central imported/normalized metadata
    images.lmdb/                original images, one per occurrence, byte-exact
    masks.parquet               canonical masks, one per occurrence-part
    reference_masks.parquet     human-vetted or otherwise trusted masks
    calibrations.parquet        px/mm and the like, keyed by what was calibrated
    runs_and_metrics.sqlite     run records + the metric values they produced
    raw_imports/                immutable raw imports + imports.jsonl, what each became and why
    exports/                    exports.jsonl, what this project has handed out
    definitions/                subsets.toml, recipes.py
    visualizations/
        pipeline/               every activity's diagnostics: sampled grids, checkpoints, figures, .report.json
        products/               rendered assets, one file per occurrence-part
    models/                     registry.json + checkpoints trained for this project
```

## `critterframe` vocabulary

### Projects and data intake

| Term | Meaning | Related concepts in other packages |
|---|---|---|
| **Project**             | A self-contained collection of organismal occurrence images, metadata, and derivations intended to be analyzed as a coherent biological dataset and sharing at least some processing steps.                | phenopype `Project`; MLflow experiment                                                 |
| **Raw import**          | Source data preserved before any structural or judgment-based decisions are applied, such as a GBIF Darwin Core archive as downloaded. Archived in the `raw_imports/` folder.                              | Kedro 'raw' data layer                                                                 |
| **Import**              | A raw import reshaped into occurrences and narrowed through explicit inclusion decisions such as `drop`, `group_col`, and `max_per_group`. A manifest records those decisions.                             | Kedro 'primary' data layer                                                             |
| **Ingest**              | The act of bringing data into a project: `ingest_occurrences` turns a raw import into an import, `ingest_images` turns a folder of images into occurrences. A call that changes nothing records no import. | Kedro 'raw' to 'primary'; phenopype Project.add_files                                  |
| **Manifest**            | A record written alongside an import or export describing what it contains and the decisions that shaped it.                                                                               | Frictionless `datapackage.json`                                                        |

### Occurrences, parts and masks

| Term | Meaning | Related concepts in other packages |
|---|---|---|
| **Occurrence image**    | Image evidence of a focal organism existing at a particular place and time.                                                                                                                                | DarwinCore                                                                             |
| **Part**                | A consistently named biological component of an organism, such as `head`. The part representing the whole organism defaults to `organism`.                                                                 | MAPHIS hierarchical anatomical labels                                                  |
| **Mask**                | The canonical spatial representation of an occurrence-part in the original image coordinates. Each occurrence-part has at most one canonical mask.                                                         |                                                                                        |
| **Canonical**           | The status of the one mask per occurrence-part that metrics are measured from by default. Replacing it leaves metrics measured from the previous mask stale.                                 |                                                                                        |
| **Reference mask**      | A mask retained for comparison rather than treated as canonical.                                                                                                                                           |                                                                                        |
| **Segment**          | An image paired with its current mask. Segments are working state and are not persisted because their images and masks already are. | albumentations `image` + `mask` pair                                                               |
| **Subset**              | A persistent named selection of occurrences.                                                                                                                                                               | FiftyOne saved view; glue subset                                                       |

### Processing

| Term | Meaning | Related concepts in other packages |
|---|---|---|
| **Operation**        | One configured processing action, classified as `transform`, `segment`, or `metric` according to what it does to a segment.        | phenopype configured function in YAML; tidymodels `step_*()`                                       |
| **Transform**        | An operation that changes the working segment without producing a value, such as orientation normalization.                        | albumentations transforms; scikit-learn transformer                                                |
| **Recipe**           | A configured chain of operations, classified as `segment`, `metric`, or `render` according to the type of output it persists.      | phenopype configured analysis in YAML; tidymodels `recipe()`; scikit-learn `Pipeline` |
| **Recipe hash**      | A reproducible hash of a recipe’s operations that allows equivalent completed work to be recognized and skipped during reruns.     | targets hash                                                                                       |
| **Run**              | One execution of a recipe over a set of occurrences.                                                                               | MLflow run                                                                                         |
| **Pipeline**         | An ordered set of scripts taking a project from ingest to export, kept as the record of what was done to it.                        | Kedro pipeline; targets pipeline |
| **Registered model** | A model stored or referenced together with its provenance information.                                                             | MLflow model registry                                                                              |
| **Record**           | A persisted package datatype, including occurrences, masks, runs, metrics, calibrations, and models.                               |                                                                                                    |

### Run outcomes and currency

| Term | Meaning | Related concepts in other packages |
|---|---|---|
| **Current / stale**  | The status of a stored metric value: current while its source mask is canonical and its run name still points to its recipe, stale otherwise. Stale values are kept, not deleted. | targets outdated target |
| **No input**         | The status of an occurrence with nothing yet to work from, such as a missing image or mask. Neither a failure nor completed work, so it is attempted again on the next run. |  |
| **Failure**          | An occurrence that raised an error during a run. Recorded so an unchanged retry is skipped; a changed recipe or source mask retries it. |  |

### Metrics

| Term | Meaning | Related concepts in other packages |
|---|---|---|
| **Metric**              | Any derived value associated with an occurrence-part.                                                                                                                                                      |                                                                                        |
| **Trait**               | A metric representing a biologically meaningful property intended for later analysis.                                                                                                                      |                                                                                        |
| **QC metric**           | A metric describing the usability or reliability of an image, segment, transformation, or derived measurement.                                                                                             |                                                                                        |
| **Group metric**        | A metric derived jointly from multiple occurrences, such as a cluster assignment or outlier score.                                                                                                         | scikit-learn fit/predict (`KMeans`, `IsolationForest`); tidymodels `prep()` / `bake()` |
| **Stored-value metric** | A metric computed from stored metric values rather than from segment pixels, such as a derived metric, outlier score, or label score.                                                      |                                                                                        |
| **Derived metric**      | A metric computed for one occurrence-part from its own stored metric values, such as a ratio of two traits.                                                                                | dplyr `mutate()`                                                                       |
| **Color threshold**     | A named combination of cutoffs on color-channel values used to identify qualifying pixels.                                                                                                                 | charisma colour lookup table (CLUT)                                                    |
| **Calibration**         | Knowledge about the imaging system used to convert metrics during export, such as a pixels-per-millimetre scale or ColorChecker-based color normalization.                                                 | ImageJ `Calibration`                                                                   |

### Annotation, filtering and validation

| Term | Meaning | Related concepts in other packages |
|---|---|---|
| **Annotation**          | A value supplied or reviewed by a person, such as a usability label, manual trait measurement, or reference mask.                                                                                          |                                                                                        |
| **Reference set**       | A collection of human-reviewed or otherwise trusted data used to evaluate a pipeline component.                                                                                                            |                                                                                        |
| **Stratified sampling** | Sampling separately within predefined groups, such as taxa or collections, to ensure that each is adequately represented.                                                                                  | scikit-learn `StratifiedShuffleSplit`; dplyr `slice_sample()` by group                 |
| **Validation**          | Comparison against a reference set to quantify how well a pipeline step performs.                                                                                                                          | FiftyOne `evaluate_segmentations()`; tidymodels `yardstick`                            |
| **Filter**              | A rule for selecting occurrences during export or downstream analysis.                                                                                                                                     | dplyr `filter()`                                                                       |

### Outputs and visualization

| Term | Meaning | Related concepts in other packages |
|---|---|---|
| **Export**              | A set of metrics & associated occurrence data where decisions about its contents and shape are based on downstream analysis intent.                                                                        | tidyr `pivot_wider()`; FiftyOne `export()`                                             |
| **Panel**            | An image showing one operation’s decision for one occurrence-part. Panels are the units from which visualizations are assembled.   | TensorBoard image summary; one panel of ggplot2 `facet_wrap()`                                     |
| **Report**           | A collection of panels, figures, and a summary file showing how one run or activity behaved, written to `visualizations/pipeline/`. | TensorBoard run log |
| **Render**           | A materialized image product often used as part of a figure.                                                                       | FiftyOne `to_patches()`                                                                            |
