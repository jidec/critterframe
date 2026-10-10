# Concepts

What a `critterframe` project is made of, and the words the package uses for it.

## The project structure

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

### Core concepts

| Term                    | Meaning                                                                                                                                                                                     |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Project**             | A self-contained collection of organismal occurrence images, metadata, and derivations intended to be analyzed as a coherent biological dataset and sharing at least some processing steps. |
| **Raw import**          | Source data preserved before any structural or judgment-based decisions are applied, such as a GBIF Darwin Core archive as downloaded. Archived in the `raw_imports/` folder.               |
| **Import**              | A raw import reshaped into occurrences and narrowed through explicit inclusion decisions such as `drop`, `group_col`, and `max_per_group`. A manifest records those decisions.              |
| **Ingest**              | The act of bringing data into a project: `ingest_occurrences` turns a raw import into an import, `ingest_images` turns a folder of images into occurrences. A call that changes nothing records no import.|
| **Occurrence image**    | Image evidence of a focal organism existing at a particular place and time.                                                                                                                 |
| **Subset**              | A named selection of occurrences.                                                                                                                                                           |
| **Part**                | A consistently named biological component of an organism, such as `head`. The part representing the whole organism defaults to `organism`.                                                  |
| **Mask**                | The canonical spatial representation of an occurrence-part in the original image coordinates. Each occurrence-part has at most one canonical mask.                                          |
| **Metric**              | Any derived value associated with an occurrence-part.                                                                                                                                       |
| **Trait**               | A metric representing a biologically meaningful property intended for later analysis.                                                                                                       |
| **QC metric**           | A metric describing the usability or reliability of an image, segment, transformation, or derived measurement.                                                                              |
| **Group metric**        | A metric derived jointly from multiple occurrences, such as a cluster assignment or outlier score.                                                                                          |
| **Color threshold**     | A named combination of cutoffs on color-channel values used to identify qualifying pixels.                                                                                                  |
| **Calibration**         | Knowledge about the imaging system used to convert metrics during export, such as a pixels-per-millimetre scale or ColorChecker-based color normalization.                                  |
| **Filter**              | A rule for selecting occurrences during export or downstream analysis.                                                                                                                      |
| **Annotation**          | A value supplied or reviewed by a person, such as a usability label, manual trait measurement, or reference mask.                                                                           |
| **Reference mask**      | A mask retained for comparison rather than treated as canonical.                                                                                                                            |
| **Reference set**       | A collection of human-reviewed or otherwise trusted data used to evaluate a pipeline component.                                                                                             |
| **Stratified sampling** | Sampling separately within predefined groups, such as taxa or collections, to ensure that each is adequately represented.                                                                   |
| **Validation**          | Comparison against a reference set to quantify how well a pipeline step performs.                                                                                                           |
| **Export**              | A set of metrics & associated occurrence data where decisions about its contents and shape are based on downstream analysis intent.                                                         |

### Processing terms

| Term                 | Meaning                                                                                                                             |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Registered model** | A model stored or referenced together with its provenance information.                                                              |
| **Segment**          | An image paired with its current mask. Segments are working state and are not persisted because their images and masks already are. |
| **Transform**        | An operation that changes the working segment without producing a value, such as orientation normalization.                         |
| **Operation**        | One configured processing action, classified as `transform`, `segment`, or `metric` according to what it does to a segment.         |
| **Recipe**           | A configured chain of operations, classified as `segment`, `metric`, or `render` according to the type of output it persists.       |
| **Recipe hash**      | A reproducible hash of a recipe’s operations that allows equivalent completed work to be recognized and skipped during reruns.      |
| **Run**              | One execution of a recipe over a set of occurrences.                                                                                |
| **Record**           | A persisted package datatype, including occurrences, masks, runs, metrics, calibrations, and models.                                |
| **Panel**            | An image showing one operation’s decision for one occurrence-part. Panels are the units from which visualizations are assembled.    |
| **Render**           | A materialized image product.                                                                                                       |
