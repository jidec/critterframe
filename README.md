# critterframe

```text
  .   .
   \ /
   ⌐■-■
 ~/ ! \~   Image frames of critters into dataframes of traits
~| o:o |~
 | o:o |
/ \_:_/ \
```

critterframe is a flexible Python package built for large-scale organismal image processing, including easy-to-understand tools to:

1. **Ingest** occurrence data and images into the framework
```python
cf.ingest_occurrences("C:/my_project", "specimens.csv", id_col="specimen_id", image_url_col="image_url")
cf.download_images("C:/my_project")
```

2. **Segment** each organism out of its occurrence image
```python
cf.run_segments("C:/my_project", steps=[cf.segment(cf.groundedsam2())])
```

3. **Extract** and export traits from segmented organisms
```python
cf.run_metrics("C:/my_project", run_name="traits",
                transforms=[cf.remove_appendages()], metrics=[cf.mean_lightness()])
cf.export_metrics("C:/my_project", "traits.csv")
```

Plus filtering, validation, human annotation, training dataset export & custom model import, support for organismal parts (i.e. head, thorax, abdomen), scale & color calibration, group-derived metrics (i.e. cluster assignments, outlier detection), 
and more — see [scripts/](scripts/) for full examples.

## The framework

1. **One focal organism per image**

   > **Why:** A single organism is the natural unit for organismal image analysis and maps cleanly onto an occurrence. We see this as a worthy simplification that removes a complex layer of bookkeeping. Tools are provided to convert multi-organism images into one-organism images for import.

2. **One canonical mask per organism or organism-part**

   > **Why:** Virtually all analyses need the best available representation of a biological part, not a growing collection of competing masks. Alternative segmentation approaches can be evaluated against reference masks before deciding which should become canonical.

3. **All derived values are metrics - whether traits, QC scores, or annotations**

   > **Why:** A common metric model lets the same machinery support biological measurements, quality control, validation, clustering results, embeddings, and lots more.

4. **Filtering is selection, not deletion**

   > **Why:** Filtering criteria are analytical decisions that may change as a project develops. We apply filters during export or post-critterframe analysis rather than removing data from the processing pipeline.

## The pipeline

1. **Small operations compose flexibly into complex pipelines**

   > **Why:** The same operations support both simple pipelines (i.e. foundational segmentation & thresholded color extraction) and complex ones (i.e. with custom part segmentation, filtering via outlier detection, metric learning embeddings, etc.).

2. **Idempotent, resumable, & evolvable**

   > **Why:** Large image datasets are expensive to process and continually evolve. Running the same recipe twice does no extra work and produces no extra data. Interrupted runs can continue. Incorporating new data is as simple as importing a new snapshot with the new data included.

3. **Full derivation provenance**

   > **Why:** Every mask and metric is linked to how it was produced and what inputs it depended on. Analyses are traceable & reproducible by default.

## Convenience features

1. Project folders are portable records with data & provenance ready for archiving alongside a publication
2. Nearly every pipeline step can be visualized with `visualize=True
3. Persistent named subsets make it easy to pass data around for validation, training, or subset-specific processing
4. Metrics exports designed for easy analysis post-critterframe
5. Multithreading for image downloading, segmentation, and metric runs

## Documentation
[jidec.github.io/critterframe](https://jidec.github.io/critterframe/)

## Installation
```
pip install "critterframe[torch] @ git+https://github.com/jidec/critterframe.git"
```

`[torch]` pulls in the ready-to-go SAM2/GroundedSAM2 torch models essential for many projects (as well as some deep-learning-based metrics), installing PyPI's default torch build for your platform if you don't yet have torch installed.

Installing [CUDA](https://developer.nvidia.com/cuda-downloads) beforehand to enable processing on GPU is recommended (GroundedSAM2 can fall back to CPU but becomes very slow).

If you need a torch version that fits a certain CUDA version (such as the version that exists on a computing cluster)
install [that torch version](https://pytorch.org/get-started/locally/) before installing `critterframe[torch]` (critterframe's `torch` dependency carries no version pin so it won't be overridden).

Projects segmenting with a custom model or working from existing masks (and not using deep-learning based metrics) don't need `torch`:
```
pip install "critterframe @ git+https://github.com/jidec/critterframe.git"
```

## Testing

```
pytest                          # ~1,100 tests, ~45s, no GPU, network, or credentials
pytest tests/unit -m "not slow" # inner loop, a few seconds
pytest -m gpu                   # opt in to what's deselected by default (gpu, network, interactive)
```

`tests/unit/` is one file per module

`tests/integration/` is one file per notable cross-module case, testing
repeat-awareness, metric staleness, coordinate inversion, calibrated export etc.

## Example pipelines

`scripts/` holds one runnable script per project shape, each documenting what it demonstrates:

| Script | Shows                                                                       |
| --- |-----------------------------------------------------------------------------|
| `simplest_pipeline.py` | The simplest five-call pipeline. Good place to start.                       |
| `antenna_moths_pipeline.py` | Pre-cropped images, so segmentation skips detection. Scale calibration.     |
| `dragonfly_wings_museums.py` | Subsets with different recipes; four wings as four parts.                   |
| `dragonfly_bodies_inat.py` | Part refinement from the organism mask; group metrics; embeddings.          |
| `salamander_boxes.py` | Local images, a specialized segmenter, position as the trait.               |
| `validation_pipeline.py` | Mask, measurement, and filter validation against human-reviewed references. |

## The project structure

```
my_project/
    occurrences.parquet         central imported/normalized metadata
    images.lmdb/                original images, one per occurrence, byte-exact
    masks.parquet               canonical masks, one per occurrence-part
    reference_masks.parquet     human-vetted or otherwise trusted masks
    calibrations.parquet        px/mm and the like, keyed by what was calibrated
    runs_and_metrics.sqlite     run records + the metric values they produced
    imports/                    immutable source imports
    definitions/                subsets.toml, recipes.py
    visualizations/
        pipeline/               one sampled QC grid per run
        products/               rendered assets, one file per occurrence-part
    models/                     registry.json + checkpoints trained for this project
```

## Vocabulary

| Term                 | Meaning                                                                                                                                                                               |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Project**          | A self-contained collection of organismal occurrence images, metadata, & derivations intended to be analyzed as a coherent biological dataset, sharing at least some processing steps |
| **Occurrence image** | Image evidence of a focal organism existing at a specific place at a specific time                                                                                                    |
| **Part**             | A consistent named biological component of an organism, such as `head`. Defaults to `organism` for the part representing the whole organism                                           |
| **Mask**             | At most one canonical mask per occurrence-part, in original image coordinates                                                                                                         |
| **Segment**          | An image plus its current mask. Not persisted because images and masks already are                                                                                                    |
| **Metric**           | Any derived value associated with an occurrence-part                                                                                                                                  |
| **Transform**        | An operation changing the working segment without producing a value, such as orientation normalization                                                                                |
| **Operation**        | One configured processing action, named `transform`, `segment`, or `metric` by what it DOES to a segment                                                                              |
| **Recipe**           | A configured chain of operations, named `segment`, `metric`, or `render` by what its output is persisted as                                                                           |
| **Run**              | One execution of a recipe over a set of occurrences                                                                                                                                   |
| **Subset**           | A named selection of occurrences                                                                                                                                                      |
| **Filter**           | A rule for selecting occurrences, at export or post-critterframe                                                                                                                      |
| **Calibration**      | Knowledge about the imaging system (e.g. px/mm scale). Resolved at metric export.                                                                                                     |
| **Record**           | A persisted datatype, including occurrences, masks, runs, metrics, calibrations, and models                                                                                           |
| **Reference mask**   | A mask kept for comparison, not treated as canonical                                                                                                                                  |
| **Recipe hash**      | The reproducible hash over a recipe's operations; what makes a rerun skip already-done work.                                                                                          |
| **Registered model** | A model attached to provenance info.                                                                                                                                                  |
| **Panel**            | One picture of one operation's decision about one occurrence-part. The unit visualizations are built from.                                                                            |
| **Render**           | A materialized image product.                                                                                                                                                         |

## Package layout

```
critterframe/
    recipes.py              classes jointly implementing recipes contract: Segment, Recipe, Operation (Transform, Segmentation, Metric) plus hashing
    ingest.py               ingest occurrence tables and optionally local images
    download.py             download images from URLs in ingested table
    export.py               export one-row-per-occurrence trait table, optionally filtered; select occurrences by stored values
    selectionhelpers.py     helpers for transient "out of these occurrences, which ones" tasks: sampling, sharding, rule matching
    project/                
        paths.py            define every path and filename in critterframe project folders
        subsets.py          create named, persisted selections of occurrences
        summarize.py        summarize what a project directory currently holds
    storage/                
        imagestore.py       the LMDB image store, better than directories for millions of images
        tables.py           parquet tables (occurrences & masks) read, snapshot write, upsert 
        sqlite.py           sqlite databases (runs & metrics) connection
    records/
        occurrences.py      normalize + save/load the occurrence table
        masks.py            RLE encode/decode, upsert, derivation hashing, sharded writes for parallel runs
        runs.py               the sqlite schema for run + metric records
        metrics.py         long-table storage, current_rows, latest_values
        calibrations.py  the scope/provenance machinery every calibration type shares
        models.py          the registry of trained models: checkpoint fingerprints, RegisteredModel
    segmentation/
        groundedsam.py  SAM2, with or without Grounding DINO detection
        manual.py          draw/correct a mask by hand -- an alternative segmentation, not a separate system
        run.py                segment() operation + run_segments(), including sharded/parallel runs
    transforms/
        orient.py            PCA orientation, axis chosen by asymmetry rather than length
        appendages.py    remove legs/antennae from a mask
        crop.py               crop, crop_to_mask, rotate, resize, remove_background
    metrics/
        dimensions.py    body_length, max_width, mask_area, bounding_box
        position.py        centroid, relative_position, image_bounds -- reported in original coordinates
        quality.py          blur, asymmetry, edge fraction -- automated QC
        color.py             mean color, hue/lightness fractions
        outliers.py        group metrics: outlier(), cluster()
        annotation.py    human labels: annotate_flags, click_two_points
        run.py                run_metrics() + RunContext + completed_keys
    calibrations/
        scale.py            px/mm from a target of known size
    validation/
        masks.py            IoU against reference masks
        metrics.py         predicted vs. reference values
        filters.py           calibrate thresholds against human labels
    visualization/
        panels.py           one picture of one operation's decision; shared drawing helpers and colour conventions
        grids.py             many panels as one image: image_grid, comparison_grid
        pipeline.py        the per-run QC grid: resolve_sample, RunReport
        products.py       assets for downstream use: render_segments, one file per occurrence-part
    training/
        splits.py            split_ids(): grouped and stratified, to avoid leakage
        datasets.py       export_training_data(): images, masks, class folders, a manifest
    extensions/                  source-specific packages that normalize INTO core, never around it
        antenna_lighttraps/
            api.py                 Antenna's HTTP API: auth, export, image URLs
            ingest.py             Antenna export -> ingest_occurrences()
            download.py         thin wrapper over download_images() for Antenna's URL column
            calibrations/
                scale.py            scale scoped per trap night (event_id), not per occurrence
        inat_insects/
            api.py                 iNaturalist's API
            ingest.py             iNaturalist observations -> ingest_occurrences()
            download.py         thin wrapper over download_images()
            metrics/
                color.py             colour clustering metrics
                bioencoder.py    embedding-based metrics
            training/
                bioencoder.py    train()/load() -- deliberately unfinished, raises NotImplementedError
```

See [Extensions](#extensions) below for what `antenna_lighttraps`/`inat_insects` are and why extensions exist
as a pattern; see [Testing](#testing) above for `tests/`, which mirrors this same tree one level up.

## Extensions

An extension normalizes one data source's quirks *into* core's representation — occurrences, images, masks —
rather than building parallel structures around it. That's the whole contract: an extension's `ingest`/`download`
map a source's API or file layout onto `ingest_occurrences`/`download_images`, and anything project-specific
(a calibration scope, a metric only that source's images support) lives beside it rather than leaking into
core. Two are shipped as worked examples of the pattern:

- **`antenna_lighttraps`** — light-trap camera monitoring. Pre-cropped detections come in pre-cropped, and
  scale calibration is scoped per trap night (`event_id`) rather than per occurrence.
- **`inat_insects`** — iNaturalist observations. Adds colour clustering and embedding-based metrics suited to
  citizen-science images shot under uncontrolled conditions.

A new data source follows the same shape: ingest/download that normalizes into core, plus whatever
calibration scope or metrics that source specifically needs — not a fork of the pipeline.

Claude Code was used to contribute code and documentation to this project (with every line examined by a human).

## License

[GPL-3.0](https://github.com/jidec/critterframe/blob/main/LICENSE)
