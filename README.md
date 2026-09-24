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

#### Plus core support for:
- Validation of segments & traits
- Advanced filtering
- Human annotation
- Training dataset export
- Importing segmentation models
- Managing organismal parts (i.e. head, thorax)
- Traits derived from stored traits: per occurrence (i.e. ratios) or per group (i.e. cluster assignments)
- Scale & color calibration

#### And support in extensions for: 
- Training a segmentation model using Segmentation Models Pytorch
- Training a bioencoder model
- Advanced customizable ingests from a GBIF DarwinCore archive

See [scripts/](scripts/) for full example pipelines

## The framework

1. **One focal organism per image**

   > **Why:** A single organism is the natural unit for organismal image analysis and maps cleanly onto an occurrence. We see this as a worthy simplification that removes a complex layer of bookkeeping. Tools will eventually be provided to help convert multi-organism images into one-organism images for import.

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

## Other key features

- Virtually every pipeline step leaves a visual report by default (`visualize=True`), making things easy to scrutinize
- Project folders are portable records with data & provenance ready for archiving alongside a publication
- Persistent named subsets make it easy to pass data around for validation, training, or subset-specific processing
- Metrics exports designed for easy analysis post-critterframe
- Multithreading/sharding for image downloading, segmentation, and metric runs

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
pytest                          # ~1,100 tests, ~45s without GPU, network, or credentials
pytest tests/unit -m "not slow" # inner loop, few seconds
pytest -m gpu                   # opt into gpu, network, interactive etc.
```

`tests/unit/` is one file per module

`tests/integration/` is one file per notable cross-module case, testing
repeat-awareness, metric staleness, coordinate inversion, calibrated export etc.

## Example pipelines

`scripts/` holds one runnable script per project shape, each documenting what it demonstrates. 
Most are untested for now, but should work as written and are good examples of different project shapes. 

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
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Project**             | A self-contained collection of organismal occurrence images, metadata, and derivations intended to be analyzed as a coherent biological dataset and sharing at least some processing steps. |
| **Raw import**          | Source data preserved before any structural or judgment-based decisions are applied, such as a GBIF Darwin Core archive as downloaded. Archived in the `raw_imports/` folder.               |
| **Import**              | A raw import reshaped into occurrences and narrowed through explicit inclusion decisions such as `drop`, `group_col`, and `max_per_group`. A manifest records those decisions.              |
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

## Package layout

```
critterframe/
    recipes.py              classes jointly implementing recipes contract: Segment, Recipe, Operation (Transform, Segmentation, Metric) plus hashing
    ingest.py               ingest occurrence tables and optionally local images
    download.py             download images from URLs in ingested table
    export.py               export one-row-per-occurrence trait table, optionally filtered, with a manifest saying what it is; select occurrences by stored values
    selectionhelpers.py     helpers for transient "out of these occurrences, which ones" tasks: sampling, sharding, rule matching
    segments.py             iterate_segments(): the per-occurrence loop most drivers walk, plus build_segment and Tally
    maskops.py              mask arithmetic with no project attached: iou, coverage, bounds, largest component
    colorspaces.py          convert(): BGR to rgb/linrgb/hsv/hls/lab/lch in canonical units, to_bgr(), in_arc() for hue arcs
    devices.py              resolve_device(): which device a loaded network runs on, asked lazily
    timing.py               timed(): one "<label> in Ns" log line around a slow step
    project/                
        paths.py            define every path and filename in critterframe project folders
        subsets.py          create named, persisted selections of occurrences
        summarize.py        summarize what a project directory currently holds
        archive.py           archive_project(): a deposit-ready copy, without images, raw data or local paths
    storage/                
        imagestore.py       the LMDB image store, better than directories for millions of images
        tables.py           parquet tables (occurrences & masks) read, snapshot write, upsert 
        jsonfiles.py        every manifest, registry and append-only log: atomic writes, JSON and JSONL
        sqlite.py           sqlite databases (runs & metrics) connection
    records/
        occurrences.py      normalize + save/load the occurrence table
        masks.py            RLE encode/decode, upsert, derivation hashing, sharded writes for parallel runs
        runs.py               the sqlite schema for run + metric records
        metrics.py         long-table storage, current_rows, latest_values
        calibrations.py  the scope/provenance machinery every calibration type shares
        models.py          the registry of trained models: checkpoint fingerprints, RegisteredModel
        failures.py         what not to retry, keyed by occurrence-part, stage, and what was attempted
    segmentation/
        groundedsam.py  SAM2, with or without Grounding DINO detection
        manual.py          draw/correct a mask by hand -- an alternative segmentation, not a separate system
        mask_import_export.py  import_masks()/export_masks(): masks in and out as <occurrence_id>__<part>.png
        run.py                segment() operation + run_segments(), including sharded/parallel runs
    transforms/
        orient.py            PCA orientation, axis chosen by asymmetry rather than length
        appendages.py    remove legs/antennae from a mask
        crop.py               crop, crop_to_mask, rotate, resize, remove_background
    metrics/
        dimensions.py    body_length, max_width, mask_area, bounding_box
        position.py        centroid, relative_position, image_bounds -- reported in original coordinates
        quality.py          blur, asymmetry, edge fraction -- automated QC
        pixels.py            masked_pixels: the organism's pixels, the one rule every colour metric shares
        color_means.py    mean_color, mean_lightness, white_balanced_color, background_color
        color_thresholds.py  ColorThreshold cutoffs across colour spaces; threshold_fractions and presets
        inductive_color_thresholds.py  the same thresholds fitted per group: a chroma gate and hue arcs
        color_clusters.py  per-group colour palette proportions
        embedding.py      EmbeddingModel over any torch network, pretrained() timm backbones, embedding()
        stored.py          StoredValues + the base for metrics computed from stored values, not pixels
        derived.py        derived(): a value from one occurrence-part's own stored values, e.g. a ratio
        outliers.py        group metrics over a population of stored values: outlier(), cluster()
        annotation.py    human labels: usability_annotation, click_two_points
        run.py                run_metrics() + RunContext + _completed_keys
    calibrations/
        scale.py            px/mm from a target of known size
    validation/
        masks.py            IoU against reference masks
        metrics.py         predicted vs. reference values
        filters.py           calibrate thresholds against human labels
    visualization/
        panels.py           one picture of one operation's decision; shared drawing helpers and colour conventions
        grids.py             many panels as one image: image_grid, comparison_grid
        figures.py          whole-population charts: line_chart, bar_chart, histogram, scatter, funnel
        pipeline.py        every activity's diagnostics: open_report, Report, resolve_sample
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
        bioencoder/
            embedding.py      BioEncoderModel (an EmbeddingModel) + load_registered()
            training.py         prepare_dataset(), load(), load_from_config(); train() deliberately unfinished
        gbif_darwincore_inat/
            archive.py          read a GBIF Darwin Core Archive, zipped or extracted
            ingest.py             one photo per occurrence -> ingest_occurrences(); iNat photo sizes, prioritize_inat
        smp_segmenter/
            segmentation.py   SMPSegmenter (UNet++, swappable encoder) + smp_segmenter() factory
            training.py         prepare_dataset() + train() + register_trained() -- a real, runnable training loop
```

## Cross-Cutting Validation

Validation should be a part of every pipeline: critterframe supports validation at multiple levels with worked examples in `/scripts`

For reference set creation/annotation
- `correct_masks` or `manual_masks` to create a reference/ground truth mask set
- `usability_flags`
- Stratified sampling of reference sets across grouping columns (e.g. taxa, collections)

For segmentation:
- `validate_masks` to directly compare a segmentation model to a reference set

For traits:
- `compare_metrics` to assess agreement between an automatically computed trait and a human-measured one

For the validity/quality of final outputs (AKA noise filtering):
- `get_validated_filters` that given candidate columns (e.g. seg model confidence, transform reliability flags) and project-scoped negative image cases (e.g. blurry, cutoff) computes filtering columns & values that screen out invalid or low quality examples under different coverage-quality tradeoffs
- Certain group-level metrics (e.g. outlier labels/scores, cluster assignments) are especially useful filtering candidates

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

Claude Code was used to contribute code, documentation, & tests to this project (with every line examined by a human).

## License

[GPL-3.0](https://github.com/jidec/critterframe/blob/main/LICENSE)
