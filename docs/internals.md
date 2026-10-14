# Internals

## Package layout

```
critterframe/
    core/                   the center of the package
        recipes.py          classes jointly implementing recipes contract: Segment, Recipe, Operation (Transform, Segmentation, Metric) plus hashing
        segments.py         iterate_segments(): the per-occurrence loop most drivers walk, plus build_segment
        drivers.py          what every per-item driver shares: Tally, Progress, NoInput
    ingest/
        occurrences.py      ingest an occurrence table: archive, reshape, narrow, record how
        images.py           ingest a folder of local images
        download.py         download images from URLs in ingested table
        archive.py          the raw-import archive: raw bytes stored once in raw_imports/
        imports.py          the import record: what an import is hashed from, its manifest and log, recognizing a repeat
    export.py               export one-row-per-occurrence trait table, optionally filtered, with a manifest saying what it is
    wide.py                 the wide view: current metric values as one row per occurrence, unit conversion, filter evaluation
    maskops.py              mask arithmetic with no project attached: iou, coverage, bounds, largest component
    colorspaces.py          convert(): BGR to rgb/linrgb/hsv/hls/lab/lch in canonical units, to_bgr(), in_arc() for hue arcs
    devices.py              resolve_device(): which device a loaded network runs on, asked lazily
    timing.py               timed(): one "<label> in Ns" log line around a slow step
    project/                
        paths.py            define every path and filename in critterframe project folders
        summarize.py        summarize what a project directory currently holds
        archive.py           archive_project(): a deposit-ready copy, without images, raw data or local paths
    selection/
        queries.py          which occurrences, read from a project: ids_matching, ids_passing, ids_completed, ids_with_mask
        algorithms.py       pure selection logic over supplied data: sampling, sharding, capping, deduplication, rule matching
        subsets.py          create named, persisted selections of occurrences
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
        run_segments.py       segment() operation + run_segments(), including sharded/parallel runs
    transforms/
        orient.py            PCA orientation, axis chosen by asymmetry rather than length
        appendages.py    remove legs/antennae from a mask
        islands.py          remove_islands: drop disconnected fragments, keeping the organism
        erode.py             erode: pull a mask in from its edges, by a share of its own thickness
        crop.py               crop, crop_to_mask, rotate, resize, remove_background
    metrics/
        base/                 what metrics are built from, not metrics themselves
            pixels.py        masked_pixels: the organism's pixels, the one rule every colour metric shares
            stored.py       StoredValues + the base for metrics computed from stored values, not pixels
            group.py        the bases for a metric fitted per group: over stored values, or pooled pixels
        dimensions.py    body_length, max_width, mask_area, bounding_box, elongation, jaggedness
        islands.py         n_islands: fragments of a mask apart from the organism
        position.py        centroid, relative_position, image_bounds -- reported in original coordinates
        quality.py          blur, asymmetry, edge fraction -- automated QC
        color/
            means.py        mean_color, mean_lightness, white_balanced_color, background_color
            thresholds.py   ColorThreshold cutoffs across colour spaces; threshold_fractions, color_presence and presets
            inductive_thresholds.py  the same thresholds fitted per group: a chroma gate and hue arcs
            clusters.py     per-group colour palette proportions
        embedding.py      EmbeddingModel over any torch network, pretrained() timm backbones, embedding()
        derived.py        derived(): a value from one occurrence-part's own values, e.g. a ratio
        outliers.py        outlier(): how unusual an occurrence is within its group's stored values
        clusters.py        cluster(): the cluster an occurrence falls in within its group's stored values
        label_score.py   label_score(): a score fitted on stored features against stored human labels
        annotation.py    human labels: exclusive_label_annotation, click_two_points
        run_metrics.py        run_metrics() + RunContext + _completed_keys
    calibrations/
        scale.py            px/mm from a target of known size
    validation/
        masks.py            IoU against reference masks
        metrics.py         predicted vs. reference values
        filters.py           calibrate thresholds against human labels; audit a filter set on held-out ones
        filter_grids.py    image grids of what a filter set catches, misses and costs
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

!!! note "Extensions"
    Extensions are typically for handling specific data sources, very specialized metrics, or trainable
    models not general enough to bundle in core. By convention they mirror the package layout.

    - **`antenna_lighttraps`**: light-trap camera monitoring. Scale calibration is scoped per trap night
      (`event_id`) rather than per occurrence.
    - **`bioencoder`**: dataset preparation for training a metric-learning model, and a loader for
      checkpoints the BioEncoder package trained (by path or by its own YAML config), feeding core's
      `embedding()`. The training loop itself is deliberately left to the BioEncoder package.
    - **`gbif_darwincore_inat`**: a GBIF Darwin Core Archive, one photo per occurrence. The way to pull
      iNaturalist observations: iNat photo sizes, cross-source deduplication, and `prioritize_inat` for
      mixed pulls.
    - **`smp_segmenter`**: a trainable UNet++ segmenter (segmentation_models_pytorch) for refining or
      replacing the bundled zero-shot segmenter on one project's own masks. Unlike `bioencoder`'s
      scaffold, its `train()` is a working training loop: binary mask segmentation doesn't carry the
      dataset-dependent backbone and loss judgment calls that make guessing at a metric-learning setup
      risky.

## Diagrams

Automatically generated by `python tools/make_diagrams.py`, so they describe the
code as it is now. Each link opens an SVG. In a call diagram, hovering a function shows the first line of its
docstring.

### Types of diagram

| Diagram | Shows | Drawn from |
|---|---|---|
| `subpackages` | Which folders and top-level modules import which. About twenty nodes. | Imports, whole package |
| `calls_<folder>`, `calls_top_level`, `calls_extensions_<name>` | Which functions call which **inside one folder**. A call into another folder is not drawn. | Calls, one folder |
| `calls_around_<function>` | One entry point, its callers one level up and what it calls two levels down, **across the whole package**. | Calls, whole package |

A call the tool cannot attribute to exactly one function is left out, so a call diagram can miss an edge. It
also cannot see a call dispatched through an operation, a factory, or a model.

### Reading order

1. **The map**: <a href="../diagrams/subpackages.svg">subpackages</a>. An arrow points from an imported module
   to its importer, so `core`, `project` and `storage` are on top, and `export`, `validation` and `training`
   are at the bottom.
2. **The center**: <a href="../diagrams/calls_core.svg">calls_core</a>. `Segment`, `Recipe` and its hash,
   `run_chain`, `framed_segment`, `iterate_segments`.
3. **What a project persists**: <a href="../diagrams/calls_storage.svg">calls_storage</a>, then
   <a href="../diagrams/calls_records.svg">calls_records</a>, the records built on it.
4. **The four entry points**, in pipeline order. These show how the package works end to end:
    - <a href="../diagrams/calls_around_ingest_occurrences.svg">calls_around_ingest_occurrences</a>
    - <a href="../diagrams/calls_around_run_segments.svg">calls_around_run_segments</a>
    - <a href="../diagrams/calls_around_run_metrics.svg">calls_around_run_metrics</a>
    - <a href="../diagrams/calls_around_export_metrics.svg">calls_around_export_metrics</a>
5. **One folder's internals**, when you want them:
    - <a href="../diagrams/calls_top_level.svg">calls_top_level</a> (`export`, `wide`, `maskops`, `colorspaces`)
    - <a href="../diagrams/calls_ingest.svg">calls_ingest</a>
    - <a href="../diagrams/calls_selection.svg">calls_selection</a>
    - <a href="../diagrams/calls_project.svg">calls_project</a>
    - <a href="../diagrams/calls_segmentation.svg">calls_segmentation</a>
    - <a href="../diagrams/calls_transforms.svg">calls_transforms</a>
    - <a href="../diagrams/calls_metrics.svg">calls_metrics</a> (the largest; most of it is individual metric
      operations)
    - <a href="../diagrams/calls_calibrations.svg">calls_calibrations</a>
    - <a href="../diagrams/calls_validation.svg">calls_validation</a>
    - <a href="../diagrams/calls_visualization.svg">calls_visualization</a>
    - <a href="../diagrams/calls_training.svg">calls_training</a>
    - <a href="../diagrams/calls_extensions_antenna_lighttraps.svg">calls_extensions_antenna_lighttraps</a>
    - <a href="../diagrams/calls_extensions_gbif_darwincore_inat.svg">calls_extensions_gbif_darwincore_inat</a>
    - <a href="../diagrams/calls_extensions_smp_segmenter.svg">calls_extensions_smp_segmenter</a>
    - <a href="../diagrams/calls_extensions_bioencoder.svg">calls_extensions_bioencoder</a>

## Testing

```
python tools/check.py           # tests, lint, format and the strict docs build, one pass/fail line each
pytest                          # ~2,200 tests, about a minute, without GPU, network, or credentials
pytest tests/unit -m "not slow" # inner loop
pytest -m gpu                   # opt into gpu, network, interactive etc.
```

`tests/unit/` mirrors the package, one file per module.

`tests/integration/` is one file per notable cross-module case, testing
repeat-awareness, metric staleness, coordinate inversion, calibrated export etc.
