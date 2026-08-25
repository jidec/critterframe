# critterframe

Image frames of critters into dataframes of traits

critterframe is a flexible Python package built for large-scale organismal image processing, including easy-to-understand tools for:

```python
# 1. ingesting
cf.ingest_occurrences("my_project", "specimens.csv", id_col="specimen_id", image_url_col="image_url")
# 2. downloading
cf.download_images("my_project")

# 3. segmentation 
cf.run_segments("my_project", steps=[cf.segment(cf.groundedsam2())])

# 4. trait extraction
cf.run_metrics(
    "my_project",
    run_name="traits",
    transforms=[cf.remove_appendages()],
    metrics=[cf.mean_lightness()],
)
cf.export_metrics("my_project", "traits.csv")

# Plus:
# filtering
# validation
# pipeline visualization
# human annotation
# custom model training & training set assembly
```
## The framework

1. **One focal organism per image**

   > **Why:** A single organism is the natural unit for organismal image analysis and maps cleanly onto an occurrence. We see this as a worthy simplification that removes a complex layer of bookkeeping. Tools are provided to convert multi-organism images into one-organism images for import.

2. **One canonical mask per organism or organism-part**

   > **Why:** Virtually all analyses need the best available representation of a biological part, not a growing collection of competing masks. Alternative segmentation approaches can be evaluated against reference masks before deciding which should become canonical.

3. **All derived values are metrics - whether traits, QC scores, or annotations**

   > **Why:** A common metric model lets the same machinery support biological measurements, quality control, validation, clustering, and other analyses. Since filtering happens also means filtering can happen during export or later analysis, so observations and derived data never need to be removed from the pipeline itself.

4. **Filtering is selection, not deletion**

   > **Why:** Filtering criteria are analytical decisions that may change as a project develops. CritterFrame therefore retains occurrences and their derived data, applying filters during export or later analysis rather than removing data from the processing pipeline.

## The pipeline

1. **Simple, small operations compose flexibly into complex pipelines**

   > **Why:** The same segmentation, transform, metric, subset, and comparison tools can support both simple trait extraction and complex pipelines involving body parts, processing subsets, embeddings, clustering, human validation, and complex filtering.

2. **Rerunnable and resumable**

   > **Why:** Large image datasets are expensive to process and continually evolve. Equivalent completed work is reused, interrupted runs can continue, and adding new occurrences or analyses does not require starting over.

3. **Full derivation provenance**

   > **Why:** Every mask and metric retains enough information to determine how it was produced and what inputs it depended on. This makes analyses interpretable, comparable, and reproducible without requiring users to manually track processing history.

## Convenience features

1. **Projects are portable analytical records ready for archiving alongside a publication**
   - A project folder contains the data, intermediate results, and provenance needed to reproduce an analysis
2. **Nearly every processing step can be visualized**
   - Most operations can produce diagnostic visualizations with `visualize=True`.
3. **Training, review, and validation sets are easy to create from project data**
   - Named subsets make it straightforward to freeze samples for annotation, model development, or benchmarking.
4. **Metrics exports are built for easy analysis outside of critterframe**
5. **Multithreading and efficiency built in (soon)** 

## Documentation
Todo

## Installation
```
pip install -e .
pip install -e ".[segmentation]"        # torch + transformers, for SAM2/GroundedSAM2
cp .env.example .env                    # only needed for extensions that call an external API
```

The core has no deep-learning dependency. Install
the `segmentation` extra when you want the bundled SAM2 models. A project segmenting with its own model doesn't
need it.

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

| Term | Meaning |
| --- | --- |
| **Project** | One self-contained analytical dataset. A directory. |
| **Occurrence** | One focal organism at a place and time, as one analysis image plus metadata. |
| **Part** | A named biological component of that organism. Defaults to `organism`. |
| **Mask** | At most one current mask per occurrence-part, in original image coordinates. |
| **Segment** | An image plus its current mask, in flight. Never persisted. |
| **Metric** | Any derived value associated with an occurrence and a part. |
| **Transform** | An operation changing the working segment without producing a value. |
| **Operation** | One configured processing action: `remove_appendages()`, `segment(groundedsam2())`. |
| **Recipe** | A reproducibly hashable configured operation chain: `segment`, `metric`, or `render`. |
| **Run** | One execution of a recipe over a set of occurrences. |
| **Subset** | A named selection of occurrences receiving a particular recipe. |
| **Filter** | A rule for selecting occurrences at export. Never deletes. |

## Package layout

- **`project/`** — where a project's files live (`paths`, all `pathlib.Path`), what's in it (`summarize`),
  which occurrences a recipe runs over (`subsets`). Nothing creates a project: directories appear lazily, as
  their first writer needs them, so what a project contains is an honest account of what's been done to it.
- **`ingest.py` / `download.py`** — occurrence tables and local images in; images fetched from URLs. Rows a
  source has already declared hold no organism can be dropped on the way in (`drop=`), while the archived
  import keeps them.
- **`recipes.py`** — `Segment`, `Operation`, `Recipe`, and the hashing that makes processing repeat-aware.
- **`calibration/`** — what the imaging system did to the image: `scale` (px/mm from a target of known
  size in the frame), and colour correction when it's written.
- **`export.py`** — the wide, one-row-per-occurrence trait table, optionally filtered, and the long-to-wide
  reshape everything else that wants that shape shares.
- **`storage/`** — the LMDB image store (byte-exact; `get()` for the 8-bit working view, `get_bytes()` for
  the original) and the parquet/sqlite mechanics, with no knowledge of any entity.
- **`records/`** — what a project persists: `occurrences`, `masks`, `runs`, `metrics`, `calibrations`, and
  `models` (which checkpoint a name refers to, what it was trained on, and the fingerprint that puts those
  weights into the recipe hash of everything they produce).
- **`transforms/`** — `orient`, `appendages`, `crop` (plus rotate, resize, remove_background).
- **`segmentation/`** — `groundedsam` (SAM2, with or without a detector), `manual` (draw/correct by hand), `run`
  (compose and execute).
- **`metrics/`** — `dimensions`, `position`, `quality`, `color`, `outliers` (group metrics), `annotation`
  (human labels), `run`.
- **`validation/`** — `masks` (IoU against reference), `metrics` (predicted vs reference values), `filters`
  (calibrate thresholds against human labels).
- **`visualization/`** — `panels` (one picture of one decision, and the colour conventions they share),
  `grids` (many panels as one image), `pipeline` (the per-run QC grid), `products` (`render_segments`, one
  image file per occurrence-part).
- **`training/`** — `splits` (`split_ids`: which occurrences answer which question, grouped and stratified so a
  specimen can't straddle train and validation) and `datasets` (`export_training_data`: those occurrences as
  images, masks, class folders, and a manifest a trainer can read). Training itself stays outside; what comes
  back is registered, and then `segment()` runs it like any other model.
- **`tests/`** — `pytest`, no GPU or network required: `unit/` mirrors the package, `integration/` is named for
  the invariant each file protects (repeat-awareness, metric staleness, coordinate inversion, calibrated
  export). The scripts under `scripts/simple_tests/` remain the *visual* check — they write debug images a
  person looks at, which is the one thing a test can't do.
- **`extensions/`** — source-specific packages that normalize *into* the core representation:
  `antenna_lighttraps` (light-trap monitoring, scale calibration per trap night) and `inat_insects` (iNaturalist, colour
  clustering, embeddings).

Claude Code was used to contribute code and documentation to this project (with every line examined by a human).

## License

[GPL-3.0](LICENSE)
