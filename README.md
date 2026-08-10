# CritterFrame

Images to traits — and image frames to dataframes.

CritterFrame turns a pile of organism images into a table of measurements: ingest the metadata, download or
import the images, segment the organism (and any body parts you care about) out of each one, measure whatever
you want off the result, and export one row per organism for analysis.

It's built for the awkward middle ground where most biological image analysis actually lives — several
collections imaged to different standards, several body parts per specimen, some measurements automated and
some checked by hand, and a processing pipeline that keeps changing while the data keeps arriving.

```python
import critterframe as cf

cf.ingest_occurrences("my_project", "specimens.csv", id_col="specimen_id", image_url_col="image_url")
cf.download_images("my_project")

cf.run_segments("my_project", steps=[cf.segment(cf.groundedsam2())])

cf.run_metrics(
    "my_project",
    run_name="traits",
    transforms=[cf.remove_appendages()],
    metrics=[cf.mean_lightness()],
)

cf.export_metrics("my_project", "traits.csv")
```

## Install

```
pip install -e .
pip install -e ".[segmentation]"        # torch + transformers, for SAM2/GroundedSAM2
cp .env.example .env                    # only needed for extensions that call an external API
```

The core — ingest, storage, transforms, metrics, export, validation — has no deep-learning dependency. Install
the `segmentation` extra when you want the bundled SAM2 models; a project segmenting with its own model doesn't
need it.

## The ideas

Thirteen of these, and everything else follows from them.

**A project is one self-contained analytical dataset.** One directory holding occurrences, images, masks,
metrics, processing definitions, and provenance. Occurrences in a project share a biological and trait model,
but subsets of them can be *processed* completely differently — three museums that photographed their specimens
three different ways still belong in one project, because you want to compare them.

**One focal organism per occurrence, per image.** The most important structural decision in the package. If
your images contain several organisms, they must be separated upstream of ingest. Everything downstream leans
on this.

**Any number of parts per occurrence.** A named biological component — the whole organism by default, or head,
abdomen, forewing. Four wings on one sheet are four parts of one occurrence, not four occurrences.

**Images are stored byte-exact.** The image store holds the encoded bytes of the analysis image exactly as
they arrived — no decode, no re-encode. A 16-bit TIFF stays 16-bit, alpha survives, EXIF survives, and a JPEG
is never recompressed a second time. Reads normalize to 8-bit BGR for the processing pipeline, which is
reversible because `get_bytes()` still returns the original.

**One canonical mask per occurrence-part.** Optional reference masks coexist alongside for validation. Masks
are always stored in the coordinates of the original analysis image, whatever crops, rotations, and resizes
produced them — so a wing mask found inside a rotated sub-crop and a whole-organism mask found on the full
frame describe pixels of the same picture.

**Metrics are any derived value.** Traits, QC scores, human labels, embeddings, cluster assignments, outlier
scores. A metric fit against a reference population is still a metric. They all store, export, and filter the
same way.

**Filtering is data, not deletion.** Filters are applied at export. Excluding blurred images from a CSV leaves
every blurred image, its mask, and its measurements exactly where they were, so a threshold can be revised and
the export rerun without recomputing or losing anything.

**Masks and metrics carry full derivation recipes.** A recipe is an immutable, hashable specification of
ordered operations. Its identity covers operation order, every parameter, every version, model and checkpoint
identity, and upstream dependencies.

**Processing is interruptible and repeat-aware.** Because recipes hash, a run skips work an equivalent recipe
already did. An interrupted run resumes. A repeated run is a no-op. An expensive metric behaves like cached
derived data rather than a calculation redone on every invocation.

**Derived data follows the masks it was derived from.** Every metric value records which mask produced it, and
every part cut out of another part records which mask it started from. Resegment an occurrence and the numbers
measured off its old mask stay on record — nothing is deleted — but they stop counting as done, so the next
run recomputes them, and stop being reported, so an export describes the masks the project currently holds
rather than ones it replaced. The same applies down a chain: resegment the organism and the wing masks cut out
of it, and everything measured on those, follow.

**Looking at the images is part of the pipeline.** `visualize=25` on any run samples 25 occurrences and writes
one grid to `visualizations/pipeline/` — a row per specimen, a column per processing stage — which is how you
judge a step across a collection instead of squinting at 10,000 debug files. Any operation in a recipe can
contribute a column, and segmentation and metric runs work the same way. Renders you actually want to keep
go to `visualizations/products/` as one image file per occurrence-part, named for the occurrence, so a figure
script or an R session can find them by id.

**A calibration describes the imaging system, not an organism.** A scale target in every frame, one reference
card per trap deployment, one calibrated copy stand for a whole collection — all three are the same knowledge
at a different grain, so a calibration is keyed by whichever occurrence column identifies what was calibrated,
and resolved onto occurrences when something needs it. Scale is the first kind; colour correction is the next,
and it shares the bookkeeping without being squeezed into scale's shape. Traits stay in pixels forever;
`export_metrics(units="mm")` divides at the end, so a corrected calibration costs one re-export instead of
re-measuring everything.

**Validation is comparison, not a separate system.** Human mask correction is a segmentation recipe. Human
labels are a metric run. Validation just measures agreement between results that both already exist.

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

## What a project looks like

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
    models/                     named custom segmenters, checkpoints
```

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
- **`records/`** — what a project persists: `occurrences`, `masks`, `runs`, `metrics`.
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
- **`training/`** — `datasets` and `splits`, for training the project-specific models `segment()` then runs.
- **`extensions/`** — source-specific packages that normalize *into* the core representation:
  `antenna_lighttraps` (light-trap monitoring, scale calibration per trap night) and `inat_insects` (iNaturalist, colour
  clustering, embeddings).

## Reference pipelines

`scripts/` holds one runnable script per project shape, each documenting what it demonstrates:

| Script | Shows |
| --- | --- |
| `simplest_pipeline.py` | The five-call pipeline. Start here. |
| `antenna_moths_pipeline.py` | Pre-cropped images, so segmentation skips detection. Scale calibration. |
| `dragonfly_wings_museums.py` | Subsets with different recipes; four wings as four parts. |
| `dragonfly_bodies_inat.py` | Part refinement from the organism mask; group metrics; embeddings. |
| `salamander_boxes.py` | Local images, a specialized segmenter, position as the trait. |
| `validation_pipeline.py` | Mask, measurement, and filter validation against human-reviewed references. |

`scripts/simple_tests/` mirrors the package layout with standalone manual scripts — no assertions, they print
output or write debug images for you to look at. `pipeline_synthetic_test.py` and `recipes_test.py` need no
data, no credentials, and no GPU, and are the fastest way to check the core machinery still works.

Claude Code contributed code and documentation to this project (with every line examined by a human).

## License

[GPL-3.0](LICENSE)
