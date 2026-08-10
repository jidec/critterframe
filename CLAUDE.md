# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CritterFrame (`critterframe`) turns organism images into trait tables: ingest occurrence metadata, download or
ingest images, segment the organism (and optionally named body parts) out of each, measure metrics off the results, and
export one row per organism.

## Setup

```
pip install -e .
pip install -e ".[segmentation]"    # torch + transformers, only for the bundled SAM2 models
cp .env.example .env                # credentials for extensions that call an external API
```

The core has no deep-learning dependency, deliberately — ingest, storage, transforms, metrics, export, and
validation all run without torch. Keep it that way: a new core module should not add a torch import.

## Running things

There is no test framework (no pytest, no test config). `scripts/simple_tests/` are standalone manual smoke
scripts mirroring the package layout — no assertions; they print output or write debug images for visual
inspection. Run them individually from the repo root after installing.

Three need nothing but the package itself and are the fastest check that the core machinery still works:

```
python scripts/simple_tests/pipeline_synthetic_test.py      # full pipeline over drawn images
python scripts/simple_tests/recipes_test.py                 # hashing + coordinate inversion
python scripts/simple_tests/visualization/grids_test.py     # grid layout + sampling, no project
```

Run the first two after touching `recipes.py`, `records/`, or either `run.py`; the synthetic pipeline test also
writes real pipeline grids and product renders, so open what it leaves behind after touching
`visualization/`. The rest need real state — a populated project, credentials, a GPU, or files under
`scripts/test_images/`.

`scripts/*.py` (excluding `simple_tests/`) are annotated reference pipelines, one per project shape. They point
at project paths that don't exist in this repo; read them as documentation of intended usage, and expect to
change `PROJECT_PATH` before running one.

No lint/format/typecheck tooling is configured. `python -m pyflakes critterframe` is clean and worth keeping
clean.

## Architecture

### The data model, in the order it constrains things

1. **A project is a directory.** Every path derives from `project_path` (see `project/paths.py`). There is no
   global `DATA_DIR` and no config module — two projects coexist without sharing state. Every public entry
   point takes `project_path` first.
2. **One focal organism per occurrence, per image.** Nothing in the package can check this; it's the ingest
   contract. Multi-organism images must be separated upstream. Where a source can tell you a row holds NO
   organism — a detector pipeline that classified a crop as debris — `ingest_occurrences(drop={column:
   values})` keeps it out of the table entirely. That's the contract being enforced, not a filter being
   applied: a row asserting nothing doesn't belong in a table whose every row asserts an organism. Safe to do
   at ingest because the source file is archived *before* parsing, so `imports/` keeps every dropped row.
   Extensions own the vocabulary (`antenna_lighttraps.ingest.NON_ORGANISM_DETERMINATIONS`).
3. **Any number of parts per occurrence**, defaulting to `"organism"` (`recipes.DEFAULT_PART`).
4. **At most one canonical mask per occurrence-part**, in `masks.parquet`, RLE-encoded, **always in the
   coordinates of the original analysis image**. Reference masks live in an identical table
   (`reference_masks.parquet`) reached with `reference=True`. Called *reference*, never "ground truth" --
   a reference is whatever you chose to compare against, and naming it truth would assert the answer
   validation exists to measure.
5. **The image store holds byte-exact encoded images.** `ImageStore.put()` takes BYTES, not arrays, and
   refuses arrays with a `TypeError`. Decoding and re-encoding would recompress JPEGs, flatten 16-bit to 8-bit,
   drop alpha, and discard EXIF — invisibly and irreversibly. `get()` decodes to 8-bit BGR (the working view
   every transform/metric/model assumes); `get_bytes()` is the full-fidelity escape hatch. **Never add a write
   path that encodes an array into the store.**
6. **Segments are never persisted** — a segment is an image plus a mask, and both already persist separately.
7. **Metrics are any derived value**: traits, QC scores, human labels, embeddings, cluster assignments, outlier
   scores. Stored long in `runs_and_metrics.sqlite`, reshaped wide at export.
8. **Metrics are immutable historical results, and a value is *current* only while its source mask is.** Every
   row records `source_mask_hash` — `records.masks.derivation_hash()` of the mask it was measured from, which
   is that mask's segmentation recipe hash, chained with its upstream's when it was cut out of another part
   (so resegmenting the organism moves the identity of every part below it). Replacing an
   occurrence-part's canonical mask deletes nothing — it just means the values derived from the old mask stop
   being current, and the long table legitimately holds both. Anything that reshapes values for analysis
   (`export.metrics_wide`, `records.metrics.latest_values`, `compare_metrics`, and `export_metrics` through
   them) reports only the current ones, via `records.metrics.current_rows`; `current_only=False` opts out where
   you want the raw history. Provenance is kept, and the working analysis follows the current masks.
   A row stores nothing the run already records — no per-row `version` (it's in the recipe spec, hence in the
   recipe hash) and no per-row timestamp (the run has one; `metric_id` is the insertion order that "newest
   wins" sorts on). Old databases carrying those columns are migrated on open by `records.runs`, which has to
   happen because the old `created_at` was NOT NULL and would reject every new insert.
9. **A calibration is knowledge about the imaging system, keyed by a scope, resolved not copied, applied
   late.** `calibrations.parquet` (`records/calibrations.py`) holds one row per
   `(calibration_type, scope, scope_value)` — `scale` today, `color` when it's written. **The scope and the
   provenance are generic; the payload is not.** `parameters` is an opaque JSON dict the record layer never
   interprets, because a scale is one number and a colour correction is a method plus a matrix plus an offset
   plus an illuminant, and flattening both into a `value` column would distort the second. Each type owns a
   module under `calibration/` that supplies the meaning: what the parameters are called, what a valid one
   looks like, how to measure it.
   A scope is just an occurrence column — `occurrence_id` for a target in every frame, `session_path` for a
   light trap's card, `device` for a fixed rig — which is how the package identifies a group everywhere else
   (`outliers.group_col`, `splits.group_col`, subset `column`), so changing what a calibration covers is a data
   change, not a code change. `resolve_for_occurrences()` maps scopes down to one answer per occurrence,
   narrowest scope winning (ranked by how many occurrences a scope value covers, so a project's own scope gets
   sensible precedence for free). Nothing is written onto occurrences: that table is snapshot-written and a
   re-ingest would erase it.
   Conversion is `export_metrics(units="mm")` at the last moment. `unit` is inside `Metric.spec()` and
   therefore inside the recipe hash, so measuring in mm would make a re-calibration a *different recipe* and
   invalidate every stored trait. Measuring stays in pixels forever; a corrected calibration costs one
   re-export.
10. **Filtering happens at export only.** Nothing is ever deleted for failing a filter. The boundary against
    item 2's `drop=`: a filter is a judgement THIS project made about degree (too blurred, score too low,
    outlier) and must stay revisable without recomputing, so it narrows at the end and keeps the data. `drop=`
    excludes on a categorical fact the SOURCE reported about whether there's an organism at all. That's why
    its rule vocabulary is membership-only — it's kept too small to express a threshold, so a quality
    judgement can't be smuggled into the one place that can't undo it.
11. Imports are complete snapshots - imported occurrences replace the current occurrence table and reimporting an image folder ingests any new images

### The three types everything is built from (`recipes.py`)

- **`Segment`** — the working representation: image, current mask, and a 2×3 affine mapping ORIGINAL image
  coordinates to its own. Spatial transforms compose onto that affine via `Segment.replace(applied=...)`;
  `mask_in_original_coordinates()` inverts the whole chain in one step before persistence. **This is the
  single most important invariant in the package.** A transform that moves pixels and forgets to pass
  `applied=` produces masks that look correct in isolation and land in the wrong place.
- **`Operation`** — one *configured* action, with a `spec()` covering everything that changes its output.
  Three kinds: `Transform` (segment in, segment out), `Segmentation` (attaches a mask), `Metric` (terminal
  value). A model reaches the hash through its own `identity()`, passed as `model=` rather than through
  `parameters` (parameters must be JSON-serializable).
- **`Recipe`** — a reproducibly hashable configured operation chain: ordered operations plus
  part/from_part/inputs, and a hash over all of it. `kind` is open-ended; `segment` and `metric` execute as
  runs and get a run record, `render` identifies a transform chain whose output is images
  (`visualization/products`). `records.runs.RUN_KINDS` deliberately accepts only the first two — a render
  derives no data, so the hash naming its folder is the whole of its provenance.

### Repeat-awareness

`run_segments` and `run_metrics` both compute their recipe hash, ask the store which `(occurrence_id, part)`
pairs that hash already covered (`records.masks.completed_keys` / `metrics.run.completed_keys` — the metric one
lives with the run because "what work is left" is a property of the run, not of the stored values), and skip
them. This is what makes runs interruptible and makes expensive metrics behave like cached derived data.
`force=True` overrides. **This is a behavioural guarantee — don't add a code path that writes results without
a recipe hash, or that mutates a recipe after a run starts.**

Completion is keyed on `(recipe hash, source mask)`, not the recipe hash alone, on both sides:

- `run_metrics` passes the `derivation_hash()` of the masks it is about to measure into
  `metrics.completed_keys`, so a value computed from a mask that has since been resegmented never counts as
  work already done. Without that, a re-run after a resegmentation silently skips every occurrence and the
  project keeps serving numbers measured off masks it no longer has.
- `run_segments` does the same for a `from_part` recipe: it loads the upstream masks *before* computing what's
  pending, records each output's upstream on the mask row (`source_mask_hash`), and passes the expected
  upstream hashes into `masks.completed_keys`. A derived part's own recipe hash doesn't move when the part it
  was cut out of is resegmented, so without this a wing mask would stay "done" against an organism mask that
  no longer exists — and, because `derivation_hash()` chains, the change propagates to the wing's metrics and
  to anything derived below it, however deep.

`records.masks.current_derivation_hashes` is the read side of both. `masks.parquet` gained `source_mask_hash`
for this; tables written before it exist are read without the column (`storage.tables.table_columns` decides),
which makes their masks look upstream-less — true of everything recorded at the time, and it costs one
recompute of any from_part chain.

### Visualization: two kinds, two contracts

`visualizations/` has exactly two meanings under it, and which one a picture belongs to is decided by who it's
for, not by what's in it.

- **`pipeline/`** — sample-level summaries of how processing BEHAVED. The principle, and it holds for every
  kind of run: *any operation in a recipe may contribute a visual panel, and a pipeline visualization
  summarizes those panels for a sample of the occurrences a run processed.* Segmentation and metric runs are
  the same object here — a stage per column, an occurrence per row — so `visualize=` means the same thing in
  both. One grid per run, `<run name>_<recipe hash>.jpg` (with `__<part>` when the part isn't the default).
  `visualize=25` samples 25, `True` samples a default 25, `["a","b"]` names them, `False` is the default.
  **There is no per-occurrence file mode** — a 10,000-occurrence run can't be inspected as 10,000 files, so
  every form samples. The sample is deterministic (`selectionhelpers.sample_occurrences`), so two versions of a
  recipe show the SAME specimens and can be compared cell by cell. One stage → `image_grid`; several →
  `comparison_grid`. A grid can only show work that happened, so a fully-cached rerun writes none — that's what
  `force=True` is for.
- **`products/<name>_<hash>/`** — assets deliberately materialized for downstream use, **one file per
  occurrence-part**, named `<occurrence_id>.png` (or `<occurrence_id>__<part>.png` when a render covers several
  parts). `render_segments()` is the one that exists. Loose files, not LMDB, because these are for figures and
  for R, and the store exists to hold original images byte-exactly for Python.

A render derives nothing and records nothing: no mask, no metric, no run row. It hashes its transform chain
only so the folder name identifies what's in it and a rerun is a no-op. Don't add a code path that makes a
picture into a measurement.

### Package layout

- **`project/`** — `paths` (every project path, returning `pathlib.Path`; creates nothing), `subsets` (named
  selections, `subsets.toml`, and `select_occurrences`, which every run funnels through), `summarize`.
- **`calibration/`** — one module per kind of calibration, holding what it MEANS. `scale` (px/mm from a
  target of known size: `scale_from_target`, `measure_scales`, `declare_scale`, `scale_for_occurrences`);
  `color` not written yet and, when it is, beside `scale.py` rather than inside it. The detector is generic on
  purpose — target, size, and search region are all arguments — and a weak match is accepted but warned about,
  since clutter can out-correlate an absent target and a plausible wrong scale is worse than none.
- **`selectionhelpers.py`** — transient "out of these occurrences, which ones" helpers: `sample_occurrences`
  (deterministic, so a sample is stable across runs and recipes) and `rows_matching` (the `{column: values}`
  test behind ingest's `drop=`; any rule matches, a missing value never does, an unknown column raises).
  Distinct from `project/subsets`, which is about named, persisted selections. Later sampling rules — worst QC
  scores, failures, stratified by taxon — belong here, not in whichever module happens to want them.
- **`storage/`** — `imagestore` (LMDB, one store per project, byte-exact) and `tables` (archive-then-replace ingest,
  parquet replace/upsert/load, sqlite connect). No knowledge of any entity.
- **`records/`** — `occurrences` (normalize + save/load; ids are strings everywhere), `masks` (RLE encode/decode,
  upsert on `(occurrence_id, part)`, `derivation_hash`/`current_derivation_hashes`), `runs` (owns the sqlite
  schema for BOTH tables, and migrates old ones), `metrics` (long storage, `load_metrics`, `current_rows`,
  `latest_values`), `calibrations` (the scope/provenance machinery every calibration type shares, with an
  opaque `parameters` payload it never interprets).
- **`ingest.py` / `download.py` / `export.py`** — the generic in/out. Ingest archives the source into
  `imports/` before parsing (which is what makes `drop=` safe), and image ingest archives a *manifest*, not
  the pixels. `export.py` owns the wide view — `column_name`, `metrics_wide`, `metric_units` — which validation
  and `training/datasets` build on too.
- **`transforms/`** — `appendages`, `orient` (PCA, axis chosen by *asymmetry* rather than length), `crop` (crop,
  crop_to_mask, rotate, resize, remove_background).
- **`segmentation/`** — `groundedsam` (SAM2 with optional Grounding DINO; `detect_bounds=False` uses the
  point-prompt path for pre-cropped images), `manual` (draw/correct by hand — an alternative segmentation, not
  a separate system), `run` (`segment()` operation + `run_segments`).
- **`metrics/`** — `dimensions`, `position` (reports in ORIGINAL coordinates), `quality`, `color`, `outliers`
  (group metrics), `annotation` (human labels), `run` (`run_metrics` + `RunContext` + `completed_keys`).
- **`validation/`** — `masks`, `metrics`, `filters`. All comparison, nothing persisted.
- **`visualization/`** — four modules that spell out the model, smallest thing first:
  - `panels` — one picture of one operation's decision about one occurrence-part. Shared drawing helpers
    (`overlay_mask`, `diff_panel`, `annotate`, `side_by_side`) and the colour conventions they all obey, so
    learning to read one panel is learning to read all of them. Also `save_panel` and `PanelFiles`.
  - `grids` — many panels as one image: `image_grid`, `comparison_grid`. Pure layout, no project, no I/O.
    Panels must arrive display-ready uint8 — it will not rescale a float array, since two probability maps
    with different ranges would stretch to look identical.
  - `pipeline` — whose panels get laid out: `resolve_sample`, `RunReport` (one recipe/one part/one sample/many
    stages), and `PanelFanout` for a multi-output run's shared steps.
  - `products` — panels materialized per occurrence-part: `render_segments`.

  A Segment's `panel_sink` is a `RunReport` under a run, or `panels.PanelFiles` when you build a segment
  yourself and want full-resolution files — the latter is deliberately not reachable through `visualize=`.
- **`training/`** — `datasets`, `splits` (grouped and stratified, to avoid leakage).
- **`extensions/`** — `antenna_lighttraps` (api/ingest/download + `calibrations/scale`, scoped to Antenna's
  `event_id` — the worked example of a project choosing its own calibration scope) and `inat_insects`
  (api/ingest/download + `metrics/color`, `metrics/bioencoder`, `training/bioencoder`). Extensions normalize
  INTO core, never around it.

### Conventions to follow when extending

- **Operations are lowercase factory functions returning a configured `Operation`.** `remove_appendages()`,
  `body_length()`, `segment(model)`. The implementation is a module-level `_name(segment, **params)`. Metric
  factories take `name=None` and `unit=...` so the same operation can appear twice under different names.
- **Transforms return `(segment, info)`; metrics return a value.** `info` is a diagnostics dict recorded on the
  run. Include reliability flags (`unreliable`, `degenerate`) that callers should check.
- **Visualization is `segment.emit_panel(panel, "<stage>")`,** called unconditionally — it no-ops when
  `segment.panel_sink` is None, which is the case for every occurrence outside the sample. EMIT, not save: an
  operation draws what it decided and hands it over; where it goes is the run's business. The stage names the
  column. Panels must be display-ready uint8 — the operation knows what its own numbers mean, so it renders
  them.
- **Anything expensive that must happen once per run goes in `Operation.prepare(context)`,** not in
  `__init__`. That's how group metrics fit their reference population.
- **Individual failures are logged and counted, never fatal.** One bad occurrence must not cost a run.
- **Docstrings explain *why*.** The existing ones document the reasoning behind a choice, the failure mode a
  guard exists for, and what a number means — not what the code plainly says. Match that density; it's the
  house style.

### Things that are deliberately unfinished

- `extensions/inat_insects/training/bioencoder.py::train()` and `load()` raise `NotImplementedError`. The
  dataset preparation above them is real; the training loop is left out rather than guessed at, and the
  docstring says which decisions a caller has to make.
- `sketch1.py` and `scripts/sketches.py` are the original design sketches this package was built from. They
  reference an older `critterframes`/`critterframes_inat` naming, and `scripts/sketches.py` isn't valid Python.
  They're kept as the design record; the realized versions are `scripts/dragonfly_bodies_inat.py` and
  `scripts/dragonfly_wings_museums.py`.
