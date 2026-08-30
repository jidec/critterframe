# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CritterFrame (`critterframe`) turns organism images into trait tables: ingest occurrence metadata, download or
ingest images, segment the organism (and optionally named body parts) out of each, measure metrics off the results, and
export one row per organism.

## Setup

```
pip install -e .
pip install -e ".[torch]"           # torch + transformers, for the bundled SAM2 models and inat_insects' embeddings
cp .env.example .env                # credentials for extensions that call an external API
```

The core has no deep-learning dependency, deliberately — ingest, storage, transforms, metrics, export, and
validation all run without torch. Keep it that way: a new core module should not add a torch import.

## Running things

### The mechanical check: pytest

```
pip install -e ".[dev]"

pytest                              # ~1,100 tests, ~45s; needs no GPU, network, or credentials
pytest tests/unit -m "not slow"     # inner loop, a few seconds
pytest -n auto                      # parallel; the suite is tmp_path-isolated
pytest -m gpu                       # opt in to what's deselected by default
```

`tests/unit/` mirrors the package; `tests/integration/` is named for the invariant or workflow each file
protects (`test_staleness.py`, `test_repeat_awareness.py`, `test_derived_parts.py`, …) rather than for a
module, because a test spanning ingest → segment → measure → export isn't "about" any one of them.
`tests/helpers/` holds the shared synthetic specimens, the stub segmenters, and the fake session/cv2 — it is
what the smoke scripts import too, so both sides are looking at the same specimens.

Markers: `slow` runs by default (opt out with `-m "not slow"`); `gpu`, `network`, and `interactive` are
deselected by default and opted into with `-m gpu`. `tests/unit/test_hash_stability.py` holds three pinned
digests — read its docstring before changing one, because a failure there means every mask and metric in every
existing project has been invalidated.

**`scripts/simple_tests/**/*_test.py` are never collected.** They are scripts, not tests, and several want a
GPU, credentials, or a project at a hardcoded path. `python_files = ["test_*.py"]` in `pyproject.toml` is what
guarantees that (the scripts are `<thing>_test.py`); don't loosen it.

### The visual check: the smoke scripts

`scripts/simple_tests/` are standalone manual scripts mirroring the package layout. They print output and write
debug images for visual inspection, and that is now their whole job — the assertions they used to state in
English have been harvested into `tests/`. What they do that a test cannot is show you WHICH pixels: whether
`remove_appendages` took legs or a wing tip, whether `orient` picked the body or the wingspan. Run them
individually from the repo root after installing.

Four need nothing but the package itself:

```
python scripts/simple_tests/pipeline_synthetic_test.py      # full pipeline over drawn images
python scripts/simple_tests/recipes_test.py                 # hashing + coordinate inversion
python scripts/simple_tests/training/training_test.py       # split, export, register, rerun
python scripts/simple_tests/visualization/grids_test.py     # grid layout + sampling, no project
```

The synthetic pipeline script writes real pipeline grids and product renders, so open what it leaves behind
after touching `visualization/`. The rest need real state — a populated project, credentials, a GPU, or files
under `scripts/test_images/`.

`scripts/*.py` (excluding `simple_tests/`) are annotated reference pipelines, one per project shape. They point
at project paths that don't exist in this repo; read them as documentation of intended usage, and expect to
change `PROJECT_PATH` before running one.

No format/typecheck tooling is configured. `python -m pyflakes critterframe tests` is clean and worth keeping
clean.

### The docs site

```
pip install -e ".[docs]"
mkdocs serve           # live preview at localhost:8000
mkdocs build --strict  # what CI runs before deploying; fails on broken refs/links
```

`docs/index.md` is a `pymdownx.snippets` include of this README, not separate content — edit the README, not
that file. `docs/api/*.md` are thin `mkdocstrings` directives, one per subpackage; they render existing
docstrings as-is, so a new subpackage needs a new `docs/api/<name>.md` plus a `mkdocs.yml` nav entry, but a new
module inside an existing subpackage needs nothing (`show_submodules` picks it up). `.github/workflows/docs.yml`
deploys to GitHub Pages via `mkdocs gh-deploy` on every push to `main` that touches `critterframe/`, `docs/`,
`mkdocs.yml`, or `README.md`.

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
12. **A registered model is provenance about weights, and the FINGERPRINT is what reaches the recipe hash.**
    Training happens outside the package (`records/models.py` imports no framework); what a project records is
    the join between a checkpoint and the data behind it — task, framework, base model, training splits as id
    digests, opaque training `parameters`. `RegisteredModel.attach(network)` binds a loaded network to that
    record and forwards `predict`/`encode`/`visualize` to it while answering `identity()` from the registry, so
    retraining into the same filename moves the recipe hash and every mask and metric below it is correctly
    redone. Name and path are deliberately NOT in `identity()`: a copied project is the same model, and a
    reused name over different weights is not.

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

### Storage invariants that fail silently

Each of these is a guard whose removal produces wrong data rather than an error, so none of them is safe to
"simplify" without reading this first.

- **`upsert_table` compares keys BY VALUE, with no coercion.** Coercing to string first looks harmless and
  does three wrong things at once: it makes the integer `1` and the string `"1"` the same key while leaving
  `1` and `1.0` different, and it turns `None`/`NaN`/`pd.NA` into `"None"`/`"nan"`/`"<NA>"` — three distinct
  keys a literal string in the data can then collide with. Guaranteeing key types is the records layer's job
  (`records.masks.make_mask_row`); storage rejects what can't be an identifier. A key that silently fails to
  match doesn't error, it **duplicates** — a mask table growing a second copy of an occurrence-part per run.
- **`records.occurrences.validate_ids` stops an ingest on a duplicate or missing id** rather than dropping
  the row. Not recoverable automatically: the fix is a judgement about the data (two photos of one specimen,
  or two specimens sharing a number?), and silently keeping whichever copy came first picks an arbitrary
  winner and loses the other.
- **RLE mask counts are stored as raw bytes, not base64.** Parquet has a binary column type, so base64 would
  cost roughly a third of the mask table's size plus an encode/decode on every read and write, for nothing.
- **`ImageStore` is keyed by occurrence id and takes a project path**, so it can't be pointed at another LMDB
  in the project. Deliberate — it is the image store, not a generic blob store.

### Run semantics that aren't obvious from the signatures

- **A run's `RunContext.occurrence_ids` is the full set the run covers, BEFORE the completed/pending filter.**
  A group metric fits its reference population from that set, so resuming an interrupted run would otherwise
  fit against only the leftovers and change what "outlier" means mid-project.
- **A fitted group model is deliberately NOT in the recipe hash.** It is determined by the reference values,
  which are determined by `from_run`, so hashing it would add nothing but instability from model randomness.
- **`subset` is recorded on the run but not hashed.** Processing the rest of the project later continues the
  same work rather than counting as a different recipe.
- **A metric run's transforms are not persisted.** They shape what gets measured, and they're in the recipe
  hash, but the segment they produce is thrown away — only the value is kept.
- **Metric runs distinguish "no mask" from "measured nothing"** with a sentinel, so an occurrence segmentation
  hasn't reached is neither a failure nor a skip.
- **`metrics.quality.WARN_THRESHOLDS` is keyed by metric NAME**, because the name is what survives into
  storage and into an export column; keying by operation would not survive the round trip.
- **Export converts units after `drop_empty` and before `filters`**, so a threshold written in millimetres
  filters millimetres.
- **`training.splits` sorts ids before assigning them.** Without that the seed doesn't pin the split, and the
  same call reproduces a different partition depending on input order.
- **Training data comes from reference masks where they exist**, since training on canonical masks teaches a
  new model the old model's mistakes.
- **`records.occurrences.save_occurrences` re-validates ids** even though `normalize()` already did:
  `ingest_images` builds rows from filenames and never goes through `normalize`, so two colliding stems would
  otherwise reach the table unchallenged.

### Concurrency: what makes parallel runs safe

`run_segments(shard=(index, total))` is the parallel entry point; the pieces below are what make it safe, and
each was added for a failure that had no error message.

- **Shards are computed, not coordinated.** `selectionhelpers.shard_occurrences` sorts then takes a
  round-robin slice, so any number of workers given the same ids and the same `total` agree on the same
  disjoint split with no communication.
- **A sharded run never upserts `masks.parquet`.** `upsert_table` is a whole-file read-merge-overwrite with no
  locking, so two concurrent writers silently lose each other's rows. Each flush writes a brand-new file
  (`paths.mask_shard_path`) instead; `merge_mask_shards()` folds them in afterwards, single-writer. File
  locking was rejected on purpose: OS locks are unreliable over the network filesystems a cluster shares,
  and a never-before-used filename has nothing to race over.
- **Shard filenames sort in write order** (zero-padded `time.time_ns()`, then a uuid tiebreak). `merge_mask_shards`
  depends on that ordering to resolve the same occurrence-part staged twice by keeping the newest.
- **In `storage/sqlite.py`, `busy_timeout` must be set BEFORE `journal_mode=WAL`.** The one-time switch into
  WAL raises `SQLITE_LOCKED`, which `busy_timeout` does not cover (it only retries `SQLITE_BUSY`), so that
  pragma gets its own bounded Python-level retry. Getting the order wrong fails only under concurrency, on a
  brand-new project.
- **`download_images` threads the fetch only.** Batching and every `store.put_many()` stay on the calling
  thread, so concurrency never reaches the image store. `max_workers=1` reproduces the sequential behaviour.
- **`project/paths.py` and `selectionhelpers.py` import nothing from the package.** That is what lets ingest,
  run drivers, and visualization use them without acquiring a dependency on the metrics or export layers;
  `selectionhelpers` reaching into `export` previously created a real import cycle.

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
- **`storage/`** — one module per backend, none with any knowledge of an entity: `imagestore` (LMDB, one store
  per project, byte-exact), `tables` (parquet replace/upsert/load), and `sqlite` (`connect`, with the WAL and
  busy_timeout pragmas that make concurrent sharded runs safe).
- **`records/`** — `occurrences` (normalize + save/load; ids are strings everywhere), `masks` (RLE encode/decode,
  upsert on `(occurrence_id, part)`, `derivation_hash`/`current_derivation_hashes`), `runs` (owns the sqlite
  schema for BOTH tables, and migrates old ones), `metrics` (long storage, `load_metrics`, `current_rows`,
  `latest_values`), `calibrations` (the scope/provenance machinery every calibration type shares, with an
  opaque `parameters` payload it never interprets), `models` (the registry of trained models —
  `models/registry.json`, checkpoint fingerprints, `RegisteredModel`; provenance only, loads nothing).
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
- **`training/`** — `splits` (`split_ids` returns `{split: ids}`; grouped and stratified, to avoid leakage) and
  `datasets` (`iterate_segments` in memory, `export_training_data` to disk — splits as subsets or ids, optional
  class folders and mask PNGs, a manifest, and a `dataset.json` whose `data_hash` is what a registered model
  points at). Splitting decides, exporting materializes, and neither does the other's job.
- **`tests/`** — `unit/` mirroring the package, `integration/` named per invariant, `helpers/` shared with the
  smoke scripts. Testing conventions: real LMDB/parquet/sqlite in `tmp_path` (three of the invariants above ARE
  storage-format invariants, so a mocked store would assert nothing), fakes only for the network and the GUI,
  hashes asserted RELATIONALLY except for three pinned digests, and timestamps stripped before comparison
  rather than frozen — except the one place where the date is the behaviour (an import archived twice in a day).
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
- **Docstrings are reference, not essays.** They render as the API site, so keep them scannable:
  - A **module** docstring is one line — the module's line from README.md's package-layout tree — plus at most
    one short sentence a reader genuinely can't use the module without. No project-tree diagrams (the README
    has one), no design essays.
  - A **function** docstring is a one-line summary, an optional one- or two-sentence caveat, then
    `param -- what it is`, one line each, and a `Returns ...` line. No rationale inside the param block.
  - Write like README.md: present tense, declarative, `&`/`e.g.`/`i.e.`. No rhetorical openers ("Worth
    knowing", "The reason is", "which is why"), no parenthetical asides longer than a clause, ALL-CAPS
    emphasis at most once per docstring.
  - **Why goes here, not there.** A design decision, a rejected alternative, or a silent failure mode a guard
    exists for belongs in the Architecture section above — one copy, findable — not restated in every module
    that touches it. The exception is a hazard someone editing *this specific code* would otherwise walk into:
    that stays as a short `#` comment at the line it guards (see `_wait_for_key`'s duplication note, and the
    pragma ordering in `storage/sqlite.py`).

### Things that are deliberately unfinished

- `extensions/inat_insects/training/bioencoder.py::train()` and `load()` raise `NotImplementedError`. The
  dataset preparation above them is real; the training loop is left out rather than guessed at, and the
  docstring says which decisions a caller has to make.
- `sketch1.py` and `scripts/sketches.py` are the original design sketches this package was built from. They
  reference an older `critterframes`/`critterframes_inat` naming, and `scripts/sketches.py` isn't valid Python.
  They're kept as the design record; the realized versions are `scripts/dragonfly_bodies_inat.py` and
  `scripts/dragonfly_wings_museums.py`.
