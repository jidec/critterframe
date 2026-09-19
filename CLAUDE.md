# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CritterFrame (`critterframe`) turns organism images into trait tables: ingest occurrence metadata, download or
ingest images, segment the organism (and optionally named body parts) out of each, measure metrics off the results, and
export one row per organism.

## Setup

```
pip install -e .
pip install -e ".[torch]"           # torch + transformers, for the bundled SAM2 models and bioencoder's embeddings
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
   at ingest because the source file is archived *before* parsing, so `raw_imports/` keeps every dropped row.
   Extensions own the vocabulary (`antenna_lighttraps.ingest.NON_ORGANISM_DETERMINATIONS`). See `ingest.py`
   under Package layout below for the raw import/import vocabulary and how turning one into the other is
   recorded.
3. **Any number of parts per occurrence**, defaulting to `"organism"` (`recipes.DEFAULT_PART`).
4. **At most one canonical mask per occurrence-part**, in `masks.parquet`, RLE-encoded, **always in the
   coordinates of the original analysis image**. Reference masks live in an identical table
   (`reference_masks.parquet`) reached with `reference=True`. Called *reference*, never "ground truth" --
   a reference is whatever you chose to compare against, and naming it truth would assert the answer
   validation exists to measure. A mask row also carries `info`: `{operation label: scalar info}` from every
   step that made it (shared steps, transforms either side of the segmenter, the segmenter itself), as JSON.
   It is never hashed and is replaced with the mask, so it can never go stale; `metrics.mask_info()` is the
   bridge that brings it into the metrics table, where it can be exported and filtered like any value.
5. **The image store holds byte-exact encoded images.** `ImageStore.put()` takes BYTES, not arrays, and
   refuses arrays with a `TypeError`. Decoding and re-encoding would recompress JPEGs, flatten 16-bit to 8-bit,
   drop alpha, and discard EXIF — invisibly and irreversibly. `get()` decodes to 8-bit BGR (the working view
   every transform/metric/model assumes); `get_bytes()` is the full-fidelity escape hatch. **Never add a write
   path that encodes an array into the store.**
6. **Segments are never persisted** — a segment is an image plus a mask, and both already persist separately.
7. **Metrics are any derived value**: traits, QC scores, human labels, embeddings, cluster assignments, outlier
   scores. Stored long in `runs_and_metrics.sqlite`, reshaped wide at export. That includes what a metric
   run's transforms reported: each transform's scalar info is stored as a sibling row named by its label
   (`segments.operation_labels`, so a repeat is `orient_2`) with unit `records.metrics.TRANSFORM_INFO_UNIT`,
   because a length measured after an unsure `orient` is suspect in a way only that row says. Export and
   `metrics_wide` leave those columns out unless `transform_info=True` (or they're named in `metric_names`) —
   `compare_metrics` would otherwise try to difference a boolean — and export drops them after filters have
   run, so a filter can use one without exporting it; a default export's manifest and hash never see them.
   `occurrences_matching` always sees them, so a rule can name `orient__unreliable`. Not in
   the recipe hash — recording them doesn't change a measured value — so occurrences measured before they were
   recorded get theirs only on a `force=True` rerun.
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

    A per-group cap at import (`ingest_occurrences(group_col=, max_per_group=)`, via
    `selectionhelpers.cap_per_group`) looks like it belongs on the filter side — it's a threshold, not
    membership — but it judges something neither `drop=` nor a filter does. `drop=` and a filter both judge
    ONE occurrence: whether it's an organism, whether its result is good enough. A per-group cap judges the
    POPULATION: every row in an oversized group is an equally valid, equally wanted candidate, so capping it
    isn't a verdict on any specimen, it's a statement that the project doesn't need this many of this kind.
    That's a different question from the one this item's threshold ban guards against — not "is this specimen
    good", but "how much of this population earns the pipeline's (expensive, per-occurrence) effort" — so it's
    safe to answer at import despite being degree rather than membership. It stays cheap to revise for the same
    reason `drop=` does: the archived import and item 11's full-snapshot reingest mean revisiting it costs
    exactly the new work the newly-admitted specimens need, never a repeat of work already done. `cap_per_group`'s
    `keep_ids` makes that literal — an occurrence the project already kept stays kept, only retrimmed by
    `cap_rule` if the cap has since shrunk — so raising `max_per_group` on a later pull is additive, never a
    reshuffle that orphans masks or metrics already computed for a specimen still worth keeping. An export
    filter could never move to import instead: what it thresholds is a computed metric that doesn't exist
    until after the pipeline runs on the very population a cap is scoping.

    A third import-time judgment, `selectionhelpers.dedupe_by`, answers a different question again: not "is
    this an organism" (`drop=`) or "does the population need this many" (`cap_per_group`), but "are these two
    rows the same real thing" — the same sighting independently published by two aggregators (GBIF fed by both
    iNaturalist and Observation.org, say), each minting its own id, so no id-based check can see the collision.
    It fingerprints rows on caller-given key columns (optionally rounding a numeric one) and delegates the
    actual keep-one-per-group work to `cap_per_group` (`max_count=1` over the computed fingerprint) rather than
    duplicating it. Unlike `drop=`, this is a probabilistic match, not a source-declared fact — two genuinely
    different sightings can share a fingerprint by coincidence — so key columns and precision should be no
    looser than the real risk of a false match warrants. A row missing any key column is exempt, kept
    untouched, the same reasoning `cap_per_group` applies to a missing `group_col` value.
    `ingest_occurrences(dedupe_key_cols=)` is where it runs, OFF by default — only a project actually
    combining more than one source needs it. It sits AFTER `drop=` and before the cap, and that order is the
    point: an ABSENT record must not be able to win a duplicate group and take the real sighting down with it,
    and a per-species cap should count real specimens rather than a sighting inflated by however many
    aggregators published it. A whole key column absent from the table (as opposed to blank on some rows, which
    `dedupe_by` already exempts) skips deduplication with a warning rather than failing the ingest, since
    narrowing the columns read without them is unexceptional, not a mistake. It is recorded in the import hash
    only when it is on, so every import archived before this stage existed keeps the hash it was archived
    under. `extensions.gbif_darwincore_inat` forwards it and supplies `DEFAULT_DEDUPE_KEY_COLS`
    (`decimalLatitude, decimalLongitude, eventDate`), the ready-made fingerprint for a GBIF-mediated pull.

    `ingest_occurrences(prefer={column: values})` is not a fourth judgment but a ranking inside the last two: it
    never excludes a row by itself. `dedupe_by` never removes a preferred row (a preferred source is trusted not
    to publish one sighting twice) and removes any other row sharing its fingerprint; `cap_per_group` fills a
    group from preferred rows before the rest. It deliberately outranks `keep_ids`: preferring a source is the
    point, so a later pull with more preferred rows displaces non-preferred specimens kept earlier — the one
    place a cap is allowed to be a reshuffle rather than additive, and only because the caller asked for it.
    Recorded in the import hash only when set. `extensions.gbif_darwincore_inat`'s `prioritize_inat=True` is
    `prefer=INAT_OCCURRENCES` (`institutionCode="iNaturalist"`).
11. Imports are complete snapshots - imported occurrences replace the current occurrence table and reimporting an image folder ingests any new images
12. **A registered model is provenance about weights, and the FINGERPRINT is what reaches the recipe hash.**
    Training happens outside the package (`records/models.py` imports no framework); what a project records is
    the join between a checkpoint and the data behind it — task, framework, base model, training splits as id
    digests, opaque training `parameters`. `RegisteredModel.attach(network)` binds a loaded network to that
    record and forwards `predict`/`embed`/`visualize` to it while answering `identity()` from the registry, so
    retraining into the same filename moves the recipe hash and every mask and metric below it is correctly
    redone. Name and path are deliberately NOT in `identity()`: a copied project is the same model, and a
    reused name over different weights is not. **A model class used standalone, outside the registry, answers
    the same way** — `SMPSegmenter` and `BioEncoderModel` hash `records.models.fingerprint_file` of their
    checkpoint, not its path, since a path got both halves of that rule backwards. Each caches the digest on
    the instance, because `Recipe.hash` is read many times per run and a checkpoint is hundreds of megabytes,
    so replacing the file under a live object keeps the old identity: build a new model after retraining.
    Registering is how a checkpoint gets the rest of its provenance, and a model type that needs construction
    knobs to load again supplies its own wrapper for it (`smp_segmenter.training.register_trained` records
    `encoder_name`/`size` in `parameters`, which `load_registered` reads back rather than trusting whatever
    the module defaults are by then).
13. **An export carries its own identity, twice.** Everything that says what an exported number IS — the
    recipe hash behind it, the run that recorded it, the mask it was measured from, the subset, the filters,
    the calibration under an mm column — is read inside `export_metrics` and then dropped by the long-to-wide
    reshape, leaving a CSV of bare numbers. So every export writes a manifest: `<name>.export.json` beside the
    file, because a CSV handed to a collaborator has to carry its identity with it, and one appended line in
    `exports/exports.jsonl`, because an export written outside the project is exactly the case where only the
    project can say what it handed out. `path=None`, the default, writes a uniquely-named file under the
    project's own `exports/` folder rather than losing the table, since a call site that names nothing still
    wants its data kept; `path=False` returns the DataFrame without writing a CSV at all, and still logs with
    a null path. `manifest=False` writes neither. `load_exports()` reads the log back.
    `export_hash` covers what was selected and what came out and nothing else — not the timestamp, not the
    filename — so one table written twice is recognizably one export, on the reasoning that keeps a path out
    of `RegisteredModel.identity()`. **An export is not a run**, for the reason `RUN_KINDS` gives: it derives
    nothing, so like a render it gets a hash-named artifact rather than a run record. Sets of occurrences are
    named the way every other record names them, `records.occurrences.ids_record` — a count and an
    order-independent digest, never a list that grows with the project — but the distinct mask
    `derivations` ARE listed, since that is what moves when the masks underneath are replaced and it is
    bounded by the number of segmentation recipes rather than by occurrences. A predicate filter is recorded
    by NAME: a lambda's repr carries a memory address that would make one export hash differently every run.

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
  derives no data, so the hash naming its folder is the whole of its provenance. `name` is the one field
  `spec()` carries that `Recipe.hash` doesn't — recorded on the run and shown by `describe_run()`, but not
  identity, the same treatment `subset`/`context` get and for the same reason (see Repeat-awareness).

### Repeat-awareness

`run_segments` and `run_metrics` both compute their recipe hash, ask the store which `(occurrence_id, part)`
pairs that hash already covered (`records.masks.completed_keys` / `metrics.run._completed_keys` — the metric one
lives with the run because "what work is left" is a property of the run, not of the stored values), and skip
them. This is what makes runs interruptible and makes expensive metrics behave like cached derived data.
`force=True` overrides. **This is a behavioural guarantee — don't add a code path that writes results without
a recipe hash, or that mutates a recipe after a run starts.**

Completion is keyed on `(recipe hash, source mask)`, not the recipe hash alone, on both sides:

- `run_metrics` passes the `derivation_hash()` of the masks it is about to measure into
  `metrics._completed_keys`, so a value computed from a mask that has since been resegmented never counts as
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

`Recipe.hash` deliberately excludes `name` (see the `Recipe` bullet above), which is where the two
`completed_keys` functions genuinely diverge for the first time. `records.masks.completed_keys` needs no name
scope: nothing ever reads a mask BY `run_name` (`masks.parquet`'s key is `occurrence_id`+`part`), so once the
hash no longer encodes it, a segmentation run renamed with no other change simply recognizes the existing
canonical mask as its own — free, and correct, since there is no per-name copy for anything to leave empty.
`metrics.run._completed_keys` is the opposite: `run_name` is what `export.column_name` and
`records.metrics.latest_values` key values back by, so an unscoped check would let a brand-new name silently
inherit another name's completion and process nothing, leaving that name's export column permanently empty.
It therefore takes an explicit `run_name` and joins to `runs.name`; `run_metrics` calls it twice — once scoped
to the run's own name (what to skip outright) and once unscoped (what some OTHER name already computed under
this exact recipe) — and for the second set, `_copyable_rows` re-inserts the existing values under the new
run's `run_id` instead of recomputing them, a plain `INSERT`-shaped read with no image, transform, or model
involved. The `copied` count in `run_metrics`' return dict is this, distinct from `processed`.

This copy step is only safe because of one more check: `metrics.run._current_for_population` (see the group
model bullet below) narrows both the same-name and cross-name candidate sets to rows whose reference
population actually matches this run's own fit, so a group metric never copies — or silently treats as
current — a score fit against a population this run wouldn't have used.

Skipping is sound only because an identical hash means identical work, and for one kind of operation it
doesn't. `Operation.deterministic` (True everywhere but `manual.draw_mask`/`correct_mask`) says whether a
rerun would reproduce the output; `Recipe.nondeterministic_operations()` reports them. Two people painting
one crop hash alike and produce different masks, so "already covered by this recipe" is genuinely ambiguous
— resume the annotation session, or make a second pass? `run_segments(force=None)`, the default, **raises**
rather than picking: explicit `force=False` resumes, `force=True` redoes. For every deterministic recipe
`None` still means `False`, so nothing else changed. **The flag is NOT in `spec()`** — it doesn't change
what one execution produces, only whether an earlier one may stand in, and hashing it would orphan every
recipe hash on disk. `metrics.annotation`'s human operations have the same hazard and deliberately keep
today's behaviour; the machinery is in place if that changes.

### Hashes: one rule, different consumers

Every hash is `recipes.hash_spec` (16 hex chars over canonical JSON), and every one follows the same rule: **it
covers exactly what changes the output; who, when, where and what-it's-called are recorded beside it, never in
it** (names, subsets, timestamps, paths, `visualize`, `force`). What differs is what reads the hash back, and that
sets what a change costs:

| Role | Hashes | Read back by | Changing its computation |
|---|---|---|---|
| Work identity | recipe hash (segment/metric) | skip logic, row currency, the per-part recipe pointer | every stored mask and metric goes stale |
| Lineage | `derivation_hash`, stored as `source_mask_hash` | currency of metrics and derived masks | the same, cascading through `from_part` |
| Content identity | `fingerprint_file`, `ids_digest`, `data_hash`, `import_hash`, `export_hash` | recipe hashes, idempotent ingest, recognizing a repeated export | work or imports re-identified |
| Scope or name only | failure `context_hash`, report `identity_hash`, products folder | retry decisions, filenames, render skip | retries or renamed files; no data affected |

Only the first two rows are pinned by `tests/unit/test_hash_stability.py`. A pipeline report usually reuses the
run's own recipe hash, but there it only locates files, so a report may be deleted freely. The one identity that
sits beside a hash rather than inside it is a group metric's reference population (`context_json`, checked by
`metrics.run._current_for_population`; see Run semantics). When adding a hash, decide which row it belongs to
first; that decides whether it must be stable forever.

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
- **A source CSV is read without type inference** (`ingest._CSV_READ`: `dtype=str`, only an empty field
  missing). Pandas' own guessing turns an id `007` into `7` (stored as `"7"`, matching nothing upstream) and a
  value `"NA"` into missing, with no error. The cost is that a column not in `numeric_cols` is a string, so a
  `drop=` rule or subset `values=` written with numbers matches nothing on it — list the column instead.
- **RLE mask counts are stored as raw bytes, not base64.** Parquet has a binary column type, so base64 would
  cost roughly a third of the mask table's size plus an encode/decode on every read and write, for nothing.
- **`ImageStore` is keyed by occurrence id and takes a project path**, so it can't be pointed at another LMDB
  in the project. Deliberate — it is the image store, not a generic blob store.
- **`write_table`/`upsert_table` write through a temp file, never straight to the destination.** Both call
  `_atomic_to_parquet`, which writes to a uniquely-named temp file beside the destination and `os.replace()`s
  it into place — atomic on the same volume on both POSIX and Windows. A process killed mid-write (crash,
  Ctrl+C, disk full, a sync client touching the file) leaves the previous complete file rather than a
  truncated one Arrow can't open. The temp name deliberately doesn't end in `.parquet`, so a leftover from an
  interrupted write can't be mistaken for a staged shard by `merge_mask_shards`' `*.parquet` glob.

### Run semantics that aren't obvious from the signatures

- **A run's `RunContext.occurrence_ids` is the full set the run covers, BEFORE the completed/pending filter.**
  A group metric fits its reference population from that set, so resuming an interrupted run would otherwise
  fit against only the leftovers and change what "outlier" means mid-project.
- **`runs.context_json` records what a run covered and what its operations fit**, beside the recipe that says
  what the work was. Neither is derivable from the other and, like `subset`, it is not hashed. Every run
  stores its occurrence set as an `ids_record`; a metric run also stores whatever `prepare()` handed back,
  keyed by metric name. That channel exists because a group metric's reference population is decided by the
  RUN, not the recipe: `limit=` narrows it and `latest_values` moves under it, so one recipe hash could mean
  two different fits with nothing recording the difference — and which groups fell back to the
  population-wide model lived only in a log line. `Operation.prepare()` may now return a JSON-serializable
  record and `Recipe.prepare_all()` collects them. Added as a nullable column, so an old database is missing
  it rather than broken by it, but `records.runs` still has to migrate on open because `start_run` names it.
- **A fitted group model is deliberately NOT in the recipe hash.** It is determined by the reference values,
  which are determined by `from_run` AND by `context.occurrence_ids` (this run's own `subset`/`limit`) — and
  the latter is exactly as unhashed as `subset` always is, for the same reason: hashing it would defeat
  incremental processing, since every occurrence added to a growing project would move the hash and force a
  full rescore of everyone already measured. Unlike every other operation, though, a group metric's VALUE
  genuinely depends on that population, so two runs can share a hash while having fit different models — the
  gap `runs.context_json` above exists to make visible. `metrics.run._current_for_population` is what
  actually CHECKS it, using data `prepare()` already returns and `context_json` already stores (no new
  column): a stored group-metric value only counts as current, or safe to copy onto a differently-named run,
  when the reference population it was fit against matches this run's own. This closes the gap for real —
  growing the reference population and rerunning the same `run_name` now correctly rescopes previously-scored
  occurrences instead of leaving them silently stale — and it costs nothing extra, because `prepare()` already
  runs, unconditionally, before the completion check even looks at it.
- **`subset` and `name` are recorded on the run but not hashed.** Processing the rest of the project later
  continues the same work rather than counting as a different recipe (`subset`); renaming a run doesn't change
  what running it produces, so it must not force every occurrence to be treated as unfinished work, or cascade
  a resegmentation through every `from_part` chain below it (`name` — see the `Recipe` bullet and
  Repeat-awareness above for how the two kinds diverge once it isn't hashed).
- **A metric `run_name` is pinned to one recipe per part, and moving it onto a different one has to be said out
  loud.** `records.runs.resolve_recipe_currency` raises if `(kind="metric", name, part)` already points at a
  different `recipe_hash`, unless `force=True` — `run_metrics`' own `force` argument, doing double duty. This
  exists only for metrics: masks.parquet upserts to a single current row per occurrence-part, so a *segment*
  run_name cycling through several recipe hashes over a project's life is exactly what resegmenting is, not
  ambiguity — `resolve_recipe_currency` is a no-op for `kind="segment"`. Metrics have no such row; the table is
  append-only, and `export.metrics_wide`/`records.metrics.latest_values` key on `run_name` alone, so two
  recipes sharing a name would otherwise silently interleave under one export column or one group-metric fit.
  The pointer itself lives in a small `current_recipes` table, separate from the immutable `runs` history, the
  same way masks.parquet sits beside the segmentation run log — moving it is not from
  `resolve_recipe_currency` itself but from a separate `commit_recipe_currency` call made only after the forced
  run has actually processed something, so a forced change that fails for every occurrence leaves the previous
  recipe's values current rather than emptying the export. `records.metrics.current_rows` checks this pointer
  alongside its existing mask-currency check — a value needs both to be current. A project with runs from
  before this pointer existed seeds it from run history's own insertion order on first use, so upgrading
  doesn't move anything under a name that hasn't actually changed. Canonical and reference measurements need
  their own names for the same reason a resegmented part does: they're meant to coexist for comparison, not
  supersede one another, and `force=True` would make whichever ran last win. `run_segments`' `run_name`
  defaults to `part` (each output's own part, for `outputs=`) rather than one fixed generic label, since name
  carries no such weight there — except `reference=True`, which folds into the default too
  (`<part>_reference`), for exactly the coexist-not-supersede reason just given: `resolve_recipe_currency`
  never runs for segments, so nothing else would stop a default-named reference pass from silently reading, in
  history and `describe_run(name=...)`, as though it superseded the canonical recipe over the same part rather
  than existing alongside it. `run_metrics`' `run_name` defaults too, but only where a default is genuinely
  unambiguous: measuring exactly one metric defaults to that metric's own `metric_name` (`metrics=
  [usability_annotation()]` needs no `run_name=` at all), since typing the same name twice is pure repetition
  for the single most common shape of call (a lone screening pass). Measuring more than one metric in a call
  has no single obvious name and keeps raising rather than guessing one — guessing wrong would silently rename
  an export column the moment a second metric gets added to an existing call, exactly the instability a
  meaningful, human-chosen `run_name` exists to avoid.
- **A metric run's transforms are not persisted.** They shape what gets measured, and they're in the recipe
  hash, but the segment they produce is thrown away — only the values are kept, plus each transform's scalar
  info as a `transform_info` row (see item 7).
- **Metric runs distinguish "no mask" from "measured nothing"** with a sentinel, so an occurrence segmentation
  hasn't reached is neither a failure nor a skip.
- **A colour threshold is its spec.** `metrics.color_thresholds.ColorThreshold` is a named set of half-open
  cutoffs, possibly across several colour spaces, and its `spec()` is what reaches the recipe hash, so moving any
  cutoff is a different recipe. Thresholds in one `threshold_fractions` are scored independently and may overlap;
  the `"unmatched"` share is opt-in rather than implied, because a partition is a claim about the thresholds the
  caller hasn't necessarily made. Fitted thresholds (`inductive_color_thresholds`) are the same type, and a group
  metric records their specs in `context_json`, so a stored value says exactly which cutoffs produced it.
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
for, not by what's in it. Interactive windows (`manual`, `annotation`, `scale_from_click`) are inputs, not
either kind; they keep their duplicated `_wait_for_key` for the cv2-stub reason documented beside it.

- **`pipeline/`** — every activity's diagnostics: how processing BEHAVED, for runs and for everything else
  (download, ingest, calibration, validation, splits, dataset export, training, group-metric fits). One
  machinery serves all of them, `visualization/pipeline.open_report`, so `visualize=`/`visualize_every=` mean
  the same thing everywhere and default to `True` everywhere — safe because every mode is bounded. A report is
  one flat filename stem, `<name>[__<part>]_<identity hash>`, and everything it writes shares it:
  - **the grid**, `<stem>.jpg` — panels for a bounded set of items, a stage per column, an item per row. Items
    are any string key: an occurrence id, a calibration scope value, an Antenna event, a validation specimen.
    `visualize=25` samples 25, `True` a default 25, `["a","b"]` names them, `False` opens a `NullReport` whose
    methods all no-op, so call sites never branch on the flag. **There is no per-item file mode** — a
    10,000-occurrence activity can't be inspected as 10,000 files. The default sample is deterministic
    (`selectionhelpers.sample_occurrences`), so two versions of a recipe show the SAME specimens and can be
    compared cell by cell. `rank="lowest"`/`"highest"` keeps the N items with the worst value passed to
    `done()` instead (worst IoU first, weakest scale match first), holding only N items' cells at once.
  - **checkpoints**, `<stem>__<label>.jpg` — two kinds. `visualize_every=N` writes `__at<N>` every N items,
    each resampled from only the items THAT window processed rather than the fixed sample above, so a very
    long run's checkpoints always have content even before the fixed sample is reached. `checkpoint(label)`
    snapshots the grid as it stands and empties it, for re-drawing the same sample repeatedly (`__epoch0005`).
    Items that finish out of order (concurrent downloads) are handed to the report in order, holding only
    what `planned()` says a grid will show.
  - **figures**, `<stem>__<name>.png` — whole-population pictures that belong to no item: loss curves, threshold
    sweeps, histograms, split counts, ingest funnels, a group metric's fitted population (through
    `RunContext.report` in `prepare()`). Built by `visualization/figures` on a bare Agg canvas: pyplot is never
    imported, so nothing needs a display or touches a global backend, and matplotlib loads only on first use.
  - **the sidecar**, `<stem>.report.json` — identity spec, the items shown, counts, a capped failure list,
    and the files written. The only place a failure with no image to draw (a dead URL) shows up.

  A report's `<name>` is the public function that opened it, `<function name>[__<qualifier>]` — `measure_scales`,
  `split_ids`, `validate_masks__candidate_a`, `measure_scales__antenna` — so a file in `pipeline/` says what
  call to re-read to understand it. The identity hash is whatever already names that output — recipe hash,
  `import_hash`, `data_hash` — or a `hash_spec` over the arguments that change it, so two configurations never
  overwrite each other. An export
  whose hash is only known after writing names its report with `identify()` before closing. A grid can only
  show work that happened, so a fully-cached rerun writes none — that's what `force=True` is for.
- **`products/<name>_<hash>/`** — assets deliberately materialized for downstream use, **one file per
  occurrence-part**, named `<occurrence_id>.png` (or `<occurrence_id>__<part>.png` when a render covers several
  parts). `render_segments()` is the one that exists. Loose files, not LMDB, because these are for figures and
  for R, and the store exists to hold original images byte-exactly for Python.

A render derives nothing and records nothing: no mask, no metric, no run row. It hashes its transform chain
only so the folder name identifies what's in it and a rerun is a no-op. Don't add a code path that makes a
picture into a measurement.

**The three per-part drivers return `{part: summary}`** — `run_segments`, `run_metrics`, `render_segments` —
one entry even for a single part, because a total over several parts is the one number that answers nothing
("12 rendered, 4 with no mask" doesn't say whether a part is missing everywhere or present everywhere).
`validate_masks` does the same with `parts=`, returning `{part: DataFrame}`, and a bare frame without it.
Only segmentation takes a per-part CHAIN (`outputs={part: steps}`), because it alone runs genuinely different
work per part: a wing segmenter and an abdomen segmenter are different models. The rest apply one chain to
each part, so a `parts=` list is all they need.

### Package layout

- **`project/`** — `paths` (every project path, returning `pathlib.Path`; creates nothing), `subsets` (named
  selections, `subsets.toml`, and `select_occurrences`, which every run funnels through), `summarize`.
- **`calibration/`** — one module per kind of calibration, holding what it MEANS. `scale` (px/mm from a
  target of known size: `scale_from_target`, `measure_scales`, `declare_scale`, `scale_for_occurrences`);
  `color` not written yet and, when it is, beside `scale.py` rather than inside it. The detector is generic on
  purpose — target, size, and search region are all arguments — and a weak match is accepted but warned about,
  since clutter can out-correlate an absent target and a plausible wrong scale is worse than none.
- **`segments.py`** — `iterate_segments`, the per-occurrence loop most drivers walk (renders, validation,
  dataset export, the pooled-pixel colour metrics), plus `build_segment` for the `from_part` framing, `Tally`
  for what a driver counts, and `NoInput` for an occurrence with nothing to work from yet. `run_segments` and
  `run_metrics` keep their own loops — a multi-part fork in one, per-occurrence writes and cross-name copies in
  the other — but build and count with the same pieces. Torch-free and records-only, so it sits in core
  rather than under `training/`, where it used to live and where two colour metrics had to reach for it.
- **`maskops.py`** — mask arithmetic with no project attached: `mask_iou`, `mask_coverage`,
  `pad_to_common_shape`, `mask_bounds`, `largest_component`. One answer to "how much do these two masks agree"
  for validation, manual correction, mirror symmetry and the trainable segmenter alike.
- **`colorspaces.py`** — colour space conversion with no project attached: `convert` (uint8 BGR to `rgb`, `linrgb`,
  `hsv`, `hls`, `lab`, `lch`), `to_bgr`, `in_arc`, `normalize`, and the `SPACES` registry. **Everything comes out in
  canonical units, never OpenCV's 8-bit encodings**: float32, hue in degrees, Lab with L 0-100 and a/b signed. The
  8-bit path stores Lab a/b offset by +128 and hue on 0-179, which is harmless for a translation-invariant distance
  and wrong for chroma or a hue arc, so no metric calls `cv2.cvtColor` for a colour space itself. `ColorSpace.circular`
  names the channels that wrap (hue), and `in_arc` is the one wrap-aware membership test. Imports nothing from the
  package, the same rule as `maskops` and `selectionhelpers`.
- **`timing.py`** — `timed()`, the "<label> in Ns" line around a slow step. A multi-gigabyte ingest spends
  minutes inside single calls, and a caller watching the log otherwise sees nothing between "start" and
  "done". Nothing is logged if the block raises: a line saying a step finished, when it didn't, is worse than
  no line.
- **`devices.py`** — `resolve_device()`, the one place that asks whether there is a CUDA device. Torch is
  imported inside the function, so core stays importable without it, and every model class stays lazy: asking
  at construction time would make BUILDING a recipe — which may never run — initialize CUDA.
- **`selectionhelpers.py`** — transient "out of these occurrences, which ones" helpers: `sample_occurrences`
  (deterministic, so a sample is stable across runs and recipes), `sample_per_group` (the same, stratified —
  smallest group first, each capped at its own size, so a common group's rollover tops up the rest rather than
  crowding out a rare one), `rows_matching` (the `{column: values}` test behind ingest's `drop=`; any rule
  matches, a missing value never does, an unknown column raises), `require_present`/`exclude_present` (AND-
  compose several "does this id exist somewhere" checks — has an image, has a mask for a part — into one call;
  the reads themselves live with the data, e.g. `storage.imagestore.ImageStore.keys()`,
  `records.masks.occurrence_ids_with_mask()`, passed in already-materialized rather than read here), and
  `worst_n` (the `{occurrence_id: value}` → worst-N-sorted rule behind `validate_masks`'/`compare_metrics`'
  `show_worst`). Distinct from `project/subsets`, which is about named, persisted selections.
  **Nothing here reads a project or imports anything else from the package** — deliberate, not incidental:
  reaching into `export` for this once created a real import cycle, so every selection rule stays a pure
  function over data the caller already has, the same split `export.occurrences_matching` uses (it reads,
  then calls `rows_matching` for the comparison). `records.masks`/`storage.imagestore` are safe to import
  directly (no transitive path back to `selectionhelpers`), but the rule stays absolute anyway: "zero package
  imports" needs no re-verification as the rest of the graph changes; "these specific ones happen to be safe"
  does, every time something downstream moves.
- **`storage/`** — one module per backend, none with any knowledge of an entity: `imagestore` (LMDB, one store
  per project, byte-exact), `tables` (parquet replace/upsert/load), `jsonfiles` (`atomic_write`, `write_json`,
  `append_jsonl`, `read_jsonl` — every manifest, registry and append-only log in the package, UTF-8 and
  all-or-nothing), and `sqlite` (`connect`, with the WAL and busy_timeout pragmas that make concurrent sharded
  runs safe). `records.runs.open_database` is a context manager that CLOSES: an unclosed read holds a file
  handle for the life of the process, which on Windows stops the project directory being moved.
- **`records/`** — `occurrences` (normalize + save/load; ids are strings everywhere, plus `ids_digest` and the
  `ids_record` every record names a set of occurrences with), `masks` (RLE encode/decode,
  upsert on `(occurrence_id, part)`, `derivation_hash`/`current_derivation_hashes`), `runs` (owns the sqlite
  schema for BOTH tables, and migrates old ones), `metrics` (long storage, `load_metrics`, `current_rows`,
  `latest_values`), `calibrations` (the scope/provenance machinery every calibration type shares, with an
  opaque `parameters` payload it never interprets), `models` (the registry of trained models —
  `models/registry.json`, checkpoint fingerprints, `RegisteredModel`; provenance only, loads nothing), and
  `failures` (what not to retry, keyed `(occurrence_id, part, stage)` — so a reference pass records under
  `segment_reference`/`metric_reference` rather than sharing `segment`/`metric` with the canonical one, whose
  success would otherwise clear it). Both run drivers persist failures and both take `retry_failed=`: the
  ledger is scoped by a `context_hash` of the recipe chained with the mask it worked from, so a retuned recipe
  or a resegmentation retries by itself and only an unchanged repeat of the same attempt is skipped.
- **`ingest.py` / `download.py` / `export.py`** — the generic in/out. A RAW IMPORT is a source's data exactly as
  it arrived; an IMPORT is what that becomes once `ingest_occurrences` reshapes it (id_col, image_url_col,
  `transform=`, which takes one callable or a SEQUENCE of them — an extension stacking its own derivations on
  a caller's must pass a sequence, since a closure around both would record only the wrapper's name and two
  different callers' transforms would then share an import hash) and narrows it by judgement (`drop=`,
  dedupe, `group_col`/`max_per_group`) into occurrences. The raw import is
  archived into `raw_imports/` before any of that runs (which is what makes `drop=` safe), content-deduplicated
  so the same bytes are never stored twice even under different decisions, and a manifest recording every
  structural and judgement decision is written beside it and into `raw_imports/imports.jsonl` (`load_imports`
  reads the log back) — the same shape `export.py`'s manifest takes for what leaves a project, applied to what
  enters one. That manifest's hash is also what makes ingest idempotent: the same raw content turned into an
  import the same way twice is recognized and skipped. A source whose own parse is the expensive step (a
  multi-gigabyte Darwin Core Archive) hands that parse to core as `read=` (and its raw bytes as `raw=`), which
  core runs only after the skip check, so the decisions are hashed in exactly one place rather than passed
  separately to each check. A CSV is read with every column as a string; only `numeric_cols`/`datetime_cols`
  are typed. Image ingest archives a *manifest* of its own, not the
  pixels, and has no import manifest of this kind — see its docstring for why. `export.py` owns the wide view —
  `column_name`, `metrics_wide`, `metric_units`, all built on one `_current_long` read — which validation and
  `training/datasets` build on too, plus the export manifest (`load_exports` reads the log back).
- **`transforms/`** — `appendages`, `orient` (PCA, axis chosen by *asymmetry* rather than length), `crop` (crop,
  crop_to_mask, rotate, resize, remove_background).
- **`segmentation/`** — `groundedsam` (SAM2 with optional Grounding DINO; `detect_bounds=False` uses the
  point-prompt path for pre-cropped images), `manual` (draw/correct by hand — an alternative segmentation, not
  a separate system), `run` (`segment()` operation + `run_segments`).
- **`metrics/`** — `dimensions`, `position` (reports in ORIGINAL coordinates), `quality`, `pixels`
  (`masked_pixels`, the one rule every colour metric reads pixels by), `color_means` (plus grey-world
  `white_balanced_color` and `background_color`, for photography whose lighting nothing controls), `color_thresholds`
  (`ColorThreshold`, `threshold_fractions`, and the black/hue presets built on them), `inductive_color_thresholds`
  (the same `ColorThreshold`s, fitted per group in `prepare()`), `color_clusters` (a KMeans palette fitted per group
  from pooled pixels, scored as each organism's share of it), `outliers` (group metrics), `annotation` (human
  labels), `mask_info` (the diagnostics a segmentation run stored on each mask, as a metric), `run`
  (`run_metrics` + `RunContext` + `_completed_keys`).
- **`validation/`** — `masks`, `metrics`, `filters`. All comparison, nothing persisted.
- **`visualization/`** — five modules that spell out the model, smallest thing first:
  - `panels` — one picture of one operation's decision about one occurrence-part. Shared drawing helpers
    (`overlay_mask`, `diff_panel`, `annotate`, `side_by_side`) and the colour conventions they all obey, so
    learning to read one panel is learning to read all of them. Also `save_panel` and `PanelFiles`.
  - `grids` — many panels as one image: `image_grid`, `comparison_grid`. Pure layout, no project, no I/O.
    Panels must arrive display-ready uint8 — it will not rescale a float array, since two probability maps
    with different ranges would stretch to look identical.
  - `figures` — whole-population charts: `line_chart`, `bar_chart`, `histogram`, `scatter`, `funnel`.
  - `pipeline` — every activity's diagnostics: `open_report`, `Report`/`NullReport` (a bounded grid,
    checkpoints, figures, a sidecar), `resolve_sample`, and `PanelFanout` for a multi-output run's shared steps.
  - `products` — panels materialized per occurrence-part: `render_segments`.

  A Segment's `panel_sink` is a `Report` under any activity, or `panels.PanelFiles` when you build a segment
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
  `event_id` — the worked example of a project choosing its own calibration scope), `bioencoder`
  (`embedding.py` — `BioEncoderModel` and the `embedding()` metric, torch imported only inside functions;
  `training.py` — `prepare_dataset()` and a deliberately unimplemented `train()`/`load()`),
  `gbif_darwincore_inat` (`archive`/`ingest` — the way iNaturalist observations enter a project, since the
  archive is what GBIF actually published and is archived byte-exact), and `smp_segmenter`
  (`segmentation.py` — a UNet++ segmenter over a swappable encoder, meeting the same predict()/identity()/
  visualize() contract as `segmentation.groundedsam`; `training.py` — `prepare_dataset()` plus a real, working
  `train()`, unlike `bioencoder`'s scaffold: binary mask segmentation doesn't carry metric-learning's
  dataset-dependent backbone/loss judgment calls, so implementing it doesn't risk a model that trains without
  complaint and performs badly the way guessing at those would). Extensions normalize INTO core, never around it.

### Conventions to follow when extending

- **Operations are lowercase factory functions returning a configured `Operation`.** `remove_appendages()`,
  `body_length()`, `segment(model)`. The implementation is a module-level `_name(segment, **params)`. Metric
  factories take `name=None` and `unit=...` so the same operation can appear twice under different names.
- **Transforms return `(segment, info)`; metrics return a value.** `info` is a diagnostics dict. Include a
  reliability flag where the operation can tell its own result is doubtful: `degenerate` means it returned the
  segment unchanged, `unreliable` that what it returned is suspect. `segments.iterate_segments` (and the run
  drivers) count those flags per run — they reach the run's `context_json` and the driver's summary, so a
  thousand-occurrence run says how many orientations it wasn't sure about rather than leaving it on whichever
  panels happened to be sampled.
- **Visualization is `segment.emit_panel(panel, "<stage>")`,** called unconditionally — it no-ops when
  `segment.panel_sink` is None, which is the case for every occurrence outside the sample. EMIT, not save: an
  operation draws what it decided and hands it over; where it goes is the run's business. The stage names the
  column. Panels must be display-ready uint8 — the operation knows what its own numbers mean, so it renders
  them.
- **Anything expensive that must happen once per run goes in `Operation.prepare(context)`,** not in
  `__init__`. That's how group metrics fit their reference population.
- **Individual failures are logged and counted, never fatal.** One bad occurrence must not cost a run.
- **`limit=` caps candidates; `max_new=` caps new work.** The two sit on either side of the already-done
  filter, and the difference matters on a project that is part-way through: `limit=10` against ten finished
  occurrences does nothing (which is what makes it a property of the population a group metric fits, recorded
  in `context_json`), while `max_new=10` does ten more every time it runs. Every driver's `limit` means the
  first; the three pending-first drivers — `download_images` and both `measure_scales` — take `max_new` for
  the second. `validate_masks` has only `limit`: it persists nothing, so it has no already-done concept for a
  `max_new` to sit after.
- **Every driver builds with `segments.build_segment` and returns `segments.Tally.summary`.** Most walk
  `segments.iterate_segments` (open the image store, build the Segment, optionally frame it by an upstream part,
  run the chain, count); `run_segments` and `run_metrics` keep their own loops for the reasons under
  `segments.py` above. One summary shape: `attempted, processed, skipped, no_input, failed, failures, flags`, plus whatever that driver
  alone has (`run_id`, `copied`, `previously_failed`, `missed`, `directory`). `no_input` is "nothing to work
  from" — no image, no mask, no `from_part` mask — which is neither a failure nor work done. A driver raises
  `segments.NoInput` for it, never writes it to the failures ledger, and logs one line per reason rather than
  one per occurrence, so it is attempted again once the input exists: a failure's retry key (recipe plus
  upstream mask) doesn't move when an image is finally downloaded, so recording it as one would skip that
  occurrence forever. `records.failures.NOT_FAILURES` ignores ledger rows written before this.
- **A segmenter's `mask_threshold` is a LOGIT.** `segment(mask_threshold=)` passes one number to whatever model
  runs, so the two bundled segmenters read it the same way: 0.0 is neutral (logit 0 is probability 0.5),
  negative grows the mask, positive shrinks it. A model thresholding its own sigmoid instead would read
  `segment()`'s neutral 0.0 as "keep every pixel above zero probability" — the whole frame.
- **Docstrings are reference, not essays.** They render as the API site, so keep them scannable:
  - A **module** docstring is one line — the module's line from README.md's package-layout tree — plus at most
    one short sentence a reader genuinely can't use the module without. No project-tree diagrams (the README
    has one), no design essays.
  - A **function** docstring is a one-line summary, an optional one- or two-sentence caveat, then
    `param -- what it is`, one line each, and a `Returns ...` line. No rationale inside the param block.
  - **The param block is a markdown bullet list, one `- `name`` per param**, e.g. `` - `project_path` --
    project to ingest into``. mkdocstrings renders a docstring's free text as raw Markdown — no docstring
    parser here recognizes this project's `--` separator as a parameters section (it looks for a colon), so
    without list markup, adjacent param lines with no blank line between them collapse into one paragraph.
    A wrapped param's continuation lines must be indented to line up with the bullet's own text (two spaces
    past the `- `), never visually aligned under the `--` a few columns further right — Markdown reads
    anything indented 4+ columns past where the list item's content starts as a nested code block, which is
    how a wrapped description ends up rendered in a copy-button panel instead of as prose.
  - Write like README.md: present tense, declarative, `&`/`e.g.`/`i.e.`. No rhetorical openers ("Worth
    knowing", "The reason is", "which is why"), no parenthetical asides longer than a clause, ALL-CAPS
    emphasis at most once per docstring.
  - **Why goes here, not there.** A design decision, a rejected alternative, or a silent failure mode a guard
    exists for belongs in the Architecture section above — one copy, findable — not restated in every module
    that touches it. The exception is a hazard someone editing *this specific code* would otherwise walk into:
    that stays as a short `#` comment at the line it guards (see `_wait_for_key`'s duplication note, and the
    pragma ordering in `storage/sqlite.py`).

### Things that are deliberately unfinished

- `extensions/bioencoder/training.py::train()` and `load()` raise `NotImplementedError`. The
  dataset preparation above them is real; the training loop is left out rather than guessed at, and the
  docstring says which decisions a caller has to make.
- `extensions/bioencoder/embedding.py`'s embeddings have no visualization. The picture worth
  having is a projection of every vector, which needs a run-end hook on `Operation` that doesn't exist yet.
- Landmarks aren't implemented, but have a settled shape. A landmark set is one dict-valued metric per
  occurrence-part, flattened to `{<name>_x, <name>_y}` so each coordinate exports, filters and compares like
  any number, and always in ORIGINAL image coordinates — mapped back through the segment's affine, the
  point-shaped counterpart of `mask_in_original_coordinates` that `Segment` doesn't have yet. Human clicks
  generalize `metrics.annotation.click_two_points`; a model gets a `landmarks(model)` factory mirroring
  `segment(model)`, its checkpoint reaching the hash through `identity()`; geometric landmarks and outline
  semilandmarks come from the mask like any metric. Procrustes alignment is a group metric: the consensus is
  fit in `prepare()` as in `metrics.outliers`, its population recorded in `context_json`. Also missing: a
  morphometrics export format (TPS, geomorph long) and a keypoint format in `training/datasets`.
- `sketch1.py` and `scripts/sketches.py` are the original design sketches this package was built from. They
  reference an older `critterframes`/`critterframes_inat` naming, and `scripts/sketches.py` isn't valid Python.
  They're kept as the design record; the realized versions are `scripts/dragonfly_bodies_inat.py` and
  `scripts/dragonfly_wings_museums.py`.
