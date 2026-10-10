# Code review ledger

Working file for the batched review of `critterframe/`. Deleted when the review ends.

Per batch: read, report findings (A bugs, B doc/code mismatch, C clarity, D out of batch), triage, apply,
verify, commit. No change to a recipe hash, `derivation_hash`, a pinned digest or an on-disk format without
explicit sign-off.

## Baseline (2026-10-08, uncommitted working tree on top of 76fb940)

- `pytest -n auto`: 2192 passed, 1 skipped (torch DLL fails to load on this machine).
- `ruff check .`: clean. `ruff format --check .`: clean.
- `mkdocs build --strict`: **fails**, 3 griffe warnings at `critterframe/validation/filters.py:1092-1098`
  (docstring continuation indent). Pre-existing; fixed in batch 17 unless pulled forward.

## Batches

| # | Batch | Modules | Status |
|---|---|---|---|
| 1 | Leaves | `project/paths`, `selection/algorithms`, `core/drivers`, `maskops`, `colorspaces`, `timing`, `devices` | findings reported, awaiting triage |
| 2 | Storage | `storage/*`, `records/occurrences`, `records/failures` | |
| 3 | Core types | `core/recipes`, `core/segments` | |
| 4 | Run records | `records/masks`, `records/runs`, `records/metrics` | |
| 5 | Calibration and models | `records/calibrations`, `records/models`, `calibrations/scale` | |
| 6 | Project | `project/summarize`, `project/archive` | |
| 6b | Selection | `selection/subsets`, `selection/queries` | |
| 7 | Visualization | `panels`, `grids`, `figures`, `pipeline`, `products` | |
| 8 | In | `ingest/occurrences`, `ingest/images`, `ingest/archive`, `ingest/imports`, `ingest/download` | |
| 9 | Transforms and segmenters | `transforms/*`, `segmentation/groundedsam`, `segmentation/manual` | |
| 10 | Segmentation driver | `segmentation/run_segments`, `segmentation/mask_import_export` | |
| 11 | Metrics driver | `metrics/run_metrics`, `metrics/stored`, `metrics/derived` | |
| 12 | Per-occurrence metrics | `dimensions`, `islands`, `position`, `quality`, `pixels`, `color/means`, `mask_info`, `annotation`, `embedding` | |
| 13 | Color thresholds | `metrics/color/thresholds`, `metrics/color/inductive_thresholds`, `metrics/color/clusters` | |
| 14 | Group metrics | `outliers`, `label_score` | |
| 15 | Out | `wide`, `export` | |
| 16 | Validation I | `validation/masks`, `validation/metrics` | |
| 17 | Validation II | `validation/filters`, `validation/filter_grids` | |
| 18 | Training and model extensions | `training/*`, `extensions/smp_segmenter`, `extensions/bioencoder` | |
| 19 | Source extensions | `extensions/antenna_lighttraps`, `extensions/gbif_darwincore_inat` | |
| 20 | Closing sweep | `critterframe/__init__.py`, this file's deferred list, README / CLAUDE.md against the code, `docs/api`, `scripts/` call sites | |

## Selection restructure (done 2026-10-08, uncommitted)

- `selection.algorithms.py` -> `selection/algorithms.py`; `project/subsets.py` -> `selection/subsets.py`; the three
  selection functions in `export.py` -> `selection/queries.py`; the wide view and filter evaluation -> `wide.py`.
- Naming rule: "occurrences" is rows, "ids" is ids; a project query is `ids_<condition>`. Renamed
  `occurrences_matching` -> `ids_matching`, `completed_ids` -> `ids_completed`, `occurrence_ids_with_mask` ->
  `ids_with_mask` (now a sorted list, in `selection.queries`), `sample_occurrences` -> `sample_ids`,
  `shard_occurrences` -> `shard_ids`. New on `cf.`: `select_ids`, `ids_with_image`, `ids_passing`.
- `wide.filtered_wide` is the frame `export_metrics` and `ids_passing` both start from.
- After it: 2208 passed, 1 skipped; ruff clean; `mkdocs build --strict` fails only on the baseline warnings.

Left for later batches:

- `validation.filters.audit_filters` still builds its frame through `export_metrics`; decide in batch 17
  whether it reads `wide.filtered_wide` instead. `validation/filters.py` also imports `export._recorded_filters`.
- The archived gate scripts (`scripts/author_pipelines/odonata_inat_obsorg/archive/`) still call
  `export_metrics(path=False, manifest=False, filters=gate)`, which keeps working.
- `scripts/` files that only needed a module path changed still deep-import
  (`from critterframe.selection.subsets import select_ids`); the `cf.` name is available.

## Folder restructure: `core/`, `metrics/color/`, `ingest/` (done 2026-10-08, uncommitted)

- `recipes.py`, `segments.py`, `drivers.py` -> `core/`.
- `metrics/color_means.py`, `color_thresholds.py`, `inductive_color_thresholds.py`, `color_clusters.py` ->
  `metrics/color/{means,thresholds,inductive_thresholds,clusters}.py`.
- `ingest.py` split into `ingest/{occurrences,images,archive}.py`; `download.py` -> `ingest/download.py`.
- **A recipe hash moved, by decision.** A derived metric records its function's import path in its spec, so
  `color_presence` now records `critterframe.metrics.color.thresholds._color_presence`. In an existing
  project, a metric run containing `color_presence` no longer matches the recipe its `run_name` is pinned
  to: the next `run_metrics` under that name raises until given `force=True`, and then recomputes every
  metric in that run (`threshold_fractions` included). Runs without `color_presence` are untouched.
- After it: 2209 passed, 1 skipped; ruff clean.
- Follow-up: `ingest/archive.py` split into `archive.py` (raw bytes) and `imports.py` (the import record).
  `already_ingested` and `archive_source_file` removed; every repeat check goes through
  `imports.import_identity` and `imports.is_recorded`. Fixed: re-ingesting an unchanged image folder
  archived a second copy of its manifest each time.
- Follow-up: `core.segments.run_chain` and `framed_segment` replace four hand-written chain loops and the
  duplicated `from_part` framing in `iterate_segments` and `run_metrics`. `run_segments` builds its
  failure context hashes once. No existing test needed editing.
- Left for batch 16: `validation/masks.py` has two bare chain loops that keep no info and count no flags.

## Docs split (done 2026-10-08, uncommitted)

- `README.md` is the pitch (about 120 lines). Its other sections moved to `docs/concepts.md`,
  `docs/guide.md` and `docs/internals.md`; the site's Home tab is now Overview.
- The example-pipelines table named six scripts that no longer exist; it is rewritten against
  `scripts/reusable_workflows/` and `scripts/author_pipelines/`.
- Removed from CLAUDE.md: the "deliberately unfinished" bullet about `sketch1.py` and `scripts/sketches.py`,
  since neither file, nor the two it called their realized versions, exists.
- Batch 20's "README against the code" step now covers the three docs pages too.
- The Examples tab is generated from the scripts by `tools/docs_examples.py`; it replaced the Guide's
  hand-written tables of reusable workflows and odonata steps.

## Mask read fix (done 2026-10-09, uncommitted)

- `run_metrics` on `odonata_inat_obsorg` failed with `ArrowMemoryError`: `mask_lookup` read the whole
  1.9-million-row mask table to keep one part. It now filters by part and reads `LOOKUP_COLUMNS` only
  (about 5 s and a 2.4 to 2.6 GB peak per part on that project). `mask_info()` raises on a row read
  without `info`.
- For batch 2 (storage): that table has two row groups of about a million rows, so no read can skip one.
  Smaller row groups, or rows sorted by part, would lower the remaining peak.
- For batch 4 (run records): an unsharded `save_masks` reads and rewrites the whole table on every flush,
  1 GB at this scale. It is the documented design, with `shard=` as the way around it, but worth a look.

## Batch 1 findings

Status column: open / accepted / rejected / done.

| ID | Kind | Where | Finding | Status |
|---|---|---|---|---|
| 1-A1 | bug | `selection.algorithms._fingerprint` | Rounding gives `-0.0` and `0.0` as different fingerprints, so two records either side of the equator or prime meridian never match. | open |
| 1-A2 | bug | `selection.algorithms.cap_per_group` | Works by index label; a table with repeated labels (a `pd.concat` without `ignore_index`) loses rows from groups that were under the cap. | open |
| 1-A3 | bug | `selection.algorithms.rows_matching` | An ndarray, Series or dict-keys of values is treated as one value and matches nothing, silently. | open |
| 1-A4 | bug | `core.drivers.log_no_input` | Any reason without a hint gets "segment that part first", including `metrics.stored`'s "no current '<run>' value". | open |
| 1-B1 | mismatch | `selection.algorithms.cap_per_group` / `_take` | `"random"` is documented as independent of row order; it samples index labels, so reordering the source picks other rows. Fixing it changes which rows a future capped ingest keeps. | open |
| 1-B2 | mismatch | `selection.algorithms.sample_per_group` | Documented as stable; groups of equal size are filled in order of first appearance, so row order changes the sample. | open |
| 1-B3 | mismatch | `selection.algorithms.cap_per_group` | A callable `rule` bypasses `keep_ids` and `prefer` without saying so. | open |
| 1-B4 | mismatch | CLAUDE.md | `calibration/` should be `calibrations/`. (The test count and the strict docs build were fixed during the docs work.) | open |
| 1-B5 | mismatch | `colorspaces.SPACES` comment | Lab a/b and LCh chroma ranges are described as the sRGB gamut's; they are the 8-bit Lab encoding's (chroma 181 is `hypot(128, 128)`). Comment only; the values feed `normalize`. | open |
| 1-C1 | clarity | `maskops.edge_distance`, `inscribed_radius` | Cast with `astype(uint8)` where the rest of the module uses `> 0`, so a fractional mask reads as empty. | open |
| 1-C2 | clarity | `paths.import_sidecar_path` | Hardcodes `.import.json` beside the `IMPORT_SIDECAR_SUFFIX` constant. | open |
| 1-C3 | clarity | `paths.recipes_path`, `RECIPES_FILE` | Used by nothing but its own test. | open |
| 1-C4 | clarity | `paths.raw_import_path` | `extension = extension or ".csv"` repeats the default. | open |
| 1-C5 | clarity | `selection.algorithms` | Layout: `_take` sits after its callers, `MEDOID_BLOCK_ROWS` mid-file; group into sampling, ranking, table narrowing, presence. | open |
| 1-C6 | clarity | `selection.algorithms.rows_matching` | The `isinstance(values, (str, bytes)) or` clause is redundant (folds into 1-A3's fix). | open |
| 1-C7 | clarity | `colorspaces` | Five private helpers have no docstring; `convert`/`to_bgr`/`normalize` say `space` is a key but a `ColorSpace` is accepted; error message spells "colour". | open |
| 1-C8 | clarity | `devices` | Unused `logging` import and `logger`. | open |
| 1-C9 | clarity | `core.drivers.Progress.finish` | Docstring says it logs "if anything was stepped through"; the test is `total`, not `done`. | open |
| 1-C10 | clarity | `selection.algorithms.sample_ids` | `count=None` (return everything) is undocumented; repeated ids are sampled as separate items. | open |

## Deferred / cross-cutting

- "colour" appears 14 times in package source (`color_thresholds` 9, `panels` 2, `colorspaces`, `color_clusters`,
  `datasets`) against the American-spelling rule. Sweep in batch 20, or per module as each is reached.
- `paths.pipeline_stem` documents `name` as "a run name or `validate_masks__head`"; CLAUDE.md says a report is
  named for the function that opened it. Check against `visualization/pipeline` in batch 7.
- `paths.relative_to_project` resolves a relative `target` against the working directory, not the project.
  Check every caller passes an absolute path (batches 4, 5, 8, 15).
- `validation/filters.py:1092-1098` docstring indentation breaks `mkdocs build --strict` (batch 17).
