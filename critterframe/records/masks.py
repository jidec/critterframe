"""
RLE encode/decode, upsert, derivation hashing, sharded writes for parallel runs.

One canonical mask per occurrence-part in masks.parquet, RLE-encoded, always in
the coordinates of the original analysis image. Writing a mask for a part that
already has one replaces it; masks are processing state, and a recipe hash says
how any of them was made.

Reference masks live in an identical table (reference_masks.parquet, via
reference=True) and are a parallel set, not a replacement -- validation compares
the two. Called reference rather than ground truth deliberately: it is whatever
you chose to compare against, which is often a human's correction but equally a
slower model or an earlier pipeline.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from ..project import paths
from ..recipes import DEFAULT_PART, hash_spec
from ..storage.tables import load_table, table_columns, upsert_table, write_table

logger = logging.getLogger(__name__)

KEY_COLS = ["occurrence_id", "part"]

COLUMNS = [
    "occurrence_id",
    "part",
    "rle_counts",
    "rle_height",
    "rle_width",
    "area",
    "score",
    "recipe_hash",
    "run_id",
    "from_part",
    "source_mask_hash",
    "created_at",
]

# Columns that identify a mask without its pixels -- what the staleness checks
# read. Kept narrow on purpose: rle_counts is most of the table's bytes, and
# neither "which masks does this recipe already cover" nor "is this value's
# source mask still current" needs a single one of them.
IDENTITY_COLUMNS = ["occurrence_id", "part", "recipe_hash", "source_mask_hash"]


def encode_mask(mask):
    """
    RLE-encode a boolean mask into the columns a mask row stores.

    COCO RLE via pycocotools: compact for the contiguous regions organism masks
    are, and widely readable. Counts are stored as raw bytes, since parquet has
    a binary column type (see CLAUDE.md).
    """
    mask = np.asfortranarray((np.asarray(mask) > 0).astype(np.uint8))
    rle = mask_utils.encode(mask)
    height, width = rle["size"]
    return {
        "rle_counts": rle["counts"],
        "rle_height": int(height),
        "rle_width": int(width),
        "area": int(mask.sum()),
    }


def decode_mask(row):
    """
    Decode one mask row back into a boolean array.

    row -- a mapping with rle_counts/rle_height/rle_width, e.g. a row from
           load_masks() or a record from make_mask_row().
    """
    rle = {
        "counts": bytes(row["rle_counts"]),
        "size": [int(row["rle_height"]), int(row["rle_width"])],
    }
    return mask_utils.decode(rle).astype(bool)


def derivation_hash(recipe_hash, source_mask_hash=None):
    """
    A mask's identity, as opposed to its recipe's: the recipe hash, chained
    with the identity of the upstream mask it started from.

    With no upstream this is just the recipe hash. Chaining matters for a part
    cut from another part, where resegmenting the organism must make the wing
    stale without the wing's own recipe changing.

    A non-string source_mask_hash -- None, or the NaN an older table reads back
    as -- means no upstream.
    """
    if not isinstance(source_mask_hash, str):
        return recipe_hash
    return hash_spec({"recipe": recipe_hash, "from": source_mask_hash})


def make_mask_row(occurrence_id, mask, part=DEFAULT_PART, recipe_hash=None,
                  run_id=None, score=None, from_part=None,
                  source_mask_hash=None):
    """
    Build one mask record.

    occurrence_id and part are the table's key, so this is where their type is
    guaranteed -- both are coerced to str and are strings everywhere after.
    Storage compares keys by value without coercing (see CLAUDE.md).

    mask        -- boolean mask in original analysis image coordinates, e.g.
                   from Segment.mask_in_original_coordinates().
    part        -- named part this mask covers; the whole organism by default.
    recipe_hash -- identity of the recipe that derived it, which is what lets a
                   rerun recognize its own previous work.
    run_id      -- the run that produced it.
    score       -- the model's own confidence, where it reports one.
    from_part   -- the upstream part this was derived from, if any.
    source_mask_hash -- derivation_hash() of the upstream MASK, if any, which is
                   what makes a derived mask stale once its source is
                   resegmented. None for a mask found straight in the image.
    """
    if occurrence_id is None or part is None:
        raise ValueError(
            f"a mask needs both an occurrence_id and a part "
            f"(got {occurrence_id!r}, {part!r})"
        )

    row = {
        "occurrence_id": str(occurrence_id),
        "part": str(part),
        "score": None if score is None else float(score),
        "recipe_hash": recipe_hash,
        "run_id": run_id,
        "from_part": from_part,
        "source_mask_hash": source_mask_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    row.update(encode_mask(mask))
    return {column: row.get(column) for column in COLUMNS}


def save_masks(project_path, rows, reference=False):
    """
    Write mask rows, replacing any existing mask for the same occurrence-part.

    rows      -- list of records from make_mask_row().
    reference -- True writes to the reference table instead of the canonical
                 one.
    """
    if not rows:
        return 0

    upsert_table(pd.DataFrame(rows, columns=COLUMNS),
                 paths.masks_path(project_path, reference=reference),
                 key_cols=KEY_COLS)
    return len(rows)


def save_mask_shard(project_path, rows, part, reference=False):
    """
    Write one batch of mask rows to a new, uniquely-named staging file rather
    than upserting into the canonical table.

    What a sharded run flushes through instead of save_masks(): a new file
    can't collide with any number of concurrent writers (see CLAUDE.md).
    merge_mask_shards() reads them back.

    rows      -- as save_masks().
    part      -- which output part these rows belong to; shards stage per part.
    reference -- as save_masks().
    """
    if not rows:
        return 0

    # A fresh name every call, sorting lexically in write order -- see
    # paths.mask_shard_path, which merge_mask_shards() below depends on.
    dest = paths.mask_shard_path(project_path, part, reference=reference)
    dest.parent.mkdir(parents=True, exist_ok=True)

    write_table(pd.DataFrame(rows, columns=COLUMNS), dest)
    return len(rows)


def merge_mask_shards(project_path, part=None, reference=False, cleanup=True):
    """
    Fold every staged shard file into the canonical mask table.

    Run once, single-process, after every shard has finished -- a cluster job
    depending on the array is the natural fit. The one place that performs the
    upsert, safe because only one process ever does. An occurrence-part staged
    twice resolves to the newest write.

    part      -- merge only this part's shards, or None for every staged part.
    reference -- merge the reference-mask shards instead.
    cleanup   -- delete each staged file once merged. False leaves them, e.g.
                 to inspect before trusting the merge.

    Returns {part: n_rows_merged}, for parts that had anything staged.
    """
    root = paths.mask_shards_dir(project_path, reference=reference)
    if not root.exists():
        return {}

    parts = [part] if part is not None else sorted(
        entry.name for entry in root.iterdir() if entry.is_dir())

    merged = {}
    for output_part in parts:
        directory = root / output_part
        shard_files = sorted(directory.glob("*.parquet")) if directory.exists() else []
        if not shard_files:
            continue

        # Filenames sort in write order (see save_mask_shard), so the last
        # occurrence of a key after concatenating in that order is the
        # chronologically newest write for it.
        combined = pd.concat([pd.read_parquet(path) for path in shard_files],
                             ignore_index=True)
        combined = combined.drop_duplicates(subset=KEY_COLS, keep="last")

        upsert_table(combined, paths.masks_path(project_path, reference=reference),
                     key_cols=KEY_COLS)
        merged[output_part] = len(combined)

        if cleanup:
            for path in shard_files:
                path.unlink()

    return merged


def load_masks(project_path, parts=None, occurrence_ids=None, recipe_hash=None,
               reference=False, columns=None):
    """
    Read mask rows as a DataFrame, still RLE-encoded -- decode_mask() a row to
    get pixels.

    parts          -- parts to include; all if None.
    occurrence_ids -- occurrence ids to include; all if None.
    recipe_hash    -- exact-recipe filter.
    reference      -- read the reference table instead of the canonical one.
    columns        -- columns to read off disk. Skip when filtering, since the
                      filter columns have to be read too.
    """
    df = load_table(paths.masks_path(project_path, reference=reference),
                    columns=columns, missing_ok=True)
    if df.empty:
        return df

    # Guarded, because `columns` may deliberately exclude the id -- parts_present
    # reads only the part column, and the identity read drops whatever the
    # stored table predates. Coercing unconditionally made asking a narrow
    # question fail on exactly the projects that had something to answer with.
    if "occurrence_id" in df.columns:
        df["occurrence_id"] = df["occurrence_id"].astype(str)
    if parts is not None:
        df = df[df["part"].isin(parts)]
    if occurrence_ids is not None:
        df = df[df["occurrence_id"].isin({str(i) for i in occurrence_ids})]
    if recipe_hash is not None:
        df = df[df["recipe_hash"] == recipe_hash]

    return df.reset_index(drop=True)


def get_mask(project_path, occurrence_id, part=DEFAULT_PART, reference=False):
    """
    One decoded mask, or None if this occurrence-part hasn't been segmented.

    Reads the whole table per call: fine for a one-off lookup, wasteful in a
    loop. Use mask_lookup() for a run.
    """
    df = load_masks(project_path, parts=[part], occurrence_ids=[occurrence_id],
                    reference=reference)
    if df.empty:
        return None
    return decode_mask(df.iloc[0])


def mask_lookup(project_path, part=DEFAULT_PART, occurrence_ids=None,
                reference=False):
    """
    {occurrence_id: mask row} for one part, read in a single pass -- what a run
    loops over, so the table is read once rather than once per occurrence.

    Rows come back encoded; decode_mask() each as you reach it, so a large run
    doesn't hold every decoded mask in memory.
    """
    df = load_masks(project_path, parts=[part], occurrence_ids=occurrence_ids,
                    reference=reference)
    return {row["occurrence_id"]: row for _, row in df.iterrows()}


def _load_identities(project_path, reference=False, **filters):
    """
    Read the identity columns (see IDENTITY_COLUMNS) of the mask table.

    source_mask_hash is dropped when the stored table predates it, since naming
    a missing column fails the read. Those rows then look upstream-less, which
    is what they effectively are.
    """
    available = set(table_columns(paths.masks_path(project_path,
                                                   reference=reference)))
    columns = [column for column in IDENTITY_COLUMNS if column in available]
    return load_masks(project_path, reference=reference,
                      columns=columns or None, **filters)


def completed_keys(project_path, recipe_hash, reference=False,
                   source_mask_hashes=None):
    """
    The (occurrence_id, part) pairs a segmentation recipe has already produced
    masks for -- the repeat-awareness check a run makes before doing any work.

    source_mask_hashes -- {(occurrence_id, part): derivation_hash} of the
                          upstream masks a from_part run is about to start
                          from. When given, a stored mask counts as complete
                          only if derived from that exact upstream, so a wing
                          goes stale when its organism is resegmented. None for
                          a recipe that segments straight from the image. A
                          mask whose upstream isn't in the map never counts.
    """
    df = _load_identities(project_path, reference=reference,
                          recipe_hash=recipe_hash)
    if df.empty:
        return set()

    keys = set(zip(df["occurrence_id"], df["part"]))
    if source_mask_hashes is None:
        return keys

    stored = {} if "source_mask_hash" not in df.columns else {
        (row.occurrence_id, row.part): row.source_mask_hash
        for row in df.itertuples(index=False)
    }
    return {
        key for key in keys
        if stored.get(key) is not None
        and stored.get(key) == source_mask_hashes.get(key)
    }


def current_derivation_hashes(project_path, parts=None, occurrence_ids=None,
                              reference=False):
    """
    {(occurrence_id, part): derivation_hash} for the masks currently in the
    table -- the lookup answering "is this still the mask that was derived
    from".

    Metric rows and derived masks record the derivation hash they were built
    from. Replacing a mask changes this and not what they recorded, so
    comparing the two is the only way to spot data left over from a superseded
    mask.

    Empty dict for a project with no mask table, read as "nothing to judge
    against" rather than "everything is stale" -- the two are
    indistinguishable here and only one is safe to assume.
    """
    df = _load_identities(project_path, reference=reference, parts=parts,
                          occurrence_ids=occurrence_ids)
    if df.empty:
        return {}
    return {
        (row.occurrence_id, row.part): derivation_hash(
            row.recipe_hash, getattr(row, "source_mask_hash", None))
        for row in df.itertuples(index=False)
    }


def parts_present(project_path, reference=False):
    """Every part name that has at least one mask -- what a project actually holds."""
    df = load_masks(project_path, reference=reference, columns=["part"])
    return sorted(df["part"].unique()) if not df.empty else []


def has_masks(project_path, reference=False):
    """True if the project has a mask table at all."""
    return paths.masks_path(project_path, reference=reference).exists()
