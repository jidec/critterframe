"""
Masks: one canonical mask per occurrence-part, plus the segmentation metadata
that says how it was derived.

Stored in project_path/masks.parquet, RLE-encoded, one row per occurrence-part

Use Parquet rather than an image store because a mask is small once run-length encoded,
plus it's often wanted alongside its metadata (which recipe produced it, what the model scored it, when)
table.

For every row here:

  1. At most ONE current mask per occurrence-part. Writing a mask for an
     occurrence-part that already has one replaces it; masks are processing
     state, and the current one is the one that counts. Superseded masks aren't
     kept -- the run record and recipe hash say how any mask was made, and a
     recipe is reproducible, so the mask itself doesn't need a version history.
  2. Every mask is in the coordinates of the ORIGINAL analysis image, never a
     crop's or a rotation's. Recipes crop and rotate freely, but
     Segment.mask_in_original_coordinates() inverts all of it before anything
     reaches this module. That's what lets a wing mask found inside a rotated
     sub-crop and a whole-organism mask found on the full frame sit in the same
     table and be compared, overlaid, or measured against the same image.
  3. A mask cut out of another part's mask records WHICH mask that was
     (source_mask_hash, see derivation_hash). Because a replaced mask leaves no
     trace behind it, that record is the only thing that can tell anything
     downstream -- a derived part, a measurement -- that what it was built on
     has moved out from under it.

REFERENCE masks live in an identical table at reference_masks.parquet and are
addressed by passing reference=True. They're a parallel set, not a
replacement: validation is comparison between the canonical masks and the
reference ones, so both must be able to exist for the same occurrence-part at
once.

They are called reference rather than ground truth deliberately. A reference
mask is whatever you have chosen to compare against -- usually a human's
correction, but equally a slower model, an earlier version of the pipeline, or
a segmenter being benchmarked. Calling it truth would assert the answer to the
question validation exists to ask, and would overstate a human annotation on
exactly the hard boundaries where annotators disagree with each other.
"""

import logging
from datetime import datetime, timezone

import numpy as np
from pycocotools import mask as mask_utils

from ..project import paths
from ..recipes import DEFAULT_PART, hash_spec
from ..storage.tables import load_table, table_columns, upsert_table

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

    COCO RLE via pycocotools: compact for the large, contiguous regions
    organism masks actually are, and a format plenty of other tools already
    read.

    The counts blob is stored as RAW BYTES. Parquet has a binary column type,
    so there is nothing to make the bytes safe for -- base64 exists to smuggle
    binary through text formats, and encoding it here would cost about a third
    of the mask table's size (compression doesn't win it back) plus an
    encode/decode on every read and write, in exchange for nothing.
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

    bytes() around the counts because pycocotools wants exactly that, while a
    parquet read can hand back any bytes-like view of the column.
    """
    rle = {
        "counts": bytes(row["rle_counts"]),
        "size": [int(row["rle_height"]), int(row["rle_width"])],
    }
    return mask_utils.decode(rle).astype(bool)


def derivation_hash(recipe_hash, source_mask_hash=None):
    """
    The identity of a MASK, as opposed to the identity of the recipe that
    produced it: its recipe hash, chained with the identity of the upstream mask
    it started from.

    For a mask segmented straight from the image -- most masks -- there is no
    upstream, and this is exactly the segmentation recipe hash. It differs only
    for a part derived from another part (run_segments' from_part), and that
    case is the reason the function exists: resegmenting the organism changes
    the wing mask derived from it WITHOUT changing the wing recipe, so a
    downstream check keyed on the recipe hash alone would see nothing move and
    conclude the wing's masks and measurements were still good. Chaining makes
    an upstream change propagate through however many derivation steps sit on
    top of it.

    Anything non-string for source_mask_hash (None, or the NaN a table written
    before the column existed reads back as) means "no upstream", which is the
    same answer both cases deserve.
    """
    if not isinstance(source_mask_hash, str):
        return recipe_hash
    return hash_spec({"recipe": recipe_hash, "from": source_mask_hash})


def make_mask_row(occurrence_id, mask, part=DEFAULT_PART, recipe_hash=None,
                  run_id=None, score=None, from_part=None,
                  source_mask_hash=None):
    """
    Build one mask record.

    mask        -- boolean mask IN ORIGINAL ANALYSIS IMAGE COORDINATES. Callers
                   running a recipe get this from
                   Segment.mask_in_original_coordinates(); a caller assembling
                   a row by hand is responsible for the same thing.
    part        -- named part this mask covers; the whole organism by default.
    recipe_hash -- identity of the recipe that derived it, which is what makes
                   a rerun able to recognize its own previous work.
    run_id      -- the run that produced it (see records.runs).
    score       -- the model's own confidence, where it reports one. Kept
                   because it's the cheapest available signal for "which of
                   these masks should a human look at first", and dropping it
                   at write time means never being able to ask that later.
    from_part   -- the upstream part this was derived from, if any.
    source_mask_hash -- derivation_hash() of the upstream mask this was derived
                   from, if any. from_part says which part it came from and this
                   says WHICH MASK of that part, which is what makes a derived
                   mask recognizably stale once the part it came from is
                   resegmented. None for a mask found straight in the image.

    occurrence_id and part are the table's key, so this is where their type is
    guaranteed: both are coerced to str here and are strings everywhere
    afterward. Storage compares keys by value without coercing (see
    storage.tables.upsert_table), so a numeric id reaching it would be a
    genuinely different key from the same id as a string -- the invariant has
    to hold at the point rows are built, not be patched up at the point they're
    compared.
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
    import pandas as pd

    if not rows:
        return 0

    upsert_table(pd.DataFrame(rows, columns=COLUMNS),
                 paths.masks_path(project_path, reference=reference),
                 key_cols=KEY_COLS)
    return len(rows)


def load_masks(project_path, parts=None, occurrence_ids=None, recipe_hash=None,
               reference=False, columns=None):
    """
    Read mask rows as a DataFrame (still RLE-encoded -- use decode_mask() on a
    row to get pixels).

    parts          -- optional list of parts to include; all if None.
    occurrence_ids -- optional list of occurrence ids to include; all if None.
    recipe_hash    -- optional exact-recipe filter.
    reference      -- read the reference table instead of the canonical one.
    columns        -- optional list of columns to read off disk. Skip it when
                     filtering, since the filter columns have to be read too.
    """
    df = load_table(paths.masks_path(project_path, reference=reference),
                    columns=columns, missing_ok=True)
    if df.empty:
        return df

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

    Reads the whole table per call, which is fine for a one-off lookup and
    wasteful in a loop -- a run over many occurrences should call load_masks()
    once and decode rows itself (see mask_lookup()).
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
    loops over, so the mask table is read once rather than once per occurrence.

    Rows are returned still encoded; decode_mask() each one as you reach it, so
    a run over a large project doesn't hold every decoded mask in memory at
    once.
    """
    df = load_masks(project_path, parts=[part], occurrence_ids=occurrence_ids,
                    reference=reference)
    return {row["occurrence_id"]: row for _, row in df.iterrows()}


def _load_identities(project_path, reference=False, **filters):
    """
    Read the identity columns (see IDENTITY_COLUMNS) of the mask table.

    source_mask_hash is dropped from the read when the stored table predates it,
    because naming a column a parquet doesn't have fails the read outright. Rows
    from such a table then look like masks with no upstream -- which is what
    every mask in a project segmented before derivation chaining existed
    effectively was, as far as anything recorded at the time can say.
    """
    available = set(table_columns(paths.masks_path(project_path,
                                                   reference=reference)))
    columns = [column for column in IDENTITY_COLUMNS if column in available]
    return load_masks(project_path, reference=reference,
                      columns=columns or None, **filters)


def completed_keys(project_path, recipe_hash, reference=False,
                   source_mask_hashes=None):
    """
    The (occurrence_id, part) pairs a given segmentation recipe has already
    produced masks for -- the repeat-awareness check segmentation.run makes
    before doing any work.

    source_mask_hashes -- {(occurrence_id, part): derivation_hash} of the
                          UPSTREAM masks the run is about to start each part
                          from, for a from_part recipe. When given, a stored
                          mask only counts as complete if it was derived from
                          that exact upstream mask: a wing mask is stale the
                          moment the organism mask it was cut out of is
                          resegmented, even though the wing recipe hasn't
                          changed and its hash still matches. Pass None for a
                          recipe that segments straight from the image, where
                          the recipe hash is the whole story.

                          A mask whose upstream isn't in the map never counts as
                          complete -- there's nothing left to confirm it against.
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
    table -- the lookup that answers "is this still the mask that thing was
    derived from".

    A metric row records the derivation hash of the mask it was measured from
    (records.metrics' source_mask_hash) and so does a mask derived from another
    part's mask. Replacing a mask changes what's here and not what they recorded,
    so comparing the two is the only way to tell derived data that describes the
    current mask from data left over from one that has since been superseded.
    Nothing else can tell them apart: masks keep no version history, by design
    (see the module docstring), so a stale value's mask is simply gone.

    Returns an empty dict for a project with no mask table. Callers read that as
    "nothing to judge staleness against" rather than "everything is stale" --
    the two are indistinguishable from here, and only one of them is safe to
    assume.
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
