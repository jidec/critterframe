"""Mask records: RLE encode and decode, upsert, derivation hashing, sharded writes for parallel runs."""

import hashlib
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from ..project import paths
from ..core.recipes import DEFAULT_PART, hash_spec
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
    "info",
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

# What a lookup row is used for: the mask itself and its identity. `info` and
# `created_at` are left out, since together they are half of a large table and
# only the mask_info metric reads one of them.
LOOKUP_COLUMNS = [
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
]


def _encode_mask(mask):
    """RLE-encode a boolean mask (COCO RLE) into the columns a mask row stores."""
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
    """Decode one mask row into a boolean array.

    Args:
        row: A mapping with `rle_counts`, `rle_height` and `rle_width`.
    """
    rle = {
        "counts": bytes(row["rle_counts"]),
        "size": [int(row["rle_height"]), int(row["rle_width"])],
    }
    return mask_utils.decode(rle).astype(bool)


def mask_digest(mask):
    """Return a short, stable digest of a mask's pixels and shape.

    Args:
        mask: A boolean or 0/nonzero array.
    """
    mask = np.asarray(mask) > 0
    content = np.packbits(mask).tobytes() + repr(mask.shape).encode("ascii")
    return hashlib.sha256(content).hexdigest()[:16]


def mask_info(row):
    """Return the `{operation label: scalar info}` a mask row recorded, or `{}`.

    Args:
        row: A mask row read with its `info`.

    Raises:
        KeyError: If the row was read without `info`, e.g. by `mask_lookup` without `info=True`.
    """
    # A row with no info key was read without the column, which is not the same
    # as a mask that recorded none: answering {} there would hide the difference.
    if "info" not in row:
        raise KeyError("this mask row was read without its info -- pass info=True to mask_lookup")
    value = row["info"]
    return json.loads(value) if isinstance(value, str) else {}


def derivation_hash(recipe_hash, source_mask_hash=None):
    """Return a mask's identity: its recipe hash, chained with its upstream mask's identity.

    Args:
        recipe_hash: Hash of the recipe that made the mask.
        source_mask_hash: Derivation hash of the upstream mask. Anything but a string means
            there is none, and the result is the recipe hash.
    """
    if not isinstance(source_mask_hash, str):
        return recipe_hash
    return hash_spec({"recipe": recipe_hash, "from": source_mask_hash})


def combined_source_hash(hashes):
    """Return one `source_mask_hash` for a mask built from one or several upstream masks.

    Args:
        hashes: `{part: derivation hash}` of each upstream mask.

    Returns:
        A single upstream's own hash, unchanged, or None if it has none. Several are
        hashed together by part name.
    """
    if len(hashes) == 1:
        return next(iter(hashes.values()))
    # A non-string is an upstream with no recorded identity (None, or the NaN
    # an older table reads back as), which canonical JSON can't carry as NaN.
    return hash_spec({str(part): value if isinstance(value, str) else None for part, value in hashes.items()})


def make_mask_row(
    occurrence_id,
    mask,
    part=DEFAULT_PART,
    recipe_hash=None,
    run_id=None,
    score=None,
    from_part=None,
    source_mask_hash=None,
    info=None,
):
    """Build one mask record, with `occurrence_id` and `part` as strings.

    Args:
        occurrence_id: The occurrence.
        mask: Boolean mask in original image coordinates.
        part: Part the mask covers.
        recipe_hash: Hash of the recipe that derived it.
        run_id: The run that produced it.
        score: The model's own confidence, where it reports one.
        from_part: The upstream part it was derived from.
        source_mask_hash: Derivation hash of the upstream mask; None for a mask found in
            the image itself.
        info: `{operation label: scalar info}` from the steps that produced it. Not hashed.
    """
    if occurrence_id is None or part is None:
        raise ValueError(f"a mask needs both an occurrence_id and a part (got {occurrence_id!r}, {part!r})")

    row = {
        "occurrence_id": str(occurrence_id),
        "part": str(part),
        "score": None if score is None else float(score),
        "info": None if info is None else json.dumps(info, sort_keys=True),
        "recipe_hash": recipe_hash,
        "run_id": run_id,
        "from_part": from_part,
        "source_mask_hash": source_mask_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    row.update(_encode_mask(mask))
    return {column: row.get(column) for column in COLUMNS}


def save_masks(project_path, rows, reference=False):
    """Write mask rows, replacing any existing mask for the same occurrence-part.

    Args:
        project_path: Project to write to.
        rows: Records from `make_mask_row`.
        reference: Write to the reference table instead of the canonical one.
    """
    if not rows:
        return 0

    upsert_table(
        pd.DataFrame(rows, columns=COLUMNS),
        paths.masks_path(project_path, reference=reference),
        key_cols=KEY_COLS,
    )
    return len(rows)


def save_mask_shard(project_path, rows, part, reference=False):
    """Write a batch of mask rows to a new staging file, for `merge_mask_shards` to fold in.

    Args:
        project_path: Project to write to.
        rows: Records from `make_mask_row`.
        part: Part the rows belong to.
        reference: Stage for the reference table.
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
    """Fold every staged shard file into the mask table.

    Run from one process, after every shard has finished. An occurrence-part staged twice
    resolves to the newest write.

    Args:
        project_path: Project to merge in.
        part: Part whose shards to merge; None for every staged part.
        reference: Merge the reference-mask shards.
        cleanup: Delete each staged file once merged.

    Returns:
        `{part: rows merged}` for the parts that had anything staged.
    """
    root = paths.mask_shards_dir(project_path, reference=reference)
    if not root.exists():
        return {}

    parts = [part] if part is not None else sorted(entry.name for entry in root.iterdir() if entry.is_dir())

    merged = {}
    for output_part in parts:
        directory = root / output_part
        shard_files = sorted(directory.glob("*.parquet")) if directory.exists() else []
        if not shard_files:
            continue

        # Filenames sort in write order (see save_mask_shard), so the last
        # occurrence of a key after concatenating in that order is the
        # chronologically newest write for it.
        combined = pd.concat([pd.read_parquet(path) for path in shard_files], ignore_index=True)
        combined = combined.drop_duplicates(subset=KEY_COLS, keep="last")

        upsert_table(combined, paths.masks_path(project_path, reference=reference), key_cols=KEY_COLS)
        merged[output_part] = len(combined)

        if cleanup:
            for path in shard_files:
                path.unlink()

    return merged


def load_masks(
    project_path, parts=None, occurrence_ids=None, recipe_hash=None, reference=False, columns=None
):
    """Read mask rows as a DataFrame, still RLE-encoded.

    Args:
        project_path: Project to read from.
        parts: Parts to include; all if None.
        occurrence_ids: Occurrences to include; all if None.
        recipe_hash: Keep only masks made by this recipe.
        reference: Read the reference table.
        columns: Columns to read. Must include `occurrence_id` when `occurrence_ids` is given.
    """
    # Applied while the file is scanned, so the other parts' rows are never loaded.
    filters = []
    if parts is not None:
        filters.append(("part", "in", [str(part) for part in parts]))
    if recipe_hash is not None:
        filters.append(("recipe_hash", "==", recipe_hash))

    df = load_table(
        paths.masks_path(project_path, reference=reference), columns=columns, missing_ok=True, filters=filters
    )
    if df.empty:
        return df

    # Guarded, because `columns` may deliberately exclude the id -- parts_present
    # reads only the part column, and the identity read drops whatever the
    # stored table predates. Coercing unconditionally made asking a narrow
    # question fail on exactly the projects that had something to answer with.
    if "occurrence_id" in df.columns:
        df["occurrence_id"] = df["occurrence_id"].astype(str)
    # In pandas, not in the scan: a project-sized id list is the wrong thing to
    # hand a parquet filter, and the part filter has already shrunk the frame.
    if occurrence_ids is not None:
        df = df[df["occurrence_id"].isin({str(i) for i in occurrence_ids})]

    return df.reset_index(drop=True)


def get_mask(project_path, occurrence_id, part=DEFAULT_PART, reference=False):
    """Return one decoded mask, or None if the occurrence-part has none.

    Reads the whole table on each call; use `mask_lookup` in a loop.

    Args:
        project_path: Project to read from.
        occurrence_id: The occurrence.
        part: The part.
        reference: Read the reference table.
    """
    df = load_masks(project_path, parts=[part], occurrence_ids=[occurrence_id], reference=reference)
    if df.empty:
        return None
    return decode_mask(df.iloc[0])


def mask_lookup(project_path, part=DEFAULT_PART, occurrence_ids=None, reference=False, info=False):
    """Return `{occurrence_id: mask row}` for one part, read in a single pass.

    Only that part's rows and `LOOKUP_COLUMNS` are read.

    Args:
        project_path: Project to read from.
        part: The part.
        occurrence_ids: Occurrences to include; all if None.
        reference: Read the reference table.
        info: Also read each mask's recorded `info`.

    Returns:
        `{occurrence_id: row}`, each row a dict.
    """
    wanted = LOOKUP_COLUMNS + (["info"] if info else [])
    available = set(table_columns(paths.masks_path(project_path, reference=reference)))
    columns = [column for column in wanted if column in available]
    df = load_masks(
        project_path,
        parts=[part],
        occurrence_ids=occurrence_ids,
        reference=reference,
        columns=columns or None,
    )
    if df.empty:
        return {}
    if info and "info" not in df.columns:
        # A table written before info was recorded: every mask in it has none.
        df["info"] = None
    return dict(zip(df["occurrence_id"], df.to_dict("records")))


def _load_identities(project_path, reference=False, **filters):
    """Read the mask table's identity columns, tolerating a table that predates `source_mask_hash`."""
    available = set(table_columns(paths.masks_path(project_path, reference=reference)))
    columns = [column for column in IDENTITY_COLUMNS if column in available]
    return load_masks(project_path, reference=reference, columns=columns or None, **filters)


def completed_keys(project_path, recipe_hash, reference=False, source_mask_hashes=None):
    """Return the `(occurrence_id, part)` pairs a segmentation recipe already has masks for.

    Args:
        project_path: Project to read from.
        recipe_hash: The recipe.
        reference: Read the reference table.
        source_mask_hashes: `{(occurrence_id, part): derivation hash}` of the upstream masks
            a `from_part` run starts from. A stored mask then counts only if it was derived
            from that exact upstream.
    """
    df = _load_identities(project_path, reference=reference, recipe_hash=recipe_hash)
    if df.empty:
        return set()

    keys = set(zip(df["occurrence_id"], df["part"]))
    if source_mask_hashes is None:
        return keys

    stored = (
        {}
        if "source_mask_hash" not in df.columns
        else {(row.occurrence_id, row.part): row.source_mask_hash for row in df.itertuples(index=False)}
    )
    return {
        key for key in keys if stored.get(key) is not None and stored.get(key) == source_mask_hashes.get(key)
    }


def current_derivation_hashes(project_path, parts=None, occurrence_ids=None, reference=False):
    """Return `{(occurrence_id, part): derivation hash}` for the masks now in the table.

    Args:
        project_path: Project to read from.
        parts: Parts to include; all if None.
        occurrence_ids: Occurrences to include; all if None.
        reference: Read the reference table.

    Returns:
        The mapping; empty for a project with no mask table.
    """
    df = _load_identities(project_path, reference=reference, parts=parts, occurrence_ids=occurrence_ids)
    if df.empty:
        return {}
    return {
        (row.occurrence_id, row.part): derivation_hash(
            row.recipe_hash, getattr(row, "source_mask_hash", None)
        )
        for row in df.itertuples(index=False)
    }


def parts_present(project_path, reference=False):
    """Return every part name that has at least one mask."""
    df = load_masks(project_path, reference=reference, columns=["part"])
    return sorted(df["part"].unique()) if not df.empty else []


def has_masks(project_path, reference=False):
    """Return whether the project has a mask table."""
    return paths.masks_path(project_path, reference=reference).exists()
