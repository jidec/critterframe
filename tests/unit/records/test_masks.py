"""
The mask table: RLE round-trips, mask identity, and what counts as work already
done.

Two invariants live here and both fail silently when they break. A mask that
round-trips imperfectly is a mask that measures slightly wrong forever. And a
derivation hash that stops moving when its upstream is resegmented is a project
that serves numbers measured off masks it no longer holds -- with no error, no
warning, and nothing in the table to notice.
"""

import numpy as np
import pandas as pd
import pytest

from critterframe.project import paths
from critterframe.records import masks as mask_records
from critterframe.storage.tables import write_table
from helpers.compare import is_iso_utc
from helpers.synthetic import blob_mask


def make_row(occurrence_id="a", mask=None, part="organism", **kwargs):
    """A mask row with a drawn mask, for tests that don't care about the pixels."""
    if mask is None:
        mask = blob_mask()
    return mask_records.make_mask_row(occurrence_id, mask, part=part, **kwargs)


# ---------------------------------------------------------------------------
# RLE round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (40, 40), (17, 129), (200, 300)])
def test_round_trip_preserves_every_pixel(shape):
    """
    Lossless in both directions, on shapes chosen to be awkward: 1x1, prime
    dimensions, and a non-square frame where a transposed encode would still
    produce a decodable mask of the wrong orientation.
    """
    rng = np.random.default_rng(0)
    mask = rng.random(shape) > 0.5
    decoded = mask_records.decode_mask(mask_records.encode_mask(mask))
    assert decoded.dtype == bool
    assert np.array_equal(decoded, mask)


@pytest.mark.parametrize("fill", [False, True])
def test_round_trip_of_a_uniform_mask(fill):
    """
    All-empty and all-full are the two RLE edge cases -- one run each -- and an
    empty mask is a legitimate thing to store even though it is not a
    legitimate thing for a segmenter to return.
    """
    mask = np.full((12, 20), fill, bool)
    assert np.array_equal(mask_records.decode_mask(mask_records.encode_mask(mask)),
                          mask)


def test_encode_records_the_area_and_the_shape():
    mask = blob_mask(shape=(200, 300))
    encoded = mask_records.encode_mask(mask)
    assert encoded["area"] == int(mask.sum())
    assert (encoded["rle_height"], encoded["rle_width"]) == (200, 300)


def test_encode_accepts_a_non_boolean_mask():
    """
    Callers hand in whatever their operation produced -- a uint8 from cv2, a
    float from a model. `> 0` is the documented rule, applied here so it is
    applied once.
    """
    integer = np.zeros((10, 10), np.uint8)
    integer[2:5, 3:8] = 7
    assert np.array_equal(
        mask_records.decode_mask(mask_records.encode_mask(integer)),
        integer > 0)


def test_decode_accepts_a_bytes_like_view():
    """
    A parquet read can hand back any bytes-like view of a binary column, while
    pycocotools wants exactly `bytes`. That conversion is why decode_mask calls
    bytes() rather than passing the value through.
    """
    mask = blob_mask()
    encoded = mask_records.encode_mask(mask)
    as_view = dict(encoded, rle_counts=memoryview(encoded["rle_counts"]))
    assert np.array_equal(mask_records.decode_mask(as_view), mask)


# ---------------------------------------------------------------------------
# derivation_hash -- the identity of a mask
# ---------------------------------------------------------------------------


def test_a_mask_with_no_upstream_is_its_recipe_hash():
    assert mask_records.derivation_hash("abc123") == "abc123"


@pytest.mark.parametrize("absent", [None, float("nan"), pd.NA, np.nan])
def test_a_missing_upstream_reads_as_no_upstream(absent):
    """
    A table written before `source_mask_hash` existed reads the column back as
    NaN, and a row that never had an upstream stores None. Both mean the same
    thing, and treating either as a real upstream would make every such mask
    permanently unmatchable.
    """
    assert mask_records.derivation_hash("abc123", absent) == "abc123"


def test_chaining_moves_the_identity():
    """A derived mask is not its own recipe -- it is its recipe on top of an upstream."""
    assert mask_records.derivation_hash("wing", "organism_v1") != "wing"


def test_a_different_upstream_gives_a_different_identity():
    """
    THE propagation property. The wing recipe hasn't changed; the organism it
    was cut out of has. Everything downstream must move with it.
    """
    assert (mask_records.derivation_hash("wing", "organism_v1")
            != mask_records.derivation_hash("wing", "organism_v2"))


def test_chaining_is_not_symmetric():
    """
    "wing cut from organism" and "organism cut from wing" are different claims,
    so a scheme that hashed the pair as a set would conflate two real cases.
    """
    assert (mask_records.derivation_hash("a", "b")
            != mask_records.derivation_hash("b", "a"))


def test_chaining_is_stable_across_calls():
    assert (mask_records.derivation_hash("wing", "organism")
            == mask_records.derivation_hash("wing", "organism"))


# ---------------------------------------------------------------------------
# make_mask_row
# ---------------------------------------------------------------------------


def test_row_has_exactly_the_table_columns():
    """
    The row IS the schema -- save_masks builds a DataFrame with COLUMNS, so a
    key that isn't one of them would be dropped silently.
    """
    assert set(make_row()) == set(mask_records.COLUMNS)


def test_keys_are_coerced_to_strings():
    """
    Storage compares keys by value without coercing, so the integer 7 and the
    string "7" would be different rows. The guarantee has to hold where rows
    are built.
    """
    row = mask_records.make_mask_row(7, blob_mask(), part=3)
    assert row["occurrence_id"] == "7"
    assert row["part"] == "3"


@pytest.mark.parametrize("occurrence_id, part",
                         [(None, "organism"), ("a", None), (None, None)])
def test_a_row_without_a_key_is_refused(occurrence_id, part):
    with pytest.raises(ValueError, match="needs both an occurrence_id and a part"):
        mask_records.make_mask_row(occurrence_id, blob_mask(), part=part)


def test_score_is_stored_as_a_float_or_left_none():
    assert make_row(score=1)["score"] == 1.0
    assert make_row()["score"] is None


def test_created_at_is_iso_utc():
    """
    Asserted once, here, because every other test strips this column before
    comparing -- which would otherwise mean nothing ever checks it.
    """
    assert is_iso_utc(make_row()["created_at"])


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_saving_nothing_writes_nothing(tmp_path):
    """An empty run is not an error, and must not create an empty mask table."""
    assert mask_records.save_masks(tmp_path, []) == 0
    assert not mask_records.has_masks(tmp_path)


def test_a_second_mask_replaces_the_first(tmp_path):
    """
    At most one canonical mask per occurrence-part. Resegmenting REPLACES;
    nothing accumulates, and nothing about the old mask survives except in
    whatever was derived from it.
    """
    mask_records.save_masks(tmp_path, [make_row(mask=blob_mask(axes=(30, 18)),
                                                recipe_hash="v1")])
    mask_records.save_masks(tmp_path, [make_row(mask=blob_mask(axes=(10, 6)),
                                                recipe_hash="v2")])

    stored = mask_records.load_masks(tmp_path)
    assert len(stored) == 1
    assert stored["recipe_hash"].iloc[0] == "v2"


def test_parts_of_one_occurrence_coexist(tmp_path):
    mask_records.save_masks(tmp_path, [
        make_row("a", part="organism"),
        make_row("a", part="wing"),
    ])
    assert mask_records.parts_present(tmp_path) == ["organism", "wing"]
    assert len(mask_records.load_masks(tmp_path)) == 2


def test_the_reference_table_is_a_different_file(tmp_path):
    """
    Validation is comparison between two tables of identical schema. Writing a
    reference mask must not touch the canonical one.
    """
    mask_records.save_masks(tmp_path, [make_row(recipe_hash="auto")])
    mask_records.save_masks(tmp_path, [make_row(recipe_hash="human")],
                            reference=True)

    assert mask_records.load_masks(tmp_path)["recipe_hash"].tolist() == ["auto"]
    assert mask_records.load_masks(tmp_path, reference=True)[
        "recipe_hash"].tolist() == ["human"]
    assert paths.masks_path(tmp_path).exists()
    assert paths.masks_path(tmp_path, reference=True).exists()


# ---------------------------------------------------------------------------
# save_mask_shard / merge_mask_shards
# ---------------------------------------------------------------------------


def test_a_shard_write_does_not_touch_the_canonical_table(tmp_path):
    """
    The whole point: a sharded run's flushes must never go anywhere near
    masks.parquet directly, since two concurrent writers upserting it at once
    would silently lose each other's rows.
    """
    mask_records.save_mask_shard(tmp_path, [make_row("a")], "organism")

    assert not mask_records.has_masks(tmp_path)
    staged = list(paths.mask_shards_dir(tmp_path, part="organism").glob("*.parquet"))
    assert len(staged) == 1


def test_saving_an_empty_shard_writes_nothing(tmp_path):
    assert mask_records.save_mask_shard(tmp_path, [], "organism") == 0
    assert not paths.mask_shards_dir(tmp_path).exists()


def test_merging_nothing_staged_is_a_no_op(tmp_path):
    assert mask_records.merge_mask_shards(tmp_path) == {}
    assert not mask_records.has_masks(tmp_path)


def test_merge_folds_every_staged_shard_into_the_canonical_table(tmp_path):
    mask_records.save_mask_shard(tmp_path, [make_row("a")], "organism")
    mask_records.save_mask_shard(tmp_path, [make_row("b")], "organism")

    merged = mask_records.merge_mask_shards(tmp_path)

    assert merged == {"organism": 2}
    assert sorted(mask_records.load_masks(tmp_path)["occurrence_id"]) == ["a", "b"]


def test_merge_resolves_a_key_written_by_two_shards_by_keeping_the_newest(tmp_path):
    """
    A shard rerun before a previous merge is the one way the same
    occurrence-part can end up staged twice -- resolved the same "newest
    wins" way the rest of the package reconciles a value recorded more than
    once.
    """
    mask_records.save_mask_shard(tmp_path, [make_row("a", recipe_hash="v1")], "organism")
    mask_records.save_mask_shard(tmp_path, [make_row("a", recipe_hash="v2")], "organism")

    mask_records.merge_mask_shards(tmp_path)

    stored = mask_records.load_masks(tmp_path)
    assert len(stored) == 1
    assert stored["recipe_hash"].iloc[0] == "v2"


def test_merge_cleans_up_consumed_shard_files_by_default(tmp_path):
    mask_records.save_mask_shard(tmp_path, [make_row("a")], "organism")
    mask_records.merge_mask_shards(tmp_path)

    assert list(paths.mask_shards_dir(tmp_path, part="organism").glob("*.parquet")) == []


def test_merge_can_leave_shard_files_in_place(tmp_path):
    mask_records.save_mask_shard(tmp_path, [make_row("a")], "organism")
    mask_records.merge_mask_shards(tmp_path, cleanup=False)

    assert len(list(
        paths.mask_shards_dir(tmp_path, part="organism").glob("*.parquet"))) == 1


def test_merge_only_touches_the_named_part(tmp_path):
    mask_records.save_mask_shard(tmp_path, [make_row("a", part="organism")], "organism")
    mask_records.save_mask_shard(tmp_path, [make_row("a", part="wing")], "wing")

    merged = mask_records.merge_mask_shards(tmp_path, part="organism")

    assert merged == {"organism": 1}
    assert mask_records.parts_present(tmp_path) == ["organism"]
    # The wing shard is untouched, still staged.
    assert len(list(
        paths.mask_shards_dir(tmp_path, part="wing").glob("*.parquet"))) == 1


def test_merge_does_not_disturb_the_reference_table(tmp_path):
    mask_records.save_mask_shard(tmp_path, [make_row("a")], "organism", reference=True)

    mask_records.merge_mask_shards(tmp_path, reference=True)

    assert mask_records.load_masks(
        tmp_path, reference=True)["occurrence_id"].tolist() == ["a"]
    assert not mask_records.has_masks(tmp_path)


def test_load_filters_by_part_ids_and_recipe(tmp_path):
    mask_records.save_masks(tmp_path, [
        make_row("a", part="organism", recipe_hash="v1"),
        make_row("b", part="organism", recipe_hash="v2"),
        make_row("a", part="wing", recipe_hash="v1"),
    ])

    assert len(mask_records.load_masks(tmp_path, parts=["wing"])) == 1
    assert len(mask_records.load_masks(tmp_path, occurrence_ids=["a"])) == 2
    assert len(mask_records.load_masks(tmp_path, recipe_hash="v2")) == 1


def test_load_matches_numeric_ids_as_strings(tmp_path):
    """
    Ids are strings everywhere. A caller filtering with the integer 7 means the
    occurrence called "7", and getting nothing back would read as "that
    occurrence has no mask".
    """
    mask_records.save_masks(tmp_path, [make_row(7)])
    assert len(mask_records.load_masks(tmp_path, occurrence_ids=[7])) == 1


def test_loading_a_project_with_no_masks_is_empty_not_an_error(tmp_path):
    assert mask_records.load_masks(tmp_path).empty
    assert mask_records.parts_present(tmp_path) == []
    assert mask_records.has_masks(tmp_path) is False


def test_get_mask_returns_pixels_or_none(tmp_path):
    mask = blob_mask()
    mask_records.save_masks(tmp_path, [make_row("a", mask=mask)])

    assert np.array_equal(mask_records.get_mask(tmp_path, "a"), mask)
    assert mask_records.get_mask(tmp_path, "missing") is None
    assert mask_records.get_mask(tmp_path, "a", part="wing") is None


def test_mask_lookup_is_keyed_by_occurrence(tmp_path):
    mask_records.save_masks(tmp_path, [make_row("a"), make_row("b")])
    lookup = mask_records.mask_lookup(tmp_path)
    assert set(lookup) == {"a", "b"}
    assert np.array_equal(mask_records.decode_mask(lookup["a"]), blob_mask())


# ---------------------------------------------------------------------------
# completed_keys -- what a rerun may skip
# ---------------------------------------------------------------------------


def test_completed_keys_covers_only_this_recipe(tmp_path):
    mask_records.save_masks(tmp_path, [
        make_row("a", recipe_hash="v1"),
        make_row("b", recipe_hash="v2"),
    ])
    assert mask_records.completed_keys(tmp_path, "v1") == {("a", "organism")}


def test_completed_keys_of_an_unrun_recipe_is_empty(tmp_path):
    mask_records.save_masks(tmp_path, [make_row("a", recipe_hash="v1")])
    assert mask_records.completed_keys(tmp_path, "other") == set()
    assert mask_records.completed_keys(tmp_path / "elsewhere", "v1") == set()


def test_a_derived_mask_counts_as_done_only_against_its_own_upstream(tmp_path):
    """
    The repeat-awareness half of the derivation invariant. The wing recipe hash
    hasn't moved, so without checking the upstream a rerun after resegmenting
    the organism would skip every occurrence and leave wing masks cut out of an
    organism the project no longer holds.
    """
    mask_records.save_masks(tmp_path, [
        make_row("a", part="wing", recipe_hash="wing_v1",
                 source_mask_hash="organism_v1"),
    ])
    key = ("a", "wing")

    same = mask_records.completed_keys(tmp_path, "wing_v1",
                                       source_mask_hashes={key: "organism_v1"})
    moved = mask_records.completed_keys(tmp_path, "wing_v1",
                                        source_mask_hashes={key: "organism_v2"})
    assert same == {key}
    assert moved == set()


def test_an_upstream_that_is_not_offered_never_counts_as_done(tmp_path):
    """
    "A mask whose upstream isn't in the map never counts as complete -- there's
    nothing left to confirm it against."
    """
    mask_records.save_masks(tmp_path, [
        make_row("a", part="wing", recipe_hash="wing_v1",
                 source_mask_hash="organism_v1"),
    ])
    assert mask_records.completed_keys(tmp_path, "wing_v1",
                                       source_mask_hashes={}) == set()


def test_a_mask_with_no_recorded_upstream_never_counts_as_done(tmp_path):
    """
    A from_part rerun over masks recorded before upstreams were tracked has
    nothing to confirm against either, so it redoes the work. One recompute is
    the honest price of not knowing.
    """
    mask_records.save_masks(tmp_path, [
        make_row("a", part="wing", recipe_hash="wing_v1"),
    ])
    assert mask_records.completed_keys(
        tmp_path, "wing_v1",
        source_mask_hashes={("a", "wing"): "organism_v1"}) == set()


# ---------------------------------------------------------------------------
# current_derivation_hashes -- the read side
# ---------------------------------------------------------------------------


def test_current_hashes_chain_the_upstream(tmp_path):
    mask_records.save_masks(tmp_path, [
        make_row("a", part="organism", recipe_hash="organism_v1"),
        make_row("a", part="wing", recipe_hash="wing_v1",
                 source_mask_hash="organism_v1"),
    ])
    current = mask_records.current_derivation_hashes(tmp_path)

    assert current[("a", "organism")] == "organism_v1"
    assert current[("a", "wing")] == mask_records.derivation_hash("wing_v1",
                                                                  "organism_v1")


def test_current_hashes_of_a_project_with_no_masks_is_empty(tmp_path):
    """
    Read as "nothing to judge staleness against", never as "everything is
    stale" -- the two are indistinguishable from here and only one is safe.
    """
    assert mask_records.current_derivation_hashes(tmp_path) == {}


def test_a_table_written_before_upstreams_were_tracked_still_reads(tmp_path):
    """
    Naming a column a parquet doesn't have fails the read outright, so the
    identity read asks the footer first. Such masks look upstream-less -- true
    of everything recorded at the time.
    """
    legacy_columns = [column for column in mask_records.COLUMNS
                      if column != "source_mask_hash"]
    row = {key: value for key, value in make_row("a").items()
           if key in legacy_columns}
    write_table(pd.DataFrame([row], columns=legacy_columns),
                paths.masks_path(tmp_path))

    assert "source_mask_hash" not in mask_records.load_masks(tmp_path).columns
    assert mask_records.current_derivation_hashes(tmp_path) == {
        ("a", "organism"): row["recipe_hash"]}
