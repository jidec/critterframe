"""
"Out of these occurrences, which ones" -- the transient kind.

Distinct from project.subsets, which is about NAMED, persisted selections.
These are computed when asked for and never stored, and two of their properties
are load-bearing:

  a sample is STABLE, so two visualizations of two versions of a recipe show
  the same specimens and can be compared cell by cell;

  a rule set is membership-only, and a missing value never matches -- not
  knowing what a row is cannot be the same as knowing it is one of these
  things.

Nothing here reads a project, which is what keeps the module importable by
ingest and by run drivers without dragging in the metrics layer.
occurrences_matching -- the selection that does read stored values -- is
tested with the wide view it reads, in test_export.py.
"""

import pandas as pd
import pytest

from critterframe.selectionhelpers import (
    SAMPLE_SEED,
    cap_per_group,
    dedupe_by,
    grow_sample,
    rows_matching,
    sample_occurrences,
    sample_per_group,
    shard_occurrences,
)


def table():
    return pd.DataFrame({
        "occurrence_id": ["a", "b", "c", "d"],
        "determination": ["Noctuidae", "Not Lepidoptera", "Debris", None],
        "count": [1, 2, 3, 4],
    })


# ---------------------------------------------------------------------------
# rows_matching
# ---------------------------------------------------------------------------


def test_a_single_value_is_one_value_not_a_string_of_characters():
    """The single-value form is what anyone writes first."""
    matched = rows_matching(table(), {"determination": "Debris"})
    assert matched.tolist() == [False, False, True, False]


def test_any_rule_matching_is_enough():
    """
    These rule sets name several kinds of a thing ("debris, or
    not-Lepidoptera"), not a conjunction one row would have to satisfy at once.
    """
    matched = rows_matching(table(), {"determination": ["Debris"],
                                      "count": [1]})
    assert matched.tolist() == [True, False, True, False]


def test_a_missing_value_never_matches():
    """
    And the difference matters most for exactly the callers that act on the
    answer -- an unclassified detection is not a detection classified as debris.
    """
    matched = rows_matching(table(), {"determination": ["Debris", None]})
    assert matched.tolist() == [False, False, True, False]


def test_a_rule_on_a_column_that_is_not_there_raises():
    """
    A typo that silently matched nothing would read as "there was none of that
    here", which is the wrong answer to have believed.
    """
    with pytest.raises(KeyError, match="rule column"):
        rows_matching(table(), {"determinaton": ["Debris"]})


def test_matching_nothing_is_an_answer_not_an_error():
    assert not rows_matching(table(), {"determination": ["Sphingidae"]}).any()


# ---------------------------------------------------------------------------
# sample_occurrences
# ---------------------------------------------------------------------------


IDS = [f"occ{index:03d}" for index in range(100)]


def test_a_sample_is_the_same_every_time():
    """
    The whole point rather than an implementation detail: a sample that
    reshuffled on every call would make two versions of a recipe incomparable,
    and you could not tell a changed method from a changed specimen.
    """
    assert sample_occurrences(IDS, 10) == sample_occurrences(IDS, 10)


def test_a_sample_does_not_depend_on_the_order_the_ids_arrive_in():
    shuffled = list(reversed(IDS))
    assert sample_occurrences(shuffled, 10) == sample_occurrences(IDS, 10)


def test_a_sample_comes_back_sorted():
    sampled = sample_occurrences(IDS, 10)
    assert sampled == sorted(sampled)


def test_a_different_seed_gives_different_specimens():
    assert sample_occurrences(IDS, 10) != sample_occurrences(IDS, 10, seed=7)
    assert SAMPLE_SEED == 20250101


def test_asking_for_more_than_there_are_gives_all_of_them():
    """Asking for 25 from a project of 8 is a reasonable thing to do."""
    assert sample_occurrences(["a", "b"], 25) == ["a", "b"]
    assert sample_occurrences(["a", "b"], None) == ["a", "b"]


def test_ids_come_back_as_strings():
    sampled = sample_occurrences([1, 2, 3], 2)
    assert len(sampled) == 2
    assert set(sampled) <= {"1", "2", "3"}
    assert all(isinstance(occurrence_id, str) for occurrence_id in sampled)


def test_sampling_nothing_is_empty():
    assert sample_occurrences([], 5) == []


# ---------------------------------------------------------------------------
# grow_sample
# ---------------------------------------------------------------------------


def test_a_first_call_with_no_keep_ids_is_a_plain_sample():
    assert grow_sample(IDS, 10) == sample_occurrences(IDS, 10)


def test_growing_keeps_every_previously_picked_id():
    """
    The whole point: raising target_size across calls only adds to what's
    already there, the same additive guarantee cap_per_group's keep_ids gives
    a capped group on a later reimport.
    """
    first = grow_sample(IDS, 10)
    second = grow_sample(IDS, 20, keep_ids=first)

    assert set(first) <= set(second)
    assert len(second) == 20


def test_growing_draws_only_the_shortfall_from_ids_not_already_kept():
    first = grow_sample(IDS, 10)
    second = grow_sample(IDS, 20, keep_ids=first)

    added = set(second) - set(first)
    assert added.isdisjoint(first)
    assert len(added) == 10


def test_a_kept_id_no_longer_in_the_candidate_pool_is_dropped():
    """
    An occurrence that dropped out of the project (e.g. a reimport) can't be
    prioritized to stay, the same way cap_per_group only ever prioritizes a
    keep_ids row still present in df.
    """
    grown = grow_sample(["a", "b", "c"], 3, keep_ids={"a", "gone"})
    assert "gone" not in grown
    assert set(grown) == {"a", "b", "c"}


def test_shrinking_the_target_trims_deterministically():
    """
    Lowering target_size below len(keep_ids) trims rather than raising or
    growing back -- and the trim is stable, not an arbitrary set order.
    """
    grown = grow_sample(IDS, 20)
    shrunk_once = grow_sample(IDS, 10, keep_ids=grown)
    shrunk_again = grow_sample(IDS, 10, keep_ids=grown)

    assert len(shrunk_once) == 10
    assert shrunk_once == shrunk_again
    assert set(shrunk_once) <= set(grown)


def test_grow_sample_ids_come_back_sorted():
    grown = grow_sample(IDS, 10)
    assert grown == sorted(grown)


# ---------------------------------------------------------------------------
# cap_per_group
# ---------------------------------------------------------------------------


def species_table():
    """
    Seven "common", two "rare", one with no identification at all -- an
    imbalance in miniature of the sort a GBIF or iNaturalist pull produces.
    """
    return pd.DataFrame({
        "occurrence_id": [chr(ord("a") + index) for index in range(10)],
        "species": ["common"] * 7 + ["rare"] * 2 + [None],
    })


def test_a_group_at_or_under_the_cap_is_untouched():
    capped = cap_per_group(species_table(), "species", 2)
    assert (capped["species"] == "rare").sum() == 2   # already exactly at the cap

    capped = cap_per_group(species_table(), "species", 7)
    assert (capped["species"] == "common").sum() == 7   # exactly at the cap
    assert (capped["species"] == "rare").sum() == 2   # well under it


def test_an_oversized_group_is_thinned_to_exactly_max_count():
    capped = cap_per_group(species_table(), "species", 2)
    assert (capped["species"] == "common").sum() == 2
    assert (capped["species"] == "rare").sum() == 2   # already at the cap


def test_a_missing_group_value_is_never_capped():
    """
    Not knowing which group a row belongs to cannot be the same as knowing it
    is one of an oversized group's extras -- the same reasoning rows_matching
    applies to a missing value never matching a drop rule.
    """
    lopsided = pd.DataFrame({
        "occurrence_id": [f"o{index}" for index in range(20)],
        "species": [None] * 20,
    })
    capped = cap_per_group(lopsided, "species", 1)
    assert len(capped) == 20


def test_the_random_rule_is_stable_across_calls():
    """The whole point: a re-ingest of the same source keeps the same specimens."""
    first = cap_per_group(species_table(), "species", 2)
    second = cap_per_group(species_table(), "species", 2)
    assert first["occurrence_id"].tolist() == second["occurrence_id"].tolist()


def test_a_different_seed_can_keep_different_specimens():
    default_seed = cap_per_group(species_table(), "species", 2)
    other_seed = cap_per_group(species_table(), "species", 2, seed=7)
    assert default_seed["occurrence_id"].tolist() != other_seed["occurrence_id"].tolist()


def test_the_first_rule_keeps_the_groups_own_leading_rows():
    capped = cap_per_group(species_table(), "species", 2, rule="first")
    common = capped[capped["species"] == "common"]["occurrence_id"].tolist()
    assert common == ["a", "b"]


def test_the_last_rule_keeps_the_groups_own_trailing_rows():
    capped = cap_per_group(species_table(), "species", 2, rule="last")
    common = capped[capped["species"] == "common"]["occurrence_id"].tolist()
    assert common == ["f", "g"]


def test_a_callable_rule_picks_explicitly():
    def last_alphabetically(group):
        return group.sort_values("occurrence_id").tail(2)

    capped = cap_per_group(species_table(), "species", 2, rule=last_alphabetically)
    common = capped[capped["species"] == "common"]["occurrence_id"].tolist()
    assert sorted(common) == ["f", "g"]


def test_an_unknown_rule_raises():
    with pytest.raises(ValueError, match="unknown cap rule"):
        cap_per_group(species_table(), "species", 2, rule="whatever")


def test_an_unknown_group_col_raises():
    with pytest.raises(KeyError, match="no 'genus' column"):
        cap_per_group(species_table(), "genus", 2)


def test_surviving_rows_keep_their_relative_order():
    capped = cap_per_group(species_table(), "species", 2, rule="first")
    assert capped["occurrence_id"].tolist() == sorted(
        capped["occurrence_id"].tolist())


def test_capping_logs_an_aggregate_count(caplog):
    with caplog.at_level("INFO"):
        cap_per_group(species_table(), "species", 2)
    assert "capped 5 of 10 row(s)" in caplog.text


def test_capping_nothing_logs_nothing(caplog):
    with caplog.at_level("INFO"):
        cap_per_group(species_table(), "species", 100)
    assert "capped" not in caplog.text


# ---------------------------------------------------------------------------
# cap_per_group: keep_ids (stable across reimports)
# ---------------------------------------------------------------------------


def test_keep_ids_survive_over_new_candidates():
    capped = cap_per_group(species_table(), "species", 2, rule="first",
                           id_col="occurrence_id", keep_ids={"g"})
    common = capped[capped["species"] == "common"]["occurrence_id"].tolist()
    # "g" survives despite sorting last in file order; rows keep their
    # relative order in the output regardless of why each one was kept.
    assert sorted(common) == ["a", "g"]


def test_raising_the_cap_with_keep_ids_is_additive():
    """
    The whole point: growing max_per_group on a reimport adds to what a
    smaller-or-equal cap already selected, rather than resampling from scratch
    and orphaning work already done on specimens that are still good
    candidates.
    """
    first = cap_per_group(species_table(), "species", 2)
    kept = set(first["occurrence_id"])

    second = cap_per_group(species_table(), "species", 4,
                           id_col="occurrence_id", keep_ids=kept)
    assert kept <= set(second["occurrence_id"])
    assert len(second[second["species"] == "common"]) == 4


def test_lowering_the_cap_retrims_the_kept_ids_by_rule():
    previously_kept = {"a", "b", "c", "d"}
    capped = cap_per_group(species_table(), "species", 2, rule="first",
                           id_col="occurrence_id", keep_ids=previously_kept)
    common = capped[capped["species"] == "common"]["occurrence_id"].tolist()
    assert common == ["a", "b"]   # rule applied among the kept ids themselves


def test_a_group_at_or_under_the_cap_ignores_keep_ids():
    capped = cap_per_group(species_table(), "species", 7,
                           id_col="occurrence_id", keep_ids={"a"})
    assert (capped["species"] == "common").sum() == 7


def test_keep_ids_does_not_apply_to_a_callable_rule():
    def last_alphabetically(group):
        return group.sort_values("occurrence_id").tail(2)

    capped = cap_per_group(species_table(), "species", 2,
                           rule=last_alphabetically,
                           id_col="occurrence_id", keep_ids={"a"})
    common = capped[capped["species"] == "common"]["occurrence_id"].tolist()
    assert sorted(common) == ["f", "g"]   # "a" gets no special treatment


def test_keep_ids_without_id_col_raises():
    with pytest.raises(ValueError, match="id_col is required"):
        cap_per_group(species_table(), "species", 2, keep_ids={"a"})


def test_keep_ids_with_an_unknown_id_col_raises():
    with pytest.raises(KeyError, match="no 'specimen_id' column"):
        cap_per_group(species_table(), "species", 2,
                      id_col="specimen_id", keep_ids={"a"})


# ---------------------------------------------------------------------------
# dedupe_by
# ---------------------------------------------------------------------------


def sightings_table():
    """
    Two aggregators independently publishing the same two real sightings
    ("a"/"b" are one sighting, "c"/"d" another), a third sighting published
    once ("e"), and one row with no coordinates at all ("f").
    """
    return pd.DataFrame({
        "occurrence_id": ["a", "b", "c", "d", "e", "f"],
        "decimalLatitude": ["40.123401", "40.123449", "51.5", "51.500049",
                            "10.0", None],
        "decimalLongitude": ["-73.987601", "-73.987649", "-0.1", "-0.100049",
                             "20.0", "20.0"],
        "eventDate": ["2024-05-01", "2024-05-01", "2024-06-15", "2024-06-15",
                     "2024-07-04", "2024-07-04"],
    })


DEDUPE_COLS = ["decimalLatitude", "decimalLongitude", "eventDate"]
DEDUPE_PRECISION = {"decimalLatitude": 4, "decimalLongitude": 4}


def test_rows_within_precision_are_deduplicated():
    deduped = dedupe_by(sightings_table(), DEDUPE_COLS, precision=DEDUPE_PRECISION)
    ids = set(deduped["occurrence_id"])
    assert len(ids & {"a", "b"}) == 1   # one survivor of the first pair
    assert len(ids & {"c", "d"}) == 1   # one survivor of the second pair
    assert "e" in ids                  # published once, untouched


def test_a_row_missing_any_key_col_is_exempt():
    """
    Not knowing where or when a row was recorded cannot be the same as
    knowing it duplicates another -- the same reasoning cap_per_group applies
    to a missing group_col value.
    """
    deduped = dedupe_by(sightings_table(), DEDUPE_COLS, precision=DEDUPE_PRECISION)
    assert "f" in set(deduped["occurrence_id"])


def test_rows_outside_precision_are_not_deduplicated():
    table = pd.DataFrame({
        "occurrence_id": ["a", "b"],
        "decimalLatitude": ["40.1234", "40.1250"],   # ~0.18 km apart
        "decimalLongitude": ["-73.9876", "-73.9876"],
        "eventDate": ["2024-05-01", "2024-05-01"],
    })
    deduped = dedupe_by(table, DEDUPE_COLS, precision=DEDUPE_PRECISION)
    assert len(deduped) == 2


def test_without_precision_matching_is_exact():
    table = pd.DataFrame({
        "occurrence_id": ["a", "b"],
        "lat": ["40.12340", "40.1234"],   # equal as floats, not as strings
    })
    assert len(dedupe_by(table, ["lat"])) == 2
    assert len(dedupe_by(table, ["lat"], precision={"lat": 4})) == 1


def test_deduping_nothing_is_a_no_op():
    table = pd.DataFrame({
        "occurrence_id": ["a", "b"],
        "decimalLatitude": ["10.0", "20.0"],
        "decimalLongitude": ["30.0", "40.0"],
        "eventDate": ["2024-01-01", "2024-01-02"],
    })
    deduped = dedupe_by(table, DEDUPE_COLS, precision=DEDUPE_PRECISION)
    assert set(deduped["occurrence_id"]) == {"a", "b"}


def test_an_unknown_key_col_raises():
    with pytest.raises(KeyError, match="no 'nope' column"):
        dedupe_by(sightings_table(), ["nope"])


def test_keep_ids_survive_a_duplicate_resolved_before():
    """
    The same additive-reimport guarantee cap_per_group gives: a duplicate
    resolved once, and already carrying downstream work, must not be
    orphaned by a later pull that would otherwise pick differently.
    """
    first = dedupe_by(sightings_table(), DEDUPE_COLS, precision=DEDUPE_PRECISION,
                      id_col="occurrence_id")
    kept_ids = set(first["occurrence_id"])
    kept_from_first_pair = kept_ids & {"a", "b"}

    # Different seeds would otherwise be free to pick either survivor --
    # keep_ids overrides that regardless.
    for seed in range(10):
        again = dedupe_by(sightings_table(), DEDUPE_COLS, precision=DEDUPE_PRECISION,
                          id_col="occurrence_id", keep_ids=kept_ids, seed=seed)
        assert kept_from_first_pair <= set(again["occurrence_id"])


def test_the_survivor_is_stable_across_calls():
    first = dedupe_by(sightings_table(), DEDUPE_COLS, precision=DEDUPE_PRECISION)
    second = dedupe_by(sightings_table(), DEDUPE_COLS, precision=DEDUPE_PRECISION)
    assert sorted(first["occurrence_id"]) == sorted(second["occurrence_id"])


def test_deduping_logs_an_aggregate_count(caplog):
    with caplog.at_level("INFO"):
        dedupe_by(sightings_table(), DEDUPE_COLS, precision=DEDUPE_PRECISION)
    assert "deduplicated 2 of 6 row(s)" in caplog.text


def test_the_fingerprint_column_does_not_leak_into_the_result():
    deduped = dedupe_by(sightings_table(), DEDUPE_COLS, precision=DEDUPE_PRECISION)
    assert "_dedupe_fingerprint" not in deduped.columns


# ---------------------------------------------------------------------------
# sample_per_group
# ---------------------------------------------------------------------------


def test_a_rare_group_is_not_crowded_out_by_a_common_one():
    """
    The whole point: a flat sample_occurrences() draw over species_table()'s 7
    common / 2 rare / 1 unidentified would very likely miss "rare" entirely at
    a small count. Stratifying guarantees it isn't.
    """
    sampled = sample_per_group(species_table(), "species", 4, id_col="occurrence_id")
    rare_ids = set(species_table().query("species == 'rare'")["occurrence_id"])
    assert len(rare_ids & set(sampled)) == 2   # both rare specimens included


def test_a_shortfall_in_one_group_rolls_over_to_the_rest():
    """
    2 rare + 7 common, count=6: an even 3/3 split isn't possible since "rare"
    only has 2, so both of those are taken and the other 4 come from "common".
    """
    sampled = sample_per_group(species_table(), "species", 6, id_col="occurrence_id")
    assert len(sampled) == 6
    counts = species_table().set_index("occurrence_id").loc[sampled, "species"].value_counts()
    assert counts["rare"] == 2
    assert counts["common"] == 4


def test_an_even_split_is_exact_when_every_group_can_supply_it():
    table = pd.DataFrame({
        "occurrence_id": [f"o{i}" for i in range(15)],
        "species": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
    })
    sampled = sample_per_group(table, "species", 6, id_col="occurrence_id")
    counts = table.set_index("occurrence_id").loc[sampled, "species"].value_counts()
    assert counts.tolist() == [2, 2, 2]


def test_a_missing_group_value_is_excluded_not_sampled():
    sampled = sample_per_group(species_table(), "species", 10, id_col="occurrence_id")
    unidentified = species_table().query("species.isna()")["occurrence_id"].tolist()
    assert not set(unidentified) & set(sampled)


def test_asking_for_more_than_every_group_holds_gives_everything_available():
    """Like sample_occurrences: count exceeding availability returns all of it."""
    sampled = sample_per_group(species_table(), "species", 1000, id_col="occurrence_id")
    assert sorted(sampled) == sorted(species_table().query("species.notna()")["occurrence_id"])


def test_the_sample_is_stable_across_calls():
    first = sample_per_group(species_table(), "species", 4, id_col="occurrence_id")
    second = sample_per_group(species_table(), "species", 4, id_col="occurrence_id")
    assert first == second


def test_a_different_seed_can_give_a_different_pick_within_a_group():
    default = sample_per_group(species_table(), "species", 6, id_col="occurrence_id")
    reseeded = sample_per_group(species_table(), "species", 6, id_col="occurrence_id", seed=7)
    assert default != reseeded


def test_the_sample_comes_back_sorted():
    sampled = sample_per_group(species_table(), "species", 4, id_col="occurrence_id")
    assert sampled == sorted(sampled)


def test_sampling_nothing_or_less_is_empty():
    assert sample_per_group(species_table(), "species", 0, id_col="occurrence_id") == []
    assert sample_per_group(species_table(), "species", -1, id_col="occurrence_id") == []


def test_sample_per_group_on_an_unknown_group_col_raises():
    with pytest.raises(KeyError, match="no 'genus' column"):
        sample_per_group(species_table(), "genus", 4, id_col="occurrence_id")


def test_an_unknown_id_col_raises():
    with pytest.raises(KeyError, match="no 'specimen_id' column"):
        sample_per_group(species_table(), "species", 4, id_col="specimen_id")


def test_id_col_defaults_to_occurrence_id():
    sampled = sample_per_group(species_table(), "species", 4)
    assert sampled  # didn't need id_col passed explicitly


# ---------------------------------------------------------------------------
# shard_occurrences
# ---------------------------------------------------------------------------


def test_shards_are_disjoint_and_cover_everything():
    """
    The whole safety property a sharded run_segments() call leans on: no two
    shards ever touch the same occurrence, and nothing is left out.
    """
    shards = [shard_occurrences(IDS, index, 7) for index in range(7)]

    seen = set()
    for shard in shards:
        assert not (seen & set(shard))
        seen.update(shard)
    assert seen == set(IDS)


def test_a_shard_is_the_same_every_time():
    """
    Deterministic regardless of which process asks -- two workers given the
    same ids and the same total have to agree on the same split with no
    coordination between them.
    """
    assert shard_occurrences(IDS, 2, 7) == shard_occurrences(IDS, 2, 7)


def test_a_shard_does_not_depend_on_the_order_the_ids_arrive_in():
    shuffled = list(reversed(IDS))
    assert shard_occurrences(shuffled, 2, 7) == shard_occurrences(IDS, 2, 7)


def test_shards_are_balanced_to_within_one():
    sizes = [len(shard_occurrences(IDS, index, 7)) for index in range(7)]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == len(IDS)


def test_one_shard_of_one_is_everything():
    assert shard_occurrences(IDS, 0, 1) == sorted(IDS)


def test_an_out_of_range_index_raises():
    with pytest.raises(ValueError, match="index"):
        shard_occurrences(IDS, 7, 7)
    with pytest.raises(ValueError, match="index"):
        shard_occurrences(IDS, -1, 7)


def test_a_non_positive_total_raises():
    with pytest.raises(ValueError, match="total"):
        shard_occurrences(IDS, 0, 0)
