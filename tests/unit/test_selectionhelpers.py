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
    rows_matching,
    sample_occurrences,
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
