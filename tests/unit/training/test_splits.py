"""
Deciding which occurrences answer which question.

Splitting looks trivial and isn't, because biological image datasets leak in
ways a random split doesn't catch. Two failure modes, and the tests below are
arranged around them:

  GROUPED LEAKAGE -- several images of one specimen, one observation, or one
  trap night are not independent. Split them at random and near-duplicates land
  on both sides, and the validation score measures memorization. `group_by`
  keeps a group whole, and that guarantee is EXACT.

  IMBALANCE -- species counts in field data are wildly uneven, and a random
  split can leave a rare one absent from validation entirely. `stratify_by`
  preserves proportions, and that guarantee is APPROXIMATE, because a group
  cannot be divided to balance a stratum. The leakage one is the one worth
  keeping exact.

The third property is reproducibility: the same ids, proportions, and seed give
the same split whatever ORDER the ids arrive in. Without it, two scripts
computing "the same" split would disagree, and previously-validation specimens
would drift into training.
"""

import pandas as pd
import pytest

import critterframe as cf
from critterframe.records.occurrences import ID_COL, save_occurrences
from critterframe.training.splits import split_dataset, split_frames

PROPORTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}


@pytest.fixture
def specimens(tmp_path):
    """
    Thirty occurrences: fifteen specimens photographed twice each, across three
    species with deliberately uneven counts.
    """
    rows = []
    for specimen in range(15):
        for shot in range(2):
            rows.append({
                ID_COL: f"spec{specimen:02d}_{shot}",
                "specimen": f"spec{specimen:02d}",
                "species": ["Anax junius", "Anax junius",
                            "Libellula lydia"][specimen % 3],
                "device": "boxA" if specimen % 2 else "boxB",
            })
    save_occurrences(tmp_path, pd.DataFrame(rows))
    return tmp_path


def sides_of(project_path, splits, column):
    """{value: set of split names it appears in} for a grouping column."""
    table = pd.read_parquet(project_path / "occurrences.parquet").set_index(ID_COL)
    sides = {}
    for name, ids in splits.items():
        for occurrence_id in ids:
            sides.setdefault(table.loc[occurrence_id, column], set()).add(name)
    return sides


# ---------------------------------------------------------------------------
# split_ids
# ---------------------------------------------------------------------------


def test_every_occurrence_lands_in_exactly_one_split(specimens):
    splits = cf.split_ids(specimens, proportions=PROPORTIONS)
    allocated = [occurrence_id for ids in splits.values() for occurrence_id in ids]

    assert len(allocated) == 30
    assert len(set(allocated)) == 30


def test_the_proportions_are_roughly_honoured(specimens):
    splits = cf.split_ids(specimens, proportions=PROPORTIONS)
    assert len(splits["train"]) == pytest.approx(18, abs=3)
    assert len(splits["val"]) == pytest.approx(6, abs=3)


def test_every_requested_name_is_a_key_even_when_empty(specimens, caplog):
    """
    So a caller looping over splits never has to guess which ones exist -- and
    an empty one says so out loud rather than appearing as a missing key three
    steps later.
    """
    with caplog.at_level("WARNING"):
        splits = cf.split_ids(specimens, proportions={"train": 0.999,
                                                      "tiny": 0.001})
    assert set(splits) == {"train", "tiny"}
    assert splits["tiny"] == []
    assert "came out empty" in caplog.text


def test_proportions_need_not_sum_to_one(specimens):
    """They are normalized -- 8:1:1 is a perfectly clear way to say it."""
    splits = cf.split_ids(specimens, proportions={"train": 8, "val": 1, "test": 1})
    assert len(splits["train"]) == pytest.approx(24, abs=3)


def test_the_default_is_seventy_fifteen_fifteen(specimens):
    splits = cf.split_ids(specimens)
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == pytest.approx(21, abs=3)


# ---------------------------------------------------------------------------
# The two guarantees
# ---------------------------------------------------------------------------


def test_grouping_is_exact(specimens):
    """
    Both images of a specimen on one side, always. This is the guarantee that
    is worth keeping exact, because breaking it doesn't produce a worse score
    -- it produces a BETTER one, for the wrong reason.
    """
    splits = cf.split_ids(specimens, proportions=PROPORTIONS,
                          group_by="specimen")
    straddling = [value for value, names in sides_of(specimens, splits,
                                                     "specimen").items()
                  if len(names) > 1]
    assert straddling == []


def test_without_grouping_near_duplicates_do_straddle(specimens):
    """
    The failure being prevented, demonstrated: with no group column, the two
    shots of a specimen are independent rows and land wherever they fall.
    """
    splits = cf.split_ids(specimens, proportions=PROPORTIONS)
    straddling = [value for value, names in sides_of(specimens, splits,
                                                     "specimen").items()
                  if len(names) > 1]
    assert straddling


def test_stratifying_keeps_the_rare_species_present(specimens):
    """
    A random split of a long-tailed table can leave a rare class out of
    validation entirely, and then the score says nothing about it.
    """
    splits = cf.split_ids(specimens, proportions=PROPORTIONS,
                          stratify_by="species")
    table = pd.read_parquet(specimens / "occurrences.parquet").set_index(ID_COL)

    for name, ids in splits.items():
        assert set(table.loc[sorted(ids), "species"]) == {"Anax junius",
                                                          "Libellula lydia"}


def test_both_at_once_keeps_the_grouping_exact(specimens):
    """
    The tension stated in the module docstring: a group can't be divided to
    balance a stratum, so grouping wins and stratification is approximate.
    """
    splits = cf.split_ids(specimens, proportions=PROPORTIONS,
                          stratify_by="species", group_by="specimen")
    straddling = [value for value, names in sides_of(specimens, splits,
                                                     "specimen").items()
                  if len(names) > 1]
    assert straddling == []


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_the_same_call_gives_the_same_split(specimens):
    assert cf.split_ids(specimens, seed=123) == cf.split_ids(specimens, seed=123)


def test_the_order_the_ids_arrive_in_does_not_matter(specimens):
    """
    A subset query one day and a hand-written list the next must produce the
    same split, or the guarantee that a seed pins a split is worthless.
    """
    ids = pd.read_parquet(specimens / "occurrences.parquet")[ID_COL].tolist()
    forward = cf.split_ids(specimens, occurrence_ids=ids, seed=7)
    backward = cf.split_ids(specimens, occurrence_ids=list(reversed(ids)), seed=7)
    assert forward == backward


def test_a_different_seed_gives_a_different_split(specimens):
    assert cf.split_ids(specimens, seed=1) != cf.split_ids(specimens, seed=2)


# ---------------------------------------------------------------------------
# Selection and errors
# ---------------------------------------------------------------------------


def test_a_subset_can_be_split_instead_of_the_whole_project(specimens):
    cf.define_subset(specimens, "boxA", column="device", values=["boxA"])
    splits = cf.split_ids(specimens, subset="boxA")
    assert sum(len(ids) for ids in splits.values()) == 14


def test_ids_and_a_subset_are_two_ways_of_saying_the_same_thing(specimens):
    with pytest.raises(ValueError, match="not both"):
        cf.split_ids(specimens, occurrence_ids=["spec00_0"], subset="boxA")


def test_an_id_that_is_not_in_the_project_raises(specimens):
    """
    A typo or a stale list. Silently dropping it would shrink a training set
    without saying so.
    """
    with pytest.raises(KeyError, match="aren't in the occurrence table"):
        cf.split_ids(specimens, occurrence_ids=["spec00_0", "ghost"])


def test_a_column_the_project_does_not_have_raises_and_lists_the_ones_it_does(
        specimens):
    with pytest.raises(KeyError, match="no column"):
        cf.split_ids(specimens, stratify_by="genus")


def test_splitting_needs_a_project(empty_project):
    with pytest.raises(FileNotFoundError, match="isn't a CritterFrame project"):
        cf.split_ids(empty_project)


def test_nothing_is_written(specimens):
    """
    A split is a decision about the data; freezing one is define_subset's job.
    Keeping them separate is what lets one decision be reused for a
    segmenter's dataset, an encoder's, and a validation pass.
    """
    from critterframe.project import paths

    cf.split_ids(specimens, proportions=PROPORTIONS)
    assert not paths.subsets_path(specimens).exists()


def test_a_split_can_be_frozen_as_a_subset(specimens):
    """The documented way to make one outlive the script that computed it."""
    splits = cf.split_ids(specimens, proportions=PROPORTIONS)
    cf.define_subset(specimens, "train", occurrence_ids=splits["train"])

    from critterframe.project.subsets import select_ids
    assert sorted(select_ids(specimens, subset="train")) == sorted(splits["train"])


# ---------------------------------------------------------------------------
# split_dataset / split_frames -- the frame-shaped callers
# ---------------------------------------------------------------------------


def manifest(rows=20):
    return pd.DataFrame({
        "occurrence_id": [f"occ{index:02d}" for index in range(rows)],
        "label": ["a", "b"] * (rows // 2),
        "group": [f"g{index // 2}" for index in range(rows)],
    })


def test_split_dataset_adds_a_column_rather_than_splitting_the_frame():
    assigned = split_dataset(manifest(), fractions=PROPORTIONS)
    assert set(assigned["split"]) <= {"train", "val", "test"}
    assert len(assigned) == 20


def test_split_frames_hands_back_the_pieces():
    pieces = split_frames(manifest(), fractions=PROPORTIONS)
    assert set(pieces) <= {"train", "val", "test"}
    assert sum(len(frame) for frame in pieces.values()) == 20
    assert "split" not in next(iter(pieces.values())).columns


def test_an_empty_frame_splits_into_nothing():
    empty = manifest(0)
    assert split_dataset(empty, fractions=PROPORTIONS).empty


def test_proportions_that_sum_to_nothing_raise():
    with pytest.raises(ValueError, match="sum to something positive"):
        split_dataset(manifest(), fractions={"train": 0, "val": 0})


def test_an_unlabelled_row_is_pooled_rather_than_dropped():
    """An unlabelled image is still training data."""
    frame = manifest()
    frame.loc[0:3, "label"] = None
    assigned = split_dataset(frame, fractions=PROPORTIONS, stratify_col="label")
    assert assigned["split"].notna().all()
