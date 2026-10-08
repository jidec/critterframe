"""
A part merged from several other parts, and what happens to it when any of them moves.

`from_part=[...]` starts a segmentation recipe from the union of several stored
parts, so a 'body' made of head, thorax and abdomen is a run like any other: it
has a run record, it skips what it has done, and it is redone when one of the
parts it was made of is replaced.

The failure this file exists for is the one a derived part has, multiplied: the
merge's own recipe never changes when ONE of its sources is resegmented, so the
only thing that can say the merged mask is stale is a source hash covering every
upstream at once.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.project import paths
from critterframe.records import masks as mask_records
from critterframe.records.runs import load_runs
from critterframe.storage.tables import write_table
from helpers.models import ThresholdModel

pytestmark = pytest.mark.slow

SPECIMENS = 8
HALVES = ["left", "right"]


def split_organism(project_path, reference=False, left_hash="left_v1", right_hash="right_v1", only=None):
    """
    Store each organism mask as a 'left' and a 'right' half, whose union is
    the organism mask exactly.

    only -- restrict to "left" or "right", which is how a test resegments one
            half (a new hash) or leaves an occurrence without the other.
    """
    rows = []
    for occurrence_id, row in mask_records.mask_lookup(project_path).items():
        mask = mask_records.decode_mask(row)
        middle = mask.shape[1] // 2
        left, right = mask.copy(), mask.copy()
        left[:, middle:] = False
        right[:, :middle] = False
        if only in (None, "left"):
            rows.append(mask_records.make_mask_row(occurrence_id, left, part="left", recipe_hash=left_hash))
        if only in (None, "right"):
            rows.append(
                mask_records.make_mask_row(occurrence_id, right, part="right", recipe_hash=right_hash)
            )
    mask_records.save_masks(project_path, rows, reference=reference)


def drop_mask(project_path, part, occurrence_id):
    """Remove one occurrence-part's mask, leaving the rest of the table alone."""
    df = mask_records.load_masks(project_path)
    keep = ~((df["part"] == part) & (df["occurrence_id"] == occurrence_id))
    write_table(df[keep], paths.masks_path(project_path))


def merge(project_path, from_part=HALVES, steps=(), **kwargs):
    kwargs.setdefault("visualize", False)
    return cf.run_segments(project_path, part="body", from_part=from_part, steps=list(steps), **kwargs)[
        "body"
    ]


def test_the_merged_mask_is_the_union_of_its_parts(segmented_project):
    split_organism(segmented_project)

    assert merge(segmented_project)["processed"] == SPECIMENS

    organism = mask_records.mask_lookup(segmented_project)
    body = mask_records.mask_lookup(segmented_project, part="body")
    assert set(body) == set(organism)
    for occurrence_id, row in body.items():
        assert np.array_equal(
            mask_records.decode_mask(row), mask_records.decode_mask(organism[occurrence_id])
        )


def test_a_merge_is_a_run_like_any_other(segmented_project):
    """
    The provenance a merge written straight into the table never had: a run
    record, and every merged row pointing at it.
    """
    split_organism(segmented_project)
    summary = merge(segmented_project)

    body = mask_records.load_masks(segmented_project, parts=["body"])
    runs = load_runs(segmented_project, name="body")
    assert len(runs) == 1
    assert set(body["run_id"]) == {summary["run_id"]}
    assert set(body["recipe_hash"]) == {runs.iloc[0]["recipe_hash"]}
    assert set(body["from_part"]) == {"left+right"}


def test_a_merge_is_repeat_aware(segmented_project):
    split_organism(segmented_project)
    merge(segmented_project)

    assert merge(segmented_project)["skipped"] == SPECIMENS


def test_resegmenting_one_source_makes_every_merged_mask_pending(segmented_project):
    """
    THE test. Only 'left' moved, and the merge recipe did not move at all --
    and yet all eight are work again, with a new identity for anything measured
    from them to go stale against.
    """
    split_organism(segmented_project)
    merge(segmented_project)
    before = mask_records.current_derivation_hashes(segmented_project, parts=["body"])

    split_organism(segmented_project, only="left", left_hash="left_v2")

    assert merge(segmented_project)["processed"] == SPECIMENS
    after = mask_records.current_derivation_hashes(segmented_project, parts=["body"])
    assert set(before) == set(after)
    assert all(before[key] != after[key] for key in before)


def test_an_occurrence_missing_one_part_has_no_input(segmented_project):
    """
    A partial union would understate the merged region, so it isn't made --
    and it isn't a failure either, since it is attempted again once the missing
    part exists.
    """
    split_organism(segmented_project)
    occurrence_id = sorted(mask_records.mask_lookup(segmented_project))[0]
    drop_mask(segmented_project, "right", occurrence_id)

    first = merge(segmented_project)
    assert (first["processed"], first["no_input"], first["failed"]) == (SPECIMENS - 1, 1, 0)

    split_organism(segmented_project, only="right")
    assert merge(segmented_project)["processed"] == 1


def test_the_order_parts_are_named_in_is_not_identity(segmented_project):
    split_organism(segmented_project)
    merge(segmented_project, from_part=["left", "right"])

    assert merge(segmented_project, from_part=["right", "left"])["skipped"] == SPECIMENS


def test_a_one_part_list_is_the_same_recipe_as_its_name(segmented_project):
    """So no recipe written with a single from_part moves by gaining the list form."""
    merge(segmented_project, from_part="organism")

    assert merge(segmented_project, from_part=["organism"])["skipped"] == SPECIMENS
    body = mask_records.load_masks(segmented_project, parts=["body"])
    assert set(body["from_part"]) == {"organism"}


def test_upstream_is_read_from_the_canonical_table_unless_told(segmented_project):
    """
    `reference=True` says where the result goes, not where the upstream comes
    from -- correcting a canonical mask into a reference depends on that -- so
    merging reference parts has to be asked for.
    """
    split_organism(segmented_project, reference=True)

    canonical = merge(segmented_project, reference=True)
    assert (canonical["processed"], canonical["no_input"]) == (0, SPECIMENS)

    assert merge(segmented_project, reference=True, from_reference=True)["processed"] == SPECIMENS
    assert len(mask_records.load_masks(segmented_project, parts=["body"], reference=True)) == SPECIMENS
    assert mask_records.load_masks(segmented_project, parts=["body"]).empty


def test_reading_the_reference_upstream_is_a_different_recipe(segmented_project):
    split_organism(segmented_project)
    split_organism(segmented_project, reference=True)
    merge(segmented_project)

    assert merge(segmented_project, from_reference=True)["processed"] == SPECIMENS


def test_from_reference_needs_something_to_read(segmented_project):
    with pytest.raises(ValueError):
        cf.run_segments(
            segmented_project, steps=[cf.segment(ThresholdModel())], from_reference=True, visualize=False
        )


def test_an_empty_list_of_parts_is_refused(segmented_project):
    with pytest.raises(ValueError):
        merge(segmented_project, from_part=[])


def test_steps_run_on_the_union(segmented_project):
    """
    What a merge gains by being a recipe: the union is a starting mask, so a
    stray fragment one part carried can be cleaned off the merged result.
    """
    split_organism(segmented_project)
    occurrence_id = sorted(mask_records.mask_lookup(segmented_project))[0]
    left = mask_records.get_mask(segmented_project, occurrence_id, part="left")
    assert not left[:4, :4].any()
    left[:4, :4] = True
    mask_records.save_masks(
        segmented_project,
        [mask_records.make_mask_row(occurrence_id, left, part="left", recipe_hash="left_v1")],
    )

    merge(segmented_project)
    assert mask_records.get_mask(segmented_project, occurrence_id, part="body")[:4, :4].all()

    merge(segmented_project, steps=[cf.remove_islands()])
    assert not mask_records.get_mask(segmented_project, occurrence_id, part="body")[:4, :4].any()


def test_parts_of_different_shapes_fail_rather_than_pad(segmented_project):
    """
    A padded union would be stored as a mask that no longer matches its image;
    one bad occurrence is counted and the rest still merge.
    """
    split_organism(segmented_project)
    occurrence_id = sorted(mask_records.mask_lookup(segmented_project))[0]
    right = mask_records.get_mask(segmented_project, occurrence_id, part="right")
    mask_records.save_masks(
        segmented_project,
        [mask_records.make_mask_row(occurrence_id, right[:-1, :-1], part="right", recipe_hash="right_v1")],
    )

    result = merge(segmented_project)

    assert (result["processed"], result["failed"]) == (SPECIMENS - 1, 1)
