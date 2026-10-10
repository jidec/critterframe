"""
The pieces every driver shares: what an operation's info keeps when stored,
the labels it is stored under, how a chain is run, and how one occurrence-part
becomes a segment framed by another part.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.core.drivers import NO_IMAGE, NoInput, Tally, no_mask
from critterframe.core.recipes import Segment, Transform
from critterframe.core.segments import framed_segment, operation_labels, run_chain, scalar_info
from critterframe.records.masks import make_mask_row
from helpers.synthetic import blob_mask


def test_only_scalars_are_kept():
    info = {
        "unreliable": True,
        "ratio": 0.2,
        "n": 3,
        "prompt": "box",
        "missing": None,
        "box": [1, 2, 3, 4],
        "nested": {"a": 1},
        "array": np.zeros(3),
    }
    assert scalar_info(info) == {"unreliable": True, "ratio": 0.2, "n": 3, "prompt": "box", "missing": None}


def test_numpy_scalars_become_python_ones():
    kept = scalar_info({"flag": np.bool_(True), "ratio": np.float32(0.5), "n": np.int64(4)})
    assert kept == {"flag": True, "ratio": 0.5, "n": 4}
    assert all(type(value) in (bool, float, int) for value in kept.values())


def test_no_info_is_empty():
    assert scalar_info(None) == {}


def test_a_repeated_operation_gets_a_numbered_label():
    labels = operation_labels([cf.orient(), cf.crop_to_mask(), cf.orient()])
    assert labels == ["orient", "crop_to_mask", "orient_2"]


# ---------------------------------------------------------------------------
# run_chain
# ---------------------------------------------------------------------------

SHAPE = (200, 300)


def reporting(name, **info):
    """A transform that changes nothing and reports `info`."""
    return Transform(name, lambda segment: (segment, dict(info)))


def blank_segment(mask=None):
    return Segment(np.zeros((*SHAPE, 3), np.uint8), mask=mask, occurrence_id="a")


def test_a_chain_keeps_each_operations_scalar_info_under_its_label():
    chain = [reporting("first", ratio=0.5, box=[1, 2]), reporting("second", n=3)]
    _state, info = run_chain(blank_segment(), chain)
    assert info == {"first": {"ratio": 0.5}, "second": {"n": 3}}
    assert list(info) == ["first", "second"]


def test_a_repeat_in_a_chain_is_numbered():
    _state, info = run_chain(blank_segment(), [reporting("step", n=1), reporting("step", n=2)])
    assert info == {"step": {"n": 1}, "step_2": {"n": 2}}


def test_labels_given_to_a_chain_are_used_as_they_are():
    """A slice of a recipe keeps the numbers the whole recipe gave it."""
    _state, info = run_chain(blank_segment(), [reporting("step", n=2)], labels=["step_2"])
    assert info == {"step_2": {"n": 2}}


def test_a_chain_counts_flags_into_every_tally_it_is_given():
    one, two = Tally(), Tally()
    chain = [reporting("orient", unreliable=True), reporting("crop", degenerate=True), reporting("fine")]
    run_chain(blank_segment(), chain, tallies=[one, two])
    assert one.flags == {"unreliable": 1, "degenerate": 1}
    assert two.flags == one.flags


def test_a_chain_returns_the_state_its_operations_made():
    mask = blob_mask(SHAPE)
    state, _info = run_chain(blank_segment(mask), [cf.crop_to_mask()])
    assert state.shape != SHAPE
    assert state.mask.any()


def test_an_empty_chain_changes_nothing():
    segment = blank_segment()
    state, info = run_chain(segment, [])
    assert state is segment
    assert info == {}


# ---------------------------------------------------------------------------
# framed_segment
# ---------------------------------------------------------------------------


class Images(dict):
    """Stands in for an open ImageStore: `get` by occurrence id, None when absent."""


@pytest.fixture
def images():
    return Images(a=np.full((*SHAPE, 3), 90, np.uint8))


def body_mask():
    return blob_mask(SHAPE, centre=(150, 100), axes=(80, 50))


def head_mask():
    return blob_mask(SHAPE, centre=(110, 90), axes=(15, 12))


def row(mask, part):
    return make_mask_row("a", mask, part=part, recipe_hash="recipe")


def test_an_occurrence_with_no_image_is_no_input(images):
    with pytest.raises(NoInput, match=NO_IMAGE):
        framed_segment(images, "missing", "organism", row(body_mask(), "organism"))


def test_a_missing_upstream_mask_is_no_input(images):
    with pytest.raises(NoInput, match=no_mask("organism")):
        framed_segment(images, "a", "head", row(head_mask(), "head"), from_part="organism", source_row=None)


def test_a_segment_carries_its_parts_own_mask(images):
    segment, info = framed_segment(images, "a", "head", row(head_mask(), "head"))
    assert segment.part == "head"
    assert np.array_equal(segment.mask, head_mask())
    assert info == {}


def test_a_segment_with_no_mask_row_has_no_mask(images):
    segment, _info = framed_segment(images, "a", "organism", None)
    assert segment.mask is None


def test_a_part_framed_by_another_keeps_its_own_mask_in_that_frame(images):
    """
    The chain crops to the ORGANISM, and the segment that comes back is the
    head's: labelled head, in the organism's crop, with the head's own mask
    reprojected into it -- which maps back onto the head where it really is.
    """
    segment, info = framed_segment(
        images,
        "a",
        "head",
        row(head_mask(), "head"),
        transforms=[cf.crop_to_mask()],
        from_part="organism",
        source_row=row(body_mask(), "organism"),
    )

    assert segment.part == "head"
    assert segment.shape != SHAPE  # the organism's crop, not the whole frame
    assert 0 < segment.mask.sum() < body_mask().sum()  # the head, not the organism
    assert np.array_equal(segment.mask_in_original_coordinates(), head_mask())
    assert list(info) == ["crop_to_mask"]


def test_a_framed_part_with_no_mask_of_its_own_has_none(images):
    segment, _info = framed_segment(
        images, "a", "head", None, from_part="organism", source_row=row(body_mask(), "organism")
    )
    assert segment.part == "head"
    assert segment.mask is None


def test_framing_counts_the_chains_flags(images):
    tally = Tally()
    framed_segment(
        images,
        "a",
        "organism",
        row(body_mask(), "organism"),
        transforms=[reporting("orient", unreliable=True)],
        tallies=[tally],
    )
    assert tally.flags == {"unreliable": 1}
