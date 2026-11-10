"""
Arrange reads stored values into an order; assign puts items into slots.

Neither measures anything: an arrangement is a function of the table it is given,
and an assignment of the arrangement and the slots.
"""

import numpy as np
import pandas as pd
import pytest

from critterframe.visualization import collage as cc
from critterframe.visualization.collage.arrange import Arrangement, principal_components
from critterframe.visualization.collage.layouts import Placed, make_slots


def table(**columns):
    n = len(next(iter(columns.values())))
    return pd.DataFrame({"occurrence_id": [f"o{i}" for i in range(n)], "part": ["organism"] * n, **columns})


def arrange(steps, frame):
    arranged = Arrangement.of(frame)
    for step in steps:
        arranged = step(arranged, frame)
    return arranged


def ids(arranged):
    return [key[0] for key in arranged.keys]


# ---------------------------------------------------------------------------
# arrange
# ---------------------------------------------------------------------------


def test_sort_by_orders_items():
    frame = table(v=[3.0, 1.0, 2.0])
    assert ids(arrange([cc.sort_by("v")], frame)) == ["o1", "o2", "o0"]
    assert ids(arrange([cc.sort_by("v", descending=True)], frame)) == ["o0", "o2", "o1"]


def test_a_circular_sort_wraps_at_its_start():
    """Hues 350 and 10 are neighbors, so starting at 340 puts 350 before 10."""
    frame = table(h=[10.0, 350.0, 180.0])
    assert ids(arrange([cc.sort_by("h", circular=True, start=340)], frame)) == ["o1", "o0", "o2"]


def test_ties_keep_the_earlier_order():
    frame = table(v=[1.0, 1.0, 0.0])
    assert ids(arrange([cc.sort_by("v")], frame)) == ["o2", "o0", "o1"]


def test_group_by_keeps_each_group_together_and_its_inner_order():
    frame = table(v=[4.0, 3.0, 2.0, 1.0], species=["b", "a", "b", "a"])
    arranged = arrange([cc.sort_by("v"), cc.group_by("species")], frame)
    assert ids(arranged) == ["o3", "o1", "o2", "o0"]
    assert arranged.groups == [("a",), ("a",), ("b",), ("b",)]


def test_group_by_size_puts_the_largest_group_first():
    frame = table(species=["a", "b", "b"])
    assert arrange([cc.group_by("species", order="size")], frame).groups[0] == ("b",)


def test_a_missing_group_label_is_its_own_group():
    frame = table(species=["a", None])
    assert arrange([cc.group_by("species")], frame).groups[1] == ("unknown",)


def test_scale_by_is_relative_to_the_median():
    frame = table(length=[10.0, 20.0, 40.0])
    arranged = arrange([cc.scale_by("length")], frame)
    assert arranged.scale.tolist() == [0.5, 1.0, 2.0]
    assert arranged.scale_reference["value"] == 20.0


def test_polar_position_puts_hue_zero_on_the_right():
    frame = table(h=[0.0, 90.0], c=[1.0, 2.0])
    targets = arrange([cc.polar_position("h", "c")], frame).targets
    assert targets == pytest.approx(np.array([[1.0, 0.0], [0.0, 2.0]]), abs=1e-9)


def test_project_reads_a_vector_column():
    frame = table(vec=[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.1]])
    targets = arrange([cc.project("vec")], frame).targets
    assert targets.shape == (3, 2)
    assert targets[2, 0] > targets[0, 0]  # signed so the first component increases with the data


def test_principal_components_are_deterministic_in_sign():
    features = np.random.default_rng(1).normal(size=(20, 4))
    assert principal_components(features) == pytest.approx(principal_components(features[:, :]))
    assert principal_components(features) == pytest.approx(principal_components(-features) * -1)


def test_path_walks_through_neighbors():
    frame = table(x=[0.0, 1.0, 2.0, 3.0, 10.0])
    arranged = arrange([cc.path("x", "o0", "o3", neighbors=2)], frame)
    assert ids(arranged) == ["o0", "o1", "o2", "o3"]


def test_path_refuses_an_unknown_end():
    with pytest.raises(ValueError, match="nope"):
        arrange([cc.path("x", "o0", "nope")], table(x=[0.0, 1.0]))


# ---------------------------------------------------------------------------
# assign
# ---------------------------------------------------------------------------


def plain(n):
    return Arrangement.of(table(v=list(range(n))))


def test_sequence_fills_slots_in_order_and_leaves_extras_out():
    placed = Placed(make_slots(np.arange(3), np.zeros(3)))
    placements = cc.sequence()(plain(5), placed)
    assert placements["item"].tolist() == [0, 1, 2]


def test_slots_sharing_a_source_get_one_item():
    placed = Placed(make_slots(np.arange(4), np.zeros(4), source=[0, 0, 1, 1]))
    assert cc.sequence()(plain(2), placed)["item"].tolist() == [0, 0, 1, 1]


def test_a_slot_size_is_scaled_by_its_item():
    arranged = plain(2)
    arranged.scale = np.array([1.0, 3.0])
    placements = cc.sequence()(arranged, Placed(make_slots([0, 1], [0, 0], size=2.0)))
    assert placements["size"].tolist() == [2.0, 6.0]


def test_match_recovers_a_planted_permutation():
    rng = np.random.default_rng(2)
    points = rng.uniform(size=(30, 2)) * 10
    permutation = rng.permutation(30)
    placed = Placed(make_slots(points[permutation, 0], points[permutation, 1]), targets=points)
    placements = cc.match()(plain(30), placed)
    assert sorted(placements["item"]) == list(range(30))
    assert placements["item"].tolist() == permutation.tolist()


def test_match_uses_each_item_once():
    placed = Placed(make_slots(np.arange(10), np.zeros(10)), targets=np.zeros((4, 2)))
    placements = cc.match()(plain(4), placed)
    assert sorted(placements["item"]) == [0, 1, 2, 3]


def test_match_with_reuse_fills_every_slot_from_its_nearest_item():
    arranged = plain(2)
    arranged.values = np.array([[0.0], [1.0]])
    placed = Placed(make_slots(np.arange(4), np.zeros(4)), slot_values=np.array([[0.1], [0.9], [0.2], [1.0]]))
    assert cc.match(on="values", reuse=True)(arranged, placed)["item"].tolist() == [0, 1, 0, 1]


def test_match_refuses_a_cost_matrix_too_large():
    placed = Placed(make_slots(np.arange(6), np.zeros(6)), targets=np.zeros((6, 2)))
    with pytest.raises(ValueError, match="max_items"):
        cc.match(max_items=5)(plain(6), placed)


def test_match_on_values_needs_values():
    with pytest.raises(ValueError, match="match_on"):
        cc.match(on="values")(plain(2), Placed(make_slots([0, 1], [0, 0])))
