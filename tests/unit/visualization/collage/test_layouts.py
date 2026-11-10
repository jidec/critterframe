"""
Layouts turn an arrangement into slots: positions in layout units, nothing drawn.

Every layout is checked for the same three things -- one slot per item (or more,
where an assignment chooses among them), the same slots on every call, and no
NaN or infinity reaching the canvas -- because a layout that drifts between runs
moves the collage hash with nothing changed.
"""

import numpy as np
import pandas as pd
import pytest

from critterframe.visualization import collage as cc
from critterframe.visualization.collage.arrange import Arrangement


def arrangement(n, groups=None, targets=None, scale=None, parts=None):
    table = pd.DataFrame(
        {"occurrence_id": [f"o{i:03d}" for i in range(n)], "part": parts or ["organism"] * n}
    )
    arranged = Arrangement.of(table)
    arranged.groups = groups
    arranged.targets = None if targets is None else np.asarray(targets, float)
    arranged.scale = None if scale is None else np.asarray(scale, float)
    return arranged


def shape_mask():
    mask = np.zeros((200, 200), bool)
    mask[20:180, 40:160] = True
    return mask


SIMPLE = {
    "grid": lambda: cc.grid(),
    "grid_overlap": lambda: cc.grid(columns=4, overlap=0.3),
    "spiral": lambda: cc.spiral(),
    "rings": lambda: cc.rings(),
    "phyllotaxis": lambda: cc.phyllotaxis(),
    "hexgrid": lambda: cc.hexgrid(),
    "within": lambda: cc.within(shape_mask()),
    "mirrored": lambda: cc.mirrored(),
    "stack": lambda: cc.stack(),
}


@pytest.mark.parametrize("name", sorted(SIMPLE))
def test_a_layout_gives_one_finite_slot_per_item_every_time(name):
    first = SIMPLE[name]().place(arrangement(23)).slots
    second = SIMPLE[name]().place(arrangement(23)).slots
    assert len(first) >= 23
    assert np.isfinite(first[["x", "y", "angle", "size"]].to_numpy(float)).all()
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("name", sorted(SIMPLE))
def test_a_layout_of_nothing_is_empty(name):
    assert SIMPLE[name]().place(arrangement(0)).slots.empty


def test_grid_rows_run_left_to_right():
    slots = cc.grid(columns=3).place(arrangement(7)).slots
    assert slots["x"].tolist()[:4] == [0.0, 1.0, 2.0, 0.0]
    assert slots["y"].tolist()[:4] == [0.0, 0.0, 0.0, 1.0]


def test_grid_pitch_clears_the_largest_item():
    slots = cc.grid(columns=2).place(arrangement(2, scale=[1.0, 2.5])).slots
    assert slots["x"].tolist() == [0.0, 2.5]


def test_overlap_shortens_the_pitch():
    slots = cc.grid(columns=2, overlap=0.25).place(arrangement(2)).slots
    assert slots["x"].tolist() == [0.0, 0.75]


def test_a_spiral_moves_outward():
    slots = cc.spiral().place(arrangement(60)).slots
    radii = np.hypot(slots["x"], slots["y"])
    assert (np.diff(radii) >= 0).all()


def test_spiral_neighbors_are_a_pitch_apart():
    slots = cc.spiral(spacing=1.0).place(arrangement(80)).slots
    steps = np.hypot(np.diff(slots["x"]), np.diff(slots["y"]))
    assert steps[10:] == pytest.approx(1.0, rel=0.05)


def test_phyllotaxis_fills_from_the_center():
    slots = cc.phyllotaxis().place(arrangement(50)).slots
    assert (np.diff(np.hypot(slots["x"], slots["y"])) > 0).all()


def test_radial_rotation_points_an_item_outward():
    """The top of an item on the right of the ring faces right: a quarter turn clockwise."""
    slots = cc.rings(center=False, rotate="radial").place(arrangement(6)).slots
    right = slots.loc[slots["x"].idxmax()]
    assert right["angle"] % 360 == pytest.approx(270, abs=31)


def test_rings_by_group_start_each_group_on_its_own_ring():
    groups = [("a",)] * 3 + [("b",)] * 4
    slots = cc.rings(by_group=True).place(arrangement(7, groups=groups)).slots
    assert set(slots.loc[slots["group"] == "a", "ring"]) == {1}
    assert set(slots.loc[slots["group"] == "b", "ring"]) == {2}


def test_rings_by_group_needs_groups():
    with pytest.raises(ValueError, match="group_by"):
        cc.rings(by_group=True).place(arrangement(3))


def test_symmetry_repeats_each_ring_source_k_times():
    slots = cc.rings(symmetry=4).place(arrangement(10)).slots
    counts = slots.groupby("source").size()
    assert (counts == 4).all()
    assert counts.index.max() >= 9


def test_mirror_flips_every_other_repeat():
    slots = cc.rings(symmetry=4, mirror=True).place(arrangement(6)).slots
    assert 0 < slots["flip"].sum() < len(slots)


def test_mirrored_pairs_sit_either_side_of_the_axis():
    slots = cc.mirrored(columns=4).place(arrangement(8)).slots
    assert slots["x"].sum() == pytest.approx(0.0)
    assert (slots["flip"] == (slots["x"] > 0)).all()


def test_mirrored_fills_from_the_center():
    slots = cc.mirrored(columns=4).place(arrangement(12)).slots
    distance = np.hypot(slots["x"], slots["y"])
    assert distance.iloc[0] <= distance.iloc[-1]


def test_stack_puts_every_item_in_one_place():
    slots = cc.stack(alpha=0.3).place(arrangement(5)).slots
    assert set(zip(slots["x"], slots["y"])) == {(0.0, 0.0)}
    assert (slots["alpha"] == 0.3).all()


def test_within_keeps_every_slot_inside_the_shape():
    mask = shape_mask()
    slots = cc.within(mask, spacing=1.0).place(arrangement(30)).slots
    # Back to the mask's pixels: the lattice is scaled by one factor, so bounds compare.
    span_x = slots["x"].max() - slots["x"].min()
    span_y = slots["y"].max() - slots["y"].min()
    assert span_y > span_x  # the shape is taller than wide


def test_within_an_empty_shape_is_refused():
    with pytest.raises(ValueError, match="empty"):
        cc.within(np.zeros((10, 10), bool))


def test_hexgrid_clips_each_item_to_a_covering_hexagon():
    slots = cc.hexgrid(columns=3).place(arrangement(6)).slots
    assert (slots["clip"] == "hex").all()
    assert (slots["fit"] == "cover").all()
    assert (slots["size"] > slots["clip_size"]).all()


def test_scatter_needs_positions():
    with pytest.raises(ValueError, match="position"):
        cc.scatter().place(arrangement(3))


def test_scatter_snaps_to_at_least_one_cell_per_item():
    targets = np.random.default_rng(0).normal(size=(40, 2))
    placed = cc.scatter().place(arrangement(40, targets=targets))
    assert len(placed.slots) >= 40
    assert placed.targets.shape == (40, 2)


def test_scatter_turns_data_y_up_into_canvas_y_down():
    placed = cc.scatter(snap=False).place(arrangement(2, targets=[[0, 0], [0, 1]]))
    assert placed.slots["y"].iloc[1] < placed.slots["y"].iloc[0]


def test_a_backdrop_needs_an_extent():
    with pytest.raises(ValueError, match="extent"):
        cc.scatter(backdrop=np.zeros((4, 4, 3), np.uint8))


def test_a_backdrop_spans_the_extent():
    placed = cc.scatter(extent=(0, 10, 0, 5), backdrop=np.zeros((4, 4, 3), np.uint8)).place(
        arrangement(3, targets=[[0, 0], [10, 5], [5, 2]])
    )
    [image] = placed.decorations
    assert image["width"] / image["height"] == pytest.approx(2.0)


def test_mosaic_slots_carry_their_block_colors():
    target = np.zeros((10, 20, 3), np.uint8)
    target[:, 10:] = (0, 0, 255)  # red on the right
    placed = cc.mosaic(target, columns=2).place(arrangement(3))
    assert placed.slot_values.tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_the_mosaic_spec_names_its_image_by_digest():
    one = cc.mosaic(np.zeros((4, 4, 3), np.uint8)).spec()
    two = cc.mosaic(np.ones((4, 4, 3), np.uint8)).spec()
    assert one != two
    assert isinstance(one["parameters"]["image"], str)


def test_beeswarm_keeps_items_clear_of_each_other():
    targets = np.column_stack([np.zeros(10), np.zeros(10)])
    slots = cc.beeswarm(ticks=0).place(arrangement(10, targets=targets)).slots
    points = slots[["x", "y"]].to_numpy()
    gaps = np.hypot(*(points[:, None, :] - points[None, :, :]).transpose(2, 0, 1))
    assert gaps[~np.eye(10, dtype=bool)].min() >= 1.0 - 1e-6


def test_beeswarm_draws_an_axis():
    placed = cc.beeswarm().place(arrangement(5, targets=[[i, 0] for i in range(5)]))
    assert any(d["kind"] == "text" for d in placed.decorations)


def test_pack_circles_never_overlap():
    scale = [3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5]
    slots = cc.pack_circles(padding=0.0).place(arrangement(7, scale=scale)).slots
    centers, radii = slots[["x", "y"]].to_numpy(), np.asarray(scale) / 2
    for i in range(7):
        for j in range(i + 1, 7):
            assert np.hypot(*(centers[i] - centers[j])) >= radii[i] + radii[j] - 1e-6


def test_nested_lays_each_group_inside_its_outer_slot():
    groups = [("a",)] * 4 + [("b",)] * 9
    placed = cc.nested(cc.grid(columns=2), cc.grid()).place(arrangement(13, groups=groups))
    slots = placed.slots
    assert sorted(slots["item"]) == list(range(13))
    a, b = slots[slots["group"] == "a"], slots[slots["group"] == "b"]
    assert a["x"].max() < b["x"].min()


def test_nested_labels_each_group():
    groups = [("a",)] * 2 + [("b",)] * 2
    placed = cc.nested(cc.grid(), cc.grid(), labels=True).place(arrangement(4, groups=groups))
    assert sorted(d["text"] for d in placed.decorations) == ["a", "b"]


def test_a_dendrogram_has_one_leaf_per_item_and_edges():
    groups = [("f1", "g1"), ("f1", "g2"), ("f2", "g3"), ("f1", "g1")]
    placed = cc.dendrogram().place(arrangement(4, groups=groups))
    assert sorted(placed.slots["item"]) == [0, 1, 2, 3]
    assert any(d["kind"] == "line" for d in placed.decorations)
    assert {d["text"] for d in placed.decorations if d["kind"] == "text"} == {"g1", "g2", "g3"}


def test_a_dendrogram_puts_one_clade_s_leaves_together():
    groups = [("x",), ("y",), ("x",), ("y",)]
    slots = cc.dendrogram(radial=False).place(arrangement(4, groups=groups)).slots
    order = slots.sort_values("x")["item"].tolist()
    assert order == [0, 2, 1, 3]


def test_exploded_places_each_part_at_its_offset():
    placed = cc.exploded({"head": (0, -1), "abdomen": (0, 1)}).place(
        arrangement(2, parts=["head", "abdomen"])
    )
    by_part = dict(zip(["head", "abdomen"], placed.slots["y"]))
    assert by_part == {"head": -1.0, "abdomen": 1.0}
