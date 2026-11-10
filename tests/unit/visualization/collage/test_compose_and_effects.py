"""
Compose puts placed cutouts and decorations on one canvas; effects style a cutout.

An effect changes how a cutout looks, never which pixels belong to it -- so the
alpha footprint is kept, or grows by exactly the margin an outline or a shadow
adds around it.
"""

import numpy as np
import pandas as pd
import pytest

from critterframe.visualization import collage as cc
from critterframe.visualization.collage import compose
from critterframe.visualization.collage.layouts import make_slots


def square(color=(0, 0, 255), side=20, alpha=255):
    image = np.zeros((side, side, 4), np.uint8)
    image[:, :, :3] = color
    image[:, :, 3] = alpha
    return image


def placements(xs, ys, **columns):
    frame = make_slots(xs, ys, **columns)
    frame.insert(0, "slot", np.arange(len(frame)))
    frame.insert(0, "part", "organism")
    frame.insert(0, "occurrence_id", [f"o{i}" for i in range(len(frame))])
    frame["item"] = np.arange(len(frame))
    return frame


# ---------------------------------------------------------------------------
# effects
# ---------------------------------------------------------------------------


def test_tint_and_desaturate_keep_the_alpha_footprint():
    cutout = square()
    cutout[:5, :5, 3] = 0
    for effect in (cc.tint(0.5, color=(255, 0, 0)), cc.desaturate()):
        styled = effect(cutout, {})
        assert (styled[:, :, 3] == cutout[:, :, 3]).all()


def test_tint_without_a_color_follows_the_slot():
    styled = cc.tint(1.0)(square(), {"color": (0.0, 1.0, 0.0)})
    assert styled[0, 0, :3].tolist() == [0, 255, 0]


def test_tint_without_any_color_changes_nothing():
    cutout = square()
    assert (cc.tint(1.0)(cutout, {"color": None}) == cutout).all()


def test_desaturate_makes_gray():
    styled = cc.desaturate()(square((0, 0, 255)), {})
    b, g, r = styled[0, 0, :3]
    assert b == g == r


def test_an_outline_grows_the_footprint_by_its_width():
    cutout = square(side=50)
    styled = cc.outline(width=0.1)(cutout, {})
    pad = (styled.shape[0] - 50) // 2
    assert pad == 5
    assert (styled[pad:-pad, pad:-pad, :3] == cutout[:, :, :3]).all()
    assert styled[pad - 2, 25, 3] == 255  # the outline itself, outside the original


def test_a_shadow_pads_equally_so_the_center_stays():
    styled = cc.shadow(offset=(0.1, 0.1), blur=0.05)(square(side=40), {})
    assert styled.shape[0] == styled.shape[1]
    assert (styled.shape[0] - 40) % 2 == 0


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------


def test_warp_without_rotation_keeps_the_shape():
    piece = compose.warp(square(side=10), 2.0, 0.0, False)
    assert piece.shape == (20, 20, 4)


def test_warp_expands_to_hold_a_rotated_cutout():
    piece = compose.warp(square(side=10), 1.0, 45.0, False)
    assert piece.shape[0] == piece.shape[1] == 15


def test_warp_flip_mirrors():
    cutout = square(side=10)
    cutout[:, :5, :3] = 0
    flipped = compose.warp(cutout, 1.0, 0.0, True)
    assert flipped[5, 8, :3].tolist() == [0, 0, 0]


def test_a_hex_clip_empties_the_corners():
    piece = compose.warp(square(side=40), 1.0, 0.0, False, clip="hex", clip_px=30)
    assert piece[0, 0, 3] == 0
    assert piece[20, 20, 3] == 255


def test_over_puts_a_later_cutout_on_top():
    frame = placements([0, 0], [0, 0])
    geometry = compose.plan(frame, [(10, 10), (10, 10)], [], cell=10, max_side=1000)
    sources = {"o0": square((255, 0, 0), 10), "o1": square((0, 0, 255), 10)}
    canvas = compose.draw(geometry, lambda row: sources[row["occurrence_id"]])
    cy, cx = geometry.height // 2, geometry.width // 2
    assert canvas[cy, cx].tolist() == [0, 0, 255, 255]


def test_mean_averages_where_cutouts_overlap():
    frame = placements([0, 0], [0, 0])
    geometry = compose.plan(frame, [(10, 10), (10, 10)], [], cell=10, max_side=1000)
    sources = {"o0": square((200, 0, 0), 10), "o1": square((0, 0, 200), 10)}
    canvas = compose.draw(geometry, lambda row: sources[row["occurrence_id"]], blend="mean")
    cy, cx = geometry.height // 2, geometry.width // 2
    assert canvas[cy, cx].tolist() == [100, 0, 100, 255]


def test_multiply_darkens_where_cutouts_overlap():
    frame = placements([0, 0], [0, 0])
    geometry = compose.plan(frame, [(10, 10), (10, 10)], [], cell=10, max_side=1000)
    sources = {"o0": square((255, 255, 0), 10), "o1": square((0, 255, 255), 10)}
    canvas = compose.draw(geometry, lambda row: sources[row["occurrence_id"]], blend="multiply")
    cy, cx = geometry.height // 2, geometry.width // 2
    assert canvas[cy, cx, :3].tolist() == [0, 255, 0]


def test_a_transparent_background_stays_transparent():
    geometry = compose.plan(placements([0, 5], [0, 0]), [(10, 10)] * 2, [], cell=10, max_side=1000)
    canvas = compose.draw(geometry, lambda row: square(side=10))
    assert canvas[0, 0, 3] == 0


def test_plan_fits_the_canvas_to_max_side():
    geometry = compose.plan(
        placements(np.arange(100), np.zeros(100)), [(50, 50)] * 100, [], cell=100, max_side=500
    )
    assert max(geometry.width, geometry.height) <= 500


def test_a_canvas_too_large_is_refused(monkeypatch):
    monkeypatch.setattr(compose, "MAX_PIXELS", 100)
    with pytest.raises(ValueError, match="MAX_PIXELS"):
        compose.plan(placements([0, 1], [0, 0]), [(10, 10)] * 2, [], cell=50, max_side=1000)


def test_a_bottom_anchor_stands_cutouts_on_one_line():
    frame = placements([0, 1], [1, 1], anchor="bottom")
    geometry = compose.plan(frame, [(10, 40), (10, 10)], [], cell=40, max_side=1000)
    bottoms = geometry.pieces["cy"] + np.array([40, 10]) * geometry.pieces["scale"] / 2
    assert bottoms.iloc[0] == pytest.approx(bottoms.iloc[1])


def test_the_geometry_spec_moves_with_any_placement():
    frame = placements([0, 1], [0, 0])
    one = compose.plan(frame, [(10, 10)] * 2, [], cell=10, max_side=1000).spec()
    moved = frame.copy()
    moved.loc[1, "angle"] = 10.0
    two = compose.plan(moved, [(10, 10)] * 2, [], cell=10, max_side=1000).spec()
    assert one != two


def test_decorations_are_drawn():
    decorations = [
        {"kind": "rect", "x": 0, "y": 2, "width": 1, "height": 1, "color": [0, 255, 0], "layer": "over"},
        {"kind": "line", "points": [[0, 4], [1, 4]], "width": 0.1, "color": [255, 0, 0], "layer": "under"},
        {
            "kind": "text",
            "x": 0,
            "y": 5,
            "text": "Anax",
            "height": 0.5,
            "color": [255, 255, 255],
            "align": "left",
            "valign": "top",
            "layer": "over",
        },
    ]
    geometry = compose.plan(placements([0], [0]), [(10, 10)], decorations, cell=40, max_side=1000)
    canvas = compose.draw(geometry, lambda row: square(side=10))
    rect = geometry.decorations[0]
    assert canvas[int(rect["y"]) + 5, int(rect["x"]) + 5, :3].tolist() == [0, 255, 0]
    text = geometry.decorations[2]
    assert canvas[int(text["y"]) : int(text["y"] + text["height"]), int(text["x"]) :, 3].max() > 0


def test_text_renders_on_a_transparent_background():
    image = compose.text_image("Aeshna", 30, (255, 255, 255))
    assert image.shape[2] == 4
    assert image[:, :, 3].min() == 0 and image[:, :, 3].max() == 255


def test_an_unknown_blend_is_refused():
    geometry = compose.plan(placements([0], [0]), [(10, 10)], [], cell=10, max_side=100)
    with pytest.raises(ValueError, match="blend"):
        compose.draw(geometry, lambda row: square(), blend="screen")


def test_pieces_keep_their_order():
    frame = placements([0, 1, 2], [0, 0, 0])
    geometry = compose.plan(frame, [(10, 10)] * 3, [], cell=10, max_side=100)
    assert geometry.pieces["occurrence_id"].tolist() == ["o0", "o1", "o2"]
    assert isinstance(geometry.pieces, pd.DataFrame)
