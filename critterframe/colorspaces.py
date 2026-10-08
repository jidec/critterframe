"""Color space conversion with no project attached: canonical units, circular channels, and the way back."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorSpace:
    """One color space: its channels, their nominal ranges, and which ones wrap.

    Attributes:
        name: The key in `SPACES`.
        channels: Lowercase channel names, in array order.
        ranges: Nominal `(low, high)` per channel, in the units `convert` returns.
        circular: Names of the channels that wrap around (hue, in degrees).
    """

    name: str
    channels: tuple
    ranges: tuple
    circular: frozenset = frozenset()

    @property
    def spans(self):
        """Return `high - low` per channel, as a float32 array."""
        return np.array([high - low for low, high in self.ranges], dtype=np.float32)

    def index(self, channel):
        """Return a channel's position in this space's arrays."""
        if channel not in self.channels:
            raise ValueError(
                f"{self.name} has no channel {channel!r} -- expected one of {list(self.channels)}"
            )
        return self.channels.index(channel)


# Nominal ranges are what a channel spans in practice, for scaling and plotting. Lab's a/b and lch's chroma are open
# ended in principle; the bounds are those of the sRGB gamut, rounded.
SPACES = {
    "rgb": ColorSpace("rgb", ("r", "g", "b"), ((0, 1), (0, 1), (0, 1))),
    "linrgb": ColorSpace("linrgb", ("r", "g", "b"), ((0, 1), (0, 1), (0, 1))),
    "hsv": ColorSpace("hsv", ("h", "s", "v"), ((0, 360), (0, 1), (0, 1)), frozenset({"h"})),
    "hls": ColorSpace("hls", ("h", "l", "s"), ((0, 360), (0, 1), (0, 1)), frozenset({"h"})),
    "lab": ColorSpace("lab", ("l", "a", "b"), ((0, 100), (-128, 127), (-128, 127))),
    "lch": ColorSpace("lch", ("l", "c", "h"), ((0, 100), (0, 181), (0, 360)), frozenset({"h"})),
}


def get_space(name):
    """Look up a color space by name.

    Args:
        name: A key of `SPACES`, or a `ColorSpace`, which is returned as is.

    Raises:
        ValueError: If the name is unknown.
    """
    if isinstance(name, ColorSpace):
        return name
    if name not in SPACES:
        raise ValueError(f"unknown colour space {name!r} -- expected one of {sorted(SPACES)}")
    return SPACES[name]


def convert(pixels, space):
    """Convert uint8 BGR pixels into a color space, in that space's canonical units.

    Args:
        pixels: uint8 array whose last axis is 3: an `(N, 3)` pixel list or an `(H, W, 3)` image.
        space: A key of `SPACES`.

    Returns:
        A float32 array shaped like `pixels`.
    """
    target = get_space(space)
    array = _as_bgr(pixels)
    if array.size == 0:
        return np.zeros(array.shape, dtype=np.float32)

    bgr = array.reshape(-1, 1, 3).astype(np.float32) / 255.0
    values = _FROM_BGR[target.name](bgr).reshape(-1, 3)
    return np.ascontiguousarray(values, dtype=np.float32).reshape(array.shape)


def to_bgr(values, space):
    """Convert values in a color space back to uint8 BGR, clipped to the displayable range.

    Args:
        values: Float array whose last axis is 3, in the units `convert` returns.
        space: A key of `SPACES`.
    """
    target = get_space(space)
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] != 3:
        raise ValueError(f"expected an array whose last axis is 3, got shape {array.shape}")
    if array.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    bgr = _TO_BGR[target.name](array.reshape(-1, 1, 3))
    return np.rint(np.clip(bgr, 0.0, 1.0) * 255.0).astype(np.uint8).reshape(array.shape)


def in_arc(angles, start, end):
    """Return which angles fall in the half-open arc `[start, end)`, in degrees.

    An arc with `start > end` wraps through 0, so `(340, 22)` is red. `start == end` is
    empty and `(0, 360)` is the whole circle.

    Args:
        angles: Angles in degrees, in `[0, 360)`.
        start: Where the arc begins, inclusive.
        end: Where the arc ends, exclusive.
    """
    angles = np.asarray(angles)
    if start <= end:
        return (angles >= start) & (angles < end)
    return (angles >= start) | (angles < end)


def normalize(values, space):
    """Rescale every channel to 0-1 by its nominal range.

    Args:
        values: Float array whose last axis is 3, in the units `convert` returns.
        space: A key of `SPACES`.
    """
    target = get_space(space)
    lows = np.array([low for low, _ in target.ranges], dtype=np.float32)
    return ((np.asarray(values, dtype=np.float32) - lows) / target.spans).astype(np.float32)


def _as_bgr(pixels):
    array = np.asarray(pixels)
    if array.dtype != np.uint8:
        raise TypeError(f"expected uint8 BGR pixels, got {array.dtype}")
    if array.ndim < 1 or array.shape[-1] != 3:
        raise ValueError(f"expected an array whose last axis is 3, got shape {array.shape}")
    return array


def _srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def _lab_to_lch(lab):
    a = lab[..., 1]
    b = lab[..., 2]
    return np.stack([lab[..., 0], np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360.0], axis=-1)


def _lch_to_lab(lch):
    angle = np.radians(lch[..., 2])
    return np.stack([lch[..., 0], lch[..., 1] * np.cos(angle), lch[..., 1] * np.sin(angle)], axis=-1)


# Each takes an (N, 1, 3) float32 array: BGR 0-1 going in, the space's own values coming out, and the reverse.
_FROM_BGR = {
    "rgb": lambda bgr: bgr[..., ::-1],
    "linrgb": lambda bgr: _srgb_to_linear(bgr[..., ::-1]),
    "hsv": lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
    "hls": lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS),
    "lab": lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB),
    "lch": lambda bgr: _lab_to_lch(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)),
}

_TO_BGR = {
    "rgb": lambda values: values[..., ::-1],
    "linrgb": lambda values: _linear_to_srgb(values)[..., ::-1],
    "hsv": lambda values: cv2.cvtColor(values, cv2.COLOR_HSV2BGR),
    "hls": lambda values: cv2.cvtColor(values, cv2.COLOR_HLS2BGR),
    "lab": lambda values: cv2.cvtColor(values, cv2.COLOR_LAB2BGR),
    "lch": lambda values: cv2.cvtColor(_lch_to_lab(values).astype(np.float32), cv2.COLOR_LAB2BGR),
}
