"""Effects: how a pasted cutout looks, never which pixels it holds."""

import cv2
import numpy as np

from .base import Step


class Effect(Step):
    """A step from one placed cutout to its styled version.

    Args:
        name: As in `Step`.
        function: Called as `function(bgra, placement, **parameters)`, returning a BGRA array
            padded equally on each side, so the cutout's center stays where it was.
        parameters: As in `Step`.
    """

    kind = "effect"

    def __init__(self, name, function, parameters=None):
        super().__init__(name, parameters)
        self.function = function

    def __call__(self, bgra, placement):
        """Apply the effect to a BGRA cutout as it will be pasted."""
        return self.function(bgra, placement, **self.parameters)


def _pad(bgra, pad):
    """Return the cutout with `pad` transparent pixels added on every side."""
    return cv2.copyMakeBorder(bgra, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))


def _under(bgra, layer):
    """Return the cutout composited over a same-sized BGRA layer."""
    alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
    under = layer[:, :, 3:4].astype(np.float32) / 255.0
    out_alpha = alpha + under * (1.0 - alpha)
    safe = np.where(out_alpha > 0, out_alpha, 1.0)
    color = (bgra[:, :, :3] * alpha + layer[:, :, :3] * under * (1.0 - alpha)) / safe
    return np.dstack([np.clip(color, 0, 255), np.clip(out_alpha * 255.0, 0, 255)]).astype(np.uint8)


def shadow(offset=(0.03, 0.03), blur=0.03, opacity=0.5, color=(0, 0, 0)):
    """Effect: a drop shadow under the cutout.

    Args:
        offset: `(dx, dy)` of the shadow, as fractions of the cutout's longest side.
        blur: Blur radius, as a fraction of the cutout's longest side.
        opacity: The shadow's opacity.
        color: BGR shadow color.
    """
    return Effect(
        "shadow",
        _shadow,
        {"offset": [float(v) for v in offset], "blur": blur, "opacity": opacity, "color": list(color)},
    )


def _shadow(bgra, placement, offset, blur, opacity, color):
    side = max(bgra.shape[:2])
    dx, dy = (int(round(value * side)) for value in offset)
    radius = max(1, int(round(blur * side)))
    pad = max(abs(dx), abs(dy)) + 2 * radius
    padded = _pad(bgra, pad)
    alpha = np.roll(np.roll(padded[:, :, 3], dy, axis=0), dx, axis=1).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), radius / 2.0) * opacity
    layer = np.zeros_like(padded)
    layer[:, :, :3] = color
    layer[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return _under(padded, layer)


def outline(width=0.02, color=(255, 255, 255)):
    """Effect: a solid outline around the cutout.

    Args:
        width: Outline width, as a fraction of the cutout's longest side.
        color: BGR outline color.
    """
    return Effect("outline", _outline, {"width": width, "color": list(color)})


def _outline(bgra, placement, width, color):
    pad = max(1, int(round(width * max(bgra.shape[:2]))))
    padded = _pad(bgra, pad)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    layer = np.zeros_like(padded)
    layer[:, :, :3] = color
    layer[:, :, 3] = cv2.dilate(padded[:, :, 3], kernel)
    return _under(padded, layer)


def tint(strength=0.25, color=None):
    """Effect: blend the cutout's colors toward a color, by default its slot's (as `mosaic` sets).

    Args:
        strength: How far toward the color, 0 to 1.
        color: BGR color; None uses the slot's color, leaving a slot without one unchanged.
    """
    return Effect("tint", _tint, {"strength": strength, "color": None if color is None else list(color)})


def _tint(bgra, placement, strength, color):
    if color is None:
        rgb = placement.get("color")
        if rgb is None or (isinstance(rgb, float) and np.isnan(rgb)):
            return bgra
        color = [255.0 * value for value in reversed(rgb)]
    out = bgra.copy()
    blended = out[:, :, :3].astype(np.float32) * (1.0 - strength) + np.asarray(color, np.float32) * strength
    out[:, :, :3] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def desaturate(amount=1.0):
    """Effect: drain the cutout's color toward gray.

    Args:
        amount: 1 for fully gray, 0 for unchanged.
    """
    return Effect("desaturate", _desaturate, {"amount": amount})


def _desaturate(bgra, placement, amount):
    out = bgra.copy()
    gray = cv2.cvtColor(np.ascontiguousarray(bgra[:, :, :3]), cv2.COLOR_BGR2GRAY).astype(np.float32)
    blended = bgra[:, :, :3].astype(np.float32) * (1.0 - amount) + gray[:, :, None] * amount
    out[:, :, :3] = np.clip(blended, 0, 255).astype(np.uint8)
    return out
