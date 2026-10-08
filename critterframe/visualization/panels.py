"""Panels: one picture of one operation's decision, with the shared drawing helpers and color conventions."""

import logging

import cv2
import numpy as np

from ..project import paths

logger = logging.getLogger(__name__)

# Shared drawing conventions, so a panel from one operation reads the same way
# as a panel from another: yellow for the annotation text, and a consistent
# agree/only-A/only-B language for every mask comparison in the package
# (appendage removal, mirror symmetry, reference-mask diffs).
TEXT_COLOR = (0, 255, 255)
AGREE_COLOR = (255, 255, 255)
REMOVED_COLOR = (0, 0, 255)
ADDED_COLOR = (0, 255, 0)

# The third colour every mask-vs-mask panel needs: pixels only the first mask
# covers. Named rather than written inline, so a hand-built comparison can't
# quietly give "only A" a colour that means something else elsewhere.
ONLY_MASK_COLOR = (0, 255, 255)

# (width, height) box an interactive window is fitted into: a 1080p screen at
# Windows' 125% scaling (effectively 1536x864, since cv2's windows aren't
# DPI-aware), less the taskbar and title bar. Read at call time, so a script
# on a bigger screen can raise it once.
DISPLAY_MAX = (1450, 780)


def save_panel(project_path, image, name, subdir=""):
    """Write one panel to `visualizations/<subdir>/<name>.png`, and return the path.

    Args:
        project_path: Project whose visualizations directory to write into.
        image: BGR, grayscale or boolean array.
        name: Filename stem; `.png` is appended if absent.
        subdir: Subfolder, conventionally the operation's name.
    """
    dest_dir = paths.visualizations_dir(project_path, subdir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not name.lower().endswith(".png"):
        name = f"{name}.png"
    dest = dest_dir / name

    image = np.asarray(image)
    if image.dtype == bool:
        image = mask_to_bgr(image)

    cv2.imwrite(str(dest), image)  # cv2 wants a str, not a Path
    return dest


class PanelFiles:
    """A panel sink that writes every panel as its own file.

    For hand-built segments, as `Segment(..., panel_sink=PanelFiles(path))`. Files land in
    `visualizations/[<prefix>/]<stage>/<occurrence_id>.png`.

    Args:
        project_path: Project whose visualizations directory to write into.
        prefix: Folder to put every stage under, to keep one batch apart from another.

    Attributes:
        paths: The files written so far.
    """

    def __init__(self, project_path, prefix=""):
        self.project_path = project_path
        self.prefix = prefix
        self.paths = []

    def wants(self, occurrence_id):
        """Return True: a file sink takes every occurrence."""
        return True

    def collect(self, occurrence_id, stage, image):
        """Write one panel, and record its path on `paths`."""
        subdir = f"{self.prefix}/{stage}" if self.prefix else stage
        dest = save_panel(self.project_path, image, str(occurrence_id), subdir=subdir)
        self.paths.append(dest)
        return dest


def segment_panel(image, mask=None, lines=()):
    """Return the panel a driver draws for one item: the mask over the image, captioned.

    Args:
        image: BGR image; copied, not modified.
        mask: Boolean mask to tint, or None.
        lines: Text lines, drawn top-left in order.
    """
    panel = overlay_mask(image, mask) if mask is not None else np.asarray(image).copy()
    for line, text in enumerate(lines):
        annotate(panel, str(text), line=line)
    return panel


def mask_to_bgr(mask):
    """Return a boolean mask as a white-on-black BGR image."""
    return cv2.cvtColor((np.asarray(mask).astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)


def overlay_mask(image, mask, color=REMOVED_COLOR, alpha=0.5):
    """Return the image with the mask's pixels tinted.

    Args:
        image: BGR image.
        mask: Boolean mask.
        color: BGR tint.
        alpha: Tint strength, 0 to 1.
    """
    out = np.asarray(image).copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    selected = np.asarray(mask) > 0
    if selected.any():
        out[selected] = (
            (1 - alpha) * out[selected].astype(np.float32) + alpha * np.array(color, dtype=np.float32)
        ).astype(np.uint8)
    return out


def diff_panel(mask, other, agree=AGREE_COLOR, only_mask=ONLY_MASK_COLOR, only_other=REMOVED_COLOR):
    """Return two masks compared as one image: agreement, and each mask's own pixels, by color.

    Args:
        mask: First boolean mask.
        other: Second boolean mask.
        agree: Color where both are set.
        only_mask: Color where only `mask` is set.
        only_other: Color where only `other` is set.
    """
    mask = np.asarray(mask) > 0
    other = np.asarray(other) > 0

    panel = np.zeros((*mask.shape, 3), dtype=np.uint8)
    panel[mask & other] = agree
    panel[mask & ~other] = only_mask
    panel[other & ~mask] = only_other
    return panel


def fit_for_display(image, max_size="screen", enlarge=True):
    """Return an image resized to fit a screen-sized box, and the factor it was resized by.

    Divide a clicked point by the factor to get it back in the image's own pixels.

    Args:
        image: Array to show; returned as is at factor 1.
        max_size: `(width, height)` box, `"screen"` for `DISPLAY_MAX`, or None for no resizing.
        enlarge: False only ever shrinks.

    Returns:
        `(display_image, scale)`.
    """
    if isinstance(max_size, str):
        max_size = DISPLAY_MAX
    if max_size is None:
        return image, 1.0
    height, width = np.asarray(image).shape[:2]
    scale = min(max_size[0] / width, max_size[1] / height)
    if not enlarge:
        scale = min(scale, 1.0)
    if scale == 1.0:
        return image, 1.0
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA
    return cv2.resize(np.asarray(image), size, interpolation=interpolation), scale


def annotate(image, text, line=0, color=TEXT_COLOR):
    """Draw one line of small text at the top-left of an image, in place.

    Args:
        image: Image to draw on.
        text: The text.
        line: 0-based line number, so several calls stack.
        color: BGR text color.
    """
    cv2.putText(image, text, (5, 15 + 17 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return image


def bordered(image, color, width=4):
    """Return a copy of an image with a solid frame drawn over its outer pixels.

    Args:
        image: Image to frame; its size is unchanged.
        color: BGR frame color.
        width: Frame width in pixels.
    """
    framed = np.asarray(image).copy()
    width = max(1, min(int(width), min(framed.shape[:2]) // 2))
    framed[:width] = color
    framed[-width:] = color
    framed[:, :width] = color
    framed[:, -width:] = color
    return framed


def side_by_side(*images):
    """Return images stacked horizontally, the shorter ones padded with black at the bottom."""
    images = [mask_to_bgr(image) if np.asarray(image).dtype == bool else image for image in images]
    images = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image for image in images]
    height = max(image.shape[0] for image in images)
    padded = [
        cv2.copyMakeBorder(image, 0, height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for image in images
    ]
    return np.hstack(padded)
