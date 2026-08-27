"""
Panels: one picture of what one operation decided about one occurrence-part.

The unit everything else in this package's visualization is built from. An
operation that makes a non-obvious decision -- which axis is the body, which
pixels were appendages, where a part boundary fell -- draws that decision here
and hands it over with segment.emit_panel(), instead of only reporting a number.
grids lays many panels out as one image; pipeline decides whose panels get laid
out; products materializes them per occurrence-part.

The drawing helpers are shared rather than reimplemented per operation for one
reason: a panel from appendage removal and a panel from reference-mask
validation should mean the same thing by the same colours. Learning to read one
is then learning to read all of them, and a white pixel never quietly means
"agreement" in one place and "mask" in another.

PanelFiles at the bottom is the other place a panel can end up: its own file, at
full resolution, for when you are looking hard at a few specimens rather than
scanning a whole run.
"""

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


def save_panel(project_path, image, name, subdir=""):
    """
    Write one panel to project_path/visualizations/<subdir>/<name>.png, and
    return the path.

    Outside pipeline/ and products/, which have their own contracts -- this is
    for a caller who wants a picture on disk and is naming the folder itself.

    project_path -- project whose visualizations directory to write into.
    image        -- BGR, grayscale, or boolean-mask array to write. A boolean
                    mask is converted here rather than refused: emit_panel's
                    contract allows one and grids lay one out happily, so a
                    sink that crashed on it would make an operation's panel
                    work in a run's grid and fail in a file.
    name         -- filename stem; ".png" is appended if absent.
    subdir       -- subfolder, conventionally the operation's name, so one
                    caller's output never mixes with another's.
    """
    dest_dir = paths.visualizations_dir(project_path, subdir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not name.lower().endswith(".png"):
        name = f"{name}.png"
    dest = dest_dir / name

    image = np.asarray(image)
    if image.dtype == bool:
        image = mask_to_bgr(image)

    cv2.imwrite(str(dest), image)   # cv2 wants a str, not a Path
    return dest


class PanelFiles:
    """
    A panel sink that writes every panel it's given as its own file.

    The counterpart to a pipeline RunReport, and the other thing a Segment's
    panel_sink can be. A report samples and composes because that's what makes a
    10,000-occurrence run inspectable; this writes one file per panel because
    that's what you want when you are looking hard at three specimens -- full
    resolution, one image per step, no grid cell shrinking the text.

    Deliberately NOT reachable through a run's visualize= argument. Per-panel
    files are right for a handful of segments you constructed yourself, and
    ruinous for a run over a collection, so getting them takes saying so:

        segment = Segment(image, mask=mask, occurrence_id="a1",
                          project_path=path, panel_sink=PanelFiles(path))

    Files land in visualizations/<stage>/<occurrence_id>.png, one folder per
    stage, so one operation's output never mixes with another's.
    """

    def __init__(self, project_path):
        self.project_path = project_path
        self.paths = []

    def wants(self, occurrence_id):
        """True always -- a file sink has no sample to be in."""
        return True

    def collect(self, occurrence_id, stage, image):
        """Write one panel, recording its path on self.paths."""
        dest = save_panel(self.project_path, image, str(occurrence_id),
                          subdir=stage)
        self.paths.append(dest)
        return dest


def mask_to_bgr(mask):
    """A boolean mask as a white-on-black BGR image, ready to draw on."""
    return cv2.cvtColor((np.asarray(mask).astype(np.uint8) * 255),
                        cv2.COLOR_GRAY2BGR)


def overlay_mask(image, mask, color=REMOVED_COLOR, alpha=0.5):
    """
    The image with mask pixels tinted -- the standard "is this segmentation
    right" view, and what every human-annotation window shows.
    """
    out = np.asarray(image).copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    selected = np.asarray(mask) > 0
    if selected.any():
        out[selected] = (
            (1 - alpha) * out[selected].astype(np.float32)
            + alpha * np.array(color, dtype=np.float32)
        ).astype(np.uint8)
    return out

def diff_panel(mask, other, agree=AGREE_COLOR, only_mask=(0, 255, 255),
               only_other=REMOVED_COLOR):
    """
    Two masks compared as one colored image: agreement in white, each mask's
    exclusive pixels in its own color.

    Used for every mask-vs-mask question in the package -- automated against
    a reference, a mask against its own mirror, before against after a
    transform -- so the colors mean the same thing wherever you see them.
    """
    mask = np.asarray(mask) > 0
    other = np.asarray(other) > 0

    panel = np.zeros((*mask.shape, 3), dtype=np.uint8)
    panel[mask & other] = agree
    panel[mask & ~other] = only_mask
    panel[other & ~mask] = only_other
    return panel


def annotate(image, text, line=0, color=TEXT_COLOR):
    """
    Draw one line of small diagnostic text at the top-left, in place.

    line -- 0-based line number, so several calls stack without each caller
            computing y offsets.
    """
    cv2.putText(image, text, (5, 15 + 17 * line), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1)
    return image


def side_by_side(*images):
    """
    Stack images horizontally, bottom-padding the shorter ones with black.

    Needed because panels being compared often differ in height -- a rotated
    image is taller than the one it came from -- and hstack refuses to join
    them.
    """
    images = [mask_to_bgr(image) if np.asarray(image).dtype == bool else image
              for image in images]
    images = [
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        for image in images
    ]
    height = max(image.shape[0] for image in images)
    padded = [
        cv2.copyMakeBorder(image, 0, height - image.shape[0], 0, 0,
                           cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for image in images
    ]
    return np.hstack(padded)
