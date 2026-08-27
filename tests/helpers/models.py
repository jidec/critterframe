"""
Stub segmenters: everything the run machinery needs from a model, minus the model.

`segment(model)` asks for `predict(image, mask_threshold) -> (mask, score, info)`
and, optionally, `identity() -> dict` and `visualize(...)`. Nothing in that
contract is about neural networks, which is why twenty lines of thresholding can
stand in for SAM2 through every test of segmentation runs, repeat-awareness,
staleness, derived parts, and training data.

**These are not a substitute for testing GroundedSAM2 -- they are a substitute
for needing it.** A test that imported torch to prove `run_segments` skips
completed work would be slow and would be testing the wrong thing. The two or
three tests that genuinely exercise SAM2 are marked `gpu` and skipped by default.

The models differ only in ways that must move a recipe hash (`cutoff`, `erode`),
which is what makes "the same model twice" and "a genuinely different segmenter"
expressible in a test.
"""

import cv2
import numpy as np


class ThresholdModel:
    """
    The smallest thing meeting the segmenter contract.

    cutoff -- greyscale value above which a pixel is organism. The drawn
              specimens are 200/180/150 on a 40 ground, so anything between
              roughly 60 and 170 finds the body.
    erode  -- pixels to shave off the result. Exists so two configurations
              genuinely disagree about where the specimen ends, which is what a
              resegmentation test needs: a second model that returns the same
              mask proves nothing about staleness.

    Both are in identity(), so the two configurations hash differently -- the
    whole reason a rerun can tell them apart.
    """

    def __init__(self, cutoff=100, erode=0):
        self.cutoff = cutoff
        self.erode = erode

    def identity(self):
        return {"class": "ThresholdModel", "cutoff": self.cutoff,
                "erode": self.erode}

    def predict(self, image, mask_threshold=0.5):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray > self.cutoff
        if self.erode:
            kernel = np.ones((self.erode * 2 + 1,) * 2, np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        return mask, 0.9, {"cutoff": self.cutoff, "erode": self.erode}


class UnidentifiedModel:
    """
    A segmenter with no identity() -- what a bare torch.nn.Module looks like.

    For asserting the documented fallback: such a model reaches the recipe hash
    as its class name only, so two different fine-tunes of it are indistinguish-
    able. That is the honest answer for an object that never said which weights
    it holds, and it is why register_model() exists.
    """

    def predict(self, image, mask_threshold=0.5):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray > 100, None, {}


class NoThresholdModel:
    """
    A segmenter whose predict() takes no mask_threshold.

    `_segment` calls with the threshold and falls back to calling without it on
    TypeError, because not every segmenter produces something thresholdable.
    This is the model that exercises the fallback.
    """

    def identity(self):
        return {"class": "NoThresholdModel"}

    def predict(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray > 100, 0.5, {}


class WrongShapeModel:
    """
    Returns a mask that doesn't match the frame it was given.

    The failure `_segment` guards against explicitly: a mask of the wrong shape
    cannot be aligned back to original coordinates, so it must raise rather than
    be persisted somewhere plausible-looking.
    """

    def identity(self):
        return {"class": "WrongShapeModel"}

    def predict(self, image, mask_threshold=0.5):
        height, width = image.shape[:2]
        return np.ones((height // 2, width // 2), bool), 0.5, {}


class EmptyMaskModel:
    """Finds nothing. An empty mask is an error, not a zero-area organism."""

    def identity(self):
        return {"class": "EmptyMaskModel"}

    def predict(self, image, mask_threshold=0.5):
        return np.zeros(image.shape[:2], bool), 0.0, {}


class FailingModel:
    """
    Raises on every occurrence.

    For the "individual failures are logged and counted, never fatal" rule: one
    bad occurrence must not cost a run, so a run over eight of these must come
    back with failed=8 rather than an exception.
    """

    def identity(self):
        return {"class": "FailingModel"}

    def predict(self, image, mask_threshold=0.5):
        raise RuntimeError("this model always fails")
