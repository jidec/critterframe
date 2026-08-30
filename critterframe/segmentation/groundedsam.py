"""
SAM2, with or without Grounding DINO detection.

detect_bounds=True finds the organism with a text-prompted detector and
segments inside the box it found, for a photograph where the location is
unknown. detect_bounds=False prompts SAM2 geometrically instead, for an image
that is already a crop around one organism.

Torch and transformers are imported lazily, so constructing a model and reading
its identity() work without the [torch] extra installed.
"""

import logging
import time

import cv2
import numpy as np

from ..visualization.panels import annotate, overlay_mask

logger = logging.getLogger(__name__)

DEFAULT_SAM_MODEL = "facebook/sam2-hiera-large"
DEFAULT_DETECTOR_MODEL = "IDEA-Research/grounding-dino-base"

# Grounding DINO expects lowercase phrases ending in a period; "organism" is
# deliberately generic since CritterFrame projects run from moths to
# salamanders, and a project narrows it via text_prompt when it can.
DEFAULT_TEXT_PROMPT = "organism."

# Confidence floors for accepting a detection box and for matching a text token
# to it. Raise if background objects get boxed.
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

# Mask-area fraction below which a center-point prompt is assumed to have landed
# on background or a sliver and pulled SAM onto the wrong thing, triggering one
# retry without it.
MIN_AREA_FRAC = 0.02

# Inset for the negative corner points, so they don't land on the boundary pixel.
CORNER_MARGIN = 0.03


class GroundedSAM2:
    """
    SAM2, optionally preceded by Grounding DINO box detection.

    model_name     -- SAM2 checkpoint to load.
    detect_bounds  -- run the text-prompted detector to find the organism
                      first. False when the image is already a crop around one
                      organism (see module docstring).
    text_prompt    -- what to look for when detecting, e.g. "dragonfly." or
                      "moth.". Lowercase, period-terminated.
    detector_name  -- Grounding DINO checkpoint; only loaded when
                      detect_bounds is True.
    box_threshold,
    text_threshold -- detector confidence floors.
    size           -- SAM2's working resolution. Lower is faster; the mask is
                      post-processed back to the input image's size either way.
    use_center_point,
    use_corner_points -- geometric prompt used when detect_bounds is False: a
                      positive point at the center (the organism) and negative
                      points at the four corners (background), which helps
                      reject shadows and clutter.
    retry_without_center -- if a center-prompted mask covers less than
                      min_area_frac of the frame, retry once with the center
                      point dropped. If corners are also in use the retry falls
                      back to corners alone; if not, to no prompt at all.
    device         -- torch device string; autodetects CUDA if omitted.
    """

    def __init__(self, model_name=DEFAULT_SAM_MODEL, detect_bounds=True,
                 text_prompt=DEFAULT_TEXT_PROMPT, detector_name=DEFAULT_DETECTOR_MODEL,
                 box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
                 size=1024, use_center_point=False, use_corner_points=False,
                 retry_without_center=True, min_area_frac=MIN_AREA_FRAC,
                 device=None):
        self.model_name = model_name
        self.detect_bounds = detect_bounds
        self.text_prompt = text_prompt
        self.detector_name = detector_name
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.size = size
        self.use_center_point = use_center_point
        self.use_corner_points = use_corner_points
        self.retry_without_center = retry_without_center
        self.min_area_frac = min_area_frac
        self._device = device

        self.processor = None
        self.model = None
        self.detector_processor = None
        self.detector = None

    def identity(self):
        """
        What this model contributes to a recipe hash: everything about it that
        changes the mask it produces. The checkpoint names are the important
        part -- two runs of "sam2" against different weights are not equivalent
        work and must not be mistaken for it -- but the prompting strategy
        belongs here too, since a center-point prompt and a detected box are
        different segmentations of the same image.
        """
        identity = {
            "class": "GroundedSAM2",
            # Bumped when this class's implementation changes the mask it
            # produces for unchanged settings, so masks derived the old way stop
            # counting as work already done. v2: mask_threshold now reaches
            # post_process_masks and is applied to logits, where it previously
            # thresholded an already-binarized mask and therefore did nothing.
            "version": "2",
            "model": self.model_name,
            "detect_bounds": self.detect_bounds,
            "size": self.size,
        }
        if self.detect_bounds:
            identity.update({
                "detector": self.detector_name,
                "text_prompt": self.text_prompt,
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
            })
        else:
            identity.update({
                "use_center_point": self.use_center_point,
                "use_corner_points": self.use_corner_points,
                "retry_without_center": self.retry_without_center,
                "min_area_frac": self.min_area_frac,
            })
        return identity

    @property
    def device(self):
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _load(self):
        """Load SAM2 (and the detector, if used) once, on first use."""
        if self.model is not None:
            return

        from transformers import Sam2Model, Sam2Processor

        logger.info("loading %s on %s", self.model_name, self.device)
        self.processor = Sam2Processor.from_pretrained(self.model_name)
        # downsize the processor's working resolution (from 1024) for speed
        self.processor.image_processor.size = {"height": self.size, "width": self.size}
        self.model = Sam2Model.from_pretrained(self.model_name).to(self.device)

        if self.detect_bounds:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            logger.info("loading detector %s on %s", self.detector_name, self.device)
            self.detector_processor = AutoProcessor.from_pretrained(self.detector_name)
            self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.detector_name).to(self.device)

    def detect(self, image):
        """
        Find the highest-scoring box matching the text prompt.

        image -- PIL RGB image or an RGB array.

        Returns (box, score) with box as [x0, y0, x1, y1] in image pixels, or
        (None, 0.0) if nothing passed the thresholds -- which the caller treats
        as a failed segmentation rather than falling back silently to a
        different prompting strategy, since a silent fallback would produce a
        mask whose recipe no longer describes how it was made.
        """
        import torch

        self._load()
        height, width = np.asarray(image).shape[:2]

        inputs = self.detector_processor(
            images=image, text=self.text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.detector(**inputs)

        results = self.detector_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]

        scores = results["scores"]
        if len(scores) == 0:
            return None, 0.0

        best = int(scores.argmax())
        box = [float(v) for v in results["boxes"][best].tolist()]
        return box, float(scores[best])

    def _prompt_points(self, width, height, use_center_point, use_corner_points):
        """
        Build the (points, labels) prompt lists. Returns ([], []) if neither
        flag is set -- a valid, deliberate "no prompt" request (see _predict),
        not an error, since that's what the retry falls back to when corners
        aren't in use either.
        """
        points, labels = [], []
        if use_center_point:
            points.append([width // 2, height // 2])   # center -- the organism
            labels.append(1)
        if use_corner_points:
            mx, my = int(width * CORNER_MARGIN), int(height * CORNER_MARGIN)
            points += [[mx, my], [width - mx, my],
                       [mx, height - my], [width - mx, height - my]]
            labels += [0, 0, 0, 0]                     # corners -- background

        return points, labels

    def _predict(self, image, points=None, labels=None, box=None,
                 mask_threshold=0.0):
        """
        Run one SAM2 inference pass for a given prompt. Returns (mask, score).

        points/labels may be empty and box may be None -- those inputs are
        omitted from the processor call entirely rather than passed as
        empty-shaped lists, so the model runs whatever prompt-free path it
        supports on its own instead of being fed a malformed prompt.

        mask_threshold is a LOGIT cutoff, and it has to reach
        post_process_masks() to do anything. That call binarizes by default, so
        thresholding its output afterwards compares an already-binary mask
        against a number and silently does nothing -- which is what this used to
        do, with an 0.5 that looked like a probability cutoff and moved no
        boundary at all. Negative values grow the mask, positive shrink it, 0.0
        is SAM2's own default.
        """
        import torch

        self._load()

        processor_kwargs = {}
        if points:
            # shape: (batch=1, num_objects=1, num_points=len(points), 2)
            processor_kwargs["input_points"] = [[points]]
            processor_kwargs["input_labels"] = [[labels]]
        if box is not None:
            # shape: (batch=1, num_boxes=1, 4)
            processor_kwargs["input_boxes"] = [[box]]

        inputs = self.processor(images=image, return_tensors="pt",
                                **processor_kwargs).to(self.device)

        started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        logger.debug("sam2 inference %.3fs", time.perf_counter() - started)

        masks = self.processor.post_process_masks(outputs.pred_masks,
                                                  inputs["original_sizes"],
                                                  mask_threshold=mask_threshold,
                                                  binarize=True)

        # SAM returns 3 candidate masks; take the highest predicted-IoU one.
        scores = outputs.iou_scores[0][0]
        best = scores.argmax().item()

        # .cpu() before .numpy() so this works when the model is on GPU
        mask = masks[0][0][best].cpu().numpy().astype(bool)
        return mask, float(scores[best])

    def predict(self, image, mask_threshold=0.0):
        """
        Segment one organism out of an image.

        image -- PIL RGB image or an RGB array.

        Returns (mask, score, info): a boolean mask the same height/width as
        image, the model's predicted IoU for it, and diagnostics naming which
        prompting path ran and whether the low-area retry fired.
        """
        array = np.asarray(image)
        height, width = array.shape[:2]
        info = {"detect_bounds": self.detect_bounds, "retried": False}

        if self.detect_bounds:
            box, box_score = self.detect(image)
            info["box"] = box
            info["box_score"] = box_score
            if box is None:
                raise ValueError(
                    f"detector found nothing matching {self.text_prompt!r} "
                    f"above box_threshold={self.box_threshold}"
                )
            mask, score = self._predict(image, box=box, mask_threshold=mask_threshold)
            info["prompt"] = "box"
            return mask, score, info

        points, labels = self._prompt_points(width, height, self.use_center_point,
                                             self.use_corner_points)
        mask, score = self._predict(image, points=points, labels=labels,
                                    mask_threshold=mask_threshold)
        info["prompt"] = "points"
        info["points"] = points
        info["labels"] = labels

        if (self.retry_without_center and self.use_center_point
                and mask.sum() < self.min_area_frac * mask.size):
            fallback = "corners only" if self.use_corner_points else "no points"
            logger.info("mask covers <%.1f%% of the frame, retrying without the "
                        "center point (%s)", self.min_area_frac * 100, fallback)
            points, labels = self._prompt_points(width, height, False,
                                                 self.use_corner_points)
            mask, score = self._predict(image, points=points, labels=labels,
                                        mask_threshold=mask_threshold)
            info["retried"] = True
            info["points"] = points
            info["labels"] = labels

        return mask, score, info

    def visualize(self, segment, image, mask, score, info):
        """
        The image with the mask tinted, prompt points drawn (green positive,
        red negative) and the detected box outlined -- the view that shows
        whether a bad mask came from a bad prompt or a bad model call.
        """
        panel = overlay_mask(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
                             mask, alpha=0.4)

        for (x, y), label in zip(info.get("points", []), info.get("labels", [])):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(panel, (int(x), int(y)), 5, color, -1)

        box = info.get("box")
        if box:
            cv2.rectangle(panel, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (255, 128, 0), 2)

        annotate(panel, f"{info['prompt']} score {score:.3f}"
                        f"{'  RETRIED' if info['retried'] else ''}")
        segment.emit_panel(panel, "segment")


def groundedsam2(**kwargs):
    """
    A GroundedSAM2 model configured for segment().

    The generic entry point: text-prompted detection followed by SAM2 by
    default, or SAM2 alone with detect_bounds=False when the image is already
    cropped to one organism. See GroundedSAM2 for every parameter.
    """
    return GroundedSAM2(**kwargs)


def sam2(**kwargs):
    """
    SAM2 alone, with no detector -- shorthand for
    groundedsam2(detect_bounds=False).

    For images that are already crops around a single organism, where detection
    would only re-find what the crop already isolated.
    """
    kwargs.setdefault("detect_bounds", False)
    return GroundedSAM2(**kwargs)
