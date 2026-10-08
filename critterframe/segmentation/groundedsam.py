"""SAM2 segmentation, with or without Grounding DINO detection."""

import logging
import time

import cv2
import numpy as np

from ..devices import resolve_device
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
    """SAM2, optionally preceded by Grounding DINO box detection.

    Args:
        model_name: SAM2 checkpoint to load.
        detect_bounds: Find the organism with the text-prompted detector first. False for
            an image already cropped around one organism.
        text_prompt: What the detector looks for, e.g. `"dragonfly."`. Lowercase and
            period-terminated, or a warning is logged.
        detector_name: Grounding DINO checkpoint; loaded only when `detect_bounds` is True.
        box_threshold: Detector confidence floor for a box.
        text_threshold: Detector confidence floor for the text match.
        size: SAM2's working resolution; the mask is returned at the image's own size.
        use_center_point: Without detection, prompt with a positive point at the center.
        use_corner_points: Without detection, prompt with negative points at the four corners.
        retry_without_center: Retry once without the center point when a center-prompted
            mask covers less than `min_area_frac` of the frame.
        min_area_frac: Share of the frame below which a mask triggers that retry.
        device: Torch device string; CUDA where available if None.
    """

    def __init__(
        self,
        model_name=DEFAULT_SAM_MODEL,
        detect_bounds=True,
        text_prompt=DEFAULT_TEXT_PROMPT,
        detector_name=DEFAULT_DETECTOR_MODEL,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        size=1024,
        use_center_point=False,
        use_corner_points=False,
        retry_without_center=True,
        min_area_frac=MIN_AREA_FRAC,
        device=None,
    ):
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

        # Warned about, never rewritten: the prompt is in identity(), so
        # normalizing it would make every mask already made with it stale.
        if detect_bounds and (text_prompt != text_prompt.lower() or not text_prompt.endswith(".")):
            logger.warning(
                "text_prompt %r: Grounding DINO expects a lowercase phrase ending in a period, e.g. %r",
                text_prompt,
                text_prompt.lower().rstrip(".") + ".",
            )

        self.processor = None
        self.model = None
        self.detector_processor = None
        self.detector = None

    def identity(self):
        """Return what this model contributes to a recipe hash: its checkpoints and prompting settings."""
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
            identity.update(
                {
                    "detector": self.detector_name,
                    "text_prompt": self.text_prompt,
                    "box_threshold": self.box_threshold,
                    "text_threshold": self.text_threshold,
                }
            )
        else:
            identity.update(
                {
                    "use_center_point": self.use_center_point,
                    "use_corner_points": self.use_corner_points,
                    "retry_without_center": self.retry_without_center,
                    "min_area_frac": self.min_area_frac,
                }
            )
        return identity

    @property
    def device(self):
        """Return the torch device, resolved on first use."""
        if self._device is None:
            self._device = resolve_device()
        return self._device

    def _load(self):
        """Load SAM2, and the detector if used, on first use.

        Each is guarded on its own, so one that failed to load is retried without skipping
        the other.
        """
        if self.model is None:
            from transformers import Sam2Model, Sam2Processor

            logger.info("loading %s on %s", self.model_name, self.device)
            self.processor = Sam2Processor.from_pretrained(self.model_name)
            # downsize the processor's working resolution (from 1024) for speed
            self.processor.image_processor.size = {"height": self.size, "width": self.size}
            self.model = Sam2Model.from_pretrained(self.model_name).to(self.device)

        if self.detect_bounds and self.detector is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            logger.info("loading detector %s on %s", self.detector_name, self.device)
            self.detector_processor = AutoProcessor.from_pretrained(self.detector_name)
            self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(self.detector_name).to(
                self.device
            )

    def detect(self, image):
        """Find the highest-scoring box matching the text prompt.

        Args:
            image: PIL RGB image or RGB array.

        Returns:
            `(box, score)` with the box as `[x0, y0, x1, y1]` in image pixels, or `(None, 0.0)`
            if nothing passed the thresholds.
        """
        boxes = self.detect_boxes(image)
        return boxes[0] if boxes else (None, 0.0)

    def detect_boxes(self, image):
        """Find every box matching the text prompt above the thresholds.

        Args:
            image: PIL RGB image or RGB array.

        Returns:
            `[(box, score), ...]`, best first, each box as `[x0, y0, x1, y1]` in image pixels.
        """
        import torch

        self._load()
        height, width = np.asarray(image).shape[:2]

        inputs = self.detector_processor(images=image, text=self.text_prompt, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.detector(**inputs)

        results = self.detector_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]

        found = [
            ([float(v) for v in box], float(score))
            for box, score in zip(results["boxes"].tolist(), results["scores"].tolist())
        ]
        return sorted(found, key=lambda item: item[1], reverse=True)

    def _prompt_points(self, width, height, use_center_point, use_corner_points):
        """Return the `(points, labels)` prompt lists; both empty when neither flag is set."""
        points, labels = [], []
        if use_center_point:
            points.append([width // 2, height // 2])  # center -- the organism
            labels.append(1)
        if use_corner_points:
            mx, my = int(width * CORNER_MARGIN), int(height * CORNER_MARGIN)
            points += [[mx, my], [width - mx, my], [mx, height - my], [width - mx, height - my]]
            labels += [0, 0, 0, 0]  # corners -- background

        return points, labels

    def _predict(self, image, points=None, labels=None, box=None, mask_threshold=0.0):
        """Run one SAM2 pass for a given prompt, and return `(mask, score)`.

        `mask_threshold` is a logit cutoff and must be passed to `post_process_masks()`, which
        binarizes: thresholding its output afterwards does nothing.
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

        inputs = self.processor(images=image, return_tensors="pt", **processor_kwargs).to(self.device)

        started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        logger.debug("sam2 inference %.3fs", time.perf_counter() - started)

        masks = self.processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], mask_threshold=mask_threshold, binarize=True
        )

        # SAM returns 3 candidate masks; take the highest predicted-IoU one.
        scores = outputs.iou_scores[0][0]
        best = scores.argmax().item()

        # .cpu() before .numpy() so this works when the model is on GPU
        mask = masks[0][0][best].cpu().numpy().astype(bool)
        return mask, float(scores[best])

    def predict(self, image, mask_threshold=0.0):
        """Segment one organism out of an image.

        Args:
            image: PIL RGB image or RGB array.
            mask_threshold: Logit cutoff: 0.0 is neutral, negative grows the mask, positive
                shrinks it.

        Returns:
            `(mask, score, info)`: a boolean mask the size of the image, the model's predicted
            IoU, and diagnostics naming the prompting path and whether the retry ran.
        """
        array = np.asarray(image)
        height, width = array.shape[:2]
        info = {"detect_bounds": self.detect_bounds, "retried": False}

        if self.detect_bounds:
            boxes = self.detect_boxes(image)
            if not boxes:
                raise ValueError(
                    f"detector found nothing matching {self.text_prompt!r} "
                    f"above box_threshold={self.box_threshold}"
                )
            (box, box_score), runner_up = boxes[0], boxes[1:2]
            info["box"] = box
            info["box_score"] = box_score
            # A second box that clears the threshold and barely overlaps the
            # first is the likeliest sign of two organisms in one image.
            # second_box_score is 0.0 rather than missing when there is none,
            # since a missing value never passes an export filter.
            info["n_boxes"] = len(boxes)
            info["second_box_score"] = runner_up[0][1] if runner_up else 0.0
            info["second_box_iou"] = _box_iou(box, runner_up[0][0]) if runner_up else None
            mask, score = self._predict(image, box=box, mask_threshold=mask_threshold)
            info["prompt"] = "box"
            return mask, score, info

        points, labels = self._prompt_points(width, height, self.use_center_point, self.use_corner_points)
        mask, score = self._predict(image, points=points, labels=labels, mask_threshold=mask_threshold)
        info["prompt"] = "points"
        info["points"] = points
        info["labels"] = labels

        if (
            self.retry_without_center
            and self.use_center_point
            and mask.sum() < self.min_area_frac * mask.size
        ):
            fallback = "corners only" if self.use_corner_points else "no points"
            logger.info(
                "mask covers <%.1f%% of the frame, retrying without the center point (%s)",
                self.min_area_frac * 100,
                fallback,
            )
            points, labels = self._prompt_points(width, height, False, self.use_corner_points)
            mask, score = self._predict(image, points=points, labels=labels, mask_threshold=mask_threshold)
            info["retried"] = True
            info["points"] = points
            info["labels"] = labels

        return mask, score, info

    def visualize(self, segment, image, mask, score, info):
        """Return the image with the mask tinted, the prompt points and the detected box drawn."""
        panel = overlay_mask(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), mask, alpha=0.4)

        for (x, y), label in zip(info.get("points", []), info.get("labels", [])):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(panel, (int(x), int(y)), 5, color, -1)

        box = info.get("box")
        if box:
            cv2.rectangle(panel, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 128, 0), 2)

        annotate(panel, f"{info['prompt']} score {score:.3f}{'  RETRIED' if info['retried'] else ''}")
        segment.emit_panel(panel, "segment")


def _box_iou(a, b):
    """Return the intersection over union of two `[x0, y0, x1, y1]` boxes."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def groundedsam2(**kwargs):
    """Return a `GroundedSAM2` model for `segment()`: detection, then SAM2.

    Args:
        **kwargs: As in `GroundedSAM2`.
    """
    return GroundedSAM2(**kwargs)


def sam2(**kwargs):
    """Return SAM2 with no detector, for images already cropped around one organism.

    Args:
        **kwargs: As in `GroundedSAM2`, apart from `detect_bounds`.
    """
    kwargs.setdefault("detect_bounds", False)
    return GroundedSAM2(**kwargs)
