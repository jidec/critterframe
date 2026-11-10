"""SAM3 segmentation, prompted by text or by points."""

import logging
import math
import time

import cv2
import numpy as np

from ..devices import resolve_device
from ..visualization.panels import annotate, overlay_mask
from .groundedsam import MIN_AREA_FRAC, _box_iou, prompt_points

logger = logging.getLogger(__name__)

DEFAULT_SAM3_MODEL = "facebook/sam3"

# SAM3 reads a short noun phrase, not Grounding DINO's period-terminated one.
# "organism" is as deliberately generic here as it is for GroundedSAM2; a project
# narrows it via text_prompt when it can.
DEFAULT_TEXT_PROMPT = "organism"

# Floor on an instance's score (its detection probability times SAM3's
# presence score for the phrase as a whole). Raise if background objects are
# segmented, lower if faint or partly hidden organisms are missed.
SCORE_THRESHOLD = 0.5


class SAM3:
    """SAM3, prompted with a text phrase or, for a pre-cropped image, with points.

    The text path is detection and segmentation in one model: SAM3 finds every instance
    of the phrase and the highest-scoring one is kept, where GroundedSAM2 needs a separate
    detector to box it first. The point path uses SAM3's tracker head, which takes the same
    prompts as SAM2. SAM3 works at a fixed 1008 px, so unlike `GroundedSAM2` it has no
    `size`.

    The `facebook/sam3` checkpoint is gated on Hugging Face: request access on its model
    page, then log in with a token (`hf auth login`) before the first load. A load refused
    for either reason raises a `RuntimeError` saying which step is missing; requests and
    their status are listed at https://huggingface.co/settings/gated-repos.

    Args:
        model_name: SAM3 checkpoint to load.
        detect_bounds: Find the organism from `text_prompt`. False for an image already
            cropped around one organism, which is prompted with points instead. Named as in
            `GroundedSAM2`, so the two are configured alike.
        text_prompt: What SAM3 looks for, a short noun phrase such as `"dragonfly"`.
        score_threshold: Score floor for an instance to count as found.
        use_center_point: Without detection, prompt with a positive point at the center.
        use_corner_points: Without detection, prompt with negative points at the four corners.
        retry_without_center: Retry once without the center point when a center-prompted
            mask covers less than `min_area_frac` of the frame.
        min_area_frac: Share of the frame below which a mask triggers that retry.
        device: Torch device string; CUDA where available if None.
    """

    def __init__(
        self,
        model_name=DEFAULT_SAM3_MODEL,
        detect_bounds=True,
        text_prompt=DEFAULT_TEXT_PROMPT,
        score_threshold=SCORE_THRESHOLD,
        use_center_point=False,
        use_corner_points=False,
        retry_without_center=False,
        min_area_frac=MIN_AREA_FRAC,
        device=None,
    ):
        self.model_name = model_name
        self.detect_bounds = detect_bounds
        self.text_prompt = text_prompt
        self.score_threshold = score_threshold
        self.use_center_point = use_center_point
        self.use_corner_points = use_corner_points
        self.retry_without_center = retry_without_center
        self.min_area_frac = min_area_frac
        self._device = device

        self.processor = None
        self.model = None
        self.tracker_processor = None
        self.tracker = None

    def identity(self):
        """Return what this model contributes to a recipe hash: its checkpoint and prompting settings."""
        identity = {
            "class": "SAM3",
            # Bumped when this class's implementation changes the mask it
            # produces for unchanged settings, so masks derived the old way stop
            # counting as work already done.
            "version": "1",
            "model": self.model_name,
            "detect_bounds": self.detect_bounds,
        }
        if self.detect_bounds:
            identity.update({"text_prompt": self.text_prompt, "score_threshold": self.score_threshold})
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
        """Load the half of SAM3 this configuration uses, on first use.

        The text path needs the concept model, the point path the tracker; only one is
        loaded, and a failed load leaves nothing behind, so the next call retries it.
        """
        if self.detect_bounds and self.model is None:
            from transformers import Sam3Model, Sam3Processor

            logger.info("loading %s on %s", self.model_name, self.device)
            processor = _from_pretrained(Sam3Processor, self.model_name)
            self.model = _from_pretrained(Sam3Model, self.model_name).to(self.device)
            self.processor = processor

        if not self.detect_bounds and self.tracker is None:
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor

            logger.info("loading %s tracker on %s", self.model_name, self.device)
            processor = _from_pretrained(Sam3TrackerProcessor, self.model_name)
            self.tracker = _from_pretrained(Sam3TrackerModel, self.model_name).to(self.device)
            self.tracker_processor = processor

    def instances(self, image, mask_threshold=0.0):
        """Find every instance of the text prompt above the score threshold.

        Args:
            image: PIL RGB image or RGB array.
            mask_threshold: Logit cutoff, as in `predict()`.

        Returns:
            `[(mask, box, score), ...]`, best first: a boolean mask the size of the image, the
            box as `[x0, y0, x1, y1]` in image pixels, and the instance's score.
        """
        import torch

        self._load()
        height, width = np.asarray(image).shape[:2]

        inputs = self.processor(images=image, text=self.text_prompt, return_tensors="pt").to(self.device)

        started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        logger.debug("sam3 inference %.3fs", time.perf_counter() - started)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            mask_threshold=logit_to_probability(mask_threshold),
            target_sizes=[(height, width)],
        )[0]

        found = [
            (mask.cpu().numpy().astype(bool), [float(v) for v in box], float(score))
            for mask, box, score in zip(
                results["masks"], results["boxes"].tolist(), results["scores"].tolist()
            )
        ]
        return sorted(found, key=lambda item: item[2], reverse=True)

    def _predict_points(self, image, points, labels, mask_threshold=0.0):
        """Run one tracker pass for a point prompt, and return `(mask, score)`.

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

        inputs = self.tracker_processor(images=image, return_tensors="pt", **processor_kwargs).to(self.device)

        started = time.perf_counter()
        with torch.no_grad():
            outputs = self.tracker(**inputs)
        logger.debug("sam3 tracker inference %.3fs", time.perf_counter() - started)

        masks = self.tracker_processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], mask_threshold=mask_threshold, binarize=True
        )

        # The tracker returns 3 candidate masks; take the highest predicted-IoU one.
        scores = outputs.iou_scores[0][0]
        best = scores.argmax().item()

        mask = masks[0][0][best].cpu().numpy().astype(bool)
        return mask, float(scores[best])

    def predict(self, image, mask_threshold=0.0):
        """Segment one organism out of an image.

        Args:
            image: PIL RGB image or RGB array.
            mask_threshold: Logit cutoff: 0.0 is neutral, negative grows the mask, positive
                shrinks it. On the text path it is converted to the probability SAM3's
                post-processing takes, so 0.0 is SAM3's own 0.5.

        Returns:
            `(mask, score, info)`: a boolean mask the size of the image, the instance score
            (text) or predicted IoU (points), and diagnostics naming the prompting path.

        Raises:
            ValueError: On the text path, when no instance clears `score_threshold`. There is
                no fallback to points, which would store a mask the recipe doesn't describe.
        """
        array = np.asarray(image)
        height, width = array.shape[:2]
        info = {"detect_bounds": self.detect_bounds, "retried": False}

        if self.detect_bounds:
            found = self.instances(image, mask_threshold=mask_threshold)
            if not found:
                raise ValueError(
                    f"SAM3 found nothing matching {self.text_prompt!r} "
                    f"above score_threshold={self.score_threshold}"
                )
            (mask, box, score), runner_up = found[0], found[1:2]
            info["prompt"] = "text"
            info["box"] = box
            info["box_score"] = score
            # The same keys GroundedSAM2 reports, so a filter on a second
            # organism in the frame carries over when the segmenter is swapped.
            # second_box_score is 0.0 rather than missing when there is none,
            # since a missing value never passes an export filter.
            info["n_boxes"] = len(found)
            info["second_box_score"] = runner_up[0][2] if runner_up else 0.0
            info["second_box_iou"] = _box_iou(box, runner_up[0][1]) if runner_up else None
            return mask, score, info

        points, labels = prompt_points(width, height, self.use_center_point, self.use_corner_points)
        mask, score = self._predict_points(image, points, labels, mask_threshold=mask_threshold)
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
            points, labels = prompt_points(width, height, False, self.use_corner_points)
            mask, score = self._predict_points(image, points, labels, mask_threshold=mask_threshold)
            info["retried"] = True
            info["points"] = points
            info["labels"] = labels

        return mask, score, info

    def visualize(self, segment, image, mask, score, info):
        """Return the image with the mask tinted, the prompt points and the instance's box drawn."""
        panel = overlay_mask(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), mask, alpha=0.4)

        for (x, y), label in zip(info.get("points", []), info.get("labels", [])):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(panel, (int(x), int(y)), 5, color, -1)

        box = info.get("box")
        if box:
            cv2.rectangle(panel, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 128, 0), 2)

        annotate(panel, f"sam3 {info['prompt']} score {score:.3f}{'  RETRIED' if info['retried'] else ''}")
        segment.emit_panel(panel, "segment")


def _from_pretrained(cls, name):
    """Load `cls` from a checkpoint, explaining a gated-repo refusal instead of passing on a bare 401.

    transformers re-raises the hub's `GatedRepoError` as a plain `OSError`, keeping the original
    as its cause, so the chain is searched rather than the class caught.
    """
    try:
        return cls.from_pretrained(name)
    except OSError as err:
        gated = _gated_cause(err)
        if gated is None:
            raise
        response = getattr(gated, "response", None)
        status = getattr(response, "status_code", None)
        page = f"https://huggingface.co/{name}"
        if status == 403:
            reason = (
                f"you are logged in, but this account has not been granted access to {name} yet. "
                f"Request it at {page} (fill in the form and accept the license); requests and their "
                "status are listed at https://huggingface.co/settings/gated-repos."
            )
        else:
            reason = (
                f"no Hugging Face login was sent. Request access at {page} if you have not, create a "
                "read token at https://huggingface.co/settings/tokens, then run `hf auth login` "
                "(or set HF_TOKEN in the environment, e.g. on a cluster). `hf auth whoami` checks it."
            )
        raise RuntimeError(f"cannot load {name}, a gated checkpoint: {reason}") from err


def _gated_cause(err):
    """Return the `GatedRepoError` in an exception's cause chain, or None."""
    try:
        from huggingface_hub.errors import GatedRepoError
    except ImportError:
        return None

    seen = set()
    while err is not None and id(err) not in seen:
        if isinstance(err, GatedRepoError):
            return err
        seen.add(id(err))
        err = err.__cause__ or err.__context__
    return None


def logit_to_probability(logit):
    """Return the probability a logit cutoff corresponds to, so 0.0 is 0.5.

    `segment(mask_threshold=)` is a logit everywhere; SAM3's instance post-processing
    thresholds probabilities, so the cutoff is converted rather than reinterpreted.
    """
    return 1.0 / (1.0 + math.exp(-logit))


def sam3(**kwargs):
    """Return a `SAM3` model for `segment()`, finding the organism from a text prompt.

    Args:
        **kwargs: As in `SAM3`.
    """
    return SAM3(**kwargs)


def sam3_points(**kwargs):
    """Return SAM3 prompted with points, for images already cropped around one organism.

    Args:
        **kwargs: As in `SAM3`, apart from `detect_bounds`.
    """
    kwargs.setdefault("detect_bounds", False)
    return SAM3(**kwargs)
