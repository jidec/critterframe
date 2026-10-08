"""SMPSegmenter: a trainable UNet++ segmenter over a swappable encoder."""

import logging
from pathlib import Path

import cv2
import numpy as np

from ...devices import resolve_device
from ...records.models import fingerprint_file, load_and_attach
from ...visualization.panels import annotate, overlay_mask

logger = logging.getLogger(__name__)

DEFAULT_ENCODER = "efficientnet-b7"
DEFAULT_SIZE = 352


class SMPSegmenter:
    """A segmentation_models_pytorch UNet++ segmenter, usable with `segment()`.

    Args:
        checkpoint: Path to a `state_dict` saved by `training.train()`. None builds an
            untrained network, which is only useful for training.
        encoder_name: The segmentation_models_pytorch encoder, e.g. `"efficientnet-b7"`.
            Must match what the checkpoint was trained with.
        size: Side length the image is resized to; must match training.
        device: Torch device string; CUDA where available if None.
    """

    def __init__(self, checkpoint=None, encoder_name=DEFAULT_ENCODER, size=DEFAULT_SIZE, device=None):
        self.checkpoint = checkpoint
        self.encoder_name = encoder_name
        self.size = size
        self._device = device
        self._fingerprint = None
        self.model = None

    def identity(self):
        """Return what this model contributes to a recipe hash: architecture, encoder, size and weights.

        The weights' digest is read once and cached, so build a new segmenter after
        replacing a checkpoint file.
        """
        return {
            "class": "SMPSegmenter",
            "architecture": "unetplusplus",
            "encoder": self.encoder_name,
            "size": self.size,
            "version": "2",
            "checkpoint": self.fingerprint,
        }

    @property
    def fingerprint(self):
        """Return the checkpoint's content digest, read once; a missing path is returned as the string."""
        if self._fingerprint is None and self.checkpoint is not None:
            path = Path(self.checkpoint)
            self._fingerprint = fingerprint_file(path) if path.exists() else str(self.checkpoint)
        return self._fingerprint

    @classmethod
    def from_checkpoint(cls, checkpoint, **kwargs):
        """Build a segmenter over a checkpoint on disk.

        Args:
            checkpoint: As in `SMPSegmenter`.
            **kwargs: The other `SMPSegmenter` arguments.
        """
        return cls(checkpoint=checkpoint, **kwargs)

    @property
    def device(self):
        """Return the torch device, resolved on first use."""
        if self._device is None:
            self._device = resolve_device()
        return self._device

    def _load(self):
        """Build the network and load its weights, on first use."""
        if self.model is not None:
            return

        import segmentation_models_pytorch as smp
        import torch

        logger.info(
            "loading %s(%s) checkpoint=%s on %s",
            "unetplusplus",
            self.encoder_name,
            self.checkpoint,
            self.device,
        )
        self.model = smp.UnetPlusPlus(
            encoder_name=self.encoder_name,
            # Skip downloading ImageNet weights when a checkpoint is about to
            # overwrite them anyway; keep them for a fresh (training) network.
            encoder_weights="imagenet" if self.checkpoint is None else None,
            in_channels=3,
            classes=1,
        ).to(self.device)

        if self.checkpoint is not None:
            state = torch.load(self.checkpoint, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, image, mask_threshold=0.0):
        """Segment one organism or part out of an image.

        Args:
            image: RGB array.
            mask_threshold: Logit cutoff: 0.0 is neutral, negative grows the mask, positive
                shrinks it.

        Returns:
            `(mask, score, info)`. `score` is the mean predicted probability over the mask,
            or None if the mask is empty.
        """
        import torch

        self._load()
        array = np.asarray(image)
        height, width = array.shape[:2]

        resized = cv2.resize(array, (self.size, self.size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)[0, 0].cpu().numpy()

        # Thresholded in logit space, so segment(mask_threshold=...) means one
        # thing across segmenters. Read as a probability, segment()'s neutral
        # 0.0 would keep every pixel whose sigmoid exceeds zero -- the whole
        # frame -- rather than every pixel the model calls organism.
        logits_full = cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = logits_full > mask_threshold

        info = {"encoder": self.encoder_name, "mask_threshold": mask_threshold, "threshold_space": "logit"}
        score = float(np.mean(1.0 / (1.0 + np.exp(-logits_full[mask])))) if mask.any() else None
        return mask, score, info

    def visualize(self, segment, image, mask, score, info):
        """Return the predicted mask tinted over the frame."""
        panel = overlay_mask(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), mask, alpha=0.4)
        label = (
            f"{self.encoder_name} score {score:.3f}" if score is not None else f"{self.encoder_name} (empty)"
        )
        annotate(panel, label)
        segment.emit_panel(panel, "segment")


def smp_segmenter(**kwargs):
    """Return an `SMPSegmenter` for `segment()`.

    Args:
        **kwargs: As in `SMPSegmenter`.
    """
    return SMPSegmenter(**kwargs)


def load_registered(project_path, name):
    """Load a registered model, ready to run.

    `encoder_name` and `size` come from the registry's `parameters`, falling back to
    `SMPSegmenter`'s defaults for a model registered without them.

    Args:
        project_path: Project the model is registered in.
        name: Registered model name.

    Returns:
        A `RegisteredModel` with an `SMPSegmenter` attached.
    """
    return load_and_attach(project_path, name, SMPSegmenter.from_checkpoint)
