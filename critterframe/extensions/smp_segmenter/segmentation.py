"""
A trainable segmenter: UNet++ (segmentation_models_pytorch) over a swappable
encoder, EfficientNet-B7 by default -- for refining GroundedSAM2's output on
one project's own masks (removing wings, isolating a body part) or for
segmenting where the bundled zero-shot model fails outright.

Torch and segmentation_models_pytorch are imported lazily, so constructing a
model and reading its identity() work without the [torch] extra installed --
the same discipline segmentation.groundedsam.GroundedSAM2 follows. See
extensions.smp_segmenter.training for training one from a project's own masks.
"""

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
    """
    A segmentation_models_pytorch UNet++ segmenter.

    Meets the same predict()/identity()/visualize() contract as GroundedSAM2,
    so it works with segment() directly. For checkpoint-fingerprint provenance
    (recommended for anything trained on real project data) wrap it instead:

        model = cf.load_model(project_path, "aux_segmenter_v1").attach(
            smp_segmenter(checkpoint=checkpoint_path))
        cf.run_segments(project_path, steps=[cf.segment(model)])

    - `checkpoint` -- path to a `state_dict` saved by `training.train()`, or
      None for a freshly-initialized (ImageNet-encoder) network --
      meaningful only while training; `segment()` needs a trained
      checkpoint to produce anything useful.
    - `encoder_name` -- `segmentation_models_pytorch` encoder, e.g.
      `"efficientnet-b7"`. Must match what the checkpoint was trained with
      -- a mismatch loads silently wrong weights shape-for-shape until it
      doesn't.
    - `size` -- side length the image is resized to before inference. Must
      match what `training.train()` used.
    - `device` -- torch device string; autodetects CUDA if omitted.
    """

    def __init__(self, checkpoint=None, encoder_name=DEFAULT_ENCODER,
                 size=DEFAULT_SIZE, device=None):
        self.checkpoint = checkpoint
        self.encoder_name = encoder_name
        self.size = size
        self._device = device
        self._fingerprint = None
        self.model = None

    def identity(self):
        """
        What this model contributes to a recipe hash: architecture, encoder,
        working size, and the checkpoint's CONTENT.

        The threshold isn't here: it belongs to `segment()`, which already
        hashes it, and it means a logit to this model exactly as it does to
        GroundedSAM2 (see `predict`).

        The fingerprint, not the path, for the reason a `RegisteredModel`
        keeps the path out of its own identity: two copies of one checkpoint
        are the same model, and two different checkpoints written to one path
        over a project's life are not. Read once and cached -- `Recipe.hash`
        is asked many times per run and a checkpoint is hundreds of megabytes
        -- so REPLACING the file under a live object keeps the old identity;
        build a new segmenter after retraining.
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
        """
        The checkpoint's content digest, read once (see `identity`).

        A path that isn't there yet is recorded as the string it is: a
        freshly-built training network has no weights to hash, and hashing a
        missing directory would quietly answer "empty" for every one of them.
        """
        if self._fingerprint is None and self.checkpoint is not None:
            path = Path(self.checkpoint)
            self._fingerprint = (fingerprint_file(path) if path.exists()
                                 else str(self.checkpoint))
        return self._fingerprint

    @classmethod
    def from_checkpoint(cls, checkpoint, **kwargs):
        """
        This segmenter over a checkpoint on disk -- the plain constructor,
        named so it pairs with `load_registered` below.
        """
        return cls(checkpoint=checkpoint, **kwargs)

    @property
    def device(self):
        if self._device is None:
            self._device = resolve_device()
        return self._device

    def _load(self):
        """Build the network and load its weights once, on first use."""
        if self.model is not None:
            return

        import segmentation_models_pytorch as smp
        import torch

        logger.info("loading %s(%s) checkpoint=%s on %s", "unetplusplus",
                   self.encoder_name, self.checkpoint, self.device)
        self.model = smp.UnetPlusPlus(
            encoder_name=self.encoder_name,
            # Skip downloading ImageNet weights when a checkpoint is about to
            # overwrite them anyway; keep them for a fresh (training) network.
            encoder_weights="imagenet" if self.checkpoint is None else None,
            in_channels=3, classes=1,
        ).to(self.device)

        if self.checkpoint is not None:
            state = torch.load(self.checkpoint, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, image, mask_threshold=0.0):
        """
        Segment one organism (or part) out of an image.

        - `image` -- RGB array, the convention every segmenter in this
          package uses.
        - `mask_threshold` -- a LOGIT cutoff, the same space GroundedSAM2
          uses and the space `segment()` documents: 0.0 is neutral (logit 0
          is probability 0.5), negative grows the mask, positive shrinks it.

        Returns (mask, score, info): score is the mean predicted probability
        over the kept mask, or None if the mask came out empty.
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

        info = {"encoder": self.encoder_name, "mask_threshold": mask_threshold,
                "threshold_space": "logit"}
        score = (float(np.mean(1.0 / (1.0 + np.exp(-logits_full[mask]))))
                 if mask.any() else None)
        return mask, score, info

    def visualize(self, segment, image, mask, score, info):
        """The predicted mask tinted over the frame -- the same drawing convention every segmenter here uses."""
        panel = overlay_mask(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
                             mask, alpha=0.4)
        label = (f"{self.encoder_name} score {score:.3f}" if score is not None
                else f"{self.encoder_name} (empty)")
        annotate(panel, label)
        segment.emit_panel(panel, "segment")


def smp_segmenter(**kwargs):
    """
    An SMPSegmenter configured for segment().

    checkpoint=None (the default) builds a fresh ImageNet-initialized network,
    useful only while training (see training.train()); pass the checkpoint
    path train() returned to segment with a trained one. See SMPSegmenter for
    every parameter.
    """
    return SMPSegmenter(**kwargs)


def load_registered(project_path, name):
    """
    A model registered via records.models.register_model(), loaded and ready
    to run.

    Delegates to records.models.load_and_attach(), the shared load-record/
    pull-parameters/attach sequence every registered model type uses. It
    reads `encoder_name`/`size` back from the registry's own `parameters`,
    recorded once at registration time (see extensions.smp_segmenter.training's
    caller), rather than this module's current DEFAULT_ENCODER/DEFAULT_SIZE --
    so a later change to those module defaults can never silently swap the
    architecture under an already-trained checkpoint. A model registered
    before `parameters` carried them falls back to SMPSegmenter's own
    constructor defaults, which are these same module constants.

    - `project_path` -- project the model is registered in.
    - `name` -- registered model name.

    Returns a RegisteredModel with an SMPSegmenter attached.
    """
    return load_and_attach(project_path, name, SMPSegmenter.from_checkpoint)
