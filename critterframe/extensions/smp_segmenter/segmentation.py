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

import cv2
import numpy as np

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
        self.model = None

    def identity(self):
        """
        What this model contributes to a recipe hash: architecture, encoder,
        working size, and the checkpoint path.

        A bare path, not a fingerprint -- used standalone (not wrapped in a
        RegisteredModel) two different checkpoints saved to the same path
        over time would collide in the hash, the same tradeoff
        inat_insects.metrics.bioencoder.BioEncoderModel makes. Wrap in
        records.models.register_model()/load_model() for a hash that moves
        when the weights do.
        """
        return {
            "class": "SMPSegmenter",
            "architecture": "unetplusplus",
            "encoder": self.encoder_name,
            "size": self.size,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
        }

    @property
    def device(self):
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
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

    def predict(self, image, mask_threshold=0.5):
        """
        Segment one organism (or part) out of an image.

        - `image` -- RGB array, the convention every segmenter in this
          package uses.
        - `mask_threshold` -- a PROBABILITY cutoff, unlike GroundedSAM2's
          logit threshold: a pixel is kept where the model's sigmoid output
          exceeds this. 0.5 is neutral.

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
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        probs_full = cv2.resize(probs, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = probs_full > mask_threshold

        info = {"encoder": self.encoder_name, "mask_threshold": mask_threshold}
        score = float(probs_full[mask].mean()) if mask.any() else None
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
    from ...records.models import load_and_attach

    return load_and_attach(project_path, name, SMPSegmenter)
