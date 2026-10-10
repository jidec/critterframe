"""Embeddings: a feature vector from any torch network, stored as a metric."""

import hashlib
import logging
from pathlib import Path

import numpy as np

from ..devices import resolve_device
from ..core.recipes import Metric, hash_spec
from ..records.models import fingerprint_file

logger = logging.getLogger(__name__)

# Default input size, the 224px ImageNet-pretrained backbones are trained at.
DEFAULT_INPUT_SIZE = (224, 224)

# Per-channel (mean, std) in 0-1 RGB that ImageNet-pretrained backbones were trained on.
IMAGENET_MEAN_STD = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

RESIZE_MODES = ("stretch", "pad")


class EmbeddingModel:
    """A network mapping an image batch to feature vectors, in the form `embedding()` needs.

    Calls the network's `encode()` if it has one, else the network itself, as
    `FloatTensor[N, 3, H, W] -> FloatTensor[N, D]`. Input is RGB scaled to 0-1 at
    `input_size`. A classifier returns logits, so pass a feature extractor.

    Args:
        model: A network with `encode()`, or a callable one.
        checkpoint: Path or identifier of the weights. A local file's content digest, or
            the string itself otherwise, reaches the recipe hash.
        input_size: `(height, width)` the network expects.
        normalize: L2-normalize embeddings before storing.
        device: Torch device string; CUDA where available if None.
        mean: Per-channel RGB mean to standardize by, on a 0-1 scale.
        std: Per-channel RGB standard deviation to standardize by.
        head: Label for which output is embedded, e.g. `"projection"`.
        resize: `"stretch"` resizes straight to `input_size`; `"pad"` letterboxes onto a
            black square first, keeping the aspect ratio.
    """

    # What identity() calls this class. A subclass keeps its own, so its
    # recorded identity (and every recipe hash built on it) survives a refactor.
    identity_class = "EmbeddingModel"
    identity_version = "1"

    def __init__(
        self,
        model,
        checkpoint,
        input_size=DEFAULT_INPUT_SIZE,
        normalize=True,
        device=None,
        mean=None,
        std=None,
        head=None,
        resize="stretch",
    ):
        if not hasattr(model, "encode") and not callable(model):
            raise TypeError(
                "an embedding model needs an encode(images) -> embeddings "
                "method or must itself be callable (a torch Module's forward)"
            )
        if resize not in RESIZE_MODES:
            raise ValueError(f"resize must be one of {RESIZE_MODES}, got {resize!r}")
        self.model = model
        self.checkpoint = checkpoint
        self.input_size = tuple(input_size)
        self.normalize = normalize
        self.mean = None if mean is None else tuple(mean)
        self.std = None if std is None else tuple(std)
        self.head = head
        self.resize = resize
        self._device = device
        self._fingerprint = None
        self._prepared = False
        self._call = None

    def identity(self):
        """Return what this model contributes to a recipe hash: its weights' digest and preprocessing."""
        identity = {
            "class": self.identity_class,
            "version": self.identity_version,
            "checkpoint": self.fingerprint,
            "input_size": list(self.input_size),
            "normalize": self.normalize,
        }
        # Only when set, so recipe hashes from before these existed hold.
        for key in ("mean", "std"):
            if getattr(self, key) is not None:
                identity[key] = list(getattr(self, key))
        if self.head is not None:
            identity["head"] = self.head
        if self.resize != "stretch":
            identity["resize"] = self.resize
        return identity

    @classmethod
    def from_checkpoint(cls, checkpoint, model, **kwargs):
        """Wrap an already-loaded network, with the checkpoint as the first argument.

        Args:
            checkpoint: As in `EmbeddingModel`.
            model: As in `EmbeddingModel`.
            **kwargs: The other `EmbeddingModel` arguments.
        """
        return cls(model, checkpoint, **kwargs)

    @property
    def fingerprint(self):
        """Return the checkpoint's content digest, read once."""
        if self._fingerprint is None and self.checkpoint is not None:
            path = Path(self.checkpoint)
            self._fingerprint = fingerprint_file(path) if path.exists() else str(self.checkpoint)
        return self._fingerprint

    @property
    def device(self):
        """Return the torch device, resolved on first use."""
        if self._device is None:
            self._device = resolve_device()
        return self._device

    def _prepare(self):
        """Move the model to its device, set eval mode, and pick `encode()` or `forward()`, once."""
        if self._prepared:
            return
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        # Bound after .to(), which may return a different object.
        self._call = self.model.encode if hasattr(self.model, "encode") else self.model
        self._prepared = True

    def preprocess(self, image):
        """Return one RGB image as the float32 HxWx3 array the network is fed.

        Args:
            image: RGB array of any size.
        """
        import cv2

        height, width = self.input_size
        image = np.asarray(image)
        if self.resize == "pad":
            image = letterbox(image)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        array = resized.astype(np.float32) / 255.0
        if self.mean is not None:
            array = array - np.asarray(self.mean, dtype=np.float32)
        if self.std is not None:
            array = array / np.asarray(self.std, dtype=np.float32)
        return array

    def embed(self, image):
        """Embed one RGB image, and return a 1D NumPy array.

        Args:
            image: RGB array of any size.
        """
        import torch

        self._prepare()
        array = self.preprocess(image)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._call(tensor)
        embedding = np.asarray(
            output[0].detach().cpu().numpy() if hasattr(output, "detach") else output[0], dtype=np.float32
        ).reshape(-1)

        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding


def letterbox(image):
    """Return the image centered on a black square as large as its longer side.

    Args:
        image: HxW or HxWxC array.
    """
    height, width = image.shape[:2]
    side = max(height, width)
    canvas = np.zeros((side, side) + image.shape[2:], dtype=image.dtype)
    top, left = (side - height) // 2, (side - width) // 2
    canvas[top : top + height, left : left + width] = image
    return canvas


def state_dict_digest(network):
    """Return a short digest of a loaded network's weights, in parameter-name order.

    Args:
        network: Anything with `state_dict()`.
    """
    digest = hashlib.sha256()
    for key, tensor in network.state_dict().items():
        digest.update(key.encode("utf-8"))
        value = tensor.detach().cpu() if hasattr(tensor, "detach") else tensor
        digest.update(np.ascontiguousarray(np.asarray(value)).tobytes())
    return hash_spec({"state_dict": digest.hexdigest()})


def pretrained(
    name,
    pretrained=True,
    input_size=None,
    mean=None,
    std=None,
    normalize=True,
    resize="stretch",
    device=None,
    **create_kwargs,
):
    """Return a timm backbone as a pooled feature extractor, ready for `embedding()`.

    Built with `num_classes=0`, which removes the classifier. Preprocessing defaults to
    what timm says the weights were trained with.

    Args:
        name: timm model name, e.g. `"resnet18"`, `"vit_small_patch14_dinov2.lvd142m"`.
        pretrained: Load timm's pretrained weights.
        input_size: Override timm's input size.
        mean: Override timm's standardization mean.
        std: Override timm's standardization standard deviation.
        normalize: L2-normalize embeddings before storing.
        resize: `"stretch"` or `"pad"`, as in `EmbeddingModel`.
        device: Torch device string; CUDA where available if None.
        **create_kwargs: Passed to `timm.create_model`.
    """
    import timm
    from timm.data import resolve_data_config

    network = timm.create_model(name, pretrained=pretrained, num_classes=0, **create_kwargs)
    network.eval()
    config = resolve_data_config({}, model=network)

    return EmbeddingModel(
        network,
        state_dict_digest(network),
        input_size=tuple(input_size or config["input_size"][1:]),
        normalize=normalize,
        device=device,
        mean=mean if mean is not None else config.get("mean"),
        std=std if std is not None else config.get("std"),
        head=f"timm:{name}",
        resize=resize,
    )


def embedding(model, name=None, unit="embedding"):
    """Metric: a learned embedding of the segment, stored as a list of floats.

    Args:
        model: An `EmbeddingModel`, a `RegisteredModel` wrapping one, or anything with the
            same `embed()` and `identity()`.
        name: Metric name; `"embedding"` if None.
        unit: Recorded unit.
    """
    return Metric("embedding", _embedding, version="1", unit=unit, metric_name=name, model=model)


def _embedding(segment, model):
    """Embed the segment's image and return the vector as a list."""
    segment.require_mask()
    vector = model.embed(segment.rgb)
    return [float(value) for value in vector]
