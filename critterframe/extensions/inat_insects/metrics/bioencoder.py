"""
BioEncoder embeddings as a metric.

A metric is any derived value associated with an occurrence and a part, and an
embedding is exactly that -- a learned vector summarizing what an organism
looks like, stored beside body length and mean lightness rather than in some
separate representation store. That's the whole reason this fits: metric values
are JSON, so a 128-float vector stores as readily as a scalar, and it exports
and filters like anything else.

Embeddings earn their keep where hand-designed traits run out. Wing venation,
subtle pattern differences, the overall gestalt that separates two similar
species -- things nobody has written a measurement for. What you get is a
vector that's only meaningful in comparison: distances between embeddings mean
something, the individual numbers don't.

Expensive to compute, which is precisely why repeat-awareness matters here more
than anywhere else. An embedding run over 50,000 images is a GPU-hours job, and
CritterFrame's recipe hashing makes the result behave like cached derived data
-- rerun the same recipe and nothing is recomputed; change the checkpoint and
everything is, because the checkpoint is in the hash.

SCAFFOLD. This defines the model contract and the metric around it; it is NOT
backed by a trained checkpoint. Train one with training.bioencoder, or load a
BioEncoder checkpoint of your own, and pass it in. What's fixed here is the
interface, so the rest of the pipeline can be written against it now.
"""

import logging

import numpy as np

from ....recipes import Metric

logger = logging.getLogger(__name__)

# Default input size. BioEncoder-style metric learning models are typically
# trained at 224px, following the ImageNet-pretrained backbones they start from.
DEFAULT_INPUT_SIZE = (224, 224)


class BioEncoderModel:
    """
    Wrapper fixing what an embedding model has to provide.

    The contract:

        encode(images: FloatTensor[N, 3, H, W]) -> FloatTensor[N, D]

    Images are RGB scaled to 0-1 at input_size. D is the embedding dimension.
    Whether the model is a BioEncoder checkpoint, a fine-tuned timm backbone, or
    something else entirely doesn't matter to anything downstream.

    model          -- an object with encode() as above (usually a
                      torch.nn.Module).
    checkpoint     -- path or identifier of the weights loaded. Goes into
                      identity() and therefore into the recipe hash, which is
                      what makes "these embeddings came from that checkpoint" a
                      recorded fact rather than something to remember.
    input_size     -- (height, width) the model was trained at.
    normalize      -- L2-normalize embeddings before storing. On by default:
                      metric-learning models are trained with cosine distance,
                      so unnormalized vectors carry a magnitude that means
                      nothing and distorts any Euclidean comparison made later.
    device         -- torch device string; autodetects CUDA if omitted.
    """

    def __init__(self, model, checkpoint, input_size=DEFAULT_INPUT_SIZE,
                 normalize=True, device=None):
        if not hasattr(model, "encode"):
            raise TypeError(
                "a BioEncoder model needs an encode(images) -> embeddings "
                "method; wrap your network in a small class providing one"
            )
        self.model = model
        self.checkpoint = checkpoint
        self.input_size = tuple(input_size)
        self.normalize = normalize
        self._device = device
        self._prepared = False

    def identity(self):
        """
        What this model contributes to a recipe hash. The checkpoint is the
        important part: two embedding sets from different checkpoints aren't
        comparable at all -- not even approximately, since the embedding spaces
        are unrelated -- so they must never be mistaken for equivalent work.
        """
        return {
            "class": "BioEncoderModel",
            "checkpoint": str(self.checkpoint),
            "input_size": list(self.input_size),
            "normalize": self.normalize,
        }

    @property
    def device(self):
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _prepare(self):
        """Move the model to its device and put it in eval mode, once."""
        if self._prepared:
            return
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._prepared = True

    def embed(self, image):
        """
        Embed one RGB image array. Returns a 1D numpy array.

        image -- RGB array; resized to input_size here, so callers don't each
                 have to know what the model expects.
        """
        import cv2
        import torch

        self._prepare()

        height, width = self.input_size
        resized = cv2.resize(np.asarray(image), (width, height),
                             interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(
            resized.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode(tensor)[0].cpu().numpy()

        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding


def embedding(model, name=None, unit="embedding"):
    """
    Metric: a learned embedding of the segment, stored as a list of floats.

    Put remove_background() in the recipe's transforms before this. An
    iNaturalist photograph's background is uncontrolled, and an embedding
    computed over the whole frame will encode the leaf the animal was sitting
    on as readily as the animal -- which then shows up as two individuals of one
    species being far apart in embedding space because they were photographed
    against different substrates.

    model -- a BioEncoderModel (or anything with the same embed()/identity()).
    """
    return Metric("embedding", _embedding, version="1", unit=unit,
                  metric_name=name, model=model)


def _embedding(segment, model):
    """Embed the segment's current image and return the vector as a plain list."""
    segment.require_mask()
    vector = model.embed(segment.rgb)
    return [float(value) for value in vector]
