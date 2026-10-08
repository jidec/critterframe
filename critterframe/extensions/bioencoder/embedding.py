"""BioEncoder embeddings as a metric: `metrics.embedding` over a BioEncoder-package checkpoint."""

import logging

from ...metrics.embedding import (
    DEFAULT_INPUT_SIZE,
    IMAGENET_MEAN_STD,
    EmbeddingModel,
    embedding,
)
from ...records.models import load_and_attach

__all__ = ["BioEncoderModel", "DEFAULT_INPUT_SIZE", "IMAGENET_MEAN_STD", "embedding", "load_registered"]

logger = logging.getLogger(__name__)


class BioEncoderModel(EmbeddingModel):
    """An `EmbeddingModel` that identifies itself as a BioEncoder model in recipe hashes.

    Takes the same arguments as `EmbeddingModel`.
    """

    identity_class = "BioEncoderModel"
    identity_version = "2"


def load_registered(project_path, name, model=None):
    """Load a registered model, ready to pass to `embedding()`.

    Construction settings come from the registry's `parameters`.

    Args:
        project_path: Project the model is registered in.
        name: Registered model name.
        model: None builds a BioEncoder-package checkpoint with `training.load()`, which
            needs `backbone` in `parameters`. Otherwise a network you loaded yourself.

    Returns:
        A `RegisteredModel` with a `BioEncoderModel` attached.
    """
    if model is None:
        from .training import load

        return load_and_attach(project_path, name, load)
    return load_and_attach(project_path, name, BioEncoderModel.from_checkpoint, model=model)
