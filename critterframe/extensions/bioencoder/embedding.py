"""
BioEncoder embeddings as a metric: `metrics.embedding` over a BioEncoder-package checkpoint.

No checkpoint ships with it. Train one with the BioEncoder package on a dataset
from `training.prepare_dataset()`, then load it with `training.load()`,
`training.load_from_config()` or `load_registered()` below.
"""

import logging

from ...metrics.embedding import (
    DEFAULT_INPUT_SIZE,
    IMAGENET_MEAN_STD,
    EmbeddingModel,
    embedding,
)
from ...records.models import load_and_attach

__all__ = ["BioEncoderModel", "DEFAULT_INPUT_SIZE", "IMAGENET_MEAN_STD",
           "embedding", "load_registered"]

logger = logging.getLogger(__name__)


class BioEncoderModel(EmbeddingModel):
    """
    An `EmbeddingModel` that records itself as a BioEncoder model.

    Identical in behaviour; kept as its own class so every recipe hash recorded
    under the `"BioEncoderModel"` identity stays valid. See `EmbeddingModel`
    for the arguments.
    """

    identity_class = "BioEncoderModel"
    identity_version = "2"


def load_registered(project_path, name, model=None):
    """
    A model registered via `records.models.register_model()`, loaded and ready
    to pass to `embedding()`.

    Construction settings are read back from the registry's own `parameters`,
    never from this module's current defaults.

    - `project_path` -- project the model is registered in.
    - `name` -- registered model name.
    - `model` -- None (default) builds a BioEncoder-package checkpoint with
      `training.load()`, which needs `backbone` in `parameters`. Pass a
      network you loaded yourself for anything else; `parameters` then holds
      only `BioEncoderModel`'s own arguments.

    Returns a RegisteredModel with a BioEncoderModel attached.
    """
    if model is None:
        from .training import load

        return load_and_attach(project_path, name, load)
    return load_and_attach(project_path, name, BioEncoderModel.from_checkpoint,
                           model=model)
