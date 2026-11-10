"""BioEncoder embeddings as a metric: `metrics.embedding` over a BioEncoder-package checkpoint."""

from ...metrics.embedding import (
    DEFAULT_INPUT_SIZE,
    IMAGENET_MEAN_STD,
    EmbeddingModel,
    embedding,
)
from ...records.models import load_and_attach

__all__ = ["BioEncoderModel", "DEFAULT_INPUT_SIZE", "IMAGENET_MEAN_STD", "embedding", "load_registered"]

# Registry parameters that only `training.load()` reads, to build the network.
_NETWORK_PARAMETERS = ("backbone", "projection_head")


class BioEncoderModel(EmbeddingModel):
    """An `EmbeddingModel` that identifies itself as a BioEncoder model in recipe hashes.

    Takes the same arguments as `EmbeddingModel`.
    """

    identity_class = "BioEncoderModel"
    identity_version = "2"


def head_label(projection_head):
    """Return the `EmbeddingModel` head label for a BioEncoder network's `projection_head` setting."""
    return "projection" if projection_head else "features"


def load_registered(project_path, name, model=None):
    """Load a registered model, ready to pass to `embedding()`.

    Construction settings come from the registry's `parameters`, as `training.register_trained()`
    records them.

    Args:
        project_path: Project the model is registered in.
        name: Registered model name.
        model: None builds a BioEncoder-package checkpoint with `training.load()`, which
            needs `backbone` in `parameters`. Otherwise a network you loaded yourself, with
            the projection head already set as `projection_head` records.

    Returns:
        A `RegisteredModel` with a `BioEncoderModel` attached.
    """
    if model is None:
        from .training import load

        return load_and_attach(project_path, name, load)
    return load_and_attach(project_path, name, _wrap, model=model)


def _wrap(checkpoint, model, **parameters):
    """Wrap a loaded network from registry parameters, leaving out those only `training.load()` reads."""
    kwargs = {key: value for key, value in parameters.items() if key not in _NETWORK_PARAMETERS}
    if "projection_head" in parameters:
        kwargs.setdefault("head", head_label(parameters["projection_head"]))
    return BioEncoderModel.from_checkpoint(checkpoint, model, **kwargs)
