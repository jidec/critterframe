"""resolve_device(): which device a loaded network should run on."""

import logging

logger = logging.getLogger(__name__)


def resolve_device(device=None):
    """Return the device to load a network onto: CUDA where available, else CPU.

    Args:
        device: An explicit device string, returned unchanged.
    """
    if device is not None:
        return device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
