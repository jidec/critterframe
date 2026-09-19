"""
resolve_device(): which device a loaded network should run on.

Torch is imported inside the function, not at module scope, so core stays
importable without it -- this module is the one place that asks the question,
not a dependency on the answer.
"""

import logging

logger = logging.getLogger(__name__)


def resolve_device(device=None):
    """
    The device to load a network onto: what the caller asked for, else CUDA
    where it's available, else CPU.

    Every model class in the package resolves this the same way, and each is
    lazy about it: asking torch at construction time would make building a
    recipe -- which may never run -- import torch and initialize CUDA.

    - `device` -- an explicit device string, returned unchanged. None (the
      default) picks one.
    """
    if device is not None:
        return device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
