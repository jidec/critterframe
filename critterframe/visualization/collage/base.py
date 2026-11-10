"""The configured step every collage factory returns: a name, its parameters, and a spec over both."""

import hashlib

import cv2
import numpy as np


class Step:
    """One configured collage step.

    Args:
        name: The factory's name, e.g. `"sort_by"`.
        parameters: Its JSON-serializable settings.
        columns: Table columns it reads; `{part}` in a name stands for each item's part.
    """

    kind = "step"

    def __init__(self, name, parameters=None, columns=()):
        self.name = name
        self.parameters = dict(parameters or {})
        self.columns = tuple(columns)

    def spec(self):
        """Return the step's kind, name and parameters."""
        return {"kind": self.kind, "name": self.name, "parameters": self.parameters}

    def __repr__(self):
        return f"{type(self).__name__}({self.name}, {self.parameters})"


def array_digest(array):
    """Return a short digest of an array's shape, type and bytes."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{array.shape}{array.dtype}".encode() + array.tobytes()).hexdigest()
    return digest[:16]


def read_image(image, flags=cv2.IMREAD_UNCHANGED):
    """Return an image given as an array or a path, as an array.

    Args:
        image: An array, or a path cv2 can read.
        flags: cv2 read flags for a path.
    """
    if isinstance(image, np.ndarray):
        return image
    array = cv2.imread(str(image), flags)
    if array is None:
        raise FileNotFoundError(f"could not read an image from {image}")
    return array
