"""
The pieces every driver shares: what an operation's info keeps when stored,
and the labels it is stored under.
"""

import numpy as np

from critterframe.segments import operation_labels, scalar_info
import critterframe as cf


def test_only_scalars_are_kept():
    info = {"unreliable": True, "ratio": 0.2, "n": 3, "prompt": "box",
            "missing": None, "box": [1, 2, 3, 4], "nested": {"a": 1},
            "array": np.zeros(3)}
    assert scalar_info(info) == {"unreliable": True, "ratio": 0.2, "n": 3,
                                 "prompt": "box", "missing": None}


def test_numpy_scalars_become_python_ones():
    kept = scalar_info({"flag": np.bool_(True), "ratio": np.float32(0.5),
                        "n": np.int64(4)})
    assert kept == {"flag": True, "ratio": 0.5, "n": 4}
    assert all(type(value) in (bool, float, int) for value in kept.values())


def test_no_info_is_empty():
    assert scalar_info(None) == {}


def test_a_repeated_operation_gets_a_numbered_label():
    labels = operation_labels([cf.orient(), cf.crop_to_mask(), cf.orient()])
    assert labels == ["orient", "crop_to_mask", "orient_2"]
