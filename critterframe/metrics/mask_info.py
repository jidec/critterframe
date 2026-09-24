"""
mask_info(): the diagnostics a segmentation run stored on each mask, as a metric.

Reads the `info` a mask row recorded (see `records.masks.make_mask_row`), so they can be exported and filtered
like any other value, and go stale when the mask is replaced.
"""

from .. import drivers
from ..recipes import Metric
from ..records import masks as mask_records


class MaskInfoMetric(Metric):
    """
    Metric: the measured mask's recorded diagnostics, flattened to one level.

    - `operations` -- operation labels to keep, e.g. `["orient", "segment"]`;
      every recorded one if None.
    - `name` -- what to store the value under; `"mask_info"` by default.
    """

    def __init__(self, operations=None, name=None):
        operations = None if operations is None else sorted(operations)
        super().__init__("mask_info", self._value, {"operations": operations},
                         version="1", unit="info", metric_name=name)
        self._info = {}

    def prepare(self, context):
        """Read the part's mask rows once, before any occurrence is measured."""
        rows = mask_records.mask_lookup(context.project_path, part=context.part,
                                        occurrence_ids=context.occurrence_ids,
                                        reference=context.reference)
        self._info = {occurrence_id: mask_records.mask_info(row)
                      for occurrence_id, row in rows.items()}
        return None

    def _value(self, segment, operations=None):
        info = self._info.get(segment.occurrence_id) or {}
        if operations is not None:
            info = {label: values for label, values in info.items()
                    if label in operations}
        if not info:
            raise drivers.NoInput(drivers.NO_MASK_INFO)
        return {f"{label}__{key}": value
                for label, values in sorted(info.items())
                for key, value in sorted(values.items())}


def mask_info(operations=None, name=None):
    """
    Metric: the diagnostics the segmentation run recorded on each mask, e.g.
    `orient__unreliable` or `segment__score`, one export column per key.

    A mask made before diagnostics were recorded, or with none for the chosen
    operations, counts as `no_input`.

    - `operations` -- operation labels to keep, e.g. `["orient", "segment"]`;
      every recorded one if None.
    - `name` -- what to store the value under; `"mask_info"` by default.

    Returns a configured MaskInfoMetric.
    """
    return MaskInfoMetric(operations=operations, name=name)
