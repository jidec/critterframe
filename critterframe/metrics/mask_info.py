"""mask_info(): the diagnostics a segmentation run stored on each mask, as a metric."""

from ..core import drivers
from ..core.recipes import Metric
from ..records import masks as mask_records


class MaskInfoMetric(Metric):
    """Metric: the measured mask's recorded diagnostics, flattened to one level.

    Args:
        operations: Operation labels to keep, e.g. `["orient", "segment"]`; all if None.
        name: Name to store the value under; `"mask_info"` if None.
    """

    def __init__(self, operations=None, name=None):
        operations = None if operations is None else sorted(operations)
        super().__init__(
            "mask_info", self._value, {"operations": operations}, version="1", unit="info", metric_name=name
        )
        self._info = {}

    def prepare(self, context):
        """Read the part's mask rows once, before any occurrence is measured."""
        rows = mask_records.mask_lookup(
            context.project_path,
            part=context.part,
            occurrence_ids=context.occurrence_ids,
            reference=context.reference,
            info=True,
        )
        self._info = {occurrence_id: mask_records.mask_info(row) for occurrence_id, row in rows.items()}
        return None

    def _value(self, segment, operations=None):
        info = self._info.get(segment.occurrence_id) or {}
        if operations is not None:
            info = {label: values for label, values in info.items() if label in operations}
        if not info:
            raise drivers.NoInput(drivers.NO_MASK_INFO)
        return {
            f"{label}__{key}": value
            for label, values in sorted(info.items())
            for key, value in sorted(values.items())
        }


def mask_info(operations=None, name=None):
    """Metric: the diagnostics the segmentation run recorded on each mask.

    Each key, e.g. `orient__unreliable` or `segment__score`, becomes an export column. A
    mask with none recorded for the chosen operations counts as `no_input`.

    Args:
        operations: Operation labels to keep, e.g. `["orient", "segment"]`; all if None.
        name: Name to store the value under; `"mask_info"` if None.
    """
    return MaskInfoMetric(operations=operations, name=name)
