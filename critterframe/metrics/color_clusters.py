"""Color clustering per group: the fraction of an organism in each color of its group's palette."""

import logging

import numpy as np

from ..colorspaces import convert, normalize
from .outliers import PooledPixelGroupMetric
from .pixels import masked_pixels

logger = logging.getLogger(__name__)

# Pixels sampled per occurrence when fitting a group palette. Every masked pixel
# of a few hundred occurrences is tens of millions of points, which KMeans does
# not need -- a few thousand per occurrence captures the same distribution and
# keeps the fit to seconds.
PIXELS_PER_OCCURRENCE = 2000

# Groups with fewer reference occurrences than this share the population-wide
# palette rather than getting their own.
MIN_GROUP_SIZE = 5


class ColorClusterMetric(PooledPixelGroupMetric):
    """Group metric: the fraction of this organism in each color of its group's shared palette.

    Fractions are comparable within a group, not across groups: each group has its own palette.

    Args:
        n_colors: Palette size per group.
        group_col: Occurrence column to group by, e.g. `"taxon"`. None fits one palette for
            the whole project.
        color_space: `"lab"` or `"hsv"`.
        transforms: Operations applied when gathering the reference pixels; pass the ones
            the run uses.
        min_group_size: Groups smaller than this share the population-wide palette.
        sample_pixels: Pixels sampled per reference occurrence.
        name: Metric name.
        unit: Recorded unit.
        reference: Fit against reference masks.
    """

    def __init__(
        self,
        n_colors=5,
        group_col=None,
        color_space="lab",
        transforms=(),
        min_group_size=MIN_GROUP_SIZE,
        sample_pixels=PIXELS_PER_OCCURRENCE,
        name=None,
        unit="fraction",
        reference=False,
    ):
        if color_space not in ("lab", "hsv"):
            raise ValueError('color_space must be "lab" or "hsv"')

        super().__init__(
            "color_clusters",
            self._score,
            group_col=group_col,
            transforms=transforms,
            min_group_size=min_group_size,
            sample_pixels=sample_pixels,
            reference=reference,
            metric_name=name or "color_clusters",
            unit=unit,
        )

        self.n_colors = n_colors
        self.color_space = color_space

    def spec(self):
        """Return the operation's spec, with the palette settings and transforms."""
        spec = super().spec()
        spec["parameters"] = {
            "n_colors": self.n_colors,
            "group_col": self.group_col,
            "color_space": self.color_space,
            "min_group_size": self.min_group_size,
            "sample_pixels": self.sample_pixels,
            "reference": self.reference,
            "transforms": [operation.spec() for operation in self.transforms],
        }
        return spec

    def prepare(self, context):
        """Fit one palette per group from pooled masked pixels, plus a population-wide fallback.

        Returns:
            The base class's fit record, plus this metric's settings.
        """
        record = super().prepare(context)
        logger.info(
            "%s fit: %d group palette(s) of %d colours + 1 population-wide fallback",
            self.metric_name,
            len(self.fits) - 1,
            self.n_colors,
        )
        return dict(record, color_space=self.color_space, n_colors=self.n_colors)

    def _convert(self, pixels):
        """Return BGR pixels in the working color space, as float."""
        converted = convert(pixels, self.color_space)
        # Lab stays in true units, where Euclidean distance approximates perceived difference; hsv is rescaled
        # so hue in degrees does not swamp saturation and value in KMeans.
        return normalize(converted, "hsv") if self.color_space == "hsv" else converted

    def _pixels(self, segment):
        """Return a sample of one segment's masked pixels, in the working color space."""
        pixels = masked_pixels(segment, cap=self.sample_pixels, required=False)
        return None if pixels is None else self._convert(pixels)

    def _fit(self, pixels):
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=self.n_colors, n_init=10, random_state=0)
        model.fit(pixels)
        return model

    def _score(self, segment):
        """Return the proportion of masked pixels nearest each palette color, and the palette's group.

        Every palette color has a key, zero where nothing fell in it.
        """
        pixels = self._pixels(segment)
        if pixels is None:
            raise ValueError("empty mask")

        palette, group = self._fit_for(segment.occurrence_id)
        assignments = palette.predict(pixels)

        counts = np.bincount(assignments, minlength=self.n_colors)
        proportions = counts / counts.sum()

        result = {f"color_{index}": float(value) for index, value in enumerate(proportions)}
        result["group"] = group
        result["dominant"] = int(np.argmax(counts))
        return result


def color_clusters(**kwargs):
    """Operation: a `ColorClusterMetric`.

    Args:
        **kwargs: As in `ColorClusterMetric`.
    """
    return ColorClusterMetric(**kwargs)
