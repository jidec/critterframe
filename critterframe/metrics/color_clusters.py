"""
Colour clustering per group: what fraction of an organism falls in each colour of its group's palette.

Rather than reducing an organism to one mean, fits a shared palette across a group and reports the proportions, which
are comparable across occurrences of the same group in a way a mean isn't, and capture pattern.
"""

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
    """
    Group metric: what fraction of this organism falls into each colour of its
    group's shared palette.

    A mean colour collapses a patterned organism to a single value -- a
    black-and-yellow dragonfly and a uniformly olive one can share a mean. This
    instead fits a palette across a whole group (usually a species), assigns
    every pixel of one occurrence to its nearest palette colour, and reports the
    proportions: "62% rust-brown, 30% cream, 8% near-black". That's a colour
    signature relative to the group's own typical palette, comparable across
    occurrences, and useful as a QC signal too -- proportions wildly off a
    species' norm suggest a bad mask, a misidentification, or a genuine outlier
    worth a look.

A `metrics.outliers.PooledPixelGroupMetric`, not a `GroupMetric`: that one fits
    on one ready-made feature ROW per reference occurrence, read from stored
    metric values, while this fits on pooled PIXELS gathered from the image
    store. The pooling, the minimum-group-size fallback and the fit record are
    the base's; the colour space and the clustering are this metric's own.

    Label stability is why the palette is fit per group and reused: "cluster 2"
    has to mean the same actual colour for every occurrence scored against a
    given group, or the proportions aren't comparable. That falls out of fitting
    once and reusing, but it's a property the per-occurrence group metrics never
    had to think about.

    - `n_colors` -- palette size per group.
    - `group_col` -- occurrence column to group by, e.g. `"taxon"`. None
      fits one palette for the whole project.
    - `color_space` -- `"lab"` (default) or `"hsv"`. Lab because distance
      in it approximates perceived colour difference, so clusters
      correspond to colours a person would call distinct.
    - `transforms` -- operations applied when gathering the reference
      pixels. Pass the SAME ones the run uses, or the palette is fit on a
      different representation than the one being scored against it.
    - `min_group_size` -- groups smaller than this share the population
      palette.
    - `sample_pixels` -- pixels sampled per reference occurrence.
    """

    def __init__(self, n_colors=5, group_col=None, color_space="lab",
                 transforms=(), min_group_size=MIN_GROUP_SIZE,
                 sample_pixels=PIXELS_PER_OCCURRENCE, name=None,
                 unit="fraction", reference=False):
        if color_space not in ("lab", "hsv"):
            raise ValueError('color_space must be "lab" or "hsv"')

        super().__init__("color_clusters", self._score, group_col=group_col,
                         transforms=transforms, min_group_size=min_group_size,
                         sample_pixels=sample_pixels, reference=reference,
                         metric_name=name or "color_clusters", unit=unit)

        self.n_colors = n_colors
        self.color_space = color_space

    def spec(self):
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
        """
        Fit one palette per group by pooling masked pixels across the reference
        population, plus a population-wide fallback.

        The pooling itself is `PooledPixelGroupMetric.prepare`; what is this
        metric's own is converting to the working colour space and clustering.

        Returns that base's fit record, plus this metric's own settings.
        """
        record = super().prepare(context)
        logger.info("%s fit: %d group palette(s) of %d colours + 1 "
                    "population-wide fallback", self.metric_name,
                    len(self.fits) - 1, self.n_colors)
        return dict(record, color_space=self.color_space, n_colors=self.n_colors)

    def _convert(self, pixels):
        """BGR pixels into the working colour space, as float."""
        converted = convert(pixels, self.color_space)
        # Lab stays in true units, where Euclidean distance approximates perceived difference; hsv is rescaled
        # so hue in degrees does not swamp saturation and value in KMeans.
        return normalize(converted, "hsv") if self.color_space == "hsv" else converted

    def _pixels(self, segment):
        """A sample of one segment's masked pixels, in the working colour space."""
        pixels = masked_pixels(segment, cap=self.sample_pixels, required=False)
        return None if pixels is None else self._convert(pixels)

    def _fit(self, pixels):
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=self.n_colors, n_init=10, random_state=0)
        model.fit(pixels)
        return model

    def _score(self, segment):
        """
        Assign every masked pixel to its nearest palette colour and return the
        proportions, plus which group's palette was used.

        Every palette colour gets a key even when its proportion is zero, so the
        exported columns are the same set for every occurrence in a group --
        otherwise a wide export would be full of holes that mean "zero" rather
        than "not measured".
        """
        pixels = self._pixels(segment)
        if pixels is None:
            raise ValueError("empty mask")

        palette, group = self._fit_for(segment.occurrence_id)
        assignments = palette.predict(pixels)

        counts = np.bincount(assignments, minlength=self.n_colors)
        proportions = counts / counts.sum()

        result = {f"color_{index}": float(value)
                  for index, value in enumerate(proportions)}
        result["group"] = group
        result["dominant"] = int(np.argmax(counts))
        return result


def color_clusters(**kwargs):
    """Operation: ColorClusterMetric, in the lowercase factory style of every other metric."""
    return ColorClusterMetric(**kwargs)
