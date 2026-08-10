"""
Colour metrics for iNaturalist photographs.

The core colour metrics (critterframe.metrics.color) assume a controlled
imaging setup, where a measured colour means something because the lighting was
the same for every specimen. iNaturalist photographs are the opposite: shot on
phones, in sun and shade, with automatic white balance doing something
different in every frame. Measured naively, "mean lightness" mostly measures
the weather.

Three responses, all of them metrics:

  white-balanced colour -- normalize each image before measuring, so a colour
      is at least internally consistent rather than an artifact of the camera's
      auto white balance.
  background colour -- measure the pixels OUTSIDE the mask and store them.
      The background is the context an observation was photographed in -- leaf,
      hand, bark, sky -- and it's simultaneously a covariate for the lighting
      problem above and a QC signal in its own right (a photograph whose
      background is the same colour as the organism is one whose segmentation
      is likely poor).
  colour clustering per group -- rather than reducing an organism to one mean,
      fit a shared palette across a group and report what fraction of the
      organism falls in each colour. Comparable across occurrences of the same
      group in a way a mean isn't, and it captures pattern.
"""

import logging

import cv2
import numpy as np

from ....recipes import Metric
from ....metrics.outliers import POPULATION, group_lookup

logger = logging.getLogger(__name__)

# Pixels sampled per occurrence when fitting a group palette. Every masked pixel
# of a few hundred occurrences is tens of millions of points, which KMeans does
# not need -- a few thousand per occurrence captures the same distribution and
# keeps the fit to seconds.
PIXELS_PER_OCCURRENCE = 2000

# Groups with fewer reference occurrences than this share the population-wide
# palette rather than getting their own.
MIN_GROUP_SIZE = 5


def white_balanced_color(name=None, unit="fraction"):
    """
    Metric: mean colour of the masked pixels after grey-world white balancing,
    as {"r", "g", "b"} on 0-1 scales.

    Grey-world assumes the WHOLE IMAGE averages to grey and scales each channel
    to make that true, then measures the organism under that correction. The
    assumption is crude and fails on an image that's genuinely mostly one
    colour, but it's computed from the image itself with no reference object,
    which is the only kind of correction available here.

    Uses the whole frame, not just the mask, to estimate the correction -- the
    organism is exactly the part whose colour you're trying to measure, so
    normalizing by it would define away the signal.
    """
    return Metric("white_balanced_color", _white_balanced_color, version="1",
                  unit=unit, metric_name=name)


def background_color(name=None, unit="fraction"):
    """
    Metric: mean colour and lightness of the pixels OUTSIDE the mask, as
    {"r", "g", "b", "lightness", "contrast"}.

    Background is information, not noise. What an organism was photographed
    against is context worth recording, and `contrast` -- the difference in
    lightness between the organism and its background -- is a direct QC signal:
    an organism close in tone to its background is one whose mask is most
    likely wrong, and one whose colour measurements are most likely
    contaminated by whatever the mask wrongly included.

    Returns lightness and contrast on 0-1 scales; contrast is signed, positive
    where the organism is lighter than its background.
    """
    return Metric("background_color", _background_color, version="1",
                  unit=unit, metric_name=name)


class ColorClusterMetric(Metric):
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

    Why it isn't a metrics.outliers.GroupMetric subclass: that one fits on one
    ready-made feature ROW per reference occurrence, read from stored metric
    values. This fits on pooled PIXELS -- thousands of rows per occurrence,
    which have to be gathered from the image store rather than read from a
    table -- and scores by reducing thousands of rows back to one dict. The
    fit/score shapes genuinely differ. What IS shared is the group-with-fallback
    lookup, which comes from metrics.outliers.group_lookup rather than being
    written twice.

    Label stability is why the palette is fit per group and reused: "cluster 2"
    has to mean the same actual colour for every occurrence scored against a
    given group, or the proportions aren't comparable. That falls out of fitting
    once and reusing, but it's a property the per-occurrence group metrics never
    had to think about.

    n_colors       -- palette size per group.
    group_col      -- occurrence column to group by, e.g. "taxon". None fits one
                     palette for the whole project.
    color_space    -- "lab" (default) or "hsv". Lab because distance in it
                     approximates perceived colour difference, so clusters
                     correspond to colours a person would call distinct.
    transforms     -- operations applied when gathering the reference pixels.
                     Pass the SAME ones the run uses, or the palette is fit on a
                     different representation than the one being scored against
                     it.
    min_group_size -- groups smaller than this share the population palette.
    sample_pixels  -- pixels sampled per reference occurrence.
    """

    def __init__(self, n_colors=5, group_col=None, color_space="lab",
                 transforms=(), min_group_size=MIN_GROUP_SIZE,
                 sample_pixels=PIXELS_PER_OCCURRENCE, name=None,
                 unit="fraction", reference=False):
        if color_space not in ("lab", "hsv"):
            raise ValueError('color_space must be "lab" or "hsv"')

        super().__init__("color_clusters", self._score, version="1", unit=unit,
                         metric_name=name or "color_clusters")

        self.n_colors = n_colors
        self.group_col = group_col
        self.color_space = color_space
        self.transforms = list(transforms)
        self.min_group_size = min_group_size
        self.sample_pixels = sample_pixels
        self.reference = reference

        self.palettes = {}
        self.group_by_id = {}

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

        This is the expensive step -- it reads every reference occurrence's
        image -- and it happens once per run rather than once per occurrence,
        which is exactly what the prepare() hook is for.
        """
        from ....training.datasets import iterate_segments

        self.group_by_id = group_lookup(context.project_path, self.group_col,
                                        context.occurrence_ids)

        pooled = {}
        for occurrence_id, segment in iterate_segments(
                context.project_path, part=context.part,
                transforms=self.transforms, reference=self.reference,
                occurrence_ids=context.occurrence_ids):
            pixels = self._sample(segment)
            if pixels is None:
                continue
            group = self.group_by_id.get(occurrence_id, POPULATION)
            pooled.setdefault(group, []).append(pixels)
            pooled.setdefault(POPULATION, []).append(pixels)

        if POPULATION not in pooled:
            raise ValueError(
                "no reference occurrences with usable pixels -- segment this "
                "part before fitting a colour palette on it"
            )

        for group, chunks in pooled.items():
            if group is not POPULATION and len(chunks) < self.min_group_size:
                logger.warning("group %r has only %d reference occurrences "
                               "(< min_group_size=%d) -- using the "
                               "population-wide palette", group, len(chunks),
                               self.min_group_size)
                continue
            self.palettes[group] = self._fit(np.concatenate(chunks))

        logger.info("%s fit: %d group palette(s) of %d colours + 1 "
                    "population-wide fallback", self.metric_name,
                    len(self.palettes) - 1, self.n_colors)

    def _convert(self, pixels):
        """BGR pixels into the working colour space, as float."""
        code = cv2.COLOR_BGR2LAB if self.color_space == "lab" else cv2.COLOR_BGR2HSV
        converted = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), code)
        return converted.reshape(-1, 3).astype(np.float32)

    def _sample(self, segment):
        """A random sample of one segment's masked pixels, in the working space."""
        if segment.mask is None or not segment.mask.any():
            return None

        image = np.asarray(segment.image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        pixels = image[segment.mask]
        if len(pixels) > self.sample_pixels:
            rng = np.random.default_rng(0)
            pixels = pixels[rng.choice(len(pixels), self.sample_pixels,
                                       replace=False)]
        return self._convert(pixels)

    def _fit(self, pixels):
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=self.n_colors, n_init=10, random_state=0)
        model.fit(pixels)
        return model

    def _palette_for(self, occurrence_id):
        group = self.group_by_id.get(occurrence_id, POPULATION)
        palette = self.palettes.get(group)
        if palette is None:
            group, palette = POPULATION, self.palettes[POPULATION]
        return palette, group

    def _score(self, segment):
        """
        Assign every masked pixel to its nearest palette colour and return the
        proportions, plus which group's palette was used.

        Every palette colour gets a key even when its proportion is zero, so the
        exported columns are the same set for every occurrence in a group --
        otherwise a wide export would be full of holes that mean "zero" rather
        than "not measured".
        """
        if not self.palettes:
            raise RuntimeError(
                f"{self.metric_name} was never fit -- group metrics are fit by "
                "their prepare() hook, which run_metrics calls for you"
            )

        pixels = self._sample(segment)
        if pixels is None:
            raise ValueError("empty mask")

        palette, group = self._palette_for(segment.occurrence_id)
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


def _grey_world_scale(image):
    """Per-channel gains that make the whole frame average to grey."""
    means = np.asarray(image).reshape(-1, 3).mean(axis=0)
    means[means == 0] = 1.0
    return means.mean() / means


def _white_balanced_color(segment):
    mask = segment.require_mask()
    if not mask.any():
        raise ValueError("empty mask")

    image = np.asarray(segment.image).astype(np.float32)
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)

    balanced = np.clip(image * _grey_world_scale(image), 0, 255)
    blue, green, red = balanced[mask].mean(axis=0) / 255.0
    return {"r": float(red), "g": float(green), "b": float(blue)}


def _lightness(pixels):
    """Mean CIELAB lightness of an (N, 3) BGR array, on a 0-1 scale."""
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
    return float(lab[:, 0, 0].mean() / 255.0)


def _background_color(segment):
    mask = segment.require_mask()
    background = ~mask
    if not background.any():
        raise ValueError("the mask covers the whole frame -- no background to measure")
    if not mask.any():
        raise ValueError("empty mask")

    image = np.asarray(segment.image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    background_pixels = image[background]
    blue, green, red = background_pixels.mean(axis=0) / 255.0

    background_lightness = _lightness(background_pixels)
    organism_lightness = _lightness(image[mask])

    return {
        "r": float(red),
        "g": float(green),
        "b": float(blue),
        "lightness": background_lightness,
        "contrast": organism_lightness - background_lightness,
    }
