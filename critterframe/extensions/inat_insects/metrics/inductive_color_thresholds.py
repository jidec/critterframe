"""
A group colour metric that discovers its own hue categories, instead of using
a fixed vocabulary the way metrics.color.HUE_BANDS or ColorClusterMetric's
chosen n_colors do.

Five steps, run once per group in prepare(), on pooled reference pixels:

  1. Convert to Lab and compute chroma C = sqrt(a^2 + b^2) per pixel, instead
     of HSV saturation -- better-behaved, and, unlike hue, not circular.
  2. Find the chroma gate empirically: bin pixels by chroma, compute CIRCULAR
     hue variance within each bin. Variance is high and noisy at low chroma
     (a near-gray/near-black/near-white pixel's hue is close to meaningless)
     and stabilizes past some chroma value -- that stabilization point is the
     gate, derived from this population's own data rather than a fixed
     cutoff like metrics.color.MIN_SATURATION.
  3. Gate on chroma: keep only pixels at or above that threshold as the
     "clean" chromatic subset the hue distribution is read from.
  4. Derive hue thresholds from the gated subset via KDE valley-finding: local
     minima in a circular kernel density estimate of hue are the natural
     boundaries between colour modes, whatever their number turns out to be.
  5. Apply back to the FULL masked population (gated pixels included, not
     just the ones that passed the gate) when scoring an occurrence, so the
     reported fractions describe the whole organism, not just its most
     colourful pixels.

Same pooled-pixel shape as ColorClusterMetric, and for the same reason it
isn't a metrics.outliers.GroupMetric subclass: that one fits on one
ready-made feature row per reference occurrence, read from stored metric
values; this fits on thousands of raw pixels per occurrence, gathered from
the image store. group_lookup/POPULATION are shared with it rather than
reimplemented.
"""

import logging

import cv2
import numpy as np

from ....recipes import Metric
from ....metrics.outliers import POPULATION, group_lookup
from ....records.occurrences import ids_record

logger = logging.getLogger(__name__)

# Pixels sampled per occurrence when POOLING for a fit. Scoring (_score) never
# samples -- step 5 is explicit that the full masked population is what gets
# classified -- this only bounds the (much larger, many-occurrence) pool a
# group's definition is fit from.
PIXELS_PER_OCCURRENCE = 2000

# Groups with fewer reference occurrences than this share the population-wide
# definition rather than fitting their own.
MIN_GROUP_SIZE = 5


class InductiveColorThresholdMetric(Metric):
    """
    Group metric: fraction of this organism in each hue range its group's own
    pixels turned out to have, plus the fraction too washed-out to have a
    reliable hue at all.

    - `group_col` -- occurrence column to group by, e.g. `"taxon"`. None fits
      one definition for the whole project.
    - `transforms` -- operations applied when gathering the reference pixels.
      Pass the SAME ones the run uses, or the definition is fit on a
      different representation than the one being scored against it.
    - `min_group_size` -- groups smaller than this share the population
      definition.
    - `sample_pixels` -- pixels sampled per reference occurrence when POOLING
      for the fit (see module docstring; scoring never samples).
    - `n_chroma_bins` -- quantile bins (equal pixel count, robust to a skewed
      chroma distribution) the empirical gate search bins pooled pixels into.
    - `min_bin_pixels` -- a chroma bin with fewer pixels than this is excluded
      from the gate search rather than trusted with a noisy circular-variance
      estimate.
    - `hue_bandwidth` -- degrees; the Gaussian kernel bandwidth for the
      circular KDE hue thresholds are read off of.
    - `valley_relative_height` -- a KDE local minimum only counts as a
      category boundary when its density is at most this fraction of the
      KDE's peak, so a shallow noise wiggle near an already-low baseline
      doesn't fragment one real colour into several.
    - `max_hue_ranges` -- safety cap on how many ranges one group's fit may
      discover; past this, only the deepest (lowest-density) valleys are kept.
    - `reference` -- fit against reference masks instead of canonical ones.

    Comparable within one group's scored occurrences, same as
    ColorClusterMetric's numbered clusters -- NOT across groups. Each group's
    fit inductively discovers its own count and placement of hue ranges, so
    `hue_0` for one group and `hue_0` for another are not the same colour.
    """

    def __init__(self, group_col=None, transforms=(), min_group_size=MIN_GROUP_SIZE,
                 sample_pixels=PIXELS_PER_OCCURRENCE, n_chroma_bins=20,
                 min_bin_pixels=50, hue_bandwidth=10.0, valley_relative_height=0.5,
                 max_hue_ranges=8, name=None, unit="fraction", reference=False):
        super().__init__("inductive_color_thresholds", self._score, version="1",
                         unit=unit, metric_name=name or "inductive_color_thresholds")

        self.group_col = group_col
        self.transforms = list(transforms)
        self.min_group_size = min_group_size
        self.sample_pixels = sample_pixels
        self.n_chroma_bins = n_chroma_bins
        self.min_bin_pixels = min_bin_pixels
        self.hue_bandwidth = hue_bandwidth
        self.valley_relative_height = valley_relative_height
        self.max_hue_ranges = max_hue_ranges
        self.reference = reference

        self.definitions = {}
        self.group_by_id = {}

    def spec(self):
        spec = super().spec()
        spec["parameters"] = {
            "group_col": self.group_col,
            "min_group_size": self.min_group_size,
            "sample_pixels": self.sample_pixels,
            "n_chroma_bins": self.n_chroma_bins,
            "min_bin_pixels": self.min_bin_pixels,
            "hue_bandwidth": self.hue_bandwidth,
            "valley_relative_height": self.valley_relative_height,
            "max_hue_ranges": self.max_hue_ranges,
            "reference": self.reference,
            "transforms": [operation.spec() for operation in self.transforms],
        }
        return spec

    def prepare(self, context):
        """
        Fit one gate+hue-ranges definition per group by pooling masked pixels
        across the reference population, plus a population-wide fallback.

        Expensive (reads every reference occurrence's image), so it happens
        once per run via this hook rather than once per occurrence.

        Returns the fit record the run stores: each definition's contributing
        occurrences as a count and a digest, plus the gate and range count it
        actually found -- the one thing about a fit a human can't infer from
        the parameters alone.
        """
        from ....training.datasets import iterate_segments

        self.group_by_id = group_lookup(context.project_path, self.group_col,
                                        context.occurrence_ids)

        pooled = {}
        contributors = {}
        for occurrence_id, segment in iterate_segments(
                context.project_path, part=context.part,
                transforms=self.transforms, reference=self.reference,
                occurrence_ids=context.occurrence_ids):
            pixels = _sample(segment, self.sample_pixels)
            if pixels is None:
                continue
            group = self.group_by_id.get(occurrence_id, POPULATION)
            for key in {group, POPULATION}:
                pooled.setdefault(key, []).append(pixels)
                contributors.setdefault(key, []).append(occurrence_id)

        if POPULATION not in pooled:
            raise ValueError(
                "no reference occurrences with usable pixels -- segment this "
                "part before fitting inductive colour thresholds on it"
            )

        groups = {}
        for group, chunks in pooled.items():
            fitted = group is POPULATION or len(chunks) >= self.min_group_size
            if fitted:
                definition = _fit_definition(
                    np.concatenate(chunks), self.n_chroma_bins, self.min_bin_pixels,
                    self.hue_bandwidth, self.valley_relative_height, self.max_hue_ranges)
                self.definitions[group] = definition
            else:
                logger.warning(
                    "group %r has only %d reference occurrences (< "
                    "min_group_size=%d) -- using the population-wide "
                    "definition instead of its own", group, len(chunks),
                    self.min_group_size)
            if group is not POPULATION:
                record = dict(ids_record(contributors[group]), fitted=fitted)
                if fitted:
                    record["gate"] = self.definitions[group]["gate"]
                    record["n_hue_ranges"] = len(self.definitions[group]["hue_ranges"])
                groups[str(group)] = record

        logger.info("%s fit: %d group definition(s) + 1 population-wide fallback "
                    "(population gate=%.2f, %d hue range(s))", self.metric_name,
                    len(self.definitions) - 1, self.definitions[POPULATION]["gate"],
                    len(self.definitions[POPULATION]["hue_ranges"]))

        return {
            "group_col": self.group_col,
            "n_chroma_bins": self.n_chroma_bins,
            "min_bin_pixels": self.min_bin_pixels,
            "hue_bandwidth": self.hue_bandwidth,
            "valley_relative_height": self.valley_relative_height,
            "max_hue_ranges": self.max_hue_ranges,
            "reference": self.reference,
            "population": dict(ids_record(contributors[POPULATION]),
                              gate=self.definitions[POPULATION]["gate"],
                              n_hue_ranges=len(self.definitions[POPULATION]["hue_ranges"])),
            "groups": groups,
        }

    def _definition_for(self, occurrence_id):
        group = self.group_by_id.get(occurrence_id, POPULATION)
        definition = self.definitions.get(group)
        if definition is None:
            group, definition = POPULATION, self.definitions[POPULATION]
        return definition, group

    def _score(self, segment):
        """
        Classify EVERY masked pixel of this occurrence (not a sample -- see
        module docstring step 5) against its group's fitted gate + hue
        ranges, and return the fraction in each.

        Every hue_N key is present for every occurrence scored against one
        group's definition, zero-valued where nothing fell in it, so a wide
        export has no holes that could be misread as "not measured".
        """
        if not self.definitions:
            raise RuntimeError(
                f"{self.metric_name} was never fit -- group metrics are fit by "
                "their prepare() hook, which run_metrics calls for you"
            )

        pixels = _sample(segment, cap=None)
        if pixels is None:
            raise ValueError("empty mask")

        definition, group = self._definition_for(segment.occurrence_id)
        lab = _to_lab(pixels)
        chroma, hue = _chroma_hue(lab)
        masks = _classify(hue, chroma, definition)

        total = len(pixels)
        result = {label: float(mask.sum()) / total for label, mask in masks.items()}
        hue_keys = [key for key in result if key.startswith("hue_")]
        result["dominant"] = (max(hue_keys, key=result.get) if any(result[k] > 0 for k in hue_keys)
                              else None)
        if result["dominant"] is not None:
            result["dominant"] = int(result["dominant"].split("_")[1])
        result["group"] = group
        return result


def inductive_color_thresholds(**kwargs):
    """Operation: InductiveColorThresholdMetric, in the lowercase factory style of every other metric."""
    return InductiveColorThresholdMetric(**kwargs)


# ---------------------------------------------------------------------------
# Pixel gathering
# ---------------------------------------------------------------------------


def _sample(segment, cap):
    """
    A segment's masked pixels, BGR, capped to `cap` by random sample -- or
    every one of them when `cap` is None, which is what scoring needs (step
    5 classifies the full population, not a sample of it).
    """
    if segment.mask is None or not segment.mask.any():
        return None

    image = np.asarray(segment.image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    pixels = image[segment.mask]
    if cap is not None and len(pixels) > cap:
        rng = np.random.default_rng(0)
        pixels = pixels[rng.choice(len(pixels), cap, replace=False)]
    return pixels


# ---------------------------------------------------------------------------
# Colour space
# ---------------------------------------------------------------------------


def _to_lab(bgr_pixels):
    """
    (N, 3) uint8 BGR -> (N, 3) float32 Lab in the TRUE range (L: 0-100, a/b:
    roughly -127..127).

    Deliberately NOT the (N, 3) uint8-in-uint8-out path ColorClusterMetric's
    _convert uses -- that one leaves a/b offset by +128 (how OpenCV stores
    Lab in an 8-bit image), which is harmless for KMeans' translation-
    invariant Euclidean distance but wrong here: chroma is a distance from
    the TRUE origin, not an arbitrary shifted one. Converting through a
    float32 0-1 BGR image instead makes OpenCV emit unscaled Lab directly.
    """
    bgr = np.asarray(bgr_pixels, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3)


def _chroma_hue(lab_pixels):
    """(N, 3) Lab -> (chroma, hue_degrees), hue in [0, 360)."""
    a = lab_pixels[:, 1]
    b = lab_pixels[:, 2]
    chroma = np.sqrt(a ** 2 + b ** 2)
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    return chroma, hue


def _circular_variance(hue_degrees):
    """
    1 - |mean resultant vector length| of a set of angles: 0 when every angle
    is identical, close to 1 when they're scattered uniformly around the
    circle. Standard circular statistics (Fisher 1993) -- unlike a plain
    linear variance, hues of 359, 0, and 1 degrees correctly read as nearly
    identical rather than wildly spread.
    """
    if len(hue_degrees) == 0:
        return float("nan")
    radians = np.radians(hue_degrees)
    resultant = np.hypot(np.mean(np.cos(radians)), np.mean(np.sin(radians)))
    return float(1.0 - resultant)


# ---------------------------------------------------------------------------
# Step 2: the empirical chroma gate
# ---------------------------------------------------------------------------


def _find_chroma_gate(chroma, hue_degrees, n_bins, min_bin_pixels):
    """
    The chroma value past which hue stops being noise.

    Bins pixels into `n_bins` equal-COUNT (quantile) bins by chroma -- robust
    to a skewed chroma distribution, unlike equal-width bins -- and computes
    circular hue variance within each bin that has at least `min_bin_pixels`
    (a bin with fewer is excluded rather than trusted with a noisy estimate).
    Returns the lower edge of the first trusted bin whose variance, and every
    trusted bin's variance from there on, stays at or below the stabilized
    (high-chroma) level -- a real stabilization point, not a bin that
    happened to dip low once by chance.

    No bin ever stabilizes (or none is trusted at all) -- logs a warning and
    returns 0.0, the safest fallback: nothing gets gated.
    """
    chroma = np.asarray(chroma)
    hue_degrees = np.asarray(hue_degrees)
    if len(chroma) == 0:
        logger.warning("no pixels to find a chroma gate from -- using gate 0.0")
        return 0.0

    edges = np.quantile(chroma, np.linspace(0.0, 1.0, n_bins + 1))
    bin_index = np.clip(np.searchsorted(edges, chroma, side="right") - 1, 0, n_bins - 1)

    trusted = []   # [(lower_edge, circular_variance), ...] in increasing chroma order
    for index in range(n_bins):
        members = hue_degrees[bin_index == index]
        if len(members) >= min_bin_pixels:
            trusted.append((float(edges[index]), _circular_variance(members)))

    if not trusted:
        logger.warning("no chroma bin had >= %d pixels -- using gate 0.0",
                       min_bin_pixels)
        return 0.0

    variances = np.array([variance for _, variance in trusted])
    tail_count = max(1, len(variances) // 4)
    # The LAST tail_count bins in chroma order -- i.e. the highest-chroma,
    # presumed-already-stable end -- not the tail_count LARGEST variance
    # values (np.sort(variances) would silently pick out the noisiest
    # low-chroma bins instead, since low chroma is exactly where variance is
    # highest).
    baseline = float(np.median(variances[-tail_count:]))
    tolerance = 1.5

    for position, (edge, _variance) in enumerate(trusted):
        rest = variances[position:]
        if np.all(rest <= baseline * tolerance):
            return edge

    logger.warning("hue variance never stabilized across chroma bins -- using gate 0.0")
    return 0.0


# ---------------------------------------------------------------------------
# Step 4: hue ranges from the gated subset
# ---------------------------------------------------------------------------


def _hue_kde(hue_degrees, bandwidth, grid_size=360):
    """
    Circular kernel density estimate of `hue_degrees`, evaluated on a
    `grid_size`-point grid (1 degree apart by default).

    A wrapped Gaussian kernel: each grid point sums a Gaussian of `hue_degrees`'
    CIRCULAR distance to it (never more than 180 degrees either way), so
    density near 0 correctly includes samples near 360.
    """
    grid = np.arange(grid_size, dtype=np.float64) * (360.0 / grid_size)
    hue_degrees = np.asarray(hue_degrees, dtype=np.float64)
    if len(hue_degrees) == 0:
        return grid, np.zeros(grid_size)

    delta = grid[:, None] - hue_degrees[None, :]
    delta = ((delta + 180.0) % 360.0) - 180.0
    density = np.exp(-0.5 * (delta / bandwidth) ** 2).sum(axis=1)
    return grid, density


def _find_hue_valleys(grid, density, valley_relative_height, max_ranges):
    """
    Grid angles at each low-density gap in `density` -- the natural
    boundaries between hue modes.

    A point counts as a valley when it's a circular local minimum (at or
    below both neighbours, wrapping) AND at or below `valley_relative_height`
    of the peak density -- the height test is what stops a shallow noise
    wiggle near an already-low baseline from fragmenting one real colour into
    several. A whole low-density stretch collapses to ONE representative
    point (its own minimum), not one valley per grid step in it. Past
    `max_ranges - 1` qualifying valleys, keeps only the deepest (lowest-
    density) ones.
    """
    n = len(density)
    if n == 0 or density.max() <= 0:
        return np.array([])

    previous = np.roll(density, 1)
    following = np.roll(density, -1)
    is_local_min = (density <= previous) & (density <= following)
    threshold = valley_relative_height * density.max()
    qualifies = is_local_min & (density <= threshold)

    indices = np.where(qualifies)[0]
    if len(indices) == 0:
        return np.array([])

    groups = [[int(indices[0])]]
    for index in indices[1:]:
        if index == groups[-1][-1] + 1:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    # a run touching both ends is one contiguous run across the wrap
    if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == n - 1:
        groups[0] = groups[-1] + groups[0]
        groups.pop()

    representatives = sorted(min(group, key=lambda i: density[i]) for group in groups)

    max_valleys = max(0, max_ranges - 1)
    if len(representatives) > max_valleys:
        by_depth = sorted(representatives, key=lambda i: density[i])[:max_valleys]
        representatives = sorted(by_depth)

    return grid[representatives]


def _ranges_from_valleys(valley_angles):
    """
    Sorted valley angles -> circular [(start, end), ...] arcs, one per gap
    between consecutive valleys (wrapping past the last back to the first).

    Fewer than 2 valleys -> one range covering the whole circle: a single
    mode's low-density far side still counts as exactly one valley by
    _find_hue_valleys' criterion, but it isn't a boundary BETWEEN two
    clusters -- there's nothing on its other side to separate FROM -- so
    treating it as a real cut would produce a single zero-width (start, start)
    range that classifies no pixel at all, rather than the one true range
    covering everything. Two genuinely separated clusters always produce two
    valleys (one gap on each side), never one, so this never misfires on a
    real bimodal population.
    """
    if len(valley_angles) < 2:
        return [(0.0, 360.0)]

    valleys = sorted(float(angle) for angle in valley_angles)
    return [(valleys[index], valleys[(index + 1) % len(valleys)])
           for index in range(len(valleys))]


def _fit_definition(bgr_pixels, n_chroma_bins, min_bin_pixels, hue_bandwidth,
                    valley_relative_height, max_hue_ranges):
    """The full 5-step fit (steps 1-4; step 5 is _classify, applied at score
    time) over one pool of BGR pixels. Returns {"gate", "hue_ranges"}."""
    lab = _to_lab(bgr_pixels)
    chroma, hue = _chroma_hue(lab)
    gate = _find_chroma_gate(chroma, hue, n_chroma_bins, min_bin_pixels)

    gated_hue = hue[chroma >= gate]
    if len(gated_hue) < min_bin_pixels:
        logger.warning("fewer than %d pixels passed the chroma gate (%.2f) -- "
                       "using one hue range covering the whole circle",
                       min_bin_pixels, gate)
        return {"gate": gate, "hue_ranges": [(0.0, 360.0)]}

    grid, density = _hue_kde(gated_hue, hue_bandwidth)
    valleys = _find_hue_valleys(grid, density, valley_relative_height, max_hue_ranges)
    return {"gate": gate, "hue_ranges": _ranges_from_valleys(valleys)}


# ---------------------------------------------------------------------------
# Step 5: apply back to the full masked population
# ---------------------------------------------------------------------------


def _classify(hue_degrees, chroma, definition):
    """
    {"achromatic": mask, "hue_0": mask, ...} for every pixel -- a partition,
    so every pixel gets exactly one label. achromatic is chroma below the
    gate; otherwise whichever hue_ranges arc contains the pixel's hue.
    """
    achromatic = chroma < definition["gate"]
    masks = {"achromatic": achromatic}
    for index, (start, end) in enumerate(definition["hue_ranges"]):
        if start <= end:
            in_range = (hue_degrees >= start) & (hue_degrees < end)
        else:
            in_range = (hue_degrees >= start) | (hue_degrees < end)
        masks[f"hue_{index}"] = in_range & ~achromatic
    return masks
