"""
Colour thresholds fitted per group: a chroma gate and hue arcs discovered from the group's own pixels.

The fitted counterpart of `color_thresholds`, which takes its thresholds as given; the rest of this docstring is the
fit. Discovers its own hue categories instead of a fixed vocabulary like `color_thresholds.HUE_BANDS` or the chosen
`n_colors` of `color_clusters`.

Five steps, run once per group in prepare(), on pooled reference pixels:

  1. Convert to Lab and compute chroma C = sqrt(a^2 + b^2) per pixel, instead
     of HSV saturation -- better-behaved, and, unlike hue, not circular.
  2. Find the chroma gate empirically: bin pixels by chroma, compute CIRCULAR
     hue variance within each bin. Variance is high and noisy at low chroma
     (a near-gray/near-black/near-white pixel's hue is close to meaningless)
     and stabilizes past some chroma value -- that stabilization point is the
     gate, derived from this population's own data rather than a fixed
     cutoff like metrics.color_thresholds.MIN_SATURATION.
  3. Gate on chroma: keep only pixels at or above that threshold as the
     "clean" chromatic subset the hue distribution is read from.
  4. Derive hue thresholds from the gated subset via KDE valley-finding: local
     minima in a circular kernel density estimate of hue are the natural
     boundaries between colour modes, whatever their number turns out to be.
  5. Express the gate and arcs as `ColorThreshold`s in lch -- `achromatic` below
     the gate, one `hue_<i>` per arc at or above it -- and apply them to the
     FULL masked population (gated pixels included, not just the ones that
     passed the gate) when scoring an occurrence, so the reported fractions
     describe the whole organism, not just its most colourful pixels.

Same pooled-pixel shape as the extension's ColorClusterMetric, and both are
metrics.outliers.PooledPixelGroupMetric subclasses: the pooling, the
minimum-group-size fallback and the fit record are that base's, and what is
here is the five-step fit itself.
"""

import logging

import numpy as np

from ..colorspaces import convert
from ..visualization import figures
from .color_thresholds import color_threshold, threshold_masks, threshold_panel
from .outliers import POPULATION, PooledPixelGroupMetric
from .pixels import masked_pixels

logger = logging.getLogger(__name__)

# Pixels sampled per occurrence when POOLING for a fit. Scoring (_score) never
# samples -- step 5 is explicit that the full masked population is what gets
# classified -- this only bounds the (much larger, many-occurrence) pool a
# group's definition is fit from.
PIXELS_PER_OCCURRENCE = 2000

# Groups with fewer reference occurrences than this share the population-wide
# definition rather than fitting their own.
MIN_GROUP_SIZE = 5

# Fitted groups drawn as figures, largest first, beside the population's own;
# a project grouped by species would otherwise leave hundreds of them.
FIGURE_GROUPS = 10


class InductiveColorThresholdMetric(PooledPixelGroupMetric):
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
        super().__init__("inductive_color_thresholds", self._score,
                         group_col=group_col, transforms=transforms,
                         min_group_size=min_group_size, sample_pixels=sample_pixels,
                         reference=reference, unit=unit,
                         metric_name=name or "inductive_color_thresholds")

        self.n_chroma_bins = n_chroma_bins
        self.min_bin_pixels = min_bin_pixels
        self.hue_bandwidth = hue_bandwidth
        self.valley_relative_height = valley_relative_height
        self.max_hue_ranges = max_hue_ranges

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

    def _fit(self, pixels):
        """The full 5-step fit over one group's pooled pixels (see `_fit_definition`)."""
        logger.debug("%s: fitting %d pooled pixel(s)", self.metric_name, len(pixels))
        definition = _fit_definition(pixels, self.n_chroma_bins, self.min_bin_pixels,
                                     self.hue_bandwidth, self.valley_relative_height,
                                     self.max_hue_ranges)
        logger.info("%s: fit gate=%.2f, %d hue range(s)", self.metric_name,
                   definition["gate"], len(definition["hue_ranges"]))
        return definition

    def _describe_fit(self, group):
        """
        What a human can't infer from the parameters: what this group's fit actually found.
        """
        definition = self.fits[group]
        return {"gate": definition["gate"],
                "n_hue_ranges": len(definition["hue_ranges"]),
                "thresholds": [threshold.spec() for threshold in definition["thresholds"]]}

    def prepare(self, context):
        """
        Fit one gate+hue-ranges definition per group, pooled across the reference population.

        The pooling and the fallback are `PooledPixelGroupMetric.prepare`; what
        is this metric's own is the five-step fit and the figures showing how
        each definition's gate and hue ranges were arrived at.

        Returns that base's fit record, plus this metric's own settings.
        """
        record = super().prepare(context)
        logger.info("%s fit: %d group definition(s) + 1 population-wide fallback "
                    "(population gate=%.2f, %d hue range(s))", self.metric_name,
                    len(self.fits) - 1, self.fits[POPULATION]["gate"],
                    len(self.fits[POPULATION]["hue_ranges"]))

        if context.report:
            drawn = [POPULATION] + sorted(
                (group for group in self.fits if group is not POPULATION),
                key=lambda group: -record["groups"][str(group)]["count"])[:FIGURE_GROUPS]
            for group in drawn:
                label = "population" if group is POPULATION else str(group)
                for suffix, figure in _definition_figures(self.fits[group], label):
                    context.report.figure(f"{self.metric_name}__{label}__{suffix}", figure)

        return dict(record,
                    n_chroma_bins=self.n_chroma_bins,
                    min_bin_pixels=self.min_bin_pixels,
                    hue_bandwidth=self.hue_bandwidth,
                    valley_relative_height=self.valley_relative_height,
                    max_hue_ranges=self.max_hue_ranges)

    def _score(self, segment):
        """
        Classify EVERY masked pixel of this occurrence (not a sample -- see
        module docstring step 5) against its group's fitted gate + hue
        ranges, and return the fraction in each.

        Every hue_N key is present for every occurrence scored against one
        group's definition, zero-valued where nothing fell in it, so a wide
        export has no holes that could be misread as "not measured".
        """
        if not self.fits:
            raise RuntimeError(
                f"{self.metric_name} was never fit -- group metrics are fit by "
                "their prepare() hook, which run_metrics calls for you"
            )

        pixels = masked_pixels(segment, required=False)
        if pixels is None:
            raise ValueError("empty mask")

        definition, group = self._fit_for(segment.occurrence_id)
        masks = threshold_masks(pixels, definition["thresholds"])
        result = {label: float(mask.mean()) for label, mask in masks.items()}

        if segment.panel_sink is not None:
            segment.emit_panel(threshold_panel(segment, masks, result), self.metric_name)

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
    if len(chroma) == 0:
        logger.warning("no pixels to find a chroma gate from -- using gate 0.0")
        return 0.0

    trusted = _chroma_variance_curve(chroma, hue_degrees, n_bins, min_bin_pixels)
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
            logger.debug("chroma gate stabilized at %.2f (bin %d/%d, baseline "
                        "variance %.4f)", edge, position + 1, len(trusted), baseline)
            return edge

    logger.warning("hue variance never stabilized across chroma bins -- using gate 0.0")
    return 0.0


def _chroma_variance_curve(chroma, hue_degrees, n_bins, min_bin_pixels):
    """
    [(lower chroma edge, circular hue variance), ...] for every trusted quantile bin, in
    increasing chroma order -- what the gate is read off of.
    """
    chroma = np.asarray(chroma)
    hue_degrees = np.asarray(hue_degrees)
    if len(chroma) == 0:
        return []

    edges = np.quantile(chroma, np.linspace(0.0, 1.0, n_bins + 1))
    bin_index = np.clip(np.searchsorted(edges, chroma, side="right") - 1, 0, n_bins - 1)

    trusted = []
    for index in range(n_bins):
        members = hue_degrees[bin_index == index]
        if len(members) >= min_bin_pixels:
            trusted.append((float(edges[index]), _circular_variance(members)))
    return trusted


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
    """
    The full 5-step fit over one pool of BGR pixels; step 5's thresholds are applied at score time
    by `threshold_masks`.

    Returns {"gate", "hue_ranges", "thresholds"} plus the curves they were read off of ("chroma_curve",
    "kde", "valleys"), kept for drawing.
    """
    _, chroma, hue = convert(bgr_pixels, "lch").T
    gate = _find_chroma_gate(chroma, hue, n_chroma_bins, min_bin_pixels)
    curve = _chroma_variance_curve(chroma, hue, n_chroma_bins, min_bin_pixels)

    gated_hue = hue[chroma >= gate]
    if len(gated_hue) < min_bin_pixels:
        logger.warning("fewer than %d pixels passed the chroma gate (%.2f) -- "
                       "using one hue range covering the whole circle",
                       min_bin_pixels, gate)
        return {"gate": gate, "hue_ranges": [(0.0, 360.0)],
                "thresholds": fitted_thresholds(gate, [(0.0, 360.0)]), "chroma_curve": curve,
                "kde": None, "valleys": []}

    grid, density = _hue_kde(gated_hue, hue_bandwidth)
    valleys = _find_hue_valleys(grid, density, valley_relative_height, max_hue_ranges)
    logger.debug("%d/%d pixel(s) passed the chroma gate; %d valley(s) found",
                len(gated_hue), len(hue), len(valleys))
    ranges = _ranges_from_valleys(valleys)
    return {"gate": gate, "hue_ranges": ranges, "thresholds": fitted_thresholds(gate, ranges),
            "chroma_curve": curve, "kde": (grid, density), "valleys": [float(angle) for angle in valleys]}


def fitted_thresholds(gate, hue_ranges):
    """
    A fitted gate and hue arcs as colour thresholds in lch.

    Disjoint by construction and together covering every pixel: `achromatic` below the gate, then one `hue_<i>` per
    arc at or above it.

    - `gate` -- the chroma below which a pixel's hue is not trusted.
    - `hue_ranges` -- `(start, end)` arcs in degrees, covering the circle.

    Returns a list of `ColorThreshold`, `achromatic` first.
    """
    thresholds = [color_threshold("achromatic", lch_c=(None, gate))]
    for index, (start, end) in enumerate(hue_ranges):
        thresholds.append(color_threshold(f"hue_{index}", lch_h=(start, end), lch_c=(gate, None)))
    return thresholds


def _definition_figures(definition, label):
    """(suffix, Figure) pairs showing how one definition's gate and hue ranges were found."""
    drawn = []
    if definition.get("chroma_curve"):
        edges, variances = zip(*definition["chroma_curve"])
        drawn.append(("gate", figures.line_chart(
            {"hue variance": (list(edges), list(variances))},
            xlabel="chroma (bin lower edge)", ylabel="circular hue variance",
            title=f"{label}: chroma gate {definition['gate']:.1f}",
            marks={"gate": definition["gate"]})))
    if definition.get("kde") is not None:
        grid, density = definition["kde"]
        drawn.append(("hue", figures.line_chart(
            {"density": (list(grid), list(density))},
            xlabel="hue (degrees)", ylabel="KDE density",
            title=f"{label}: {len(definition['hue_ranges'])} hue range(s)",
            marks={f"{angle:.0f}": angle for angle in definition["valleys"]})))
    return drawn
