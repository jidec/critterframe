"""Color thresholds fitted per group: a chroma gate and hue arcs found from the group's own pixels."""

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
    """Group metric: the fraction of this organism in each hue range its group's pixels turned out to have.

    Also reports `achromatic`, the fraction below the group's chroma gate. Fractions are
    comparable within a group, not across groups: `hue_0` is a different color in each.

    Args:
        group_col: Occurrence column to group by, e.g. `"taxon"`. None fits one definition
            for the whole project.
        transforms: Operations applied when gathering the reference pixels; pass the ones
            the run uses.
        min_group_size: Groups smaller than this share the population-wide definition.
        sample_pixels: Pixels sampled per reference occurrence for the fit. Scoring uses
            every masked pixel.
        n_chroma_bins: Equal-count chroma bins the gate search uses.
        min_bin_pixels: A chroma bin with fewer pixels than this is left out of the search.
        hue_bandwidth: Gaussian kernel bandwidth of the hue density estimate, in degrees.
        valley_relative_height: A density minimum counts as a boundary only when it is at
            most this fraction of the peak.
        max_hue_ranges: Cap on the hue ranges one group's fit may find; the deepest valleys
            are kept.
        name: Metric name.
        unit: Recorded unit.
        reference: Fit against reference masks.
    """

    def __init__(
        self,
        group_col=None,
        transforms=(),
        min_group_size=MIN_GROUP_SIZE,
        sample_pixels=PIXELS_PER_OCCURRENCE,
        n_chroma_bins=20,
        min_bin_pixels=50,
        hue_bandwidth=10.0,
        valley_relative_height=0.5,
        max_hue_ranges=8,
        name=None,
        unit="fraction",
        reference=False,
    ):
        super().__init__(
            "inductive_color_thresholds",
            self._score,
            group_col=group_col,
            transforms=transforms,
            min_group_size=min_group_size,
            sample_pixels=sample_pixels,
            reference=reference,
            unit=unit,
            metric_name=name or "inductive_color_thresholds",
        )

        self.n_chroma_bins = n_chroma_bins
        self.min_bin_pixels = min_bin_pixels
        self.hue_bandwidth = hue_bandwidth
        self.valley_relative_height = valley_relative_height
        self.max_hue_ranges = max_hue_ranges

    def spec(self):
        """Return the operation's spec, with the fit settings and transforms."""
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
        """Fit one group's pooled pixels (see `_fit_definition`)."""
        logger.debug("%s: fitting %d pooled pixel(s)", self.metric_name, len(pixels))
        definition = _fit_definition(
            pixels,
            self.n_chroma_bins,
            self.min_bin_pixels,
            self.hue_bandwidth,
            self.valley_relative_height,
            self.max_hue_ranges,
        )
        logger.info(
            "%s: fit gate=%.2f, %d hue range(s)",
            self.metric_name,
            definition["gate"],
            len(definition["hue_ranges"]),
        )
        return definition

    def _describe_fit(self, group):
        """Return what one group's fit found: its gate and hue ranges."""
        definition = self.fits[group]
        return {
            "gate": definition["gate"],
            "n_hue_ranges": len(definition["hue_ranges"]),
            "thresholds": [threshold.spec() for threshold in definition["thresholds"]],
        }

    def prepare(self, context):
        """Fit one gate and set of hue ranges per group, and draw how each was arrived at.

        Returns:
            The base class's fit record, plus this metric's settings.
        """
        record = super().prepare(context)
        logger.info(
            "%s fit: %d group definition(s) + 1 population-wide fallback "
            "(population gate=%.2f, %d hue range(s))",
            self.metric_name,
            len(self.fits) - 1,
            self.fits[POPULATION]["gate"],
            len(self.fits[POPULATION]["hue_ranges"]),
        )

        if context.report:
            drawn = [POPULATION] + sorted(
                (group for group in self.fits if group is not POPULATION),
                key=lambda group: -record["groups"][str(group)]["count"],
            )[:FIGURE_GROUPS]
            for group in drawn:
                label = "population" if group is POPULATION else str(group)
                for suffix, figure in _definition_figures(self.fits[group], label):
                    context.report.figure(f"{self.metric_name}__{label}__{suffix}", figure)

        return dict(
            record,
            n_chroma_bins=self.n_chroma_bins,
            min_bin_pixels=self.min_bin_pixels,
            hue_bandwidth=self.hue_bandwidth,
            valley_relative_height=self.valley_relative_height,
            max_hue_ranges=self.max_hue_ranges,
        )

    def _score(self, segment):
        """Return the fraction of every masked pixel in each of its group's fitted thresholds.

        Every threshold has a key, zero where nothing fell in it.
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
        result["dominant"] = max(hue_keys, key=result.get) if any(result[k] > 0 for k in hue_keys) else None
        if result["dominant"] is not None:
            result["dominant"] = int(result["dominant"].split("_")[1])
        result["group"] = group
        return result


def inductive_color_thresholds(**kwargs):
    """Operation: an `InductiveColorThresholdMetric`.

    Args:
        **kwargs: As in `InductiveColorThresholdMetric`.
    """
    return InductiveColorThresholdMetric(**kwargs)


def _circular_variance(hue_degrees):
    """Return 1 minus the mean resultant length of a set of angles: 0 when identical, near 1 when scattered."""
    if len(hue_degrees) == 0:
        return float("nan")
    radians = np.radians(hue_degrees)
    resultant = np.hypot(np.mean(np.cos(radians)), np.mean(np.sin(radians)))
    return float(1.0 - resultant)


# ---------------------------------------------------------------------------
# Step 2: the empirical chroma gate
# ---------------------------------------------------------------------------


def _find_chroma_gate(chroma, hue_degrees, n_bins, min_bin_pixels):
    """Return the chroma value past which hue stops being noise.

    The lower edge of the first trusted equal-count chroma bin from which circular hue
    variance stays at or below its high-chroma level. 0.0, with a warning, if none does.
    """
    chroma = np.asarray(chroma)
    if len(chroma) == 0:
        logger.warning("no pixels to find a chroma gate from -- using gate 0.0")
        return 0.0

    trusted = _chroma_variance_curve(chroma, hue_degrees, n_bins, min_bin_pixels)
    if not trusted:
        logger.warning("no chroma bin had >= %d pixels -- using gate 0.0", min_bin_pixels)
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
            logger.debug(
                "chroma gate stabilized at %.2f (bin %d/%d, baseline variance %.4f)",
                edge,
                position + 1,
                len(trusted),
                baseline,
            )
            return edge

    logger.warning("hue variance never stabilized across chroma bins -- using gate 0.0")
    return 0.0


def _chroma_variance_curve(chroma, hue_degrees, n_bins, min_bin_pixels):
    """Return `[(lower chroma edge, circular hue variance), ...]` for every trusted bin, by increasing chroma."""
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
    """Return a circular kernel density estimate of hue on a grid, using a wrapped Gaussian kernel."""
    grid = np.arange(grid_size, dtype=np.float64) * (360.0 / grid_size)
    hue_degrees = np.asarray(hue_degrees, dtype=np.float64)
    if len(hue_degrees) == 0:
        return grid, np.zeros(grid_size)

    delta = grid[:, None] - hue_degrees[None, :]
    delta = ((delta + 180.0) % 360.0) - 180.0
    density = np.exp(-0.5 * (delta / bandwidth) ** 2).sum(axis=1)
    return grid, density


def _find_hue_valleys(grid, density, valley_relative_height, max_ranges):
    """Return the grid angles of the low-density gaps between hue modes.

    A valley is a circular local minimum at or below `valley_relative_height` of the peak.
    A low stretch gives one valley, and only the deepest `max_ranges - 1` are kept.
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
    """Return circular `(start, end)` arcs, one per gap between consecutive valley angles.

    Fewer than two valleys give one arc covering the whole circle: a single valley
    separates nothing.
    """
    if len(valley_angles) < 2:
        return [(0.0, 360.0)]

    valleys = sorted(float(angle) for angle in valley_angles)
    return [(valleys[index], valleys[(index + 1) % len(valleys)]) for index in range(len(valleys))]


# The fit, run once per group on pooled reference pixels:
#   1. Convert to Lab and take chroma C = sqrt(a^2 + b^2) per pixel. Chroma, not
#      HSV saturation: it is better behaved and, unlike hue, not circular.
#   2. Find the chroma gate: bin pixels by chroma and take the circular hue
#      variance within each bin. It is high and noisy at low chroma, where a
#      near-gray pixel's hue means little, and settles past some value. That
#      value is the gate, read from this population and not a fixed cutoff.
#   3. Keep only pixels at or above the gate as the subset hue is read from.
#   4. Find hue boundaries as the valleys of a circular kernel density estimate
#      of that subset's hue, however many modes there turn out to be.
#   5. Express the gate and arcs as ColorThresholds in lch: `achromatic` below
#      the gate, one `hue_<i>` per arc. Scoring applies them to every masked
#      pixel, gated or not, so the fractions describe the whole organism.
def _fit_definition(
    bgr_pixels, n_chroma_bins, min_bin_pixels, hue_bandwidth, valley_relative_height, max_hue_ranges
):
    """Fit a chroma gate and hue ranges to one pool of BGR pixels.

    Returns:
        `{"gate", "hue_ranges", "thresholds"}`, plus the curves they were read from
        (`"chroma_curve"`, `"kde"`, `"valleys"`) for drawing.
    """
    _, chroma, hue = convert(bgr_pixels, "lch").T
    gate = _find_chroma_gate(chroma, hue, n_chroma_bins, min_bin_pixels)
    curve = _chroma_variance_curve(chroma, hue, n_chroma_bins, min_bin_pixels)

    gated_hue = hue[chroma >= gate]
    if len(gated_hue) < min_bin_pixels:
        logger.warning(
            "fewer than %d pixels passed the chroma gate (%.2f) -- "
            "using one hue range covering the whole circle",
            min_bin_pixels,
            gate,
        )
        return {
            "gate": gate,
            "hue_ranges": [(0.0, 360.0)],
            "thresholds": fitted_thresholds(gate, [(0.0, 360.0)]),
            "chroma_curve": curve,
            "kde": None,
            "valleys": [],
        }

    grid, density = _hue_kde(gated_hue, hue_bandwidth)
    valleys = _find_hue_valleys(grid, density, valley_relative_height, max_hue_ranges)
    logger.debug(
        "%d/%d pixel(s) passed the chroma gate; %d valley(s) found", len(gated_hue), len(hue), len(valleys)
    )
    ranges = _ranges_from_valleys(valleys)
    return {
        "gate": gate,
        "hue_ranges": ranges,
        "thresholds": fitted_thresholds(gate, ranges),
        "chroma_curve": curve,
        "kde": (grid, density),
        "valleys": [float(angle) for angle in valleys],
    }


def fitted_thresholds(gate, hue_ranges):
    """Return a fitted gate and hue arcs as color thresholds in lch.

    They are disjoint and together cover every pixel: `achromatic` below the gate, then
    one `hue_<i>` per arc.

    Args:
        gate: Chroma below which a pixel's hue is not trusted.
        hue_ranges: `(start, end)` arcs in degrees, covering the circle.
    """
    thresholds = [color_threshold("achromatic", lch_c=(None, gate))]
    for index, (start, end) in enumerate(hue_ranges):
        thresholds.append(color_threshold(f"hue_{index}", lch_h=(start, end), lch_c=(gate, None)))
    return thresholds


def _definition_figures(definition, label):
    """Return `(suffix, Figure)` pairs showing how one definition's gate and hue ranges were found."""
    drawn = []
    if definition.get("chroma_curve"):
        edges, variances = zip(*definition["chroma_curve"])
        drawn.append(
            (
                "gate",
                figures.line_chart(
                    {"hue variance": (list(edges), list(variances))},
                    xlabel="chroma (bin lower edge)",
                    ylabel="circular hue variance",
                    title=f"{label}: chroma gate {definition['gate']:.1f}",
                    marks={"gate": definition["gate"]},
                ),
            )
        )
    if definition.get("kde") is not None:
        grid, density = definition["kde"]
        drawn.append(
            (
                "hue",
                figures.line_chart(
                    {"density": (list(grid), list(density))},
                    xlabel="hue (degrees)",
                    ylabel="KDE density",
                    title=f"{label}: {len(definition['hue_ranges'])} hue range(s)",
                    marks={f"{angle:.0f}": angle for angle in definition["valleys"]},
                ),
            )
        )
    return drawn
