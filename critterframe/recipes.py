"""
Recipes: reproducible compositions of configured operations, and the working
representation they operate on.

Three things live here because they're three halves of one contract, and
splitting them would just mean three modules importing each other:

  Segment   -- the temporary working representation an operation reads and
               writes: an image plus its current mask. Never persisted (a
               segment is just an image and a mask, both of which already are
               persisted separately), so it carries only what an operation
               needs while a recipe is running.
  Operation -- one CONFIGURED processing action: remove_appendages(),
               segment(groundedsam2()), mean_lightness(). Configured is the
               important word -- an operation is the callable plus the exact
               parameters it will run with, which is what makes it hashable.
  Recipe    -- a reproducibly hashable configured operation chain: the ordered
               operations plus the inputs they consume, and the hash that
               identifies the whole thing.

Recipe identity includes operation order, every operation's parameters and
version, model/checkpoint identity, and the upstream derived inputs the recipe
consumes (which part it starts from, which mask table). That's what makes
processing repeat-aware: run.py asks "does this occurrence-part already have
output from this exact recipe hash?" and skips or reuses it if so, so an
interrupted run resumes instead of restarting, and rerunning an identical
config is a no-op instead of silently duplicating work.

A recipe says what a chain IS, not what is done with the result, so its `kind`
is open-ended -- "segment" and "metric" are the two that produce a run record,
and "render" (see visualization.products) hashes a transform chain to name and
identify a folder of images without recording anything at all. The run database
deliberately accepts only the first two: a render derives no data, so there is
nothing about it to keep provenance for beyond the hash that names it.

Spatial bookkeeping is Segment's other job. Persisted masks are always in the
coordinates of the ORIGINAL analysis image, but recipes freely crop, rotate,
and resize on the way to producing one. Rather than making every operation
remember how to invert itself, a Segment carries the affine mapping original
coordinates to its own current ones; operations that move pixels compose onto
it, and mask_in_original_coordinates() inverts the whole chain in one step
right before persistence.
"""

import hashlib
import json
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# The part every occurrence has unless a recipe says otherwise: the whole
# focal organism. Any number of parts are allowed per occurrence, but a project
# that never names one still gets a well-formed occurrence-part row rather than
# a null part special case running through every table.
DEFAULT_PART = "organism"

# Length of the hex digest kept as a recipe hash. Full sha256 is unwieldy in a
# parquet column and a log line; 16 hex chars is 64 bits, far past collision
# concerns for the number of recipes one project will ever run.
HASH_LENGTH = 16

# Identity affine: original coordinates and current coordinates are the same.
IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _json_default(value):
    """
    Convert common NumPy values into JSON-compatible Python values.

    value -- a value json.dumps() couldn't serialize directly (passed via its
             default= hook); only np.ndarray and np.generic are handled.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def canonical_json(value):
    """
    Serialize value to a deterministic JSON string (sorted keys, compact
    separators) so identical specs always dump identically -- which is the
    whole basis of recipe hashing, and also how metric values are stored.

    value -- Python value (dict/list/etc, possibly containing NumPy values
             handled via _json_default) to serialize.
    """
    return json.dumps(value, default=_json_default, sort_keys=True,
                      separators=(",", ":"))


def load_json(value):
    """Deserialize a JSON string column back to a Python value; None passes through."""
    return json.loads(value) if value is not None else None


def hash_spec(spec):
    """
    Hash any recipe/operation spec dict to a short, stable hex digest.

    spec -- JSON-serializable dict describing the thing being identified.
    """
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()
    return digest[:HASH_LENGTH]


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------


def compose(existing, applied):
    """
    Compose two 2x3 affines: `existing` maps original -> current, `applied`
    maps current -> new, and the result maps original -> new.

    Done in homogeneous 3x3 form and truncated back, which is just the
    arithmetic; the reason it's a named function is that every spatial
    transform needs the multiplication in this order, and getting it backwards
    produces masks that look plausible but land in the wrong place once
    inverted.
    """
    def to_3x3(matrix):
        return np.vstack([np.asarray(matrix, dtype=np.float64), [0.0, 0.0, 1.0]])

    return (to_3x3(applied) @ to_3x3(existing))[:2]


class Segment:
    """
    A temporary working representation of a masked image: the image, its
    current mask, and where both sit relative to the original analysis image.

    image           -- current working image (OpenCV BGR, or grayscale).
    mask            -- current working mask as a boolean array the same
                       height/width as image, or None before any segmentation
                       operation has produced one.
    occurrence_id   -- the occurrence this segment belongs to.
    part            -- the named biological part being derived/measured.
    project_path    -- project this came from; needed so operations can save
                       diagnostics without being handed a second argument.
    matrix          -- 2x3 affine mapping ORIGINAL analysis-image coordinates
                       to this segment's current coordinates. Identity for a
                       segment that hasn't been spatially transformed.
    original_shape  -- (height, width) of the original analysis image, needed
                       to size the canvas when inverting back to it.
    panel_sink      -- where this segment's diagnostic panels go: an object with
                       collect(occurrence_id, stage, image), normally a
                       visualization.pipeline.RunReport. None (the default) is
                       what makes every emit_panel() call a no-op without each
                       operation re-checking a flag, and is what the great
                       majority of segments run with -- panels are built for the
                       sampled few, not for everyone.
    """

    def __init__(self, image, mask=None, occurrence_id=None, part=DEFAULT_PART,
                 project_path=None, matrix=None, original_shape=None,
                 panel_sink=None):
        self.image = image
        self.mask = None if mask is None else (np.asarray(mask) > 0)
        self.occurrence_id = occurrence_id
        self.part = part
        self.project_path = project_path
        self.matrix = IDENTITY.copy() if matrix is None else np.asarray(matrix, dtype=np.float64)
        self.original_shape = original_shape if original_shape is not None \
            else image.shape[:2]
        self.panel_sink = panel_sink

    @property
    def shape(self):
        """(height, width) of the segment's current working frame."""
        return self.image.shape[:2]

    @property
    def rgb(self):
        """
        The working image as RGB -- what image models expect, while everything
        else in this package works in OpenCV's BGR.
        """
        image = np.asarray(self.image)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def require_mask(self):
        """
        The current mask, raising if there isn't one -- for metrics and
        transforms that are meaningless without a segmentation, so they fail
        with a useful message instead of a NoneType error three frames down.
        """
        if self.mask is None:
            raise ValueError(
                f"segment for occurrence {self.occurrence_id} part "
                f"'{self.part}' has no mask yet -- put a segment(...) "
                "operation before this one in the recipe"
            )
        return self.mask

    def replace(self, image=None, mask=None, applied=None):
        """
        Return a new Segment with some parts swapped out, leaving this one
        untouched -- operations return new segments rather than mutating,
        so a recipe's intermediate states stay inspectable (and a failed
        operation can't leave a half-modified segment behind).

        image   -- new working image; keeps the current one if omitted.
        mask    -- new working mask; keeps the current one if omitted. Pass
                   False to explicitly clear it.
        applied -- 2x3 affine mapping this segment's CURRENT coordinates to
                   the new one's, for an operation that moves pixels (crop,
                   rotate, resize). Composed onto the running original->current
                   mapping. Omit for operations that don't move pixels.
        """
        if mask is False:
            new_mask = None
        elif mask is None:
            new_mask = self.mask
        else:
            new_mask = mask

        return Segment(
            image=self.image if image is None else image,
            mask=new_mask,
            occurrence_id=self.occurrence_id,
            part=self.part,
            project_path=self.project_path,
            matrix=self.matrix if applied is None else compose(self.matrix, applied),
            original_shape=self.original_shape,
            panel_sink=self.panel_sink,
        )

    def for_part(self, part):
        """
        The same working state relabeled as a different part -- how a
        multi-output segmentation recipe forks one shared, preprocessed segment
        into a branch per part without redoing the shared work.
        """
        new = self.replace()
        new.part = part
        return new

    def mask_in_original_coordinates(self):
        """
        This segment's mask warped back into the ORIGINAL analysis image's
        frame, which is the only coordinate system masks are ever persisted in.

        Inverts the accumulated affine in one step, so it doesn't matter how
        many crops/rotations/resizes a recipe applied along the way, or in what
        order -- a mask a part-specific model produced inside a rotated crop
        still lands on the right pixels of the parent image, which is what
        makes parts and whole organisms comparable in the same table.
        """
        mask = self.require_mask()
        height, width = self.original_shape

        # The shape check matters as much as the matrix one: an upper-left crop
        # translates by (0, 0), so its affine IS the identity while its canvas
        # is smaller. Returning early on the matrix alone would persist a mask
        # sized to the crop, which then silently disagrees with every other
        # mask of the same occurrence.
        if np.allclose(self.matrix, IDENTITY) and mask.shape == (height, width):
            return mask

        inverse = cv2.invertAffineTransform(self.matrix)
        warped = cv2.warpAffine(mask.astype(np.uint8), inverse, (width, height),
                                flags=cv2.INTER_NEAREST)
        return warped > 0

    def emit_panel(self, image, stage):
        """
        Offer a diagnostic panel, or do nothing when nothing is listening.
        Operations call this unconditionally rather than each guarding on a
        flag.

        EMIT, not save: an operation makes a picture of what it decided and
        hands it over. Where it goes is the run's business, not the operation's.
        Today a run collecting a pipeline grid takes it as one cell (see
        visualization.pipeline); a different sink could write it as its own file
        or stream it somewhere, and no operation would change.

        image -- a DISPLAY-READY uint8 (or boolean) panel. An operation knows
                 what its own numbers mean, so it renders them; nothing
                 downstream will rescale a float array on its behalf.
        stage -- what this panel shows, e.g. "orientation", "crop_to_mask". It
                 titles the column the panel lands in, so name it for the step
                 rather than for the occurrence.
        """
        if self.panel_sink is None:
            return
        self.panel_sink.collect(self.occurrence_id, stage, image)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class Operation:
    """
    One configured processing action.

    Subclasses fix what an operation DOES with a segment (Transform rewrites
    it, Segmentation gives it a mask, Metric reads a terminal value off it);
    this base fixes what every operation must be able to say about ITSELF, so a
    recipe containing it can be hashed.

    name       -- operation identifier, e.g. "remove_appendages". Also the
                  default column/metric name and visualization subfolder.
    function   -- the callable doing the work. Called as
                  function(segment, **parameters) and returning whatever the
                  subclass's __call__ contract expects.
    parameters -- the exact settings this operation is configured with. Must be
                  JSON-serializable, since they go into the recipe hash --
                  configuration that can't be recorded can't be reproduced.
    version    -- method version, bumped by hand when an implementation changes
                  in a way that changes its output for unchanged parameters.
                  Part of the hash, so a bump correctly invalidates cached work.
    model      -- optional model backing this operation; contributes its own
                  identity() to the hash so two runs with different checkpoints
                  are never mistaken for equivalent work.
    """

    kind = "operation"

    def __init__(self, name, function, parameters=None, version="1", model=None):
        self.name = name
        self.function = function
        self.parameters = dict(parameters or {})
        self.version = str(version)
        self.model = model

    def spec(self):
        """
        The hashable description of this configured operation. Everything that
        can change the output belongs in here and nothing that can't -- a
        visualize flag, for instance, produces debug images but identical
        results, so it never reaches this.
        """
        spec = {
            "name": self.name,
            "kind": self.kind,
            "version": self.version,
            "parameters": self.parameters,
        }
        if self.model is not None:
            spec["model"] = model_identity(self.model)
        return spec

    def invoke(self, segment):
        """
        Call this operation's function with its configured parameters.

        A model, when there is one, is passed as `model=` rather than through
        `parameters` -- parameters have to be JSON-serializable to be hashable,
        and a loaded neural network isn't. The model reaches the hash via its
        own identity() instead (see model_identity).
        """
        if self.model is not None:
            return self.function(segment, model=self.model, **self.parameters)
        return self.function(segment, **self.parameters)

    def prepare(self, context):
        """
        Optional hook run ONCE before a run's per-occurrence loop starts.

        Almost every operation is self-contained per occurrence and ignores
        this. Group metrics are the exception -- they need a reference
        population fit across many occurrences before any single one can be
        scored -- so this is where that fit happens, rather than each group
        metric inventing its own way to get called once up front.

        context -- a metrics.run.RunContext: the project path, the occurrence
                   ids this run covers, and the part being processed.
        """
        return None

    def __repr__(self):
        return f"{type(self).__name__}({self.name}, {self.parameters})"


class Transform(Operation):
    """
    An operation that modifies the working image and/or mask without itself
    producing a metric.

    Its function is called as function(segment, **parameters) and returns
    (segment, info): the new working segment, plus a dict of diagnostics that
    gets recorded on the run. Transforms that move pixels are responsible for
    passing `applied` to Segment.replace() so the mapping back to original
    coordinates stays correct -- that is the one thing a transform must not
    forget.
    """

    kind = "transform"

    def __call__(self, segment):
        return self.invoke(segment)


class Segmentation(Operation):
    """
    An operation that derives or refines a mask -- from an image, from an
    existing mask, or from another part's mask.

    Automatic models and hand-drawn masks are alternative segmentations, not
    different systems: segment(groundedsam2()) and draw_mask() are both
    Segmentation operations returning (segment, info), and both feed the same
    mask table. What differs is only which one a run's recipe names, and
    therefore which recipe hash the resulting mask carries.
    """

    kind = "segment"

    def __call__(self, segment):
        return self.invoke(segment)


class Metric(Operation):
    """
    An operation producing a terminal value: a trait, a QC value, a human
    label, an embedding, a cluster assignment, an outlier score. Metrics end a
    chain -- nothing composes onto a metric's output the way operations compose
    onto a segment.

    Its function is called as function(segment, **parameters) and returns the
    value: usually a scalar, but a dict is allowed and is how a metric reports
    several related numbers at once (export gives each key its own column). The
    value must be JSON-serializable, since that's how it's stored.

    unit        -- what the value is expressed in ("px", "px2", "mm",
                   "category", "fraction", ...). Recorded alongside the value,
                   because a bare number whose unit lives only in a variable
                   name is the easiest thing in this package to misread later.
    metric_name -- what to store the value under, defaulting to the operation
                   name. Overridable so the same operation can appear twice in
                   one recipe under different configurations (mask_area() as
                   "area_px", say) without the second overwriting the first.
    """

    kind = "metric"

    def __init__(self, name, function, parameters=None, version="1", model=None,
                 unit=None, metric_name=None):
        super().__init__(name, function, parameters=parameters, version=version,
                         model=model)
        self.unit = unit
        self.metric_name = metric_name or name

    def spec(self):
        spec = super().spec()
        spec["metric_name"] = self.metric_name
        spec["unit"] = self.unit
        return spec

    def __call__(self, segment):
        return self.invoke(segment)


def model_identity(model):
    """
    A model's contribution to a recipe hash: whatever the model reports about
    which weights it actually is.

    A model may define identity() -> dict to say this precisely (see
    segmentation.groundedsam.GroundedSAM2). Anything else -- a bare
    torch.nn.Module a caller trained themselves, say -- falls back to its class
    name, which identifies the architecture but NOT the checkpoint. That's the
    honest answer for an object that never told us its checkpoint, but it does
    mean two different fine-tunes of the same class hash alike; give a
    caller-supplied model an identity() when that matters.
    """
    if hasattr(model, "identity"):
        return model.identity()
    return {"class": type(model).__name__}


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


class Recipe:
    """
    An immutable, hashable specification of a configured operation chain and
    the inputs it consumes.

    kind       -- what the chain is for. "segment" and "metric" are the two that
                  execute as runs and get a run record (records.runs.RUN_KINDS);
                  "render" identifies a chain of transforms whose output is
                  images on disk rather than data (visualization.products). New
                  kinds are cheap -- the hash doesn't care -- but a kind with no
                  runner is just a name.
    name       -- the run name a caller gave, e.g. "body_parts" or "traits".
                  Part of identity, so deliberately rerunning the same
                  operations under a new name records a genuinely new run
                  rather than being skipped as already done.
    operations -- ordered list of Operations. For a metric recipe this is the
                  transforms followed by the metrics, in the order they run.
    part       -- the part this recipe produces (segmentation) or measures
                  (metrics).
    from_part  -- the upstream part whose mask this recipe starts from, if any.
                  Part of identity because a recipe that refines the organism
                  mask is not the same recipe when pointed at a wing.
    inputs     -- any other upstream derived dependency worth pinning into
                  identity, e.g. {"masks": "reference"} for a metric recipe
                  measuring reference masks instead of canonical ones.
    """

    def __init__(self, kind, name, operations, part=DEFAULT_PART, from_part=None,
                 inputs=None):
        self.kind = kind
        self.name = name
        self.operations = list(operations)
        self.part = part
        self.from_part = from_part
        self.inputs = dict(inputs or {})

    def spec(self):
        """The full, hashable description of this recipe."""
        return {
            "kind": self.kind,
            "name": self.name,
            "part": self.part,
            "from_part": self.from_part,
            "inputs": self.inputs,
            "operations": [operation.spec() for operation in self.operations],
        }

    @property
    def hash(self):
        """
        Stable identity of this recipe. Two recipes hash alike exactly when
        running them would do the same work, which is what lets a run skip
        occurrence-parts that already carry this hash.
        """
        return hash_spec(self.spec())

    def operations_of(self, kind):
        """The operations of one kind, in order -- e.g. the transforms of a metric recipe."""
        return [operation for operation in self.operations if operation.kind == kind]

    def prepare_all(self, context):
        """Run every operation's prepare() hook once, before the per-occurrence loop."""
        for operation in self.operations:
            operation.prepare(context)

    def __repr__(self):
        names = ", ".join(operation.name for operation in self.operations)
        return f"Recipe({self.kind}:{self.name} part={self.part} [{names}] {self.hash})"


def describe(recipe):
    """
    A recipe's spec plus its hash, ready to be stored on a run record -- the
    reproducible half of provenance. Stored in full rather than as a hash alone
    so a run stays readable years later even if the operation that produced it
    has since been renamed or deleted from the package.
    """
    return {"recipe_hash": recipe.hash, "recipe": recipe.spec()}
