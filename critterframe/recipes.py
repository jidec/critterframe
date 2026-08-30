"""
Classes jointly implementing the recipes contract: Segment, Recipe, Operation
(Transform, Segmentation, Metric), plus hashing.

  Segment   -- the working representation an operation reads and writes: an
               image plus its current mask. Never persisted.
  Operation -- one configured action, e.g. remove_appendages(),
               segment(groundedsam2()). Configured is the point: the callable
               plus the exact parameters it runs with, which is what makes it
               hashable.
  Recipe    -- an ordered chain of operations plus the inputs they consume, and
               the hash identifying the whole thing.

A recipe's hash covers operation order, every operation's parameters, version,
and model identity, and the upstream inputs it consumes. That is what makes
processing repeat-aware.

Segment's other job is spatial bookkeeping: persisted masks are always in
original image coordinates, so a Segment carries the affine mapping original
coordinates to its current ones, and mask_in_original_coordinates() inverts the
whole chain in one step before persistence.
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

    Named because every spatial transform needs the multiplication in this
    order; backwards produces masks that look plausible and land in the wrong
    place once inverted.
    """
    def to_3x3(matrix):
        return np.vstack([np.asarray(matrix, dtype=np.float64), [0.0, 0.0, 1.0]])

    return (to_3x3(applied) @ to_3x3(existing))[:2]


class Segment:
    """
    A working masked image: the image, its current mask, and where both sit
    relative to the original analysis image.

    image           -- current working image, BGR or grayscale.
    mask            -- current working mask, a boolean array matching image's
                       height/width, or None before any segmentation.
    occurrence_id   -- the occurrence this segment belongs to.
    part            -- the named biological part being derived or measured.
    project_path    -- project this came from, so operations can save
                       diagnostics without a second argument.
    matrix          -- 2x3 affine mapping original analysis-image coordinates
                       to this segment's. Identity if untransformed.
    original_shape  -- (height, width) of the original image, for sizing the
                       canvas when inverting back to it.
    panel_sink      -- where diagnostic panels go: an object with
                       collect(occurrence_id, stage, image), normally a
                       RunReport. None makes every emit_panel() a no-op, which
                       is what most segments run with -- panels are for the
                       sampled few.
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
        untouched, so intermediate states stay inspectable and a failed
        operation leaves nothing half-modified.

        image   -- new working image; keeps the current one if omitted.
        mask    -- new working mask; keeps the current one if omitted. False
                   clears it.
        applied -- 2x3 affine mapping this segment's current coordinates to the
                   new one's, for an operation that moves pixels. Composed onto
                   the running original->current mapping. Omit otherwise.
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
        This segment's mask warped back into the original analysis image's
        frame, the only coordinate system masks are persisted in.

        Inverts the accumulated affine in one step, so however many crops,
        rotations and resizes a recipe applied, a mask found inside a rotated
        crop still lands on the right pixels of the parent image.
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
        Operations call this unconditionally rather than guarding on a flag.

        Emit, not save: an operation draws what it decided and hands it over;
        where it goes is the run's business.

        image -- a display-ready uint8 or boolean panel. Nothing downstream
                 will rescale a float array on the operation's behalf.
        stage -- what this panel shows, e.g. "orientation". It titles the
                 column, so name it for the step rather than the occurrence.
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

    Subclasses fix what an operation does with a segment; this base fixes what
    every operation says about itself, so a recipe containing it can be hashed.

    name       -- operation identifier, e.g. "remove_appendages". Also the
                  default column/metric name and visualization subfolder.
    function   -- the callable doing the work, called as
                  function(segment, **parameters).
    parameters -- the exact settings this operation runs with. Must be
                  JSON-serializable, since they go into the recipe hash.
    version    -- method version, bumped by hand when an implementation changes
                  its output for unchanged parameters, so cached work is
                  correctly invalidated.
    model      -- optional model backing this operation; contributes its own
                  identity() to the hash.
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
        The hashable description of this configured operation.

        Everything that can change the output belongs here and nothing that
        can't -- a visualize flag produces debug images but identical results.
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

        A model is passed as `model=` rather than through `parameters`, which
        must stay JSON-serializable; it reaches the hash via its own
        identity().
        """
        if self.model is not None:
            return self.function(segment, model=self.model, **self.parameters)
        return self.function(segment, **self.parameters)

    def prepare(self, context):
        """
        Optional hook run once before a run's per-occurrence loop.

        Almost every operation ignores this. Group metrics are the exception:
        they fit a reference population before any occurrence can be scored.

        context -- a metrics.run.RunContext: project path, the occurrence ids
                   this run covers, and the part being processed.
        """
        return None

    def __repr__(self):
        return f"{type(self).__name__}({self.name}, {self.parameters})"


class Transform(Operation):
    """
    An operation that changes the working image and/or mask without producing
    a value.

    Returns (segment, info): the new segment, plus diagnostics recorded on the
    run. A transform that moves pixels MUST pass `applied` to Segment.replace(),
    or the mapping back to original coordinates is wrong.
    """

    kind = "transform"

    def __call__(self, segment):
        return self.invoke(segment)


class Segmentation(Operation):
    """
    An operation that derives or refines a mask, from an image or an existing
    mask.

    Automatic models and hand-drawn masks are alternative segmentations, not
    different systems: segment(groundedsam2()) and draw_mask() both return
    (segment, info) and feed the same mask table.
    """

    kind = "segment"

    def __call__(self, segment):
        return self.invoke(segment)


class Metric(Operation):
    """
    An operation producing a terminal value: a trait, a QC value, a human
    label, an embedding, a cluster assignment, an outlier score. Metrics end a
    chain.

    Returns the value, usually a scalar; a dict reports several related numbers
    at once and export gives each key its own column. Must be
    JSON-serializable, since that is how it is stored.

    unit        -- what the value is expressed in, e.g. "px", "px2", "category".
                   Recorded alongside the value, since a bare number whose unit
                   lives in a variable name is easy to misread later.
    metric_name -- what to store the value under, defaulting to the operation
                   name. Override so one operation can appear twice in a recipe
                   without the second overwriting the first.
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
    A model's contribution to a recipe hash: whatever it reports about which
    weights it is.

    A model may define identity() -> dict to say so precisely. Anything else
    falls back to its class name, which identifies the architecture but not the
    checkpoint -- so two fine-tunes of one class hash alike. Give a model an
    identity() when that matters.
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

    kind       -- what the chain is for. "segment" and "metric" execute as runs
                  and get a run record; "render" identifies a transform chain
                  whose output is images rather than data.
    name       -- the run name, e.g. "traits". Part of identity, so rerunning
                  the same operations under a new name records a new run.
    operations -- ordered Operations; for a metric recipe, transforms then
                  metrics.
    part       -- the part this recipe produces or measures.
    from_part  -- the upstream part whose mask this starts from, if any. In
                  identity, since refining the organism mask is not the same
                  recipe pointed at a wing.
    inputs     -- any other upstream dependency worth pinning into identity,
                  e.g. {"masks": "reference"}.
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
