"""The recipe contract: Segment, Operation (Transform, Segmentation, Metric), Recipe, and hashing."""

import difflib
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


def json_default(value):
    """Convert a NumPy array or scalar into a JSON-compatible value, for `json.dumps(default=)`."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def canonical_json(value):
    """Serialize a value to deterministic JSON: sorted keys, compact separators.

    Args:
        value: A JSON-serializable value, possibly holding NumPy values.
    """
    return json.dumps(value, default=json_default, sort_keys=True, separators=(",", ":"))


def load_json(value):
    """Deserialize a JSON string; None passes through."""
    return json.loads(value) if value is not None else None


def recorded_callable(value):
    """Return a callable's name, for a spec that gets hashed.

    Args:
        value: A callable, a string, or None; the last two pass through.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [recorded_callable(item) for item in value]
    return getattr(value, "__qualname__", "<callable>")


def recorded_rules(rules):
    """Return a `{column: values}` rule set in a hashable, order-independent form.

    Args:
        rules: `{column: value}` or `{column: values}`, or None.
    """
    recorded = {}
    for column, values in (rules or {}).items():
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set, frozenset)):
            recorded[column] = values
            continue
        recorded[column] = sorted(values, key=str)
    return recorded


def hash_spec(spec):
    """Hash a spec to a 16-character hex digest of its canonical JSON.

    Args:
        spec: JSON-serializable value describing what is being identified.
    """
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()
    return digest[:HASH_LENGTH]


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------


def _compose(existing, applied):
    """Compose two 2x3 affines into one mapping original coordinates to the new frame.

    `existing` maps original to current and `applied` maps current to new. The order
    matters: reversed, masks look plausible and land in the wrong place once inverted.
    """

    def to_3x3(matrix):
        return np.vstack([np.asarray(matrix, dtype=np.float64), [0.0, 0.0, 1.0]])

    return (to_3x3(applied) @ to_3x3(existing))[:2]


class Segment:
    """A working masked image: the image, its mask, and where both sit in the original image.

    Args:
        image: Working image, BGR or grayscale.
        mask: Boolean array matching the image's height and width, or None before any
            segmentation.
        occurrence_id: The occurrence this segment belongs to.
        part: The part being derived or measured.
        project_path: Project this came from.
        matrix: 2x3 affine mapping original image coordinates to this segment's; identity
            if None.
        original_shape: `(height, width)` of the original image.
        panel_sink: Where diagnostic panels go, an object with
            `collect(occurrence_id, stage, image)`; None discards them.
        original_image: The image the segment started from, kept by reference through
            every transform. Defaults to `image` when no `matrix` is given.
    """

    def __init__(
        self,
        image,
        mask=None,
        occurrence_id=None,
        part=DEFAULT_PART,
        project_path=None,
        matrix=None,
        original_shape=None,
        panel_sink=None,
        original_image=None,
    ):
        if original_image is None and matrix is None:
            original_image = image
        self.original_image = original_image
        self.image = image
        self.mask = None if mask is None else (np.asarray(mask) > 0)
        self.occurrence_id = occurrence_id
        self.part = part
        self.project_path = project_path
        self.matrix = IDENTITY.copy() if matrix is None else np.asarray(matrix, dtype=np.float64)
        self.original_shape = original_shape if original_shape is not None else image.shape[:2]
        self.panel_sink = panel_sink

    @property
    def shape(self):
        """Return `(height, width)` of the working frame."""
        return self.image.shape[:2]

    @property
    def rgb(self):
        """Return the working image as RGB."""
        image = np.asarray(self.image)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def require_mask(self):
        """Return the mask.

        Raises:
            ValueError: If the segment has no mask.
        """
        if self.mask is None:
            raise ValueError(
                f"segment for occurrence {self.occurrence_id} part "
                f"'{self.part}' has no mask yet -- put a segment(...) "
                "operation before this one in the recipe"
            )
        return self.mask

    def replace(self, image=None, mask=None, applied=None):
        """Return a new Segment with some parts swapped out.

        Args:
            image: New working image; the current one if None.
            mask: New working mask; the current one if None, and False clears it.
            applied: 2x3 affine mapping this segment's coordinates to the new one's. Required
                for any operation that moves pixels.
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
            matrix=self.matrix if applied is None else _compose(self.matrix, applied),
            original_shape=self.original_shape,
            panel_sink=self.panel_sink,
            original_image=self.original_image,
        )

    def for_part(self, part):
        """Return the same working state labeled as a different part."""
        new = self.replace()
        new.part = part
        return new

    def mask_in_original_coordinates(self):
        """Return the mask warped back into the original image's frame."""
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
        warped = cv2.warpAffine(mask.astype(np.uint8), inverse, (width, height), flags=cv2.INTER_NEAREST)
        return warped > 0

    def project_mask(self, mask):
        """Return a mask in original image coordinates, warped into this segment's frame.

        Args:
            mask: Boolean array in the original image's coordinates.
        """
        height, width = self.shape
        warped = cv2.warpAffine(
            np.asarray(mask).astype(np.uint8), self.matrix, (width, height), flags=cv2.INTER_NEAREST
        )
        return warped > 0

    def emit_panel(self, image, stage):
        """Hand a diagnostic panel to the panel sink, if there is one.

        Args:
            image: A display-ready uint8 or boolean panel.
            stage: What the panel shows, e.g. `"orientation"`; it titles the grid column.
        """
        if self.panel_sink is None:
            return
        self.panel_sink.collect(self.occurrence_id, stage, image)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class Operation:
    """One configured processing action.

    Args:
        name: Operation identifier, e.g. `"remove_appendages"`.
        function: The callable doing the work, called as `function(segment, **parameters)`.
        parameters: The settings it runs with; JSON-serializable, since they are hashed.
        version: Method version, bumped by hand when the output changes for unchanged
            parameters.
        model: Model backing the operation; its `identity()` is hashed.
        note: Free text on how the operation is meant to be used. Recorded with each run,
            never hashed.
    """

    kind = "operation"

    # Whether rerunning this operation reproduces its output. Deliberately NOT
    # in spec(): it doesn't change what one execution produces, only whether an
    # earlier one may stand in for this one, and hashing it would move every
    # recipe hash already stored on disk.
    deterministic = True

    def __init__(self, name, function, parameters=None, version="1", model=None, note=None):
        if note is not None and not isinstance(note, str):
            raise TypeError(f"note must be text, got {type(note).__name__}")
        self.name = name
        self.function = function
        self.parameters = dict(parameters or {})
        self.version = str(version)
        self.model = model
        self.note = note

    def spec(self):
        """Return the hashable description of this operation: everything that changes its output."""
        spec = {
            "name": self.name,
            "kind": self.kind,
            "version": self.version,
            "parameters": self.parameters,
        }
        if self.model is not None:
            spec["model"] = _model_identity(self.model)
        return spec

    def invoke(self, segment):
        """Call the operation's function with its parameters, and its model as `model=`."""
        if self.model is not None:
            return self.function(segment, model=self.model, **self.parameters)
        return self.function(segment, **self.parameters)

    def prepare(self, context):
        """Run once before a run's per-occurrence loop.

        Args:
            context: A `metrics.run_metrics.RunContext`.

        Returns:
            None, or a JSON-serializable record the run stores beside its recipe. The default
            returns the operation's `note`, if it has one.
        """
        if self.note is None:
            return None
        logger.info("%s: %s", getattr(self, "metric_name", self.name), self.note)
        return {"note": self.note}

    def __repr__(self):
        return f"{type(self).__name__}({self.name}, {self.parameters})"


class Transform(Operation):
    """An operation that changes the working image or mask without producing a value.

    Calling it returns `(segment, info)`, the new segment and a diagnostics dict.
    """

    kind = "transform"

    def __call__(self, segment):
        """Run the operation on a segment."""
        return self.invoke(segment)


class Segmentation(Operation):
    """An operation that derives or refines a mask.

    Calling it returns `(segment, info)`.

    Args:
        name: As in `Operation`.
        function: As in `Operation`.
        parameters: As in `Operation`.
        version: As in `Operation`.
        model: As in `Operation`.
        deterministic: False where a rerun can produce a different mask, as hand-drawing does.
    """

    kind = "segment"

    def __init__(self, name, function, parameters=None, version="1", model=None, deterministic=True):
        super().__init__(name, function, parameters=parameters, version=version, model=model)
        self.deterministic = bool(deterministic)

    def __call__(self, segment):
        """Run the operation on a segment."""
        return self.invoke(segment)


class Metric(Operation):
    """An operation producing a value: a trait, a QC score, a label, an embedding.

    Calling it returns the value, a JSON-serializable scalar, list or dict. Each key of a
    dict becomes its own export column.

    Args:
        name: As in `Operation`.
        function: As in `Operation`.
        parameters: As in `Operation`.
        version: As in `Operation`.
        model: As in `Operation`.
        unit: What the value is expressed in, e.g. `"px"`, `"category"`.
        metric_name: Name the value is stored under; the operation name if None.
        requires_mask: False for a metric that judges the image itself and can run before
            segmentation. Not hashed.
        note: As in `Operation`.

    Attributes:
        input: `"segment"` for a metric computed from pixels, `"stored"` for one computed
            from stored values (see `metrics.base.stored`). Not hashed.
    """

    kind = "metric"
    input = "segment"

    def __init__(
        self,
        name,
        function,
        parameters=None,
        version="1",
        model=None,
        unit=None,
        metric_name=None,
        requires_mask=True,
        note=None,
    ):
        super().__init__(name, function, parameters=parameters, version=version, model=model, note=note)
        self.unit = unit
        self.metric_name = metric_name or name
        self.requires_mask = bool(requires_mask)

    def spec(self):
        """Return the operation's spec, with its metric name and unit."""
        spec = super().spec()
        spec["metric_name"] = self.metric_name
        spec["unit"] = self.unit
        return spec

    def __call__(self, segment):
        """Run the operation on a segment."""
        return self.invoke(segment)


def _model_identity(model):
    """Return a model's contribution to a recipe hash.

    Its `identity()` if it has one, else its class name, which does not tell two
    checkpoints of one class apart.
    """
    if hasattr(model, "identity"):
        return model.identity()
    return {"class": type(model).__name__}


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


class Recipe:
    """A hashable specification of an operation chain and the inputs it consumes.

    Args:
        kind: `"segment"` or `"metric"` for a run, `"render"` for a chain whose output is images.
        name: The run name. Recorded, but not part of the hash.
        operations: Ordered operations; for a metric recipe, transforms then metrics.
        part: The part this recipe produces or measures.
        from_part: The upstream part whose mask it starts from, or a sorted list of parts
            whose union it starts from.
        inputs: Other upstream dependencies to hash, e.g. `{"masks": "reference"}`.
    """

    def __init__(self, kind, name, operations, part=DEFAULT_PART, from_part=None, inputs=None):
        self.kind = kind
        self.name = name
        self.operations = list(operations)
        self.part = part
        self.from_part = from_part
        self.inputs = dict(inputs or {})

    def spec(self):
        """Return the full description of this recipe, including its name."""
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
        """Return the recipe's identity: the hash of its spec without `name`."""
        identity = self.spec()
        del identity["name"]
        return hash_spec(identity)

    def operations_of(self, kind):
        """Return the operations of one kind, in order."""
        return [operation for operation in self.operations if operation.kind == kind]

    def nondeterministic_operations(self):
        """Return the operations whose output a rerun would not reproduce, in order."""
        return [operation for operation in self.operations if not operation.deterministic]

    def prepare_all(self, context):
        """Run every operation's `prepare()` once.

        Returns:
            `{name: record}` for the operations that returned one, keyed by metric name where
            there is one.
        """
        prepared = {}
        for operation in self.operations:
            record = operation.prepare(context)
            if record is not None:
                prepared[getattr(operation, "metric_name", operation.name)] = record
        return prepared

    def __repr__(self):
        return f"Recipe({describe_spec(self.spec())} {self.hash})"


def describe_spec(spec):
    """Return one readable line for a recipe spec.

    Args:
        spec: The dict `Recipe.spec()` produces, or a stored run's recipe.
    """
    names = ", ".join(operation["name"] for operation in spec.get("operations", []))
    from_part = f" from_part={spec['from_part']}" if spec.get("from_part") else ""
    return f"{spec['kind']}:{spec['name']} part={spec['part']}{from_part} [{names}]"


def _shown(value, limit=70):
    """Return a spec value as short text."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _operation_key(operation):
    """Return what makes two operations the same step when lining two chains up."""
    return (operation.get("name"), operation.get("metric_name"))


def _operation_shown(operation):
    """Return one operation as `kind name(param=value, ...)`."""
    parameters = ", ".join(
        f"{key}={_shown(value, 30)}" for key, value in (operation.get("parameters") or {}).items()
    )
    name = operation.get("metric_name") or operation.get("name")
    return f"{operation.get('kind', 'operation')} {name}({parameters})"


def _dict_changes(label, old, new):
    """Return one line per key added, removed or changed between two dicts."""
    old, new = old or {}, new or {}
    lines = []
    for key in sorted(set(old) | set(new), key=str):
        if key not in old:
            lines.append(f"{label}{key} added: {_shown(new[key])}")
        elif key not in new:
            lines.append(f"{label}{key} removed (was {_shown(old[key])})")
        elif old[key] != new[key]:
            lines.append(f"{label}{key}: {_shown(old[key])} -> {_shown(new[key])}")
    return lines


def describe_recipe_change(old_spec, new_spec):
    """Return what differs between two recipe specs, one readable line per difference.

    `name` is ignored, since it is not part of a recipe's identity.

    Args:
        old_spec: The dict `Recipe.spec()` produces, or a stored run's recipe.
        new_spec: The spec to compare it with.

    Returns:
        A list of lines, never empty.
    """
    old = load_json(canonical_json(old_spec))
    new = load_json(canonical_json(new_spec))

    lines = []
    for field in sorted((set(old) | set(new)) - {"name", "operations"}, key=str):
        if old.get(field) != new.get(field):
            lines.append(f"{field}: {_shown(old.get(field))} -> {_shown(new.get(field))}")

    old_operations = old.get("operations") or []
    new_operations = new.get("operations") or []
    matcher = difflib.SequenceMatcher(
        a=[_operation_key(operation) for operation in old_operations],
        b=[_operation_key(operation) for operation in new_operations],
        autojunk=False,
    )

    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            for before, after in zip(old_operations[old_start:old_end], new_operations[new_start:new_end]):
                label = (after.get("metric_name") or after.get("name")) + ": "
                for field in sorted(
                    (set(before) | set(after)) - {"name", "metric_name", "parameters", "model"}, key=str
                ):
                    if before.get(field) != after.get(field):
                        lines.append(
                            f"{label}{field} {_shown(before.get(field))} -> {_shown(after.get(field))}"
                        )
                lines += _dict_changes(label, before.get("parameters"), after.get("parameters"))
                if before.get("model") != after.get("model"):
                    if isinstance(before.get("model"), dict) and isinstance(after.get("model"), dict):
                        lines += _dict_changes(f"{label}model ", before["model"], after["model"])
                    else:
                        lines.append(
                            f"{label}model {_shown(before.get('model'))} -> {_shown(after.get('model'))}"
                        )
            continue

        for operation in old_operations[old_start:old_end]:
            lines.append(f"removed {_operation_shown(operation)}")
        for index in range(new_start, new_end):
            where = "at the start" if index == 0 else f"after {new_operations[index - 1].get('name')}"
            lines.append(f"added {_operation_shown(new_operations[index])} {where}")

    return lines or ["nothing this comparison reads -- the two specs are the same"]


def _describe(recipe):
    """Return a recipe's spec with its hash, as stored on a run record."""
    return {"recipe_hash": recipe.hash, "recipe": recipe.spec()}
