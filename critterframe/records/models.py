"""The registry of trained models: checkpoint fingerprints and RegisteredModel. Provenance only; loads nothing."""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from ..storage.jsonfiles import read_json, write_json
from ..project import paths
from ..core.recipes import hash_spec
from .occurrences import ids_record

logger = logging.getLogger(__name__)

# A registered name is a key in a JSON file, a thing typed into scripts, and
# often a filename stem, so it is kept to what all three tolerate.
_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Read size for checkpoint hashing. Checkpoints are hundreds of megabytes to
# tens of gigabytes; streaming keeps registration off the heap.
_CHUNK = 1024 * 1024


def register_model(
    project_path,
    name,
    path=None,
    task=None,
    framework=None,
    base_model=None,
    training_data=None,
    training_splits=None,
    parameters=None,
    notes=None,
    fingerprint=True,
):
    """Record a trained model in the project.

    Registering an existing name replaces its record, with a loud log line when the
    fingerprint changed.

    Args:
        project_path: Project the model belongs to.
        name: The model's name, e.g. `"dragonfly_segmenter_v1"`.
        path: Checkpoint file or directory, absolute or relative to the project. None for
            weights the package can't see, e.g. a hosted endpoint.
        task: What it does, e.g. `"segment"`, `"embedding"`. Free text.
        framework: What it was trained with, e.g. `"torch"`.
        base_model: What it was fine-tuned from, e.g. `"sam2_hiera_large"`.
        training_data: A directory written by `export_training_data`, its `dataset.json`,
            or a dict of your own.
        training_splits: `{split name: occurrence ids}` for a model trained without an
            export; stored as counts and digests.
        parameters: Dict of training settings, stored as given.
        notes: Free text.
        fingerprint: False skips hashing the checkpoint, with a warning: replacing the file
            then no longer changes the recipe hash.

    Returns:
        A `RegisteredModel` with no network attached.
    """
    if not _VALID_NAME.match(str(name)):
        raise ValueError(
            f"model name {name!r} must start with a letter or digit and hold "
            "only letters, digits, dot, dash, and underscore -- it is a "
            "registry key and usually a filename"
        )

    record = {
        "name": str(name),
        "task": task,
        "framework": framework,
        "base_model": base_model,
        "notes": notes,
        "parameters": dict(parameters or {}),
        "training_data": _training_data_record(project_path, training_data, training_splits),
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    record.update(_checkpoint_record(project_path, path, fingerprint))

    registry = _load_registry(project_path)
    previous = registry.get(str(name))
    if previous and previous.get("fingerprint") != record["fingerprint"]:
        logger.warning(
            "model '%s' re-registered with different weights (%s -> %s): every "
            "recipe using it now hashes differently, so its masks and metrics "
            "will be recomputed on the next run",
            name,
            previous.get("fingerprint"),
            record["fingerprint"],
        )

    registry[str(name)] = record
    _save_registry(project_path, registry)
    logger.info("registered model '%s' (%s%s)", name, record["fingerprint"], f", {task}" if task else "")
    return RegisteredModel(record, project_path)


def load_model(project_path, name):
    """Return the `RegisteredModel` for a name, with no network attached.

    Args:
        project_path: Project to read from.
        name: The model's name.

    Raises:
        KeyError: If the project has no such model.
    """
    registry = _load_registry(project_path)
    if str(name) not in registry:
        raise KeyError(
            f"no model named '{name}' in "
            f"{paths.models_registry_path(project_path)} "
            f"(registered: {sorted(registry)})"
        )
    return RegisteredModel(registry[str(name)], project_path)


def load_and_attach(project_path, name, factory, **extra_kwargs):
    """Load a registered model and attach a freshly built runtime to it.

    Args:
        project_path: Project to read from.
        name: The model's name.
        factory: Called as `factory(checkpoint_path, **parameters, **extra_kwargs)`, where
            `parameters` is what `register_model(parameters=)` stored.
        **extra_kwargs: Passed to `factory` after the stored parameters, e.g. `device=`.

    Returns:
        A `RegisteredModel` with `factory`'s result attached.
    """
    registered = load_model(project_path, name)
    parameters = dict(registered.record.get("parameters") or {})
    parameters.update(extra_kwargs)
    return registered.attach(factory(registered.path, **parameters))


def list_models(project_path):
    """Return every registered model as `{name: record}`."""
    return _load_registry(project_path)


def unregister_model(project_path, name):
    """Remove a model's registry entry and return its record.

    The checkpoint, and the masks and metrics it produced, are left as they are.

    Args:
        project_path: Project to edit.
        name: The model's name.
    """
    registry = _load_registry(project_path)
    record = registry.pop(str(name), None)
    if record is None:
        raise KeyError(f"no model named '{name}' to unregister")
    _save_registry(project_path, registry)
    logger.info("unregistered model '%s' (its checkpoint and results are untouched)", name)
    return record


def _load_registry(project_path):
    """Return the registry as `{name: record}`; empty when nothing is registered."""
    return (read_json(paths.models_registry_path(project_path), default={}) or {}).get("models", {})


def _save_registry(project_path, registry):
    """Write the whole registry, atomically."""
    write_json(paths.models_registry_path(project_path), {"models": registry})
    return registry


class RegisteredModel:
    """A model's provenance, optionally bound to a loaded network.

    `identity()` answers from the record; every other attribute is forwarded to the
    attached network.

    Args:
        record: The registry entry.
        project_path: Project it was read from.
        runtime: The loaded network, or None until `attach` supplies one.
    """

    def __init__(self, record, project_path, runtime=None):
        self.record = dict(record)
        self.project_path = project_path
        self.runtime = runtime

    @property
    def name(self):
        """Return the name the model is registered under."""
        return self.record["name"]

    @property
    def path(self):
        """Return the checkpoint as an absolute Path, or None for a model with no local weights."""
        return paths.resolve_in_project(self.project_path, self.record.get("path"))

    def identity(self):
        """Return what this model contributes to a recipe hash: its fingerprint and name.

        The path is left out, so a copied checkpoint is the same model.
        """
        return {
            "class": "RegisteredModel",
            "name": self.record["name"],
            "fingerprint": self.record.get("fingerprint"),
        }

    def attach(self, runtime):
        """Return a new `RegisteredModel` bound to a loaded network.

        Args:
            runtime: An object meeting the contract of the operation it is used in:
                `predict()` for `segment()`, `embed()` for an embedding metric.
        """
        return RegisteredModel(self.record, self.project_path, runtime)

    def __getattr__(self, attribute):
        """Forward an attribute this class does not define to the attached network.

        Raises `AttributeError` when nothing is attached: `hasattr()` is how a run asks whether
        a model has `visualize()`.
        """
        runtime = self.__dict__.get("runtime")
        if runtime is None:
            raise AttributeError(
                f"registered model '{self.__dict__['record']['name']}' has "
                f"provenance but no loaded network, so it has no "
                f"'{attribute}' -- load the checkpoint yourself and pass it to "
                ".attach()"
            )
        return getattr(runtime, attribute)

    def __repr__(self):
        state = "attached" if self.runtime is not None else "provenance only"
        return f"RegisteredModel({self.record['name']}, {self.record.get('fingerprint')}, {state})"


def _checkpoint_record(project_path, path, fingerprint):
    """Return the stored path (relative when inside the project) and fingerprint of a checkpoint."""
    if path is None:
        logger.warning(
            "model registered with no checkpoint path -- its identity rests on "
            "its name alone, so reusing the name for different weights would "
            "not change any recipe hash"
        )
        return {"path": None, "fingerprint": None, "fingerprint_method": None, "size_bytes": None}

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = paths.project_dir(project_path) / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"no checkpoint at {resolved}")

    stored = paths.relative_to_project(project_path, resolved)

    size = _size_of(resolved)
    if not fingerprint:
        logger.warning(
            "registering '%s' without a fingerprint -- replacing the file "
            "later will not change any recipe hash, so results measured from "
            "the old weights would keep counting as current",
            resolved.name,
        )
        return {"path": stored, "fingerprint": None, "fingerprint_method": None, "size_bytes": size}

    logger.info("fingerprinting %s (%.1f MB)", resolved.name, size / 1e6)
    return {
        "path": stored,
        "fingerprint": fingerprint_file(resolved),
        "fingerprint_method": "sha256",
        "size_bytes": size,
    }


def fingerprint_file(path):
    """Return a short digest of a checkpoint's contents.

    A directory is hashed from its sorted `(relative path, file digest)` pairs. Slow on a
    large checkpoint, so a caller should cache the result.

    Args:
        path: Checkpoint file or directory.
    """
    path = Path(path)
    if path.is_file():
        return hash_spec({"file": _sha256(path)})

    files = sorted(item for item in path.rglob("*") if item.is_file())
    return hash_spec(
        {"dir": [[str(item.relative_to(path)).replace("\\", "/"), _sha256(item)] for item in files]}
    )


def _sha256(path):
    """Return the sha256 of one file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _size_of(path):
    """Return the bytes on disk, summed over a directory's files."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _training_data_record(project_path, training_data, training_splits):
    """Return what the model was trained on, as an export's record or as counts and id digests."""
    record = {}

    if training_data is not None:
        if isinstance(training_data, dict):
            record["dataset"] = training_data
        else:
            record["dataset"] = _read_dataset_record(project_path, training_data)

    if training_splits is not None:
        record["splits"] = {name: ids_record(ids) for name, ids in training_splits.items()}

    return record or None


def _read_dataset_record(project_path, training_data):
    """Read an export's `dataset.json`, given it or its directory, and record where it came from."""
    location = Path(training_data)
    if not location.is_absolute():
        candidate = paths.project_dir(project_path) / location
        location = candidate if candidate.exists() else location

    dataset_path = location / "dataset.json" if location.is_dir() else location
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"no dataset record at {dataset_path} -- training_data should be a "
            "directory written by export_training_data(), its dataset.json, or "
            "a dict of your own"
        )

    with dataset_path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)

    record["source"] = paths.relative_to_project(project_path, dataset_path.parent)
    return record
