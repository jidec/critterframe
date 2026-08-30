"""
The registry of trained models: checkpoint fingerprints, RegisteredModel.

Provenance only -- nothing here loads a network, imports torch, or knows what a
checkpoint contains. Training happens outside the package; what a project
records is the join between a checkpoint and the data behind it.

    cf.register_model(project_path, "dragonfly_segmenter_v1",
                      path="models/segmenter_v1.pt", task="segment",
                      framework="torch", base_model="sam2_hiera_large")

    model = cf.load_model(project_path, "dragonfly_segmenter_v1").attach(my_net)
    cf.run_segments(project_path, steps=[cf.segment(model)])

`attach` binds a loaded network to the record and forwards predict/encode/
visualize to it while answering identity() from the registry. Identity is the
checkpoint's FINGERPRINT, not its name or path, so retraining into the same
filename moves the recipe hash and everything below it is correctly redone.
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from ..project import paths
from ..recipes import hash_spec
from .occurrences import ids_digest

logger = logging.getLogger(__name__)

# A registered name is a key in a JSON file, a thing typed into scripts, and
# often a filename stem, so it is kept to what all three tolerate.
_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Read size for checkpoint hashing. Checkpoints are hundreds of megabytes to
# tens of gigabytes; streaming keeps registration off the heap.
_CHUNK = 1024 * 1024


def register_model(project_path, name, path=None, task=None, framework=None,
                   base_model=None, training_data=None, training_splits=None,
                   parameters=None, notes=None, fingerprint=True):
    """
    Record a trained model in the project, and return it as a RegisteredModel.

    Re-registering a name replaces the record. That is logged loudly when the
    fingerprint moves, since every recipe using it then hashes differently and
    correctly redoes its work.

    project_path    -- project the model belongs to; models are registered per
                       project.
    name            -- what this model is called, e.g. "dragonfly_segmenter_v1".
    path            -- checkpoint file or directory, absolute or relative to the
                       project. Stored relative when inside it, so a copied
                       project still resolves. None for weights this package
                       can't see, e.g. a hosted endpoint.
    task            -- what it does: "segment", "embedding", "classify". Free
                       text; nothing dispatches on it.
    framework       -- what it was trained with, e.g. "torch", "ultralytics".
    base_model      -- what it was fine-tuned from, e.g. "sam2_hiera_large".
    training_data   -- directory written by export_training_data(), its
                       dataset.json, or a dict you assembled yourself.
    training_splits -- {split name: occurrence ids} for a model trained without
                       an export. Stored as counts and id digests, not lists.
    parameters      -- opaque dict of training settings, stored as given and
                       never interpreted.
    notes           -- free text.
    fingerprint     -- False skips hashing the checkpoint, for a large file on
                       slow storage. Costs the guarantee that replacing the file
                       changes the recipe hash, so it warns.

    Returns a RegisteredModel with no network attached -- call .attach(network)
    to run it.
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
        "training_data": _training_data_record(project_path, training_data,
                                               training_splits),
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    record.update(_checkpoint_record(project_path, path, fingerprint))

    registry = load_registry(project_path)
    previous = registry.get(str(name))
    if previous and previous.get("fingerprint") != record["fingerprint"]:
        logger.warning(
            "model '%s' re-registered with different weights (%s -> %s): every "
            "recipe using it now hashes differently, so its masks and metrics "
            "will be recomputed on the next run",
            name, previous.get("fingerprint"), record["fingerprint"])

    registry[str(name)] = record
    save_registry(project_path, registry)
    logger.info("registered model '%s' (%s%s)", name, record["fingerprint"],
                f", {task}" if task else "")
    return RegisteredModel(record, project_path)


def load_model(project_path, name):
    """
    The RegisteredModel for `name`, raising if the project has no such model.

    Provenance only until something is attached to it: the returned object
    knows which weights it is and nothing about how to run them.
    """
    registry = load_registry(project_path)
    if str(name) not in registry:
        raise KeyError(
            f"no model named '{name}' in "
            f"{paths.models_registry_path(project_path)} "
            f"(registered: {sorted(registry)})"
        )
    return RegisteredModel(registry[str(name)], project_path)


def list_models(project_path):
    """Every registered model as {name: record}, empty if none are registered."""
    return load_registry(project_path)


def unregister_model(project_path, name):
    """
    Forget a registered model, returning its record.

    Removes the registry entry and nothing else: the checkpoint stays on disk,
    and so do the masks and metrics it produced, which remain valid results of
    a recipe whose hash still names those weights. Forgetting the record only
    means nothing new can be run under that name.
    """
    registry = load_registry(project_path)
    record = registry.pop(str(name), None)
    if record is None:
        raise KeyError(f"no model named '{name}' to unregister")
    save_registry(project_path, registry)
    logger.info("unregistered model '%s' (its checkpoint and results are untouched)",
                name)
    return record


def load_registry(project_path):
    """
    The raw registry as {name: record}. Empty when nothing has been registered
    -- a project with no models of its own is the normal case, not an error.
    """
    registry_path = paths.models_registry_path(project_path)
    if not registry_path.exists():
        return {}
    with registry_path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("models", {})


def save_registry(project_path, registry):
    """Write the whole registry, replacing whatever was there."""
    registry_path = paths.models_registry_path(project_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8") as handle:
        json.dump({"models": registry}, handle, indent=2, sort_keys=True)
    return registry


class RegisteredModel:
    """
    A model's provenance, optionally bound to a loaded network.

    identity() answers from the record, so a recipe hash carries the checkpoint
    fingerprint; everything else forwards to the attached network, so this is
    accepted anywhere a model is.

    record       -- the registry entry.
    project_path -- project it was read from, so `path` resolves.
    runtime      -- the loaded network, or None until attach() supplies one.
    """

    def __init__(self, record, project_path, runtime=None):
        self.record = dict(record)
        self.project_path = project_path
        self.runtime = runtime

    @property
    def name(self):
        return self.record["name"]

    @property
    def path(self):
        """
        The checkpoint as an absolute Path, or None for a model with no local
        weights. Relative records resolve against the project, which is what
        makes a copied project still find its own models.
        """
        stored = self.record.get("path")
        if stored is None:
            return None
        stored = Path(stored)
        return stored if stored.is_absolute() else paths.project_dir(self.project_path) / stored

    def identity(self):
        """
        What this model contributes to a recipe hash: the weights' fingerprint,
        plus the name for readability.

        The path is absent on purpose -- two copies of one checkpoint are the
        same model and must hash alike, or moving a project would invalidate
        every mask in it. With no fingerprint, only the name is left.
        """
        return {
            "class": "RegisteredModel",
            "name": self.record["name"],
            "fingerprint": self.record.get("fingerprint"),
        }

    def attach(self, runtime):
        """
        Bind a loaded network to this record, returning a new RegisteredModel.

        New rather than mutated, so one record can back several loaded networks
        -- a CPU copy and a GPU copy, say -- without either changing the other.

        runtime -- whatever meets the contract of the operation it is used in:
                   predict() for segment(), encode() for an embedding metric.
        """
        return RegisteredModel(self.record, self.project_path, runtime)

    def __getattr__(self, attribute):
        """
        Forward anything this class does not define to the attached network.

        Raises AttributeError, not something louder, when nothing is attached:
        hasattr() is how a run asks whether a model has visualize().
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
        return (f"RegisteredModel({self.record['name']}, "
                f"{self.record.get('fingerprint')}, {state})")


def _checkpoint_record(project_path, path, fingerprint):
    """
    The stored-path and fingerprint half of a record.

    Stored relative to the project when the checkpoint is inside it: a project
    is meant to be copyable, and an absolute path baked into its registry is
    the thing that breaks first when it moves to a cluster.
    """
    if path is None:
        logger.warning(
            "model registered with no checkpoint path -- its identity rests on "
            "its name alone, so reusing the name for different weights would "
            "not change any recipe hash")
        return {"path": None, "fingerprint": None, "fingerprint_method": None,
                "size_bytes": None}

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = paths.project_dir(project_path) / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"no checkpoint at {resolved}")

    stored = _relative_to_project(project_path, resolved)

    size = _size_of(resolved)
    if not fingerprint:
        logger.warning(
            "registering '%s' without a fingerprint -- replacing the file "
            "later will not change any recipe hash, so results measured from "
            "the old weights would keep counting as current", resolved.name)
        return {"path": stored, "fingerprint": None,
                "fingerprint_method": None, "size_bytes": size}

    logger.info("fingerprinting %s (%.1f MB)", resolved.name, size / 1e6)
    return {
        "path": stored,
        "fingerprint": _fingerprint(resolved),
        "fingerprint_method": "sha256",
        "size_bytes": size,
    }


def _relative_to_project(project_path, target):
    """
    A path as the registry should store it: relative to the project when it
    sits inside, absolute otherwise, posix-separated either way so a registry
    written on Windows reads on Linux.
    """
    absolute = Path(target).resolve()
    try:
        return absolute.relative_to(paths.project_dir(project_path).resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def _fingerprint(path):
    """
    A short digest of the weights themselves.

    A directory is hashed as its sorted (relative path, file digest) pairs, so
    a checkpoint saved as a folder of shards is identified as precisely as a
    single file, and a file appearing or moving inside it changes the answer.
    Truncated to recipe-hash length, which is what it feeds.
    """
    path = Path(path)
    if path.is_file():
        return hash_spec({"file": _sha256(path)})

    files = sorted(item for item in path.rglob("*") if item.is_file())
    return hash_spec({"dir": [[str(item.relative_to(path)).replace("\\", "/"),
                               _sha256(item)] for item in files]})


def _sha256(path):
    """Stream one file through sha256 -- checkpoints do not fit comfortably in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _size_of(path):
    """Bytes on disk, summed over a directory checkpoint's files."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _training_data_record(project_path, training_data, training_splits):
    """
    What the model was trained on, as something small enough to keep and exact
    enough to check.

    An exported dataset already answers this -- export_training_data() writes
    dataset.json holding the split id digests, the part, the mask table, and
    the transform chain -- so pointing at one carries the whole answer. Ids
    given directly are reduced to counts and digests for the same reason
    dataset.json does it: the record must be able to prove which set was used,
    which a digest does, and listing thousands of ids in a registry file would
    only make it unreadable.
    """
    record = {}

    if training_data is not None:
        if isinstance(training_data, dict):
            record["dataset"] = training_data
        else:
            record["dataset"] = _read_dataset_record(project_path, training_data)

    if training_splits is not None:
        record["splits"] = {
            name: {"count": len(list(ids)), "ids_hash": ids_digest(ids)}
            for name, ids in training_splits.items()
        }

    return record or None


def _read_dataset_record(project_path, training_data):
    """
    Read an export's dataset.json, given either it or the directory holding it.

    The directory it came from is recorded alongside, relative to the project
    where possible: the digests say WHICH data, and the path says where to go
    looking for it, which are different questions and both worth an answer.
    """
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

    record["source"] = _relative_to_project(project_path, dataset_path.parent)
    return record
