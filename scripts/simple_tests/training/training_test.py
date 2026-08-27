"""
Splitting, exporting, and registering, over synthetic images -- no real data,
no credentials, no GPU, no network.

Covers the path from "a project holds masks and labels" to "a model trained
outside CritterFrame is back inside it as an ordinary dependency":

    split_ids()            -> which occurrences answer which question
    export_training_data() -> those occurrences as files a trainer can read
    register_model()       -> the checkpoint that came back, and what it saw

Run from the repo root:
    python scripts/simple_tests/training/training_test.py
"""

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import critterframe as cf

# Shared with the test suite rather than written out twice -- see
# tests/helpers/. The suite asserts what these produce; this script shows it.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from helpers.models import ThresholdModel          # noqa: E402

logging.basicConfig(level=logging.WARNING)

WORKSPACE = os.path.join(tempfile.gettempdir(), "critterframe_training_test")
PROJECT_PATH = os.path.join(WORKSPACE, "project")
IMAGE_DIR = os.path.join(WORKSPACE, "images")

shutil.rmtree(WORKSPACE, ignore_errors=True)
os.makedirs(IMAGE_DIR)

# Three "species" of drawn blob, two images of each of nine specimens. Two
# images per specimen is the whole point of the group column: they are
# near-duplicates, and a split that puts one in train and the other in
# validation reports a score for memorization.
SPECIES = ["Anax junius", "Anax junius", "Libellula lydia"]
ROWS = []
for specimen in range(9):
    for shot in range(2):
        occurrence_id = f"spec{specimen:02d}_{shot}"
        image = np.full((200, 240, 3), 30, np.uint8)
        radius = 30 + 6 * (specimen % 3)
        cv2.circle(image, (120, 100), radius, (210, 190, 160), -1)
        cv2.imwrite(os.path.join(IMAGE_DIR, f"{occurrence_id}.png"), image)
        ROWS.append({"occurrence_id": occurrence_id,
                     "species": SPECIES[specimen % 3],
                     "specimen": f"spec{specimen:02d}"})

metadata = pd.DataFrame(ROWS)


print("== a project with labels and masks ==")
print(cf.ingest_images(PROJECT_PATH, IMAGE_DIR, metadata=metadata))
print(cf.run_segments(PROJECT_PATH, steps=[cf.segment(ThresholdModel())]))

print("\n== split_ids ==")
splits = cf.split_ids(
    PROJECT_PATH,
    proportions={"train": 0.6, "val": 0.2, "test": 0.2},
    stratify_by="species",
    group_by="specimen",
    seed=123,
)
for name, ids in splits.items():
    print(f"  {name:<6} {len(ids):>2}  {sorted(ids)}")

occurrences = metadata.set_index("occurrence_id")
sides = {}
for name, ids in splits.items():
    for occurrence_id in ids:
        sides.setdefault(occurrences.loc[occurrence_id, "specimen"], set()).add(name)
straddling = [specimen for specimen, names in sides.items() if len(names) > 1]
print(f"\n  specimens split across sides: {len(straddling)} (expect 0) -- both "
      "images of\n  a specimen are on one side, which is what group_by "
      "guarantees exactly")

for name, ids in splits.items():
    counts = occurrences.loc[sorted(ids), "species"].value_counts().to_dict()
    print(f"  {name:<6} species: {counts}")
print("  <- stratification is approximate where grouping is exact: a group "
      "can't be\n     divided to balance a stratum, and the leakage guarantee "
      "is the one to keep")

shuffled = cf.split_ids(
    PROJECT_PATH,
    occurrence_ids=list(reversed(metadata["occurrence_id"].tolist())),
    proportions={"train": 0.6, "val": 0.2, "test": 0.2},
    stratify_by="species",
    group_by="specimen",
    seed=123,
)
print(f"\n  same ids in reverse order give the same split: "
      f"{shuffled == splits} (expect True)")

changed = cf.split_ids(PROJECT_PATH, proportions={"train": 0.6, "val": 0.2, "test": 0.2},
                       stratify_by="species", group_by="specimen", seed=7)
print(f"  a different seed gives a different one: {changed != splits} (expect True)")

print("\n== export_training_data: an ImageFolder-shaped classification set ==")
CLASSES_DIR = os.path.join(WORKSPACE, "classes")
manifest = cf.export_training_data(
    PROJECT_PATH, CLASSES_DIR,
    splits=splits,
    class_by="species",
    transforms=[cf.remove_background(), cf.crop_to_mask()],
    metadata=["specimen"],
)
print(manifest[["occurrence_id", "split", "class", "image_path", "specimen"]]
      .head(6).to_string(index=False))
for root, _directories, files in sorted(os.walk(CLASSES_DIR)):
    if files and root != CLASSES_DIR:
        relative = os.path.relpath(root, CLASSES_DIR).replace(os.sep, "/")
        print(f"  {relative}/  {len(files)} file(s)")
print("  <- 'Anax junius' became the folder Anax_junius; a class holding a "
      "slash would\n     otherwise have quietly become a nested directory")

print("\n== export_training_data: image/mask pairs for a segmenter ==")
SEGMENTER_DIR = os.path.join(WORKSPACE, "segmenter")
seg_manifest = cf.export_training_data(
    PROJECT_PATH, SEGMENTER_DIR,
    splits={"train": splits["train"], "val": splits["val"]},
    masks=True,
)
print(seg_manifest[["occurrence_id", "split", "image_path", "mask_path",
                    "mask_area"]].head(4).to_string(index=False))

with open(os.path.join(SEGMENTER_DIR, "dataset.json"), encoding="utf-8") as handle:
    dataset = json.load(handle)
print("\n  dataset.json:")
print("   ", json.dumps({key: dataset[key] for key in
                         ("part", "reference", "masks", "splits", "data_hash")},
                        indent=2).replace("\n", "\n    "))
print("  <- the id digests identify WHICH occurrences were exported without "
      "listing them,\n     and data_hash identifies the export as a whole")

print("\n== an occurrence in two splits is refused ==")
try:
    cf.export_training_data(PROJECT_PATH, os.path.join(WORKSPACE, "leaky"),
                            splits={"train": splits["train"],
                                    "test": splits["train"][:1] + splits["test"]})
except ValueError as exc:
    print(f"  ValueError: {exc}")

print("\n== register_model ==")
# Stand-in for a checkpoint trained outside CritterFrame on what was exported.
checkpoint = os.path.join(PROJECT_PATH, "models", "blobnet_v1.pt")
os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
with open(checkpoint, "wb") as handle:
    handle.write(b"pretend weights, revision 1")

registered = cf.register_model(
    PROJECT_PATH, "blobnet_v1",
    path=checkpoint,
    task="segment",
    framework="torch",
    base_model="sam2_hiera_large",
    training_data=SEGMENTER_DIR,
    parameters={"epochs": 40, "lr": 1e-4},
)
print(" ", registered)
print("  identity in a recipe:", registered.identity())
print("  training data it saw :",
      json.dumps(registered.record["training_data"]["dataset"]["splits"]))
print("  stored path          :", registered.record["path"],
      "<- relative, so a copied project still finds it")

print("\n== a registered model runs like any other ==")
model = cf.load_model(PROJECT_PATH, "blobnet_v1").attach(ThresholdModel(cutoff=120))
before = cf.Recipe("segment", "custom", [cf.segment(model)]).hash
print("  run:", cf.run_segments(PROJECT_PATH, run_name="custom",
                                steps=[cf.segment(model)]))
print(f"  recipe hash: {before}")

# Retraining into the same filename is the failure a fingerprint exists for.
with open(checkpoint, "wb") as handle:
    handle.write(b"pretend weights, revision 2 -- retrained for longer")
cf.register_model(PROJECT_PATH, "blobnet_v1", path=checkpoint, task="segment",
                  framework="torch", base_model="sam2_hiera_large",
                  training_data=SEGMENTER_DIR, parameters={"epochs": 80})

retrained = cf.load_model(PROJECT_PATH, "blobnet_v1").attach(ThresholdModel(cutoff=120))
after = cf.Recipe("segment", "custom", [cf.segment(retrained)]).hash
print(f"  after retraining into the same file: {after}")
print(f"  hash moved: {before != after} (expect True) -- same name, same path, "
      "same\n  class, different weights, so the work is correctly redone")
print("  rerun:", cf.run_segments(PROJECT_PATH, run_name="custom",
                                  steps=[cf.segment(retrained)]),
      " <- expect processed=18, skipped=0")

print("\n== the registry ==")
for name, record in cf.list_models(PROJECT_PATH).items():
    print(f"  {name}: {record['task']} / {record['framework']} / "
          f"{record['fingerprint']} / from {record['base_model']}")
print("\nworkspace left at", WORKSPACE)
