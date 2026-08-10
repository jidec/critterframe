"""
Recipe hashing and segment coordinate bookkeeping -- the two mechanisms
everything else quietly depends on, checked in isolation.

Hashing is what makes processing repeat-aware, and coordinate inversion is what
makes parts-via-crops work. Both fail silently when they're wrong: a hash that
changes when it shouldn't just means work is redone, and an inversion that's
wrong produces a mask that looks fine on its own and is in the wrong place.
This prints enough to see either.

Needs nothing -- no project, no data.

Run from the repo root:
    python scripts/simple_tests/recipes_test.py
"""

import numpy as np

import critterframe as cf
from critterframe.recipes import Recipe, Segment

print("== recipe identity ==")


def build(steps, name="traits", part="organism"):
    return Recipe("metric", name, steps, part=part)


same_a = build([cf.remove_appendages(), cf.orient(), cf.body_length()])
same_b = build([cf.remove_appendages(), cf.orient(), cf.body_length()])
reordered = build([cf.orient(), cf.remove_appendages(), cf.body_length()])
reparameterized = build([cf.remove_appendages(relative_radius=0.05), cf.orient(),
                         cf.body_length()])
renamed = build([cf.remove_appendages(), cf.orient(), cf.body_length()], name="other")
repartd = build([cf.remove_appendages(), cf.orient(), cf.body_length()], part="head")

print(f"  identical recipes  : {same_a.hash} == {same_b.hash}  "
      f"-> {same_a.hash == same_b.hash} (expect True)")
for label, other in (("reordered      ", reordered),
                     ("reparameterized", reparameterized),
                     ("renamed        ", renamed),
                     ("different part ", repartd)):
    print(f"  {label}: {other.hash}  differs -> "
          f"{other.hash != same_a.hash} (expect True)")

print("\n  a hash covers the whole recipe, so this is what's actually stored:")
print("  ", same_a)

print("\n== model identity reaches the hash ==")


class FakeModel:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def identity(self):
        return {"class": "FakeModel", "checkpoint": self.checkpoint}

    def predict(self, image, mask_threshold=0.5):
        raise NotImplementedError


v1 = Recipe("segment", "seg", [cf.segment(FakeModel("v1.pt"))])
v2 = Recipe("segment", "seg", [cf.segment(FakeModel("v2.pt"))])
print(f"  checkpoint v1: {v1.hash}")
print(f"  checkpoint v2: {v2.hash}  differs -> {v1.hash != v2.hash} (expect True)")

print("\n== coordinate inversion ==")
# A 200x300 frame with one blob at a known place. Whatever a recipe does to the
# frame, the mask must come back pointing at that same place.
image = np.zeros((200, 300, 3), np.uint8)
image[60:100, 200:240] = 255
truth = np.zeros((200, 300), bool)
truth[60:100, 200:240] = True

blob_x, blob_y = 220.0, 80.0
print(f"  blob sits at ({blob_x:.0f}, {blob_y:.0f}) in a 300x200 frame\n")

chains = {
    "no transforms": [],
    "crop upper_right": [cf.crop(region="upper_right")],
    "crop then rotate": [cf.crop(region="upper_right"), cf.rotate(30)],
    "crop, rotate, resize": [cf.crop(region="upper_right"), cf.rotate(30),
                             cf.resize(scale=2.0)],
    "crop_to_mask": [cf.crop_to_mask()],
}

for label, chain in chains.items():
    state = Segment(image, mask=truth, occurrence_id="test")
    for operation in chain:
        state, _info = operation(state)

    working = state.shape
    restored = state.mask_in_original_coordinates()
    ys, xs = np.nonzero(restored)
    overlap = (restored & truth).sum() / (restored | truth).sum()

    print(f"  {label:<22} working frame {working[1]}x{working[0]:<4} "
          f"restored ({xs.mean():5.0f}, {ys.mean():5.0f}) "
          f"shape {restored.shape}  iou vs truth {overlap:.3f}")

print("\n  every row should restore near (220, 80) at shape (200, 300); iou is")
print("  below 1.0 where a rotation or resize resampled the mask, which is")
print("  expected -- interpolation is lossy, the coordinate frame is not.")
