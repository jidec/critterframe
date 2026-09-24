"""
Masks in and out of a project as `<occurrence_id>__<part>.png` files.

The round trip has to be exact -- same pixels, same place in the original image
-- and an imported mask's identity is its own pixels, so re-importing a corrected
one makes exactly its own metrics stale and re-importing an identical one does
nothing at all.
"""

import shutil

import cv2
import numpy as np
import pandas as pd
import pytest

import critterframe as cf
from critterframe.project import paths
from critterframe.records import masks as mask_records
from critterframe.records import occurrences as occurrence_records
from critterframe.records.runs import load_runs
from critterframe.segmentation.mask_import_export import MANIFEST_NAME
from critterframe.storage.imagestore import ImageStore

pytestmark = pytest.mark.slow


def ids_of(project_path):
    return sorted(occurrence_records.load_occurrences(project_path)["occurrence_id"])


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.asarray(mask, np.uint8) * 255)
    path.write_bytes(encoded.tobytes())


def image_shape(project_path, occurrence_id):
    with ImageStore(project_path, readonly=True) as store:
        return store.get(occurrence_id).shape[:2]


def add_occurrence(project_path, occurrence_id):
    """An occurrence with no image, copied from the first row."""
    table = occurrence_records.load_occurrences(project_path)
    extra = table.iloc[[0]].copy()
    extra["occurrence_id"] = occurrence_id
    occurrence_records.save_occurrences(project_path, pd.concat([table, extra],
                                                                ignore_index=True))


def unsegmented_copy(project_path, tmp_path):
    """The same images and occurrences, in a second project with no masks."""
    target = tmp_path / "target"
    shutil.copytree(project_path, target)
    paths.masks_path(target).unlink()
    return target


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------


def test_masks_round_trip_pixel_for_pixel(segmented_project, tmp_path):
    image_project = unsegmented_copy(segmented_project, tmp_path)
    folder = tmp_path / "masks"
    exported = cf.export_masks(segmented_project, folder)
    assert exported["organism"]["processed"] == len(ids_of(segmented_project))
    assert (folder / MANIFEST_NAME).exists()

    result = cf.import_masks(image_project, folder, description="from the template",
                             visualize=False)["organism"]
    assert result["processed"] == len(ids_of(image_project))

    source = mask_records.mask_lookup(segmented_project)
    imported = mask_records.mask_lookup(image_project)
    for occurrence_id, row in source.items():
        assert np.array_equal(mask_records.decode_mask(row),
                              mask_records.decode_mask(imported[occurrence_id]))


def test_reimporting_the_same_folder_changes_nothing(segmented_project, tmp_path):
    image_project = unsegmented_copy(segmented_project, tmp_path)
    folder = tmp_path / "masks"
    cf.export_masks(segmented_project, folder)
    cf.import_masks(image_project, folder, visualize=False)
    again = cf.import_masks(image_project, folder, visualize=False)["organism"]
    assert (again["processed"], again["skipped"]) == (0, len(ids_of(image_project)))


def test_the_import_run_records_where_the_masks_came_from(segmented_project, tmp_path):
    image_project = unsegmented_copy(segmented_project, tmp_path)
    folder = tmp_path / "masks"
    cf.export_masks(segmented_project, folder)
    cf.import_masks(image_project, folder, description="hand-drawn", visualize=False)

    [run] = load_runs(image_project, name="organism_imported").to_dict("records")
    assert run["context"]["description"] == "hand-drawn"
    assert run["context"]["folder"] == "masks"
    assert run["context"]["source"]["project"] == "project"
    assert run["context"]["source"]["runs"]


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def test_a_file_without_a_part_lands_on_part(image_project, tmp_path):
    occurrence_id = ids_of(image_project)[0]
    mask = np.zeros(image_shape(image_project, occurrence_id), bool)
    mask[10:30, 10:30] = True
    write_mask(tmp_path / "masks" / f"{occurrence_id}.png", mask)

    result = cf.import_masks(image_project, tmp_path / "masks", part="wing",
                             visualize=False)
    assert result["wing"]["processed"] == 1
    assert occurrence_id in mask_records.mask_lookup(image_project, part="wing")


def test_an_id_containing_the_separator_still_round_trips(image_project, tmp_path):
    add_occurrence(image_project, "x__y")
    mask = np.zeros((20, 20), bool)
    mask[5:10, 5:10] = True
    write_mask(tmp_path / "named" / "x__y__organism.png", mask)
    write_mask(tmp_path / "bare" / "x__y.png", mask)

    named = cf.import_masks(image_project, tmp_path / "named", visualize=False)
    bare = cf.import_masks(image_project, tmp_path / "bare", part="wing", visualize=False)
    assert named["organism"]["processed"] == 1
    assert bare["wing"]["processed"] == 1


def test_parts_and_subset_narrow_an_export(segmented_project, tmp_path):
    chosen = ids_of(segmented_project)[:2]
    cf.define_subset(segmented_project, "two", occurrence_ids=chosen)
    cf.export_masks(segmented_project, tmp_path / "masks", parts=["organism"], subset="two")
    written = sorted(path.name for path in (tmp_path / "masks").glob("*.png"))
    assert written == sorted(f"{occurrence_id}__organism.png" for occurrence_id in chosen)


# ---------------------------------------------------------------------------
# What can't be imported, and what can only be imported unchecked
# ---------------------------------------------------------------------------


def test_a_mask_of_the_wrong_size_is_refused(image_project, tmp_path):
    occurrence_id = ids_of(image_project)[0]
    write_mask(tmp_path / "masks" / f"{occurrence_id}__organism.png", np.ones((10, 12), bool))

    result = cf.import_masks(image_project, tmp_path / "masks", visualize=False)["organism"]
    assert result["failed"] == 1
    assert "12x10" in result["failures"][0]["error"]


def test_an_empty_mask_is_refused(image_project, tmp_path):
    occurrence_id = ids_of(image_project)[0]
    write_mask(tmp_path / "masks" / f"{occurrence_id}__organism.png",
               np.zeros(image_shape(image_project, occurrence_id), bool))
    result = cf.import_masks(image_project, tmp_path / "masks", visualize=False)["organism"]
    assert result["failures"][0]["error"] == "empty mask"


def test_a_file_for_no_known_occurrence_is_unmatched(image_project, tmp_path):
    write_mask(tmp_path / "masks" / "nobody__organism.png", np.ones((10, 10), bool))
    result = cf.import_masks(image_project, tmp_path / "masks", visualize=False)["organism"]
    assert (result["unmatched"], result["failed"], result["processed"]) == (1, 0, 0)


def test_an_occurrence_with_no_image_is_imported_unverified(image_project, tmp_path):
    add_occurrence(image_project, "no_image_yet")
    write_mask(tmp_path / "masks" / "no_image_yet__organism.png", np.ones((10, 10), bool))
    result = cf.import_masks(image_project, tmp_path / "masks", visualize=False)["organism"]
    assert (result["processed"], result["unverified"]) == (1, 1)


# ---------------------------------------------------------------------------
# Replacing
# ---------------------------------------------------------------------------


def test_a_corrected_mask_makes_only_its_own_metrics_stale(measured_project, tmp_path):
    before = cf.export_metrics(measured_project, path=False)
    target = ids_of(measured_project)[0]
    mask = np.zeros(image_shape(measured_project, target), bool)
    mask[20:60, 20:90] = True
    write_mask(tmp_path / "masks" / f"{target}__organism.png", mask)

    result = cf.import_masks(measured_project, tmp_path / "masks", visualize=False)
    assert result["organism"]["replaced"] == 1

    after = cf.export_metrics(measured_project, path=False, drop_empty=False)
    column = "traits__organism__body_length"
    assert pd.isna(after.set_index("occurrence_id").loc[target, column])
    others = after[after["occurrence_id"] != target][column]
    assert others.notna().all() and len(others) == len(before) - 1


def test_replace_false_keeps_the_existing_mask(segmented_project, tmp_path):
    target = ids_of(segmented_project)[0]
    original = mask_records.decode_mask(mask_records.mask_lookup(segmented_project)[target])
    write_mask(tmp_path / "masks" / f"{target}__organism.png", ~original)

    result = cf.import_masks(segmented_project, tmp_path / "masks", replace=False,
                             visualize=False)["organism"]
    assert result["skipped"] == 1
    kept = mask_records.decode_mask(mask_records.mask_lookup(segmented_project)[target])
    assert np.array_equal(kept, original)


def test_a_reference_import_goes_to_the_reference_table(segmented_project, tmp_path):
    image_project = unsegmented_copy(segmented_project, tmp_path)
    cf.export_masks(segmented_project, tmp_path / "masks")
    cf.import_masks(image_project, tmp_path / "masks", reference=True, visualize=False)
    assert mask_records.mask_lookup(image_project, reference=True)
    assert not mask_records.mask_lookup(image_project)
    assert not load_runs(image_project, name="organism_reference_imported").empty


# ---------------------------------------------------------------------------
# What export refuses to write
# ---------------------------------------------------------------------------


def _mask_for(project_path, occurrence_id):
    add_occurrence(project_path, occurrence_id)
    mask_records.save_masks(project_path, [mask_records.make_mask_row(
        occurrence_id, np.ones((5, 5), bool))])


def test_an_id_no_file_system_can_carry_is_refused(segmented_project, tmp_path):
    _mask_for(segmented_project, "a:b")
    with pytest.raises(ValueError, match="every OS"):
        cf.export_masks(segmented_project, tmp_path / "masks")
    assert not (tmp_path / "masks").exists()


def test_names_differing_only_by_case_are_refused(segmented_project, tmp_path):
    _mask_for(segmented_project, "Specimen")
    _mask_for(segmented_project, "specimen")
    with pytest.raises(ValueError, match="only by case"):
        cf.export_masks(segmented_project, tmp_path / "masks")


def test_an_existing_file_is_never_overwritten(segmented_project, tmp_path):
    cf.export_masks(segmented_project, tmp_path / "masks")
    with pytest.raises(FileExistsError):
        cf.export_masks(segmented_project, tmp_path / "masks")


def test_a_folder_with_a_non_ascii_name_works_both_ways(segmented_project, tmp_path):
    image_project = unsegmented_copy(segmented_project, tmp_path)
    folder = tmp_path / "mäsks_é"
    cf.export_masks(segmented_project, folder)
    result = cf.import_masks(image_project, folder, visualize=False)["organism"]
    assert result["processed"] == len(ids_of(image_project))
