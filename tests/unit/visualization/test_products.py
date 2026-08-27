"""
Assets deliberately materialized for downstream use: one file per
occurrence-part.

The opposite contract to a pipeline grid. These are outputs rather than
diagnostics -- figure panels, per-specimen plates, images fed to something that
isn't Python -- so they are one-file-per-thing with the occurrence id in the
filename, which is what makes them joinable back to an exported trait table by
anything that can read a directory listing (R very much included).

And the rule with the sharpest edge: **a render derives nothing and records
nothing.** No mask, no metric, no run row. It hashes its transform chain only so
the folder name identifies what is in it and a rerun is a no-op. A code path
that made a picture into a measurement would be the one way this could go wrong
invisibly.
"""

import pytest

import critterframe as cf
from critterframe.project import paths
from helpers.models import ThresholdModel
from critterframe.records.runs import load_runs
from critterframe.visualization.products import product_filename

pytestmark = pytest.mark.slow

PLATE_CHAIN = [cf.remove_background(), cf.crop_to_mask(pad=0.1)]


def render(project_path, name="plates", **kwargs):
    kwargs.setdefault("transforms", PLATE_CHAIN)
    return cf.render_segments(project_path, name, **kwargs)


# ---------------------------------------------------------------------------
# product_filename
# ---------------------------------------------------------------------------


def test_a_file_is_named_for_its_occurrence():
    assert product_filename("specimen0") == "specimen0.png"


def test_several_parts_qualify_the_name():
    """
    So the mapping stays one row per file even when an occurrence contributes
    several.
    """
    assert product_filename("specimen0", "forewing_left") == \
        "specimen0__forewing_left.png"


def test_the_format_is_part_of_the_name():
    assert product_filename("specimen0", extension="jpg") == "specimen0.jpg"


# ---------------------------------------------------------------------------
# render_segments
# ---------------------------------------------------------------------------


def test_a_render_writes_one_file_per_occurrence(segmented_project):
    summary = render(segmented_project)
    written = sorted(summary["directory"].glob("*.png"))

    assert summary["rendered"] == 8
    assert len(written) == 8
    assert written[0].stem == "specimen0"


def test_the_folder_is_named_for_the_render_and_its_chain(segmented_project):
    """
    The hash is the whole of a render's provenance -- there is no run record to
    look it up in -- so it has to be in the name.
    """
    directory = render(segmented_project)["directory"]
    assert directory.parent == paths.products_dir(segmented_project)
    assert directory.name.startswith("plates_")


def test_a_different_chain_renders_into_a_different_folder(segmented_project):
    """
    Two versions of a plate must not overwrite each other -- comparing them is
    the reason to make the second one.
    """
    first = render(segmented_project)["directory"]
    second = render(segmented_project, transforms=[cf.remove_background()])["directory"]
    assert first != second


def test_rerunning_the_same_render_writes_nothing_new(segmented_project):
    """
    Which makes an interrupted render resumable. Safe because the output is
    deterministic for a given hash, so skipping cannot leave a stale file.
    """
    render(segmented_project)
    again = render(segmented_project)
    assert (again["rendered"], again["skipped"]) == (0, 8)


def test_force_re_renders(segmented_project):
    render(segmented_project)
    assert render(segmented_project, force=True)["rendered"] == 8


def test_a_render_records_no_run(segmented_project):
    """
    THE rule. A render derives no data, so there is nothing to keep provenance
    for beyond the hash naming its folder -- and the run database deliberately
    refuses the "render" kind outright.
    """
    before = len(load_runs(segmented_project))
    render(segmented_project)
    assert len(load_runs(segmented_project)) == before


def test_a_render_writes_no_mask_and_no_metric(segmented_project):
    from critterframe.records import masks as mask_records

    before = len(mask_records.load_masks(segmented_project))
    render(segmented_project)

    assert len(mask_records.load_masks(segmented_project)) == before
    assert len(cf.export_metrics(segmented_project)) == 0


def test_an_occurrence_without_that_part_is_skipped_not_failed(segmented_project):
    """
    A project legitimately has parts only some occurrences carry, and a render
    of "every wing" should write the wings it has and say how many it didn't.
    """
    summary = render(segmented_project, name="wings", part="wing")
    assert (summary["rendered"], summary["failed"]) == (0, 0)
    assert summary["skipped"] == 8


def test_rendering_several_parts_qualifies_every_filename(segmented_project):
    """
    Including for the part that would otherwise have been the default -- a
    folder where some names carry a part and others don't is unreadable by a
    directory listing.
    """
    cf.run_segments(segmented_project, run_name="core", part="core",
                    from_part="organism",
                    shared_steps=[cf.remove_background()],
                    steps=[cf.segment(ThresholdModel(cutoff=120))],
                    visualize=False)

    summary = render(segmented_project, name="both",
                     parts=["organism", "core"])
    names = {path.name for path in summary["directory"].glob("*.png")}
    assert "specimen0__organism.png" in names
    assert "specimen0__core.png" in names


def test_a_subset_and_a_limit_narrow_the_render(segmented_project):
    cf.define_subset(segmented_project, "boxA", column="device", values=["boxA"])
    assert render(segmented_project, subset="boxA")["rendered"] == 4
    assert render(segmented_project, name="few", limit=2)["rendered"] == 2


def test_explicit_ids_override_the_selection(segmented_project):
    summary = render(segmented_project, occurrence_ids=["specimen3"])
    assert summary["rendered"] == 1


def test_the_format_is_a_choice_with_a_lossless_default(segmented_project):
    """
    PNG because a render usually goes into a figure, where an exact edge beats
    a smaller file with JPEG ringing along the specimen boundary.
    """
    summary = render(segmented_project, name="jpegs", extension="jpg")
    assert all(path.suffix == ".jpg"
               for path in summary["directory"].glob("*.*")
               if path.suffix != ".json")


def test_rendering_needs_a_project(empty_project):
    with pytest.raises(FileNotFoundError, match="isn't a CritterFrame project"):
        render(empty_project)
