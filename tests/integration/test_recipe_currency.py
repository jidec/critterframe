"""
What happens when a metric run_name is reused for a different recipe.

Masks solve this for free: masks.parquet upserts to one row per
occurrence-part, so "current" is never ambiguous no matter how many recipe
hashes a segment run_name has cycled through -- that's exactly what
resegmenting is (see test_staleness.py). The metrics table is append-only, so
it needs its own answer: a run_name is pinned to one recipe per part, and
moving it onto a different one is something a caller has to say out loud
(force=True) rather than something that falls out of write order. Nothing is
deleted either way -- the previous recipe's values just stop being current,
the same as a mask's own history does.
"""

import pytest

import critterframe as cf
from critterframe.records.runs import load_runs

pytestmark = pytest.mark.slow

SPECIMENS = 8
LENGTH_COLUMN = "traits__organism__body_length"


def measure(project_path, **kwargs):
    kwargs.setdefault("visualize", False)
    return cf.run_metrics(project_path, run_name="traits", metrics=[cf.body_length()], **kwargs)["organism"]


def test_a_changed_recipe_under_one_name_is_refused_without_force(segmented_project):
    measure(segmented_project)
    with pytest.raises(ValueError, match="currently points at a different"):
        measure(segmented_project, transforms=[cf.orient()])


def test_an_identical_rerun_needs_no_acknowledgement(segmented_project):
    """force is about a CHANGED recipe -- rerunning the same one (retrying
    after an interruption, say) has nothing to acknowledge."""
    measure(segmented_project)
    result = measure(segmented_project)
    assert result["skipped"] == SPECIMENS


def test_force_moves_the_name_onto_the_new_recipe(segmented_project):
    measure(segmented_project)
    result = measure(segmented_project, transforms=[cf.orient()], force=True)

    assert result["processed"] == SPECIMENS
    assert load_runs(segmented_project, name="traits")["recipe_hash"].nunique() == 2


def test_the_old_recipe_s_values_stop_being_current(segmented_project):
    """
    Nothing is deleted -- export_metrics(current_only=False) still shows both
    -- but the default view follows whichever recipe "traits" points at now.
    """
    measure(segmented_project)
    before = cf.export_metrics(segmented_project, path=False, drop_empty=False)
    assert before[LENGTH_COLUMN].notna().all()

    measure(segmented_project, transforms=[cf.orient()], force=True)
    after = cf.export_metrics(segmented_project, path=False, drop_empty=False)
    after_history = cf.export_metrics(segmented_project, path=False, drop_empty=False, current_only=False)

    assert after[LENGTH_COLUMN].notna().all()  # remeasured for everyone
    assert not before[LENGTH_COLUMN].equals(after[LENGTH_COLUMN])
    assert len(after_history) >= len(after)  # the old values are still on record


def test_a_forced_run_that_processes_nothing_does_not_move_the_pointer(segmented_project, caplog):
    measure(segmented_project)
    cf.define_subset(segmented_project, "none", occurrence_ids=[])

    with caplog.at_level("WARNING"):
        result = measure(segmented_project, transforms=[cf.orient()], force=True, subset="none")

    assert result["processed"] == 0
    assert "still points at the previous recipe" in caplog.text

    exported = cf.export_metrics(segmented_project, path=False, drop_empty=False)
    assert exported[LENGTH_COLUMN].notna().all()  # the original recipe's values


def counted_width(interrupt_at=None):
    """
    max_width under its own operation name, optionally interrupted on its Nth
    call. The function isn't in the spec, so the interrupted metric and the
    resumed one are the same recipe -- as a real rerun of one script is.
    """
    from critterframe.metrics.dimensions import _max_width
    from critterframe.recipes import Metric

    calls = []

    def measure_width(segment):
        calls.append(segment.occurrence_id)
        if interrupt_at is not None and len(calls) == interrupt_at:
            raise KeyboardInterrupt
        return _max_width(segment)

    return Metric("counted_width", measure_width, version="1", unit="px")


def widths(project_path, metric, **kwargs):
    return cf.run_metrics(project_path, run_name="widths", metrics=[metric], visualize=False, **kwargs)[
        "organism"
    ]


def test_an_interrupted_forced_move_has_already_moved_the_name(segmented_project):
    """
    Values are stored per occurrence, so the name has to move when the first one
    is. Moved only at the end, an interruption leaves them stored under a recipe
    the name doesn't point at: invisible to an export, and not resumable, since
    force is what gets past the name check and force also redoes everything.
    """
    widths(segmented_project, counted_width())

    with pytest.raises(KeyboardInterrupt):
        widths(segmented_project, counted_width(interrupt_at=4), transforms=[cf.orient()], force=True)

    # Resumed the way any interrupted run is: the same call, without force.
    resumed = widths(segmented_project, counted_width(), transforms=[cf.orient()])
    assert (resumed["skipped"], resumed["processed"]) == (3, SPECIMENS - 3)


def test_what_an_interrupted_forced_move_stored_is_current(segmented_project):
    """The three values written before the interruption are the export's, not the old recipe's."""
    widths(segmented_project, counted_width())
    before = cf.export_metrics(segmented_project, path=False, manifest=False, drop_empty=False)[
        "widths__organism__counted_width"
    ]

    with pytest.raises(KeyboardInterrupt):
        widths(segmented_project, counted_width(interrupt_at=4), transforms=[cf.orient()], force=True)

    after = cf.export_metrics(segmented_project, path=False, manifest=False, drop_empty=False)[
        "widths__organism__counted_width"
    ]
    assert before.notna().sum() == SPECIMENS
    assert after.notna().sum() == 3


def test_reference_and_canonical_need_separate_names(segmented_project):
    """
    inputs={"masks": ...} changes the hash, so measuring the reference table
    under the SAME name as the canonical one is exactly the drift this guards
    -- and force would be the wrong tool here, since reference values are
    meant to coexist with canonical ones, not supersede them.
    """
    measure(segmented_project)
    with pytest.raises(ValueError, match="currently points at a different"):
        measure(segmented_project, reference=True)


def label(project_path, monkeypatch, keys, note):
    """A scripted labelling pass under one note."""
    from critterframe.metrics import annotation
    from helpers.stubs import FakeCv2

    monkeypatch.setattr(annotation, "cv2", FakeCv2(keys=[ord(key) for key in keys]))
    return cf.run_metrics(
        project_path,
        metrics=[cf.exclusive_label_annotation(["good", "bad"], name="quality", note=note)],
        visualize=False,
    )["organism"]


def test_a_labelling_note_is_recorded_with_the_run_and_rewording_it_reasks_nothing(
    segmented_project, monkeypatch
):
    """
    The note says how the labels were meant to be given, so it belongs with the
    run that gave them. It is not the recipe: a reworded note finds every label
    already there, asks nobody anything, and records the new wording.
    """
    first = label(segmented_project, monkeypatch, "1" * SPECIMENS, "bad: any part missing")
    assert first["processed"] == SPECIMENS

    again = label(segmented_project, monkeypatch, "", "bad = any part of it missing")
    assert (again["processed"], again["skipped"]) == (0, SPECIMENS)

    runs = load_runs(segmented_project, name="quality")  # newest first
    notes = [run["operations"]["quality"]["note"] for run in runs["context"]]
    assert notes == ["bad = any part of it missing", "bad: any part missing"]
    assert runs["recipe_hash"].nunique() == 1


def test_the_refusal_says_what_changed(segmented_project):
    """
    "The hash is different" reads like a bug when nothing seems to have changed.
    The stored recipe and the new one are both in hand at that moment, so the
    error names the difference instead of leaving two hashes to compare.
    """
    measure(segmented_project)
    with pytest.raises(ValueError) as refused:
        measure(segmented_project, transforms=[cf.orient(axis_strategy="longer")])

    message = str(refused.value)
    assert "What this run changes from the recipe the name points at" in message
    assert "added transform orient(" in message and "at the start" in message


def test_a_forced_move_logs_what_changed(segmented_project, caplog):
    measure(segmented_project)
    with caplog.at_level("INFO"):
        measure(segmented_project, transforms=[cf.orient()], force=True)

    assert "force=True moves it from recipe" in caplog.text
    assert "added transform orient(" in caplog.text
