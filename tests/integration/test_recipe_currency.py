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
    return cf.run_metrics(project_path, run_name="traits",
                          metrics=[cf.body_length()], **kwargs)["organism"]


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
    after_history = cf.export_metrics(segmented_project, path=False,
                                      drop_empty=False, current_only=False)

    assert after[LENGTH_COLUMN].notna().all()   # remeasured for everyone
    assert not before[LENGTH_COLUMN].equals(after[LENGTH_COLUMN])
    assert len(after_history) >= len(after)     # the old values are still on record


def test_a_forced_run_that_processes_nothing_does_not_move_the_pointer(
        segmented_project, caplog):
    measure(segmented_project)
    cf.define_subset(segmented_project, "none", occurrence_ids=[])

    with caplog.at_level("WARNING"):
        result = measure(segmented_project, transforms=[cf.orient()],
                         force=True, subset="none")

    assert result["processed"] == 0
    assert "still points at the previous recipe" in caplog.text

    exported = cf.export_metrics(segmented_project, path=False, drop_empty=False)
    assert exported[LENGTH_COLUMN].notna().all()   # the original recipe's values


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
