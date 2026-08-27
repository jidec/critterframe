"""
What happens to a number when the mask it was measured from is replaced.

Nothing is deleted. Replacing an occurrence-part's canonical mask means the
values derived from the old one stop being CURRENT, and the long table
legitimately holds both -- provenance kept, working analysis following the
current masks. Two things have to be true at once for that to work, and each
fails silently on its own:

  the export must stop reporting the old values, or the project serves numbers
  measured off masks it no longer holds;

  a re-run must not skip the work, or there is nothing to replace them with.

The second is the subtler one. The metric recipe hash did not change when the
mask did, so a completion check keyed on the recipe alone would report every
occurrence as already done and the project would keep its stale numbers forever
-- with `force=True` as the only escape, and nothing to suggest it was needed.
"""

import pytest

import critterframe as cf
from critterframe.records.metrics import load_metrics
from helpers.models import ThresholdModel

pytestmark = pytest.mark.slow

SPECIMENS = 8
LENGTH_COLUMN = "traits__organism__body_length"


def measure(project_path, **kwargs):
    kwargs.setdefault("visualize", False)
    return cf.run_metrics(project_path, run_name="traits",
                          transforms=[cf.remove_appendages(), cf.orient()],
                          metrics=[cf.body_length(), cf.max_width(),
                                   cf.mask_area(name="area_px", unit="px2"),
                                   cf.mean_lightness(), cf.blur_variance(),
                                   cf.bilateral_asymmetry(),
                                   cf.edge_fraction()],
                          **kwargs)["organism"]


def resegment(project_path, erode=2):
    return cf.run_segments(project_path,
                           steps=[cf.segment(ThresholdModel(erode=erode))],
                           visualize=False)["organism"]


def test_the_export_holds_every_occurrence_before_anything_moves(measured_project):
    exported = cf.export_metrics(measured_project)
    assert len(exported) == SPECIMENS
    assert exported[LENGTH_COLUMN].notna().all()


def test_resegmenting_empties_the_export_without_deleting_a_value(measured_project):
    """
    Every stored value was measured off the mask that just got replaced, so
    none of them describes what the project now holds and none is exported.
    They are not gone -- the long table still has every one, with the
    provenance that says why they no longer count.
    """
    before = load_metrics(measured_project)
    resegment(measured_project)

    exported = cf.export_metrics(measured_project)
    assert len(exported) == 0
    assert LENGTH_COLUMN not in exported.columns
    assert len(load_metrics(measured_project)) == len(before)


def test_an_occurrence_with_no_current_value_can_be_kept_in_the_export(
        measured_project):
    """
    `drop_empty=False` keeps the occurrence and its metadata, with no trait
    columns to show for it -- for an export that has to line up with a full
    specimen list.
    """
    resegment(measured_project)
    kept = cf.export_metrics(measured_project, drop_empty=False)
    assert len(kept) == SPECIMENS
    assert LENGTH_COLUMN not in kept.columns


def test_a_rerun_after_resegmenting_redoes_the_work_without_force(measured_project):
    """
    The recipe has run over these occurrences before, but not over these masks,
    so there is no cached work to reuse -- and needing force= here would mean
    the staleness was invisible to the machinery that decides what to skip.
    """
    resegment(measured_project)
    assert measure(measured_project)["processed"] == SPECIMENS


def test_the_new_values_differ_and_both_are_on_record(measured_project):
    """
    A tighter mask is a shorter body. Both numbers stay in the long table,
    distinguished by the mask they came from rather than by which is "right".
    """
    before = cf.export_metrics(measured_project).set_index("occurrence_id")
    resegment(measured_project)
    measure(measured_project)
    after = cf.export_metrics(measured_project).set_index("occurrence_id")

    specimen = before.index[0]
    assert after.loc[specimen, LENGTH_COLUMN] < before.loc[specimen, LENGTH_COLUMN]

    history = load_metrics(measured_project, metric_names=["body_length"])
    history = history[history["occurrence_id"] == specimen]
    assert len(history) == 2
    assert history["recipe_hash"].nunique() == 1
    assert history["source_mask_hash"].nunique() == 2


def test_the_full_history_is_still_reachable(measured_project):
    """
    `current_only=False` is the opt-out, for asking what a project used to say.
    """
    resegment(measured_project)
    measure(measured_project)

    current = cf.export_metrics(measured_project)
    everything = cf.export_metrics(measured_project, current_only=False)
    assert len(current) == SPECIMENS
    assert len(everything) == SPECIMENS
    assert len(load_metrics(measured_project)) == SPECIMENS * 7 * 2


def test_resegmenting_one_occurrence_leaves_its_neighbours_current(measured_project):
    """
    Staleness is per occurrence-part. A subset resegmented with a better recipe
    must not blank the export for everything else.
    """
    before = cf.export_metrics(measured_project)["occurrence_id"].tolist()
    cf.run_segments(measured_project, steps=[cf.segment(ThresholdModel(erode=2))],
                    limit=1, visualize=False)

    after = cf.export_metrics(measured_project)["occurrence_id"].tolist()
    assert len(after) == SPECIMENS - 1
    assert set(before) - set(after) == {before[0]}


def test_resegmenting_back_to_the_original_recipe_revives_the_old_values(
        measured_project):
    """
    A consequence of identity being a hash rather than a timestamp, and worth
    pinning: the values were never stale in themselves, only stale relative to
    a mask. Restore the mask and they describe the project again.
    """
    resegment(measured_project)
    assert len(cf.export_metrics(measured_project)) == 0

    cf.run_segments(measured_project, steps=[cf.segment(ThresholdModel())],
                    force=True, visualize=False)
    revived = cf.export_metrics(measured_project)
    assert len(revived) == SPECIMENS
    assert revived[LENGTH_COLUMN].notna().all()


def test_stale_values_are_gone_from_the_narrow_lookup_too(measured_project):
    """
    `latest_values` is what a group metric fits its reference population from,
    so a stale value there doesn't just misreport one occurrence -- it shifts
    the distribution every other occurrence is scored against.
    """
    from critterframe.records.metrics import latest_values

    resegment(measured_project)
    assert latest_values(measured_project, "traits",
                         metric_name="body_length").empty
    assert len(latest_values(measured_project, "traits",
                             metric_name="body_length",
                             current_only=False)) == SPECIMENS
