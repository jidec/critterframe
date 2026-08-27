"""
Sharded segmentation: several run_segments() calls, one project, no overlap
and no lost work.

run_segments(shard=(index, total)) is what lets several workers -- a cluster
job array, a plain multiprocessing.Pool, or a ThreadPoolExecutor -- process
one project in parallel: shards are deterministic and disjoint by
construction (selectionhelpers.shard_occurrences), and a sharded run writes
to a private staging area rather than masks.parquet directly, so no amount
of concurrent writing can lose a row the way upsert_table's whole-file
rewrite would. That last part is the one thing a sequential test can't fully
show on its own -- so one test here actually runs two shards from separate
threads AT THE SAME TIME against one project.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

import critterframe as cf
from critterframe.records import masks as mask_records
from helpers.models import ThresholdModel

pytestmark = pytest.mark.slow

SPECIMENS = 8


def shard_run(project_path, index, total, **kwargs):
    kwargs.setdefault("visualize", False)
    return cf.run_segments(project_path, steps=[cf.segment(ThresholdModel())],
                           shard=(index, total), **kwargs)["organism"]


def test_two_shards_together_cover_every_occurrence_once(image_project):
    first = shard_run(image_project, 0, 2)
    second = shard_run(image_project, 1, 2)

    assert first["processed"] + second["processed"] == SPECIMENS
    # Nothing lands in the canonical table until merged.
    assert not mask_records.has_masks(image_project)


def test_merging_after_both_shards_makes_every_mask_appear(image_project):
    shard_run(image_project, 0, 2)
    shard_run(image_project, 1, 2)

    merged = cf.merge_mask_shards(image_project)

    assert merged == {"organism": SPECIMENS}
    stored = mask_records.load_masks(image_project)
    assert len(stored) == SPECIMENS
    assert len(set(stored["occurrence_id"])) == SPECIMENS


def test_a_shard_rerun_before_merging_is_resolved_rather_than_duplicated(image_project):
    """
    Rerunning a shard before merging is the one way the same occurrence-part
    can end up staged twice (harmless -- resegmenting an occurrence-part is
    always safe, just wasted work) -- merge must still land on exactly one
    mask per occurrence-part afterward, not two.
    """
    shard_run(image_project, 0, 2)
    shard_run(image_project, 1, 2)
    shard_run(image_project, 0, 2, force=True)   # redo shard 0's work

    merged = cf.merge_mask_shards(image_project)

    assert merged == {"organism": SPECIMENS}
    stored = mask_records.load_masks(image_project)
    assert len(stored) == SPECIMENS


def test_an_unsharded_run_still_writes_directly(image_project):
    """
    shard=None is the default and must remain exactly today's behaviour: no
    staging, straight into masks.parquet.
    """
    result = cf.run_segments(image_project, steps=[cf.segment(ThresholdModel())],
                             visualize=False)["organism"]

    assert result["processed"] == SPECIMENS
    assert len(mask_records.load_masks(image_project)) == SPECIMENS


def test_two_shards_running_at_the_same_time_lose_nothing(image_project):
    """
    The actual concurrency claim, not just two sequential calls: run both
    shards from separate threads at once, against the same project, then
    confirm the merge still accounts for every occurrence exactly once.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(shard_run, image_project, index, 2)
                  for index in range(2)]
        results = [future.result() for future in futures]

    assert sum(result["processed"] for result in results) == SPECIMENS

    merged = cf.merge_mask_shards(image_project)
    assert merged == {"organism": SPECIMENS}
    stored = mask_records.load_masks(image_project)
    assert len(stored) == SPECIMENS
    assert len(set(stored["occurrence_id"])) == SPECIMENS
