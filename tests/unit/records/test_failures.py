"""
The failure ledger: what makes a rerun skip a previously-failed attempt, and
what makes it stop skipping once the attempt would genuinely differ.
"""

from critterframe.records import failures as failure_records


def test_a_recorded_failure_is_found_by_matching_context(tmp_path):
    failure_records.record_failures(tmp_path, "segment", [
        {"occurrence_id": "a", "part": "organism", "context_hash": "h1",
         "error": "boom"},
    ])

    found = failure_records.failed_keys(
        tmp_path, "segment", {("a", "organism"): "h1"})
    assert found == {("a", "organism")}


def test_a_changed_context_hash_is_not_returned_as_failed(tmp_path):
    """
    The whole point: a retuned recipe (or a corrected URL) is a different
    attempt, so the old failure must not suppress it.
    """
    failure_records.record_failures(tmp_path, "segment", [
        {"occurrence_id": "a", "part": "organism", "context_hash": "h1",
         "error": "boom"},
    ])

    found = failure_records.failed_keys(
        tmp_path, "segment", {("a", "organism"): "h2"})
    assert found == set()


def test_a_different_stage_does_not_see_another_stage_s_failure(tmp_path):
    failure_records.record_failures(tmp_path, "download", [
        {"occurrence_id": "a", "context_hash": "h1", "error": "404"},
    ])

    found = failure_records.failed_keys(
        tmp_path, "segment", {("a", failure_records.NO_PART): "h1"})
    assert found == set()


def test_recording_again_replaces_the_prior_failure(tmp_path):
    failure_records.record_failures(tmp_path, "segment", [
        {"occurrence_id": "a", "part": "organism", "context_hash": "h1",
         "error": "first error"},
    ])
    failure_records.record_failures(tmp_path, "segment", [
        {"occurrence_id": "a", "part": "organism", "context_hash": "h1",
         "error": "second error"},
    ])

    df = failure_records.load_failures(tmp_path, stage="segment")
    assert len(df) == 1
    assert df.iloc[0]["error"] == "second error"


def test_clearing_a_failure_lets_it_be_seen_as_failed_no_longer(tmp_path):
    failure_records.record_failures(tmp_path, "segment", [
        {"occurrence_id": "a", "part": "organism", "context_hash": "h1",
         "error": "boom"},
    ])
    failure_records.clear_failures(tmp_path, "segment", [("a", "organism")])

    found = failure_records.failed_keys(
        tmp_path, "segment", {("a", "organism"): "h1"})
    assert found == set()


def test_a_download_style_failure_has_no_part(tmp_path):
    """Download has no part concept; NO_PART is the fixed key it uses."""
    failure_records.record_failures(tmp_path, "download", [
        {"occurrence_id": "a", "context_hash": "urlhash", "error": "404"},
    ])

    found = failure_records.failed_keys(
        tmp_path, "download", {("a", failure_records.NO_PART): "urlhash"})
    assert found == {("a", failure_records.NO_PART)}


def test_failed_keys_on_an_empty_project_is_empty(tmp_path):
    assert failure_records.failed_keys(tmp_path, "segment", {("a", "organism"): "h1"}) == set()


def test_load_failures_on_an_empty_project_is_empty(tmp_path):
    assert failure_records.load_failures(tmp_path).empty
