"""
The JSON and JSON Lines files a project keeps.

Two properties matter here and neither is visible in the happy path: a
whole-file write must be all-or-nothing, so a crash leaves the previous
complete file rather than half of the new one; and everything is UTF-8
whatever the platform's default is, because a species name with an accent in
it is ordinary and tomllib/json readers only accept UTF-8.
"""

import json

import pytest

from critterframe.storage.jsonfiles import (
    append_jsonl,
    atomic_write,
    read_json,
    read_jsonl,
    write_json,
)


def test_a_record_round_trips(tmp_path):
    path = write_json(tmp_path / "manifest.json", {"b": 1, "a": [2, 3]})
    assert read_json(path) == {"b": 1, "a": [2, 3]}


def test_keys_are_sorted_so_two_identical_records_are_identical_files(tmp_path):
    first = write_json(tmp_path / "a.json", {"b": 1, "a": 2}).read_text(encoding="utf-8")
    second = write_json(tmp_path / "b.json", {"a": 2, "b": 1}).read_text(encoding="utf-8")
    assert first == second


def test_numpy_values_serialize_like_every_other_spec(tmp_path):
    import numpy as np

    path = write_json(tmp_path / "spec.json", {"n": np.int64(3), "xs": np.arange(2)})
    assert read_json(path) == {"n": 3, "xs": [0, 1]}


def test_reading_a_file_that_is_not_there_gives_the_default(tmp_path):
    assert read_json(tmp_path / "missing.json", default={}) == {}


def test_a_failed_write_leaves_the_previous_file_intact(tmp_path):
    """The whole point of writing through a temp file."""
    path = write_json(tmp_path / "registry.json", {"models": {"a": 1}})

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        write_json(path, {"models": Unserializable()})

    assert read_json(path) == {"models": {"a": 1}}
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_an_interrupted_write_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "half.json"
    with pytest.raises(KeyboardInterrupt):
        with atomic_write(path) as handle:
            handle.write("{")
            raise KeyboardInterrupt

    assert not path.exists()
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_non_ascii_survives_the_round_trip(tmp_path):
    """Windows writes cp1252 when asked for nothing, and json only reads UTF-8."""
    path = write_json(tmp_path / "note.json", {"note": "Libellule à Ánax"})
    assert read_json(path)["note"] == "Libellule à Ánax"

    with atomic_write(tmp_path / "note.toml") as handle:
        handle.write('note = "Libellule à Ánax"\n')
    assert "Ánax" in (tmp_path / "note.toml").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON Lines logs
# ---------------------------------------------------------------------------


def test_a_log_grows_by_appending(tmp_path):
    log = tmp_path / "imports.jsonl"
    append_jsonl(log, {"import_hash": "a"})
    append_jsonl(log, {"import_hash": "b"})

    records = read_jsonl(log)
    assert records["import_hash"].tolist() == ["a", "b"]     # oldest first


def test_an_absent_log_reads_as_empty(tmp_path):
    assert read_jsonl(tmp_path / "nothing.jsonl").empty


def test_one_truncated_line_does_not_cost_the_rest(tmp_path, caplog):
    """
    A process killed mid-append leaves a partial line. The history either side
    of it is still the record of what happened.
    """
    log = tmp_path / "exports.jsonl"
    append_jsonl(log, {"export_hash": "a"})
    with open(log, "a", encoding="utf-8") as handle:
        handle.write('{"export_hash": "b"')          # killed here
    append_jsonl(log, {"export_hash": "c"})

    with caplog.at_level("WARNING"):
        records = read_jsonl(log, what="export")

    assert records["export_hash"].tolist() == ["a", "c"]
    assert "unreadable export on line 2" in caplog.text


def test_a_log_line_is_one_line(tmp_path):
    """So `wc -l` counts records and a reader can stream them."""
    log = tmp_path / "runs.jsonl"
    append_jsonl(log, {"recipe": {"nested": {"deep": [1, 2]}}})
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert json.loads(log.read_text(encoding="utf-8"))["recipe"]["nested"]["deep"] == [1, 2]
