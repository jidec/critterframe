"""
Parquet table mechanics, with no knowledge of any entity. The other tabular
backend is tested in test_sqlite.py.

Two write shapes, and which one is right is a property of the data rather than
of the caller: `write_table` replaces wholesale, for tables fed by full
snapshots; `upsert_table` merges on a key, for tables built up incrementally
across many runs. The tests that matter most here are the ones about keys --
because a key that silently fails to match doesn't error, it duplicates, and a
mask table that grows a second copy of an occurrence-part on every run is a
project that quietly stops meaning anything.
"""

import pandas as pd
import pytest

from critterframe.storage.tables import (
    load_table,
    table_columns,
    upsert_table,
    write_table,
)

KEYS = ["occurrence_id", "part"]


def rows(*records):
    return pd.DataFrame(list(records))


def a_row(occurrence_id="a", part="organism", value=1):
    return {"occurrence_id": occurrence_id, "part": part, "value": value}


# ---------------------------------------------------------------------------
# write_table -- snapshot replace
# ---------------------------------------------------------------------------


def test_writing_creates_the_parent_directory(tmp_path):
    """
    A project comes into existence lazily, as its first writer needs it. This
    is that writer.
    """
    destination = tmp_path / "project" / "table.parquet"
    write_table(rows(a_row()), destination)
    assert destination.exists()


def test_a_second_write_replaces_everything(tmp_path):
    """
    Snapshot semantics: the newest export IS the complete desired state, so
    what was there before is gone rather than merged.
    """
    path = tmp_path / "table.parquet"
    write_table(rows(a_row("a"), a_row("b")), path)
    write_table(rows(a_row("c")), path)
    assert load_table(path)["occurrence_id"].tolist() == ["c"]


def test_the_index_is_not_written(tmp_path):
    """
    A stored index would come back as a column on the next read and start
    appearing in exports.
    """
    path = tmp_path / "table.parquet"
    frame = rows(a_row("a"), a_row("b")).set_index("occurrence_id")
    write_table(frame.reset_index(), path)
    assert load_table(path).index.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# upsert_table -- incremental merge
# ---------------------------------------------------------------------------


def test_upserting_into_nothing_writes_the_rows(tmp_path):
    path = tmp_path / "table.parquet"
    upsert_table(rows(a_row("a")), path, KEYS)
    assert len(load_table(path)) == 1


def test_a_matching_key_is_replaced_and_the_rest_appended(tmp_path):
    path = tmp_path / "table.parquet"
    upsert_table(rows(a_row("a", value=1), a_row("b", value=2)), path, KEYS)
    upsert_table(rows(a_row("a", value=99), a_row("c", value=3)), path, KEYS)

    stored = load_table(path).set_index("occurrence_id")["value"]
    assert stored.to_dict() == {"a": 99, "b": 2, "c": 3}


def test_the_whole_key_has_to_match(tmp_path):
    """
    (occurrence_id, part) is one key. A new part of an existing occurrence is a
    new row, not a replacement -- which is what lets an organism and its wing
    coexist.
    """
    path = tmp_path / "table.parquet"
    upsert_table(rows(a_row("a", part="organism")), path, KEYS)
    upsert_table(rows(a_row("a", part="wing")), path, KEYS)
    assert len(load_table(path)) == 2


def test_upserting_one_subsets_rows_leaves_anothers_alone(tmp_path):
    """
    Why this table is merged rather than replaced: a project's subsets may be
    processed with entirely different recipes, and a run over one must not
    destroy the other's work.
    """
    path = tmp_path / "table.parquet"
    upsert_table(rows(a_row("amnh1"), a_row("amnh2")), path, KEYS)
    upsert_table(rows(a_row("mcz1")), path, KEYS)
    assert len(load_table(path)) == 3


@pytest.mark.parametrize("key_columns", [["missing"], ["occurrence_id", "nope"]])
def test_a_missing_key_column_raises(tmp_path, key_columns):
    with pytest.raises(KeyError, match="missing key column"):
        upsert_table(rows(a_row()), tmp_path / "table.parquet", key_columns)


def test_a_null_key_raises(tmp_path):
    """
    A null key isn't an identifier: it can never match, so the row would append
    a duplicate on every single run. Loud beats a table that quietly doubles.
    """
    with pytest.raises(ValueError, match="null values in key column"):
        upsert_table(rows(a_row(None)), tmp_path / "table.parquet", KEYS)


def test_a_null_key_in_the_existing_table_raises(tmp_path):
    """
    The same check on the way in, because a table written by an older, laxer
    writer is exactly where a null key would be hiding.
    """
    path = tmp_path / "table.parquet"
    write_table(rows(a_row(None)), path)
    with pytest.raises(ValueError, match="null values in key column"):
        upsert_table(rows(a_row("a")), path, KEYS)


def test_a_changed_key_type_raises_rather_than_duplicating(tmp_path):
    """
    Keys are compared BY VALUE with no coercion, so the integer 1 and the
    string "1" are different keys. Without this guard every row would append
    instead of replacing, and the failure would surface later as an Arrow type
    error naming neither the cause nor the fix.
    """
    path = tmp_path / "table.parquet"
    upsert_table(rows(a_row(1)), path, KEYS)
    with pytest.raises(TypeError, match="key column type"):
        upsert_table(rows(a_row("1")), path, KEYS)


def test_upserting_into_an_empty_stored_table_is_fine(tmp_path):
    """An existing file with no rows has no keys to check and nothing to keep."""
    path = tmp_path / "table.parquet"
    write_table(rows().reindex(columns=["occurrence_id", "part", "value"]), path)
    upsert_table(rows(a_row("a")), path, KEYS)
    assert len(load_table(path)) == 1


# ---------------------------------------------------------------------------
# load_table / table_columns
# ---------------------------------------------------------------------------


def test_a_missing_table_raises_by_default(tmp_path):
    """
    So a missing occurrence table doesn't quietly read as "this project has no
    occurrences" -- the two need different answers.
    """
    with pytest.raises(FileNotFoundError, match="no table at"):
        load_table(tmp_path / "absent.parquet")


def test_a_missing_table_can_be_legitimately_absent(tmp_path):
    """Masks before any segmentation run: absent is the normal state."""
    empty = load_table(tmp_path / "absent.parquet", missing_ok=True,
                       columns=["occurrence_id"])
    assert empty.empty
    assert empty.columns.tolist() == ["occurrence_id"]


def test_columns_can_be_read_selectively(tmp_path):
    path = tmp_path / "table.parquet"
    write_table(rows(a_row()), path)
    assert load_table(path, columns=["value"]).columns.tolist() == ["value"]


def test_table_columns_reads_the_footer_only(tmp_path):
    """
    How a records module asks what an EXISTING table has before deciding what
    to read -- because naming a column the file predates fails the read
    outright, and "read these if they're there" needs an answer that isn't a
    swallowed exception.
    """
    path = tmp_path / "table.parquet"
    write_table(rows(a_row()), path)
    assert set(table_columns(path)) == {"occurrence_id", "part", "value"}
    assert table_columns(tmp_path / "absent.parquet") == []


def test_naming_a_column_a_table_lacks_fails_the_read(tmp_path):
    """The failure table_columns exists to let callers avoid."""
    path = tmp_path / "table.parquet"
    write_table(rows(a_row()), path)
    with pytest.raises(Exception):
        load_table(path, columns=["value", "not_a_column"])
