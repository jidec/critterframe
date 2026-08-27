"""
The occurrence table: two column names agreed on, ids that work, everything
else left alone.

The id rules are the strict part, and deliberately so. A duplicate id or a
blank one used to drop the offending row with a warning, which was wrong: both
mean the source disagrees with the package's central rule that one occurrence is
one organism in one image, and the fix is a judgement about the DATA -- are
these two rows the same specimen photographed twice, or two specimens given the
same number? Nothing here can answer that, so it stops instead of guessing.
"""

import numpy as np
import pandas as pd
import pytest

from critterframe.records import occurrences as occurrence_records
from critterframe.records.occurrences import ID_COL, IMAGE_URL_COL


def source_table(**columns):
    columns.setdefault("id", ["a", "b"])
    return pd.DataFrame(columns)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_the_id_column_is_renamed():
    normalized = occurrence_records.normalize(source_table(), id_col="id")
    assert ID_COL in normalized.columns
    assert "id" not in normalized.columns


def test_an_already_canonical_table_needs_no_id_col():
    frame = pd.DataFrame({ID_COL: ["a"]})
    assert occurrence_records.normalize(frame)[ID_COL].tolist() == ["a"]


def test_a_table_with_no_id_says_what_to_pass():
    with pytest.raises(KeyError, match="pass id_col"):
        occurrence_records.normalize(pd.DataFrame({"name": ["a"]}))


def test_the_image_url_column_is_optional():
    """
    Nothing requires it -- only download_images() reads it, and a project whose
    images came from a local folder has no URLs at all.
    """
    normalized = occurrence_records.normalize(source_table(), id_col="id")
    assert IMAGE_URL_COL not in normalized.columns


def test_the_image_url_column_is_renamed_when_given():
    frame = source_table(photo=["http://x/1.jpg", "http://x/2.jpg"])
    normalized = occurrence_records.normalize(frame, id_col="id",
                                              image_col="photo")
    assert normalized[IMAGE_URL_COL].tolist() == ["http://x/1.jpg",
                                                  "http://x/2.jpg"]


def test_everything_else_the_source_sent_is_kept():
    """
    Normalization means agreeing on the two columns the package needs, not
    discarding the rest -- the extra columns are the science.
    """
    frame = source_table(species=["Anax", "Libellula"], notes=["x", "y"])
    normalized = occurrence_records.normalize(frame, id_col="id")
    assert {"species", "notes"} <= set(normalized.columns)


def test_ids_become_strings():
    """
    Sources number their occurrences. The same id read back from parquet, an
    LMDB key, and a sqlite column would otherwise compare unequal depending on
    integer-vs-string typing -- which surfaces as "the pipeline processed
    nothing".
    """
    frame = pd.DataFrame({ID_COL: [1, 2]})
    assert occurrence_records.normalize(frame)[ID_COL].tolist() == ["1", "2"]


def test_datetime_and_numeric_columns_are_coerced():
    frame = source_table(when=["2024-01-01", "2024-06-30"], size=["1.5", "2"])
    normalized = occurrence_records.normalize(frame, id_col="id",
                                              datetime_cols=["when"],
                                              numeric_cols=["size"])
    assert pd.api.types.is_datetime64_any_dtype(normalized["when"])
    assert normalized["size"].tolist() == [1.5, 2.0]


def test_coercion_is_best_effort():
    """
    One malformed timestamp shouldn't cost the whole ingest, so an unparseable
    value becomes NaT/NaN rather than raising.
    """
    frame = source_table(when=["2024-01-01", "not a date"],
                         size=["1.5", "not a number"])
    normalized = occurrence_records.normalize(frame, id_col="id",
                                              datetime_cols=["when"],
                                              numeric_cols=["size"])
    assert pd.isna(normalized["when"].iloc[1])
    assert np.isnan(normalized["size"].iloc[1])


def test_naming_a_column_the_source_lacks_is_harmless():
    """
    Every caller lists a superset, because external exports vary in which
    optional columns they carry.
    """
    normalized = occurrence_records.normalize(
        source_table(), id_col="id", datetime_cols=["absent"],
        numeric_cols=["also_absent"])
    assert len(normalized) == 2


def test_normalize_does_not_touch_the_caller_s_frame():
    frame = source_table()
    occurrence_records.normalize(frame, id_col="id")
    assert "id" in frame.columns


# ---------------------------------------------------------------------------
# validate_ids
# ---------------------------------------------------------------------------


def test_duplicate_ids_raise_and_name_the_offenders():
    """
    Not a row to discard: a repeated id means the source has two rows for one
    organism, and keeping whichever came first picks an arbitrary winner.
    """
    frame = pd.DataFrame({ID_COL: ["a", "b", "a"]})
    with pytest.raises(ValueError, match="duplicate occurrence id"):
        occurrence_records.validate_ids(frame)


@pytest.mark.parametrize("blank", [None, np.nan, "", pd.NA, pd.NaT])
def test_a_row_with_no_id_raises(blank):
    """
    Every spelling a null takes after astype(str) -- "", "nan", "None",
    "<NA>", "NaT". Any of them means a row nothing downstream can key on.
    """
    frame = pd.DataFrame({ID_COL: ["a", blank]})
    frame[ID_COL] = frame[ID_COL].astype(str)
    with pytest.raises(ValueError, match="have no occurrence id"):
        occurrence_records.validate_ids(frame)


def test_the_error_names_the_source_column():
    """So the message points at what to fix in the export, not at our name for it."""
    frame = pd.DataFrame({ID_COL: ["a", "a"]})
    with pytest.raises(ValueError, match="'detection_id'"):
        occurrence_records.validate_ids(frame, source="detection_id")


def test_validating_a_table_without_the_column_raises():
    with pytest.raises(KeyError, match="no 'occurrence_id' column"):
        occurrence_records.validate_ids(pd.DataFrame({"other": [1]}))


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_saving_validates_again(tmp_path):
    """
    The one place every write passes through -- ingest_images builds its rows
    from filenames without going near normalize(), and two colliding stems
    would otherwise reach the table unchallenged.
    """
    with pytest.raises(ValueError, match="duplicate occurrence id"):
        occurrence_records.save_occurrences(tmp_path,
                                            pd.DataFrame({ID_COL: ["a", "a"]}))


def test_a_save_replaces_the_whole_table(tmp_path):
    """
    Snapshot semantics: an external export is the complete desired state, not a
    delta.
    """
    occurrence_records.save_occurrences(tmp_path,
                                        pd.DataFrame({ID_COL: ["a", "b"]}))
    occurrence_records.save_occurrences(tmp_path, pd.DataFrame({ID_COL: ["c"]}))
    assert occurrence_records.occurrence_ids(tmp_path) == ["c"]


def test_ids_come_back_as_strings(tmp_path):
    occurrence_records.save_occurrences(tmp_path,
                                        pd.DataFrame({ID_COL: ["1", "2"]}))
    assert occurrence_records.load_occurrences(tmp_path)[ID_COL].tolist() == ["1", "2"]


def test_columns_can_be_read_selectively(tmp_path):
    """The id is always included, since every caller keys on it."""
    occurrence_records.save_occurrences(
        tmp_path, pd.DataFrame({ID_COL: ["a"], "species": ["Anax"],
                                "notes": ["x"]}))
    narrow = occurrence_records.load_occurrences(tmp_path, columns=["species"])
    assert narrow.columns.tolist() == [ID_COL, "species"]


def test_an_uningested_project_can_be_counted_without_raising(tmp_path):
    assert occurrence_records.occurrence_count(tmp_path) == 0


def test_loading_an_uningested_project_raises_by_default(tmp_path):
    with pytest.raises(FileNotFoundError):
        occurrence_records.load_occurrences(tmp_path)


# ---------------------------------------------------------------------------
# ids_digest
# ---------------------------------------------------------------------------


def test_a_digest_identifies_the_set_not_the_order():
    """
    The same 4,000 specimens listed in a different order are the same training
    data, and a record that said otherwise would report a difference nobody
    made.
    """
    assert (occurrence_records.ids_digest(["b", "a", "c"])
            == occurrence_records.ids_digest(["a", "b", "c"]))


def test_a_digest_ignores_duplicates():
    assert (occurrence_records.ids_digest(["a", "a", "b"])
            == occurrence_records.ids_digest(["a", "b"]))


def test_a_digest_reads_numbers_as_the_ids_they_are():
    assert occurrence_records.ids_digest([1, 2]) == occurrence_records.ids_digest(["1", "2"])


def test_a_different_set_digests_differently():
    assert (occurrence_records.ids_digest(["a", "b"])
            != occurrence_records.ids_digest(["a", "c"]))


def test_an_empty_set_has_a_digest():
    """A split that came out empty is still a fact about what was trained on."""
    assert isinstance(occurrence_records.ids_digest([]), str)
