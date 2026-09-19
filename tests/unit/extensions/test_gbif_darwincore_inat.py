"""
The GBIF Darwin Core extension: reading the archive, picking one multimedia row
per occurrence, and the iNaturalist-only photo URL rewrite.

Everything here is offline -- a GBIF Darwin Core Archive is a file a project
already downloaded, never something this package fetches, so the whole surface
is string/table parsing and is testable against small hand-built archives.
"""

import io
import zipfile

import pandas as pd
import pytest

import critterframe as cf
from critterframe.extensions.gbif_darwincore_inat import archive, ingest

OCCURRENCE_TXT = (
    "gbifID\toccurrenceStatus\tscientificName\n"
    "1\tPRESENT\tOrthetrum trinacria\n"
    "2\tPRESENT\tTrithemis aconita\n"
    "3\tABSENT\tNothing observed\n"
)

MULTIMEDIA_TXT = (
    "gbifID\ttype\tidentifier\tcreator\n"
    "1\tStillImage\thttps://inaturalist-open-data.s3.amazonaws.com/photos/111/original.jpg\tAlice\n"
    "2\tSound\thttps://example.com/call.mp3\tBob\n"
    "2\tStillImage\thttps://observation.org/photos/222.jpg\tBob\n"
)


def write_archive_zip(path, occurrence=OCCURRENCE_TXT, multimedia=MULTIMEDIA_TXT):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("occurrence.txt", occurrence)
        zf.writestr("multimedia.txt", multimedia)
    return path


def write_archive_dir(directory, occurrence=OCCURRENCE_TXT, multimedia=MULTIMEDIA_TXT):
    nested = directory / "dwca" / "data"
    nested.mkdir(parents=True)
    (nested / "occurrence.txt").write_text(occurrence)
    (nested / "multimedia.txt").write_text(multimedia)
    return directory


# ---------------------------------------------------------------------------
# archive: reading tables, zipped or not
# ---------------------------------------------------------------------------


def test_a_table_keeps_every_column_as_a_string():
    """
    Left as strings rather than pandas' guess -- a catalogNumber like "007" or
    a stateProvince abbreviated "NA" must not be silently reinterpreted.
    """
    from io import BytesIO

    table = archive.read_darwincore_table(BytesIO(OCCURRENCE_TXT.encode()))
    assert table["gbifID"].tolist() == ["1", "2", "3"]
    assert table["gbifID"].dtype == object


def test_only_an_empty_field_reads_as_missing():
    """
    keep_default_na is off: GBIF text is full of words that are real values,
    not blanks, and only a genuinely empty field should read as missing.
    """
    from io import BytesIO

    table = archive.read_darwincore_table(
        BytesIO(b"gbifID\tstateProvince\n1\tNA\n2\t\n"))
    assert table["stateProvince"].iloc[0] == "NA"
    assert pd.isna(table["stateProvince"].iloc[1])


def test_a_malformed_row_is_dropped_and_logged(caplog):
    """A row with the wrong field count for its header is dropped, not fatal."""
    from io import BytesIO

    text = (
        "gbifID\ttype\tidentifier\n"
        "1\tStillImage\thttps://example.com/1.jpg\n"
        "2\tStillImage\thttps://example.com/2.jpg\textra\tfields\n"
        "3\tStillImage\thttps://example.com/3.jpg\n"
    )
    with caplog.at_level("WARNING"):
        table = archive.read_darwincore_table(BytesIO(text.encode()))

    assert table["gbifID"].tolist() == ["1", "3"]
    assert "dropped 1 malformed row" in caplog.text


def test_a_leaked_verbatim_multimedia_row_is_recovered_by_name(caplog):
    """
    A GBIF export bug can leak a row shaped like verbatim/multimedia.txt (extra
    datasetKey/datasetID columns, different field order) into the processed
    multimedia.txt -- it's remapped by name onto the real header rather than
    dropped.
    """
    from io import BytesIO

    verbatim_values = {
        "gbifID": "2", "datasetKey": "ds-1", "type": "StillImage", "format": "image/jpeg",
        "identifier": "https://example.com/2.jpg", "references": "", "title": "",
        "description": "", "created": "", "creator": "Bob", "contributor": "",
        "publisher": "", "audience": "", "source": "", "license": "",
        "rightsHolder": "", "datasetID": "",
    }
    bad_row = "\t".join(verbatim_values[field] for field in archive.VERBATIM_MULTIMEDIA_FIELDS)
    text = (
        "gbifID\ttype\tidentifier\tcreator\n"
        "1\tStillImage\thttps://example.com/1.jpg\tAlice\n"
        f"{bad_row}\n"
    )
    with caplog.at_level("WARNING"):
        table = archive.read_darwincore_table(BytesIO(text.encode()))

    assert table["gbifID"].tolist() == ["1", "2"]
    recovered = table.loc[table["gbifID"] == "2"].iloc[0]
    assert recovered["identifier"] == "https://example.com/2.jpg"
    assert recovered["creator"] == "Bob"
    assert "recovered 1 leaked verbatim Multimedia row" in caplog.text


def test_the_archive_is_read_straight_out_of_the_zip(tmp_path):
    """No unzip step required -- the whole point of accepting a .zip at all."""
    zpath = write_archive_zip(tmp_path / "gbif.zip")

    occurrence_df, multimedia_df = archive.read_darwincore_archive(zpath)
    assert len(occurrence_df) == 3
    assert len(multimedia_df) == 3


def test_an_already_extracted_directory_works_too_however_deep_it_sits(tmp_path):
    directory = write_archive_dir(tmp_path)

    occurrence_df, multimedia_df = archive.read_darwincore_archive(directory)
    assert len(occurrence_df) == 3
    assert len(multimedia_df) == 3


def test_usecols_narrows_both_tables_and_always_keeps_the_join_key(tmp_path):
    """
    A processed occurrence.txt is 200+ columns wide under dtype=str -- for a
    multi-gigabyte export, reading only what's needed is the difference
    between fitting in memory and not. gbifID survives even when a caller
    forgets to ask for it, since dropping it would silently break the merge.
    """
    zpath = write_archive_zip(tmp_path / "gbif.zip")

    occurrence_df, multimedia_df = archive.read_darwincore_archive(
        zpath, occurrence_usecols=["scientificName"], multimedia_usecols=["type"])

    assert set(occurrence_df.columns) == {"scientificName", "gbifID"}
    assert set(multimedia_df.columns) == {"type", "gbifID"}
    assert len(occurrence_df) == 3   # every row still read, just narrower


def test_usecols_none_reads_every_column_as_before(tmp_path):
    zpath = write_archive_zip(tmp_path / "gbif.zip")
    occurrence_df, _ = archive.read_darwincore_archive(zpath)
    assert list(occurrence_df.columns) == ["gbifID", "occurrenceStatus", "scientificName"]


def test_raw_archive_bytes_is_the_zip_verbatim(tmp_path):
    zpath = write_archive_zip(tmp_path / "gbif.zip")
    raw_bytes, extension = archive.raw_archive_bytes(zpath)
    assert raw_bytes == zpath.read_bytes()
    assert extension == ".zip"


def test_raw_archive_bytes_from_a_directory_zips_the_two_tables_untouched(tmp_path):
    directory = write_archive_dir(tmp_path)
    raw_bytes, extension = archive.raw_archive_bytes(directory)
    assert extension == ".zip"

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as bundle:
        assert set(bundle.namelist()) == {"occurrence.txt", "multimedia.txt"}
        occurrence = pd.read_csv(bundle.open("occurrence.txt"), sep="\t", dtype=str)
        assert occurrence["gbifID"].tolist() == ["1", "2", "3"]


def test_a_missing_table_says_so(tmp_path):
    zpath = tmp_path / "gbif.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("occurrence.txt", OCCURRENCE_TXT)

    with pytest.raises(FileNotFoundError, match="multimedia.txt"):
        archive.read_darwincore_archive(zpath)


def test_two_matching_files_is_ambiguous_rather_than_a_silent_pick(tmp_path):
    zpath = tmp_path / "gbif.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("occurrence.txt", OCCURRENCE_TXT)
        zf.writestr("a/multimedia.txt", MULTIMEDIA_TXT)
        zf.writestr("b/multimedia.txt", MULTIMEDIA_TXT)

    with pytest.raises(ValueError, match="more than one"):
        archive.read_darwincore_archive(zpath)


def test_a_root_level_file_wins_over_a_nested_verbatim_copy(tmp_path):
    """
    GBIF sometimes ships a second multimedia.txt under verbatim/ alongside the
    root-level processed one -- the root-level file wins rather than raising.
    """
    zpath = tmp_path / "gbif.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("occurrence.txt", OCCURRENCE_TXT)
        zf.writestr("multimedia.txt", MULTIMEDIA_TXT)
        zf.writestr("verbatim/multimedia.txt",
                   "gbifID\tdatasetKey\ttype\n1\tds-1\tStillImage\n")

    occurrence_df, multimedia_df = archive.read_darwincore_archive(zpath)
    assert list(multimedia_df.columns) == ["gbifID", "type", "identifier", "creator"]


def test_neither_a_zip_nor_a_directory_raises(tmp_path):
    plain = tmp_path / "occurrence.txt"
    plain.write_text(OCCURRENCE_TXT)

    with pytest.raises(ValueError, match="neither a directory nor a zip"):
        archive.read_darwincore_archive(plain)


# ---------------------------------------------------------------------------
# rewrite_inat_photo_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", [
    "inaturalist-open-data.s3.amazonaws.com",
    "static.inaturalist.org",
])
def test_an_inat_photo_url_is_rewritten_to_the_requested_size(host):
    url = f"https://{host}/photos/404810759/original.jpg"
    assert ingest.rewrite_inat_photo_size(url, "medium") == \
        f"https://{host}/photos/404810759/medium.jpg"


def test_a_non_inat_url_passes_through_untouched():
    """
    This dataset mixes providers freely -- rewriting a path from Flickr or a
    museum server would just break it.
    """
    url = "https://observation.org/photos/7726784.jpg"
    assert ingest.rewrite_inat_photo_size(url, "small") == url


def test_a_missing_identifier_passes_through():
    assert ingest.rewrite_inat_photo_size(None, "small") is None
    assert pd.isna(ingest.rewrite_inat_photo_size(pd.NA, "small"))


def test_an_unknown_size_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="unknown iNaturalist photo size"):
        ingest.rewrite_inat_photo_size("https://static.inaturalist.org/photos/1/original.jpg",
                                       "huge")


# ---------------------------------------------------------------------------
# select_media
# ---------------------------------------------------------------------------


def multimedia_frame():
    return pd.DataFrame({
        "gbifID": ["1", "1", "2"],
        "type": ["StillImage", "StillImage", "StillImage"],
        "identifier": ["https://x/first.jpg", "https://x/second.jpg", "https://x/only.jpg"],
    })


def test_the_default_rule_takes_the_first_row_per_occurrence():
    selected = ingest.select_media(multimedia_frame())
    assert selected.set_index("gbifID")["identifier"].to_dict() == {
        "1": "https://x/first.jpg", "2": "https://x/only.jpg"}


def test_the_last_rule_takes_the_last_row_per_occurrence():
    selected = ingest.select_media(multimedia_frame(), rule="last")
    assert selected.set_index("gbifID")["identifier"]["1"] == "https://x/second.jpg"


def test_type_filtering_happens_before_the_one_per_occurrence_rule():
    """
    Otherwise "first" could hand back a sound recording for an occurrence
    whose actual first photo sorts later in the file.
    """
    media = pd.DataFrame({
        "gbifID": ["1", "1"],
        "type": ["Sound", "StillImage"],
        "identifier": ["https://x/call.mp3", "https://x/photo.jpg"],
    })
    selected = ingest.select_media(media, media_type="StillImage")
    assert selected["identifier"].tolist() == ["https://x/photo.jpg"]


def test_media_type_none_keeps_every_row_regardless_of_type():
    media = pd.DataFrame({
        "gbifID": ["1", "1"],
        "type": ["Sound", "StillImage"],
        "identifier": ["https://x/call.mp3", "https://x/photo.jpg"],
    })

    filtered = ingest.select_media(media, media_type="StillImage")
    assert filtered["identifier"].tolist() == ["https://x/photo.jpg"]

    unfiltered = ingest.select_media(media, media_type=None)
    assert unfiltered["identifier"].tolist() == ["https://x/call.mp3"]


def test_a_missing_type_column_warns_and_skips_the_filter(caplog):
    media = multimedia_frame().drop(columns=["type"])
    with caplog.at_level("WARNING"):
        selected = ingest.select_media(media)
    assert len(selected) == 2
    assert "no '" in caplog.text


def test_a_callable_rule_picks_explicitly():
    def prefer_second(group):
        return group.iloc[-1]

    selected = ingest.select_media(multimedia_frame(), rule=prefer_second)
    assert selected.set_index("gbifID")["identifier"]["1"] == "https://x/second.jpg"


def test_a_callable_rule_may_decline_an_occurrence_entirely():
    def only_id_one(group):
        return group.iloc[0] if group["gbifID"].iloc[0] == "1" else None

    selected = ingest.select_media(multimedia_frame(), rule=only_id_one)
    assert selected["gbifID"].tolist() == ["1"]


def test_an_unknown_rule_raises():
    with pytest.raises(ValueError, match="unknown media selection rule"):
        ingest.select_media(multimedia_frame(), rule="whatever")


# ---------------------------------------------------------------------------
# merge_occurrence_media
# ---------------------------------------------------------------------------


def test_multimedia_columns_are_prefixed_to_avoid_colliding_with_occurrences():
    """
    occurrence.txt and multimedia.txt both define columns named type and
    license for different things.
    """
    occurrence_df = pd.DataFrame({"gbifID": ["1"], "type": ["Occurrence"]})
    media_df = pd.DataFrame({"gbifID": ["1"], "type": ["StillImage"],
                             "identifier": ["https://x/photo.jpg"]})

    merged = ingest.merge_occurrence_media(occurrence_df, media_df)
    assert merged["type"].iloc[0] == "Occurrence"
    assert merged["media_type"].iloc[0] == "StillImage"


def test_an_occurrence_with_no_usable_image_is_excluded(caplog):
    """
    No image in THIS export isn't the source declaring no organism, so it
    isn't the drop= contract's job -- but an occurrence with no image can't
    be an occurrence-per-image record either, so it's excluded here, with
    the count logged rather than a per-row manifest.
    """
    occurrence_df = pd.DataFrame({"gbifID": ["1", "2"]})
    media_df = pd.DataFrame({"gbifID": ["1"], "identifier": ["https://x/photo.jpg"]})

    with caplog.at_level("INFO"):
        merged = ingest.merge_occurrence_media(occurrence_df, media_df)

    assert len(merged) == 1
    assert "2" not in merged["gbifID"].values
    assert "excluding 1 of 2" in caplog.text


# ---------------------------------------------------------------------------
# ingest_occurrences, end to end against a small archive
# ---------------------------------------------------------------------------


def test_ingesting_an_archive_drops_absent_occurrences_by_default(tmp_path):
    """GBIF's occurrenceStatus="ABSENT" is the source declaring no organism."""
    zpath = write_archive_zip(tmp_path / "gbif.zip")

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath)
    assert sorted(table["occurrence_id"]) == ["1", "2"]


def test_drop_none_keeps_absent_occurrences(tmp_path):
    # Occurrence 3 needs its own media row here: the shared fixture's ABSENT
    # row has none, and this test means to isolate drop= from the separate
    # imageless-exclusion behaviour covered elsewhere.
    occurrence = OCCURRENCE_TXT
    multimedia = MULTIMEDIA_TXT + "3\tStillImage\thttps://x/3.jpg\n"
    zpath = write_archive_zip(tmp_path / "gbif.zip", occurrence=occurrence,
                              multimedia=multimedia)

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath, drop=None)
    assert sorted(table["occurrence_id"]) == ["1", "2", "3"]


def test_ingesting_picks_the_still_image_over_the_sound_recording(tmp_path):
    zpath = write_archive_zip(tmp_path / "gbif.zip")

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath)
    row = table.set_index("occurrence_id").loc["2"]
    assert row["image_url"] == "https://observation.org/photos/222.jpg"


def test_a_present_occurrence_with_no_media_row_is_excluded(tmp_path):
    """
    A PRESENT occurrence GBIF says nothing is absent about, but with no
    multimedia row at all, still doesn't reach the saved occurrence table --
    it has no image to be an occurrence-per-image record with.
    """
    occurrence = (
        "gbifID\toccurrenceStatus\tscientificName\n"
        "1\tPRESENT\tOrthetrum trinacria\n"
        "4\tPRESENT\tNo photo taken\n"
    )
    multimedia = (
        "gbifID\ttype\tidentifier\n"
        "1\tStillImage\thttps://x/1.jpg\n"
    )
    zpath = write_archive_zip(tmp_path / "gbif.zip", occurrence=occurrence,
                              multimedia=multimedia)

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath)
    assert sorted(table["occurrence_id"]) == ["1"]


def test_inat_photo_size_rewrites_only_the_inat_row(tmp_path):
    zpath = write_archive_zip(tmp_path / "gbif.zip")

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath,
                                      inat_photo_size="small")
    by_id = table.set_index("occurrence_id")
    assert by_id.loc["1", "image_url"] == \
        "https://inaturalist-open-data.s3.amazonaws.com/photos/111/small.jpg"
    assert by_id.loc["2", "image_url"] == "https://observation.org/photos/222.jpg"


def test_an_extracted_directory_ingests_the_same_as_its_zip(tmp_path):
    directory = write_archive_dir(tmp_path / "extracted")

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=directory)
    assert sorted(table["occurrence_id"]) == ["1", "2"]


def test_preloaded_tables_work_without_touching_disk(tmp_path):
    occurrence_df, multimedia_df = archive.read_darwincore_archive(
        write_archive_zip(tmp_path / "gbif.zip"))

    table = ingest.ingest_occurrences(tmp_path / "project", occurrence_df=occurrence_df,
                                      multimedia_df=multimedia_df)
    assert sorted(table["occurrence_id"]) == ["1", "2"]


def test_archive_path_and_preloaded_tables_together_is_an_error(tmp_path):
    occurrence_df, multimedia_df = archive.read_darwincore_archive(
        write_archive_zip(tmp_path / "gbif.zip"))

    with pytest.raises(ValueError, match="not both"):
        ingest.ingest_occurrences(tmp_path / "project",
                                  archive_path=tmp_path / "gbif.zip",
                                  occurrence_df=occurrence_df,
                                  multimedia_df=multimedia_df)


def test_no_source_at_all_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="pass archive_path"):
        ingest.ingest_occurrences(tmp_path / "project")


def test_the_staged_csv_does_not_linger_after_ingest(tmp_path):
    """The dated raw import core_ingest.ingest_occurrences archived is the durable copy."""
    zpath = write_archive_zip(tmp_path / "gbif.zip")
    project = tmp_path / "project"

    ingest.ingest_occurrences(project, archive_path=zpath)
    assert not (project / "raw_imports" / ".gbif_occurrences.csv").exists()


def test_the_raw_import_is_the_original_archive_byte_for_byte(tmp_path):
    """
    What gets archived is the .zip exactly as GBIF sent it, not a
    re-serialization of what was parsed out of it and not the merged,
    one-photo-per-occurrence CSV built downstream. Recovering from a bad
    media_rule choice means still having every candidate GBIF offered
    (occurrence 2's sound recording included), not just the one selected.
    """
    zpath = write_archive_zip(tmp_path / "gbif.zip")
    project = tmp_path / "project"

    ingest.ingest_occurrences(project, archive_path=zpath)

    [raw] = (project / "raw_imports").glob("*.zip")
    assert raw.read_bytes() == zpath.read_bytes()

    with zipfile.ZipFile(raw) as archived:
        assert set(archived.namelist()) == {"occurrence.txt", "multimedia.txt"}
        multimedia = pd.read_csv(archived.open("multimedia.txt"), sep="\t", dtype=str)
        assert len(multimedia[multimedia["gbifID"] == "2"]) == 2


def test_the_raw_import_from_an_extracted_directory_zips_both_tables(tmp_path):
    """
    No original .zip to copy here, so the two tables are zipped back together
    straight from disk -- still pandas-free, still not the merged CSV.
    """
    directory = write_archive_dir(tmp_path / "extracted")
    project = tmp_path / "project"

    ingest.ingest_occurrences(project, archive_path=directory)

    [raw] = (project / "raw_imports").glob("*.zip")
    with zipfile.ZipFile(raw) as archived:
        assert set(archived.namelist()) == {"occurrence.txt", "multimedia.txt"}
        multimedia = pd.read_csv(archived.open("multimedia.txt"), sep="\t", dtype=str)
        assert len(multimedia[multimedia["gbifID"] == "2"]) == 2


def test_media_rule_and_related_decisions_land_in_the_import_manifest(tmp_path):
    zpath = write_archive_zip(tmp_path / "gbif.zip")
    project = tmp_path / "project"

    ingest.ingest_occurrences(project, archive_path=zpath, media_rule="last",
                              inat_photo_size="small")

    manifests = cf.load_imports(project)
    assert len(manifests) == 1
    extra = manifests.iloc[0]["extra"]
    assert extra["media_rule"] == "last"
    assert extra["inat_photo_size"] == "small"
    assert extra["source"] == str(zpath)


def test_group_col_and_max_per_group_reach_the_core_ingest(tmp_path):
    """
    A GBIF download is routinely dominated by a few common species -- this is
    the option that thins it back out, forwarded straight through to
    critterframe.ingest.ingest_occurrences.
    """
    occurrence = (
        "gbifID\toccurrenceStatus\tspecies\n"
        "1\tPRESENT\tcommon\n"
        "2\tPRESENT\tcommon\n"
        "3\tPRESENT\tcommon\n"
        "4\tPRESENT\trare\n"
    )
    multimedia = (
        "gbifID\ttype\tidentifier\n"
        "1\tStillImage\thttps://x/1.jpg\n"
        "2\tStillImage\thttps://x/2.jpg\n"
        "3\tStillImage\thttps://x/3.jpg\n"
        "4\tStillImage\thttps://x/4.jpg\n"
    )
    zpath = write_archive_zip(tmp_path / "gbif.zip", occurrence=occurrence,
                              multimedia=multimedia)

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath,
                                      group_col="species", max_per_group=1)
    counts = table["species"].value_counts()
    assert counts["common"] == 1
    assert counts["rare"] == 1


# ---------------------------------------------------------------------------
# ingest_occurrences: cross-source deduplication (dedupe_key_cols)
# ---------------------------------------------------------------------------


DUPLICATE_OCCURRENCE_TXT = (
    "gbifID\toccurrenceStatus\tspecies\tdecimalLatitude\tdecimalLongitude\teventDate\n"
    "1\tPRESENT\tOrthetrum trinacria\t40.123401\t-73.987601\t2024-05-01\n"
    "2\tPRESENT\tOrthetrum trinacria\t40.123449\t-73.987649\t2024-05-01\n"
    "3\tPRESENT\tTrithemis aconita\t10.0\t20.0\t2024-06-15\n"
)
DUPLICATE_MULTIMEDIA_TXT = (
    "gbifID\ttype\tidentifier\n"
    "1\tStillImage\thttps://inaturalist.org/1.jpg\n"
    "2\tStillImage\thttps://observation.org/2.jpg\n"
    "3\tStillImage\thttps://inaturalist.org/3.jpg\n"
)


def test_deduplication_is_off_by_default(tmp_path):
    """
    Only a project actually combining more than one GBIF-mediated source
    needs this -- gbifID 1 and 2 would fingerprint alike (~5m apart, same
    eventDate) if dedupe_key_cols were given, but nothing is asked for here.
    """
    zpath = write_archive_zip(tmp_path / "gbif.zip", occurrence=DUPLICATE_OCCURRENCE_TXT,
                              multimedia=DUPLICATE_MULTIMEDIA_TXT)

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath)
    assert sorted(table["occurrence_id"]) == ["1", "2", "3"]


def test_opting_in_deduplicates_the_same_sighting_from_two_sources(tmp_path):
    """
    gbifID 1 and 2 are one real sighting published through two aggregators --
    close enough in lat/lon (~5m) and on the same eventDate to fingerprint
    alike under DEFAULT_DEDUPE_KEY_COLS, so only one survives even though
    nothing shares an id.
    """
    zpath = write_archive_zip(tmp_path / "gbif.zip", occurrence=DUPLICATE_OCCURRENCE_TXT,
                              multimedia=DUPLICATE_MULTIMEDIA_TXT)

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath,
                                      dedupe_key_cols=ingest.DEFAULT_DEDUPE_KEY_COLS)
    assert len(table) == 2   # one of {1, 2}, plus 3
    assert "3" in set(table["occurrence_id"])


def test_missing_dedupe_columns_is_not_a_failure(tmp_path, caplog):
    """
    The shared OCCURRENCE_TXT fixture has no coordinates at all -- narrower
    than DEFAULT_INAT_OCCURRENCE_COLUMNS would ever be in practice, but a
    caller narrowing occurrence_columns without them (while still opting in to
    dedupe_key_cols) is a real, unexceptional case, not a mistake worth
    failing a multi-gigabyte ingest over.
    """
    zpath = write_archive_zip(tmp_path / "gbif.zip")

    with caplog.at_level("WARNING"):
        table = ingest.ingest_occurrences(
            tmp_path / "project", archive_path=zpath,
            dedupe_key_cols=ingest.DEFAULT_DEDUPE_KEY_COLS)
    assert sorted(table["occurrence_id"]) == ["1", "2"]
    assert "skipping deduplication" in caplog.text


def test_dedupe_happens_before_the_group_cap(tmp_path):
    """
    A per-species cap must count real specimens, not a sighting inflated by
    however many aggregators published it -- capping "common" to 1 before
    deduping could keep two rows that are the same sighting.
    """
    occurrence = (
        "gbifID\toccurrenceStatus\tspecies\tdecimalLatitude\tdecimalLongitude\teventDate\n"
        "1\tPRESENT\tcommon\t40.123401\t-73.987601\t2024-05-01\n"
        "2\tPRESENT\tcommon\t40.123449\t-73.987649\t2024-05-01\n"
        "3\tPRESENT\tcommon\t51.5\t-0.1\t2024-06-15\n"
    )
    multimedia = (
        "gbifID\ttype\tidentifier\n"
        "1\tStillImage\thttps://x/1.jpg\n"
        "2\tStillImage\thttps://x/2.jpg\n"
        "3\tStillImage\thttps://x/3.jpg\n"
    )
    zpath = write_archive_zip(tmp_path / "gbif.zip", occurrence=occurrence,
                              multimedia=multimedia)

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath,
                                      dedupe_key_cols=ingest.DEFAULT_DEDUPE_KEY_COLS,
                                      group_col="species", max_per_group=2)
    # Without dedupe-before-cap this could land on {1, 2} -- one real sighting
    # twice -- and never reach 3 at all.
    assert len(table) == 2
    assert "3" in set(table["occurrence_id"])


def test_dedupe_decisions_land_in_the_import_manifest(tmp_path):
    zpath = write_archive_zip(tmp_path / "gbif.zip", occurrence=DUPLICATE_OCCURRENCE_TXT,
                              multimedia=DUPLICATE_MULTIMEDIA_TXT)
    project = tmp_path / "project"

    ingest.ingest_occurrences(project, archive_path=zpath,
                              dedupe_key_cols=("decimalLatitude",), dedupe_rule="first")

    extra = cf.load_imports(project).iloc[0]["extra"]
    assert extra["dedupe_key_cols"] == ["decimalLatitude"]
    assert extra["dedupe_rule"] == "first"


def test_omitting_dedupe_key_cols_is_recorded_as_none(tmp_path):
    """The default (nothing passed) and an explicit dedupe_key_cols=None must
    record identically -- they're the same decision."""
    zpath = write_archive_zip(tmp_path / "gbif.zip")
    project = tmp_path / "project"

    ingest.ingest_occurrences(project, archive_path=zpath)

    extra = cf.load_imports(project).iloc[0]["extra"]
    assert extra["dedupe_key_cols"] is None


def test_reingesting_with_a_changed_dedupe_decision_does_reparse(tmp_path, monkeypatch):
    """Same reasoning as group_col/max_per_group: a changed judgement about
    the same bytes is still a different import, not a skip."""
    zpath = write_archive_zip(tmp_path / "gbif.zip", occurrence=DUPLICATE_OCCURRENCE_TXT,
                              multimedia=DUPLICATE_MULTIMEDIA_TXT)
    project = tmp_path / "project"

    ingest.ingest_occurrences(project, archive_path=zpath,
                              dedupe_key_cols=ingest.DEFAULT_DEDUPE_KEY_COLS,
                              trust_source_file_unchanged=True)
    table = ingest.ingest_occurrences(project, archive_path=zpath,
                                      trust_source_file_unchanged=True)

    assert sorted(table["occurrence_id"]) == ["1", "2", "3"]
    assert len(cf.load_imports(project)) == 2


def test_a_repeat_ingest_with_dedupe_on_skips_the_parse(tmp_path, monkeypatch):
    """The skip check has to hash the dedupe decisions too, or it never
    recognizes an import that deduplicated."""
    zpath = write_archive_zip(tmp_path / "gbif.zip", occurrence=DUPLICATE_OCCURRENCE_TXT,
                              multimedia=DUPLICATE_MULTIMEDIA_TXT)
    project = tmp_path / "project"
    kwargs = dict(archive_path=zpath, dedupe_key_cols=ingest.DEFAULT_DEDUPE_KEY_COLS)

    ingest.ingest_occurrences(project, **kwargs)
    monkeypatch.setattr(archive, "read_darwincore_archive", _parse_forbidden)
    ingest.ingest_occurrences(project, **kwargs)


# ---------------------------------------------------------------------------
# ingest_occurrences: prioritize_inat
# ---------------------------------------------------------------------------


def _parse_forbidden(*a, **kw):
    raise AssertionError("read_darwincore_archive ran on a repeat ingest")


def write_sourced_archive(path, rows):
    """rows: (gbifID, institutionCode, species, lat) -- one photo each."""
    occurrence = "gbifID\toccurrenceStatus\tinstitutionCode\tspecies\t" \
                 "decimalLatitude\tdecimalLongitude\teventDate\n"
    multimedia = "gbifID\ttype\tidentifier\n"
    for gbif_id, institution, species, lat in rows:
        occurrence += f"{gbif_id}\tPRESENT\t{institution}\t{species}\t{lat}\t20.0\t2024-05-01\n"
        multimedia += f"{gbif_id}\tStillImage\thttps://x/{gbif_id}.jpg\n"
    return write_archive_zip(path, occurrence=occurrence, multimedia=multimedia)


MIXED_SOURCE_ROWS = [
    ("1", "iNaturalist", "common", 1.0),
    ("2", "iNaturalist", "common", 2.0),
    ("3", "iNaturalist", "common", 3.0),
    ("4", "Observation.org", "common", 4.0),
    ("5", "Observation.org", "common", 5.0),
    ("6", "iNaturalist", "rare", 6.0),
    ("7", "Observation.org", "rare", 7.0),
    ("8", "", "rare", 8.0),
    ("9", "Observation.org", "rare", 9.0),
]


def test_prioritize_inat_caps_from_inat_rows_only_when_there_are_enough(tmp_path):
    zpath = write_sourced_archive(tmp_path / "gbif.zip", MIXED_SOURCE_ROWS)

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath,
                                      group_col="species", max_per_group=2,
                                      prioritize_inat=True)
    common = table[table["species"] == "common"]
    rare = table[table["species"] == "rare"]
    assert (common["institutionCode"] == "iNaturalist").all()
    assert len(common) == 2
    assert "6" in set(rare["occurrence_id"])   # the only iNat row, plus one other
    assert len(rare) == 2


def test_prioritize_inat_lets_dedupe_remove_only_the_other_source(tmp_path):
    rows = [("1", "Observation.org", "common", 1.0),
            ("2", "iNaturalist", "common", 1.0),     # same sighting as 1
            ("3", "iNaturalist", "common", 3.0),
            ("4", "iNaturalist", "common", 3.0)]     # coincidental iNat match
    zpath = write_sourced_archive(tmp_path / "gbif.zip", rows)

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath,
                                      dedupe_key_cols=ingest.DEFAULT_DEDUPE_KEY_COLS,
                                      prioritize_inat=True)
    assert sorted(table["occurrence_id"]) == ["2", "3", "4"]


def test_prioritize_inat_off_leaves_the_import_hash_alone(tmp_path):
    zpath = write_sourced_archive(tmp_path / "gbif.zip", MIXED_SOURCE_ROWS)
    project = tmp_path / "project"

    ingest.ingest_occurrences(project, archive_path=zpath, prioritize_inat=False)
    manifest = cf.load_imports(project).iloc[0]
    assert "prefer" not in manifest.dropna().index
    assert "prioritize_inat" not in manifest["extra"]


def test_toggling_prioritize_inat_reparses_and_repeating_it_does_not(tmp_path, monkeypatch):
    zpath = write_sourced_archive(tmp_path / "gbif.zip", MIXED_SOURCE_ROWS)
    project = tmp_path / "project"
    kwargs = dict(archive_path=zpath, group_col="species", max_per_group=2)

    ingest.ingest_occurrences(project, **kwargs)
    ingest.ingest_occurrences(project, prioritize_inat=True, **kwargs)
    assert len(cf.load_imports(project)) == 2

    monkeypatch.setattr(archive, "read_darwincore_archive", _parse_forbidden)
    ingest.ingest_occurrences(project, prioritize_inat=True, **kwargs)
    ingest.ingest_occurrences(project, prioritize_inat=True,
                              trust_source_file_unchanged=True, **kwargs)


def test_prioritize_inat_needs_institution_code(tmp_path):
    zpath = write_archive_zip(tmp_path / "gbif.zip")
    with pytest.raises(KeyError, match="institutionCode"):
        ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath,
                                  group_col="scientificName", max_per_group=1,
                                  prioritize_inat=True)


def test_occurrence_columns_narrows_the_read_end_to_end(tmp_path):
    """
    The merge needs gbifID -- that survives without being named -- but
    occurrenceStatus (what default drop=ABSENT_OCCURRENCES reads) has to be
    asked for explicitly, same as any other column drop=/group_col/transform
    touches. A column nobody asked for (scientificName) does not make it in.
    """
    zpath = write_archive_zip(tmp_path / "gbif.zip")

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath,
                                      occurrence_columns=["occurrenceStatus"])
    assert "scientificName" not in table.columns
    assert sorted(table["occurrence_id"]) == ["1", "2"]   # ABSENT still dropped


def test_the_raw_import_keeps_columns_usecols_would_have_dropped(tmp_path):
    """
    occurrence_columns narrows what's PARSED for this ingest; the archived raw
    import is copied from the untouched source, so every column is still
    recoverable regardless of what a particular ingest needed.
    """
    zpath = write_archive_zip(tmp_path / "gbif.zip")
    project = tmp_path / "project"

    ingest.ingest_occurrences(project, archive_path=zpath,
                              occurrence_columns=["occurrenceStatus"])

    [raw] = (project / "raw_imports").glob("*.zip")
    with zipfile.ZipFile(raw) as archived:
        occurrence = pd.read_csv(archived.open("occurrence.txt"), sep="\t", dtype=str)
        assert "scientificName" in occurrence.columns


def test_a_numeric_looking_string_column_is_not_silently_coerced(tmp_path):
    """
    The guarantee read_darwincore_table exists for (a catalogNumber like "007"
    stays a string) has to survive all the way to the saved table, not just
    the initial parse -- passing the merged table to core directly (df=)
    rather than round-tripping it through a written-then-reread CSV is what
    keeps that true; a plain CSV round trip would let "007" come back as 7.
    """
    occurrence = (
        "gbifID\toccurrenceStatus\tcatalogNumber\n"
        "1\tPRESENT\t007\n"
    )
    multimedia = "gbifID\ttype\tidentifier\n1\tStillImage\thttps://x/1.jpg\n"
    zpath = write_archive_zip(tmp_path / "gbif.zip", occurrence=occurrence,
                              multimedia=multimedia)

    table = ingest.ingest_occurrences(tmp_path / "project", archive_path=zpath)
    assert table.set_index("occurrence_id").loc["1", "catalogNumber"] == "007"


def test_default_inat_occurrence_columns_works_end_to_end(tmp_path):
    """
    DEFAULT_INAT_OCCURRENCE_COLUMNS is meant to be passed as
    occurrence_columns= against a real GBIF export, whose processed
    occurrence.txt carries every one of these as a standard interpreted
    field (blank where the source has no value) -- it has to include
    occurrenceStatus (default drop=ABSENT_OCCURRENCES needs it) and actually
    narrow the read, not just exist as an unused constant.
    """
    header = list(ingest.DEFAULT_INAT_OCCURRENCE_COLUMNS) + ["catalogNumber"]
    row1 = {c: "" for c in header}
    row1.update(gbifID="1", occurrenceStatus="PRESENT",
               scientificName="Orthetrum trinacria", catalogNumber="XYZ-1")
    row2 = {c: "" for c in header}
    row2.update(gbifID="2", occurrenceStatus="ABSENT",
               scientificName="Nothing observed", catalogNumber="XYZ-2")
    occurrence = "\t".join(header) + "\n" + "\n".join(
        "\t".join(row[c] for c in header) for row in (row1, row2)) + "\n"
    zpath = write_archive_zip(tmp_path / "gbif.zip", occurrence=occurrence)

    table = ingest.ingest_occurrences(
        tmp_path / "project", archive_path=zpath,
        occurrence_columns=ingest.DEFAULT_INAT_OCCURRENCE_COLUMNS)

    assert sorted(table["occurrence_id"]) == ["1"]        # ABSENT still dropped
    assert table.set_index("occurrence_id").loc["1", "scientificName"] == \
        "Orthetrum trinacria"
    assert "catalogNumber" not in table.columns            # not in the default set


def test_reingesting_the_same_archive_skips_the_expensive_parse(tmp_path, monkeypatch):
    """
    The scenario this exists for: a multi-gigabyte GBIF archive whose parse
    takes minutes and gigabytes of RAM. A repeat ingest with identical
    decisions must not pay that cost again just to discover, afterward, that
    there was nothing new -- already_ingested is checked BEFORE
    read_darwincore_archive runs, not after.
    """
    zpath = write_archive_zip(tmp_path / "gbif.zip")
    project = tmp_path / "project"

    first = ingest.ingest_occurrences(project, archive_path=zpath)

    def _no_parse(*a, **kw):
        raise AssertionError("read_darwincore_archive ran on a repeat ingest "
                             "-- already_ingested should have skipped it")

    monkeypatch.setattr(archive, "read_darwincore_archive", _no_parse)

    second = ingest.ingest_occurrences(project, archive_path=zpath)
    pd.testing.assert_frame_equal(first, second)


def test_reingesting_with_different_decisions_does_reparse(tmp_path, monkeypatch):
    """The parse-skipping check has to be sensitive to the same decisions
    core's own idempotency check is, not just the raw bytes."""
    zpath = write_archive_zip(tmp_path / "gbif.zip")
    project = tmp_path / "project"

    ingest.ingest_occurrences(project, archive_path=zpath, drop=None)
    table = ingest.ingest_occurrences(project, archive_path=zpath)   # default drop

    assert sorted(table["occurrence_id"]) == ["1", "2"]    # ABSENT dropped this time


def test_trust_source_file_unchanged_skips_the_copy_and_hash(tmp_path, monkeypatch):
    """
    Same path, size, and mtime as a previous import: raw_archive_bytes is
    never called at all, not just the parse.
    """
    zpath = write_archive_zip(tmp_path / "gbif.zip")
    project = tmp_path / "project"

    first = ingest.ingest_occurrences(project, archive_path=zpath,
                                      trust_source_file_unchanged=True)

    def _no_copy(*a, **kw):
        raise AssertionError("raw_archive_bytes ran on a fingerprint hit")
    monkeypatch.setattr(archive, "raw_archive_bytes", _no_copy)

    second = ingest.ingest_occurrences(project, archive_path=zpath,
                                       trust_source_file_unchanged=True)
    pd.testing.assert_frame_equal(first, second)


def test_trust_source_file_unchanged_falls_back_when_the_archive_changed(tmp_path):
    """A fingerprint miss still does a real copy+hash and a correct re-ingest."""
    zpath = write_archive_zip(tmp_path / "gbif.zip")
    project = tmp_path / "project"
    ingest.ingest_occurrences(project, archive_path=zpath, trust_source_file_unchanged=True)

    changed_occurrence = OCCURRENCE_TXT + "4\tPRESENT\tAnax junius\n"
    changed_multimedia = MULTIMEDIA_TXT + "4\tStillImage\thttps://x/4.jpg\n"
    write_archive_zip(zpath, occurrence=changed_occurrence, multimedia=changed_multimedia)

    table = ingest.ingest_occurrences(project, archive_path=zpath,
                                      trust_source_file_unchanged=True)
    assert "4" in table["occurrence_id"].tolist()


def test_trust_source_file_unchanged_still_reingests_on_a_changed_decision(tmp_path):
    """Same archive, different drop= -- still correctly re-ingests, not skips."""
    zpath = write_archive_zip(tmp_path / "gbif.zip")
    project = tmp_path / "project"

    ingest.ingest_occurrences(project, archive_path=zpath, drop=None,
                              trust_source_file_unchanged=True)
    table = ingest.ingest_occurrences(project, archive_path=zpath,
                                      trust_source_file_unchanged=True)   # default drop

    assert sorted(table["occurrence_id"]) == ["1", "2"]


def test_the_zip_member_is_streamed_not_materialized_whole(tmp_path, monkeypatch):
    """
    read_darwincore_archive must not call ZipFile.read() on either member --
    that would decompress the whole table into one Python bytes object before
    parsing, which is exactly the extra copy streaming exists to avoid.
    """
    zpath = write_archive_zip(tmp_path / "gbif.zip")

    def _no_eager_read(self, name, *a, **kw):
        raise AssertionError(f"ZipFile.read() was called for {name!r} -- "
                             "the member should be streamed via .open() instead")

    monkeypatch.setattr(zipfile.ZipFile, "read", _no_eager_read)

    occurrence_df, multimedia_df = archive.read_darwincore_archive(zpath)
    assert len(occurrence_df) == 3
    assert len(multimedia_df) == 3
