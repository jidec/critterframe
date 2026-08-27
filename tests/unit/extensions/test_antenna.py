"""
The Antenna light-trap extension: parsing, paging, and one lazily-read .env.

Antenna has already done the "one organism per image" separation CritterFrame
requires upstream of ingest -- it hands back per-detection crops cut from a
sheet image. What the extension adds is the way back UP that hierarchy: from a
crop's URL to the sheet it came from, and from the sheet to the deployment
session, which is the thing a reference card is set up for and therefore the
scope a scale calibration attaches to.

All of it is string parsing and paging, so all of it is testable offline: the
parsers take URLs, and every network call takes an injectable session.
"""

import numpy as np
import pandas as pd
import pytest

from critterframe.extensions.antenna_lighttraps import api, ingest
from critterframe.extensions.antenna_lighttraps.calibrations import (
    scale as antenna_scale,
)
from helpers.stubs import FakeResponse, FakeSession

CROP_URL = ("https://object-arbutus.cloud.computecanada.ca/ami-media-staging/"
            "uploads/detections/199/2026-06-11/"
            "bronzeBobcat_2026_06_11__01_15_20_HDR0_detection_2121474.jpg")
SHEET_ID = "bronzeBobcat/2026-06-11/bronzeBobcat_2026_06_11__01_15_20_HDR0.jpg"


# ---------------------------------------------------------------------------
# Parsing a crop back to its sheet and session
# ---------------------------------------------------------------------------


def test_a_crop_url_names_the_sheet_it_was_cut_from():
    assert ingest.parse_sheet_image_id(CROP_URL) == SHEET_ID


def test_the_session_is_the_device_and_the_night():
    """
    A reference card is set up once per deployment rather than once per image,
    so this is the key a scale measurement attaches to.
    """
    assert ingest.parse_session_path(SHEET_ID) == "bronzeBobcat/2026-06-11"


@pytest.mark.parametrize("url", [
    None,
    np.nan,
    "https://example.com/nothing",
    "https://example.com/uploads/plain_crop.jpg",
])
def test_a_url_that_does_not_parse_gives_nothing_rather_than_a_guess(url, caplog):
    """
    A wrong sheet id would attach a scale measured on one night to crops from
    another, which is worse than having no scale at all.
    """
    with caplog.at_level("WARNING"):
        assert pd.isna(ingest.parse_sheet_image_id(url))


def test_a_sheet_id_that_does_not_parse_gives_nothing():
    assert pd.isna(ingest.parse_session_path("no_slashes_here"))
    assert pd.isna(ingest.parse_session_path(None))


def test_the_derived_columns_land_on_the_table():
    table = pd.DataFrame({"occurrence_id": ["1", "2"],
                          "image_url": [CROP_URL, None]})
    derived = ingest.add_derived_columns(table)

    assert derived["sheet_image_id"].iloc[0] == SHEET_ID
    assert derived["session_path"].iloc[0] == "bronzeBobcat/2026-06-11"
    assert pd.isna(derived["session_path"].iloc[1])


def test_deriving_reads_the_normalized_column_name():
    """
    It runs after normalization, so it reads image_url rather than Antenna's
    own column name -- and says so when there is none.
    """
    with pytest.raises(KeyError, match="no 'image_url' column"):
        ingest.add_derived_columns(pd.DataFrame({"occurrence_id": ["1"]}))


def test_the_non_organism_vocabulary_belongs_to_the_extension():
    """
    Which determinations mean "no organism here" is knowledge about the source,
    so the extension owns the vocabulary and the core owns the mechanism.
    """
    assert ingest.NON_ORGANISM_DETERMINATIONS == {
        "determination_name": ["Not Lepidoptera"]}


# ---------------------------------------------------------------------------
# The API surface, against a fake session
# ---------------------------------------------------------------------------


def test_paging_follows_the_server_s_own_next_link():
    """
    Rather than counting pages ourselves: a page size the server silently caps,
    or a record inserted mid-walk, then costs us no rows.
    """
    session = FakeSession({"captures/": [
        FakeResponse(json_data={"count": 3, "results": [{"id": 1}, {"id": 2}],
                                "next": "https://antenna/api/v2/captures/?page=2"}),
        FakeResponse(json_data={"count": 3, "results": [{"id": 3}], "next": None}),
    ]})

    walked = list(api.paginate(session, "captures/"))
    assert [record["id"] for record in walked] == [1, 2, 3]


def test_the_first_request_carries_the_query_and_the_rest_do_not():
    """`next` already encodes them, and re-sending them can conflict."""
    session = FakeSession({"captures/": [
        FakeResponse(json_data={"count": 1, "results": [{"id": 1}],
                                "next": "https://antenna/api/v2/captures/?page=2"}),
        FakeResponse(json_data={"count": 1, "results": [], "next": None}),
    ]})

    list(api.paginate(session, "captures/", params={"event": 5}))
    assert session.calls[0][2]["params"]["event"] == 5
    assert session.calls[1][2]["params"] is None


def test_an_export_is_requested_polled_and_downloaded(monkeypatch, tmp_path):
    """
    The three-step dance the API requires, with the polling loop shortened --
    a real export of a large project takes minutes, which is why the timeout is
    generous and why a timed-out poll doesn't cancel it.
    """
    monkeypatch.setattr(api.time, "sleep", lambda seconds: None)
    session = FakeSession({
        "/exports/12/": [
            FakeResponse(json_data={"id": 12, "job": {"progress": 0.5}}),
            FakeResponse(json_data={"id": 12, "file_url": "https://antenna/x.csv"}),
        ],
        "/exports/": FakeResponse(json_data={"id": 12}),
        "x.csv": FakeResponse(content=b"occurrence_id\n1\n"),
    })

    export_id = api.request_export(session, project=199)
    export = api.poll_export(session, export_id, interval=0)
    destination = api.download_export(session, export, tmp_path / "export.csv")

    assert export["file_url"].endswith("x.csv")
    assert destination.read_bytes() == b"occurrence_id\n1\n"


def test_an_export_that_never_becomes_ready_times_out(monkeypatch):
    """
    And the message names the export, because a timed-out poll doesn't cancel
    it -- it can be fetched later by id.
    """
    monkeypatch.setattr(api.time, "sleep", lambda seconds: None)
    session = FakeSession({"/exports/12/": (
        lambda url, kwargs: FakeResponse(json_data={"id": 12, "job": {}}))})

    with pytest.raises(TimeoutError, match="export 12 not ready"):
        api.poll_export(session, 12, interval=0, timeout=0.05)


# ---------------------------------------------------------------------------
# Credentials and configuration, read lazily
# ---------------------------------------------------------------------------


# That importing this module reads no .env is asserted in a subprocess, by
# tests/integration/test_public_api.py -- in-process it is unfalsifiable, since
# any earlier test that authenticated would have loaded the file already.


def test_the_project_id_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ANTENNA_PROJECT_ID", "199")
    assert api.project_id() == 199


def test_the_environment_beats_the_fallback(monkeypatch):
    """
    The argument is named `default` because that is what it is: a server states
    which project it is working on through its environment, and a hardcoded
    fallback in a script must not override the deployment it is running in.
    """
    monkeypatch.setenv("ANTENNA_PROJECT_ID", "199")
    assert api.project_id(42) == 199

    monkeypatch.delenv("ANTENNA_PROJECT_ID")
    monkeypatch.setattr(api, "_environment_loaded", True)
    assert api.project_id(42) == 42


def test_no_project_id_anywhere_raises_rather_than_guessing(monkeypatch):
    """Silently exporting the wrong project is worse than failing."""
    monkeypatch.delenv("ANTENNA_PROJECT_ID", raising=False)
    monkeypatch.setattr(api, "_environment_loaded", True)   # don't read .env
    with pytest.raises(RuntimeError, match="no Antenna project id"):
        api.project_id()


def test_missing_credentials_say_which_ones(monkeypatch):
    monkeypatch.setattr(api, "_environment_loaded", True)
    monkeypatch.delenv("ANTENNA_EMAIL", raising=False)
    monkeypatch.delenv("ANTENNA_PW", raising=False)

    with pytest.raises(RuntimeError, match="ANTENNA_EMAIL"):
        api.get_session()


def test_the_api_root_is_the_public_default_unless_overridden(monkeypatch):
    monkeypatch.setattr(api, "_environment_loaded", True)
    monkeypatch.delenv("ANTENNA_BASE", raising=False)
    assert api.base() == api.DEFAULT_BASE

    monkeypatch.setenv("ANTENNA_BASE", "https://staging.example/api/v2")
    assert api.base() == "https://staging.example/api/v2"


def test_a_blank_setting_means_unset(monkeypatch):
    """
    .env.example ships `ANTENNA_BASE=` with nothing after it, which sets the
    variable to the empty string -- and a blank line in a config file means "I
    did not set this", not "the API lives at ''".
    """
    monkeypatch.setattr(api, "_environment_loaded", True)
    monkeypatch.setenv("ANTENNA_BASE", "")
    assert api.base() == api.DEFAULT_BASE


# ---------------------------------------------------------------------------
# Scale, with an injected session
# ---------------------------------------------------------------------------


def test_a_sheet_is_fetched_through_the_session_it_is_given(draw_target_sheet):
    """
    The last bare requests.get in the package now takes a session, which is
    what makes the whole scale pass reachable without credentials -- and lets
    one session cover a scale pass and a download pass.
    """
    import cv2

    sheet, _template, _expected = draw_target_sheet()
    encoded = cv2.imencode(".jpg", sheet)[1].tobytes()
    session = FakeSession({"sheet.jpg": FakeResponse(content=encoded)})

    image = antenna_scale._download_sheet("https://antenna/sheet.jpg",
                                          session=session)
    assert image.shape == sheet.shape
    assert session.urls() == ["https://antenna/sheet.jpg"]


def test_an_undecodable_sheet_raises(draw_target_sheet):
    session = FakeSession({"sheet.jpg": FakeResponse(content=b"<html>nope</html>")})
    with pytest.raises(ValueError, match="could not decode"):
        antenna_scale._download_sheet("https://antenna/sheet.jpg", session=session)


def test_the_template_lives_in_the_project_that_uses_it(metadata_project):
    """
    A target is a property of the rig, and the rig belongs to the project -- so
    the template is a file in the project rather than a package asset.
    """
    from critterframe.project import paths

    template = antenna_scale.template_path(metadata_project)
    assert template.parent == paths.project_dir(metadata_project)
    assert template.suffix == ".png"


def test_a_missing_template_says_where_to_put_one(metadata_project):
    with pytest.raises(FileNotFoundError):
        antenna_scale.load_template(metadata_project)
