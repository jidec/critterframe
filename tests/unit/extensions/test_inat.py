"""
The iNaturalist extension: normalizing INTO the core representation.

An extension's job is to turn a source's idea of a record into CritterFrame's,
and everything worth testing here is that translation -- done offline, against
canned API payloads, because the translation is where the bugs are and the
network is not.

The one judgement worth naming: `taxon_name` falls back to whatever an
observation IS identified to when it isn't identified to the rank asked for. An
observation determined only to genus still has a usable group for a group
metric, and dropping it would silently bias the reference population toward
well-studied taxa.
"""

import pytest

from critterframe.extensions.inat_insects import api, ingest
from helpers.stubs import FakeResponse, FakeSession


def an_observation(**overrides):
    """A canned API payload, shaped like the real thing and no deeper."""
    observation = {
        "id": 123456,
        "uri": "https://www.inaturalist.org/observations/123456",
        "photos": [{"url": "https://inaturalist/photos/1/square.jpg"}],
        "taxon": {
            "id": 47792,
            "rank": "species",
            "name": "Anax junius",
            "ancestors": [
                {"rank": "family", "name": "Aeshnidae"},
                {"rank": "genus", "name": "Anax"},
            ],
        },
        "geojson": {"coordinates": [-71.05, 42.36]},
        "observed_on": "2024-06-01",
        "place_guess": "Boston, MA",
        "quality_grade": "research",
        "license_code": "cc-by-nc",
        "user": {"login": "anobserver"},
    }
    observation.update(overrides)
    return observation


# ---------------------------------------------------------------------------
# photo_url
# ---------------------------------------------------------------------------


def test_the_requested_size_is_substituted_into_the_url():
    """
    iNaturalist gives a 75px "square" URL and expects callers to substitute the
    size they want.
    """
    assert api.photo_url(an_observation(), size="large") == \
        "https://inaturalist/photos/1/large.jpg"


def test_the_first_photo_is_the_one_taken():
    """
    One occurrence is one image, so taking several photos of one observation
    would create several occurrences of the same organism -- and the first is
    the one the observer chose as representative.
    """
    observation = an_observation(photos=[
        {"url": "https://inaturalist/photos/1/square.jpg"},
        {"url": "https://inaturalist/photos/2/square.jpg"}])
    assert "photos/1/" in api.photo_url(observation)


def test_an_observation_with_no_photo_has_no_url():
    """It can't become a CritterFrame occurrence, and saying None is how."""
    assert api.photo_url(an_observation(photos=[])) is None
    assert api.photo_url(an_observation(photos=None)) is None


# ---------------------------------------------------------------------------
# taxon_name
# ---------------------------------------------------------------------------


def test_the_identified_rank_is_returned_directly():
    assert api.taxon_name(an_observation(), rank="species") == "Anax junius"


def test_a_higher_rank_is_read_off_the_ancestry():
    assert api.taxon_name(an_observation(), rank="genus") == "Anax"
    assert api.taxon_name(an_observation(), rank="family") == "Aeshnidae"


def test_an_observation_identified_no_further_than_genus_still_has_a_group():
    """
    The fallback that keeps the reference population honest: dropping these
    would bias it toward well-studied taxa.
    """
    observation = an_observation(taxon={"rank": "genus", "name": "Anax",
                                        "ancestors": []})
    assert api.taxon_name(observation, rank="species") == "Anax"


def test_an_unidentified_observation_has_no_name():
    assert api.taxon_name(an_observation(taxon={})) is None
    assert api.taxon_name(an_observation(taxon=None)) is None


# ---------------------------------------------------------------------------
# observation_row
# ---------------------------------------------------------------------------


def test_an_observation_becomes_the_columns_a_project_keeps():
    """
    Deliberately narrow: the API returns a deep nested payload, and a
    column-per-field parquet would be unreadable. What is kept is identity,
    image, taxonomy, place, time, and license.
    """
    row = ingest.observation_row(an_observation())

    assert row["occurrence_id"] == "123456"
    assert row["taxon"] == "Anax junius"
    assert row["genus"] == "Anax"
    assert row["family"] == "Aeshnidae"
    assert row["observer"] == "anobserver"
    assert row["license_code"] == "cc-by-nc"


def test_the_id_is_a_string_like_every_other_occurrence_id():
    """
    iNaturalist numbers its observations, and the same id read back from
    parquet, LMDB, and sqlite must compare equal in all three.
    """
    assert ingest.observation_row(an_observation())["occurrence_id"] == "123456"


def test_coordinates_are_split_into_latitude_and_longitude():
    """In that order, whatever GeoJSON's own ordering is."""
    row = ingest.observation_row(an_observation())
    assert (row["latitude"], row["longitude"]) == (42.36, -71.05)


def test_an_observation_with_no_location_still_flattens():
    """
    A missing coordinate is normal -- obscured taxa have theirs withheld -- and
    it must not cost the observation its row.
    """
    row = ingest.observation_row(an_observation(geojson=None))
    assert row["latitude"] is None and row["occurrence_id"] == "123456"


def test_the_image_url_is_what_download_images_will_fetch():
    row = ingest.observation_row(an_observation())
    assert row["image_url"].startswith("https://inaturalist/photos/1/")


# ---------------------------------------------------------------------------
# search_observations, against a fake session
# ---------------------------------------------------------------------------


def paged(*pages):
    """A session that answers the observations endpoint with successive pages."""
    return FakeSession({"/observations": [FakeResponse(json_data={"results": page})
                                          for page in pages]})


def test_the_search_walks_every_page(monkeypatch):
    monkeypatch.setattr(api, "REQUEST_INTERVAL", 0)
    session = paged([an_observation(id=1), an_observation(id=2)],
                    [an_observation(id=3)],
                    [])

    found = list(api.search_observations(taxon_name="Odonata", session=session))
    assert [observation["id"] for observation in found] == [1, 2, 3]


def test_paging_asks_for_what_comes_after_the_last_id(monkeypatch):
    """
    id_above paging rather than page numbers: a result set that grows while you
    walk it would otherwise shift under you and duplicate or skip rows.
    """
    monkeypatch.setattr(api, "REQUEST_INTERVAL", 0)
    session = paged([an_observation(id=7)], [])

    list(api.search_observations(session=session))
    assert session.calls[0][2]["params"]["id_above"] == 0
    assert session.calls[1][2]["params"]["id_above"] == 7


def test_a_limit_stops_the_walk_early(monkeypatch, caplog):
    """
    Which is what makes trying a query cheap -- the generator stops rather than
    the caller filtering afterwards.
    """
    monkeypatch.setattr(api, "REQUEST_INTERVAL", 0)
    session = paged([an_observation(id=index) for index in range(1, 6)])

    with caplog.at_level("INFO"):
        found = list(api.search_observations(session=session, limit=2))
    assert len(found) == 2
    assert "stopped at limit" in caplog.text


def test_the_query_defaults_to_verified_and_licensed(monkeypatch):
    """
    An unverified identification makes a per-species group metric meaningless,
    since the groups themselves would be wrong -- and an unlicensed photo can't
    be redistributed with the dataset.
    """
    monkeypatch.setattr(api, "REQUEST_INTERVAL", 0)
    session = paged([])

    list(api.search_observations(taxon_name="Odonata", session=session))
    params = session.calls[0][2]["params"]
    assert params["quality_grade"] == "research"
    assert "photo_license" in params


def test_the_defaults_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(api, "REQUEST_INTERVAL", 0)
    session = paged([])

    list(api.search_observations(session=session, quality_grade=None,
                                 licensed=False))
    params = session.calls[0][2]["params"]
    assert "quality_grade" not in params
    assert "photo_license" not in params


def test_an_empty_result_ends_the_walk(monkeypatch):
    monkeypatch.setattr(api, "REQUEST_INTERVAL", 0)
    session = paged([])
    assert list(api.search_observations(session=session)) == []


# ---------------------------------------------------------------------------
# The training scaffold
# ---------------------------------------------------------------------------


def test_the_unwritten_training_loop_says_so_rather_than_pretending():
    """
    Deliberately unfinished, and the raise is what stops it being quietly
    stubbed into a no-op that trains nothing.
    """
    from critterframe.extensions.inat_insects.training import bioencoder

    with pytest.raises(NotImplementedError, match="isn't implemented"):
        bioencoder.train(manifest=None, output_dir=None)

    with pytest.raises(NotImplementedError, match="no loader is implemented"):
        bioencoder.load("some_checkpoint.pt")


def test_an_embedding_model_must_be_able_to_encode():
    """
    The whole contract: whether it is a BioEncoder checkpoint, a fine-tuned timm
    backbone, or something else entirely doesn't matter downstream.
    """
    from critterframe.extensions.inat_insects.metrics.bioencoder import (
        BioEncoderModel,
    )

    with pytest.raises(TypeError, match="encode"):
        BioEncoderModel(object(), checkpoint="weights.pt")


def test_an_embedding_is_identified_by_its_checkpoint():
    """
    Two embedding sets from different checkpoints aren't comparable at all --
    not even approximately, since the spaces are unrelated -- so they must never
    be mistaken for equivalent work.
    """
    from critterframe.extensions.inat_insects.metrics.bioencoder import (
        BioEncoderModel,
    )

    class Encoder:
        def encode(self, images):
            raise NotImplementedError

    first = BioEncoderModel(Encoder(), checkpoint="a.pt")
    second = BioEncoderModel(Encoder(), checkpoint="b.pt")
    assert first.identity() != second.identity()
    assert first.identity()["checkpoint"] == "a.pt"
