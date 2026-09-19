"""
Fetching pixels, without a network.

Every network function in the package takes an injectable `session=`, which is
what makes this file possible at all: a fake session with a routing table
exercises the whole loop -- pending selection, batching, validation, failure
counting -- offline and in milliseconds.

The rule worth pinning hardest is that a stored image is never replaced from
here. Re-running a download is a resume, not a refresh, and an occurrence that
already has pixels is left alone with no parameter to change that.
"""

import cv2
import numpy as np
import pandas as pd
import pytest

import critterframe as cf
from critterframe.download import _check_decodable
from critterframe.records.occurrences import ID_COL, save_occurrences
from critterframe.storage.imagestore import ImageStore
from helpers.stubs import FakeResponse, FakeSession
from helpers.synthetic import draw_specimen


def image_bytes(index=0):
    return cv2.imencode(".png", draw_specimen(index))[1].tobytes()


@pytest.fixture
def url_project(tmp_path):
    """Four occurrences with image URLs and nothing downloaded yet."""
    save_occurrences(tmp_path, pd.DataFrame({
        ID_COL: ["a", "b", "c", "d"],
        "image_url": [f"http://example/{name}.png" for name in "abcd"],
        "device": ["boxA", "boxA", "boxB", "boxB"],
    }))
    return tmp_path


@pytest.fixture
def all_ok():
    """A session that serves a drawn specimen for any url."""
    return FakeSession({"http://example/": FakeResponse(image_bytes())})


# ---------------------------------------------------------------------------
# _check_decodable
# ---------------------------------------------------------------------------


def test_a_real_image_passes_through_unchanged():
    data = image_bytes()
    assert _check_decodable(data, "a") is data


def test_an_html_error_page_is_refused():
    """
    A URL that 200s with an error page is otherwise only discovered much later,
    by a segmentation run that fails on an image nobody can look at.
    """
    with pytest.raises(ValueError, match="isn't a decodable image"):
        _check_decodable(b"<html>404</html>", "a")


def test_an_empty_response_is_refused():
    with pytest.raises(ValueError, match="empty response"):
        _check_decodable(b"", "a")


# ---------------------------------------------------------------------------
# download_images
# ---------------------------------------------------------------------------


def test_every_pending_image_is_fetched_and_stored(url_project, all_ok):
    summary = cf.download_images(url_project, session=all_ok)

    assert summary["attempted"] == 4 and summary["processed"] == 4
    with ImageStore(url_project, readonly=True) as store:
        assert sorted(store.keys()) == ["a", "b", "c", "d"]


def test_the_bytes_are_stored_exactly_as_served(url_project, all_ok):
    """No decode, no re-encode -- the archive of the original is the original."""
    served = image_bytes()
    cf.download_images(url_project, session=all_ok)

    with ImageStore(url_project, readonly=True) as store:
        assert store.get_bytes("a") == served


def test_a_second_run_downloads_nothing(url_project, all_ok):
    """
    Re-running is a resume, not a refresh. An occurrence that already has an
    image is left alone, and there is no parameter to ask otherwise.
    """
    cf.download_images(url_project, session=all_ok)
    second = FakeSession({"http://example/": FakeResponse(image_bytes())})
    summary = cf.download_images(url_project, session=second)

    assert summary["attempted"] == 0
    assert second.calls == []


def test_an_interrupted_download_resumes(url_project):
    """
    Batches are flushed periodically rather than at the end, so an interruption
    costs at most one batch.
    """
    cf.download_images(url_project, max_new=2,
                       session=FakeSession({"http://example/":
                                            FakeResponse(image_bytes())}))
    summary = cf.download_images(
        url_project, session=FakeSession({"http://example/":
                                          FakeResponse(image_bytes())}))
    assert summary["attempted"] == 2


def test_a_failure_is_counted_and_the_rest_still_download(url_project):
    """
    One bad URL out of four must not cost the other three -- and the failure
    has to say which occurrence and which url, or it can't be chased down.
    """
    session = FakeSession({
        "http://example/b.png": FakeResponse(b"<html>gone</html>"),
        "http://example/": FakeResponse(image_bytes()),
    })
    summary = cf.download_images(url_project, session=session)

    assert (summary["processed"], summary["failed"]) == (3, 1)
    assert summary["failures"][0]["occurrence_id"] == "b"
    assert summary["failures"][0]["url"].endswith("b.png")


def test_a_failed_download_is_not_retried_until_asked(url_project):
    """
    A failure is recorded, not just logged -- a rerun against the same URL
    makes no HTTP request for it at all, until retry_failed asks anyway.
    """
    session = FakeSession({
        "http://example/b.png": FakeResponse(b"<html>gone</html>"),
        "http://example/": FakeResponse(image_bytes()),
    })
    first = cf.download_images(url_project, session=session)
    assert (first["processed"], first["failed"]) == (3, 1)

    second = FakeSession({
        "http://example/b.png": FakeResponse(b"<html>gone</html>"),
        "http://example/": FakeResponse(image_bytes()),
    })
    summary = cf.download_images(url_project, session=second)
    assert (summary["attempted"], summary["previously_failed"]) == (0, 1)
    assert second.calls == []

    retried = cf.download_images(
        url_project, retry_failed=True,
        session=FakeSession({"http://example/": FakeResponse(image_bytes())}))
    assert (retried["attempted"], retried["processed"]) == (1, 1)


def test_a_corrected_url_is_retried_without_asking(url_project):
    """
    The failure is scoped to the URL that failed -- a re-ingest that fixes it
    is automatically new work, no retry_failed needed.
    """
    cf.download_images(url_project, session=FakeSession({
        "http://example/b.png": FakeResponse(b"<html>gone</html>"),
        "http://example/": FakeResponse(image_bytes()),
    }))

    table = pd.read_parquet(url_project / "occurrences.parquet")
    table.loc[table[ID_COL] == "b", "image_url"] = "http://example/b-fixed.png"
    save_occurrences(url_project, table)

    summary = cf.download_images(
        url_project, session=FakeSession({"http://example/": FakeResponse(image_bytes())}))
    assert (summary["attempted"], summary["processed"]) == (1, 1)


def test_an_http_error_is_a_failure_not_a_crash(url_project):
    session = FakeSession({
        "http://example/c.png": FakeResponse(b"", status_code=404),
        "http://example/": FakeResponse(image_bytes()),
    })
    summary = cf.download_images(url_project, session=session)
    assert summary["failed"] == 1


def test_a_missing_url_is_skipped_not_attempted(url_project, all_ok):
    """
    A null URL means nothing to fetch. It is not a failed download, because no
    download was ever possible.
    """
    table = pd.read_parquet(url_project / "occurrences.parquet")
    table.loc[table[ID_COL] == "d", "image_url"] = None
    save_occurrences(url_project, table)

    summary = cf.download_images(url_project, session=all_ok)
    assert (summary["attempted"], summary["processed"]) == (3, 3)


def test_a_subset_narrows_what_is_fetched(url_project, all_ok):
    cf.define_subset(url_project, "boxA", column="device", values=["boxA"])
    summary = cf.download_images(url_project, subset="boxA", session=all_ok)
    assert summary["processed"] == 2


def test_max_new_caps_the_attempt(url_project, all_ok):
    """For trying a source out before committing to a collection."""
    assert cf.download_images(url_project, max_new=1,
                              session=all_ok)["processed"] == 1


def test_limit_caps_candidates_and_max_new_caps_new_work(url_project, all_ok):
    """
    The two caps sit on either side of the already-downloaded filter, which is
    the whole difference between them. `limit` narrows the population the pass
    considers -- so a limit covering only occurrences that already have an
    image downloads nothing, exactly like every other driver's `limit`.
    `max_new` caps what is left after that: "fetch two more", every time.
    """
    cf.download_images(url_project, max_new=2, session=all_ok)

    assert cf.download_images(url_project, limit=2,
                              session=all_ok)["attempted"] == 0
    assert cf.download_images(url_project, max_new=1,
                              session=all_ok)["attempted"] == 1


def test_max_workers_one_still_downloads_everything(url_project, all_ok):
    """
    Fetches run on a thread pool by default, but a source with a strict
    rate limit needs to be able to fall back to one at a time -- this pins
    that escape hatch as a real, tested option rather than an unverified
    parameter.
    """
    summary = cf.download_images(url_project, max_workers=1, session=all_ok)

    assert (summary["attempted"], summary["processed"]) == (4, 4)
    with ImageStore(url_project, readonly=True) as store:
        assert sorted(store.keys()) == ["a", "b", "c", "d"]


def test_batching_still_saves_the_remainder(url_project, all_ok):
    """
    The last partial batch has to be flushed after the loop, or the tail of
    every download would be silently lost.
    """
    summary = cf.download_images(url_project, batch_size=3, session=all_ok)
    assert summary["processed"] == 4


def test_a_project_without_urls_says_which_function_to_use(url_project, all_ok):
    """
    A URL column is optional -- a project whose images came from a local folder
    has none -- so its absence is a wrong-function mistake, not a broken table.
    """
    save_occurrences(url_project, pd.DataFrame({ID_COL: ["a"]}))
    with pytest.raises(KeyError, match="ingest_images"):
        cf.download_images(url_project, session=all_ok)


def test_a_differently_named_url_column_can_be_named(url_project, all_ok):
    table = pd.read_parquet(url_project / "occurrences.parquet")
    save_occurrences(url_project, table.rename(columns={"image_url": "photo"}))

    assert cf.download_images(url_project, url_col="photo",
                              session=all_ok)["processed"] == 4


def test_downloading_into_a_directory_that_is_not_a_project_raises(empty_project,
                                                                   all_ok):
    with pytest.raises(FileNotFoundError, match="isn't a CritterFrame project"):
        cf.download_images(empty_project, session=all_ok)


def test_a_16_bit_image_survives_the_round_trip(url_project):
    """
    Whatever format the server serves is what gets stored -- the validation
    decode on the way past uses IMREAD_UNCHANGED precisely so it neither
    rejects nor flattens one.
    """
    deep = (np.arange(400, dtype=np.uint16) * 160).reshape(20, 20)
    served = cv2.imencode(".png", deep)[1].tobytes()
    cf.download_images(url_project,
                       session=FakeSession({"http://example/":
                                            FakeResponse(served)}))

    with ImageStore(url_project, readonly=True) as store:
        assert store.get_bytes("a") == served
        assert store.get("a", flags=cv2.IMREAD_UNCHANGED).dtype == np.uint16


# ---------------------------------------------------------------------------
# download_images(visualize=)
# ---------------------------------------------------------------------------


def _download_sidecar(project_path):
    import json

    from critterframe.project import paths

    [sidecar] = paths.pipeline_dir(project_path).glob("download_*.report.json")
    return json.loads(sidecar.read_text(encoding="utf-8"))


def test_a_download_leaves_a_thumbnail_grid_and_its_failures(url_project):
    session = FakeSession({
        "http://example/b.png": FakeResponse(b"<html>gone</html>"),
        "http://example/": FakeResponse(image_bytes()),
    })
    cf.download_images(url_project, session=session)

    record = _download_sidecar(url_project)
    assert record["counts"] == {"items": 4, "done": 4, "failed": 1}
    assert record["failures"][0]["item"] == "b"
    assert "b.png" in record["failures"][0]["error"]
    assert any(name.endswith(".jpg") for name in record["files"])


def test_out_of_order_downloads_still_fill_every_checkpoint(url_project, all_ok):
    """
    Concurrent fetches finish in any order, but checkpoint windows are positional --
    so a window's thumbnails must not be lost to a neighbour finishing first.
    """
    cf.download_images(url_project, session=all_ok, max_workers=4, visualize=1,
                       visualize_every=2)

    record = _download_sidecar(url_project)
    windows = sorted(name.rsplit("__", 1)[1] for name in record["files"] if "__at" in name)
    assert windows == ["at00000002.jpg", "at00000004.jpg"]


def test_nothing_pending_writes_nothing(url_project, all_ok):
    from critterframe.project import paths

    cf.download_images(url_project, session=all_ok, visualize=False)
    cf.download_images(url_project, session=all_ok)
    assert not paths.pipeline_dir(url_project).exists()
