"""
The image store: bytes in, the same bytes out.

`put()` takes encoded bytes and refuses arrays. That refusal is the whole point
of the module, and it is worth restating why, because the alternative looks
convenient: decoding and re-encoding an array would recompress JPEGs, flatten
16-bit to 8-bit, drop alpha, and discard EXIF -- invisibly, irreversibly, and to
every image in the project.

So the tests here are mostly about fidelity: what comes back is what went in,
and the 8-bit BGR working view is a convenience laid over the top rather than
what is stored.
"""

import cv2
import numpy as np
import pytest

from critterframe.project import paths
from critterframe.storage.imagestore import ImageStore


@pytest.fixture
def store(tmp_path):
    """
    A store on its own tmp_path, opened and closed per test.

    Function-scoped and never shared: LMDB holds a writer lock on its
    environment, so a fixture handing out a long-lived store would contend with
    the code under test for no benefit.
    """
    with ImageStore(tmp_path) as opened:
        yield opened


def encoded(image=None, extension=".png"):
    """An image as its encoded file bytes -- what the store actually takes."""
    if image is None:
        image = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
    return cv2.imencode(extension, image)[1].tobytes()


# ---------------------------------------------------------------------------
# What put() accepts
# ---------------------------------------------------------------------------


def test_an_array_is_refused(store):
    """
    The invariant. A TypeError naming the fix, not a silent re-encode.
    """
    with pytest.raises(TypeError, match="stores encoded image bytes"):
        store.put("a", np.zeros((4, 4, 3), np.uint8))


@pytest.mark.parametrize("wrong", ["a string of bytes", 12, None, [1, 2, 3]])
def test_anything_that_is_not_bytes_is_refused(store, wrong):
    with pytest.raises(TypeError):
        store.put("a", wrong)


@pytest.mark.parametrize("shape", [bytes, bytearray, memoryview])
def test_every_bytes_like_spelling_is_accepted(store, shape):
    """
    `response.content`, `path.read_bytes()`, and a buffer view all arrive here
    from real callers.
    """
    store.put("a", shape(encoded()))
    assert store.has("a")


def test_empty_bytes_are_refused(store):
    """
    A zero-length body is a failed download, not an image. Storing it would
    make the occurrence look done and fail at decode time instead.
    """
    with pytest.raises(ValueError, match="empty image bytes"):
        store.put("a", b"")


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------


def test_bytes_come_back_byte_for_byte(store):
    """
    Byte-exact, not merely visually identical. This is what makes the store
    safe to treat as the archive of the original.
    """
    original = encoded(extension=".jpg")
    store.put("a", original)
    assert store.get_bytes("a") == original


def test_a_jpeg_is_not_recompressed(store):
    """
    Storing and re-reading a JPEG a hundred times must not degrade it. The way
    to be sure is that the file never changes at all.
    """
    original = encoded(np.full((32, 32, 3), 120, np.uint8), extension=".jpg")
    store.put("a", original)
    for _ in range(3):
        store.put("a", store.get_bytes("a"))
    assert store.get_bytes("a") == original


def test_sixteen_bit_depth_survives_in_the_stored_bytes(store):
    """
    `get()` gives 8-bit BGR because that is what every transform and metric
    expects. The DEPTH is still there -- `get_bytes` plus an unchanged decode
    is the escape hatch, and a store that had re-encoded on write could not
    offer one.
    """
    deep = (np.arange(400, dtype=np.uint16) * 160).reshape(20, 20)
    store.put("a", encoded(deep, extension=".png"))

    working = store.get("a")
    unchanged = store.get("a", flags=cv2.IMREAD_UNCHANGED)

    assert working.dtype == np.uint8 and working.ndim == 3
    assert unchanged.dtype == np.uint16
    assert np.array_equal(unchanged, deep)


def test_alpha_survives_in_the_stored_bytes(store):
    """Same bargain as bit depth: the working view drops it, the store doesn't."""
    with_alpha = np.zeros((8, 8, 4), np.uint8)
    with_alpha[..., 3] = 128
    store.put("a", encoded(with_alpha))

    assert store.get("a").shape == (8, 8, 3)
    assert store.get("a", flags=cv2.IMREAD_UNCHANGED).shape == (8, 8, 4)


def test_the_working_view_is_always_8_bit_bgr(store):
    """
    What every transform, metric, and model in the package assumes. A grayscale
    source must arrive as three channels, not one.
    """
    store.put("a", encoded(np.full((6, 6), 200, np.uint8)))
    image = store.get("a")
    assert image.shape == (6, 6, 3)
    assert image.dtype == np.uint8


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def test_a_missing_image_is_none_not_an_error(store):
    """
    "Not downloaded yet" is the normal state of half a project mid-ingest.
    """
    assert store.get("missing") is None
    assert store.get_bytes("missing") is None
    assert store.has("missing") is False


def test_undecodable_bytes_raise_on_read(store):
    """
    An HTML error page stored as an image. It cannot be caught at write time --
    the store takes bytes without looking at them -- so it must be loud at read
    time rather than returning None, which would read as "not downloaded".
    """
    store.put("a", b"<html>404</html>")
    with pytest.raises(ValueError, match="could not be decoded"):
        store.get("a")


def test_ids_are_keyed_as_strings(store):
    """Ids are strings everywhere; 7 and "7" are the same occurrence."""
    store.put(7, encoded())
    assert store.has("7")
    assert store.get_bytes(7) == store.get_bytes("7")


def test_a_second_put_replaces_the_first(store):
    first, second = encoded(), encoded(np.zeros((4, 4, 3), np.uint8))
    store.put("a", first)
    store.put("a", second)
    assert store.get_bytes("a") == second
    assert store.keys() == ["a"]


def test_put_many_stores_a_batch(store):
    store.put_many([("a", encoded()), ("b", encoded())])
    assert sorted(store.keys()) == ["a", "b"]


def test_put_many_skips_an_empty_body_without_failing_the_batch(store, caplog):
    """
    Individual failures are logged and counted, never fatal: one bad download
    must not cost the other 4,999 images in the transaction.
    """
    with caplog.at_level("WARNING"):
        store.put_many([("a", encoded()), ("bad", b""), ("b", encoded())])
    assert sorted(store.keys()) == ["a", "b"]
    assert "empty image bytes for bad" in caplog.text


def test_get_many_reports_missing_as_none(store):
    store.put("a", encoded())
    images = store.get_many(["a", "missing"])
    assert images[0].shape == (10, 10, 3)
    assert images[1] is None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_the_store_lives_inside_its_project(tmp_path, store):
    assert store.path == paths.images_path(tmp_path)


def test_a_reader_can_open_a_store_a_writer_created(tmp_path):
    """
    readonly=True opens without a write lock, which is what lets a run read
    images while another reader does the same.
    """
    with ImageStore(tmp_path) as writer:
        writer.put("a", encoded())

    with ImageStore(tmp_path, readonly=True) as reader:
        assert reader.has("a")


def test_map_size_is_read_at_call_time(tmp_path, monkeypatch):
    """
    The default is None so DEFAULT_MAP_SIZE is looked up when a store is
    opened, not bound when the module was imported. Nothing inside the package
    passes a map_size, so that lookup is the only way the constant can be
    overridden at all -- which is what keeps a test suite from allocating 5 GiB
    per fixture on Windows.
    """
    from critterframe.storage import imagestore

    monkeypatch.setattr(imagestore, "DEFAULT_MAP_SIZE", 1024 ** 2)
    with imagestore.ImageStore(tmp_path) as small:
        assert small.env.info()["map_size"] == 1024 ** 2
