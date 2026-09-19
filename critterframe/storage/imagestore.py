"""
The LMDB image store, better than directories for millions of images.

One store per project, holding the exact encoded bytes of one analysis image per
occurrence. Nothing is decoded or re-encoded on the way in; `get()` decodes to
8-bit BGR for processing, `get_bytes()` returns the original bytes.
"""

import logging

import cv2
import lmdb
import numpy as np

from ..project import paths

logger = logging.getLogger(__name__)

# Starting size of the memory-mapped file, in bytes. Hitting the limit
# mid-write raises MapFullError, which put()/put_many() catch and grow past
# (see MAP_SIZE_INCREMENT) -- so this no longer has to be sized for a project's
# eventual size, only its opening one. It still isn't free to start large: on
# Linux and macOS the file is sparse and an oversized map costs no real disk,
# but on WINDOWS LMDB allocates the full size up front, so a map opened bigger
# than needed burns that much disk immediately. 5GB holds tens of thousands of
# typical analysis images; raise it explicitly via ImageStore(map_size=...)
# for a project that starts bigger than that.
DEFAULT_MAP_SIZE = 5 * 1024 ** 3

# How far the map grows, in bytes, each time a write overflows it. Fixed
# rather than multiplicative (doubling): the same Windows up-front allocation
# above means doubling an already-large map would burn that much extra disk in
# one jump. Overridable via ImageStore(map_size_increment=...).
MAP_SIZE_INCREMENT = DEFAULT_MAP_SIZE

# Growth attempts allowed within one put()/put_many() call before giving up
# and letting MapFullError propagate. Bounds a write that can never fit --
# bigger than any reasonable number of increments -- rather than growing
# forever.
MAX_GROWTHS_PER_CALL = 10

# How get() decodes for the processing pipeline: always 8-bit, always 3-channel
# BGR. Storage is faithful; this is the normalized working view, and
# get_bytes() is the way past it.
WORKING_FLAGS = cv2.IMREAD_COLOR


class ImageStore:
    """
    Keyed image blob store using LMDB. Key = occurrence id; value = the image's
    exact encoded bytes.

    - `project_path` -- project whose images.lmdb to open.
    - `map_size` -- starting size of the environment, in bytes. None reads
      DEFAULT_MAP_SIZE at call time, so the module constant stays overridable.
      Grows automatically past this if a write overflows it.
    - `map_size_increment` -- how far map_size grows, in bytes, each time a
      write overflows it. None reads MAP_SIZE_INCREMENT at call time.
    - `readonly` -- open without a write lock; lets several readers run at
      once.
    """

    def __init__(self, project_path, map_size=None, map_size_increment=None,
                 readonly=False):
        self.project_path = project_path
        self.path = paths.images_path(project_path)
        self.path.mkdir(parents=True, exist_ok=True)
        map_size = DEFAULT_MAP_SIZE if map_size is None else map_size
        self.map_size_increment = (
            MAP_SIZE_INCREMENT if map_size_increment is None else map_size_increment
        )
        self.readonly = readonly
        # subdir=True lets LMDB manage a directory rather than a single file;
        # lock=False for readonly is safe and avoids lock contention on reads.
        # py-lmdb wants a str, not a Path.
        self.env = lmdb.open(str(self.path), map_size=map_size,
                             readonly=readonly, lock=not readonly)
        self._map_size = map_size
        logger.info("opened image store at %s (readonly=%s)", self.path, readonly)

    def _key(self, occurrence_id):
        return str(occurrence_id).encode("utf-8")

    def _grow(self):
        """Raise the map size by one increment after a write overflowed it."""
        new_size = self._map_size + self.map_size_increment
        self.env.set_mapsize(new_size)
        logger.warning(
            "image store at %s filled its map, growing %d -> %d bytes",
            self.path, self._map_size, new_size,
        )
        self._map_size = new_size

    def _resync_map_size(self):
        """
        Pick up a map size grown by another process's ImageStore on the same
        path. set_mapsize(0) is LMDB's signal to reread the current size from
        the environment's meta pages rather than set a new one.
        """
        self.env.set_mapsize(0)
        self._map_size = self.env.info()["map_size"]

    def _run_write(self, fn):
        """
        Run fn(txn) in a write transaction, growing the map and retrying on
        overflow. fn must be safe to call again from scratch -- a failed
        attempt's transaction is aborted, so nothing it did persists.
        """
        for attempt in range(MAX_GROWTHS_PER_CALL + 1):
            try:
                with self.env.begin(write=True) as txn:
                    return fn(txn)
            except lmdb.MapFullError:
                if attempt == MAX_GROWTHS_PER_CALL:
                    raise
                self._grow()

    def _run_read(self, fn):
        """
        Run fn(txn) in a read transaction, resyncing and retrying once if
        another process grew the map since this environment last checked.
        """
        try:
            with self.env.begin() as txn:
                return fn(txn)
        except lmdb.MapResizedError:
            self._resync_map_size()
            with self.env.begin() as txn:
                return fn(txn)

    def put(self, occurrence_id, data):
        """
        Store one image's encoded bytes under occurrence_id, exactly as given.

        - `data` -- the image file's bytes, e.g. an HTTP response body or a
          file read in binary mode. Not a decoded array; there is no way to
          write one, because encoding it here would be lossy.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"ImageStore stores encoded image bytes, got "
                f"{type(data).__name__}. Pass the file's bytes "
                "(response.content, or path.read_bytes()) rather than a decoded "
                "array -- re-encoding an array here would recompress or "
                "downconvert the original irreversibly."
            )
        if not data:
            raise ValueError(f"empty image bytes for {occurrence_id}")

        key, value = self._key(occurrence_id), bytes(data)
        self._run_write(lambda txn: txn.put(key, value))

    def put_many(self, items):
        """Store many (occurrence_id, bytes) pairs in one transaction."""
        # Filtered and logged once here, outside the retry loop below, so a
        # write that overflows and grows the map doesn't re-log the same
        # empty-body warning once per attempt.
        pairs = []
        for occurrence_id, data in items:
            if not data:
                logger.warning("empty image bytes for %s, skipping", occurrence_id)
                continue
            pairs.append((self._key(occurrence_id), bytes(data)))

        def _write(txn):
            for key, value in pairs:
                txn.put(key, value)

        self._run_write(_write)

    def get_bytes(self, occurrence_id):
        """
        The stored bytes, exactly as written, or None if absent.

        The full-fidelity read, for anything the 8-bit BGR working view would
        destroy: 16-bit intensities, alpha, EXIF, or re-exporting the original.
        """
        key = self._key(occurrence_id)
        raw = self._run_read(lambda txn: txn.get(key))
        return None if raw is None else bytes(raw)

    def get(self, occurrence_id, flags=WORKING_FLAGS):
        """
        Fetch and decode one image as 8-bit 3-channel BGR, or None if absent.

        - `flags` -- override the decode, e.g. cv2.IMREAD_UNCHANGED to preserve
          bit depth and alpha. Transforms and metrics expect 8-bit BGR, so pass
          it only when the caller handles what comes back.
        """
        raw = self.get_bytes(occurrence_id)
        if raw is None:
            return None

        image = cv2.imdecode(np.frombuffer(raw, np.uint8), flags)
        if image is None:
            raise ValueError(
                f"stored bytes for {occurrence_id} could not be decoded as an image"
            )
        return image

    def get_many(self, occurrence_ids, flags=WORKING_FLAGS):
        """Fetch and decode several images; missing keys come back as None."""
        return [self.get(occurrence_id, flags=flags)
                for occurrence_id in occurrence_ids]

    def has(self, occurrence_id):
        """True if a blob exists for this id (cheap -- no decode)."""
        key = self._key(occurrence_id)
        return self._run_read(lambda txn: txn.get(key) is not None)

    def keys(self):
        """
        All occurrence ids in the store, as strings. A full scan, so use it for
        a pending check rather than in a hot loop.
        """
        return self._run_read(
            lambda txn: [k.decode("utf-8") for k in txn.cursor().iternext(values=False)]
        )

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
