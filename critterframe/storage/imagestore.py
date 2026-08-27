"""
LMDB blob storage for per-occurrence images.

Large image blobs read randomly by occurrence id
This is kinda LMDB's sweet spot and better than millions of loose files.

THE STORE HOLDS THE EXACT ENCODED BYTES OF THE ANALYSIS IMAGE. Whatever arrived
-- a JPEG off a URL, a 16-bit TIFF, a PNG with an alpha channel -- is what comes
back out, byte for byte. The store never decodes, re-encodes, or converts on the
way in.

That's load-bearing for a scientific package. Decoding and re-encoding would
silently recompress JPEGs a second time (generational loss on exactly the fine
detail a measurement depends on), turn lossless PNG/TIFF into lossy JPEG,
flatten 16-bit data to 8, drop alpha, and discard EXIF. None of that is visible
in the resulting array, and all of it is unrecoverable. A project's images are
its raw observations.

Reading is where a normalization happens instead, and it's reversible:

  get()       decodes to 8-bit 3-channel BGR, which is what every transform,
              metric, and model here is written against. Uniform and
              predictable: a grayscale image comes back 3-channel, a 16-bit one
              comes back 8-bit.
  get_bytes() returns the stored bytes untouched, for anything needing full
              fidelity -- 16-bit intensities, alpha, EXIF, re-exporting the
              original file, checksums.

So the lossy step sits at the point of USE, where a caller can see it and opt
out, rather than at the point of STORAGE, where it would be permanent.

A project has ONE image store (project_path/images.lmdb): the original
analysis image, one per occurrence. Masks are not stored here: they go in
masks.parquet as RLE, in the coordinates of the original image (see
records.masks) and segments are never persisted at all, since a segment is
just an image plus its current mask.
"""

import logging

import cv2
import lmdb
import numpy as np

from ..project import paths

logger = logging.getLogger(__name__)

# Max size the memory-mapped file may grow to, in bytes. Hitting the limit
# mid-write raises MapFullError, so it wants headroom -- but not unlimited
# headroom: on Linux and macOS the file is sparse and an oversized map costs no
# real disk, while on WINDOWS LMDB allocates the full size up front, so a map
# sized for a project you might one day have will consume that much disk today.
# 5GB holds tens of thousands of typical analysis images; raise it explicitly
# via ImageStore(map_size=...) for a project that needs more.
DEFAULT_MAP_SIZE = 5 * 1024 ** 3

# How get() decodes for the processing pipeline: always 8-bit, always 3-channel
# BGR. Storage is faithful; this is the normalized working view, and
# get_bytes() is the way past it.
WORKING_FLAGS = cv2.IMREAD_COLOR


class ImageStore:
    """
    Keyed image blob store using LMDB. Key = occurrence id; value = the image's
    exact encoded bytes.

    project_path -- project whose images.lmdb to open.
    map_size     -- maximum size the environment may grow to, in bytes.
                    None (the default) reads DEFAULT_MAP_SIZE at CALL time
                    rather than binding it as a default argument at def time --
                    which is what lets the module constant be overridden at all,
                    since every caller inside the package opens a store without
                    passing one.
    readonly     -- open without a write lock; safe and faster for readers,
                    and lets several readers run at once.
    """

    def __init__(self, project_path, map_size=None, readonly=False):
        self.project_path = project_path
        self.path = paths.images_path(project_path)
        self.path.mkdir(parents=True, exist_ok=True)
        map_size = DEFAULT_MAP_SIZE if map_size is None else map_size
        self.readonly = readonly
        # subdir=True lets LMDB manage a directory rather than a single file;
        # lock=False for readonly is safe and avoids lock contention on reads.
        # py-lmdb wants a str, not a Path.
        self.env = lmdb.open(str(self.path), map_size=map_size,
                             readonly=readonly, lock=not readonly)
        logger.info("opened image store at %s (readonly=%s)", self.path, readonly)

    def _key(self, occurrence_id):
        return str(occurrence_id).encode("utf-8")

    def put(self, occurrence_id, data):
        """
        Store one image's encoded bytes under occurrence_id, exactly as given.

        data -- the image file's bytes: an HTTP response body, or a file read
                in binary mode. NOT a decoded array -- this store deliberately
                has no way to write one, because encoding an array here is the
                lossy step the module docstring exists to prevent.
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

        with self.env.begin(write=True) as txn:
            txn.put(self._key(occurrence_id), bytes(data))

    def put_many(self, items):
        """
        Store many (occurrence_id, bytes) pairs in one transaction -- far
        faster than individual puts for batches.
        """
        with self.env.begin(write=True) as txn:
            for occurrence_id, data in items:
                if not data:
                    logger.warning("empty image bytes for %s, skipping", occurrence_id)
                    continue
                txn.put(self._key(occurrence_id), bytes(data))

    def get_bytes(self, occurrence_id):
        """
        The stored bytes, exactly as written, or None if absent.

        The full-fidelity read: use it for anything the 8-bit BGR working view
        would destroy -- 16-bit intensities, alpha, EXIF, or writing the
        original file back out.
        """
        with self.env.begin() as txn:
            raw = txn.get(self._key(occurrence_id))
        return None if raw is None else bytes(raw)

    def get(self, occurrence_id, flags=WORKING_FLAGS):
        """
        Fetch and decode one image into the pipeline's working representation
        (8-bit 3-channel BGR), or None if absent.

        flags -- override the decode. cv2.IMREAD_UNCHANGED preserves bit depth
                 and alpha, but transforms and metrics expect 8-bit BGR, so
                 pass it only when the caller handles what comes back.
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
        with self.env.begin() as txn:
            return txn.get(self._key(occurrence_id)) is not None

    def keys(self):
        """
        All occurrence ids in the store (as strings). A full scan -- use for
        the pending check, not in hot loops.
        """
        with self.env.begin() as txn:
            return [k.decode("utf-8") for k in txn.cursor().iternext(values=False)]

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def pending_ids(all_ids, store):
    """Ids in all_ids that aren't yet in the image store."""
    done = set(store.keys())
    return [i for i in all_ids if str(i) not in done]
