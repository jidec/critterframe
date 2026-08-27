"""
Shared fixtures.

Two ideas run through everything here.

**Templates are built once; tests get copies.** Building a project means ingest,
a segmentation run, and a seven-metric run over eight specimens. Doing that per
test would cost minutes across the suite. Doing it once and sharing it would be
worse than slow -- it would be wrong, because tests mutate projects (a run
upserts masks, appends metric rows, writes a grid), and the tests whose whole
subject is "what has already been done here" would start depending on each
other's order. So: session-scoped templates that nothing ever writes into, and a
function-scoped `copytree` per test.

**Fixtures yield paths and data, never open handles.** No fixture returns a live
`ImageStore` or a sqlite connection. LMDB holds a writer lock on its
environment, and the package's own code opens and closes stores as it needs
them; a long-lived handle from a fixture would contend with the code under test
for no benefit. The one test file that needs a store opens it itself, on its own
tmp_path, inside a `with`.
"""

import os
import shutil

import pytest

import critterframe as cf
from critterframe.storage import imagestore

from helpers import synthetic
from helpers.compare import strip_volatile as _strip_volatile
from helpers.models import ThresholdModel
from helpers.stubs import FakeSession

# Metrics measured on the template project. Deliberately several kinds -- two
# lengths, an area, a colour, three quality scores -- so tests of the wide
# export, of unit conversion, and of staleness all have something of the right
# shape to work on without each building its own run.
TEMPLATE_METRICS = [
    "body_length", "max_width", "area_px",
    "mean_lightness", "blur_variance", "bilateral_asymmetry", "edge_fraction",
]

SPECIMEN_COUNT = 8


# ---------------------------------------------------------------------------
# Session guards
# ---------------------------------------------------------------------------


# Big enough for eight drawn specimens with room to spare, and small enough
# that a few hundred copied projects don't fill a disk. See _small_image_store.
TEST_MAP_SIZE = 4 * 1024 ** 2


@pytest.fixture(scope="session", autouse=True)
def _small_image_store():
    """
    Open every image store with a 4 MiB map instead of the 5 GiB default.

    On Windows LMDB allocates the full map up front, and `shutil.copytree` does
    not preserve sparseness on the platforms where it would be sparse -- so
    every one of the ~200 project fixtures in this suite costs a full map on
    disk. At the default that is a terabyte; at 64 MiB it was still 12 GB per
    run, which filled the drive. Patching the module constant reaches every
    caller in the package, none of which passes a map_size.

    An explicit `pytest.MonkeyPatch` because the built-in `monkeypatch` fixture
    is function-scoped and cannot be requested from a session-scoped one.
    """
    patcher = pytest.MonkeyPatch()
    patcher.setattr(imagestore, "DEFAULT_MAP_SIZE", TEST_MAP_SIZE)
    yield
    patcher.undo()


@pytest.fixture(scope="session", autouse=True)
def _no_ambient_credentials():
    """
    Remove Antenna/iNat variables from the environment for the whole session.

    A developer running the suite has a real .env beside the repo. Without this,
    a test asserting "no credentials means RuntimeError" passes on a clean
    machine and fails on theirs -- the worst kind of test, since the failure
    says nothing about the code.
    """
    patcher = pytest.MonkeyPatch()
    for name in list(os.environ):
        if name.startswith(("ANTENNA_", "INAT_")):
            patcher.delenv(name, raising=False)
    yield
    patcher.undo()


# ---------------------------------------------------------------------------
# Data builders (no I/O)
# ---------------------------------------------------------------------------


@pytest.fixture
def template_metrics():
    """The metric names recorded by the 'traits' run on the template project."""
    return list(TEMPLATE_METRICS)


@pytest.fixture
def draw_specimen():
    """Callable (index=0, legs=True) -> a drawn BGR specimen image."""
    return synthetic.draw_specimen


@pytest.fixture
def draw_target_sheet():
    """Callable () -> (sheet, template, expected_px_per_mm)."""
    return synthetic.draw_target_sheet


@pytest.fixture
def stub_model():
    """The default stand-in segmenter: threshold at 100, no erosion."""
    return ThresholdModel()


@pytest.fixture
def eroding_model():
    """
    Callable (erode) -> a segmenter that genuinely disagrees with `stub_model`
    about where the specimen ends. What a resegmentation test needs: a second
    model returning the same mask would prove nothing about staleness.
    """
    return lambda erode=2: ThresholdModel(erode=erode)


@pytest.fixture
def fake_session():
    """Callable (routes) -> FakeSession. Function-scoped: it records calls."""
    return FakeSession


@pytest.fixture
def strip_volatile():
    """Drop timestamp/id columns before comparing two frames exactly."""
    return _strip_volatile


# ---------------------------------------------------------------------------
# Project templates (session) -- built once, never written to
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _specimen_images(tmp_path_factory):
    """A directory of drawn specimen PNGs, plus their metadata frame."""
    directory = tmp_path_factory.mktemp("specimens")
    ids = synthetic.write_specimens(directory, count=SPECIMEN_COUNT)
    return directory, synthetic.specimen_metadata(ids)


@pytest.fixture(scope="session")
def _metadata_template(tmp_path_factory, _specimen_images, _small_image_store):
    """
    The smallest valid project: an occurrence table and an archived import,
    ingested from a CSV. No images, no store.
    """
    _directory, metadata = _specimen_images
    root = tmp_path_factory.mktemp("metadata_template")
    project, source = root / "project", root / "source.csv"
    metadata.to_csv(source, index=False)
    cf.ingest_occurrences(project, source)
    return project


@pytest.fixture(scope="session")
def _image_template(tmp_path_factory, _specimen_images, _small_image_store):
    """A project with eight images in the store and metadata joined on."""
    directory, metadata = _specimen_images
    project = tmp_path_factory.mktemp("image_template") / "project"
    cf.ingest_images(project, directory, metadata=metadata)
    return project


@pytest.fixture(scope="session")
def _segmented_template(tmp_path_factory, _image_template):
    """The image project with one canonical organism mask per occurrence."""
    project = tmp_path_factory.mktemp("segmented_template") / "project"
    shutil.copytree(_image_template, project)
    cf.run_segments(project, steps=[cf.segment(ThresholdModel())],
                    visualize=False)
    return project


@pytest.fixture(scope="session")
def _measured_template(tmp_path_factory, _segmented_template):
    """The segmented project with a seven-metric 'traits' run recorded."""
    project = tmp_path_factory.mktemp("measured_template") / "project"
    shutil.copytree(_segmented_template, project)
    cf.run_metrics(project, run_name="traits",
                   transforms=[cf.remove_appendages(), cf.orient()],
                   metrics=[cf.body_length(), cf.max_width(),
                            cf.mask_area(name="area_px", unit="px2"),
                            cf.mean_lightness(), cf.blur_variance(),
                            cf.bilateral_asymmetry(), cf.edge_fraction()],
                   visualize=False)
    return project


# ---------------------------------------------------------------------------
# Project copies (function) -- what tests actually use
# ---------------------------------------------------------------------------


def _copy(template, tmp_path):
    destination = tmp_path / "project"
    shutil.copytree(template, destination)
    return destination


@pytest.fixture
def metadata_project(tmp_path, _metadata_template):
    """Occurrences only: eight rows, no images, no masks."""
    return _copy(_metadata_template, tmp_path)


@pytest.fixture
def image_project(tmp_path, _image_template):
    """Occurrences and images, nothing segmented yet."""
    return _copy(_image_template, tmp_path)


@pytest.fixture
def segmented_project(tmp_path, _segmented_template):
    """One canonical organism mask per occurrence, no metrics."""
    return _copy(_segmented_template, tmp_path)


@pytest.fixture
def measured_project(tmp_path, _measured_template):
    """Masks and a full 'traits' metric run -- the state most tests want."""
    return _copy(_measured_template, tmp_path)


@pytest.fixture
def empty_project(tmp_path):
    """
    A directory that is NOT a project: no occurrence table.

    For the require_project guard, which is the difference between "this project
    has no occurrences" and "you typed the path wrong".
    """
    directory = tmp_path / "not_a_project"
    directory.mkdir()
    return directory
