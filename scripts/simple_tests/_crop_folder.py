"""
Shared setup for the smoke scripts that run over a folder of real crops
(scripts/simple_tests/metrics/, transforms/, segmentation/).

Not a smoke test itself -- nothing here is collected by pytest (it isn't
named *_test.py in the first place) and nothing here asserts anything. It
exists because five scripts were each independently reimplementing the same
few lines: read a folder of jpgs, segment each with the bundled point-prompt
model, and wire up a Segment ready to write full-resolution panels via
PanelFiles. Keeping that in one place means a change to Segment's
constructor, or to how panels get sunk to disk, is one edit instead of five.
"""

from pathlib import Path

import cv2

from critterframe.recipes import Segment
from critterframe.segmentation.groundedsam import sam2
from critterframe.visualization.panels import PanelFiles


def iter_crops(crop_dir, pattern="*.jpg"):
    """
    Yield (path, image) for every readable image in crop_dir, sorted by name.

    Unreadable files are skipped rather than raising -- a smoke script is for
    looking at what DOES work, not for asserting every file in the folder is
    valid.
    """
    for path in sorted(Path(crop_dir).glob(pattern)):
        image = cv2.imread(str(path))
        if image is not None:
            yield path, image


def segmented_crops(crop_dir, project_path, pattern="*.jpg"):
    """
    Yield (path, Segment, score) for every image in crop_dir, pre-segmented
    with the bundled point-prompt SAM2 model (sam2(), no detector -- these
    are already tight crops) and wired to write full-resolution panels to
    project_path/visualizations/ via PanelFiles.

    The shared setup every metric/transform smoke script needs before it can
    run its own operation over real crops and look at the result. score is
    SAM2's own confidence for the mask it found, yielded alongside the
    Segment for scripts that want to report it.
    """
    model = sam2()
    for path, image in iter_crops(crop_dir, pattern=pattern):
        mask, score, _info = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        segment = Segment(image, mask=mask, occurrence_id=path.stem,
                          project_path=project_path,
                          panel_sink=PanelFiles(project_path))
        yield path, segment, score
