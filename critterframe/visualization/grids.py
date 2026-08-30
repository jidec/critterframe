"""
Many panels as one image: image_grid, comparison_grid.

Pure layout -- no project, no I/O. Panels must arrive display-ready uint8:
nothing here will rescale a float array, since two probability maps with
different ranges would stretch to look identical.
"""

import logging

import cv2
import numpy as np

from .panels import TEXT_COLOR

logger = logging.getLogger(__name__)

DEFAULT_CELL = (240, 240)      # (height, width) of one image cell
DEFAULT_COLUMNS = 5
BACKGROUND = (24, 24, 24)      # near-black, so a dark specimen still has an edge
GRID_COLOR = (60, 60, 60)
LABEL_HEIGHT = 18
TITLE_HEIGHT = 26


def as_bgr(image):
    """
    A display-ready panel as 3-channel 8-bit BGR.

    Handles only the conversions with one obvious answer: a boolean mask is
    black and white, a grayscale image is the same image in three channels, an
    image with alpha drops it. Anything else raises rather than being guessed
    at -- see the module docstring on why rescaling a float panel here would be
    the wrong place to make that call.
    """
    image = np.asarray(image)

    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    elif image.dtype != np.uint8:
        raise TypeError(
            f"panel is {image.dtype}, not uint8 -- grids lay out display-ready "
            "images and won't rescale one for you, because how a float array "
            "should map to pixels is a question only whatever computed it can "
            "answer. Render it where it's made."
        )

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def fit_cell(image, cell=DEFAULT_CELL, background=BACKGROUND):
    """
    One image centred in a fixed-size cell, scaled to fit and letterboxed.

    Scaled DOWN only where it doesn't fit and up where it's smaller, both by the
    same factor in both axes: a grid whose cells each stretched to fill would
    make a long specimen and a round one look alike, which is exactly the
    judgement a QC grid exists to support.
    """
    image = as_bgr(image)
    height, width = cell
    source_height, source_width = image.shape[:2]
    if source_height == 0 or source_width == 0:
        return np.full((height, width, 3), background, np.uint8)

    scale = min(height / source_height, width / source_width)
    new_size = (max(1, int(round(source_width * scale))),
                max(1, int(round(source_height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
    resized = cv2.resize(image, new_size, interpolation=interpolation)

    canvas = np.full((height, width, 3), background, np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def _text_strip(text, width, height, background=BACKGROUND, color=TEXT_COLOR,
                scale=0.4, centered=False):
    """A single line of text on its own strip, for labels and titles."""
    strip = np.full((height, width, 3), background, np.uint8)
    if not text:
        return strip

    text = str(text)
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    left = max(2, (width - text_width) // 2) if centered else 3
    cv2.putText(strip, text, (left, height - 5), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)
    return strip


def _labelled_cell(image, label, cell, background):
    """One fitted cell with its caption strip underneath."""
    fitted = fit_cell(image, cell=cell, background=background)
    if label is None:
        return fitted
    return np.vstack([fitted, _text_strip(label, cell[1], LABEL_HEIGHT,
                                          background=background)])


def _with_title(grid, title, background=BACKGROUND):
    """The grid with a title bar above it, if there's a title."""
    if not title:
        return grid
    bar = _text_strip(title, grid.shape[1], TITLE_HEIGHT, background=background,
                      color=(255, 255, 255), scale=0.5)
    return np.vstack([bar, grid])


def image_grid(images, labels=None, title=None, columns=DEFAULT_COLUMNS,
               cell=DEFAULT_CELL, background=BACKGROUND):
    """
    A grid of images, row-major, each captioned.

    images  -- list of arrays. Any sizes; each is fitted into a cell.
    labels  -- caption per image (occurrence ids, usually). None for no captions.
    title   -- headline drawn across the top.
    columns -- images per row. The last row is padded with empty cells rather
               than being narrower, so the grid stays rectangular and the eye
               can track columns.
    cell    -- (height, width) of one image cell.
    """
    images = list(images)
    if not images:
        raise ValueError("image_grid needs at least one image")
    labels = list(labels) if labels is not None else [None] * len(images)
    if len(labels) != len(images):
        raise ValueError(
            f"got {len(images)} image(s) and {len(labels)} label(s) -- a "
            "grid captions each cell, so they have to correspond"
        )

    cells = [_labelled_cell(image, label, cell, background)
             for image, label in zip(images, labels)]
    blank = np.full(cells[0].shape, background, np.uint8)

    rows = []
    for start in range(0, len(cells), columns):
        row = cells[start:start + columns]
        row += [blank] * (columns - len(row))
        strip = np.hstack(row)
        rows.append(strip)
        # A rule under each row, because a caption sits under its image with a
        # letterbox above it -- without a line, a caption reads as belonging to
        # whatever is below it instead.
        rows.append(np.full((1, strip.shape[1], 3), GRID_COLOR, np.uint8))

    grid = np.vstack(rows)
    return _with_title(grid, title, background=background)


def comparison_grid(rows, column_titles=None, row_labels=None, title=None,
                    cell=DEFAULT_CELL, background=BACKGROUND):
    """
    One row per occurrence, one column per processing stage.

    rows          -- list of image lists, one list per occurrence, in stage
                     order. Rows may be ragged: a recipe that skipped a stage
                     for one occurrence leaves that cell empty rather than
                     shifting everything left, which would silently compare one
                     specimen's crop against another's rotation.
    column_titles -- stage names, drawn once across the top.
    row_labels    -- occurrence ids, drawn down the left edge.
    title         -- headline drawn across the top.

    Ragged rows are padded, so `rows` is addressed as `rows[occurrence][stage]`.
    """
    rows = [list(row) for row in rows]
    if not rows:
        raise ValueError("comparison_grid needs at least one row")

    width = max(len(row) for row in rows)
    if column_titles is not None:
        width = max(width, len(column_titles))

    cell_height, cell_width = cell
    # Sized to the longest id actually present, within reason: occurrence ids
    # range from "12" to a 40-character UUID, and a fixed gutter either wastes
    # half the grid or clips every label to the same unusable prefix.
    label_width = 0
    if row_labels is not None:
        longest = max((len(str(label)) for label in row_labels), default=0)
        label_width = int(np.clip(8 + 6 * longest, 48, 160))
    blank = np.full((cell_height, cell_width, 3), background, np.uint8)

    assembled = []
    if column_titles is not None:
        headers = [_text_strip(column_titles[index] if index < len(column_titles) else "",
                               cell_width, LABEL_HEIGHT, background=background,
                               centered=True)
                   for index in range(width)]
        header_row = np.hstack(headers)
        if label_width:
            corner = np.full((LABEL_HEIGHT, label_width, 3), background, np.uint8)
            header_row = np.hstack([corner, header_row])
        assembled.append(header_row)

    for index, row in enumerate(rows):
        cells = [fit_cell(image, cell=cell, background=background) if image is not None
                 else blank
                 for image in row]
        cells += [blank] * (width - len(cells))
        strip = np.hstack(cells)

        if label_width:
            label = np.full((cell_height, label_width, 3), background, np.uint8)
            text = str(row_labels[index]) if index < len(row_labels) else ""
            # An id too long for even the widest gutter keeps its TAIL: ids
            # sharing a prefix (specimen0001, specimen0002) are told apart by
            # their end, so a head-first truncation would label every row alike.
            fits = max(4, (label_width - 6) // 6)
            cv2.putText(label, text[-fits:], (3, cell_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1, cv2.LINE_AA)
            strip = np.hstack([label, strip])

        assembled.append(strip)
        assembled.append(np.full((1, strip.shape[1], 3), GRID_COLOR, np.uint8))

    grid = np.vstack(assembled)
    return _with_title(grid, title, background=background)
