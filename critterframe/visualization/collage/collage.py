"""The collage driver: render_collage, stored values and cutouts arranged into one image."""

import difflib
import logging
import struct
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd

from ...core import drivers
from ...core.recipes import DEFAULT_PART, hash_spec
from ...export import _recorded_filters, manifest_columns, manifest_runs
from ...project import paths
from ...records.masks import current_derivation_hashes
from ...records.occurrences import ID_COL, ids_record
from ...selection import subsets as subset_selection
from ...storage.jsonfiles import read_json, write_json
from ...transforms.crop import crop_to_mask
from ...wide import filtered_wide
from .. import products
from . import compose
from .arrange import PART_COL, Arrangement, Arranger
from .compose import decoration_specs

logger = logging.getLogger(__name__)

# Every collage renders its cutouts under one name, so collages sharing a transform chain share
# one folder of cutouts.
CUTOUT_RENDER = "cutouts"
PLACEMENTS_SUFFIX = ".placements.csv"
PART_PLACEHOLDER = "{part}"


def default_transforms():
    """Return the cutout chain used when none is given: the frame cropped close to the mask."""
    return [crop_to_mask(pad=0.02)]


def _resolved(column, part):
    """Return a column name with `{part}` replaced by the item's part."""
    return column.replace(PART_PLACEHOLDER, part)


def _require_columns(names, available):
    """Raise naming the closest matches if any column is missing from the frame."""
    missing = sorted(set(names) - set(available))
    if missing:
        hints = {name: difflib.get_close_matches(name, list(available), n=3) for name in missing}
        raise KeyError(
            f"no column(s) {missing} to arrange by. Closest: {hints}. A metric column is "
            "'<run>__<part>__<metric>[__<key>]'; run_metrics() it first, or name an occurrence column"
        )


def _image_size(path):
    """Return an image file's `(width, height)`, from the header alone for a PNG."""
    with open(path, "rb") as handle:
        head = handle.read(24)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", head[16:24])
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return image.shape[1], image.shape[0]


def _read_cutout(path):
    """Return a cutout file as BGRA."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not read cutout {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        image = np.dstack([image, np.full(image.shape[:2], 255, np.uint8)])
    return image


def _render_cutouts(project_path, items, target_parts, transforms, reference, derivations):
    """Render every item's cutout, rendering again those whose mask changed since; return their paths.

    Returns:
        `({item: path}, render directory)` for the items that have a cutout.
    """
    common = dict(
        transforms=transforms,
        parts=target_parts if len(target_parts) > 1 else None,
        part=target_parts[0],
        reference=reference,
        extension="png",
        transparent=True,
        visualize=False,
    )
    directory = next(
        iter(products.render_segments(project_path, CUTOUT_RENDER, occurrence_ids=[], **common).values())
    )["directory"]
    qualify = len(target_parts) > 1

    def file_of(item):
        return directory / paths.product_filename(item[0], item[1] if qualify else None, "png")

    # A render skips a file that exists, so a cutout of a mask since replaced would stand in for
    # the new one: the derivation each was rendered from is kept beside the collages.
    record_path = paths.collages_dir(project_path) / f"{directory.name}.json"
    recorded = read_json(record_path, default={})
    stale = sorted(
        {
            item[0]
            for item in items
            if item in derivations
            and file_of(item).exists()
            and recorded.get("__".join(item)) != derivations[item]
        }
    )
    if stale:
        logger.info(
            "collage: %d cutout(s) were rendered from masks since replaced; rendering again", len(stale)
        )
        products.render_segments(project_path, CUTOUT_RENDER, occurrence_ids=stale, force=True, **common)
    with_mask = sorted({item[0] for item in items if item in derivations})
    products.render_segments(project_path, CUTOUT_RENDER, occurrence_ids=with_mask, **common)

    files = {item: file_of(item) for item in items if item in derivations and file_of(item).exists()}
    recorded.update({"__".join(item): derivations[item] for item in files})
    write_json(record_path, recorded)
    return files, directory


def _value_frame(project_path, run_names, filters, units, subset, occurrence_ids):
    """Return the filtered wide frame the collage's values come from, and the long rows behind it."""
    selected = filtered_wide(
        project_path,
        run_names=run_names,
        filters=filters,
        units=units,
        subset=None if occurrence_ids is not None else subset,
        drop_empty=False,
    )
    frame = selected.df.copy()
    frame[ID_COL] = frame[ID_COL].astype(str)
    if occurrence_ids is not None:
        frame = frame[frame[ID_COL].isin({str(occurrence_id) for occurrence_id in occurrence_ids})]
    return frame, selected


def _missing(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return False
    return value is None or pd.isna(value)


def render_collage(
    project_path,
    name,
    layout,
    arrange=(),
    assign=None,
    effects=(),
    decorations=(),
    transforms=None,
    part=DEFAULT_PART,
    parts=None,
    subset=None,
    occurrence_ids=None,
    filters=None,
    units=None,
    run_names=None,
    reference=False,
    cell=256,
    max_side=8000,
    background=None,
    blend="over",
    extension="png",
    force=False,
):
    """Write one image of many specimens: cutouts arranged, laid out, assigned to slots and composed.

    Every value that orders, sizes, groups, positions or matches an item is a stored metric or
    an occurrence column, read from the frame `export_metrics` starts from. An item missing a
    value an arrange step reads is left out and counted. A rerun whose placements, cutouts and
    styling are unchanged writes nothing.

    Args:
        project_path: Project to read from and write into.
        name: The collage's name, which starts its filenames.
        layout: A layout from `collage.layouts`, e.g. `grid()`.
        arrange: Arrange steps applied in order, e.g. `[sort_by(...), group_by("species")]`.
        assign: How items fill slots; the layout's default if None.
        effects: Effects applied to each pasted cutout, in order.
        decorations: Decorations drawn from the placed items, e.g. `label("species")`.
        transforms: The cutout chain; the frame cropped to the mask if None.
        part: The part each item shows.
        parts: Several parts, one item per occurrence-part.
        subset: Named subset to draw from.
        occurrence_ids: Draw from exactly these, ignoring `subset`.
        filters: Export filters an occurrence must pass, as `export_metrics` takes them.
        units: `"mm"` converts pixel columns, as in `export_metrics`.
        run_names: Metric runs to read values from; all if None.
        reference: Cut out from the reference masks.
        cell: Canvas pixels per layout unit, before fitting to `max_side`.
        max_side: The canvas's longest side at most, in pixels.
        background: BGR fill; None leaves it transparent.
        blend: `"over"`, `"mean"` or `"multiply"` (see `compose.draw`).
        extension: Image format.
        force: Write the collage again where it exists.

    Returns:
        A `core.drivers.Tally.summary` over the items, plus `path`, `collage_hash`, `placed`,
        `missing_value`, `unarranged` and `unassigned`. `path` is None where nothing was placed.
    """
    paths.require_project(project_path)
    extension = str(extension).lstrip(".").lower()
    if background is None and extension not in products.ALPHA_FORMATS:
        raise ValueError(f"a '{extension}' collage has no transparency -- pass background=(b, g, r)")
    if blend not in compose.BLENDS:
        raise ValueError(f"blend must be one of {compose.BLENDS}, got {blend!r}")

    steps = [arrange] if isinstance(arrange, Arranger) else list(arrange or ())
    effects, decorations = list(effects), list(decorations)
    transforms = default_transforms() if transforms is None else list(transforms)
    target_parts = list(parts) if parts else [part]

    # An item's own occurrence id and part need no lookup.
    own = {ID_COL, PART_COL}
    required = list(dict.fromkeys(column for step in steps for column in step.columns))
    optional = list(dict.fromkeys(column for step in decorations for column in step.columns))

    long_values, provenance = pd.DataFrame(), {}
    if set(required + optional) - own or filters:
        frame, selected = _value_frame(project_path, run_names, filters, units, subset, occurrence_ids)
        long_values, provenance = selected.long_df, selected.provenance
        ids = sorted(frame[ID_COL].unique())
    else:
        frame = None
        if occurrence_ids is None:
            ids = sorted(subset_selection.select_ids(project_path, subset=subset))
        else:
            ids = sorted({str(occurrence_id) for occurrence_id in occurrence_ids})

    items = [(occurrence_id, target_part) for occurrence_id in ids for target_part in target_parts]
    tally = drivers.Tally(attempted=len(items))

    table = pd.DataFrame({ID_COL: [item[0] for item in items], PART_COL: [item[1] for item in items]})
    used_columns = set()
    if frame is not None:
        indexed = frame.set_index(ID_COL)
        for column in required + optional:
            if column in table.columns:
                continue
            resolved = [_resolved(column, item[1]) for item in items]
            _require_columns(set(resolved), indexed.columns)
            used_columns |= set(resolved)
            table[column] = [indexed.at[item[0], name] for item, name in zip(items, resolved)]

    derivations = current_derivation_hashes(
        project_path, parts=target_parts, occurrence_ids=ids, reference=reference
    )
    files, directory = _render_cutouts(project_path, items, target_parts, transforms, reference, derivations)
    has_cutout = np.array([item in files for item in items], bool)
    tally.no_input = int((~has_cutout).sum())
    if tally.no_input:
        logger.warning(
            "collage '%s': %d item(s) with no cutout (no mask yet, or it failed to render)",
            name,
            tally.no_input,
        )

    complete = np.ones(len(items), bool)
    for column in required:
        complete &= ~table[column].map(_missing).to_numpy(bool)
    missing_value = int((has_cutout & ~complete).sum())
    if missing_value:
        logger.warning("collage '%s': %d item(s) left out for a missing value", name, missing_value)
    table = table[has_cutout & complete].reset_index(drop=True)

    columns = manifest_columns(provenance, sorted(used_columns))
    arrangement = Arrangement.of(table)
    for step in steps:
        arrangement = step(arrangement, table)
    if arrangement.scale_reference is not None and arrangement.scale_reference.get("unit") is None:
        column = _resolved(arrangement.scale_reference["column"], target_parts[0])
        arrangement.scale_reference["unit"] = (columns.get(column) or {}).get("unit")
    unarranged = len(table) - len(arrangement)

    placed = layout.place(arrangement)
    assignment = assign or placed.default_assign
    placements = assignment(arrangement, placed)
    placements.insert(0, ID_COL, [arrangement.keys[item][0] for item in placements["item"]])
    placements.insert(1, PART_COL, [arrangement.keys[item][1] for item in placements["item"]])
    unassigned = len(arrangement) - placements["item"].nunique()
    if unassigned:
        logger.warning("collage '%s': %d item(s) got no slot in the layout", name, unassigned)

    if placements.empty:
        logger.warning("collage '%s': nothing to place, so nothing is written", name)
        return tally.summary(
            path=None,
            collage_hash=None,
            placed=0,
            missing_value=missing_value,
            unarranged=unarranged,
            unassigned=unassigned,
        )

    drawn = list(placed.decorations)
    for decoration in decorations:
        drawn += decoration.resolve(placements, table, arrangement)

    sizes = [_image_size(files[(row[ID_COL], row[PART_COL])]) for _, row in placements.iterrows()]
    geometry = compose.plan(placements, sizes, drawn, cell, max_side)

    placed_keys = sorted({(row[ID_COL], row[PART_COL]) for _, row in placements.iterrows()})
    collage_hash = hash_spec(
        {
            "cutouts": directory.name,
            "masks": hash_spec(sorted(derivations[key] for key in placed_keys)),
            "geometry": geometry.spec(),
            "effects": [effect.spec() for effect in effects],
            "canvas": {
                "background": None if background is None else list(background),
                "blend": blend,
                "format": extension,
            },
        }
    )
    path = paths.collage_path(project_path, name, collage_hash, f".{extension}")
    sidecar = paths.collage_path(project_path, name, collage_hash, paths.COLLAGE_SIDECAR_SUFFIX)
    extra = {
        "path": path,
        "collage_hash": collage_hash,
        "placed": len(placed_keys),
        "missing_value": missing_value,
        "unarranged": unarranged,
        "unassigned": unassigned,
    }
    if path.exists() and sidecar.exists() and not force:
        logger.info("collage '%s' is unchanged -> %s", name, path)
        tally.skipped = len(placed_keys)
        return tally.summary(**extra)

    canvas = compose.draw(
        geometry, lambda row: _read_cutout(files[(row[ID_COL], row[PART_COL])]), effects, blend, background
    )
    if background is not None or extension not in products.ALPHA_FORMATS:
        canvas = canvas[:, :, :3]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"could not write {path}")
    tally.processed = len(placed_keys)

    pieces = geometry.pieces
    pd.DataFrame(
        {
            ID_COL: pieces[ID_COL],
            PART_COL: pieces[PART_COL],
            "slot": pieces["slot"],
            "group": pieces["group"],
            "x": pieces["cx"].round(1),
            "y": pieces["cy"].round(1),
            "scale": pieces["scale"].round(5),
            "angle": pieces["angle"].round(2),
            "flip": pieces["flip"],
            "alpha": pieces["alpha"],
            "z": np.arange(len(pieces)),
        }
    ).to_csv(paths.collage_path(project_path, name, collage_hash, PLACEMENTS_SUFFIX), index=False)

    # The long rows behind the columns read, for the placed occurrences only.
    used_rows = long_values
    if not long_values.empty:
        triples = {(info["run_name"], info["part"], info["metric_name"]) for info in columns.values()}
        keys = list(zip(long_values["run_name"], long_values["part"], long_values["metric_name"]))
        used_rows = long_values[
            pd.Series([key in triples for key in keys], index=long_values.index)
            & long_values[ID_COL].astype(str).isin({key[0] for key in placed_keys})
        ]

    record = {
        "name": name,
        "collage_hash": collage_hash,
        "occurrences": ids_record(sorted({key[0] for key in placed_keys})),
        "counts": {
            "requested": len(items),
            "placed": len(placed_keys),
            "pasted": len(pieces),
            "no_cutout": tally.no_input,
            "missing_value": missing_value,
            "unarranged": unarranged,
            "unassigned": unassigned,
        },
        "steps": {
            "arrange": [step.spec() for step in steps],
            "layout": layout.spec(),
            "assign": assignment.spec(),
            "effects": [effect.spec() for effect in effects],
            "decorations": [decoration.spec() for decoration in decorations],
            "drawn": decoration_specs(drawn),
        },
        "cutouts": {
            "directory": paths.relative_to_project(project_path, directory),
            "transforms": [operation.spec() for operation in transforms],
            "masks": "reference" if reference else "canonical",
            "parts": target_parts,
            "derivations": sorted({derivations[key] for key in placed_keys}),
        },
        "selection": {
            "subset": subset if occurrence_ids is None else None,
            "occurrence_ids": None if occurrence_ids is None else ids_record(ids),
            "filters": _recorded_filters(filters),
            "units": units,
            "run_names": None if run_names is None else list(run_names),
        },
        "columns": columns,
        "runs": manifest_runs(project_path, used_rows),
        "canvas": {
            "width": geometry.width,
            "height": geometry.height,
            "px_per_unit": round(geometry.px_per_unit, 4),
            "cell": cell,
            "max_side": max_side,
            "background": None if background is None else list(background),
            "blend": blend,
            "format": extension,
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(sidecar, record)
    logger.info("collage '%s': %d item(s) -> %s", name, len(placed_keys), path)
    return tally.summary(**extra)
