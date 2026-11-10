"""Assign: which item goes in which slot, in order or by an optimal match."""

import numpy as np

from .base import Step

MAX_MATCH = 3000


class Assignment(Step):
    """A step from an arrangement and a layout's slots to placements.

    Args:
        name: As in `Step`.
        function: Called as `function(arrangement, placed, **parameters)`, returning
            `{source: item}`.
        parameters: As in `Step`.
    """

    kind = "assign"

    def __init__(self, name, function, parameters=None):
        super().__init__(name, parameters)
        self.function = function

    def __call__(self, arrangement, placed):
        """Return the placed slots, one row per slot that received an item.

        Returns:
            The slots frame with `slot` (its row in the layout's slots) and `item` filled in, and
            `size` multiplied by the item's scale, in slot order.
        """
        slots = placed.slots.reset_index(drop=True)
        slots.insert(0, "slot", np.arange(len(slots)))
        if len(slots) and (slots["item"] >= 0).all():
            chosen = slots.copy()
        else:
            items = self.function(arrangement, placed, **self.parameters)
            chosen = slots[slots["source"].isin(list(items))].copy()
            chosen["item"] = [items[source] for source in chosen["source"]]
        chosen["item"] = chosen["item"].astype(int)
        chosen["size"] = chosen["size"].to_numpy(float) * arrangement.scales()[chosen["item"].to_numpy()]
        return chosen.reset_index(drop=True)


def _sources(placed):
    """Return each distinct source in the order its first slot appears."""
    return list(dict.fromkeys(placed.slots["source"].tolist()))


def sequence():
    """Assign: the first item to the first slot, the second to the second, and so on.

    Slots sharing a source all receive its item; items beyond the last slot are left out.
    """
    return Assignment("sequence", _sequence)


def _sequence(arrangement, placed):
    return {source: item for item, source in zip(range(len(arrangement)), _sources(placed))}


def match(on="position", reuse=False, max_items=MAX_MATCH):
    """Assign: items to slots by the least total distance, each item used once.

    The cost matrix is items by slots, so either may be at most `max_items`.

    Args:
        on: `"position"` matches each item's target position to a slot's (`scatter`);
            `"values"` matches its `match_on` features to a slot's (`mosaic`).
        reuse: Fill every slot with its nearest item, using an item as often as it is nearest.
        max_items: The most items or slots it accepts.
    """
    if on not in ("position", "values"):
        raise ValueError(f"on must be 'position' or 'values', got {on!r}")
    return Assignment("match", _match, {"on": on, "reuse": bool(reuse), "max_items": int(max_items)})


def _match(arrangement, placed, on, reuse, max_items):
    from scipy.optimize import linear_sum_assignment

    sources = _sources(placed)
    # drop_duplicates keeps each source's first slot, in slot order, as `sources` lists them.
    first = placed.slots.reset_index(drop=True).drop_duplicates("source")
    if on == "position":
        if placed.targets is None:
            raise ValueError(f"match(on='position') needs a layout with target positions, not {placed!r}")
        items, slots = np.asarray(placed.targets, float), first[["x", "y"]].to_numpy(float)
    else:
        if arrangement.values is None or placed.slot_values is None:
            raise ValueError(
                "match(on='values') needs item values from match_on() and a layout whose slots carry values"
            )
        items = np.asarray(arrangement.values, float)
        slots = np.asarray(placed.slot_values, float)[first.index]
    if items.shape[1] != slots.shape[1]:
        raise ValueError(f"items have {items.shape[1]} values and slots {slots.shape[1]}; they must match")
    if max(len(items), len(slots)) > max_items:
        raise ValueError(
            f"match compares {len(items)} items with {len(slots)} slots, more than max_items={max_items} "
            "-- sample fewer items, use a coarser layout, or raise max_items"
        )

    cost = np.sqrt(((items[:, None, :] - slots[None, :, :]) ** 2).sum(axis=2))
    if reuse:
        return {source: int(np.argmin(cost[:, index])) for index, source in enumerate(sources)}
    rows, columns = linear_sum_assignment(cost)
    return {sources[column]: int(row) for row, column in zip(rows, columns)}
