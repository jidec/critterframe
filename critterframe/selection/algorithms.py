"""Transient selections over data the caller already has: sample, shard, cap, dedupe, match."""

import logging
import random

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fixed so a project's sample is stable across runs, recipes, and sessions.
SAMPLE_SEED = 20250101


def rows_matching(df, rules):
    """Return which rows a `{column: values}` rule set picks out, as a boolean Series.

    A row matches when any rule does. A missing value never matches.

    Args:
        df: Table to test.
        rules: `{column: value}` or `{column: [values, ...]}`. Membership only.

    Raises:
        KeyError: If a rule names a column `df` doesn't have.
    """
    matched = pd.Series(False, index=df.index)

    for column, values in rules.items():
        if column not in df.columns:
            raise KeyError(f"rule column '{column}' isn't in the table (columns: {sorted(df.columns)})")
        # A bare string is one value, not an iterable of characters -- the
        # single-value form is what anyone writes first.
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set, frozenset)):
            values = [values]

        series = df[column]
        matched |= series.isin(list(values)) & series.notna()

    return matched


def require_present(occurrence_ids, **named_sets):
    """Narrow ids to the ones present in every named set.

    Logs how many ids each set drops, in the order given.

    Args:
        occurrence_ids: Ids to narrow.
        **named_sets: `{label: ids}`, e.g. `has_image=store.keys()`.

    Returns:
        A sorted list of string ids.
    """
    return _narrow(occurrence_ids, named_sets, keep_if_present=True)


def exclude_present(occurrence_ids, **named_sets):
    """Narrow ids to the ones present in none of the named sets.

    Args:
        occurrence_ids: Ids to narrow.
        **named_sets: `{label: ids}`, e.g. `downloaded=store.keys()`.

    Returns:
        A sorted list of string ids.
    """
    return _narrow(occurrence_ids, named_sets, keep_if_present=False)


def _narrow(occurrence_ids, named_sets, keep_if_present):
    """Keep the ids present in every named set, or in none, logging what each set drops."""
    remaining = {str(occurrence_id) for occurrence_id in occurrence_ids}
    verb = "require_present" if keep_if_present else "exclude_present"
    relation = "missing" if keep_if_present else "present in"

    for label, ids in named_sets.items():
        present = {str(occurrence_id) for occurrence_id in ids}
        kept = (remaining & present) if keep_if_present else (remaining - present)
        dropped = len(remaining) - len(kept)
        if dropped:
            logger.info(
                "%s: dropped %d of %d occurrence(s) %s '%s'", verb, dropped, len(remaining), relation, label
            )
        remaining = kept

    return sorted(remaining)


def sample_ids(occurrence_ids, count, seed=SAMPLE_SEED):
    """Return a stable pseudo-random sample of ids, sorted.

    The same ids, count and seed give the same sample whatever order the ids arrive in.

    Args:
        occurrence_ids: Ids to sample from.
        count: How many to take; every id is returned when there are fewer.
        seed: Sampling seed.
    """
    ids = sorted(str(occurrence_id) for occurrence_id in occurrence_ids)
    if count is None or count >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, count))


def grow_sample(candidate_ids, target_size, keep_ids=None, seed=SAMPLE_SEED):
    """Extend or trim a previous sample toward a target size, deterministically.

    Args:
        candidate_ids: The whole eligible pool, `keep_ids` included.
        target_size: Size wanted. Raising it only adds; lowering it trims `keep_ids`.
        keep_ids: Ids from a previous call, kept where still in `candidate_ids`.
        seed: Sampling seed.

    Returns:
        Sorted ids.
    """
    pool = {str(occurrence_id) for occurrence_id in candidate_ids}
    kept = pool & {str(occurrence_id) for occurrence_id in (keep_ids or [])}

    survivors = set(sample_ids(kept, min(target_size, len(kept)), seed=seed))
    remaining = target_size - len(survivors)
    if remaining > 0:
        survivors |= set(sample_ids(pool - kept, remaining, seed=seed))

    return sorted(survivors)


def sample_per_group(df, group_col, count, id_col="occurrence_id", seed=SAMPLE_SEED):
    """Return a stable sample of up to `count` ids, spread across the groups of a column.

    Groups are filled smallest first, so a rare group is taken whole before a common one
    absorbs what is left.

    Args:
        df: Table holding the ids and the group column.
        group_col: Column to stratify by. A row with a missing value is excluded.
        count: Total ids wanted, across every group.
        id_col: Column holding the ids.
        seed: Sampling seed.
    """
    if group_col not in df.columns:
        raise KeyError(f"no '{group_col}' column to group by (columns: {sorted(df.columns)})")
    if id_col not in df.columns:
        raise KeyError(f"no '{id_col}' column to read ids from (columns: {sorted(df.columns)})")
    if count <= 0:
        return []

    groups = [ids for _, ids in df[df[group_col].notna()].groupby(group_col, sort=False)[id_col]]
    groups.sort(key=len)

    picked = []
    remaining_count, remaining_groups = count, len(groups)
    for ids in groups:
        share = -(-remaining_count // remaining_groups)  # ceil division
        take = min(len(ids), share)
        picked.extend(sample_ids(ids, take, seed=seed))
        remaining_count -= take
        remaining_groups -= 1

    return sorted(picked)


# Rows of a group measured against the whole group at once: bounds the memory
# of a pairwise distance table, which is otherwise the square of the group.
MEDOID_BLOCK_ROWS = 1024


def _summed_distances(points):
    """Return each row's summed Euclidean distance to every row of an `(n, d)` array."""
    squared = (points**2).sum(axis=1)
    totals = np.empty(len(points))
    for start in range(0, len(points), MEDOID_BLOCK_ROWS):
        block = slice(start, start + MEDOID_BLOCK_ROWS)
        between = squared[block, None] + squared[None, :] - 2 * points[block] @ points.T
        totals[block] = np.sqrt(np.clip(between, 0, None)).sum(axis=1)
    return totals


def group_medoids(vectors, groups, count=1):
    """Return, per group, the `count` ids with the smallest summed distance to the rest of it.

    Distance is Euclidean, and equal sums are broken by sorted id. Time grows with the
    square of a group's size.

    Args:
        vectors: `{occurrence_id: vector}`, or a DataFrame indexed by id with one numeric
            column per dimension. A row with a missing value is dropped.
        groups: `{occurrence_id: group}` or a Series indexed by id. An id with a missing
            group is excluded.
        count: How many to take per group; a smaller group is taken whole.

    Returns:
        A sorted list of ids.
    """
    if count <= 0:
        return []

    if not isinstance(vectors, pd.DataFrame):
        vectors = pd.DataFrame.from_dict(
            {
                occurrence_id: np.atleast_1d(np.asarray(vector, dtype=float))
                for occurrence_id, vector in dict(vectors).items()
            },
            orient="index",
        )
    table = vectors.dropna()
    table.index = table.index.map(str)

    groups = groups if isinstance(groups, pd.Series) else pd.Series(groups, dtype=object)
    groups = groups.dropna()
    groups.index = groups.index.map(str)
    groups = groups[groups.index.isin(table.index)]

    picked = []
    for _group, members in groups.groupby(groups, sort=False):
        ids = sorted(members.index)
        totals = _summed_distances(table.loc[ids].to_numpy(dtype=float))
        # stable, over ids already sorted: that is the tie-break
        picked.extend(ids[index] for index in np.argsort(totals, kind="stable")[:count])
    return sorted(picked)


def worst_n(values, count, ascending=True):
    """Return the `count` ids with the lowest or highest values, worst first.

    Args:
        values: `{occurrence_id: value}` or a Series indexed by id. NaN values are dropped.
        count: How many to return.
        ascending: True when a low value is worst, False when a high one is.

    Returns:
        `[(occurrence_id, value), ...]`.
    """
    if count <= 0:
        return []

    series = values if isinstance(values, pd.Series) else pd.Series(values)
    ranked = series.dropna().sort_values(ascending=ascending).head(count)
    return [(str(occurrence_id), value) for occurrence_id, value in ranked.items()]


def cap_per_group(
    df, group_col, max_count, rule="random", seed=SAMPLE_SEED, id_col=None, keep_ids=None, prefer=None
):
    """Narrow a table to at most `max_count` rows per value of a column.

    Args:
        df: Table to cap.
        group_col: Column naming the group, e.g. `"species"`. A row with a missing value
            is never removed.
        max_count: Cap applied within each group.
        rule: Which rows of an oversized group survive: `"random"` (seeded, independent of
            row order), `"first"`, `"last"`, or a callable taking and returning the group's rows.
        seed: Seed for the `"random"` rule.
        id_col: Column holding the ids `keep_ids` refers to.
        keep_ids: Ids to keep ahead of the rest of an oversized group, e.g. what an earlier
            cap selected, so raising the cap only adds.
        prefer: `{column: values}` rule naming rows to fill a group from first. Ranks above
            `keep_ids`.

    Raises:
        KeyError: If `group_col` isn't in `df`.
    """
    if group_col not in df.columns:
        raise KeyError(f"no '{group_col}' column to group by (columns: {sorted(df.columns)})")
    if keep_ids and not id_col:
        raise ValueError("id_col is required when keep_ids is given")
    if keep_ids and id_col not in df.columns:
        raise KeyError(f"no '{id_col}' column to match keep_ids against (columns: {sorted(df.columns)})")

    keep = pd.Series(True, index=df.index)
    keep_ids = {str(occurrence_id) for occurrence_id in keep_ids} if keep_ids else set()
    preferred = rows_matching(df, prefer) if prefer else None

    for _, rows in df[df[group_col].notna()].groupby(group_col, sort=False):
        if len(rows) <= max_count:
            continue

        if callable(rule):
            survivors = rule(rows).index
        elif rule in ("random", "first", "last"):
            if keep_ids:
                is_kept = rows[id_col].astype(str).isin(keep_ids)
            else:
                is_kept = pd.Series(False, index=rows.index)
            tiers = [is_kept, ~is_kept]
            if preferred is not None:
                is_preferred = preferred.loc[rows.index]
                tiers = [is_preferred & tier for tier in tiers] + [~is_preferred & tier for tier in tiers]

            survivors = rows.index[:0]
            for tier in tiers:
                remaining = max_count - len(survivors)
                if remaining <= 0:
                    break
                candidates = rows[tier]
                survivors = survivors.union(_take(candidates, min(remaining, len(candidates)), rule, seed))
        else:
            raise ValueError(
                f"unknown cap rule {rule!r} -- use 'random', 'first', 'last', "
                "or a callable(group_df) -> group_df"
            )

        keep.loc[rows.index.difference(survivors)] = False

    dropped = int((~keep).sum())
    if dropped:
        logger.info("capped %d of %d row(s) to at most %d per '%s'", dropped, len(df), max_count, group_col)

    return df[keep].reset_index(drop=True)


def _fingerprint(df, key_cols, precision):
    """Return one string per row identifying its key columns, or NaN where any is missing."""
    valid = pd.Series(True, index=df.index)
    parts = []
    for column in key_cols:
        if column not in df.columns:
            raise KeyError(f"no '{column}' column to fingerprint on (columns: {sorted(df.columns)})")
        series = df[column]
        if column in precision:
            series = pd.to_numeric(series, errors="coerce").round(precision[column])
        valid &= series.notna()
        parts.append(series.astype(str))

    fingerprint = parts[0]
    for part in parts[1:]:
        fingerprint = fingerprint.str.cat(part, sep="\x1f")
    return fingerprint.where(valid)


def dedupe_by(
    df, key_cols, precision=None, rule="random", seed=SAMPLE_SEED, id_col=None, keep_ids=None, prefer=None
):
    """Narrow a table to one row per distinct combination of key columns.

    The match is by value and so can be wrong: two different records can share a fingerprint.

    Args:
        df: Table to deduplicate.
        key_cols: Columns that together identify one real occurrence, e.g. latitude,
            longitude and date. A row missing any of them is never removed.
        precision: `{column: ndigits}` to round a numeric key column by before matching.
        rule: Which row of a duplicate group survives, as in `cap_per_group`.
        seed: Seed for the `"random"` rule.
        id_col: Column holding the ids `keep_ids` refers to.
        keep_ids: Ids to keep ahead of the rest of a duplicate group.
        prefer: `{column: values}` rule naming rows trusted to be unique. A preferred row is
            never removed, and any other row sharing its fingerprint is.
    """
    key_cols = list(key_cols)
    working = df.copy()
    working["_dedupe_fingerprint"] = _fingerprint(df, key_cols, precision or {})
    # Carried through cap_per_group's reset_index so the source order survives.
    working["_dedupe_position"] = range(len(working))

    protected = working.iloc[0:0]
    if prefer:
        is_preferred = rows_matching(working, prefer)
        protected = working[is_preferred]
        others = working[~is_preferred]
        claimed = set(protected["_dedupe_fingerprint"].dropna())
        working = others[~others["_dedupe_fingerprint"].isin(claimed)]

    deduped = cap_per_group(
        working, "_dedupe_fingerprint", 1, rule=rule, seed=seed, id_col=id_col, keep_ids=keep_ids
    )
    deduped = pd.concat([protected, deduped]).sort_values("_dedupe_position").reset_index(drop=True)
    removed = len(df) - len(deduped)
    if removed:
        logger.info("deduplicated %d of %d row(s) sharing (%s)", removed, len(df), ", ".join(key_cols))
    return deduped.drop(columns=["_dedupe_fingerprint", "_dedupe_position"])


def _take(rows, count, rule, seed):
    """Return the index of `count` rows, by the `random`, `first` or `last` rule."""
    if count <= 0 or rows.empty:
        return rows.index[:0]
    if rule == "random":
        kept_positions = set(sample_ids(rows.index.astype(str), count, seed=seed))
        return rows.index[rows.index.astype(str).isin(kept_positions)]
    if rule == "first":
        return rows.index[:count]
    return rows.index[-count:]  # "last"


def shard_ids(occurrence_ids, index, total):
    """Return this shard's disjoint slice of the ids.

    Ids are sorted and dealt round-robin, so any worker given the same ids and `total`
    computes the same split, and shard sizes differ by at most one.

    Args:
        occurrence_ids: Ids to split.
        index: This shard's number, from 0 to `total - 1`.
        total: How many shards there are.
    """
    if total < 1:
        raise ValueError(f"total must be at least 1, got {total}")
    if not (0 <= index < total):
        raise ValueError(f"index must be in [0, {total}), got {index}")

    ids = sorted(str(occurrence_id) for occurrence_id in occurrence_ids)
    return ids[index::total]
