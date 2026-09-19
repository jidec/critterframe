"""
Choosing which occurrences something applies to.

Selection questions recur all over the package and are the same question every
time: out of these occurrences, which ones? A visualization wants a
representative handful, a trial run wants a few to test a recipe on, a training
split wants a held-out set, an annotation pass wants the ones a model is least
sure about. They differ in the RULE, not in the shape, so they belong together
rather than one per module.

Distinct from project.subsets, which is about NAMED, persisted selections that a
project carries in its definitions -- "the 2019 Peru material", "the specimens
Ana checked". These are transient: computed when asked for, never stored.

Everything here takes data and answers from it -- nothing reads a project, and
nothing else in the package is imported. That is deliberate rather than
incidental: this module is imported by ingest, by segmentation runs, and by
pipeline visualization, none of which should acquire a dependency on the
metrics or export layers just to sample or shard a list of ids.
export.occurrences_matching is the selection that DOES read stored values, and
it lives with the wide-form view it reads rather than here.
"""

import logging
import random

import pandas as pd

logger = logging.getLogger(__name__)

# Fixed so a project's sample is stable across runs, recipes, and sessions.
SAMPLE_SEED = 20250101


def rows_matching(df, rules):
    """
    Which rows a {column: values} rule set picks out, as a boolean Series.

    - `rules` -- {column: value} or {column: [values...]}. A row matches when
      ANY rule matches: these name several kinds of a thing ("debris, or
      not-Lepidoptera"), not a conjunction one row must satisfy at once.

    Membership only, deliberately -- no <=, no >, no predicate. A threshold is a
    judgement about degree that a caller will want to revise, and revisable
    judgements belong where they keep the data (export filters, subsets). Keeping
    the vocabulary too small to express one is what stops it being smuggled into
    a place that can't undo it.

    A missing value never matches: not knowing what a row is cannot be the same
    as knowing it is one of these things. A named column that isn't in df raises,
    since a typo matching nothing would read as "there was none of that here".
    """
    matched = pd.Series(False, index=df.index)

    for column, values in rules.items():
        if column not in df.columns:
            raise KeyError(
                f"rule column '{column}' isn't in the table "
                f"(columns: {sorted(df.columns)})"
            )
        # A bare string is one value, not an iterable of characters -- the
        # single-value form is what anyone writes first.
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set, frozenset)):
            values = [values]

        series = df[column]
        matched |= series.isin(list(values)) & series.notna()

    return matched


def require_present(occurrence_ids, **named_sets):
    """
    Narrow occurrence_ids to ones present in EVERY named set -- composing
    several "does this exist" checks (has an image, has a mask for a part,
    ...) into one call instead of one isin()/set-intersection per condition
    at each call site.

    - `named_sets` -- {label: ids}, e.g. `has_image=store.keys()`,
      `head_mask=occurrence_ids_with_mask(project_path, "head")`.

    Every rule must match (AND) -- these name independent preconditions, not
    alternative spellings of one fact, unlike rows_matching's ANY-of-these-
    values OR. Logs how many ids each named set drops, in the order given,
    so a caller can see which condition is the one actually narrowing things.

    Returns a sorted list of str ids present in occurrence_ids and every
    named set.
    """
    return _narrow(occurrence_ids, named_sets, keep_if_present=True)


def exclude_present(occurrence_ids, **named_sets):
    """
    The inverse of require_present: keep ids present in NONE of the named
    sets -- the "not already downloaded", "not already measured" shape the
    pending-first drivers use. Same composition, same logging.
    """
    return _narrow(occurrence_ids, named_sets, keep_if_present=False)


def _narrow(occurrence_ids, named_sets, keep_if_present):
    """Shared body of require_present/exclude_present -- see either's docstring."""
    remaining = {str(occurrence_id) for occurrence_id in occurrence_ids}
    verb = "require_present" if keep_if_present else "exclude_present"
    relation = "missing" if keep_if_present else "present in"

    for label, ids in named_sets.items():
        present = {str(occurrence_id) for occurrence_id in ids}
        kept = (remaining & present) if keep_if_present else (remaining - present)
        dropped = len(remaining) - len(kept)
        if dropped:
            logger.info("%s: dropped %d of %d occurrence(s) %s '%s'",
                       verb, dropped, len(remaining), relation, label)
        remaining = kept

    return sorted(remaining)


def sample_occurrences(occurrence_ids, count, seed=SAMPLE_SEED):
    """
    A stable pseudo-random sample of occurrence ids, in sorted order.

    Sorted input before sampling and sorted output after, from a fixed seed: the
    same ids and the same count give the same specimens every time, in the same
    order. That is the whole point rather than an implementation detail --
    a sample that reshuffled on every call would make two visualizations of two
    versions of a recipe incomparable, and you would be unable to tell a changed
    method from a changed specimen.

    Returns every id when count exceeds how many there are, rather than raising:
    asking for 25 from a project of 8 is a reasonable thing to do.
    """
    ids = sorted(str(occurrence_id) for occurrence_id in occurrence_ids)
    if count is None or count >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, count))


def grow_sample(candidate_ids, target_size, keep_ids=None, seed=SAMPLE_SEED):
    """
    Extend (or, if target_size has shrunk, trim) a previous sample toward
    target_size, deterministically.

    For growing a named subset toward a target size over several runs without
    reshuffling who is already in it -- the same guarantee cap_per_group's
    keep_ids gives a capped group, one level up: a subset growing toward a
    size rather than a group shrinking toward a cap.

    - `candidate_ids` -- the whole eligible pool, keep_ids included. An id in
      keep_ids no longer present here is dropped, the same way cap_per_group
      only prioritizes a keep_ids row still in df.
    - `target_size` -- desired size of the result. Raising it across calls only
      adds; lowering it below len(keep_ids) trims keep_ids down via
      sample_occurrences rather than an arbitrary set order.
    - `keep_ids` -- ids from a previous call to keep if still in
                     - `candidate_ids` -- typically what a named subset already
                     holds. None (or empty) is a first call, equivalent to a
                     plain sample_occurrences(candidate_ids, target_size).
    - `seed` -- passed through to sample_occurrences.

    Returns ids, sorted like sample_occurrences.
    """
    pool = {str(occurrence_id) for occurrence_id in candidate_ids}
    kept = pool & {str(occurrence_id) for occurrence_id in (keep_ids or [])}

    survivors = set(sample_occurrences(kept, min(target_size, len(kept)), seed=seed))
    remaining = target_size - len(survivors)
    if remaining > 0:
        survivors |= set(sample_occurrences(pool - kept, remaining, seed=seed))

    return sorted(survivors)


def sample_per_group(df, group_col, count, id_col="occurrence_id", seed=SAMPLE_SEED):
    """
    A stable, stratified sample of up to `count` ids, spread across group_col.

    Smallest group first: each group's fair share is ceil(what's left / groups
    not yet visited), capped at the group's own size, so a handful of common
    groups can't crowd out a rare one the way a flat sample_occurrences() draw
    over the whole table would -- a rare group is taken in full (or as close
    as its size allows) before a common group absorbs the rollover. Each
    group's own picks still go through sample_occurrences, so the sample is
    stable across runs the same way a flat one is.

    - `group_col` -- occurrence column to stratify by, e.g. "taxon". A row with
      a missing value is excluded, the same way rows_matching and cap_per_group
      treat one -- not knowing a row's group cannot be the same as knowing it
      belongs to one being sampled.
    - `count` -- total ids wanted, across every group. Returns every qualifying
      id when count exceeds how many there are, like sample_occurrences -- here
      that can mean every id in every group, if the whole table is smaller than
      count.
    - `id_col` -- occurrence id column to read and return.
    - `seed` -- passed through to sample_occurrences for each group's draw.
    """
    if group_col not in df.columns:
        raise KeyError(
            f"no '{group_col}' column to group by (columns: {sorted(df.columns)})"
        )
    if id_col not in df.columns:
        raise KeyError(
            f"no '{id_col}' column to read ids from (columns: {sorted(df.columns)})"
        )
    if count <= 0:
        return []

    groups = [ids for _, ids in
             df[df[group_col].notna()].groupby(group_col, sort=False)[id_col]]
    groups.sort(key=len)

    picked = []
    remaining_count, remaining_groups = count, len(groups)
    for ids in groups:
        share = -(-remaining_count // remaining_groups)  # ceil division
        take = min(len(ids), share)
        picked.extend(sample_occurrences(ids, take, seed=seed))
        remaining_count -= take
        remaining_groups -= 1

    return sorted(picked)


def worst_n(values, count, ascending=True):
    """
    The `count` occurrence ids with the lowest (ascending=True, the default)
    or highest values, sorted worst-first.

    - `values` -- {occurrence_id: value} or a pandas Series indexed by id.
      NaN values are dropped first -- an undefined value can't be ranked.
    - `count` -- how many to return. 0 or fewer returns [].
    - `ascending` -- True when a LOW value is worst (an IoU, a match score);
      False when a HIGH value is worst (a percent disagreement).

    Returns [(occurrence_id, value), ...], worst first -- both, not just
    ids, since every existing caller wants the value alongside it for a log
    line. Take just the ids with `[i for i, _ in worst_n(...)]` when that's
    all that's needed, e.g. to hand to `project.subsets.grow_subset`.
    """
    if count <= 0:
        return []

    series = values if isinstance(values, pd.Series) else pd.Series(values)
    ranked = series.dropna().sort_values(ascending=ascending).head(count)
    return [(str(occurrence_id), value) for occurrence_id, value in ranked.items()]


def cap_per_group(df, group_col, max_count, rule="random", seed=SAMPLE_SEED,
                  id_col=None, keep_ids=None, prefer=None):
    """
    Narrow df to at most max_count rows per distinct group_col value.

    A GBIF or iNaturalist pull is routinely dominated by a handful of common
    species with thousands of photos each, next to rare ones with a few -- this
    is how a project caps that at ingest, one group_col value at a time (see
    ingest.ingest_occurrences' group_col/max_per_group).

    - `group_col` -- occurrence column naming the group, e.g. "species". An
      unknown column raises, like rows_matching. A missing value in it is
      exempt from capping, kept untouched no matter how many rows share that
      gap -- not knowing which group a row belongs to cannot be the same as
      knowing it is one of an oversized group's extras, the same reasoning
      rows_matching applies to a missing value never matching a rule.
    - `max_count` -- cap applied independently within each group; a group at or
      under this size is returned untouched.
    - `rule` -- which rows survive an oversized group. "random" (default) is a
      stable pseudo-random choice, via sample_occurrences' seeded rule applied
      to the candidates' own row positions -- so which specimens survive
      doesn't depend on the source's row order. "first"/"last" keep that many
      rows in the candidates' own order instead. A callable(group_df) ->
      group_df picks explicitly, for the whole group at once -- keep_ids does
      not apply to it, since a callable already owns the decision.
    - `seed` -- passed through to sample_occurrences for the "random" rule.
    - `id_col`, `keep_ids` -- ids (read from id_col, matched against keep_ids)
      to prioritize keeping over the rest of an oversized group -- typically
      the occurrences a previous, smaller-or-equal cap already selected for
      this project. This is what makes raising max_per_group on a reimport ADDITIVE
      instead of a reshuffle: ids already kept stay kept (retrimmed by rule if
      the cap has since shrunk), and only the shortfall is drawn fresh from ids
      not in keep_ids. Ignored for a group at or under max_count, and for a
      callable rule.
    - `prefer` -- optional `{column: values}` rule (see rows_matching) naming
      rows to fill an oversized group from first, e.g. one source over
      another. Ranks ABOVE keep_ids: the order is preferred & kept, preferred
      & new, other & kept, other & new, so a preferred newcomer displaces an
      already-kept row that isn't. Ignored for a callable rule.

    Rows a cap removes are logged as one aggregate count, the same as
    ingest.ingest_occurrences' drop= -- the archived import is the recovery
    path for these too, not a per-row manifest.
    """
    if group_col not in df.columns:
        raise KeyError(
            f"no '{group_col}' column to group by (columns: {sorted(df.columns)})"
        )
    if keep_ids and not id_col:
        raise ValueError("id_col is required when keep_ids is given")
    if keep_ids and id_col not in df.columns:
        raise KeyError(
            f"no '{id_col}' column to match keep_ids against (columns: "
            f"{sorted(df.columns)})"
        )

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
                tiers = [is_preferred & tier for tier in tiers] + \
                        [~is_preferred & tier for tier in tiers]

            survivors = rows.index[:0]
            for tier in tiers:
                remaining = max_count - len(survivors)
                if remaining <= 0:
                    break
                candidates = rows[tier]
                survivors = survivors.union(
                    _take(candidates, min(remaining, len(candidates)), rule, seed))
        else:
            raise ValueError(
                f"unknown cap rule {rule!r} -- use 'random', 'first', 'last', "
                "or a callable(group_df) -> group_df"
            )

        keep.loc[rows.index.difference(survivors)] = False

    dropped = int((~keep).sum())
    if dropped:
        logger.info("capped %d of %d row(s) to at most %d per '%s'",
                   dropped, len(df), max_count, group_col)

    return df[keep].reset_index(drop=True)


def _fingerprint(df, key_cols, precision):
    """
    One string per row identifying its key_cols, or NaN if any of them is
    missing -- a composite version of a single group_col value, built so
    dedupe_by can hand it straight to cap_per_group.

    precision rounds a column after coercing it to numeric; a column not
    named there is matched on its own value, cast to string.
    """
    valid = pd.Series(True, index=df.index)
    parts = []
    for column in key_cols:
        if column not in df.columns:
            raise KeyError(
                f"no '{column}' column to fingerprint on (columns: "
                f"{sorted(df.columns)})"
            )
        series = df[column]
        if column in precision:
            series = pd.to_numeric(series, errors="coerce").round(precision[column])
        valid &= series.notna()
        parts.append(series.astype(str))

    fingerprint = parts[0]
    for part in parts[1:]:
        fingerprint = fingerprint.str.cat(part, sep="\x1f")
    return fingerprint.where(valid)


def dedupe_by(df, key_cols, precision=None, rule="random", seed=SAMPLE_SEED,
             id_col=None, keep_ids=None, prefer=None):
    """
    Narrow df to one row per distinct combination of key_cols.

    Distinct from cap_per_group: capping thins an oversized but equally valid
    population (too many photos of one common species), while this is for
    rows that are the same real thing recorded more than once -- the same
    sighting independently published by two aggregators, say, each minting
    its own id, so no id-based check can see the collision. Built on top of
    cap_per_group (max_count=1 over a computed fingerprint column) rather than
    duplicating its group-and-keep logic.

    - `key_cols` -- columns that together fingerprint one real occurrence, e.g.
      ("decimalLatitude", "decimalLongitude", "eventDate") for a sighting
      published through two different aggregators. A row missing ANY key_col is
      exempt, kept untouched -- not knowing where or when a row was recorded
      cannot be the same as knowing it duplicates another, the reasoning
      cap_per_group already applies to a missing group_col value.
    - `precision` -- optional {column: ndigits} rounding a numeric key_col
      before matching, e.g. {"decimalLatitude": 4, "decimalLongitude": 4} (~11m
      at the equator) so two sources recording the same spot to different
      decimal precision still fingerprint alike. A column not named here is
      matched on its own value exactly.
    - `rule`, `seed`, `id_col`, `keep_ids` -- as cap_per_group: which row of a
      duplicate group survives, and which ids to prioritize keeping over a
      fresh pick -- the same reimport-additivity guarantee, so a duplicate
      resolved once (and already carrying masks or metrics) doesn't get
      orphaned by a later pull that happens to pick differently.
    - `prefer` -- optional `{column: values}` rule (see rows_matching) naming
      rows trusted to be unique among themselves, e.g. a source that never
      publishes one sighting twice. A preferred row is never removed, and any
      other row sharing its fingerprint is, keep_ids or not.

    This is a probabilistic match, not a source-declared fact like drop=' --
    two independent, genuinely different sightings can share a fingerprint by
    coincidence. Choose key_cols and precision no looser than the real risk of
    a false match warrants.
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

    deduped = cap_per_group(working, "_dedupe_fingerprint", 1, rule=rule,
                            seed=seed, id_col=id_col, keep_ids=keep_ids)
    deduped = (pd.concat([protected, deduped])
               .sort_values("_dedupe_position")
               .reset_index(drop=True))
    removed = len(df) - len(deduped)
    if removed:
        logger.info("deduplicated %d of %d row(s) sharing (%s)",
                   removed, len(df), ", ".join(key_cols))
    return deduped.drop(columns=["_dedupe_fingerprint", "_dedupe_position"])


def _take(rows, count, rule, seed):
    """Index of `count` rows out of rows, by the "random"/"first"/"last" rule."""
    if count <= 0 or rows.empty:
        return rows.index[:0]
    if rule == "random":
        kept_positions = set(sample_occurrences(rows.index.astype(str), count, seed=seed))
        return rows.index[rows.index.astype(str).isin(kept_positions)]
    if rule == "first":
        return rows.index[:count]
    return rows.index[-count:]  # "last"


def shard_occurrences(occurrence_ids, index, total):
    """
    This shard's disjoint slice of occurrence_ids, out of `total` shards.

    Sorted first, then handed out round-robin -- deterministic regardless of
    the order occurrence_ids arrives in or which process computes it, so any
    number of workers given the same ids and the same total always agree on
    the same non-overlapping split with no coordination between them. That's
    what lets a cluster job array (or a plain multiprocessing.Pool, or a
    handful of manual terminal invocations) run one shard per worker safely:
    worker i calls run_segments(..., shard=(i, n)) and never touches an
    occurrence any other worker is also touching.

    Sizes differ by at most one shard-to-shard (round-robin, not chunked), so
    no worker sits idle waiting on a shard several times the size of its
    neighbours'.

    - `index` -- this shard's number, 0 <= index < total.
    - `total` -- how many shards `occurrence_ids` is being split into.
    """
    if total < 1:
        raise ValueError(f"total must be at least 1, got {total}")
    if not (0 <= index < total):
        raise ValueError(f"index must be in [0, {total}), got {index}")

    ids = sorted(str(occurrence_id) for occurrence_id in occurrence_ids)
    return ids[index::total]
