"""
Subsets: named selections of occurrences intended to receive a particular
recipe.

A subset selects rows; it never copies or moves them. An occurrence can belong
to several or to none, and running a recipe over one leaves every other
subset's masks and metrics untouched.

The definitions live in a hand-editable TOML file, which is why the serializer
gets a round-trip test of its own: deciding which collection is which is human
knowledge about the data, and a file a person is invited to correct has to
survive being written by us and read back after they have.
"""

import pandas as pd
import pytest

import critterframe as cf
from critterframe.project import paths
from critterframe.project import subsets as subset_selection
from critterframe.records.occurrences import ID_COL, save_occurrences

try:                                    # 3.11+
    import tomllib
except ModuleNotFoundError:             # 3.10, via the tomli backport
    import tomli as tomllib


@pytest.fixture
def collections_project(tmp_path):
    """Six occurrences across three collections and two years."""
    save_occurrences(tmp_path, pd.DataFrame({
        ID_COL: [f"occ{index}" for index in range(6)],
        "collection": ["AMNH", "AMNH", "MCZ", "MCZ", "Alabama Museum",
                       "Alabama Museum"],
        "year": [2019, 2020, 2019, 2021, 2021, 2022],
    }))
    return tmp_path


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


def test_a_project_without_subsets_is_not_an_error(tmp_path):
    """The normal case: most projects process everything the same way."""
    assert subset_selection.load_subsets(tmp_path) == {}


def test_definitions_land_in_a_hand_editable_file(collections_project):
    cf.define_subset(collections_project, "amnh", column="collection",
                     values=["AMNH"])
    path = paths.subsets_path(collections_project)

    assert path.exists()
    assert "[subsets.amnh]" in path.read_text()


def test_what_we_write_is_what_tomllib_reads_back(collections_project):
    """
    The serializer is hand-rolled, and reading uses the standard library --
    which is where correctness actually matters, since that is what has to cope
    with whatever a human hand-edits.
    """
    written = subset_selection.save_subsets(collections_project, {
        "quoted": {"column": 'a "difficult" name', "values": ["x\\y", "z"]},
        "numeric": {"column": "year", "values": [2019, 2020]},
        "flagged": {"column": "ok", "values": [True]},
    })
    with paths.subsets_path(collections_project).open("rb") as handle:
        assert tomllib.load(handle)["subsets"] == written


def test_saving_replaces_the_whole_table(collections_project):
    cf.define_subset(collections_project, "amnh", column="collection",
                     values=["AMNH"])
    subset_selection.save_subsets(collections_project, {"only": {"query": "year > 2020"}})
    assert list(subset_selection.load_subsets(collections_project)) == ["only"]


# ---------------------------------------------------------------------------
# define_subset
# ---------------------------------------------------------------------------


def test_a_column_and_values_rule(collections_project):
    cf.define_subset(collections_project, "amnh", column="collection",
                     values=["AMNH"])
    assert subset_selection.load_subsets(collections_project)["amnh"] == {
        "column": "collection", "values": ["AMNH"]}


def test_a_query_rule_for_what_a_column_cannot_express(collections_project):
    cf.define_subset(collections_project, "recent", query="year >= 2021")
    assert subset_selection.select_ids(collections_project,
                                       subset="recent") == ["occ3", "occ4", "occ5"]


def test_an_explicit_id_list_is_its_own_definition(collections_project):
    """
    For a hand-picked selection with no rule behind it -- a validation set
    chosen by eye -- where writing the ids down IS the definition.
    """
    cf.define_subset(collections_project, "checked",
                     occurrence_ids=["occ0", "occ4"])
    assert subset_selection.select_ids(collections_project,
                                       subset="checked") == ["occ0", "occ4"]


def test_ids_are_stored_as_strings(collections_project):
    cf.define_subset(collections_project, "numeric", occurrence_ids=[1, 2])
    assert subset_selection.load_subsets(collections_project)["numeric"] == {
        "occurrence_ids": ["1", "2"]}


@pytest.mark.parametrize("kwargs", [
    {},
    {"values": ["AMNH"], "query": "year > 2020"},
    {"column": "collection", "values": ["AMNH"], "occurrence_ids": ["occ0"]},
])
def test_exactly_one_rule_is_required(collections_project, kwargs):
    with pytest.raises(ValueError, match="exactly one of"):
        cf.define_subset(collections_project, "bad", **kwargs)


def test_values_need_a_column_to_match_on(collections_project):
    with pytest.raises(ValueError, match="needs column="):
        cf.define_subset(collections_project, "bad", values=["AMNH"])


def test_redefining_a_subset_replaces_it(collections_project):
    cf.define_subset(collections_project, "one", column="collection",
                     values=["AMNH"])
    cf.define_subset(collections_project, "one", column="collection",
                     values=["MCZ"])
    assert subset_selection.select_ids(collections_project,
                                       subset="one") == ["occ2", "occ3"]


def test_several_subsets_are_defined_from_one_column(collections_project):
    """
    The usual shape: a single metadata column already separates the groups and
    only the names need tidying.
    """
    cf.define_subsets(collections_project, "collection", {
        "AMNH": "amnh", "MCZ": "mcz", "Alabama Museum": "alabama"})
    assert sorted(subset_selection.load_subsets(collections_project)) == [
        "alabama", "amnh", "mcz"]


def test_several_values_can_map_to_one_subset(collections_project):
    """Which merges them -- two spellings of one collection, say."""
    cf.define_subsets(collections_project, "collection",
                      {"AMNH": "east", "MCZ": "east"})
    assert subset_selection.select_ids(collections_project, subset="east") == [
        "occ0", "occ1", "occ2", "occ3"]


# ---------------------------------------------------------------------------
# select_occurrences
# ---------------------------------------------------------------------------


def test_no_subset_selects_the_whole_project(collections_project):
    """
    "No subset" and "this subset" are the same code path with the same
    guarantees rather than two -- every run funnels through here.
    """
    assert len(subset_selection.select_occurrences(collections_project)) == 6


def test_an_unknown_subset_name_raises_and_lists_the_real_ones(collections_project):
    cf.define_subset(collections_project, "amnh", column="collection",
                     values=["AMNH"])
    with pytest.raises(KeyError, match="no subset named 'amhn'"):
        subset_selection.select_occurrences(collections_project, subset="amhn")


def test_a_subset_selecting_on_a_missing_column_raises(collections_project):
    subset_selection.save_subsets(collections_project,
                                  {"ghost": {"column": "site", "values": ["x"]}})
    with pytest.raises(KeyError, match="which the occurrence table doesn't have"):
        subset_selection.select_occurrences(collections_project, subset="ghost")


def test_the_rule_column_is_read_even_when_other_columns_were_asked_for(
        collections_project):
    """
    Otherwise narrowing the read would break the very rule doing the
    narrowing.
    """
    cf.define_subset(collections_project, "amnh", column="collection",
                     values=["AMNH"])
    selected = subset_selection.select_occurrences(collections_project,
                                                   subset="amnh",
                                                   columns=["year"])
    assert len(selected) == 2
    assert {"year", "collection", ID_COL} <= set(selected.columns)


def test_a_limit_applies_after_selection(collections_project):
    """For trying a recipe on a handful before committing to the project."""
    cf.define_subset(collections_project, "amnh", column="collection",
                     values=["AMNH"])
    assert subset_selection.select_ids(collections_project, subset="amnh",
                                       limit=1) == ["occ0"]


def test_selection_returns_a_clean_index(collections_project):
    cf.define_subset(collections_project, "mcz", column="collection",
                     values=["MCZ"])
    selected = subset_selection.select_occurrences(collections_project,
                                                   subset="mcz")
    assert selected.index.tolist() == [0, 1]


def test_an_occurrence_can_belong_to_several_subsets(collections_project):
    cf.define_subset(collections_project, "amnh", column="collection",
                     values=["AMNH"])
    cf.define_subset(collections_project, "2020", column="year", values=[2020])

    assert "occ1" in subset_selection.select_ids(collections_project, subset="amnh")
    assert "occ1" in subset_selection.select_ids(collections_project, subset="2020")


def test_ids_come_back_as_a_list_of_strings(collections_project):
    ids = subset_selection.select_ids(collections_project)
    assert isinstance(ids, list)
    assert all(isinstance(occurrence_id, str) for occurrence_id in ids)
