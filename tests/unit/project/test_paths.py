"""
Project layout: every path derives from one directory, and nothing creates
anything.

Small tests for a small module, but the property they protect is the one the
whole package is arranged around -- there is no global DATA_DIR and no config,
so two projects coexist without sharing state. A path function that reached
outside its project_path, or created a directory on the way to naming one, would
break that quietly: the first symptom would be a project directory containing
folders for work nobody did.
"""

from pathlib import Path

import pytest

from critterframe.project import paths

# Every function taking only a project path, so "derives from project_path" and
# "creates nothing" can be asserted once for all of them rather than eight
# times with a copy-pasted body.
PATH_FUNCTIONS = [
    paths.project_dir,
    paths.occurrences_path,
    paths.images_path,
    paths.masks_path,
    paths.calibrations_path,
    paths.runs_and_metrics_path,
    paths.imports_dir,
    paths.definitions_dir,
    paths.subsets_path,
    paths.recipes_path,
    paths.visualizations_dir,
    paths.pipeline_dir,
    paths.products_dir,
    paths.models_dir,
    paths.models_registry_path,
]


@pytest.mark.parametrize("function", PATH_FUNCTIONS,
                         ids=lambda function: function.__name__)
def test_every_path_is_inside_the_project(tmp_path, function):
    result = function(tmp_path)
    assert isinstance(result, Path)
    assert tmp_path in result.parents or result == tmp_path


@pytest.mark.parametrize("function", PATH_FUNCTIONS,
                         ids=lambda function: function.__name__)
def test_naming_a_path_creates_nothing(tmp_path, function):
    """
    A project comes into existence lazily, as its first writer needs it. Asking
    where something WOULD live is not that writer.
    """
    project = tmp_path / "project"
    function(project)
    assert not project.exists()


@pytest.mark.parametrize("function", PATH_FUNCTIONS,
                         ids=lambda function: function.__name__)
def test_a_string_and_a_path_agree(tmp_path, function):
    """
    Every public entry point takes project_path first and callers pass whatever
    they have. `project_dir` funnels both spellings through `Path`, and this is
    what says so.
    """
    assert function(str(tmp_path)) == function(tmp_path)


def test_two_projects_share_no_path(tmp_path):
    """The point of deriving everything from one directory."""
    first, second = tmp_path / "a", tmp_path / "b"
    for function in PATH_FUNCTIONS:
        assert function(first) != function(second)


def test_reference_masks_are_a_separate_table(tmp_path):
    """
    Reference masks live in an identical table reached with reference=True.
    Validation is comparison between the two, so they must not be the same file.
    """
    assert paths.masks_path(tmp_path) != paths.masks_path(tmp_path, reference=True)
    assert paths.masks_path(tmp_path).name == "masks.parquet"
    assert paths.masks_path(tmp_path, reference=True).name == "reference_masks.parquet"


def test_products_are_one_folder_per_render(tmp_path):
    """A render's name is a folder under products/, not a suffix on one folder."""
    assert paths.products_dir(tmp_path, "plates").parent == paths.products_dir(tmp_path)
    assert paths.products_dir(tmp_path, "plates").name == "plates"


def test_pipeline_and_products_are_siblings_under_visualizations(tmp_path):
    """
    The two visualization contracts -- sampled QC sheets, and assets
    materialized per occurrence-part -- are separate directories precisely so
    which one a picture belongs to is decided by where it lands.
    """
    assert paths.pipeline_dir(tmp_path).parent == paths.visualizations_dir(tmp_path)
    assert paths.products_dir(tmp_path).parent == paths.visualizations_dir(tmp_path)
    assert paths.pipeline_dir(tmp_path) != paths.products_dir(tmp_path)


def test_registry_lives_with_the_checkpoints(tmp_path):
    """A registered model is a record OF the files under models/."""
    assert paths.models_registry_path(tmp_path).parent == paths.models_dir(tmp_path)


def test_require_project_rejects_a_directory_without_occurrences(empty_project):
    """
    The check every reader makes, so a typo'd path fails loudly instead of
    quietly reporting an empty project.
    """
    with pytest.raises(FileNotFoundError, match="isn't a CritterFrame project"):
        paths.require_project(empty_project)


def test_require_project_accepts_a_real_one(metadata_project):
    assert paths.require_project(metadata_project) == Path(metadata_project)


def test_require_project_does_not_create_the_directory(tmp_path):
    """A missing project must raise, not be conjured into existence."""
    missing = tmp_path / "nowhere"
    with pytest.raises(FileNotFoundError):
        paths.require_project(missing)
    assert not missing.exists()
