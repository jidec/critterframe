"""
Selections that read a project: the occurrences whose stored values match a
rule, and the ones most typical of their group.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.metrics.annotation import exclusive_label_annotation
from critterframe.core.recipes import Recipe
from critterframe.records.metrics import append_metrics, make_metric_row
from critterframe.records.occurrences import ID_COL, load_occurrences
from critterframe.records.runs import start_run
from critterframe.records import masks as mask_records
from critterframe.selection.queries import ids_matching, ids_passing, ids_with_image, ids_with_mask
from helpers.synthetic import blob_mask
from helpers.stored import store_values


# ---------------------------------------------------------------------------
# exemplars_per_group
# ---------------------------------------------------------------------------


def pointing(degrees, length=1.0):
    """A 2-long vector of that direction and length."""
    angle = np.deg2rad(degrees)
    return [float(length * np.cos(angle)), float(length * np.sin(angle))]


@pytest.fixture
def embedded_project(metadata_project):
    """
    Six specimens with a stored 2-long vector, three per species. In each
    species one points between the other two; the lydia one is also 100 times
    as long, so it is the medoid by direction and far from both by raw distance.
    """
    occurrences = load_occurrences(metadata_project)
    lydia = occurrences.loc[occurrences["species"] == "Libellula lydia", ID_COL].tolist()[:3]
    anax = occurrences.loc[occurrences["species"] == "Anax junius", ID_COL].tolist()[:3]
    vectors = {
        lydia[0]: pointing(0),
        lydia[1]: pointing(10, length=100.0),
        lydia[2]: pointing(20),
        anax[0]: pointing(90),
        anax[1]: pointing(100),
        anax[2]: pointing(80),
    }
    store_values(metadata_project, vectors, run_name="embedding", metric_name="embedding", unit="embedding")
    return metadata_project, lydia, anax


def test_one_exemplar_per_species_is_its_medoid(embedded_project):
    project, lydia, anax = embedded_project
    assert cf.exemplars_per_group(project, "embedding", "species") == sorted([lydia[1], anax[0]])


def test_vectors_are_compared_by_direction_unless_asked_otherwise(embedded_project):
    """By raw distance the long lydia vector is the furthest from the other two, not the medoid."""
    project, lydia, anax = embedded_project
    raw = cf.exemplars_per_group(project, "embedding", "species", normalize=False)
    assert lydia[1] not in raw and anax[0] in raw


def test_only_the_given_occurrences_are_measured_against(embedded_project):
    """Without the first lydia, the other two are each the other's only neighbour."""
    project, lydia, anax = embedded_project
    chosen = cf.exemplars_per_group(
        project, "embedding", "species", occurrence_ids=[lydia[1], lydia[2], *anax]
    )
    # a tie, which goes to the first id
    assert chosen == sorted([min(lydia[1], lydia[2]), anax[0]])


def test_several_exemplars_per_group(embedded_project):
    project, lydia, anax = embedded_project
    assert len(cf.exemplars_per_group(project, "embedding", "species", count=2)) == 4


def test_a_vector_of_another_length_is_left_out(embedded_project, caplog):
    project, lydia, anax = embedded_project
    store_values(
        project, {lydia[1]: [1.0, 0.0, 5.0]}, run_name="embedding", metric_name="embedding", unit="embedding"
    )
    with caplog.at_level("WARNING"):
        chosen = cf.exemplars_per_group(project, "embedding", "species")
    assert lydia[1] not in chosen and "left out" in caplog.text


def test_a_single_number_picks_the_median_member_without_being_scaled_away(metadata_project):
    """Scaled to unit length every length would be 1 and the first id would win."""
    occurrences = load_occurrences(metadata_project)
    lydia = occurrences.loc[occurrences["species"] == "Libellula lydia", ID_COL].tolist()[:3]
    store_values(metadata_project, {lydia[0]: 90.0, lydia[1]: 10.0, lydia[2]: 50.0})

    assert cf.exemplars_per_group(metadata_project, "traits", "species", metric_name="body_length") == [
        lydia[2]
    ]


def test_a_dict_of_numbers_is_a_feature_and_its_other_entries_are_skipped(metadata_project, caplog):
    """Colour proportions: the member whose mix lies between the other two is the exemplar."""
    occurrences = load_occurrences(metadata_project)
    lydia = occurrences.loc[occurrences["species"] == "Libellula lydia", ID_COL].tolist()[:3]
    anax = occurrences.loc[occurrences["species"] == "Anax junius", ID_COL].tolist()[:1]

    def mix(red, blue, top):
        return {
            "red": red,
            "blue": blue,
            "unmatched": 1.0 - red - blue,
            "ranked_color_1": top,
            "any_present": True,
        }

    store_values(
        metadata_project,
        {
            lydia[0]: mix(0.9, 0.0, "red"),
            lydia[1]: mix(0.1, 0.8, "blue"),
            lydia[2]: mix(0.5, 0.4, "red"),
            anax[0]: {"red": 0.2, "ranked_color_1": "red"},
        },  # no blue: other keys
        run_name="colours",
        metric_name="color_bins",
        unit="fraction",
    )

    with caplog.at_level("WARNING"):
        chosen = cf.exemplars_per_group(
            metadata_project, "colours", "species", metric_name="color_bins", normalize=False
        )
    assert chosen == [lydia[2]]
    assert "left out" in caplog.text


def test_exemplars_need_a_real_group_column_and_stored_values(embedded_project, caplog):
    project, _lydia, _anax = embedded_project
    with pytest.raises(KeyError, match="group by"):
        cf.exemplars_per_group(project, "embedding", "nope")
    with caplog.at_level("WARNING"):
        assert cf.exemplars_per_group(project, "never_run", "species") == []
    assert "no exemplars" in caplog.text


# ---------------------------------------------------------------------------
# ids_matching
# ---------------------------------------------------------------------------


def screen():
    """A project's own screening label: one about the image, so asked without a mask."""
    return exclusive_label_annotation(["usable", "cut_off", "blurry"], name="usability", requires_mask=False)


def store_flags(project_path, flags, run_name="screening", source_mask_hash=None):
    recipe = Recipe("metric", run_name, [screen()], part="organism")
    run_id = start_run(project_path, recipe)
    append_metrics(
        project_path,
        run_id,
        recipe.hash,
        [
            make_metric_row(
                occurrence_id,
                "organism",
                "usability",
                flag,
                unit="category",
                source_mask_hash=source_mask_hash,
            )
            for occurrence_id, flag in flags.items()
        ],
    )


def test_stored_labels_can_be_selected_on_by_bare_metric_name(metadata_project):
    """
    The run and part prefixes are added for you, so a screening pass's usable
    crops are {"usability": "usable"}.
    """
    store_flags(metadata_project, {"specimen0": "usable", "specimen1": "cut_off", "specimen2": "usable"})
    assert ids_matching(metadata_project, "screening", {"usability": "usable"}) == [
        "specimen0",
        "specimen2",
    ]


def test_a_mistyped_metric_name_raises_once_there_is_data(metadata_project):
    """
    Because the usual thing to do with the answer is define a subset from it,
    and a subset that is silently empty looks exactly like a review pass nobody
    has done yet.
    """
    store_flags(metadata_project, {"specimen0": "usable"})
    with pytest.raises(KeyError, match="rule column"):
        ids_matching(metadata_project, "screening", {"annotate_flag": "usable"})


def test_a_run_nobody_has_done_yet_selects_none_and_says_so(metadata_project, caplog):
    """
    The one empty case that ISN'T a typo. There is nothing to check a rule
    against, so the typo guard can't apply -- it applies from the first
    recorded value onward.
    """
    with caplog.at_level("WARNING"):
        assert ids_matching(metadata_project, "screening", {"usability": "usable"}) == []
    assert "nothing to match" in caplog.text


def test_labels_survive_a_resegmentation_by_default(metadata_project):
    """
    current_only is False here, against the grain of everything else that reads
    stored values: a label like "cut_off" describes the CROP and stays true
    whatever mask was on screen. Left at True, resegmenting would void a review
    session over a change the labels never depended on.
    """
    from critterframe.records import masks as mask_records
    from helpers.synthetic import blob_mask

    store_flags(metadata_project, {"specimen0": "usable"}, source_mask_hash="the_mask_that_was_on_screen")
    mask_records.save_masks(
        metadata_project, [mask_records.make_mask_row("specimen0", blob_mask(), recipe_hash="brand_new")]
    )

    assert ids_matching(metadata_project, "screening", {"usability": "usable"}) == ["specimen0"]
    assert ids_matching(metadata_project, "screening", {"usability": "usable"}, current_only=True) == []


def test_numeric_labels_match_as_stored(metadata_project):
    """Values are compared as they were stored, without coercion."""
    recipe = Recipe("metric", "qc", [screen()], part="organism")
    run_id = start_run(metadata_project, recipe)
    append_metrics(
        metadata_project,
        run_id,
        recipe.hash,
        [
            make_metric_row("specimen0", "organism", "grade", 3),
            make_metric_row("specimen1", "organism", "grade", np.int64(4)),
        ],
    )
    assert ids_matching(metadata_project, "qc", {"grade": 4}) == ["specimen1"]


# ---------------------------------------------------------------------------
# ids_with_mask, ids_with_image
# ---------------------------------------------------------------------------


def mask_row(occurrence_id, part="organism"):
    return mask_records.make_mask_row(occurrence_id, blob_mask(), recipe_hash="recipe", part=part)


def test_ids_with_mask_is_per_part_and_sorted(tmp_path):
    mask_records.save_masks(tmp_path, [mask_row("b"), mask_row("a"), mask_row("a", part="wing")])
    assert ids_with_mask(tmp_path, part="organism") == ["a", "b"]
    assert ids_with_mask(tmp_path, part="wing") == ["a"]
    assert ids_with_mask(tmp_path, part="head") == []


def test_ids_with_mask_reads_the_reference_table(tmp_path):
    mask_records.save_masks(tmp_path, [mask_row("a")], reference=True)
    assert ids_with_mask(tmp_path, reference=True) == ["a"]
    assert ids_with_mask(tmp_path) == []


def test_ids_with_image_lists_what_the_store_holds(image_project):
    assert ids_with_image(image_project) == sorted(cf.select_ids(image_project))


def test_asking_who_has_an_image_does_not_create_a_store(metadata_project):
    """A query reads; a project with no images must not gain an empty LMDB for having been asked."""
    assert ids_with_image(metadata_project) == []
    assert not (metadata_project / "images.lmdb").exists()


# ---------------------------------------------------------------------------
# ids_passing: a gate keeps exactly the rows an export with its filters would
# ---------------------------------------------------------------------------

QC_AREA = "qc__organism__mask_area"


@pytest.fixture
def screened_project(measured_project):
    """The measured project plus a 'qc' run, the kind a gate reads."""
    cf.run_metrics(
        measured_project, run_name="qc", visualize=False, metrics=[cf.mask_area(), cf.mask_fraction()]
    )
    return measured_project


def exported_ids(project_path, **kwargs):
    return sorted(cf.export_metrics(project_path, path=False, manifest=False, **kwargs)[ID_COL].astype(str))


def test_a_gate_keeps_what_an_export_with_its_filters_keeps(screened_project):
    cutoff = cf.export_metrics(screened_project, path=False, manifest=False)[QC_AREA].median()
    gate = {QC_AREA: (">", cutoff)}

    passing = ids_passing(screened_project, gate)

    assert 0 < len(passing) < len(cf.select_ids(screened_project))
    assert passing == exported_ids(screened_project, filters=gate)


def test_a_gate_reads_a_column_its_own_selection_left_out(screened_project):
    cutoff = cf.export_metrics(screened_project, path=False, manifest=False)[QC_AREA].median()
    gate = {QC_AREA: (">", cutoff)}

    passing = ids_passing(screened_project, gate, run_names=["traits"])

    assert passing == exported_ids(screened_project, filters=gate, run_names=["traits"])
    assert passing == ids_passing(screened_project, gate)


def test_a_gate_can_be_written_in_millimetres(screened_project):
    cf.declare_scale(screened_project, 4.0, scope="device", scope_value="boxA")
    converted = f"{QC_AREA}_mm2"
    cutoff = cf.export_metrics(screened_project, path=False, manifest=False, units="mm")[converted].median()
    gate = {converted: (">", cutoff)}

    passing = ids_passing(screened_project, gate, units="mm")

    assert 0 < len(passing)
    assert passing == exported_ids(screened_project, filters=gate, units="mm")


def test_a_gate_is_narrowed_by_a_subset(screened_project):
    ids = cf.select_ids(screened_project)
    cf.define_subset(screened_project, "half", occurrence_ids=ids[:4])

    passing = ids_passing(screened_project, {QC_AREA: (">", 0)}, subset="half")

    assert passing == sorted(ids[:4])


def test_a_gate_writes_no_export_and_logs_none(screened_project):
    ids_passing(screened_project, {QC_AREA: (">", 0)})
    assert len(cf.load_exports(screened_project)) == 0
    assert not (screened_project / "exports").exists()


def test_a_gate_on_a_column_no_run_holds_raises(screened_project):
    with pytest.raises(KeyError, match="filter column"):
        ids_passing(screened_project, {"qc__organism__no_such_metric": (">", 0)})
