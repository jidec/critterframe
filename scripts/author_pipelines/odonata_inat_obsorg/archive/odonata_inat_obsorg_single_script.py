"""
Dragonfly bodies from iNaturalist observations published through GBIF

The most elaborate of the reference pipelines, and the one that shows what the
package is actually for

This is the unattended part, and it is run more than once. Every step skips
what it has already done, so a rerun only does what has become possible since:

  1. this script                                   stops after the organism segment
  2. odonata_inat_obsorg_annotate.py               usability labels, reference part masks
  3. odonata_inat_obsorg_train_part_segmenters.py  head / thorax / abdomen segmenters
  4. this script                                   organism gate, parts, abdomen scores
  5. odonata_inat_obsorg_annotate.py               screen abdomens
  6. this script                                   filters, traits, export

odonata_inat_obsorg_validate.py holds the reports that feed nothing here.
"""

import logging

import critterframe as cf
from critterframe.extensions.gbif_darwincore_inat import ingest as gbif_ingest
from critterframe.extensions.smp_segmenter import segmentation
from critterframe.metrics.embedding import pretrained
from critterframe.project.paths import exports_dir
from critterframe.validation.filters import SEGMENT_BAD_LABELS

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

gbif_ingest.ingest_occurrences(
    PROJECT_PATH,
    archive_path="D:/0036785-260806074905277.zip",
    occurrence_columns=gbif_ingest.DEFAULT_INAT_OCCURRENCE_COLUMNS,
    inat_photo_size="medium",
    dedupe_key_cols=["eventDate","decimalLatitude","decimalLongitude"],
    group_col="scientificName",
    max_per_group=500,
    trust_source_file_unchanged=True,
    prioritize_inat=True,
)

cf.download_images(PROJECT_PATH)

# Whole organism. Persisted, and everything below starts from it.
cf.run_segments(
    PROJECT_PATH,
    run_name="organism_groundedsam2",
    steps=[
        cf.segment(cf.groundedsam2(text_prompt="insect.", box_threshold=0.25, text_threshold=0.25),
                   mask_threshold=0.0),
    ],
)

# --- organism gate: which occurrences are worth running parts on --------------
#
# The organism segment says things about validity the part crops can't: how sure
# the detector was, whether it saw a second organism, whether the mask runs off
# the frame. Conservative on purpose: a good specimen gated out never gets
# parts, and nothing downstream would notice. Nothing is deleted, and loosening
# the gate later only adds the newly admitted occurrences.

# mask_info is what groundedsam2 recorded on each mask: box_score, score,
# n_boxes, second_box_score, ...
cf.run_metrics(
    PROJECT_PATH, run_name="organism_qc",
    metrics=[cf.mask_info(operations=["segment"]), cf.edge_fraction(),
             cf.mask_fraction(), cf.blur_variance()],
)

# One cutoff per score, each allowed to cost 2% of usable images. A score that
# catches no unusable image is left out, and what the cutoffs do together is
# logged, along with the reasons they can't see: expect bad_angle,
# wrong_life_stage and dead to pass through to the abdomen filters.
gate = cf.get_validated_filters(
    PROJECT_PATH,
    metric_specs={
        "mask_info__segment__box_score": "below",          # detector unsure
        "mask_info__segment__score": "below",              # SAM2 unsure of its mask
        "mask_info__segment__second_box_score": "above",   # a second organism in frame
        "edge_fraction": "above",                          # mask runs off the frame
        "blur_variance": "below",                          # out of focus
    },
    predicted_run="organism_qc",
    annotation_run="usability_annotation",
    subset="usability_annotation_set",
    max_fpr=0.02,
)

passing = cf.export_metrics(PROJECT_PATH, path=False, manifest=False,
                            run_names=["organism_qc"], parts=["organism"], filters=gate)

cf.define_subset(PROJECT_PATH, name="organism_gate_pass",
                 occurrence_ids=sorted(passing["occurrence_id"]),
                 note=f"passes the organism gate: {gate}")

cf.run_segments(
    PROJECT_PATH,
    run_name="body_parts",
    from_part="organism",
    subset="organism_gate_pass",
    shared_steps=[cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
    outputs={
        "head":    [cf.segment(segmentation.load_registered(PROJECT_PATH, "head_segmenter_v1"))],
        "thorax":  [cf.segment(segmentation.load_registered(PROJECT_PATH, "thorax_segmenter_v1"))],
        "abdomen": [cf.segment(segmentation.load_registered(PROJECT_PATH, "abdomen_segmenter_v1"))],
    },
)

cf.run_segments(PROJECT_PATH, part="body", from_part=["head", "thorax", "abdomen"], steps=[],
                subset="organism_gate_pass")

# --- abdomen scores: candidates for filtering out bad abdomen segments --------

cf.run_metrics(PROJECT_PATH, run_name="part_qc", part="abdomen", subset="organism_gate_pass",
               metrics=[cf.mask_info(), cf.mask_fraction()])

resnet_embedding = cf.embedding(pretrained("resnet18", resize="pad"),
                                name="resnet18_embedding")
cf.run_metrics(
    PROJECT_PATH, part="abdomen", subset="organism_gate_pass",
    transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
                cf.orient(axis_strategy="longer")],
    metrics=[resnet_embedding],   # run_name defaults to "resnet18_embedding"
)

cf.run_metrics(PROJECT_PATH, run_name="abdomen_outliers", part="abdomen", subset="organism_gate_pass",
               metrics=[cf.outlier([resnet_embedding], from_run="resnet18_embedding")])

cf.run_metrics(PROJECT_PATH, run_name="abdomen_clusters", part="abdomen", subset="organism_gate_pass",
               metrics=[cf.cluster([resnet_embedding], from_run="resnet18_embedding",
                                   n_clusters=10, n_components=16)])

# fitted on the screened sample's labels
cf.run_metrics(PROJECT_PATH, run_name="abdomen_bad_score", part="abdomen", subset="organism_gate_pass",
               metrics=[cf.label_score([resnet_embedding], from_run="resnet18_embedding",
                                       labels_run="segment_quality",
                                       labels_subset="part_screening_calibrate")])

# --- abdomen filters, calibrated against the screening labels -----------------
#
# Calibration only for now, so every rate logged here describes the labels the
# filters were chosen on. Stating how clean an export is needs a second sample,
# screened from outside part_screening_calibrate and passed as audit_subset=.
#
# Each candidate may cost 5% of good abdomens on its own. Delete a line to
# export without that candidate.
abdomen_filters = cf.get_validated_filters(
    PROJECT_PATH,
    metric_specs={
        "part_qc":           {"mask_info__segment__score": "below"},
        # IsolationForest's decision function: lower is more anomalous
        "abdomen_outliers":  {"outlier__anomaly_score": "below"},
        # the worst clusters are dropped, within the same 5%
        "abdomen_clusters":  {"cluster__cluster_id": "category"},
        "abdomen_bad_score": {"label_score__bad_probability": "above"},
    },
    annotation_run="segment_quality",
    label_metric="segment_quality_annotation",
    bad_labels=SEGMENT_BAD_LABELS,
    part="abdomen",
    subset="part_screening_calibrate",
    max_fpr=0.05,
)

filters = {**gate, **abdomen_filters}

# --- traits --------------------------------------------------------------------

# every bin also needs high chroma, so browns, greys and pruinose whites land in "unmatched".
# light blue gets a lower chroma floor because pale blues are inherently low-chroma (sky blue ~26, brown ~24)
cf.run_metrics(
    PROJECT_PATH,
    run_name="fixed_thresholds",
    subset="organism_gate_pass",
    part="abdomen",
    metrics=[
        cf.threshold_fractions([
            cf.color_threshold("red",        lch_h=(345, 55),  lch_c=(30, None)),
            cf.color_threshold("yellow",     lch_h=(55, 105),  lch_c=(30, None)),
            cf.color_threshold("green",      lch_h=(105, 195), lch_c=(30, None)),
            cf.color_threshold("blue",       lch_h=(195, 320), lch_c=(30, None), lch_l=(None, 55)),
            cf.color_threshold("light_blue", lch_h=(195, 320), lch_c=(20, None), lch_l=(55, None)),
        ], unmatched=True, name="color_bins"),
    ],
)

# --- export --------------------------------------------------------------------

# the trait run, plus every run a filter reads
df = cf.export_metrics(
    PROJECT_PATH,
    run_names=["fixed_thresholds", "organism_qc", "part_qc", "abdomen_outliers",
               "abdomen_clusters", "abdomen_bad_score"],
    parts=["abdomen", "organism"],
    subset="organism_gate_pass",
    filters=filters,
    occurrence_columns=["sex","occurrence_id","eventDate","decimalLatitude","decimalLongitude",
                        "coordinateUncertaintyInMeters","order","family","genus","species","elevation","image_url"],
)

# Derived colour columns. A colour is "present" when it covers at least 10% of the abdomen;
# "unmatched" is not a colour, so it never counts. An occurrence with no value for a colour counts it as absent.
colors = ["red", "yellow", "green", "blue", "light_blue"]
fractions = df[["fixed_thresholds__abdomen__color_bins__" + color for color in colors]].set_axis(colors, axis=1)
present = fractions.ge(0.10)

for color in colors:
    df[f"{color}_present"] = present[color]
df["n_colors_present"] = present.sum(axis=1)


def dominant(row, rank):
    """The rank-th (0-based) largest present colour in one row, or None if fewer are present; ties keep `colors` order."""
    ordered = row.dropna().sort_values(ascending=False, kind="stable")
    return ordered.index[rank] if len(ordered) > rank else None


present_fractions = fractions.where(present)       # absent colours become NaN, so they can't be ranked
df["dominant_color_1"] = present_fractions.apply(dominant, axis=1, rank=0)
df["dominant_color_2"] = present_fractions.apply(dominant, axis=1, rank=1)

print(df["n_colors_present"].value_counts().sort_index())
print(df[["dominant_color_1", "dominant_color_2"]].value_counts(dropna=False).head(15))

exports_dir(PROJECT_PATH).mkdir(parents=True, exist_ok=True)   # paths creates nothing itself
df.to_csv(exports_dir(PROJECT_PATH) / "abdomen_color_bins_derived.csv")
