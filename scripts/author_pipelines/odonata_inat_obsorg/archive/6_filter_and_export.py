"""
Step 6 of 6: score the abdomens, calibrate filters against the screening
labels, measure the traits, export.

Unattended.

Before: 5_screen_abdomens.py.
"""

import logging

import critterframe as cf
from critterframe.metrics.embedding import pretrained
from critterframe.project.paths import exports_dir

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

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
                                       labels_run="abdomen_quality",
                                       labels_subset="part_screening_calibrate",
                                       label_metric="abdomen_quality",
                                       good_labels=["good"])])

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
    annotation_run="abdomen_quality",
    label_metric="abdomen_quality",
    good_labels=["good"],          # every other label is a bad abdomen
    part="abdomen",
    subset="part_screening_calibrate",
    max_fpr=0.05,
)

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

# The trait run, plus every run a filter reads. Only what passed the organism
# gate: the subset is the gate, and its note records the cutoffs.
df = cf.export_metrics(
    PROJECT_PATH,
    run_names=["fixed_thresholds", "part_qc", "abdomen_outliers",
               "abdomen_clusters", "abdomen_bad_score"],
    parts=["abdomen"],
    subset="organism_gate_pass",
    filters=abdomen_filters,
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
