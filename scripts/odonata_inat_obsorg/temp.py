from critterframe.metrics.inductive_color_thresholds import inductive_color_thresholds
from critterframe.records.masks import occurrence_ids_with_mask
from critterframe.selectionhelpers import sample_occurrences

import logging

import critterframe as cf
from critterframe.extensions.gbif_darwincore_inat import ingest as gbif_ingest
from critterframe.metrics.outliers import outlier

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

MIN_CHROMA = 30
LIGHT_BLUE_MIN_CHROMA = 20

cf.run_metrics(
    PROJECT_PATH,
    run_name="fixed_thresholds",
    subset="thresholds_test",
    part="abdomen",
    metrics=[
        cf.threshold_fractions([
            cf.color_threshold("red",        lch_h=(345, 55),  lch_c=(MIN_CHROMA, None)),
            cf.color_threshold("yellow",     lch_h=(55, 105),  lch_c=(MIN_CHROMA, None)),
            cf.color_threshold("green",      lch_h=(105, 195), lch_c=(MIN_CHROMA, None)),
            cf.color_threshold("blue",       lch_h=(195, 320), lch_c=(MIN_CHROMA, None), lch_l=(None, 55)),
            cf.color_threshold("light_blue", lch_h=(195, 320), lch_c=(LIGHT_BLUE_MIN_CHROMA, None), lch_l=(55, None)),
        ], unmatched=True, name="color_bins"),
    ],
    visualize=True,
)

df = cf.export_metrics(PROJECT_PATH, run_names=["fixed_thresholds"], parts=["abdomen"],
                       occurrence_columns=["sex","occurrence_id","eventDate","decimalLatitude","decimalLongitude",
                                           "coordinateUncertaintyInMeters","order","family","genus","species","elevation","image_url"])

# Derived colour columns. A colour is "present" when it covers at least PRESENT_THRESHOLD of the abdomen;
# "unmatched" is not a colour, so it never counts. An occurrence with no value for a colour counts it as absent.
COLORS = ["red", "yellow", "green", "blue", "light_blue"]
PRESENT_THRESHOLD = 0.10
PREFIX = "fixed_thresholds__abdomen__color_bins__"

fractions = df[[PREFIX + color for color in COLORS]].set_axis(COLORS, axis=1)
present = fractions.ge(PRESENT_THRESHOLD)

for color in COLORS:
    df[f"{color}_present"] = present[color]
df["n_colors_present"] = present.sum(axis=1)


def _dominant(row, rank):
    """The rank-th (0-based) largest present colour in one row, or None if fewer are present; ties keep COLORS order."""
    ordered = row.dropna().sort_values(ascending=False, kind="stable")
    return ordered.index[rank] if len(ordered) > rank else None


present_fractions = fractions.where(present)       # absent colours become NaN, so they can't be ranked
df["dominant_color_1"] = present_fractions.apply(_dominant, axis=1, rank=0)
df["dominant_color_2"] = present_fractions.apply(_dominant, axis=1, rank=1)

print(df["n_colors_present"].value_counts().sort_index())
print(df[["dominant_color_1", "dominant_color_2"]].value_counts(dropna=False).head(15))

from critterframe.project.paths import exports_dir
exports_dir(PROJECT_PATH).mkdir(parents=True, exist_ok=True)   # paths creates nothing itself
df.to_csv(exports_dir(PROJECT_PATH) / "abdomen_color_bins_derived.csv")