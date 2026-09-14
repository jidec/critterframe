import critterframe as cf

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# create reference set for 100 segments


# create sub
cf.define_subset(
    PROJECT_PATH, name="groundedsam_tests",
    occurrence_ids=cf.sample_occurrences(count=100),
)

cf.run_segments