"""
A map of appearance: each specimen placed by the first two principal components
of its embedding, then snapped to a grid so none overlap.

Specimens that look alike sit together, so a cluster of mis-segmented crops or a
mislabelled species stands out at a glance. Snapping is an optimal assignment of
specimens to grid cells, so positions stay as close to the projection as a
non-overlapping layout allows; raise density= for truer positions and more empty
space. The match compares every specimen with every cell, so sample a large
project down first (a subset, or exemplars_per_group).

Before: an embedding run named "embeddings" (cf.embedding()), after
remove_background() and crop_to_mask().
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

cf.render_collage(
    PROJECT_PATH,
    "embedding_map",
    layout=cf.collage.scatter(snap=True, density=1.5),
    arrange=[cf.collage.project("embeddings__organism__embedding")],
    run_names=["embeddings"],
)
