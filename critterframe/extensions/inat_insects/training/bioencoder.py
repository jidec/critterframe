"""
Training and fine-tuning a BioEncoder embedding model on a project.

The path this completes: segment organisms, remove their backgrounds, train an
embedding model on the results using the project's own taxonomy as labels, then
plug the trained model back in as an embedding metric
(extensions.inat_insects.metrics.bioencoder). A project ends up producing the
model that measures it.

Why metric learning rather than classification: the useful output isn't "which
species is this" -- the project usually already knows, from the community
identification -- it's a SPACE where similar organisms are close together. That
space keeps working for taxa the model never saw, supports outlier detection
and clustering, and doesn't have to be retrained when a new species appears in
the data. A classifier gives you a label and nothing else.

SCAFFOLD. The dataset preparation below is real and runnable; the training loop
is a defined contract with the pieces that need a decision left explicit
(backbone, loss, augmentation). Those are choices that depend on the project's
size and taxonomy, and guessing at them here would produce a model that trains
without complaint and embeds badly, which is worse than a NotImplementedError.
"""

import logging
import os

from ....recipes import DEFAULT_PART
from ....training.datasets import write_dataset
from ....training.splits import split_dataset

logger = logging.getLogger(__name__)

# Label column used for metric learning. Species is the natural choice: it's
# what the identification actually asserts, and it's the level at which "these
# two images should be close together" is a claim worth training on.
DEFAULT_LABEL_COL = "taxon"

# Group column for splitting. Observer rather than the obvious choice of taxon,
# because one observer photographing the same individual repeatedly is the main
# source of near-duplicates in iNaturalist data -- a random split puts the same
# animal in train and validation and reports an excellent score for it.
DEFAULT_GROUP_COL = "observer"

# Minimum images per label. Metric learning needs several examples of a class
# to have anything to pull together; a class with one image contributes no
# positive pair at all and just adds noise.
MIN_IMAGES_PER_LABEL = 5


def prepare_dataset(project_path, output_dir, part=DEFAULT_PART, transforms=(),
                    label_col=DEFAULT_LABEL_COL, group_col=DEFAULT_GROUP_COL,
                    min_per_label=MIN_IMAGES_PER_LABEL, fractions=None,
                    reference=False, subset=None, limit=None, seed=0):
    """
    Build a training dataset out of a project: image/mask pairs on disk, labels
    attached, split into train/val/test.

    project_path  -- project to build from.
    output_dir    -- directory to write into.
    part          -- part to train on.
    transforms    -- operations applied to each segment. Pass the SAME chain
                    the embedding metric will use at inference time, typically
                    [remove_background(), crop_to_mask(), orient()] -- a model
                    trained on background-removed, oriented segments and then
                    run on raw photographs will embed badly and give no
                    indication of why.
    label_col     -- occurrence column holding the label.
    group_col     -- occurrence column to group by when splitting, so
                    near-duplicates can't straddle train and validation.
    min_per_label -- labels with fewer images than this are dropped.
    fractions     -- split fractions; 70/15/15 by default.
    reference     -- train on reference masks rather than canonical ones.
    subset, limit -- restrict which occurrences are used.
    seed          -- split seed, so the split is reproducible across runs.

    Returns the manifest DataFrame, with `split` and `label` columns added.
    """
    label_columns = [column for column in {label_col, group_col} if column]

    manifest = write_dataset(project_path, output_dir, part=part,
                             transforms=transforms, reference=reference,
                             subset=subset, limit=limit,
                             label_columns=label_columns)
    if manifest.empty:
        return manifest

    manifest = manifest.rename(columns={label_col: "label"})
    manifest = manifest[manifest["label"].notna()]

    counts = manifest["label"].value_counts()
    keep = counts[counts >= min_per_label].index
    dropped = int((~manifest["label"].isin(keep)).sum())
    if dropped:
        logger.info("dropped %d image(s) whose label had fewer than %d examples",
                    dropped, min_per_label)
    manifest = manifest[manifest["label"].isin(keep)]

    if manifest.empty:
        logger.warning("no labels have at least %d images -- nothing to train on",
                       min_per_label)
        return manifest

    manifest = split_dataset(manifest, fractions=fractions, group_col=group_col,
                             stratify_col="label", seed=seed)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    logger.info("prepared %d images across %d labels -> %s",
                len(manifest), manifest["label"].nunique(), output_dir)
    return manifest


def train(manifest, output_dir, backbone=None, loss=None, augmentations=None,
          epochs=30, batch_size=32, embedding_dim=128, seed=0):
    """
    Train an embedding model on a prepared dataset.

    NOT IMPLEMENTED. What's fixed here is the interface and what a caller must
    decide; the training loop itself is deliberately left out rather than
    guessed at.

    manifest      -- DataFrame from prepare_dataset(), with image_path, label,
                    and split columns.
    output_dir    -- where to write the checkpoint and training log. The
                    checkpoint path becomes a BioEncoderModel's `checkpoint`,
                    and therefore part of the recipe hash of every embedding it
                    ever produces -- so write a distinct path per training run
                    rather than overwriting one file, or two different models'
                    embeddings become indistinguishable in the record.
    backbone      -- pretrained feature extractor to fine-tune. The decision
                    that most affects the result and the one most dependent on
                    dataset size: a large backbone on a few thousand images
                    overfits, a small one on a hundred thousand underuses them.
    loss          -- metric-learning loss (triplet, ArcFace, supervised
                    contrastive...). Interacts with batch size, since
                    pair-based losses need enough examples per class IN A BATCH
                    to form informative pairs -- which is why a batch sampler
                    matters here in a way it doesn't for classification.
    augmentations -- training augmentations. Be careful what you make the model
                     invariant to: colour jitter is standard practice and
                     directly destroys the colour signal an entomological
                     embedding probably wants, and horizontal flip is safe for
                     a dorsal view and wrong for anything asymmetric.
    epochs, batch_size, embedding_dim, seed -- the usual.

    Should return the checkpoint path, ready to hand to
    metrics.bioencoder.BioEncoderModel.
    """
    raise NotImplementedError(
        "BioEncoder training isn't implemented -- prepare_dataset() produces a "
        "standard image/label/split manifest, so train with the BioEncoder "
        "package or any metric-learning setup of your choice, then wrap the "
        "checkpoint in metrics.bioencoder.BioEncoderModel to use it as a "
        "metric. See this function's docstring for the decisions to make."
    )


def load(checkpoint, **kwargs):
    """
    Load a trained checkpoint as a BioEncoderModel, ready to pass to
    metrics.bioencoder.embedding().

    NOT IMPLEMENTED, for the same reason as train(): how to load depends on
    what trained it. Construct BioEncoderModel directly with your own loaded
    network -- it only requires an encode(images) -> embeddings method.
    """
    raise NotImplementedError(
        "no loader is implemented -- construct "
        "metrics.bioencoder.BioEncoderModel(model, checkpoint) directly with "
        "your loaded network; it only needs an encode(images) -> embeddings "
        "method"
    )
