"""
Training and fine-tuning a BioEncoder embedding model on a project.

The path this completes: segment organisms, remove their backgrounds, train an
embedding model on the results using the project's own taxonomy as labels, then
plug the trained model back in as an embedding metric
(extensions.bioencoder.embedding). A project ends up producing the
model that measures it.

Why metric learning rather than classification: the useful output isn't "which
species is this" -- the project usually already knows, from the community
identification -- it's a SPACE where similar organisms are close together. That
space keeps working for taxa the model never saw, supports outlier detection
and clustering, and doesn't have to be retrained when a new species appears in
the data. A classifier gives you a label and nothing else.

Dataset preparation and load() are real; train() is a scaffold. Its training
loop is a defined contract with the pieces that need a decision left explicit
(backbone, loss, augmentation). Those are choices that depend on the project's
size and taxonomy, and guessing at them here would produce a model that trains
without complaint and embeds badly, which is worse than a NotImplementedError.
Train with the BioEncoder package instead, then load() its checkpoint.
"""

import logging
import os

import pandas as pd

from ...devices import resolve_device
from ...project import subsets as subset_selection
from ...recipes import DEFAULT_PART
from ...records.occurrences import ID_COL
from ...training.datasets import export_training_data
from ...training.splits import split_ids
from .embedding import DEFAULT_INPUT_SIZE, IMAGENET_MEAN_STD, BioEncoderModel

# Where the BioEncoder package keeps its own root_dir/run_name between calls.
BIOENCODER_SETTINGS = "~/.bioencoder.yaml"

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
                    reference=False, subset=None, limit=None, seed=0, visualize=True):
    """
    Build a training dataset out of a project: image/mask pairs on disk, labels
    attached, split into train/val/test.

    - `project_path` -- project to build from.
    - `output_dir` -- directory to write into.
    - `part` -- part to train on.
    - `transforms` -- operations applied to each segment. Pass the SAME
      chain the embedding metric will use at inference time, typically
      `[remove_background(), crop_to_mask(), orient()]` -- a model trained
      on background-removed, oriented segments and then run on raw
      photographs will embed badly and give no indication of why.
    - `label_col` -- occurrence column holding the label.
    - `group_col` -- occurrence column to group by when splitting, so
      near-duplicates can't straddle train and validation.
    - `min_per_label` -- labels with fewer occurrences than this are dropped,
      before the split and the export, so the dataset record describes what
      was actually written.
    - `fractions` -- split fractions; 70/15/15 by default.
    - `reference` -- train on reference masks rather than canonical ones.
    - `subset`, `limit` -- restrict which occurrences are used.
    - `seed` -- split seed, so the split is reproducible across runs.
    - `visualize` -- True (default): the exported dataset's grid (see
      `export_training_data`), plus a pipeline figure of images per label
      per split for the labels kept. False writes nothing.

    Returns the manifest DataFrame, with `split` and `label` columns added.
    """
    # Labels are filtered and split BEFORE anything is exported, so the
    # dataset.json export_training_data writes describes the images that are
    # actually in the directory. Filtering afterwards (and rewriting only
    # manifest.csv) left its data_hash covering a population the model never
    # saw, which is what a registered model's training_data then pointed at.
    columns = [column for column in {label_col, group_col} if column]
    occurrences = subset_selection.select_occurrences(project_path, subset=subset,
                                                       columns=columns)
    labelled = occurrences[occurrences[label_col].notna()]

    counts = labelled[label_col].value_counts()
    keep = counts[counts >= min_per_label].index
    dropped = int((~labelled[label_col].isin(keep)).sum())
    if dropped:
        logger.info("dropped %d occurrence(s) whose label had fewer than %d examples",
                    dropped, min_per_label)
    labelled = labelled[labelled[label_col].isin(keep)]

    if labelled.empty:
        logger.warning("no labels have at least %d occurrence(s) -- nothing to train on",
                       min_per_label)
        return pd.DataFrame()

    occurrence_ids = labelled[ID_COL].tolist()
    if limit is not None:
        occurrence_ids = occurrence_ids[:limit]

    splits = split_ids(project_path, occurrence_ids=occurrence_ids,
                       fractions=fractions, stratify_col=label_col,
                       group_col=group_col, seed=seed, visualize=visualize)

    manifest = export_training_data(project_path, output_dir, splits=splits, part=part,
                                    transforms=transforms, reference=reference,
                                    masks=True, metadata=columns, visualize=visualize)
    if manifest.empty:
        return manifest

    # `label` is this extension's own name for the training target, and what
    # train() reads; the export carries the occurrence column's own name.
    manifest = manifest.rename(columns={label_col: "label"})
    manifest.to_csv(os.path.join(output_dir, "manifest.csv"), index=False)

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

    - `manifest` -- DataFrame from `prepare_dataset()`, with `image_path`,
      `label`, and `split` columns.
    - `output_dir` -- where to write the checkpoint and training log. The
      checkpoint path becomes a `BioEncoderModel`'s `checkpoint`, and
      therefore part of the recipe hash of every embedding it ever
      produces -- so write a distinct path per training run rather than
      overwriting one file, or two different models' embeddings become
      indistinguishable in the record. `records.models.register_model()`
      removes that trap by hashing the weights themselves, and is also
      where the training data behind the checkpoint gets recorded.
    - `backbone` -- pretrained feature extractor to fine-tune. The decision
      that most affects the result and the one most dependent on dataset
      size: a large backbone on a few thousand images overfits, a small
      one on a hundred thousand underuses them.
    - `loss` -- metric-learning loss (triplet, ArcFace, supervised
      contrastive...). Interacts with batch size, since pair-based losses
      need enough examples per class IN A BATCH to form informative pairs
      -- which is why a batch sampler matters here in a way it doesn't for
      classification.
    - `augmentations` -- training augmentations. Be careful what you make
      the model invariant to: colour jitter is standard practice and
      directly destroys the colour signal an entomological embedding
      probably wants, and horizontal flip is safe for a dorsal view and
      wrong for anything asymmetric.
    - `epochs`, `batch_size`, `embedding_dim`, `seed` -- the usual.

    Should return the checkpoint path, ready to hand to
    embedding.BioEncoderModel.
    """
    raise NotImplementedError(
        "BioEncoder training isn't implemented -- prepare_dataset() produces a "
        "standard image/label/split manifest, so train with the BioEncoder "
        "package (then load() its checkpoint) or any metric-learning setup of "
        "your choice (then wrap it in embedding.BioEncoderModel) to use it as "
        "a metric. See this function's docstring for the decisions to make."
    )


def load(checkpoint, backbone, projection_head=True, input_size=DEFAULT_INPUT_SIZE,
         normalize=True, mean=IMAGENET_MEAN_STD[0], std=IMAGENET_MEAN_STD[1],
         device=None):
    """
    Load a stage-one checkpoint written by the BioEncoder package as a
    BioEncoderModel, ready to pass to `embedding.embedding()`.

    Loads strictly, so a `backbone` that doesn't match the checkpoint raises
    instead of leaving most of the network at its initial weights. Needs the
    `bioencoder` package; a checkpoint from anything else goes straight into
    `embedding.BioEncoderModel` with your own loaded network.

    - `checkpoint` -- a weights file under `weights/<run>/first/`, usually
      `swa`.
    - `backbone` -- the `model.backbone` the training config named, e.g.
      `"timm_resnet18"`.
    - `projection_head` -- True embeds with the trained projection head (the
      space the contrastive loss shaped); False with the backbone's features
      before it.
    - `input_size` -- (height, width); the config's `img_size`, both ways.
    - `normalize` -- L2-normalize embeddings before storing.
    - `mean`, `std` -- input standardization; ImageNet's by default, which
      BioEncoder's `Normalize()` augmentation applies in training.
    - `device` -- torch device string; autodetects CUDA if omitted.

    Returns a BioEncoderModel.
    """
    import torch
    from bioencoder.core.models import BioEncoderModel as BioEncoderNetwork

    device = resolve_device(device)
    # Building the encoder fetches its ImageNet weights (cached after the
    # first time) before the checkpoint overwrites them.
    network = BioEncoderNetwork(backbone=backbone)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    # strict: BioEncoder's own build_model loads with strict=False, which
    # turns a mismatched backbone into a silently mostly-random network.
    network.load_state_dict(state, strict=True)
    network.use_projection_head(projection_head)

    return BioEncoderModel(network, checkpoint, input_size=input_size,
                           normalize=normalize, device=device, mean=mean, std=std,
                           head="projection" if projection_head else "features")


def load_from_config(config_path, root_dir=None, run_name=None, checkpoint=None,
                     projection_head=True, normalize=True, device=None):
    """
    Load the checkpoint a BioEncoder inference/training config names, as a BioEncoderModel.

    Reads the same keys `bioencoder_inference` does -- `model.backbone`,
    `img_size`, `model.checkpoint`/`model.checkpoint_path`, `model.stage` -- so
    a project already driving BioEncoder by config needs no second description
    of its model. Only first-stage (embedding) checkpoints are supported.

    - `config_path` -- BioEncoder YAML config.
    - `root_dir`, `run_name` -- BioEncoder's own; read from `~/.bioencoder.yaml`
      (written by `bioencoder_configure`) when omitted. Unused when the config
      or `checkpoint` gives a path.
    - `checkpoint` -- explicit weights path, overriding the config.
    - `projection_head`, `normalize`, `device` -- as `load()`.

    Returns a BioEncoderModel.
    """
    import yaml

    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    model_config = config.get("model") or {}

    stage = model_config.get("stage", "first")
    if stage != "first":
        raise ValueError(
            f"{config_path} names a stage-{stage!r} model; an embedding needs "
            "the first-stage (metric-learning) checkpoint")
    backbone = model_config.get("backbone")
    if not backbone:
        raise ValueError(f"{config_path} has no model.backbone")

    checkpoint = checkpoint or model_config.get("checkpoint_path")
    if not checkpoint:
        if root_dir is None or run_name is None:
            settings_path = os.path.expanduser(BIOENCODER_SETTINGS)
            settings = {}
            if os.path.isfile(settings_path):
                with open(settings_path, encoding="utf-8") as handle:
                    settings = yaml.safe_load(handle) or {}
            root_dir = root_dir or settings.get("root_dir")
            run_name = run_name or settings.get("run_name")
        if root_dir is None or run_name is None:
            raise ValueError(
                "no checkpoint path in the config, and no root_dir/run_name "
                "given or configured -- pass checkpoint= or root_dir= and run_name=")
        checkpoint = os.path.join(root_dir, "weights", run_name, stage,
                                  model_config.get("checkpoint", "swa"))

    size = int(config.get("img_size", DEFAULT_INPUT_SIZE[0]))
    return load(checkpoint, backbone, projection_head=projection_head,
                input_size=(size, size), normalize=normalize, device=device)
