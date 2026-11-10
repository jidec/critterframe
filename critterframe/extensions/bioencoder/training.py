"""BioEncoder training support: prepare_dataset(), register_trained(), load() and load_from_config()."""

import logging
import os

import pandas as pd

from ...core.recipes import DEFAULT_PART
from ...records.models import register_model
from ...records.occurrences import ID_COL
from ...selection import subsets as subset_selection
from ...selection.algorithms import cap_per_group
from ...selection.queries import ids_with_mask
from ...training.datasets import export_training_data
from ...training.splits import split_ids
from .embedding import DEFAULT_INPUT_SIZE, IMAGENET_MEAN_STD, BioEncoderModel, head_label

# Where the BioEncoder package keeps its own root_dir/run_name between calls.
BIOENCODER_SETTINGS = "~/.bioencoder.yaml"

logger = logging.getLogger(__name__)

# The fewest images a label needs; bioencoder_split_dataset's own min_per_class.
MIN_IMAGES_PER_LABEL = 20

# The two splits BioEncoder reads as ImageFolders, which must hold the same classes.
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"


def prepare_dataset(
    project_path,
    output_dir,
    label_col,
    group_col,
    part=DEFAULT_PART,
    transforms=(),
    min_per_label=MIN_IMAGES_PER_LABEL,
    max_per_label=None,
    fractions=None,
    reference=False,
    subset=None,
    seed=0,
    visualize=True,
):
    """Write a BioEncoder training dataset, `<split>/<label>/<occurrence_id>.png`, split by group.

    A label missing from either the train or the val split is dropped from every split.
    `output_dir` must be empty or absent.

    Args:
        project_path: Project to build from.
        output_dir: BioEncoder's `<root_dir>/data/<run_name>`.
        label_col: Occurrence column holding the label, e.g. `"species"`.
        group_col: Occurrence column whose members must stay in one split, e.g. `"recordedBy"`;
            None splits without grouping.
        part: Part to train on.
        transforms: Operations applied to each segment; pass the chain the embedding
            metric will use at inference.
        min_per_label: Labels with fewer occurrences than this, after the cap, are dropped.
        max_per_label: Cap on occurrences per label, chosen at random with `seed`; None keeps all.
        fractions: Split fractions, including `"train"` and `"val"`; 70/15/15 train/val/test if None.
        reference: Frame segments by reference masks.
        subset: Named subset to use.
        seed: Seed for the cap and the split.
        visualize: Write the split's figure and the exported dataset's grid.

    Returns:
        The manifest DataFrame, with `split` and `class` columns.

    Raises:
        RuntimeError: If occurrences failing to export leave train and val with different labels.
    """
    if fractions is not None and not {TRAIN_SPLIT, VAL_SPLIT} <= set(fractions):
        raise ValueError(f"fractions must include '{TRAIN_SPLIT}' and '{VAL_SPLIT}', got {sorted(fractions)}")
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise ValueError(
            f"{output_dir} is not empty -- clear it first, or an image left from an earlier "
            "export would be trained on as though it were selected"
        )

    columns = [column for column in (label_col, group_col) if column]
    occurrences = subset_selection.select_occurrences(project_path, subset=subset, columns=columns)
    with_mask = set(ids_with_mask(project_path, part, reference=reference))
    labels = occurrences[label_col]
    occurrences = occurrences[
        occurrences[ID_COL].isin(with_mask) & labels.notna() & (labels.astype(str).str.strip() != "")
    ]
    if max_per_label is not None:
        occurrences = cap_per_group(occurrences, label_col, max_per_label, seed=seed)

    counts = occurrences[label_col].value_counts()
    occurrences = occurrences[occurrences[label_col].isin(counts[counts >= min_per_label].index)]
    if occurrences.empty:
        logger.warning("no labels have at least %d occurrence(s) -- nothing to train on", min_per_label)
        return pd.DataFrame()
    logger.info(
        "%d occurrence(s) across %d label(s) with at least %d each",
        len(occurrences),
        occurrences[label_col].nunique(),
        min_per_label,
    )

    splits = split_ids(
        project_path,
        occurrence_ids=occurrences[ID_COL].tolist(),
        fractions=fractions,
        stratify_col=label_col,
        group_col=group_col,
        seed=seed,
        visualize=visualize,
    )

    # Filtered before the export, so dataset.json describes the images on disk.
    label_of = dict(zip(occurrences[ID_COL], occurrences[label_col]))
    shared = {label_of[i] for i in splits[TRAIN_SPLIT]} & {label_of[i] for i in splits[VAL_SPLIT]}
    dropped = occurrences[label_col].nunique() - len(shared)
    if dropped:
        logger.warning(
            "dropping %d label(s) not present in both '%s' and '%s'", dropped, TRAIN_SPLIT, VAL_SPLIT
        )
    splits = {name: [i for i in ids if label_of[i] in shared] for name, ids in splits.items()}

    manifest = export_training_data(
        project_path,
        output_dir,
        splits=splits,
        part=part,
        transforms=transforms,
        reference=reference,
        class_by=label_col,
        metadata=[group_col] if group_col else None,
        visualize=visualize,
    )
    if manifest.empty:
        return manifest

    exported = {
        name: set(manifest.loc[manifest["split"] == name, "class"]) for name in (TRAIN_SPLIT, VAL_SPLIT)
    }
    if exported[TRAIN_SPLIT] != exported[VAL_SPLIT]:
        raise RuntimeError(
            f"'{TRAIN_SPLIT}' and '{VAL_SPLIT}' hold different labels after export "
            f"({TRAIN_SPLIT} only: {sorted(exported[TRAIN_SPLIT] - exported[VAL_SPLIT])}, "
            f"{VAL_SPLIT} only: {sorted(exported[VAL_SPLIT] - exported[TRAIN_SPLIT])}) -- BioEncoder "
            "would misalign them; see the export log for occurrences that failed"
        )

    logger.info(
        "prepared %d images across %d labels -> %s", len(manifest), len(exported[TRAIN_SPLIT]), output_dir
    )
    return manifest


def register_trained(
    project_path,
    name,
    checkpoint,
    dataset_dir,
    backbone,
    input_size,
    projection_head=True,
    normalize=True,
    **kwargs,
):
    """Register a BioEncoder checkpoint with what `embedding.load_registered()` needs to load it.

    These settings are not in the registered model's identity, so register changed ones under a new name.

    Args:
        project_path: Project to register in.
        name: Registered model name, e.g. `"bioencoder_odonata_v1"`.
        checkpoint: The weights file, usually `weights/<run>/first/swa`.
        dataset_dir: The directory `prepare_dataset()` wrote; its `dataset.json` becomes `training_data`.
        backbone: As in `load`.
        input_size: As in `load`.
        projection_head: As in `load`.
        normalize: As in `load`.
        **kwargs: Passed to `records.models.register_model`; `parameters=` is merged with
            this function's own.

    Returns:
        The `RegisteredModel`.
    """
    parameters = {
        "backbone": backbone,
        "input_size": list(input_size),
        "projection_head": projection_head,
        "normalize": normalize,
    }
    parameters.update(kwargs.pop("parameters", None) or {})
    return register_model(
        project_path,
        name,
        path=checkpoint,
        task="embedding",
        framework="torch",
        base_model=kwargs.pop("base_model", backbone),
        training_data=str(dataset_dir),
        parameters=parameters,
        **kwargs,
    )


def train(
    manifest,
    output_dir,
    backbone=None,
    loss=None,
    augmentations=None,
    epochs=30,
    batch_size=32,
    embedding_dim=128,
    seed=0,
):
    """Train an embedding model on a prepared dataset. Not implemented.

    Raises `NotImplementedError`: train with the BioEncoder package on what `prepare_dataset()`
    wrote, then `register_trained()` its checkpoint. The arguments name what a caller has to decide.

    Args:
        manifest: DataFrame from `prepare_dataset()`, with `image_path`, `class` and `split`.
        output_dir: Where to write the checkpoint and log; use a distinct path per run.
        backbone: Pretrained feature extractor to fine-tune; size it to the dataset.
        loss: Metric-learning loss, e.g. triplet, ArcFace, supervised contrastive. Pair-based
            losses need enough examples per class in each batch.
        augmentations: Training augmentations. Each makes the model invariant to
            something: color jitter removes color signal, and a horizontal flip is wrong
            for an asymmetric view.
        epochs: Training epochs.
        batch_size: Batch size.
        embedding_dim: Embedding dimensions.
        seed: Random seed.
    """
    raise NotImplementedError(
        "BioEncoder training isn't implemented -- prepare_dataset() writes BioEncoder's own "
        "<split>/<class>/ layout, so train with the BioEncoder package, then register_trained() "
        "its checkpoint and embed with embedding.load_registered(), or wrap a network of your "
        "own in embedding.BioEncoderModel. See this function's docstring for the decisions to make."
    )


def load(
    checkpoint,
    backbone,
    projection_head=True,
    input_size=DEFAULT_INPUT_SIZE,
    normalize=True,
    mean=IMAGENET_MEAN_STD[0],
    std=IMAGENET_MEAN_STD[1],
    device=None,
):
    """Load a stage-one checkpoint written by the BioEncoder package, as a `BioEncoderModel`.

    Loads strictly, so a `backbone` that doesn't match the checkpoint raises. Needs the
    `bioencoder` package.

    Args:
        checkpoint: A weights file under `weights/<run>/first/`, usually `swa`.
        backbone: The `model.backbone` the training config named, e.g. `"timm_resnet18"`.
        projection_head: Embed with the trained projection head; False uses the
            backbone's features before it.
        input_size: `(height, width)`: the config's `img_size`, both ways.
        normalize: L2-normalize embeddings before storing.
        mean: Input standardization mean; ImageNet's by default.
        std: Input standardization standard deviation; ImageNet's by default.
        device: Torch device string; CUDA where available if None, resolved on first use.
    """
    import torch
    from bioencoder.core.models import BioEncoderModel as BioEncoderNetwork

    # Building the encoder fetches its ImageNet weights (cached after the
    # first time) before the checkpoint overwrites them.
    network = BioEncoderNetwork(backbone=backbone)
    # Onto the CPU: the device is chosen when the model first embeds.
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    # strict: BioEncoder's own build_model loads with strict=False, which
    # turns a mismatched backbone into a silently mostly-random network.
    network.load_state_dict(state, strict=True)
    network.use_projection_head(projection_head)

    return BioEncoderModel(
        network,
        checkpoint,
        input_size=input_size,
        normalize=normalize,
        device=device,
        mean=mean,
        std=std,
        head=head_label(projection_head),
    )


def load_from_config(
    config_path,
    root_dir=None,
    run_name=None,
    checkpoint=None,
    projection_head=True,
    normalize=True,
    device=None,
):
    """Load the checkpoint a BioEncoder config names, as a `BioEncoderModel`.

    Reads the keys `bioencoder_inference` does: `model.backbone`, `img_size`,
    `model.checkpoint` or `model.checkpoint_path`, and `model.stage`. Only first-stage
    checkpoints are supported.

    Args:
        config_path: BioEncoder YAML config.
        root_dir: BioEncoder's root directory; from `~/.bioencoder.yaml` if None.
        run_name: BioEncoder's run name; from `~/.bioencoder.yaml` if None.
        checkpoint: Explicit weights path, overriding the config.
        projection_head: As in `load`.
        normalize: As in `load`.
        device: As in `load`.
    """
    import yaml

    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    model_config = config.get("model") or {}

    stage = model_config.get("stage", "first")
    if stage != "first":
        raise ValueError(
            f"{config_path} names a stage-{stage!r} model; an embedding needs "
            "the first-stage (metric-learning) checkpoint"
        )
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
                "given or configured -- pass checkpoint= or root_dir= and run_name="
            )
        checkpoint = os.path.join(root_dir, "weights", run_name, stage, model_config.get("checkpoint", "swa"))

    size = int(config.get("img_size", DEFAULT_INPUT_SIZE[0]))
    return load(
        checkpoint,
        backbone,
        projection_head=projection_head,
        input_size=(size, size),
        normalize=normalize,
        device=device,
    )
