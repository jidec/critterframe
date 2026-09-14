"""
Prepare a dataset for, and train, the UNet++ segmenter in
extensions.smp_segmenter.segmentation.

prepare_dataset() is a thin wrapper over training.splits.split_ids() and
training.datasets.export_training_data() -- real and runnable, like every
other dataset-prep step in this package. train() is a real, opinionated
training loop (ImageNet-pretrained encoder, BCEWithLogitsLoss, Adam) adapted
from a working script, not a stub: unlike
extensions.inat_insects.training.bioencoder's train(), which is deliberately
left unimplemented because a metric-learning backbone/loss/batching strategy
is a dataset-dependent judgment call, binary mask segmentation has a far more
standard shape, so guessing at reasonable defaults here doesn't carry the same
risk of a model that "trains without complaint and embeds badly" (bioencoder's
own reasoning for staying a stub).

    from critterframe.extensions.smp_segmenter import training, segmentation

    manifest = training.prepare_dataset(project_path, "training/aux_v1",
                                        transforms=[cf.remove_background()])
    checkpoint = training.train(manifest, "training/aux_v1")
    cf.register_model(project_path, "aux_segmenter_v1", path=checkpoint,
                      task="segment", framework="torch",
                      base_model=segmentation.DEFAULT_ENCODER,
                      training_data="training/aux_v1")

Torch, segmentation_models_pytorch, and tqdm are imported lazily inside
train(), so this module -- and prepare_dataset() -- import and run without the
[torch] extra installed; only train() needs it.
"""

import copy
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score, roc_auc_score

from ...recipes import DEFAULT_PART
from ...training.datasets import export_training_data
from ...training.splits import DEFAULT_FRACTIONS, split_ids
from .segmentation import DEFAULT_ENCODER, DEFAULT_SIZE

logger = logging.getLogger(__name__)


def prepare_dataset(project_path, output_dir, part=DEFAULT_PART, transforms=(),
                    reference=True, fractions=None, group_by=None,
                    stratify_by=None, subset=None, seed=0, from_part=None):
    """
    Export a train/val/test image+mask dataset from a project's own masks.

    reference defaults to True, unlike export_training_data()'s own default:
    the whole reason to train a model here is that the automated masks weren't
    good enough, so training on them would teach this model to reproduce the
    old one's mistakes.

    - `project_path` -- project to export from.
    - `output_dir` -- directory the dataset is written into; see
      `training.datasets.export_training_data` for the exact layout
      (`<split>/images/<id>.png`, `<split>/masks/<id>.png`).
    - `part` -- occurrence-part to export masks for.
    - `transforms` -- the SAME chain this model will see at inference time,
      e.g. `[remove_background()]` -- a mismatch between how training
      images were framed and how inference images arrive is the single
      most common way a segmenter trains cleanly and performs badly.
    - `fractions` -- `{split name: fraction}`, defaulting to
      `training.splits`' own 70/15/15 train/val/test. Any names `train()`
      is told to use -- but keep a THIRD split `test()` never touches:
      `train()` already uses val to pick the best epoch, so scoring a
      model against val too is grading it on data it (indirectly) already
      saw. A held-out test split is what `validation.masks.validate_masks`
      needs for an honest number. Leave `"val"` out entirely for a two-way
      train/test split too small to spare a third slice, and pass
      `train(val_split=None)` to match -- not a `"val": 0.0` entry, which
      would still come back as its own (empty) split and log a warning
      about it.
    - `group_by`, `stratify_by` -- passed to `training.splits.split_ids()`.
      `group_by` is the leakage guard -- several photos of one specimen
      must land on the same side.
    - `subset` -- restrict to a named subset instead of the whole project.
    - `seed` -- split seed.
    - `from_part` -- build each image from an upstream part's CANONICAL
      mask instead of `part`'s own; see
      `training.datasets.export_training_data`. Pass the same `from_part`
      the `run_segments()` call that made `part`'s mask used, e.g. a part
      carved out of the organism mask, so the exported image matches the
      shared crop this model will see at inference rather than one cropped
      to `part`'s own, usually much smaller, mask.

    Returns the manifest DataFrame export_training_data() wrote.
    """
    splits = split_ids(project_path, subset=subset,
                       proportions=fractions or DEFAULT_FRACTIONS,
                       group_by=group_by, stratify_by=stratify_by, seed=seed)
    return export_training_data(project_path, output_dir, splits=splits, part=part,
                                transforms=transforms, reference=reference,
                                masks=True, from_part=from_part)


def train(manifest, dataset_dir, encoder_name=DEFAULT_ENCODER, size=DEFAULT_SIZE,
         num_epochs=30, batch_size=6, num_workers=0, lr=1e-4, pos_weight=8.0,
         train_split="train", val_split="val", device=None, seed=0,
         show=False, checkpoint_name=None):
    """
    Train a UNet++/`encoder_name` segmenter on a dataset prepare_dataset()
    wrote, and save the best epoch's weights.

    ImageNet-pretrained encoder, BCEWithLogitsLoss(pos_weight=pos_weight) --
    the organism is usually the minority class in a crop -- and Adam, tracking
    F1/AUROC/IoU each epoch in probability space (sigmoid applied before
    thresholding; the loss itself trains on raw logits, which is what
    BCEWithLogitsLoss expects). Keeps the epoch with the best validation loss
    rather than the last one -- unless there is no validation split, see
    val_split.

    - `manifest` -- the DataFrame `prepare_dataset()` returned, or a path to
      its `manifest.csv`.
    - `dataset_dir` -- the directory `prepare_dataset()` wrote into --
      image and mask paths in the manifest are relative to this.
    - `encoder_name` -- `segmentation_models_pytorch` encoder. Must be
      passed again to `segmentation.smp_segmenter()` at inference time.
    - `size` -- side length images/masks are resized to for training; must
      match `segmentation.smp_segmenter()`'s size at inference time.
    - `num_epochs`, `batch_size`, `num_workers`, `lr` -- the usual.
    - `pos_weight` -- BCEWithLogitsLoss's positive-class weight; raise it
      if the organism covers a small fraction of most frames.
    - `train_split` -- which of `prepare_dataset()`'s splits to train on.
    - `val_split` -- which split to validate on, or None to train with no
      validation at all -- for a reference set too small to spare a third
      split. None trains for every epoch with no early stopping, saves the
      FINAL epoch's weights instead of the best-val-loss one, and skips
      the `show=True` sample-prediction plot (nothing held out to draw
      from). Naming a split that isn't actually in the manifest still
      raises, same as `train_split` -- only passing None explicitly opts
      out of validation.
    - `device` -- torch device string; autodetects CUDA if omitted.
    - `seed` -- torch/numpy seed.
    - `show` -- plot loss/F1/AUROC/IoU curves and a few validation
      predictions with matplotlib as training goes. Needs matplotlib
      installed; off by default so training never requires a display.
    - `checkpoint_name` -- filename the weights are saved under, inside
      `dataset_dir`. None (default) names it
      `smp_segmenter_<encoder>_<today>[_n].pt` -- distinct per run, so
      retraining doesn't silently overwrite a checkpoint
      `records.models.register_model()` already fingerprinted.

    Returns the checkpoint path, absolute, ready to hand to
    records.models.register_model() and then
    segmentation.smp_segmenter(checkpoint=...).
    """
    import segmentation_models_pytorch as smp
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    if isinstance(manifest, (str, Path)):
        manifest = pd.read_csv(manifest)
    if "split" not in manifest.columns:
        raise ValueError(
            "manifest has no 'split' column -- prepare_dataset() always "
            "passes splits= to export_training_data(), so this manifest "
            "wasn't written by it"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # val_split=None is the only way to skip validation -- an absent "val"
    # split with the default val_split="val" still raises below, the same
    # protection train_split gets against a typo'd or missing name.
    phase_splits = [("train", train_split)]
    if val_split is not None:
        phase_splits.append(("val", val_split))
    phases = tuple(name for name, _split_name in phase_splits)
    if val_split is None:
        logger.warning("training with no validation split -- no early "
                       "stopping; the final epoch's weights will be saved")

    rows = {}
    for phase, split_name in phase_splits:
        phase_rows = manifest[manifest["split"] == split_name]
        missing_mask = phase_rows["mask_path"].isna().sum()
        if missing_mask:
            logger.warning("dropping %d '%s' row(s) with no mask", missing_mask, split_name)
            phase_rows = phase_rows.dropna(subset=["mask_path"])
        if phase_rows.empty:
            raise ValueError(
                f"no usable rows with split=={split_name!r} in the manifest "
                f"(splits present: {sorted(manifest['split'].unique())})"
            )
        rows[phase] = phase_rows.reset_index(drop=True)

    class _MaskDataset(Dataset):
        """One (image, mask) tensor pair per manifest row, resized the same
        way SMPSegmenter.predict() resizes at inference."""

        def __init__(self, frame):
            self.frame = frame

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, index):
            import cv2

            row = self.frame.iloc[index]
            image = cv2.imread(str(Path(dataset_dir) / row["image_path"]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

            mask = cv2.imread(str(Path(dataset_dir) / row["mask_path"]),
                              cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy((mask > 0).astype("float32")).unsqueeze(0)

            return image, mask

    datasets = {phase: _MaskDataset(frame) for phase, frame in rows.items()}
    loaders = {
        phase: DataLoader(dataset, batch_size=batch_size,
                          shuffle=(phase == "train"), num_workers=num_workers)
        for phase, dataset in datasets.items()
    }

    model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights="imagenet",
                             in_channels=3, classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    history = {f"{phase}_{metric}": [] for phase in phases
              for metric in ("loss", "f1", "auroc", "iou")}

    for epoch in range(num_epochs):
        for phase in phases:
            model.train(phase == "train")
            running_loss = 0.0
            y_true, y_pred = [], []

            for images, masks in tqdm(loaders[phase],
                                      desc=f"epoch {epoch + 1}/{num_epochs} {phase}"):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    logits = model(images)
                    loss = criterion(logits, masks)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                # sigmoid before recording -- BCEWithLogitsLoss trains on raw
                # logits, but F1/threshold need probabilities, not logits.
                probs = torch.sigmoid(logits)
                y_true.append(masks.detach().cpu().numpy().ravel())
                y_pred.append(probs.detach().cpu().numpy().ravel())

            epoch_loss = running_loss / len(datasets[phase])
            y_true_arr = np.concatenate(y_true) > 0.5
            y_pred_arr = np.concatenate(y_pred)
            f1 = f1_score(y_true_arr, y_pred_arr > 0.5)
            iou = jaccard_score(y_true_arr, y_pred_arr > 0.5)
            try:
                auroc = roc_auc_score(y_true_arr, y_pred_arr)
            except ValueError:
                auroc = float("nan")   # a batch/epoch with only one class present

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_f1"].append(f1)
            history[f"{phase}_auroc"].append(auroc)
            history[f"{phase}_iou"].append(iou)
            logger.info("epoch %d %s: loss=%.4f f1=%.4f auroc=%.4f iou=%.4f",
                       epoch + 1, phase, epoch_loss, f1, auroc, iou)

            if "val" in phases:
                if phase == "val" and epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_state = copy.deepcopy(model.state_dict())
                    if show:
                        _plot_sample_predictions(model, loaders["val"], device, epoch + 1)
            elif phase == "train":
                # No held-out split to pick a best epoch from -- keep the
                # latest, i.e. the final epoch's weights once the loop ends.
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    checkpoint_path = (Path(dataset_dir) / checkpoint_name if checkpoint_name
                       else _default_checkpoint_path(dataset_dir, encoder_name))
    torch.save(model.state_dict(), checkpoint_path)
    # Absolute, not whatever form dataset_dir arrived in: register_model()
    # treats a relative path as relative to the PROJECT directory, not to the
    # caller's cwd, and dataset_dir is very often itself already
    # project-relative (e.g. "<project>/training/head") -- returning that
    # as-is would have register_model() double the project prefix.
    checkpoint_path = checkpoint_path.resolve()
    if "val" in phases:
        logger.info("saved checkpoint -> %s (best val loss %.4f)",
                   checkpoint_path, best_val_loss)
    else:
        logger.info("saved checkpoint -> %s (final epoch, no held-out val split)",
                   checkpoint_path)

    if show:
        _plot_training_metrics(history)

    return checkpoint_path


def _default_checkpoint_path(dataset_dir, encoder_name):
    """
    `smp_segmenter_<encoder>_<today>[_n].pt`.

    The `_n` suffix stops a same-day rerun silently overwriting an earlier
    checkpoint -- and going stale against whatever fingerprint
    records.models.register_model() already recorded for it.
    """
    directory = Path(dataset_dir)
    base = f"smp_segmenter_{encoder_name}_{date.today().isoformat()}"
    dest = directory / f"{base}.pt"
    n = 1
    while dest.exists():
        dest = directory / f"{base}_{n}.pt"
        n += 1
    return dest


def _plot_sample_predictions(model, loader, device, epoch, threshold=0.5, num_samples=3):
    """A few validation images with their ground-truth and predicted masks -- what show=True is for."""
    import matplotlib.pyplot as plt
    import torch

    model.eval()
    with torch.no_grad():
        images, true_masks = next(iter(loader))
        images, true_masks = images.to(device), true_masks.to(device)
        probs = torch.sigmoid(model(images))

        indices = np.random.choice(range(images.size(0)),
                                   min(num_samples, images.size(0)), replace=False)
        plt.figure(figsize=(12, 4 * len(indices)))
        plt.suptitle(f"epoch {epoch}", fontsize=16)
        for row, index in enumerate(indices):
            image = images[index].cpu().permute(1, 2, 0).numpy()
            true_mask = true_masks[index].cpu().squeeze().numpy()
            pred_mask = (probs[index].cpu().squeeze().numpy() > threshold).astype("uint8")

            for column, (panel, title) in enumerate(
                    ((image, "image"), (true_mask, "reference mask"),
                    (pred_mask, f"predicted (t={threshold})"))):
                plt.subplot(len(indices), 3, row * 3 + column + 1)
                plt.imshow(panel, cmap=None if column == 0 else "gray")
                plt.title(title)
                plt.axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def _plot_training_metrics(history):
    """
    Loss/F1/AUROC/IoU over epochs, train vs. val -- what show=True is for.

    Plots train alone when history has no val_* keys, i.e. train(val_split=None).
    """
    import matplotlib.pyplot as plt

    has_val = "val_loss" in history
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    if has_val:
        plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("loss")

    plt.subplot(1, 2, 2)
    for metric in ("f1", "auroc", "iou"):
        plt.plot(epochs, history[f"train_{metric}"], "--", label=f"train {metric}")
        if has_val:
            plt.plot(epochs, history[f"val_{metric}"], label=f"val {metric}")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.legend()
    plt.title("F1 / AUROC / IoU")

    plt.tight_layout()
    plt.show()
