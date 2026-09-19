"""
Prepare a dataset for, and train, the UNet++ segmenter in
extensions.smp_segmenter.segmentation.

prepare_dataset() is a thin wrapper over training.splits.split_ids() and
training.datasets.export_training_data() -- real and runnable, like every
other dataset-prep step in this package. train() is a real, opinionated
training loop (ImageNet-pretrained encoder, BCEWithLogitsLoss, Adam) adapted
from a working script, not a stub: unlike
extensions.bioencoder.training's train(), which is deliberately
left unimplemented because a metric-learning backbone/loss/batching strategy
is a dataset-dependent judgment call, binary mask segmentation has a far more
standard shape, so guessing at reasonable defaults here doesn't carry the same
risk of a model that "trains without complaint and embeds badly" (bioencoder's
own reasoning for staying a stub).

    from critterframe.extensions.smp_segmenter import training, segmentation

    manifest = training.prepare_dataset(project_path, "training/aux_v1",
                                        transforms=[cf.remove_background()])
    checkpoint = training.train(manifest, "training/aux_v1")
    training.register_trained(project_path, "aux_segmenter_v1", checkpoint,
                              "training/aux_v1")

Torch, segmentation_models_pytorch, and tqdm are imported lazily inside
train(), so this module -- and prepare_dataset() -- import and run without the
[torch] extra installed; only train() needs it.
"""

import copy
import json
import logging
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score, roc_auc_score

from ...devices import resolve_device
from ...maskops import mask_iou
from ...records.models import register_model
from ...recipes import DEFAULT_PART, hash_spec
from ...training.datasets import DATASET_FILE, export_training_data
from ...training.splits import split_ids
from ...visualization import figures
from ...visualization import pipeline as pipeline_visualization
from ...visualization.panels import annotate, overlay_mask
from .segmentation import DEFAULT_ENCODER, DEFAULT_SIZE

logger = logging.getLogger(__name__)


def prepare_dataset(project_path, output_dir, part=DEFAULT_PART, transforms=(),
                    reference=True, fractions=None, group_col=None,
                    stratify_col=None, subset=None, seed=0, from_part=None,
                    visualize=True):
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
    - `group_col`, `stratify_col` -- passed to
      `training.splits.split_ids()`. `group_col` is the leakage guard --
      several photos of one specimen must land on the same side.
    - `subset` -- restrict to a named subset instead of the whole project.
    - `seed` -- split seed.
    - `from_part` -- build each image from an upstream part's CANONICAL
      mask instead of `part`'s own; see
      `training.datasets.export_training_data`. Pass the same `from_part`
      the `run_segments()` call that made `part`'s mask used, e.g. a part
      carved out of the organism mask, so the exported image matches the
      shared crop this model will see at inference rather than one cropped
      to `part`'s own, usually much smaller, mask.
    - `visualize` -- passed to `split_ids()` and `export_training_data()`.

    Returns the manifest DataFrame export_training_data() wrote.
    """
    splits = split_ids(project_path, subset=subset, fractions=fractions,
                       group_col=group_col, stratify_col=stratify_col,
                       seed=seed, visualize=visualize)
    return export_training_data(project_path, output_dir, splits=splits, part=part,
                                transforms=transforms, reference=reference,
                                masks=True, from_part=from_part, visualize=visualize)


def train(manifest, dataset_dir, encoder_name=DEFAULT_ENCODER, size=DEFAULT_SIZE,
         num_epochs=30, batch_size=6, num_workers=0, lr=1e-4, pos_weight=8.0,
         train_split="train", val_split="val", device=None, seed=0,
         checkpoint_name=None, project_path=None, visualize=True, visualize_every=1):
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
      FINAL epoch's weights instead of the best-val-loss one, and draws
      its prediction grids from the train split instead. Naming a split
      that isn't actually in the manifest still raises, same as
      `train_split` -- only passing None explicitly opts out of validation.
    - `device` -- torch device string; autodetects CUDA if omitted.
    - `seed` -- torch/numpy seed.
    - `checkpoint_name` -- filename the weights are saved under, inside
      `dataset_dir`. None (default) names it
      `smp_segmenter_<encoder>_<today>[_n].pt` -- distinct per run, so
      retraining doesn't silently overwrite a checkpoint
      `records.models.register_model()` already fingerprinted.
    - `project_path` -- project whose `visualizations/pipeline/` the
      training diagnostics are written into. None writes them under
      `dataset_dir` instead.
    - `visualize` -- True (default), an int, or ids: a fixed, seeded sample
      of validation specimens drawn as image | reference | prediction at
      each checkpoint epoch (`__epoch<N>`), plus loss and F1/AUROC/IoU
      curves at the end. False writes nothing.
    - `visualize_every` -- draw the prediction grid every N epochs, and on
      every new best validation loss.

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
    device = resolve_device(device)

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
    best_epoch = None
    history = {f"{phase}_{metric}": [] for phase in phases
              for metric in ("loss", "f1", "auroc", "iou")}

    identity = {"kind": "train_smp_segmenter", "data_hash": _dataset_hash(dataset_dir),
                "encoder_name": encoder_name, "size": size, "num_epochs": num_epochs,
                "batch_size": batch_size, "lr": lr, "pos_weight": pos_weight,
                "train_split": train_split, "val_split": val_split, "seed": seed}
    report = pipeline_visualization.open_report(
        project_path or dataset_dir, f"train__smp_{encoder_name}", hash_spec(identity),
        visualize=visualize, identity=identity)
    shown = rows["val" if "val" in phases else "train"]
    report.begin(shown["occurrence_id"].astype(str))
    shown = shown[shown["occurrence_id"].astype(str).map(report.wants)]

    for epoch in range(num_epochs):
        improved = False
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
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    improved = True
            elif phase == "train":
                # No held-out split to pick a best epoch from -- keep the
                # latest, i.e. the final epoch's weights once the loop ends.
                best_state = copy.deepcopy(model.state_dict())

        due = improved or epoch + 1 == num_epochs or (
            visualize_every and (epoch + 1) % visualize_every == 0)
        if report and due and len(shown):
            _prediction_panels(report, model, shown, dataset_dir, size, device)
            report.checkpoint(f"epoch{epoch + 1:04d}")

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

    if report:
        _training_figures(report, history, phases, best_epoch)
    report.close()

    return checkpoint_path


def register_trained(project_path, name, checkpoint, dataset_dir,
                     encoder_name=DEFAULT_ENCODER, size=DEFAULT_SIZE, **kwargs):
    """
    Register a checkpoint `train()` just wrote, with what
    `segmentation.load_registered()` needs to load it again.

    `records.models.register_model` with this model type's own fields filled
    in: `parameters` carries `encoder_name`/`size`, without which loading
    later falls back to whatever this module's defaults happen to be by then
    -- a checkpoint loaded under a different architecture than it was trained
    with, silently, which is the failure `load_registered` exists to prevent.

    - `project_path` -- project to register in.
    - `name` -- registered model name, e.g. `"wing_segmenter_v1"`.
    - `checkpoint` -- the path `train()` returned.
    - `dataset_dir` -- the dataset `prepare_dataset()` wrote and `train()`
      trained on; its `dataset.json` becomes the model's `training_data`.
    - `encoder_name`, `size` -- what `train()` was given. Pass the same
      values, or the registry records a model that loads wrong.
    - `kwargs` -- anything else `records.models.register_model` takes, e.g.
      `notes=`. `parameters=` is merged with this function's own rather than
      replaced.

    Returns the RegisteredModel.
    """
    parameters = {"encoder_name": encoder_name, "size": size}
    parameters.update(kwargs.pop("parameters", None) or {})
    return register_model(
        project_path, name, path=checkpoint, task="segment", framework="torch",
        base_model=kwargs.pop("base_model", encoder_name),
        training_data=str(dataset_dir), parameters=parameters, **kwargs)


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


def _dataset_hash(dataset_dir):
    """The data_hash from a dataset directory's dataset.json, or None if it has none."""
    record = Path(dataset_dir) / DATASET_FILE
    if not record.exists():
        return None
    with record.open(encoding="utf-8") as handle:
        return json.load(handle).get("data_hash")


def _prediction_panels(report, model, shown, dataset_dir, size, device, threshold=0.5):
    """Image, reference, and prediction panels for the report's sampled specimens."""
    import torch

    model.eval()
    with torch.no_grad():
        for row in shown.itertuples(index=False):
            item = str(row.occurrence_id)
            image = cv2.imread(str(Path(dataset_dir) / row.image_path))
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            reference = cv2.imread(str(Path(dataset_dir) / row.mask_path), cv2.IMREAD_GRAYSCALE)
            reference = cv2.resize(reference, (size, size), interpolation=cv2.INTER_NEAREST) > 0

            tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).float()
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            predicted = torch.sigmoid(model(tensor))[0, 0].cpu().numpy() > threshold

            iou = mask_iou(predicted, reference)

            report.panel(item, "image", image)
            report.panel(item, "reference", overlay_mask(image, reference))
            panel = overlay_mask(image, predicted)
            annotate(panel, f"iou {iou:.2f}")
            report.panel(item, "prediction", panel)


def _training_figures(report, history, phases, best_epoch):
    """Loss and F1/AUROC/IoU curves over epochs, train against val."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    marks = {"best": best_epoch} if best_epoch is not None else None
    report.figure("loss", figures.line_chart(
        {phase: (epochs, history[f"{phase}_loss"]) for phase in phases},
        xlabel="epoch", ylabel="loss", title="loss", marks=marks))
    report.figure("scores", figures.line_chart(
        {f"{phase} {metric}": (epochs, history[f"{phase}_{metric}"])
         for phase in phases for metric in ("f1", "auroc", "iou")},
        xlabel="epoch", ylabel="score", title="F1 / AUROC / IoU", marks=marks))
