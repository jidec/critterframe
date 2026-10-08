"""Prepare a dataset for, train and register the UNet++ segmenter."""

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


def prepare_dataset(
    project_path,
    output_dir,
    part=DEFAULT_PART,
    transforms=(),
    reference=True,
    fractions=None,
    group_col=None,
    stratify_col=None,
    subset=None,
    seed=0,
    from_part=None,
    visualize=True,
):
    """Export an image and mask dataset, split, from a project's own masks.

    Args:
        project_path: Project to export from.
        output_dir: Directory to write into, laid out as `export_training_data` describes.
        part: Part to export masks for.
        transforms: Operations applied to each segment; pass the chain the model will
            get at inference.
        reference: Read masks from the reference table. True by default here.
        fractions: `{split name: fraction}`; 70/15/15 train/val/test if None. Leave `"val"`
            out for a two-way split, and pass `train(val_split=None)` to match.
        group_col: Occurrence column whose members must stay in one split.
        stratify_col: Occurrence column whose distribution each split should preserve.
        subset: Named subset to export.
        seed: Split seed.
        from_part: Frame each image by this upstream part's canonical mask, as the
            `run_segments` call that made `part` did.
        visualize: Passed to `split_ids` and `export_training_data`.

    Returns:
        The manifest DataFrame.
    """
    splits = split_ids(
        project_path,
        subset=subset,
        fractions=fractions,
        group_col=group_col,
        stratify_col=stratify_col,
        seed=seed,
        visualize=visualize,
    )
    return export_training_data(
        project_path,
        output_dir,
        splits=splits,
        part=part,
        transforms=transforms,
        reference=reference,
        masks=True,
        from_part=from_part,
        visualize=visualize,
    )


def train(
    manifest,
    dataset_dir,
    encoder_name=DEFAULT_ENCODER,
    size=DEFAULT_SIZE,
    num_epochs=30,
    batch_size=6,
    num_workers=0,
    lr=1e-4,
    pos_weight=8.0,
    train_split="train",
    val_split="val",
    device=None,
    seed=0,
    checkpoint_name=None,
    project_path=None,
    visualize=True,
    visualize_every=1,
):
    """Train a UNet++ segmenter on a dataset `prepare_dataset()` wrote, and save its weights.

    ImageNet-pretrained encoder, `BCEWithLogitsLoss` and Adam. The epoch with the best
    validation loss is saved; with no validation split, the last one.

    Args:
        manifest: The DataFrame `prepare_dataset()` returned, or a path to its `manifest.csv`.
        dataset_dir: The directory `prepare_dataset()` wrote into.
        encoder_name: The segmentation_models_pytorch encoder; needed again at inference.
        size: Side length images and masks are resized to; needed again at inference.
        num_epochs: Training epochs.
        batch_size: Batch size.
        num_workers: Data loader workers.
        lr: Learning rate.
        pos_weight: The loss's positive-class weight; raise it where the organism covers
            little of most frames.
        train_split: Split to train on.
        val_split: Split to validate on. None trains every epoch with no validation.
        device: Torch device string; CUDA where available if None.
        seed: Random seed.
        checkpoint_name: Filename for the weights inside `dataset_dir`; None names it
            `smp_segmenter_<encoder>_<today>[_n].pt`, distinct per run.
        project_path: Project whose pipeline folder gets the training diagnostics; None
            writes them under `dataset_dir`.
        visualize: True, an int, or ids: a fixed sample drawn as image, reference and
            prediction at each checkpoint epoch, plus loss and score curves.
        visualize_every: Draw the prediction grid every N epochs, and on each new best
            validation loss.

    Returns:
        The absolute checkpoint path.
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
        logger.warning(
            "training with no validation split -- no early stopping; the final epoch's weights will be saved"
        )

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
        """One `(image, mask)` tensor pair per manifest row, resized as `SMPSegmenter.predict()` resizes."""

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

            mask = cv2.imread(str(Path(dataset_dir) / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy((mask > 0).astype("float32")).unsqueeze(0)

            return image, mask

    datasets = {phase: _MaskDataset(frame) for phase, frame in rows.items()}
    loaders = {
        phase: DataLoader(dataset, batch_size=batch_size, shuffle=(phase == "train"), num_workers=num_workers)
        for phase, dataset in datasets.items()
    }

    model = smp.UnetPlusPlus(
        encoder_name=encoder_name, encoder_weights="imagenet", in_channels=3, classes=1
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = None
    history = {f"{phase}_{metric}": [] for phase in phases for metric in ("loss", "f1", "auroc", "iou")}

    identity = {
        "kind": "train_smp_segmenter",
        "data_hash": _dataset_hash(dataset_dir),
        "encoder_name": encoder_name,
        "size": size,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "pos_weight": pos_weight,
        "train_split": train_split,
        "val_split": val_split,
        "seed": seed,
    }
    report = pipeline_visualization.open_report(
        project_path or dataset_dir,
        f"train__smp_{encoder_name}",
        hash_spec(identity),
        visualize=visualize,
        identity=identity,
    )
    shown = rows["val" if "val" in phases else "train"]
    report.begin(shown["occurrence_id"].astype(str))
    shown = shown[shown["occurrence_id"].astype(str).map(report.wants)]

    for epoch in range(num_epochs):
        improved = False
        for phase in phases:
            model.train(phase == "train")
            running_loss = 0.0
            y_true, y_pred = [], []

            for images, masks in tqdm(loaders[phase], desc=f"epoch {epoch + 1}/{num_epochs} {phase}"):
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
                auroc = float("nan")  # a batch/epoch with only one class present

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_f1"].append(f1)
            history[f"{phase}_auroc"].append(auroc)
            history[f"{phase}_iou"].append(iou)
            logger.info(
                "epoch %d %s: loss=%.4f f1=%.4f auroc=%.4f iou=%.4f",
                epoch + 1,
                phase,
                epoch_loss,
                f1,
                auroc,
                iou,
            )

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

        due = improved or epoch + 1 == num_epochs or (visualize_every and (epoch + 1) % visualize_every == 0)
        if report and due and len(shown):
            _prediction_panels(report, model, shown, dataset_dir, size, device)
            report.checkpoint(f"epoch{epoch + 1:04d}")

    model.load_state_dict(best_state)

    checkpoint_path = (
        Path(dataset_dir) / checkpoint_name
        if checkpoint_name
        else _default_checkpoint_path(dataset_dir, encoder_name)
    )
    torch.save(model.state_dict(), checkpoint_path)
    # Absolute, not whatever form dataset_dir arrived in: register_model()
    # treats a relative path as relative to the PROJECT directory, not to the
    # caller's cwd, and dataset_dir is very often itself already
    # project-relative (e.g. "<project>/training/head") -- returning that
    # as-is would have register_model() double the project prefix.
    checkpoint_path = checkpoint_path.resolve()
    if "val" in phases:
        logger.info("saved checkpoint -> %s (best val loss %.4f)", checkpoint_path, best_val_loss)
    else:
        logger.info("saved checkpoint -> %s (final epoch, no held-out val split)", checkpoint_path)

    if report:
        _training_figures(report, history, phases, best_epoch)
    report.close()

    return checkpoint_path


def register_trained(
    project_path, name, checkpoint, dataset_dir, encoder_name=DEFAULT_ENCODER, size=DEFAULT_SIZE, **kwargs
):
    """Register a checkpoint `train()` wrote, with what `load_registered()` needs to load it.

    Args:
        project_path: Project to register in.
        name: Registered model name, e.g. `"wing_segmenter_v1"`.
        checkpoint: The path `train()` returned.
        dataset_dir: The dataset trained on; its `dataset.json` becomes `training_data`.
        encoder_name: What `train()` was given.
        size: What `train()` was given.
        **kwargs: Passed to `records.models.register_model`; `parameters=` is merged with
            this function's own.

    Returns:
        The `RegisteredModel`.
    """
    parameters = {"encoder_name": encoder_name, "size": size}
    parameters.update(kwargs.pop("parameters", None) or {})
    return register_model(
        project_path,
        name,
        path=checkpoint,
        task="segment",
        framework="torch",
        base_model=kwargs.pop("base_model", encoder_name),
        training_data=str(dataset_dir),
        parameters=parameters,
        **kwargs,
    )


def _default_checkpoint_path(dataset_dir, encoder_name):
    """Return `smp_segmenter_<encoder>_<today>[_n].pt`, with `_n` so a same-day rerun gets a new file."""
    directory = Path(dataset_dir)
    base = f"smp_segmenter_{encoder_name}_{date.today().isoformat()}"
    dest = directory / f"{base}.pt"
    n = 1
    while dest.exists():
        dest = directory / f"{base}_{n}.pt"
        n += 1
    return dest


def _dataset_hash(dataset_dir):
    """Return the `data_hash` from a dataset directory's `dataset.json`, or None."""
    record = Path(dataset_dir) / DATASET_FILE
    if not record.exists():
        return None
    with record.open(encoding="utf-8") as handle:
        return json.load(handle).get("data_hash")


def _prediction_panels(report, model, shown, dataset_dir, size, device, threshold=0.5):
    """Emit image, reference and prediction panels for the report's sampled specimens."""
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
    """Write loss and F1/AUROC/IoU curves over epochs, train against val."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    marks = {"best": best_epoch} if best_epoch is not None else None
    report.figure(
        "loss",
        figures.line_chart(
            {phase: (epochs, history[f"{phase}_loss"]) for phase in phases},
            xlabel="epoch",
            ylabel="loss",
            title="loss",
            marks=marks,
        ),
    )
    report.figure(
        "scores",
        figures.line_chart(
            {
                f"{phase} {metric}": (epochs, history[f"{phase}_{metric}"])
                for phase in phases
                for metric in ("f1", "auroc", "iou")
            },
            xlabel="epoch",
            ylabel="score",
            title="F1 / AUROC / IoU",
            marks=marks,
        ),
    )
