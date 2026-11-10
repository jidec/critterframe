"""
The BioEncoder extension: an embedding model's contract, and the training
scaffold that stops short of guessing at a training loop.
"""

import pytest

from critterframe.extensions.bioencoder import training
from critterframe.extensions.bioencoder.embedding import IMAGENET_MEAN_STD, BioEncoderModel
from critterframe.records.occurrences import load_occurrences


def test_the_unwritten_training_loop_says_so_rather_than_pretending():
    """
    Deliberately unfinished, and the raise is what stops it being quietly
    stubbed into a no-op that trains nothing.
    """
    with pytest.raises(NotImplementedError, match="isn't implemented"):
        training.train(manifest=None, output_dir=None)


def test_an_embedding_model_must_be_able_to_encode():
    """
    The whole contract: whether it is a BioEncoder checkpoint, a fine-tuned timm
    backbone, or something else entirely doesn't matter downstream.
    """
    with pytest.raises(TypeError, match="encode"):
        BioEncoderModel(object(), checkpoint="weights.pt")


def test_a_forward_only_network_is_called_and_encode_wins_when_both_exist():
    """
    A torch Module's forward() is the universal contract; encode() is the
    explicit one, and takes precedence.
    """

    class ForwardOnly:
        def __call__(self, images):
            return "forward"

    class Both(ForwardOnly):
        def encode(self, images):
            return "encode"

    forward_only = BioEncoderModel(ForwardOnly(), checkpoint="a.pt", device="cpu")
    both = BioEncoderModel(Both(), checkpoint="a.pt", device="cpu")
    forward_only._prepare()
    both._prepare()
    assert forward_only._call("x") == "forward"
    assert both._call("x") == "encode"


def test_preprocessing_reaches_the_hash_only_when_set():
    """
    Standardizing the input changes every embedding, so it is identity -- but
    adding the keys unconditionally would move every recipe hash recorded before
    they existed.
    """

    class Encoder:
        def encode(self, images):
            raise NotImplementedError

    plain = BioEncoderModel(Encoder(), checkpoint="a.pt")
    assert plain.identity() == {
        "class": "BioEncoderModel",
        "version": "2",
        "checkpoint": "a.pt",
        "input_size": [224, 224],
        "normalize": True,
    }

    mean, std = IMAGENET_MEAN_STD
    variants = [
        BioEncoderModel(Encoder(), checkpoint="a.pt", mean=mean),
        BioEncoderModel(Encoder(), checkpoint="a.pt", std=std),
        BioEncoderModel(Encoder(), checkpoint="a.pt", head="features"),
    ]
    identities = [plain.identity()] + [variant.identity() for variant in variants]
    assert len({repr(identity) for identity in identities}) == len(identities)


def test_a_registered_bioencoder_checkpoint_loads_from_its_own_parameters(tmp_path, monkeypatch):
    """
    With no network passed, load_registered builds one with training.load(),
    from the backbone and sizes the registry recorded rather than today's
    defaults.
    """
    from critterframe.extensions.bioencoder import embedding
    from critterframe.records.models import register_model

    project = tmp_path / "project"
    project.mkdir()
    (project / "swa").write_bytes(b"weights")
    parameters = {"backbone": "timm_resnet18", "projection_head": False, "input_size": [256, 256]}
    register_model(project, "embed_v1", path="swa", task="embedding", parameters=parameters)

    calls = []

    def fake_load(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return "network"

    monkeypatch.setattr(training, "load", fake_load)
    loaded = embedding.load_registered(project, "embed_v1")

    assert loaded.runtime == "network"
    assert calls == [(project / "swa", parameters)]


def test_an_embedding_is_identified_by_its_checkpoint():
    """
    Two embedding sets from different checkpoints aren't comparable at all --
    not even approximately, since the spaces are unrelated -- so they must never
    be mistaken for equivalent work.
    """

    class Encoder:
        def encode(self, images):
            raise NotImplementedError

    first = BioEncoderModel(Encoder(), checkpoint="a.pt")
    second = BioEncoderModel(Encoder(), checkpoint="b.pt")
    assert first.identity() != second.identity()
    assert first.identity()["checkpoint"] == "a.pt"


HALVES = {"train": 0.5, "val": 0.5}


@pytest.mark.slow
def test_a_bioencoder_dataset_is_written_in_its_own_class_folder_layout(segmented_project, tmp_path):
    """
    BioEncoder reads train/ and val/ as ImageFolders, so the export is
    `<split>/<label>/`, every label in both, no group straddling the line, and
    dataset.json describing exactly the images on disk -- which is what a
    model registered with `training_data=` points at.
    """
    import json

    out = tmp_path / "bio"
    manifest = training.prepare_dataset(
        segmented_project, out, label_col="species", group_col="device", min_per_label=2, fractions=HALVES
    )
    assert not manifest.empty

    classes = {split: set(manifest.loc[manifest["split"] == split, "class"]) for split in HALVES}
    assert classes["train"] == classes["val"] == {"Anax junius", "Libellula lydia"}
    for row in manifest.itertuples():
        assert (out / row.image_path).parent.parent.name == row.split
    assert manifest.groupby("device")["split"].nunique().max() == 1

    record = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    recorded = {split: entry["count"] for split, entry in record["splits"].items()}
    assert recorded == manifest["split"].value_counts().to_dict()


@pytest.mark.slow
def test_a_label_on_only_one_side_of_the_split_is_dropped_from_every_split(
    segmented_project, tmp_path, monkeypatch
):
    """
    An ImageFolder numbers classes by its own folder list, so a label missing
    from val would shift every label after it, silently.
    """
    occurrences = load_occurrences(segmented_project)
    lydia = sorted(occurrences.loc[occurrences["species"] == "Libellula lydia", "occurrence_id"])
    anax = sorted(occurrences.loc[occurrences["species"] == "Anax junius", "occurrence_id"])
    splits = {"train": lydia + anax[:2], "val": anax[2:4], "test": anax[4:]}
    monkeypatch.setattr(training, "split_ids", lambda *args, **kwargs: splits)

    manifest = training.prepare_dataset(
        segmented_project, tmp_path / "bio", label_col="species", group_col=None, min_per_label=1
    )

    assert set(manifest["class"]) == {"Anax junius"}
    assert set(manifest["split"]) == {"train", "val", "test"}


@pytest.mark.slow
def test_a_label_is_capped_before_the_minimum_is_applied(segmented_project, tmp_path):
    manifest = training.prepare_dataset(
        segmented_project,
        tmp_path / "bio",
        label_col="species",
        group_col=None,
        min_per_label=2,
        max_per_label=2,
        fractions=HALVES,
    )
    assert manifest["class"].value_counts().to_dict() == {"Anax junius": 2, "Libellula lydia": 2}


def test_a_dataset_is_never_written_over_an_earlier_one(segmented_project, tmp_path):
    """A leftover image in a class folder would be trained on as though it were selected."""
    out = tmp_path / "bio"
    (out / "train" / "Anax junius").mkdir(parents=True)
    (out / "train" / "Anax junius" / "old.png").write_bytes(b"")
    with pytest.raises(ValueError, match="not empty"):
        training.prepare_dataset(segmented_project, out, label_col="species", group_col=None)


def test_bioencoder_needs_a_train_and_a_val_split(segmented_project, tmp_path):
    with pytest.raises(ValueError, match="'val'"):
        training.prepare_dataset(
            segmented_project, tmp_path / "bio", label_col="species", group_col=None, fractions={"train": 1}
        )


def test_a_registered_checkpoint_loads_back_with_what_register_trained_recorded(tmp_path, monkeypatch):
    from critterframe.extensions.bioencoder import embedding

    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    (project / "data" / "dataset.json").write_text('{"data_hash": "abc"}', encoding="utf-8")
    (project / "swa").write_bytes(b"weights")

    registered = training.register_trained(
        project, "embed_v1", "swa", project / "data", backbone="timm_resnet18", input_size=(256, 256)
    )
    assert registered.record["task"] == "embedding"
    assert registered.record["base_model"] == "timm_resnet18"
    assert registered.record["training_data"]["dataset"]["data_hash"] == "abc"

    calls = []
    monkeypatch.setattr(training, "load", lambda checkpoint, **kwargs: calls.append(kwargs))
    embedding.load_registered(project, "embed_v1")

    assert calls == [
        {"backbone": "timm_resnet18", "input_size": [256, 256], "projection_head": True, "normalize": True}
    ]


def test_a_network_loaded_by_hand_wraps_with_the_registered_preprocessing(tmp_path):
    """
    The registry holds `backbone` for training.load(); a network the caller
    built needs only the preprocessing, and its head label from `projection_head`.
    """
    from critterframe.extensions.bioencoder import embedding
    from critterframe.records.models import register_model

    class Encoder:
        def encode(self, images):
            raise NotImplementedError

    project = tmp_path / "project"
    project.mkdir()
    (project / "swa").write_bytes(b"weights")
    parameters = {"backbone": "timm_resnet18", "projection_head": False, "input_size": [256, 256]}
    register_model(project, "embed_v1", path="swa", task="embedding", parameters=parameters)

    loaded = embedding.load_registered(project, "embed_v1", model=Encoder())

    assert isinstance(loaded.runtime, BioEncoderModel)
    assert loaded.runtime.head == "features"
    assert loaded.runtime.input_size == (256, 256)


def fake_loader(monkeypatch):
    calls = []
    monkeypatch.setattr(
        training, "load", lambda checkpoint, backbone, **kwargs: calls.append((checkpoint, backbone, kwargs))
    )
    return calls


def test_a_bioencoder_config_names_the_checkpoint_it_was_trained_into(tmp_path, monkeypatch):
    """The same keys bioencoder_inference reads, so one config describes the model."""
    config = tmp_path / "inference.yml"
    config.write_text(
        "model:\n  backbone: timm_resnet18\n  checkpoint: last\nimg_size: 384\n", encoding="utf-8"
    )
    calls = fake_loader(monkeypatch)

    training.load_from_config(config, root_dir="root", run_name="odonata")

    checkpoint, backbone, kwargs = calls[0]
    assert str(checkpoint).replace("\\", "/") == "root/weights/odonata/first/last"
    assert backbone == "timm_resnet18"
    assert kwargs["input_size"] == (384, 384)


def test_an_explicit_checkpoint_path_needs_no_bioencoder_root(tmp_path, monkeypatch):
    config = tmp_path / "inference.yml"
    config.write_text("model:\n  backbone: timm_resnet18\n  checkpoint_path: w/swa\n", encoding="utf-8")
    calls = fake_loader(monkeypatch)

    training.load_from_config(config)

    assert calls[0][0] == "w/swa"
    assert calls[0][2]["input_size"] == (224, 224)


def test_a_second_stage_config_is_not_an_embedding(tmp_path):
    config = tmp_path / "inference.yml"
    config.write_text("model:\n  backbone: timm_resnet18\n  stage: second\n", encoding="utf-8")
    with pytest.raises(ValueError, match="first-stage"):
        training.load_from_config(config, checkpoint="w/swa")
