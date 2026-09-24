"""
The BioEncoder extension: an embedding model's contract, and the training
scaffold that stops short of guessing at a training loop.
"""

import pytest

from critterframe.extensions.bioencoder import training
from critterframe.extensions.bioencoder.embedding import IMAGENET_MEAN_STD, BioEncoderModel


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
    assert plain.identity() == {"class": "BioEncoderModel", "version": "2",
                                "checkpoint": "a.pt", "input_size": [224, 224],
                                "normalize": True}

    mean, std = IMAGENET_MEAN_STD
    variants = [BioEncoderModel(Encoder(), checkpoint="a.pt", mean=mean),
                BioEncoderModel(Encoder(), checkpoint="a.pt", std=std),
                BioEncoderModel(Encoder(), checkpoint="a.pt", head="features")]
    identities = [plain.identity()] + [variant.identity() for variant in variants]
    assert len({repr(identity) for identity in identities}) == len(identities)


def test_a_registered_bioencoder_checkpoint_loads_from_its_own_parameters(
        tmp_path, monkeypatch):
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
    parameters = {"backbone": "timm_resnet18", "projection_head": False,
                  "input_size": [256, 256]}
    register_model(project, "embed_v1", path="swa", task="embedding",
                   parameters=parameters)

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


@pytest.mark.slow
def test_a_bioencoder_dataset_record_describes_what_it_shipped(segmented_project, tmp_path):
    """
    The label filter and the split run BEFORE the export, so dataset.json's
    splits are the images on disk. Filtering afterwards left its data_hash
    covering a population the model never saw -- and a model registered with
    `training_data=` then pointed at exactly that.
    """
    import json

    from critterframe.project import paths

    out = tmp_path / "bio"
    manifest = training.prepare_dataset(segmented_project, out, label_col="species",
                                        group_col=None, min_per_label=2)
    assert not manifest.empty

    record = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    recorded = {split: entry["count"] for split, entry in record["splits"].items()}
    assert recorded == manifest["split"].value_counts().to_dict()
    assert list(record["splits"]) == sorted(manifest["split"].unique())

    # The split figure covers what a bespoke per-label figure used to.
    assert list(paths.pipeline_dir(segmented_project).glob("split_ids_*__counts.png"))


def fake_loader(monkeypatch):
    calls = []
    monkeypatch.setattr(training, "load",
                        lambda checkpoint, backbone, **kwargs:
                        calls.append((checkpoint, backbone, kwargs)))
    return calls


def test_a_bioencoder_config_names_the_checkpoint_it_was_trained_into(
        tmp_path, monkeypatch):
    """The same keys bioencoder_inference reads, so one config describes the model."""
    config = tmp_path / "inference.yml"
    config.write_text("model:\n  backbone: timm_resnet18\n  checkpoint: last\n"
                      "img_size: 384\n", encoding="utf-8")
    calls = fake_loader(monkeypatch)

    training.load_from_config(config, root_dir="root", run_name="odonata")

    checkpoint, backbone, kwargs = calls[0]
    assert str(checkpoint).replace("\\", "/") == "root/weights/odonata/first/last"
    assert backbone == "timm_resnet18"
    assert kwargs["input_size"] == (384, 384)


def test_an_explicit_checkpoint_path_needs_no_bioencoder_root(tmp_path, monkeypatch):
    config = tmp_path / "inference.yml"
    config.write_text("model:\n  backbone: timm_resnet18\n  checkpoint_path: w/swa\n",
                      encoding="utf-8")
    calls = fake_loader(monkeypatch)

    training.load_from_config(config)

    assert calls[0][0] == "w/swa"
    assert calls[0][2]["input_size"] == (224, 224)


def test_a_second_stage_config_is_not_an_embedding(tmp_path):
    config = tmp_path / "inference.yml"
    config.write_text("model:\n  backbone: timm_resnet18\n  stage: second\n",
                      encoding="utf-8")
    with pytest.raises(ValueError, match="first-stage"):
        training.load_from_config(config, checkpoint="w/swa")
