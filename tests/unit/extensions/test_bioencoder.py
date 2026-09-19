"""
The BioEncoder extension: an embedding model's contract, and the training
scaffold that stops short of guessing at a training loop.
"""

import pytest

from critterframe.extensions.bioencoder import training
from critterframe.extensions.bioencoder.embedding import BioEncoderModel


def test_the_unwritten_training_loop_says_so_rather_than_pretending():
    """
    Deliberately unfinished, and the raise is what stops it being quietly
    stubbed into a no-op that trains nothing.
    """
    with pytest.raises(NotImplementedError, match="isn't implemented"):
        training.train(manifest=None, output_dir=None)

    with pytest.raises(NotImplementedError, match="no loader is implemented"):
        training.load("some_checkpoint.pt")


def test_an_embedding_model_must_be_able_to_encode():
    """
    The whole contract: whether it is a BioEncoder checkpoint, a fine-tuned timm
    backbone, or something else entirely doesn't matter downstream.
    """
    with pytest.raises(TypeError, match="encode"):
        BioEncoderModel(object(), checkpoint="weights.pt")


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
