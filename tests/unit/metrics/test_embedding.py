"""
Embeddings from any network: the wrapper's contract, what reaches the hash,
and that core stays importable without torch.
"""

import subprocess
import sys

import numpy as np
import pytest

from critterframe.extensions.bioencoder.embedding import BioEncoderModel
from critterframe.metrics.embedding import (
    EmbeddingModel,
    embedding,
    letterbox,
    state_dict_digest,
)
from critterframe.recipes import Recipe


class Encoder:
    def encode(self, images):
        raise NotImplementedError


def test_importing_embeddings_does_not_import_torch():
    """
    Core runs without torch; a module that merely DEFINES a torch wrapper must
    not pull it in. Checked in a fresh interpreter, since this one may already
    have torch loaded by another test.
    """
    code = ("import sys, critterframe, critterframe.metrics.embedding, "
            "critterframe.extensions.bioencoder.embedding; "
            "print('torch' in sys.modules, 'timm' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, check=True)
    assert result.stdout.split() == ["False", "False"]


def test_a_recipe_naming_an_embedding_hashes_without_torch():
    recipe = Recipe("metric", "embed", [embedding(EmbeddingModel(Encoder(), "a.pt"))],
                    part="organism")
    assert len(recipe.hash) == 16


def test_the_bioencoder_identity_survives_the_move_into_core():
    """
    Every embedding recipe hash already recorded was built on this exact dict,
    so moving the machinery into core must not change a key of it.
    """
    assert BioEncoderModel(Encoder(), checkpoint="a.pt").identity() == {
        "class": "BioEncoderModel", "version": "2", "checkpoint": "a.pt",
        "input_size": [224, 224], "normalize": True}
    assert isinstance(BioEncoderModel(Encoder(), "a.pt"), EmbeddingModel)


def test_a_generic_model_is_not_mistaken_for_a_bioencoder_one():
    generic = EmbeddingModel(Encoder(), checkpoint="a.pt").identity()
    assert generic["class"] == "EmbeddingModel"
    assert generic != BioEncoderModel(Encoder(), checkpoint="a.pt").identity()


def test_letterboxing_reaches_the_hash_only_when_set():
    plain = EmbeddingModel(Encoder(), checkpoint="a.pt")
    padded = EmbeddingModel(Encoder(), checkpoint="a.pt", resize="pad")
    assert "resize" not in plain.identity()
    assert padded.identity()["resize"] == "pad"


def test_an_unknown_resize_mode_is_refused():
    with pytest.raises(ValueError, match="resize"):
        EmbeddingModel(Encoder(), checkpoint="a.pt", resize="crop")


def test_letterbox_keeps_the_aspect_ratio_on_a_square():
    image = np.full((10, 40, 3), 200, np.uint8)
    boxed = letterbox(image)
    assert boxed.shape == (40, 40, 3)
    assert boxed[:15].max() == 0 and boxed[25:].max() == 0
    assert (boxed[15:25] == 200).all()


def test_padding_changes_what_the_network_sees():
    """Stretching an elongated organism distorts it; padding keeps its shape."""
    image = np.zeros((20, 80, 3), np.uint8)
    image[:, :40] = 255
    stretched = EmbeddingModel(Encoder(), "a.pt", input_size=(16, 16)).preprocess(image)
    padded = EmbeddingModel(Encoder(), "a.pt", input_size=(16, 16),
                            resize="pad").preprocess(image)
    assert stretched.shape == padded.shape == (16, 16, 3)
    assert padded[0].max() == 0          # a black letterbox band on top
    assert stretched[0].max() > 0


def test_a_state_dict_digest_follows_the_weights():
    class Network:
        def __init__(self, weights):
            self.weights = weights

        def state_dict(self):
            return {"layer.weight": np.asarray(self.weights, np.float32)}

    assert (state_dict_digest(Network([1, 2]))
            == state_dict_digest(Network([1, 2])))
    assert (state_dict_digest(Network([1, 2]))
            != state_dict_digest(Network([1, 3])))


def test_embedding_a_segment_returns_a_normalized_plain_list():
    try:
        import torch
    except (ImportError, OSError) as exc:  # OSError: a torch whose DLLs won't load
        pytest.skip(f"torch unavailable: {exc}")

    class MeanColour:
        def __call__(self, images):
            return torch.stack([images.mean(dim=(2, 3))[0]])

    from critterframe.recipes import Segment

    model = EmbeddingModel(MeanColour(), "a.pt", input_size=(8, 8), device="cpu")
    segment = Segment(np.full((10, 10, 3), 128, np.uint8),
                      mask=np.ones((10, 10), bool))
    vector = embedding(model)(segment)

    assert isinstance(vector, list) and len(vector) == 3
    assert np.isclose(np.linalg.norm(vector), 1.0)
