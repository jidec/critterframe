"""
Pinned digests. Read this before changing one.

IF A TEST IN THIS FILE FAILS, YOU HAVE INVALIDATED EVERY MASK AND EVERY METRIC
IN EVERY EXISTING PROJECT. Recipe hashes are stored on disk, in `masks.parquet`
and in the metrics table, and they are how the package decides what work is
already done. Move the hashing machinery and every stored hash becomes
unmatchable: the next run recomputes everything, and every value measured before
the change stops being reported as current.

That may be exactly the right thing to do. It must be deliberate. If it is:
update the digest here in the same commit, and say so in the commit message.

**Why these three and not thirty.** The rest of the suite tests hashing
RELATIONALLY -- a changed parameter must move the hash, a display flag must not
(see `test_recipes.py`). Relational tests catch every mistake about what belongs
in identity, and are blind to exactly one thing: a global change to the
machinery. Flip `sort_keys`, change `HASH_LENGTH`, rename a key in `spec()`, and
every relational test still passes, because everything moved together. These
three digests are the only thing standing between that and silent data loss.

Three, because a wall of pinned hex gets bulk-regenerated instead of read, and a
golden test nobody reads protects nothing. One per layer:

  hash_spec        the serializer and the digest itself
  Recipe.hash      the whole spec() key structure, end to end
  derivation_hash  the chaining that propagates a resegmentation downstream
"""

from critterframe.metrics.dimensions import body_length
from critterframe.recipes import Recipe, canonical_json, hash_spec
from critterframe.records.masks import derivation_hash
from critterframe.segmentation.run import segment


class FrozenModel:
    """
    A model whose identity() will never change, defined here rather than taken
    from `helpers.models` on purpose: a golden digest must depend only on the
    package's hashing, never on a test helper somebody might reasonably edit.
    """

    def identity(self):
        return {"class": "FrozenModel", "checkpoint": "frozen-v1"}


def test_hash_spec_digest():
    """The serializer plus sha256 plus the 16-character truncation."""
    assert hash_spec({"a": 1, "b": [2, 3]}) == "efbd0040190fb087"


def test_canonical_json_is_sorted_and_compact():
    """
    What `hash_spec` hashes. Pinned as text as well as as a digest, because
    when the digest above moves this line says which of the two changed.
    """
    assert canonical_json({"b": [2, 3], "a": 1}) == '{"a":1,"b":[2,3]}'


def test_segmentation_recipe_digest():
    """
    A fully-specified segmentation recipe: kind, name, part, from_part, inputs,
    and one operation with its own name, kind, version, parameters, and model
    identity. Every key of `spec()` at both levels is inside this number.
    """
    recipe = Recipe("segment", "organisms", [segment(FrozenModel())],
                    part="organism")
    assert recipe.hash == "8bac4483268b4304"


def test_metric_recipe_digest():
    """
    The metric side of the same structure -- `Metric.spec()` adds `metric_name`
    and `unit`, and `unit` being in here is what makes measuring in millimetres
    a different recipe from measuring in pixels.
    """
    recipe = Recipe("metric", "traits", [body_length()], part="organism")
    assert recipe.hash == "c708c1489971f5c6"


def test_derivation_hash_chain_digest():
    """
    A derived part's identity: its own recipe hash chained with the hash of the
    mask it was cut out of. This is what moves every wing metric when the
    organism underneath is resegmented.
    """
    assert derivation_hash("aaaaaaaaaaaaaaaa",
                           "bbbbbbbbbbbbbbbb") == "b710b93cc6715f34"


def test_derivation_hash_without_an_upstream_is_the_recipe_hash():
    """
    Not a golden so much as the base case the chain rests on: a mask with no
    upstream IS its recipe hash. Anything else would mean tables written before
    `source_mask_hash` existed could never match, and every project predating
    the column would recompute from scratch.
    """
    assert derivation_hash("aaaaaaaaaaaaaaaa") == "aaaaaaaaaaaaaaaa"
