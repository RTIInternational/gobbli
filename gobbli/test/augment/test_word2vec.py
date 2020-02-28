from pathlib import Path

import gensim
import pytest

from gobbli.augment.word2vec import _WORD2VEC_MODEL_ARCHIVES, WORD2VEC_MODELS, Word2Vec
from gobbli.util import TokenizeMethod


def test_word2vec_models():
    for model_name, (archive_name, filename) in WORD2VEC_MODELS.items():
        assert archive_name in _WORD2VEC_MODEL_ARCHIVES


_TEST_WORD2VEC_VECTORS = """2 3
test 0 1 1
test2 1 0 1
"""


def make_weights_file(tmpdir):
    weights_file = Path(tmpdir) / "word2vec.txt"
    weights_file.write_text(_TEST_WORD2VEC_VECTORS)
    return weights_file


def test_word2vec_init(tmpdir):
    # Path corresponding to existing weights file should be loaded
    weights_file = make_weights_file(tmpdir)
    Word2Vec(weights_file)

    # Existing model should be loaded
    m = gensim.models.KeyedVectors.load_word2vec_format(str(weights_file))
    Word2Vec(m)

    # String corresponding to one of the pretrained models should download and
    # use the weights
    Word2Vec("fasttext-simple")

    # Anything else should error
    with pytest.raises(TypeError):
        Word2Vec(None)


@pytest.mark.parametrize(
    "n_similar,diversity,tokenizer,exception_cls",
    [
        # wrong type n_similar
        (5.5, 1.0, TokenizeMethod.SPLIT, TypeError),
        # wrong type diversity
        (5, 1, TokenizeMethod.SPLIT, TypeError),
        # wrong type tokenizer
        (5, 0.8, 1, TypeError),
        # bad value n_similar
        (0, 1.0, TokenizeMethod.SPLIT, ValueError),
        # bad value diversity (<= 0)
        (5, 0.0, TokenizeMethod.SPLIT, ValueError),
        # bad value diversity (> 1)
        (5, 1.1, TokenizeMethod.SPLIT, ValueError),
        # bad value str tokenizer (no match)
        (5, 0.8, "no such tokenizer", KeyError),
        # ok (enum tokenizer)
        (5, 0.8, TokenizeMethod.SPLIT, None),
        # ok (callable tokenizer)
        (5, 0.8, lambda s: s.split(), None),
        # ok (str tokenizer)
        (5, 0.8, "SPLIT", None),
    ],
)
def test_word2vec_kwargs(tmpdir, n_similar, diversity, tokenizer, exception_cls):
    weights_file = make_weights_file(tmpdir)
    kwargs = {"n_similar": n_similar, "diversity": diversity, "tokenizer": tokenizer}

    if exception_cls is None:
        Word2Vec(weights_file, **kwargs)
    else:
        with pytest.raises(exception_cls):
            Word2Vec(weights_file, **kwargs)


def test_word2vec_replace(tmpdir):
    weights_file = make_weights_file(tmpdir)

    m = Word2Vec(weights_file)

    # we should find a replacement for a word in the vocabulary
    assert m._maybe_replace_token("test") == "test2"

    # out of vocabulary words shouldn't be replaced
    assert m._maybe_replace_token("bad_token") == "bad_token"
