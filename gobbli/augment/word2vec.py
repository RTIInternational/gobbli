import functools
import logging
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Union, cast

import numpy as np

from gobbli.augment.base import BaseAugment
from gobbli.util import (
    TokenizeMethod,
    assert_in,
    assert_type,
    download_archive,
    download_file,
    is_archive,
    tokenize,
)

LOGGER = logging.getLogger(__name__)

_WORD2VEC_MODEL_ARCHIVES = {
    "glove.6B": "http://nlp.stanford.edu/data/glove.6B.zip",
    "glove.42B.300d": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
    "glove.840B.300d": "http://nlp.stanford.edu/data/glove.840B.300d.zip",
    "glove.twitter.27B": "http://nlp.stanford.edu/data/glove.twitter.27B.zip",
    "fasttext.en.300d": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec",
    "fasttext.simple.300d": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec",
    "charngram.100d": "http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz",
}
"""
A mapping from archive names to downloadable archives.  See :obj:`WORD2VEC_MODELS`
for a list of names users can pass to get pretrained word2vec models.
"""

WORD2VEC_MODELS = {
    "glove.6B.50d": ("glove.6B", "glove.6B.50d.txt"),
    "glove.6B.100d": ("glove.6B", "glove.6B.100d.txt"),
    "glove.6B.200d": ("glove.6B", "glove.6B.200d.txt"),
    "glove.6B.300d": ("glove.6B", "glove.6B.300d.txt"),
    "glove.42B.300d": ("glove.42B.300d", "glove.42B.300d.txt"),
    "glove.840B.300d": ("glove.840B.300d", "glove.840B.300d.txt"),
    "glove.twitter.27B.25d": ("glove.twitter.27B", "glove.twitter.27B.25d"),
    "glove.twitter.27B.50d": ("glove.twitter.27B", "glove.twitter.27B.50d"),
    "glove.twitter.27B.100d": ("glove.twitter.27B", "glove.twitter.27B.100d"),
    "glove.twitter.27B.200d": ("glove.twitter.27B", "glove.twitter.27B.200d"),
    "fasttext-en": ("fasttext.en.300d", "wiki.en.vec"),
    "fasttext-simple": ("fasttext.simple.300d", "wiki.simple.vec"),
    "charngram": ("charngram.100d", "charNgram.txt"),
}
"""
A mapping from word2vec model names to archive URLs and filenames (since
some models contain multiple files per archive).  Pass one of these key names
to :class:`Word2Vec` to use pretrained model weights.
"""


class Word2Vec(BaseAugment):
    """
    Data augmentation method based on word2vec.  Replaces words with similar words
    according to vector similarity.

    Args:
      model: Pretrained word2vec model to use.  If a string, it should correspond
        to one of the keys in :obj:`WORD2VEC_MODELS`.  The corresponding pretrained
        vectors will be downloaded and used.  If a Path, it's assumed to be
        a file containing pretrained vectors, which will be loaded into a gensim word2vec
        model.  If a gensim Word2Vec model, it will be used directly.
      tokenizer: Function used for tokenizing texts to do word replacement. If an instance
        of :class:`gobbli.util.TokenizeMethod`, the corresponding preset tokenization method
        will be used.  If a callable, it should accept a list of strings and return a
        list of tokenized strings.
      n_similar: Number of similar words to be considered for replacement.
      diversity: 0 < diversity <= 1; determines the likelihood of selecting replacement words
        based on their similarity to the original word.
        At 1, the most similar words are most likely to be selected
        as replacements.  As diversity decreases, likelihood of selection becomes less
        dependent on similarity.
    """

    def __init__(
        self,
        # Can't make this type more restrictive since gensim might not be
        # available, and we need to make the union include a gensim type
        model: Any,
        tokenizer: Union[
            str, TokenizeMethod, Callable[[List[str]], List[List[str]]]
        ] = TokenizeMethod.SPLIT,
        n_similar: int = 10,
        diversity: float = 0.8,
    ):
        try:
            import gensim
            from gensim.scripts.glove2word2vec import glove2word2vec
        except ImportError:
            raise ImportError(
                "word2vec-based data augmentation requires gensim to be installed."
            )

        if isinstance(model, str):
            # Download and extract pretrained weights from a public source
            assert_in("word2vec model", model, set(WORD2VEC_MODELS.keys()))
            archive_name, filename = WORD2VEC_MODELS[model]
            archive_url = _WORD2VEC_MODEL_ARCHIVES[archive_name]

            LOGGER.debug(f"Downloading word2vec model '{model}'")
            # Some downloads aren't contained in archives
            if is_archive(Path(archive_url)):
                extract_dir = download_archive(
                    archive_url, self.data_dir(), junk_paths=True
                )
                model_file = extract_dir / filename
            else:
                model_file = download_file(archive_url)

            if model.startswith("glove"):
                LOGGER.debug(f"Converting GloVe format to word2vec format")
                # Need to convert the downloaded file to word2vec format,
                # since GloVe vectorsr are formatted slightly differently
                with tempfile.NamedTemporaryFile() as f:
                    tempfile_path = Path(f.name)
                    glove2word2vec(model_file, tempfile_path)
                    shutil.copy2(tempfile_path, model_file)

            LOGGER.debug(f"Loading word2vec model '{model}'")
            self._model = gensim.models.KeyedVectors.load_word2vec_format(model_file)
            LOGGER.debug(f"word2vec model loaded")
        elif isinstance(model, Path):
            LOGGER.debug(f"Loading word2vec model from path '{model}'")
            self._model = gensim.models.KeyedVectors.load_word2vec_format(str(model))
            LOGGER.debug(f"word2vec model loaded")
        elif isinstance(model, (gensim.models.Word2Vec, gensim.models.KeyedVectors)):
            self._model = model
        else:
            raise TypeError(
                f"unsupported type for initializing word2vec model: '{type(model)}'"
            )

        assert_type("n_similar", n_similar, int)
        if n_similar <= 0:
            raise ValueError("n_similar must be > 0")
        self.n_similar = n_similar

        assert_type("diversity", diversity, float)
        if not 0 < diversity <= 1:
            raise ValueError("diversity must be > 0 and <= 1")
        self.diversity = diversity

        if isinstance(tokenizer, str):
            tokenizer = TokenizeMethod[tokenizer]

        if isinstance(tokenizer, TokenizeMethod):
            # Avoid mypy error when passing a partially-applied function created by
            # functools.partial
            self.tokenizer = cast(
                Callable[[List[str]], List[List[str]]],
                functools.partial(tokenize, tokenizer),
            )
        elif callable(tokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError(f"unsupported type for tokenizer: '{type(tokenizer)}'")

    def _maybe_replace_token(self, token: str) -> str:
        """
        Generate a replacement for the given token using our model and parameters.
        """
        try:
            similar = self._model.most_similar(positive=token, topn=self.n_similar)
        except KeyError:
            # This token doesn't exist in the word2vec model's vocabulary --
            # don't change it
            return token
        similar_probs = np.array([s[1] for s in similar])

        # Scale according to diversity and normalize
        scaled_probs = np.power(similar_probs, 1 / self.diversity)
        scaled_probs = scaled_probs / np.sum(scaled_probs)
        selected_ndx = np.argmax(np.random.multinomial(1, scaled_probs))
        return similar[selected_ndx][0]

    def augment(self, X: List[str], times: int = 5, p: float = 0.1) -> List[str]:
        new_texts = []
        tokenized_texts = self.tokenizer(X)
        for _ in range(times):
            for tokens in tokenized_texts:
                new_tokens = tokens[:]
                for i in range(len(new_tokens)):
                    if random.random() < p:
                        new_tokens[i] = self._maybe_replace_token(new_tokens[i])
                # TODO can we do better with detokenization here?
                new_texts.append(" ".join(new_tokens))
        return new_texts
