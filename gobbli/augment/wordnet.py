import functools
import random
from typing import Any, List

from gobbli.augment.base import BaseAugment


def _detokenize_doc(doc: Any) -> str:
    """
    Detokenize a spaCy Doc object back into a string, applying our custom replacements
    as needed. This requires the associated extension to have been registered appropriately.
    The :class:`WordNet` constructor should handle registering the extension.
    """
    return "".join([f"{tok._.replacement}{tok.whitespace_}" for tok in doc])


def _get_lemmas(synsets: List[Any]) -> List[str]:
    """
    Return all the lemma names associated with a list of synsets.
    """
    return [lemma_name for synset in synsets for lemma_name in synset.lemma_names()]


@functools.lru_cache(maxsize=256)
def _get_wordnet_lemmas(word: str, pos: str) -> List[str]:
    """
    Determine all the lemmas for a given word to be considered candidates for
    replacement.  Wrap this function in an LRU cache to keep from recalculating common words
    or terms reused frequently in the same document.
    """
    # We should have properly guarded this import in the WordNet constructor
    from nltk.corpus import wordnet

    synsets = wordnet.synsets(word, pos)
    hypernyms = [hypernym for synset in synsets for hypernym in synset.hypernyms()]
    hyponyms = [hyponym for synset in synsets for hyponym in synset.hyponyms()]
    return list(
        frozenset(_get_lemmas(synsets) + _get_lemmas(hypernyms) + _get_lemmas(hyponyms))
    )


class WordNet(BaseAugment):
    """
    Data augmentation method based on WordNet.  Replaces words with similar
    words according to the WordNet ontology.  Texts will be Part of Speech-tagged
    using spaCy to help ensure only sensible replacements (i.e., within the same part of speech)
    are considered.

    Args:
      skip_download_check: If True, don't try to download the WordNet corpus;
        assume it's already been downloaded.
      spacy_model: The language model to be used for Part of Speech tagging by spaCy.
        The model must already have been installed.
    """

    def __init__(self, skip_download_check: bool = False, spacy_model="en_core_web_sm"):
        try:
            from nltk.corpus import wordnet
            import nltk
        except ImportError:
            raise ImportError(
                "WordNet-based data augmentation requires nltk to be installed."
            )

        self.wn = wordnet

        try:
            import spacy
            from spacy.tokens import Token
        except ImportError:
            raise ImportError(
                "WordNet-based data augmentation requires spaCy and a language "
                "model to be installed (for part of speech tagging)."
            )

        if not skip_download_check:
            nltk.download("wordnet")

        self.nlp = spacy.load(spacy_model, parser=False, tagger=True, entity=False)
        Token.set_extension("replacement", default=None, force=True)

    def _maybe_replace_token(self, token: Any) -> str:
        if token.pos_ == "ADJ":
            wordnet_pos = self.wn.ADJ
        elif token.pos_ == "NOUN":
            wordnet_pos = self.wn.NOUN
        elif token.pos_ == "VERB":
            wordnet_pos = self.wn.VERB
        elif token.pos_ == "ADV":
            wordnet_pos = self.wn.ADV
        else:
            # If the token's part of speech isn't recognized by WordNet,
            # return it without replacing.
            return token.text

        all_lemma_names = _get_wordnet_lemmas(token.text, wordnet_pos)
        if len(all_lemma_names) > 0:
            # WordNet lemmas have underscores where spaces should be, so apply spaces
            # appropriately
            return random.choice(all_lemma_names).replace("_", " ")
        else:
            return token.text

    def augment(self, X: List[str], times: int = 5, p: float = 0.1) -> List[str]:
        new_texts = []
        tagged_docs = [doc for doc in self.nlp.pipe(X)]

        for _ in range(times):
            for doc in tagged_docs:
                for tok in doc:
                    if random.random() < p:
                        tok._.replacement = self._maybe_replace_token(tok)
                    else:
                        tok._.replacement = tok.text

                new_texts.append(_detokenize_doc(doc))
        return new_texts
