import pytest
from spacy.lang.en import English

from gobbli.augment.wordnet import WordNet, _detokenize_doc


@pytest.mark.parametrize(
    "text",
    [
        "This is a test.",
        "Test  with double space.",
        "Test-with hyphen.",
        "Testing some 1 2 3 numbers.",
    ],
)
def test_detokenize_doc(text):
    # Initialize the spaCy extension needed to detokenize text
    WordNet()

    nlp = English()
    doc = nlp(text)

    # Fill out the replacement attribute as WordNet would.
    for tok in doc:
        tok._.replacement = tok.text
    assert _detokenize_doc(doc) == text


def test_wordnet_augment():
    wn = WordNet()
    times = 5
    new_texts = wn.augment(["This is a test."], times=times)
    assert len(new_texts) == times
