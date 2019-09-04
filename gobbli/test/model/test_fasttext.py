import pytest

from gobbli.model.fasttext import FastText


@pytest.mark.parametrize(
    "params,exception",
    [
        # Unknown param
        ({"unknown": None}, ValueError),
        # Bad type (word_ngrams)
        ({"word_ngrams": 1.0}, TypeError),
        # Bad type (lr)
        ({"lr": 1}, TypeError),
        # Bad type (dim)
        ({"dim": 100.0}, TypeError),
        # Bad type (ws)
        ({"ws": 3.0}, TypeError),
        # Bad value (fasttext_model)
        ({"fasttext_model": "bert"}, ValueError),
        # OK value (fasttext_model)
        ({"fasttext_model": "crawl-300d"}, None),
        # Dim mismatch (pretrained vectors vs user-passed dim)
        ({"fasttext_model": "crawl-300d", "dim": 100}, ValueError),
        # OK values (all)
        (
            {
                "word_ngrams": 2,
                "lr": 0.01,
                "dim": 300,
                "ws": 3,
                "fasttext_model": "crawl-300d",
            },
            None,
        ),
    ],
)
def test_init(params, exception):
    if exception is None:
        FastText(**params)
    else:
        with pytest.raises(exception):
            FastText(**params)
