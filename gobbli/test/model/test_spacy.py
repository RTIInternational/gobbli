import pytest

from gobbli.model.spacy import SpaCyModel


@pytest.mark.parametrize(
    "params,exception",
    [
        # Unknown param
        ({"unknown": None}, ValueError),
        # Bad type (dropout)
        ({"dropout": "100"}, TypeError),
        # OK type (dropout)
        ({"dropout": 0.3}, None),
        # Bad type (full_pipeline)
        ({"full_pipeline": 1}, TypeError),
        # OK type (full_pipeline)
        ({"full_pipeline": True}, None),
        # OK types (all params)
        ({"full_pipeline": True, "dropout": 0.3}, None),
    ],
)
def test_init(params, exception):
    if exception is None:
        SpaCyModel(**params)
    else:
        with pytest.raises(exception):
            SpaCyModel(**params)
