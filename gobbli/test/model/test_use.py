import pytest

from gobbli.model.use import USE


@pytest.mark.parametrize(
    "params,exception",
    [
        # Unknown param
        ({"unknown": None}, ValueError),
        # Bad value (use_model)
        ({"use_model": "bert"}, ValueError),
        # OK value (use_model)
        ({"use_model": "universal-sentence-encoder"}, None),
    ],
)
def test_init(params, exception):
    if exception is None:
        USE(**params)
    else:
        with pytest.raises(exception):
            USE(**params)
