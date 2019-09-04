import pytest

from gobbli.model.mtdnn import MTDNN


@pytest.mark.parametrize(
    "params,exception",
    [
        # Unknown param
        ({"unknown": None}, ValueError),
        # Bad type (max_seq_length)
        ({"max_seq_length": "100"}, TypeError),
        # Bad value (mtdnn_model)
        ({"mtdnn_model": "bert"}, ValueError),
        # OK type (max_seq_length)
        ({"max_seq_length": 100}, None),
        # OK value (mtdnn_model)
        ({"mtdnn_model": "mt-dnn-base"}, None),
        # OK values (both params)
        ({"max_seq_length": 100, "mtdnn_model": "mt-dnn-base"}, None),
    ],
)
def test_init(params, exception):
    if exception is None:
        MTDNN(**params)
    else:
        with pytest.raises(exception):
            MTDNN(**params)
