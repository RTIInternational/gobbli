import pytest

from gobbli.model.bert import BERT


@pytest.mark.parametrize(
    "params,exception",
    [
        # Unknown param
        ({"unknown": None}, ValueError),
        # Bad type (max_seq_length)
        ({"max_seq_length": "100"}, TypeError),
        # Bad value (bert_model)
        ({"bert_model": "ernie"}, ValueError),
        # OK type (max_seq_length)
        ({"max_seq_length": 100}, None),
        # OK value (bert_model)
        ({"bert_model": "bert-base-uncased"}, None),
        # OK values (both params)
        ({"max_seq_length": 100, "bert_model": "bert-base-uncased"}, None),
    ],
)
def test_init(params, exception):
    if exception is None:
        BERT(**params)
    else:
        with pytest.raises(exception):
            BERT(**params)
