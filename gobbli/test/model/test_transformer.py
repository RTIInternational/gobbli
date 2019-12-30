import pytest

from gobbli.model.transformer import Transformer


@pytest.mark.parametrize(
    "params,exception",
    [
        # Unknown param
        ({"unknown": None}, ValueError),
        # Bad type (max_seq_length)
        ({"max_seq_length": "100"}, TypeError),
        # OK type (max_seq_length)
        ({"max_seq_length": 100}, None),
        # Bad type (config_overrides)
        ({"config_overrides": 1}, TypeError),
        # OK type (config_overrides)
        ({"config_overrides": {}}, None),
        # Bad type (lr)
        ({"lr": 1}, TypeError),
        # OK type (lr)
        ({"lr": 1e-3}, None),
        # Bad type (adam_eps)
        ({"adam_eps": 1}, TypeError),
        # OK type (adam_eps)
        ({"adam_eps": 1e-5}, None),
        # Bad type (gradient_accumulation_steps)
        ({"gradient_accumulation_steps": 1.0}, TypeError),
        # OK type (gradient_accumulation_steps)
        ({"gradient_accumulation_steps": 2}, None),
        # OK values (all params),
        (
            {
                "max_seq_length": 100,
                "config_overrides": {},
                "lr": 1e-3,
                "adam_eps": 1e-5,
                "gradient_accumulation_steps": 2,
            },
            None,
        ),
    ],
)
def test_init(params, exception):
    if exception is None:
        Transformer(**params)
    else:
        with pytest.raises(exception):
            Transformer(**params)
