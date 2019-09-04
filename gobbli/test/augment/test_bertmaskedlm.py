import pytest

from gobbli.augment.bert import BERTMaskedLM
from gobbli.test.util import model_test_dir


@pytest.mark.parametrize(
    "params,exception",
    [
        # Unknown param
        ({"unknown": None}, ValueError),
        # Bad type (diversity)
        ({"diversity": 2}, TypeError),
        # Bad type (batch size)
        ({"batch_size": 2.5}, TypeError),
        # Bad type (n_probable)
        ({"n_probable": 2.5}, TypeError),
        # Bad value (diversity)
        ({"diversity": 0.0}, ValueError),
        # Bad value (batch_size)
        ({"batch_size": 0}, ValueError),
        # Bad value (n_probable)
        ({"n_probable": 0}, ValueError),
        # OK values
        ({"diversity": 0.5, "n_probable": 3, "batch_size": 16}, None),
    ],
)
def test_init(params, exception):
    if exception is None:
        BERTMaskedLM(**params)
    else:
        with pytest.raises(exception):
            BERTMaskedLM(**params)


def test_bertmaskedlm_augment(model_gpu_config, gobbli_dir):
    model = BERTMaskedLM(
        data_dir=model_test_dir(BERTMaskedLM), load_existing=True, **model_gpu_config
    )
    model.build()

    times = 5
    new_texts = model.augment(["This is a test."], times=times)
    assert len(new_texts) == times
