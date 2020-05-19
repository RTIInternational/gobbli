import pytest

from gobbli.augment.marian import MarianMT
from gobbli.test.util import model_test_dir


@pytest.mark.parametrize(
    "params,exception",
    [
        # Unknown param
        ({"unknown": None}, ValueError),
        # Bad type (batch_size)
        ({"batch_size": 2.5}, TypeError),
        # Bad type (target_languages)
        ({"target_languages": "english"}, TypeError),
        # Bad value (batch_size)
        ({"batch_size": 0}, ValueError),
        # Bad value (target_languages)
        ({"target_languages": ["not a language"]}, TypeError),
        # Bad value, one OK value (target_languages)
        ({"target_languages": ["spanish", "not a language"]}, TypeError),
        # OK values
        ({"batch_size": 16, "target_languages": ["spanish", "french"]}, None),
    ],
)
def test_init(params, exception):
    if exception is None:
        MarianMT(**params)
    else:
        with pytest.raises(exception):
            MarianMT(**params)


def test_marianmt_augment(model_gpu_config, gobbli_dir):
    # Don't go overboard with the languages here, since each
    # one requires a separate model (few hundred MB) to be downloaded
    target_languages = ["russian", "french"]
    model = MarianMT(
        data_dir=model_test_dir(MarianMT),
        load_existing=True,
        target_languages=target_languages,
        **model_gpu_config,
    )
    model.build()

    new_texts = model.augment(["This is a test."])
    assert len(new_texts) == len(target_languages)
