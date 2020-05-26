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
        ({"target_languages": ["not a language"]}, ValueError),
        # Bad value, one OK value (target_languages)
        ({"target_languages": ["french", "not a language"]}, ValueError),
        # OK values
        ({"batch_size": 16, "target_languages": ["russian", "french"]}, None),
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

    # Can't augment more times than target languages
    invalid_num_times = len(target_languages) + 1
    with pytest.raises(ValueError):
        model.augment(["This is a test."], times=invalid_num_times)

    valid_num_times = len(target_languages)
    new_texts = model.augment(["This is a test."], times=valid_num_times)
    assert len(new_texts) == valid_num_times
