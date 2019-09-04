import pytest

from gobbli.dataset.newsgroups import NewsgroupsDataset
from gobbli.dataset.trivial import TrivialDataset
from gobbli.model.bert import BERT
from gobbli.model.fasttext import FastText
from gobbli.model.majority import MajorityClassifier
from gobbli.model.mtdnn import MTDNN
from gobbli.model.transformer import Transformer
from gobbli.test.util import model_test_dir, skip_if_low_resource, validate_checkpoint


def check_predict_output(train_output, predict_input, predict_output):
    assert predict_output.y_pred_proba.shape == (
        len(predict_input.X),
        len(train_output.labels),
    )

    assert len(predict_output.y_pred) == len(predict_input.X)
    label_set = set(train_output.labels)
    for p in predict_output.y_pred:
        assert p in label_set


@pytest.mark.parametrize(
    "model_cls,dataset_cls,model_kwargs,train_kwargs,predict_kwargs",
    [
        (MajorityClassifier, TrivialDataset, {}, {}, {}),
        (MajorityClassifier, NewsgroupsDataset, {}, {}, {}),
        (
            BERT,
            TrivialDataset,
            {},
            {"num_train_epochs": 1, "train_batch_size": 1, "valid_batch_size": 1},
            {"predict_batch_size": 1},
        ),
        (
            BERT,
            NewsgroupsDataset,
            {},
            {"num_train_epochs": 1, "train_batch_size": 32, "valid_batch_size": 8},
            {"predict_batch_size": 32},
        ),
        (
            MTDNN,
            TrivialDataset,
            {},
            {"num_train_epochs": 1, "train_batch_size": 1, "valid_batch_size": 1},
            {"predict_batch_size": 1},
        ),
        (
            MTDNN,
            NewsgroupsDataset,
            {},
            {"num_train_epochs": 1, "train_batch_size": 32, "valid_batch_size": 32},
            {"predict_batch_size": 32},
        ),
        (
            FastText,
            TrivialDataset,
            {},
            {"num_train_epochs": 1, "train_batch_size": 1, "valid_batch_size": 1},
            {"predict_batch_size": 1},
        ),
        (
            FastText,
            NewsgroupsDataset,
            {},
            {"num_train_epochs": 1, "train_batch_size": 32, "valid_batch_size": 32},
            {"predict_batch_size": 32},
        ),
        (
            Transformer,
            TrivialDataset,
            {"max_seq_length": 128},
            {"num_train_epochs": 1, "train_batch_size": 1, "valid_batch_size": 1},
            {"predict_batch_size": 1},
        ),
        (
            Transformer,
            NewsgroupsDataset,
            {"max_seq_length": 128},
            {"num_train_epochs": 1, "train_batch_size": 16, "valid_batch_size": 32},
            {"predict_batch_size": 32},
        ),
    ],
)
def test_classifier(
    model_cls,
    dataset_cls,
    model_kwargs,
    train_kwargs,
    predict_kwargs,
    model_gpu_config,
    gobbli_dir,
    request,
):
    """
    Ensure classifiers train and predict appropriately across a few example datasets.
    """
    # These combinations of model and dataset require a lot of memory
    if model_cls in (BERT, MTDNN, Transformer) and dataset_cls in (NewsgroupsDataset,):
        skip_if_low_resource(request.config)

    model = model_cls(
        data_dir=model_test_dir(model_cls),
        load_existing=True,
        **model_gpu_config,
        **model_kwargs,
    )

    model.build()
    ds = dataset_cls.load()

    # Verify training runs, results are sensible
    train_output = model.train(ds.train_input(limit=50, **train_kwargs))
    assert train_output.valid_loss is not None
    assert train_output.train_loss is not None
    assert 0 <= train_output.valid_accuracy <= 1

    validate_checkpoint(model.__class__, train_output.checkpoint)

    predict_input = ds.predict_input(limit=50, **predict_kwargs)

    if isinstance(model, FastText):
        # fastText requires a trained checkpoint for prediction
        pass
    else:
        # Verify prediction runs without a trained checkpoint
        predict_output = model.predict(predict_input)
        check_predict_output(train_output, predict_input, predict_output)

    # Verify prediction runs with a trained checkpoint
    predict_input.checkpoint = train_output.checkpoint
    predict_output = model.predict(predict_input)
    check_predict_output(train_output, predict_input, predict_output)
