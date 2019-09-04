import tempfile
from pathlib import Path

import pandas as pd
import pytest

from gobbli.dataset.newsgroups import NewsgroupsDataset
from gobbli.experiment.classification import (
    DEFAULT_METRICS,
    ClassificationError,
    ClassificationExperiment,
    ClassificationExperimentResults,
)
from gobbli.model.bert import BERT
from gobbli.model.fasttext import FastText
from gobbli.model.majority import MajorityClassifier
from gobbli.model.mtdnn import MTDNN
from gobbli.model.use import USE
from gobbli.util import dir_to_blob


def test_classification_results_checkpoint(tmpdir):
    # Verify checkpoints can be extracted correctly regardless of format
    tempdir_path = Path(tmpdir)
    checkpoint_path = tempdir_path / "test_checkpoint"
    checkpoint_path.mkdir(parents=True)
    checkpoint_file = checkpoint_path / "checkpoint.txt"

    checkpoint_contents = "test"
    checkpoint_file.write_text(checkpoint_contents)
    checkpoint_bytes = dir_to_blob(checkpoint_path)

    common_args = {
        "training_results": [],
        "labels": [],
        "X": [],
        "y_true": [],
        "y_pred_proba": pd.DataFrame(),
    }
    bytes_results = ClassificationExperimentResults(
        **common_args,
        best_model_checkpoint=checkpoint_bytes,
        best_model_checkpoint_name=checkpoint_file.name,
    )

    path_results = ClassificationExperimentResults(
        **common_args,
        best_model_checkpoint=checkpoint_path,
        best_model_checkpoint_name=checkpoint_file.name,
    )

    # Bytes checkpoint, no base_path (results object creates tempdir)
    bytes_checkpoint = bytes_results.get_checkpoint()
    assert bytes_checkpoint.read_text() == checkpoint_contents

    # Bytes checkpoint, base path
    with tempfile.TemporaryDirectory() as test_dir:
        test_dir_path = Path(test_dir) / "test"
        bytes_checkpoint = bytes_results.get_checkpoint(base_path=test_dir_path)
        assert bytes_checkpoint.parent == test_dir_path
        assert bytes_checkpoint.read_text() == checkpoint_contents

    # Path checkpoint, no base path
    path_checkpoint = path_results.get_checkpoint()
    assert path_checkpoint == checkpoint_path / checkpoint_file
    assert path_checkpoint.read_text() == checkpoint_contents

    # Path checkpoint, base path
    with tempfile.TemporaryDirectory() as test_dir:
        test_dir_path = Path(test_dir) / "test"
        path_checkpoint = path_results.get_checkpoint(base_path=test_dir_path)
        assert path_checkpoint.parent == test_dir_path
        assert path_checkpoint.read_text() == checkpoint_contents


def test_classification_results():
    results = ClassificationExperimentResults(
        training_results=[],
        labels=["a", "b"],
        X=["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4", "b5"],
        y_true=["a", "a", "a", "a", "b", "b", "b", "b", "b"],
        y_pred_proba=pd.DataFrame(
            {
                "a": [0.7, 0.3, 0.4, 0.6, 0.9, 0.3, 0.6, 0.4, 0.7],
                "b": [0.3, 0.7, 0.6, 0.4, 0.1, 0.7, 0.4, 0.6, 0.3],
            }
        ),
        best_model_checkpoint=bytes(),
        best_model_checkpoint_name="",
    )

    # Ensure predicted labels are calculated correctly
    expected_y_pred = ["a", "b", "b", "a", "a", "b", "a", "b", "a"]
    assert results.y_pred == expected_y_pred

    # Ensure errors are calculated correctly
    # Error observations are a2, a3, b1, b3, b5
    ea2 = ClassificationError(X="a2", y_true="a", y_pred_proba={"a": 0.3, "b": 0.7})
    ea3 = ClassificationError(X="a3", y_true="a", y_pred_proba={"a": 0.4, "b": 0.6})
    eb1 = ClassificationError(X="b1", y_true="b", y_pred_proba={"a": 0.9, "b": 0.1})
    eb3 = ClassificationError(  # noqa: F841
        X="b3", y_true="b", y_pred_proba={"a": 0.6, "b": 0.4}
    )
    eb5 = ClassificationError(X="b5", y_true="b", y_pred_proba={"a": 0.7, "b": 0.3})

    # Cut off at k=2 and ensure the correct errors are in there and ordered correctly
    errors = results.errors(k=2)

    # Expected errors
    a_false_positives = [eb1, eb5]
    a_false_negatives = [ea2, ea3]
    b_false_positives = a_false_negatives
    b_false_negatives = a_false_positives

    a_errors = errors["a"]
    b_errors = errors["b"]
    assert a_errors[0] == a_false_positives
    assert a_errors[1] == a_false_negatives
    assert b_errors[0] == b_false_positives
    assert b_errors[1] == b_false_negatives


@pytest.mark.parametrize(
    "model_cls,valid", [(FastText, True), (BERT, True), (MTDNN, True), (USE, False)]
)
def test_classification_init_validation(model_cls, valid):
    if valid:
        ClassificationExperiment._validate_model_cls(model_cls)
    else:
        with pytest.raises(ValueError):
            ClassificationExperiment._validate_model_cls(model_cls)


@pytest.mark.parametrize(
    "bad_value",
    [
        # Not enough values
        ((0.8, 0.2),),
        # Too many values
        ((0.6, 0.2, 0.1, 0.1),),
        # sum > 1
        ((0.7, 0.2, 0.2),),
        # sum < 1
        ((0.6, 0.2, 0.1),),
    ],
)
def test_classification_run_validation(bad_value):
    with pytest.raises(ValueError):
        ClassificationExperiment._validate_split(bad_value)


@pytest.mark.parametrize(
    "model_cls,dataset_cls,param_grid,limit,ray_local_mode",
    [
        # Can't use the TrivialDataset here because it's too small for the standard
        # train/valid/test split
        # Trivial model, no ray
        (MajorityClassifier, NewsgroupsDataset, {}, 1000, True),
        # "Real" model/dataset, use ray cluster
        (FastText, NewsgroupsDataset, {"lr": [0.1, 0.01]}, 1000, False),
        # "Real" model/dataset with more complicated checkpoint structure, use ray cluster
        # Use smaller limit since this model takes a while to train
        (BERT, NewsgroupsDataset, {}, 50, False),
    ],
)
def test_classification_run(
    request, model_cls, dataset_cls, param_grid, limit, ray_local_mode, gobbli_dir
):
    if model_cls == BERT:
        pytest.skip(
            "BERT model takes up too much disk space; this test is currently disabled"
        )

    dataset = dataset_cls.load()

    exp = ClassificationExperiment(
        model_cls=model_cls,
        dataset=dataset,
        param_grid=param_grid,
        task_num_cpus=1,
        task_num_gpus=0,
        worker_gobbli_dir=gobbli_dir,
        worker_log_level=request.config.getoption("worker_log_level"),
        limit=limit,
        ignore_ray_initialized_error=True,
        ray_kwargs={"local_mode": ray_local_mode},
    )

    results = exp.run()

    if not model_cls == MajorityClassifier:
        assert results.best_model_checkpoint is not None
        assert results.best_model_checkpoint_name is not None

    metrics = results.metrics()
    assert len(metrics) == len(DEFAULT_METRICS)
    for metric, value in metrics.items():
        assert isinstance(value, float)

    metrics_report = results.metrics_report()
    assert len(metrics_report) > 0

    k = 5
    errors = results.errors(k=k)
    for label, (false_positives, false_negatives) in errors.items():
        assert len(false_positives) <= k
        assert len(false_negatives) <= k

    errors_report = results.errors_report(k=k)
    assert len(errors_report) > 0

    # Verify the plot runs without errors
    results.plot()
