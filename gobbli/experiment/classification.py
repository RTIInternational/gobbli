import logging
import math
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import altair as alt
import pandas as pd
import ray
from sklearn.model_selection import ParameterGrid, train_test_split

import gobbli.io
from gobbli.experiment.base import (
    BaseExperiment,
    get_worker_ip,
    init_gpu_config,
    init_worker_env,
)
from gobbli.inspect.evaluate import ClassificationError, ClassificationEvaluation
from gobbli.model.mixin import PredictMixin, TrainMixin
from gobbli.util import blob_to_dir, dir_to_blob, is_multilabel


@dataclass
class ClassificationExperimentResults:
    """
    Results from a classification experiment.  An experiment entails training a set of models
    based on a grid of parameters, retraining on the full train/validation dataset with the
    best set of parameters, and evaluating predictions on the test set.

    Args:
      training_results: A list of dictionaries containing information about each training run,
        one for each unique combination of hyperparameters in :paramref:`BaseExperiment.params.param_grid`.
      labels: The set of unique labels in the dataset.
      X: The list of texts to classify.
      y_true: The true labels for the test set, as passed by the user.
      y_pred_proba: A dataframe containing a row for each observation in the test set and a
        column for each label in the training data.  Cells are predicted probabilities.
      best_model_checkpoint: If results came from another process on the master node, this is
        the directory containing the checkpoint.  If the results came from a worker node, this
        is a bytes object containing the compressed model weights.
      best_model_checkpoint_name: Path to the best checkpoint within the directory or
        or compressed blob.
      metric_funcs: Overrides for the default set of metric functions used to evaluate
        the classifier's performance.
    """

    training_results: List[Dict[str, Any]]
    labels: List[str]
    X: List[str]
    y_true: Union[List[str], List[List[str]]]
    y_pred_proba: pd.DataFrame
    best_model_checkpoint: Union[bytes, Path]
    best_model_checkpoint_name: str
    metric_funcs: Optional[Dict[str, Callable[[Sequence, Sequence], float]]] = None

    def __post_init__(self):
        self.evaluation = ClassificationEvaluation(
            labels=self.labels,
            X=self.X,
            y_true=self.y_true,
            y_pred_proba=self.y_pred_proba,
            metric_funcs=self.metric_funcs,
        )
        self.multilabel = is_multilabel(self.y_true)

    def get_checkpoint(self, base_path: Optional[Path] = None) -> Path:
        """
        Return a filesystem path to our checkpoint, which can be used to initialize
        future models from the same state. If a base_path is provided, copy/extract
        the checkpoint under that path.

        NOTE: If no base_path is provided and the checkpoint comes from a remote
        worker, the checkpoint will be extracted to a temporary directory, and a
        warning will be emitted.  gobbli will make no effort to ensure the temporary
        directory is cleaned up after creation.

        Args:
          base_path: Optional directory to extract/copy the checkpoint to. If not provided,
            the original path will be returned if the checkpoint already existed on the
            current machine's filesystem.  If the checkpoint is a bytes object, a temporary
            directory will be created.  The directory must not already exist.

        Returns:
          The path to the extracted checkpoint.
        """
        if isinstance(self.best_model_checkpoint, bytes):
            if base_path is None:
                warnings.warn(
                    "No base_path provided; checkpoint extracting to temporary "
                    "directory."
                )
                base_path = Path(tempfile.mkdtemp())

            blob_to_dir(self.best_model_checkpoint, base_path)
            return base_path / self.best_model_checkpoint_name

        elif isinstance(self.best_model_checkpoint, Path):
            if base_path is None:
                base_path = self.best_model_checkpoint
            else:
                # Copy the checkpoint to the user-provided base path
                shutil.copytree(self.best_model_checkpoint, base_path)
            return base_path / self.best_model_checkpoint_name
        else:
            raise TypeError(
                f"unsupported checkpoint type: '{type(self.best_model_checkpoint)}'"
            )

    def metrics(self, *args, **kwargs) -> Dict[str, float]:
        """
        See :meth:`ClassificationEvaluation.metrics`.
        """
        return self.evaluation.metrics(*args, **kwargs)

    def metrics_report(self, *args, **kwargs) -> str:
        """
        See :meth:`ClassificationEvaluation.metrics_report`.
        """
        return self.evaluation.metrics_report(*args, **kwargs)

    def plot(self, *args, **kwargs) -> alt.Chart:
        """
        See :meth:`ClassificationEvaluation.plot`.
        """
        return self.evaluation.plot(*args, **kwargs)

    def errors(
        self, *args, **kwargs
    ) -> Dict[str, Tuple[List[ClassificationError], List[ClassificationError]]]:
        """
        See :meth:`ClassificationEvaluation.errors`.
        """
        return self.evaluation.errors(*args, **kwargs)

    def errors_report(self, *args, **kwargs) -> str:
        """
        See :meth:`ClassificationEvaluation.errors_report`.
        """
        return self.evaluation.errors_report(*args, **kwargs)


@dataclass
class RemoteTrainResult:
    """
    Results from a training process on a (possibly remote) worker.

    Args:
      metadata: Metadata from the training output.
      labels: List of labels identified in the data.
      checkpoint_name: Name of the checkpoint under the checkpoint directory.
      checkpoint_id: ray ObjectID for the checkpoint directory (path or bytes)
      model_params: Parameters used to initialize the model.
      ip_address: IP address of the node that ran training.
    """

    metadata: Dict[str, Any]
    labels: List[str]
    checkpoint_name: str
    checkpoint_id: ray.ObjectID
    model_params: Dict[str, Any]
    ip_address: str


class ClassificationExperiment(BaseExperiment):
    """
    Run a classification experiment.  This entails training a model to
    make predictions given some input.

    The experiment will, for each combination of model hyperparameters,
    train the model on a training set and evaluate it on a validation set.
    The best combination of hyperparameters will be retrained on the combined
    training/validation sets and evaluated on the test set. After completion,
    the experiment will return :class:`ClassificationExperimentResults`, which will
    allow the user to examine the results in various ways.
    """

    _DEFAULT_TRAIN_VALID_TEST_SPLIT = (0.7, 0.1, 0.2)
    _DEFAULT_TRAIN_VALID_SPLIT = (0.8, 0.2)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ClassificationExperiment._validate_model_cls(self.model_cls)

    @staticmethod
    def _validate_model_cls(model_cls: Any):
        if not (
            issubclass(model_cls, TrainMixin) and issubclass(model_cls, PredictMixin)
        ):
            raise ValueError(
                f"Model of class {model_cls.__name__} can't be used for a"
                " classification experiment because it doesn't support both"
                " training and prediction."
            )

    @staticmethod
    def _validate_split(
        train_valid_test_split: Union[Tuple[float, float, float], Tuple[float, float]],
        expected_len=3,
    ):
        if (
            not len(train_valid_test_split) == expected_len
            or sum(train_valid_test_split) != 1
        ):
            raise ValueError(
                f"train_valid_test_split must be length {expected_len} and sum to 1"
            )

    def run(
        self,
        dataset_split: Optional[
            Union[Tuple[float, float], Tuple[float, float, float]]
        ] = None,
        seed: int = 1,
        train_batch_size: int = 32,
        valid_batch_size: int = 32,
        test_batch_size: int = 32,
        num_train_epochs: int = 5,
    ) -> ClassificationExperimentResults:
        """
        Run the experiment.

        Args:
          dataset_split: A tuple describing the proportion of the dataset
            to be added to the train/validation/test splits.  If the experiment uses an explicit
            test set (passes :paramref:`BaseExperiment.params.test_dataset`), this should be a
            2-tuple describing the train/validation split.  Otherwise, it should be a 3-tuple
            describing the train/validation/test split. The tuple must sum to 1.
          seed: Random seed to be used for dataset splitting for reproducibility.
          train_batch_size: Number of observations per batch on the training dataset.
          valid_batch_size: Number of observations per batch on the validation dataset.
          test_batch_size: Number of observations per batch on the test dataset.
          num_train_epochs: Number of epochs to use for training.

        Returns:
          The results of the experiment.
        """
        _dataset_split = dataset_split

        # If the user didn't pass an explicit test set, create one
        # using a split
        if self.X_test is None:
            if _dataset_split is None:
                _dataset_split = (
                    ClassificationExperiment._DEFAULT_TRAIN_VALID_TEST_SPLIT
                )

            ClassificationExperiment._validate_split(_dataset_split, expected_len=3)

            # cast needed to satisfy mypy
            train_prop, valid_prop, test_prop = cast(
                Tuple[float, float, float], _dataset_split
            )
            train_valid_prop = train_prop + valid_prop

            X_train_valid, X_test, y_train_valid, y_test = train_test_split(
                self.X, self.y, train_size=train_valid_prop, test_size=test_prop
            )

        else:
            if _dataset_split is None:
                _dataset_split = ClassificationExperiment._DEFAULT_TRAIN_VALID_SPLIT

            ClassificationExperiment._validate_split(_dataset_split, expected_len=2)

            # cast needed to satisfy mypy
            train_prop, valid_prop = cast(Tuple[float, float], _dataset_split)
            train_valid_prop = 1

            X_train_valid, y_train_valid = self.X, self.y
            X_test, y_test = self.X_test, self.y_test

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_valid,
            y_train_valid,
            # Round to prevent floating point imprecision errors
            train_size=round(train_prop / train_valid_prop, 4),
            test_size=round(valid_prop / train_valid_prop, 4),
        )

        if self.param_grid is not None:
            for param, values in self.param_grid.items():
                if isinstance(values, str):
                    raise TypeError(
                        f"String detected in parameter grid values for parameter '{param}'. "
                        "This will be treated as a list of character parameter values, "
                        "which probably isn't what you want.  If you're really sure, "
                        "convert the string to a list of characters and try again."
                    )
        grid = ParameterGrid(self.param_grid)
        if len(grid) == 0:
            raise ValueError("empty parameter grid")

        dataset_ids = [ray.put(d) for d in (X_train, y_train, X_valid, y_valid)]

        # Return the checkpoint blob separately from the train result so it doesn't
        # have to be copied to the object store again when used by the predict function
        @ray.remote(num_cpus=self.task_num_cpus, num_gpus=self.task_num_gpus)
        def train(
            X_train: Any,
            y_train: Any,
            X_valid: Any,
            y_valid: Any,
            train_batch_size: int,
            valid_batch_size: int,
            num_train_epochs: int,
            model_cls: Any,
            model_params: Dict[str, Any],
            gobbli_dir: Optional[Path] = None,
            log_level: Union[int, str] = logging.WARNING,
            local_mode: bool = False,
            distributed: bool = False,
        ) -> RemoteTrainResult:

            logger = init_worker_env(gobbli_dir=gobbli_dir, log_level=log_level)
            use_gpu, nvidia_visible_devices = init_gpu_config()

            if not distributed and len(ray.nodes()) > 1:
                raise RuntimeError(
                    "Experiments must be started with distributed = True to run "
                    "tasks on remote workers."
                )

            clf = model_cls(
                **model_params,
                use_gpu=use_gpu,
                nvidia_visible_devices=nvidia_visible_devices,
                logger=logger,
            )

            clf.build()

            train_input = gobbli.io.TrainInput(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                train_batch_size=train_batch_size,
                valid_batch_size=valid_batch_size,
                num_train_epochs=num_train_epochs,
            )
            train_output = clf.train(train_input)
            checkpoint = train_output.checkpoint
            checkpoint_name = getattr(checkpoint, "name", None)

            if distributed:
                # Copy weights into the object store, since we don't share a filesystem
                # with the master node
                checkpoint = (
                    dir_to_blob(checkpoint.parent) if checkpoint is not None else None
                )

            # Ray throws an error here if we try to put this checkpoint in local mode
            if not local_mode:
                checkpoint = ray.put(checkpoint)

            worker_ip = get_worker_ip()

            return RemoteTrainResult(
                metadata=train_output.metadata(),
                labels=train_output.labels,
                checkpoint_name=checkpoint_name,
                checkpoint_id=checkpoint,
                model_params=model_params,
                ip_address=worker_ip,
            )

        @ray.remote(num_cpus=self.task_num_cpus, num_gpus=self.task_num_gpus)
        def predict(
            X_test: List[str],
            test_batch_size: int,
            model_cls: Any,
            model_params: Dict[str, Any],
            labels: List[str],
            checkpoint: Union[bytes, Path],
            checkpoint_name: Optional[str],
            gobbli_dir: Optional[Path] = None,
            log_level: Union[int, str] = logging.WARNING,
            distributed: bool = False,
        ) -> pd.DataFrame:

            logger = init_worker_env(gobbli_dir=gobbli_dir, log_level=log_level)
            use_gpu, nvidia_visible_devices = init_gpu_config()

            if not distributed and len(ray.nodes()) > 1:
                raise RuntimeError(
                    "Experiments must be started with distributed = True to run "
                    "tasks on remote workers."
                )

            clf = model_cls(
                **model_params,
                use_gpu=use_gpu,
                nvidia_visible_devices=nvidia_visible_devices,
                logger=logger,
            )

            # This step isn't necessary in all cases if the build step just downloads
            # pretrained weights we weren't going to use anyway, but sometimes it's needed
            # Ex. for BERT to download vocabulary files and config
            clf.build()

            # Use the current working directory (CWD) as the base for the tempdir, under the
            # assumption that the CWD is included in any bind mounts/volumes the user may have
            # created if they're running this in a Docker container
            # If it's not part of a host mount, the files won't be mounted properly in the container
            with tempfile.TemporaryDirectory(dir=".") as tempdir:
                tempdir_path = Path(tempdir)

                checkpoint_path = None  # type: Optional[Path]
                if isinstance(checkpoint, bytes):
                    if checkpoint_name is not None:
                        blob_to_dir(checkpoint, tempdir_path)
                        checkpoint_path = tempdir_path / checkpoint_name
                elif isinstance(checkpoint, Path):
                    checkpoint_path = checkpoint
                elif checkpoint is None:
                    pass
                else:
                    raise TypeError(f"invalid checkpoint type: '{type(checkpoint)}'")

                predict_input = gobbli.io.PredictInput(
                    X=X_test,
                    labels=labels,
                    checkpoint=checkpoint_path,
                    predict_batch_size=test_batch_size,
                )
                predict_output = clf.predict(predict_input)

            return predict_output.y_pred_proba

        # Run training in parallel using the Ray cluster
        raw_results = ray.get(
            [
                train.remote(
                    *dataset_ids,
                    train_batch_size,
                    valid_batch_size,
                    num_train_epochs,
                    self.model_cls,
                    params,
                    self.worker_gobbli_dir,
                    self.worker_log_level,
                    self.is_ray_local_mode,
                    self.distributed,
                )
                for params in grid
            ]
        )

        training_results = []  # type: List[Dict[str, Any]]
        best_valid_loss = math.inf
        best_result = None  # type: Optional[RemoteTrainResult]
        best_checkpoint_id = None  # type: Optional[ray.ObjectID]

        for train_results in raw_results:
            result = {
                **train_results.metadata,
                "node_ip_address": train_results.ip_address,
                "model_params": train_results.model_params,
            }
            if result["valid_loss"] < best_valid_loss:
                best_result = train_results
                best_checkpoint_id = train_results.checkpoint_id
                best_valid_loss = result["valid_loss"]

            training_results.append(result)

        if best_result is None:
            raise ValueError(
                "failed to find parameter combination with finite validation loss"
            )

        # Evaluate the best model on the test set
        # TODO is this no longer necessary?
        # if is_ray_local_mode():
        #     X_test_id = X_test
        # else:
        X_test_id = ray.put(X_test)
        y_pred_proba = ray.get(
            predict.remote(
                X_test_id,
                test_batch_size,
                self.model_cls,
                best_result.model_params,
                best_result.labels,
                best_checkpoint_id,
                best_result.checkpoint_name,
                self.worker_gobbli_dir,
                self.worker_log_level,
                self.distributed,
            )
        )

        best_checkpoint = best_checkpoint_id
        # We will have stored the checkpoint in the object store if not
        # running in local mode
        if not self.is_ray_local_mode:
            best_checkpoint = ray.get(best_checkpoint_id)

        return ClassificationExperimentResults(
            X=X_test,
            labels=best_result.labels,  # type: ignore
            y_true=y_test,
            y_pred_proba=y_pred_proba,
            training_results=training_results,
            best_model_checkpoint=cast(Union[bytes, Path], best_checkpoint),
            best_model_checkpoint_name=best_result.checkpoint_name,
        )
