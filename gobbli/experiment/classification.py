import logging
import math
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import matplotlib
import pandas as pd
import ray
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split

import gobbli.io
from gobbli.experiment.base import (
    BaseExperiment,
    get_worker_ip,
    init_gpu_config,
    init_worker_env,
    is_ray_local_mode,
)
from gobbli.model.mixin import PredictMixin, TrainMixin
from gobbli.util import (
    blob_to_dir,
    dir_to_blob,
    escape_line_delimited_text,
    pred_prob_to_pred_label,
    truncate_text,
)

MetricFunc = Callable[[Sequence[str], pd.DataFrame], float]
"""
A function used to calculate some metric.  It should accept a sequence of true labels (y_true)
and a dataframe of shape (n_samples, n_classes) containing predicted probabilities; it should
output a real number.
"""

DEFAULT_METRICS: Dict[str, MetricFunc] = {
    "Weighted F1 Score": lambda y_true, y_pred_proba: f1_score(
        y_true, pred_prob_to_pred_label(y_pred_proba), average="weighted"
    ),
    "Weighted Precision Score": lambda y_true, y_pred_proba: precision_score(
        y_true, pred_prob_to_pred_label(y_pred_proba), average="weighted"
    ),
    "Weighted Recall Score": lambda y_true, y_pred_proba: recall_score(
        y_true, pred_prob_to_pred_label(y_pred_proba), average="weighted"
    ),
    "Accuracy": lambda y_true, y_pred_proba: accuracy_score(
        y_true, pred_prob_to_pred_label(y_pred_proba)
    ),
}
"""
The default set of metrics to be reported in experiment results.  Users may want to extend
this.
"""


@dataclass
class ClassificationError:
    """
    Describes an error in classification.  Reports the original text,
    the true label, and the predicted probability.

    Args:
      X: The original text.
      y_true: The true label.
      y_pred_proba: The model predicted probability for each class.
    """

    X: str
    y_true: str
    y_pred_proba: Dict[str, float]

    @property
    def y_pred(self) -> str:
        """
        Returns:
          The predicted class for this observation.
        """
        return max(self.y_pred_proba, key=lambda k: self.y_pred_proba[k])


@dataclass
class ClassificationExperimentResults:
    """
    Results from a classification experiment.  An experiment entails training a set of models
    based on a grid of parameters, retraining on the full train/validation dataset with the
    best set of parameters, and evaluating predictions on the test set.

    Args:
      training_results: A list of dictionaries containing information about each training run,
        one for each unique combination of hyperparameters in :paramref:`BaseExperiment.params.param_grid`.
      y_true: The true labels for the test set, as passed by the user.
      y_pred_proba: A dataframe containing a row for each observation in the test set and a
        column for each label in the training data.  Cells are predicted probabilities.
      best_model_checkpoint: If results came from another process on the master node, this is
        the directory containing the checkpoint.  If the results came from a worker node, this
        is a bytes object containing the compressed model weights.
      best_model_checkpoint_name: Path to the best checkpoint within the directory or
        or compressed blob.
    """

    training_results: List[Dict[str, Any]]
    labels: List[str]
    X: List[str]
    y_true: List[str]
    y_pred_proba: pd.DataFrame
    best_model_checkpoint: Union[bytes, Path]
    best_model_checkpoint_name: str
    metric_funcs: Optional[Dict[str, Callable[[Sequence, Sequence], float]]] = None

    def __post_init__(self):
        if not len(self.y_true) == self.y_pred_proba.shape[0]:
            raise ValueError(
                "y_true and y_pred_proba must have the same number of observations"
            )

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

    @property
    def y_pred(self) -> List[str]:
        """
        Returns:
          The predicted class for each observation.
        """
        return pred_prob_to_pred_label(self.y_pred_proba)

    def metrics(self) -> Dict[str, float]:
        """
        Returns:
          A dictionary containing various metrics of model performance on the test dataset.
        """
        metric_funcs = self.metric_funcs
        if metric_funcs is None:
            metric_funcs = DEFAULT_METRICS

        return {
            name: metric_func(self.y_true, self.y_pred_proba)
            for name, metric_func in metric_funcs.items()
        }

    def metrics_report(self) -> str:
        """
        Returns:
          A nicely formatted human-readable report describing metrics of model performance
          on the test dataset.
        """
        metric_string = "\n".join(
            f"{name}: {metric}" for name, metric in self.metrics().items()
        )
        return (
            "Metrics:\n"
            "--------\n"
            f"{metric_string}\n\n"
            "Classification Report:\n"
            "----------------------\n"
            f"{classification_report(self.y_true, self.y_pred)}\n"
        )

    def plot(self, ax: Optional[matplotlib.axes.Axes] = None) -> matplotlib.axes.Axes:
        """
        Returns:
          A plot visualizing predicted probabilities and true classes to visually identify
          where errors are being made.
        """
        pred_prob_df = self.y_pred_proba.copy()
        pred_prob_df["True Class"] = self.y_true

        plot_df = pred_prob_df.melt(
            id_vars=["True Class"], var_name="Class", value_name="Predicted Probability"
        )
        plot_df["Belongs to Class"] = plot_df["True Class"] == plot_df["Class"]

        plot_ax = sns.stripplot(
            y="Class",
            x="Predicted Probability",
            hue="Belongs to Class",
            dodge=True,
            data=plot_df,
            size=3,
            palette="muted",
            ax=ax,
        )
        plot_ax.set_xticks([0, 0.5, 1])
        plot_ax.legend(loc="lower right", framealpha=0, fontsize="small")
        return plot_ax

    def errors(
        self, k: int = 10
    ) -> Dict[str, Tuple[List[ClassificationError], List[ClassificationError]]]:
        """
        Output the biggest mistakes for each class by the classifier.

        Args:
          k: The number of results to return for each of false positives and false negatives.

        Returns:
          A dictionary whose keys are class names and values are 2-tuples.  The first
          element is a list of the top ``k`` false positives, and the second element is a list
          of the top ``k`` false negatives.
        """
        errors = {}
        y_pred_series = pd.Series(self.y_pred)
        y_true_series = pd.Series(self.y_true)

        error_pred_prob = self.y_pred_proba[y_pred_series != y_true_series]
        for label in self.labels:
            pred_label = y_pred_series == label
            true_label = y_true_series == label

            # Order false positives/false negatives by the degree of the error;
            # i.e. we want the false positives with highest predicted probability first
            # and false negatives with lowest predicted probability first
            # Take the top `k` of each
            false_positives = (
                error_pred_prob.loc[pred_label & ~true_label]
                .sort_values(by=label, ascending=False)
                .iloc[:k]
            )
            false_negatives = (
                error_pred_prob.loc[~pred_label & true_label]
                .sort_values(by=label, ascending=True)
                .iloc[:k]
            )

            def create_classification_errors(
                y_pred_proba: pd.DataFrame,
            ) -> List[ClassificationError]:
                classification_errors = []
                for ndx, row in y_pred_proba.iterrows():
                    classification_errors.append(
                        ClassificationError(
                            X=self.X[ndx],
                            y_true=self.y_true[ndx],
                            y_pred_proba=row.to_dict(),
                        )
                    )
                return classification_errors

            errors[label] = (
                create_classification_errors(false_positives),
                create_classification_errors(false_negatives),
            )

        return errors

    def errors_report(self, k: int = 10) -> str:
        """
        Args:
          k: The number of results to return for each of false positives and false negatives.

        Returns:
          A nicely-formatted human-readable report describing the biggest mistakes made by
          the classifier for each class.
        """
        errors = self.errors(k=k)
        output = "Errors Report\n" "------------\n\n"

        for label, (false_positives, false_negatives) in errors.items():

            def make_errors_str(errors: List[ClassificationError]) -> str:
                return "\n".join(
                    (
                        f"True Class: {e.y_true}\n"
                        f"Predicted Class: {e.y_pred} (Probability: {e.y_pred_proba[e.y_pred]})\n"
                        f"Text: {truncate_text(escape_line_delimited_text(e.X), 500)}\n"
                    )
                    for e in errors
                )

            false_positives_str = make_errors_str(false_positives)
            if len(false_positives_str) == 0:
                false_positives_str = "None"
            false_negatives_str = make_errors_str(false_negatives)
            if len(false_negatives_str) == 0:
                false_negatives_str = "None"

            output += (
                " -------\n"
                f"| CLASS: {label}\n"
                " -------\n\n"
                "False Positives\n"
                "***************\n\n"
                f"{false_positives_str}\n\n"
                "False Negatives\n"
                "***************\n\n"
                f"{false_negatives_str}\n\n"
            )

        return output


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
          train_valid_test_split: A tuple describing the proportion of the dataset
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

        grid = ParameterGrid(self.param_grid)
        if len(grid) == 0:
            raise ValueError("empty parameter grid")

        # Transfer datasets to the Ray distributed object store
        # if not running in local mode
        # In local mode, this causes problems: https://github.com/ray-project/ray/issues/5379
        if is_ray_local_mode():
            dataset_ids = [X_train, y_train, X_valid, y_valid]
        else:
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
            master_ip: str,
            gobbli_dir: Optional[Path] = None,
            log_level: Union[int, str] = logging.WARNING,
            distributed: bool = False,
        ) -> RemoteTrainResult:

            logger = init_worker_env(gobbli_dir=gobbli_dir, log_level=log_level)
            use_gpu, nvidia_visible_devices = init_gpu_config()

            worker_ip = get_worker_ip()
            if not distributed and worker_ip != master_ip:
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

            if not is_ray_local_mode():
                checkpoint = ray.put(checkpoint)

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
            master_ip: str,
            gobbli_dir: Optional[Path] = None,
            log_level: Union[int, str] = logging.WARNING,
            distributed: bool = False,
        ) -> pd.DataFrame:

            logger = init_worker_env(gobbli_dir=gobbli_dir, log_level=log_level)
            use_gpu, nvidia_visible_devices = init_gpu_config()

            worker_ip = get_worker_ip()
            if not distributed and worker_ip != master_ip:
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

        # Record the IP address of the master node so workers can detect
        # whether they're remote and not running in distributed mode, at which
        # point they should raise an error
        master_ip = get_worker_ip()

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
                    master_ip,
                    self.worker_gobbli_dir,
                    self.worker_log_level,
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
        if is_ray_local_mode():
            X_test_id = X_test
        else:
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
                master_ip,
                self.worker_gobbli_dir,
                self.worker_log_level,
                self.distributed,
            )
        )

        best_checkpoint = best_checkpoint_id
        if not is_ray_local_mode():
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
