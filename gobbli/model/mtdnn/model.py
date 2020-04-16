import json
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import gobbli.io
from gobbli.docker import maybe_mount, run_container
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.model.mixin import PredictMixin, TrainMixin
from gobbli.util import (
    assert_in,
    assert_type,
    copy_file,
    download_file,
    escape_line_delimited_texts,
)

MTDNN_MODEL_FILES = {
    "mt-dnn-base": "https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_base.pt",
    "mt-dnn-large": "https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_large.pt",
}
"""
A mapping from model names to weight files.
"mt-dnn-base" is a safe default for most situations.
Larger models require more time and GPU memory to run.
"""


def _preprocess_text(text_series: pd.Series) -> pd.Series:
    """
    Preprocess a Series of text for the MT-DNN uncased model.
    """
    return text_series.str.lower().str.replace(r"\s", " ", regex=True)


def _df_to_tsv(df: pd.DataFrame, output_file: Path):
    """
    Write a dataframe with "X" and "y" (optional) columns to the given
    output file in the format expected by MT-DNN for a TSV file.
    """
    df.loc[:, "X"] = _preprocess_text(df["X"])
    df.to_csv(output_file, index=False, header=True)


def _write_labels(labels: List[Any], output_file: Path):
    """
    Write the given set of labels to the given file.
    """
    output_file.write_text(escape_line_delimited_texts(labels))


class MTDNN(BaseModel, TrainMixin, PredictMixin):
    """
    Classifier wrapper for Microsoft's MT-DNN:
    https://github.com/namisan/mt-dnn
    """

    _BUILD_PATH = Path(__file__).parent

    _TRAIN_INPUT_FILE = "train.csv"
    _VALID_INPUT_FILE = "valid.csv"
    _TEST_INPUT_FILE = "test.csv"
    _LABELS_INPUT_FILE = "labels.csv"

    _PREDICT_OUTPUT_FILE = "predict.csv"

    _LOG_FILE = "log.log"

    _WEIGHTS_FILE_NAME = "weights.pt"

    def init(self, params: Dict[str, Any]):
        """
        See :meth:`gobbli.model.base.BaseModel.init`.

        MT-DNN parameters:

        - ``max_seq_length`` (:obj:`int`): The maximum total input sequence length after
          WordPiece tokenization.  Sequences longer than this will be truncated,
          and sequences shorter than this will be padded.  Default: 128
        - ``mtdnn_model`` (:obj:`str`): Name of a pretrained MT-DNN model to use.
          See :obj:`MTDNN_MODEL_FILES` for a listing of available MT-DNN models.
        """
        self.max_seq_length = 128
        self.mtdnn_model = "mt-dnn-base"

        for name, value in params.items():
            if name == "max_seq_length":
                assert_type(name, value, int)
                self.max_seq_length = value
            elif name == "mtdnn_model":
                assert_in(name, value, set(MTDNN_MODEL_FILES.keys()))
                self.mtdnn_model = value
            else:
                raise ValueError(f"Unknown param '{name}'")

    @property
    def weights_dir(self) -> Path:
        """
        Returns:
          The directory containing pretrained weights for this instance.
        """
        return self.class_weights_dir / self.mtdnn_model

    @property
    def image_tag(self) -> str:
        """
        Returns:
          The Docker image tag to be used for the MT-DNN container.
        """
        return "gobbli-mt-dnn-classifier"

    def _build(self):
        # Download data if we don't already have it
        if not self.weights_dir.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_weights_dir = Path(tmpdir) / self.weights_dir.name
                tmp_weights_dir.mkdir()
                self.logger.info("Downloading pre-trained weights.")
                downloaded_file = download_file(MTDNN_MODEL_FILES[self.mtdnn_model])
                copy_file(downloaded_file, tmp_weights_dir / MTDNN._WEIGHTS_FILE_NAME)
                shutil.move(tmp_weights_dir, self.weights_dir)
                self.logger.info("Weights downloaded.")

        # Build the docker image
        self.docker_client.images.build(
            path=str(MTDNN._BUILD_PATH),
            tag=self.image_tag,
            **self._base_docker_build_kwargs,
        )

    @staticmethod
    def _get_checkpoint(
        user_checkpoint: Optional[Path], context: ContainerTaskContext
    ) -> Tuple[Optional[Path], Path]:
        """
        Determines the host checkpoint file and container checkpoint file
        using the user-requested checkpoint (if any) and the container context.

        Args:
          user_checkpoint: An optional checkpoint passed in by the user.  If the user doesn't
            pass one, use the default pretrained checkpoint.
          context: The container context to create the checkpoint in.

        Returns:
          A 2-tuple: the host checkpoint file (if any) and
            the container checkpoint file
        """
        if user_checkpoint is None:
            # Default weights
            host_checkpoint_file = None
            container_checkpoint_file = (
                BaseModel._CONTAINER_WEIGHTS_PATH / MTDNN._WEIGHTS_FILE_NAME
            )
        else:
            # Trained weights, which will be mounted in the container
            host_checkpoint_file = user_checkpoint
            container_checkpoint_file = context.container_root_dir / "checkpoint.pt"

        return host_checkpoint_file, container_checkpoint_file

    def _write_input(self, X: List[str], y: Optional[List[str]], input_file: Path):
        """
        Write the given gobbli input into the format expected by MT-DNN.
        Make sure the given directory exists first.
        """
        df_data = {"X": X}
        if y is not None:
            df_data["y"] = y

        df = pd.DataFrame(df_data)

        _df_to_tsv(df, input_file)

    def _train(
        self, train_input: gobbli.io.TrainInput, context: ContainerTaskContext
    ) -> gobbli.io.TrainOutput:
        if train_input.multilabel:
            raise ValueError(
                "gobbli MT-DNN model doesn't support multilabel classification."
            )

        self._write_input(
            train_input.X_train,
            train_input.y_train_multiclass,
            context.host_input_dir / MTDNN._TRAIN_INPUT_FILE,
        )
        self._write_input(
            train_input.X_valid,
            train_input.y_valid_multiclass,
            context.host_input_dir / MTDNN._VALID_INPUT_FILE,
        )

        labels = train_input.labels()
        labels_path = context.host_input_dir / MTDNN._LABELS_INPUT_FILE
        _write_labels(labels, labels_path)

        if train_input.valid_batch_size != train_input.train_batch_size:
            warnings.warn(
                "MT-DNN model does not support separate validation batch size; "
                f"using train batch size '{train_input.train_batch_size}' for both "
                "training and validation."
            )

        # Determine checkpoint to use
        host_checkpoint_file, container_checkpoint_file = self._get_checkpoint(
            train_input.checkpoint, context
        )

        cmd = (
            "python gobbli_train.py"
            " --data_dir=data/mt_dnn"
            f" --init_checkpoint={container_checkpoint_file}"
            f" --batch_size={train_input.train_batch_size}"
            f" --output_dir={context.container_output_dir}"
            f" --log_file={context.container_output_dir / MTDNN._LOG_FILE}"
            " --optimizer=adamax"
            " --grad_clipping=0"
            " --global_grad_clipping=1"
            " --lr=2e-5"
            f" --train_file={context.container_input_dir / MTDNN._TRAIN_INPUT_FILE}"
            f" --valid_file={context.container_input_dir / MTDNN._VALID_INPUT_FILE}"
            f" --label_file={context.container_input_dir / MTDNN._LABELS_INPUT_FILE}"
            f" --epochs={train_input.num_train_epochs}"
            f" --max_seq_len={self.max_seq_length}"
        )

        run_kwargs = self._base_docker_run_kwargs(context)

        # Mount the checkpoint in the container if needed
        maybe_mount(
            run_kwargs["volumes"], host_checkpoint_file, container_checkpoint_file
        )

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        # MT-DNN counts epochs starting from 0
        final_epoch = train_input.num_train_epochs - 1

        # Parse the generated evaluation results files
        eval_results = {}  # type: Dict[str, Any]
        for name in ("train", "valid"):
            results_file = context.host_output_dir / f"{name}_scores_{final_epoch}.json"
            with open(results_file, "r") as f:
                results = json.load(f)
                eval_results.update(
                    {f"{name}_{key}": val for key, val in results.items()}
                )

        return gobbli.io.TrainOutput(
            valid_loss=eval_results["valid_metrics"]["loss"],
            valid_accuracy=eval_results["valid_metrics"]["accuracy"] / 100,
            train_loss=eval_results["train_metrics"]["loss"],
            labels=labels,
            multilabel=False,
            checkpoint=context.host_output_dir / f"model_{final_epoch}.pt",
            _console_output=container_logs,
        )

    def _predict(
        self, predict_input: gobbli.io.PredictInput, context: ContainerTaskContext
    ) -> gobbli.io.PredictOutput:
        self._write_input(
            predict_input.X, None, context.host_input_dir / MTDNN._TEST_INPUT_FILE
        )

        labels_path = context.host_input_dir / MTDNN._LABELS_INPUT_FILE
        _write_labels(predict_input.labels, labels_path)

        # Determine checkpoint to use
        host_checkpoint_file, container_checkpoint_file = self._get_checkpoint(
            predict_input.checkpoint, context
        )

        cmd = (
            "python gobbli_train.py"
            " --data_dir=data/mt_dnn"
            f" --init_checkpoint={container_checkpoint_file}"
            f" --batch_size={predict_input.predict_batch_size}"
            f" --output_dir={context.container_output_dir}"
            f" --log_file={context.container_output_dir / MTDNN._LOG_FILE}"
            " --optimizer=adamax"
            " --grad_clipping=0"
            " --global_grad_clipping=1"
            " --lr=2e-5"
            f" --test_file={context.container_input_dir / MTDNN._TEST_INPUT_FILE}"
            f" --label_file={context.container_input_dir / MTDNN._LABELS_INPUT_FILE}"
            f" --max_seq_len={self.max_seq_length}"
        )

        run_kwargs = self._base_docker_run_kwargs(context)

        # Mount the checkpoint in the container if needed
        maybe_mount(
            run_kwargs["volumes"], host_checkpoint_file, container_checkpoint_file
        )

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        # Retrieve the generated predictions
        return gobbli.io.PredictOutput(
            y_pred_proba=pd.read_csv(
                context.host_output_dir / MTDNN._PREDICT_OUTPUT_FILE
            ),
            _console_output=container_logs,
        )
