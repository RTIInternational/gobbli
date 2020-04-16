import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import gobbli.io
from gobbli.docker import maybe_mount, run_container
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.model.mixin import EmbedMixin, PredictMixin, TrainMixin
from gobbli.util import (
    assert_in,
    assert_type,
    download_archive,
    escape_line_delimited_text,
    multilabel_to_indicator_df,
    pred_prob_to_pred_label,
    pred_prob_to_pred_multilabel,
)

FASTTEXT_VECTOR_ARCHIVES = {
    "wiki-news-300d": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
    "wiki-news-300d-subword": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip",
    "crawl-300d": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",
    "crawl-300d-subword": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip",
    "wiki-crawl-300d": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz",
    "wiki-aligned-300d": "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec",
}
"""
A mapping from pretrained vector names to archives.
See `the fastText docs <https://fasttext.cc/docs/en/english-vectors.html>`__ for information
about each set of vectors.  Note, some sets of vectors are very, very large.
"""

_DIM_REGEX = re.compile(r"([0-9]+)d")


def _parse_dim(model_name: str) -> int:
    """
    Parse the number of dimensions from a FastText model name.
    """
    match = _DIM_REGEX.search(model_name)
    if match is None:
        raise ValueError(
            f"Failed to parse number of dimensions from model name {model_name}"
        )

    return int(match.group(1))


def _fasttext_preprocess(text: str) -> str:
    """
    Preprocess text for the fasttext model.

    Lowercase text and escape newlines.  Removing or separately
    tokenizing punctuation is recommended, but there are too many different
    ways to do it, so we leave that up to the user.
    """
    return escape_line_delimited_text(text).lower()


@dataclass
class FastTextCheckpoint:
    path: Path

    @property
    def vectors(self) -> Path:
        """
        From a checkpoint, return the path to the text vectors.
        """
        return self.path.parent / f"{self.path.stem}.vec"

    @property
    def model(self) -> Path:
        """
        From a checkpoint, return the path to the binary model.
        """
        return self.path.parent / f"{self.path.stem}.bin"


class FastText(BaseModel, TrainMixin, PredictMixin, EmbedMixin):
    """
    Wrapper for Facebook's fastText model:
    https://github.com/facebookresearch/fastText

    Note: fastText benefits from some preprocessing steps:
    https://fasttext.cc/docs/en/supervised-tutorial.html#preprocessing-the-data

    gobbli will only lowercase and escape newlines in your input by default.
    If you want more sophisticated preprocessing for punctuation, stemming, etc,
    consider performing some preprocessing on your own beforehand.
    """

    _BUILD_PATH = Path(__file__).parent

    _TRAIN_INPUT_FILE = "train.txt"
    _VALID_INPUT_FILE = "valid.txt"
    _TEST_INPUT_FILE = "test.txt"

    _PREDICT_OUTPUT_FILE = "predict.txt"

    _EMBEDDING_INPUT_FILE = "input.txt"
    _EMBEDDING_OUTPUT_FILE = "embeddings.txt"

    _CHECKPOINT_BASE = "model"

    _LABEL_SPACE_ESCAPE = "$gobbli_space$"

    @property
    def image_tag(self) -> str:
        """
        Returns:
          The tag to use for the fastText image.
        """
        return "gobbli-fasttext"

    @property
    def weights_dir(self) -> Path:
        """
        Returns:
          The directory containing pretrained weights for this instance.
        """
        # Weights won't be used if we don't have a model to use
        if self.fasttext_model is None:
            return self.class_weights_dir
        return self.class_weights_dir / self.fasttext_model

    def init(self, params: Dict[str, Any]):
        """
        See :meth:`gobbli.model.base.BaseModel.init`.

        For more info on fastText parameter semantics, see
        `the docs <https://fasttext.cc/docs/en/options.html>`__.  The fastText
        `supervised tutorial <https://fasttext.cc/docs/en/supervised-tutorial.html>`__ has
        some more detailed explanation.

        fastText parameters:

        - ``word_ngrams`` (:obj:`int`): Max length of word n-grams.
        - ``lr`` (:obj:`float`): Learning rate.
        - ``dim`` (:obj:`int`): Dimension of learned vectors.
        - ``ws`` (:obj:`int`): Context window size.
        - ``autotune_duration`` (:obj:`int`): Duration in seconds to spend autotuning parameters.
          Any of the above parameters will not be autotuned if they are manually specified.
        - ``autotune_modelsize`` (:obj:`str`): Maximum size of autotuned model (ex "2M" for 2
          megabytes).  Any of the above parameters will not be autotuned if they are manually
          specified.
        - ``fasttext_model`` (:obj:`str`): Name of a pretrained fastText model to use.
          See :obj:`FASTTEXT_VECTOR_ARCHIVES` for a listing of available pretrained models.
        """
        self.word_ngrams = None
        self.lr = None
        self.ws = None
        self.fasttext_model = None
        # Default to dimensionality of the passed model, if any
        if "fasttext_model" in params:
            self.dim: Optional[int] = _parse_dim(params["fasttext_model"])
        else:
            self.dim = None
        self.autotune_duration = None
        self.autotune_modelsize = None

        for name, value in params.items():
            if name == "word_ngrams":
                assert_type(name, value, int)
                self.word_ngrams = value
            elif name == "lr":
                assert_type(name, value, float)
                self.lr = value
            elif name == "dim":
                assert_type(name, value, int)
                self.dim = value
            elif name == "ws":
                assert_type(name, value, int)
                self.ws = value
            elif name == "fasttext_model":
                assert_in(name, value, set(FASTTEXT_VECTOR_ARCHIVES.keys()))
                self.fasttext_model = value
            elif name == "autotune_duration":
                assert_type(name, value, int)
                self.autotune_duration = value
            elif name == "autotune_modelsize":
                assert_type(name, value, str)
                self.autotune_modelsize = value
            else:
                raise ValueError(f"Unknown param '{name}'")

        if (
            self.fasttext_model is not None
            and f"{self.dim}d" not in self.fasttext_model
        ):
            raise ValueError(
                "When using pretrained vectors, 'dim' must match the"
                f" dimensionality of the vectors; 'dim' value of {self.dim}"
                f" is incompatible with vectors {self.fasttext_model}."
            )

    def _build(self):
        # Download data if we need it and don't already have it
        if (
            self.fasttext_model is not None
            and not (self.weights_dir / self.fasttext_model).exists()
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_weights_dir = Path(tmpdir) / self.weights_dir.name
                tmp_weights_dir.mkdir()
                self.logger.info("Downloading pre-trained weights.")
                download_archive(
                    FASTTEXT_VECTOR_ARCHIVES[self.fasttext_model], tmp_weights_dir
                )
                shutil.move(tmp_weights_dir, self.weights_dir)
                self.logger.info("Weights downloaded.")

        # Build the custom docker image
        self.docker_client.images.build(
            path=str(FastText._BUILD_PATH),
            tag=self.image_tag,
            **self._base_docker_build_kwargs,
        )

    @staticmethod
    def _escape_label(label: str) -> str:
        """
        Escape a label for use in fastText's label format.  Spaces
        must be replaced, or the label will be interpreted as part of the text.

        Args:
          label: Label to escape

        Returns:
          The escaped label
        """
        return label.replace(" ", FastText._LABEL_SPACE_ESCAPE)

    @staticmethod
    def _unescape_label(label: str) -> str:
        """
        Reverse escaping for a label read from fastText's output format.

        Args:
          label: Label to unescape

        Returns:
          The unescaped label
        """
        return label.replace(FastText._LABEL_SPACE_ESCAPE, " ")

    @staticmethod
    def _locate_checkpoint(weights_dir: Path) -> FastTextCheckpoint:
        """
        Locate a fastText checkpoint under the given directory,
        regardless of its filename.

        Args:
          weights_dir: The directory to search for a checkpoint (not recursive).

        Returns:
          A fastText checkpoint.
        """
        candidates = list(weights_dir.glob("*.vec"))
        if len(candidates) == 0:
            raise ValueError(f"No weights files found in '{weights_dir}'.")
        elif len(candidates) > 1:
            raise ValueError(
                f"Multiple weights files found in '{weights_dir}': {candidates}"
            )

        return FastTextCheckpoint(path=candidates[0].parent / candidates[0].stem)

    def _get_checkpoint(
        self, user_checkpoint: Optional[Path], context: ContainerTaskContext
    ) -> Tuple[Optional[FastTextCheckpoint], Optional[FastTextCheckpoint]]:
        """
        Determines, if any, the host checkpoint file and container checkpoint file
        using the user-requested checkpoint and the container context.

        Args:
          user_checkpoint: An optional checkpoint passed in by the user.  If the user doesn't
            pass one, use the default pretrained checkpoint, if any, or no checkpoint.
          context: The container context to create the checkpoint in.

        Returns:
          A 2-tuple: the host checkpoint (if any) and
            the container checkpoint (if any)
        """
        host_checkpoint = None  # type: Optional[FastTextCheckpoint]
        container_checkpoint = None  # type: Optional[FastTextCheckpoint]

        if self.fasttext_model is None and user_checkpoint is None:
            # No pretrained vectors
            return host_checkpoint, container_checkpoint

        elif self.fasttext_model is not None and user_checkpoint is None:
            host_checkpoint = FastText._locate_checkpoint(self.weights_dir)
            container_checkpoint = FastTextCheckpoint(
                BaseModel._CONTAINER_WEIGHTS_PATH / host_checkpoint.path.name
            )
            return host_checkpoint, container_checkpoint

        else:  # user_checkpoint is not None; user_checkpoint overrides pretrained model

            # This should never happen by the conditional checks above
            assert user_checkpoint is not None

            host_checkpoint = FastTextCheckpoint(user_checkpoint)
            container_checkpoint = FastTextCheckpoint(
                BaseModel._CONTAINER_WEIGHTS_PATH / host_checkpoint.path.name
            )

            return host_checkpoint, container_checkpoint

    def _write_input(
        self, X: List[str], y: Optional[List[List[str]]], input_path: Path
    ):
        """
        Write the given input and labels (if any) into the format expected by fastText.
        Make sure the given directory exists first.
        """
        with open(input_path, "w") as f:
            if y is not None:
                for text, labels in zip(X, y):
                    label_str = " ".join(
                        f"__label__{FastText._escape_label(label)}" for label in labels
                    )
                    f.write(f"{label_str} {_fasttext_preprocess(text)}\n")
            elif y is None:
                for text in X:
                    f.write(f"{_fasttext_preprocess(text)}\n")

    def _run_supervised(
        self,
        user_checkpoint: Optional[Path],
        container_input_path: Path,
        container_output_path: Path,
        context: ContainerTaskContext,
        num_epochs: int,
        autotune_validation_file_path: Optional[Path] = None,
        freeze_vectors: bool = False,
    ) -> Tuple[str, float]:
        """
        Run the fastText "supervised" command.  Used for both training and getting
        validation loss.

        Args:
          user_checkpoint: A checkpoint passed by the user
          container_input_path: Path to the input file in the container
          container_output_path: Path to the output checkpoint in the container
          context: Container task context.
        validation_file_path: Optional file to use for autotune validation when training.
          freeze_vectors: If true, use 0 learning rate; train solely for
            the purpose of calculating loss.

        Returns:
          A 2-tuple: container logs and loss.
        """
        host_checkpoint, container_checkpoint = self._get_checkpoint(
            user_checkpoint, context
        )

        cmd = (
            "supervised"
            f" -input {container_input_path}"
            f" -output {container_output_path}"
            f" -epoch {num_epochs}"
        )

        if autotune_validation_file_path is not None:
            cmd += f" -autotune-validation {autotune_validation_file_path}"

        lr = self.lr
        if freeze_vectors:
            lr = 0.0
        if lr is not None:
            cmd += f" -lr {lr}"

        for arg_name, attr in (
            ("wordNgrams", "word_ngrams"),
            ("dim", "dim"),
            ("ws", "ws"),
            ("autotune-duration", "autotune_duration"),
            ("autotune-modelsize", "autotune_modelsize"),
        ):
            attr_val = getattr(self, attr)
            if attr_val is not None:
                cmd += f" -{arg_name} {attr_val}"

        run_kwargs = self._base_docker_run_kwargs(context)

        if host_checkpoint is not None and container_checkpoint is not None:
            maybe_mount(
                run_kwargs["volumes"],
                host_checkpoint.vectors,
                container_checkpoint.vectors,
            )
            cmd += f" -pretrainedVectors {container_checkpoint.vectors}"

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        # Parse the training loss out of the console output
        last_loss_ndx = container_logs.rfind("avg.loss:")
        failed_parse_msg = (
            "Failed to parse loss information from fastText container logs."
            " Run with debug logging to"
            " see why this might have happened."
        )
        if last_loss_ndx == -1:
            raise ValueError(failed_parse_msg)

        # Skip over the word "avg.loss:" - next field in the output is "ETA:"
        loss_start_ndx = last_loss_ndx + len("avg.loss:")
        loss_end_ndx = container_logs.find("ETA:", loss_start_ndx)
        loss = float(container_logs[loss_start_ndx:loss_end_ndx].strip())
        return container_logs, loss

    def _run_predict_prob(
        self,
        user_checkpoint: Path,
        labels: List[str],
        container_input_path: Path,
        context: ContainerTaskContext,
    ) -> Tuple[str, pd.DataFrame]:
        """
        Run the fastText "predict-prob" command.  Used for obtaining
        label predicted probabilities on a dataset.

        Args:
          container_trained_model_path: Trained model passed by the user (.bin file)
          labels: Set of all labels to be used in prediction.
          container_input_path: Path to the input file in the container.
          context: Container task context.

        Returns:
          A 2-tuple: container logs and a dataframe of predicted probabilities.
        """
        host_checkpoint, container_checkpoint = self._get_checkpoint(
            user_checkpoint, context
        )

        if host_checkpoint is None or container_checkpoint is None:
            raise ValueError("A trained checkpoint is required to run prediction.")

        host_output_path = context.host_output_dir / FastText._PREDICT_OUTPUT_FILE
        container_output_path = (
            context.container_output_dir / FastText._PREDICT_OUTPUT_FILE
        )

        cmd = (
            "bash -c './fasttext predict-prob"
            f" {container_checkpoint.model}"
            f" {container_input_path}"
            f" {len(labels)}"
            f" >{container_output_path}'"
        )

        run_kwargs = self._base_docker_run_kwargs(context)
        # Override the entrypoint so we can use 'bash -c ...' above
        run_kwargs["entrypoint"] = ""
        maybe_mount(
            run_kwargs["volumes"], host_checkpoint.model, container_checkpoint.model
        )

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        # Parse the predicted probabilities out of the output file
        pred_prob_data = []
        with open(host_output_path, "r") as f:
            for line in f:
                tokens = line.split()
                # Seems that fastText doesn't always return a probability for
                # every label, so start out with default = 0.0 so the shape of
                # the returned DataFrame will be consistent with the number
                # of labels
                row_data = {label: 0.0 for label in labels}
                for raw_label, prob in zip(tokens[0::2], tokens[1::2]):
                    # Strip the "__label__" prefix and undo escaping
                    label = FastText._unescape_label(raw_label[9:])
                    row_data[label] = float(prob)
                pred_prob_data.append(row_data)

        return (container_logs, pd.DataFrame(pred_prob_data))

    def _train(
        self, train_input: gobbli.io.TrainInput, context: ContainerTaskContext
    ) -> gobbli.io.TrainOutput:
        self._write_input(
            train_input.X_train,
            train_input.y_train_multilabel,
            context.host_input_dir / FastText._TRAIN_INPUT_FILE,
        )
        self._write_input(
            train_input.X_valid,
            train_input.y_valid_multilabel,
            context.host_input_dir / FastText._VALID_INPUT_FILE,
        )

        container_validation_input_path = (
            context.container_input_dir / FastText._VALID_INPUT_FILE
        )
        train_logs, train_loss = self._run_supervised(
            train_input.checkpoint,
            context.container_input_dir / FastText._TRAIN_INPUT_FILE,
            context.container_output_dir / FastText._CHECKPOINT_BASE,
            context,
            train_input.num_train_epochs,
            autotune_validation_file_path=container_validation_input_path,
        )

        host_checkpoint_path = context.host_output_dir / f"{FastText._CHECKPOINT_BASE}"

        labels = train_input.labels()

        # Calculate validation accuracy on our own, since the CLI only provides
        # precision/recall
        predict_logs, pred_prob_df = self._run_predict_prob(
            host_checkpoint_path, labels, container_validation_input_path, context
        )

        if train_input.multilabel:
            pred_labels = pred_prob_to_pred_multilabel(pred_prob_df)
            gold_labels = multilabel_to_indicator_df(
                train_input.y_valid_multilabel, labels
            )
        else:
            pred_labels = pred_prob_to_pred_label(pred_prob_df)
            gold_labels = train_input.y_valid_multiclass

        valid_accuracy = accuracy_score(gold_labels, pred_labels)

        # Not ideal, but fastText doesn't provide a way to get validation loss;
        # Negate the validation accuracy instead
        valid_loss = -valid_accuracy

        return gobbli.io.TrainOutput(
            train_loss=train_loss,
            valid_loss=valid_loss,
            valid_accuracy=valid_accuracy,
            labels=labels,
            multilabel=train_input.multilabel,
            checkpoint=host_checkpoint_path,
            _console_output="\n".join((train_logs, predict_logs)),
        )

    def _predict(
        self, predict_input: gobbli.io.PredictInput, context: ContainerTaskContext
    ) -> gobbli.io.PredictOutput:
        host_input_path = context.host_input_dir / FastText._TEST_INPUT_FILE
        self._write_input(predict_input.X, None, host_input_path)
        container_input_path = context.to_container(host_input_path)

        if predict_input.checkpoint is None:
            raise ValueError("fastText requires a trained checkpoint for prediction.")

        predict_logs, pred_prob_df = self._run_predict_prob(
            predict_input.checkpoint,
            predict_input.labels,
            container_input_path,
            context,
        )

        return gobbli.io.PredictOutput(
            y_pred_proba=pred_prob_df, _console_output=predict_logs
        )

    def _embed(
        self, embed_input: gobbli.io.EmbedInput, context: ContainerTaskContext
    ) -> gobbli.io.EmbedOutput:
        # Check for null checkpoint here to give quick feedback to the user
        if embed_input.checkpoint is None:
            raise ValueError(
                "fastText requires a trained checkpoint to generate embeddings."
            )
        if embed_input.pooling == gobbli.io.EmbedPooling.NONE:
            raise ValueError(
                "fastText prints sentence vectors, so pooling is required."
            )

        host_input_path = context.host_input_dir / FastText._EMBEDDING_INPUT_FILE
        self._write_input(embed_input.X, None, host_input_path)
        container_input_path = context.to_container(host_input_path)

        host_checkpoint, container_checkpoint = self._get_checkpoint(
            embed_input.checkpoint, context
        )

        # We shouldn't get Nones here if the user didn't pass a null checkpoint, but
        # check anyway to satisfy mypy
        if host_checkpoint is None or container_checkpoint is None:
            raise ValueError(
                "fastText requires a trained checkpoint to generate embeddings."
            )

        host_output_path = context.host_output_dir / FastText._EMBEDDING_OUTPUT_FILE
        container_output_path = (
            context.container_output_dir / FastText._EMBEDDING_OUTPUT_FILE
        )

        cmd = (
            "bash -c './fasttext print-sentence-vectors"
            f" {container_checkpoint.model}"
            f" <{container_input_path}"
            f" >{container_output_path}'"
        )

        run_kwargs = self._base_docker_run_kwargs(context)
        # Override the entrypint so we can use 'bash -c ...' above
        run_kwargs["entrypoint"] = ""
        maybe_mount(
            run_kwargs["volumes"], host_checkpoint.model, container_checkpoint.model
        )

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        # Parse the embeddings out of the output file
        embeddings = np.loadtxt(host_output_path, comments=None, ndmin=2)

        return gobbli.io.EmbedOutput(
            X_embedded=embeddings, embed_tokens=None, _console_output=container_logs
        )
