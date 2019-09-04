import re
import shutil
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
    pred_prob_to_pred_label,
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

    _TRAIN_INPUT_FILE = "train.txt"
    _VALID_INPUT_FILE = "valid.txt"
    _TEST_INPUT_FILE = "test.txt"

    _PREDICT_OUTPUT_FILE = "predict.txt"

    _EMBEDDING_INPUT_FILE = "input.txt"
    _EMBEDDING_OUTPUT_FILE = "embeddings.txt"

    _CHECKPOINT_BASE = "model"

    @property
    def image_tag(self) -> str:
        """
        Using a prebuilt image directly:
        https://github.com/xeb/fastText-docker

        Returns:
          The tag to use for the fastText image.
        """
        return "xebxeb/fasttext-docker:binary"

    def init(self, params: Dict[str, Any]):
        """
        See :meth: `gobbli.model.base.BaseModel.init`.

        For more info on fastText parameter semantics, see
        `the docs <https://fasttext.cc/docs/en/options.html>`__.  The fastText
        `supervised tutorial <https://fasttext.cc/docs/en/supervised-tutorial.html>`__ has
        some more detailed explanation.

        fastText parameters:

        - ``word_ngrams`` (:obj:`int`): Max length of word n-grams.
        - ``lr`` (:obj:`float`): Learning rate.
        - ``dim`` (:obj:`int`): Dimension of learned vectors.
        - ``ws`` (:obj:`int`): Context window size.
        - ``fasttext_model`` (:obj:`str`): Name of a pretrained fastText model to use.
          See :obj:`FASTTEXT_VECTOR_ARCHIVES` for a listing of available pretrained models.
        """
        self.word_ngrams = 1
        self.lr = 0.1
        self.ws = 5
        self.fasttext_model = None
        # Default to dimensionality of the passed model, if any;
        # otherwise, default to 100
        if "fasttext_model" in params:
            self.dim = _parse_dim(params["fasttext_model"])
        else:
            self.dim = 100

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
        if self.fasttext_model is not None and not self.weights_dir.exists():
            self.weights_dir.mkdir(parents=True)
            try:
                self.logger.info("Downloading pre-trained weights.")
                download_archive(
                    FASTTEXT_VECTOR_ARCHIVES[self.fasttext_model], self.weights_dir
                )
                self.logger.info("Weights downloaded.")
            except Exception:
                # Don't leave the weights directory in a partially downloaded state
                if self.weights_dir.exists():
                    shutil.rmtree(self.weights_dir)
                raise

        # Using a prebuilt Docker image, so no build step required

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
            raise ValueError(f"No .vec files found in '{weights_dir}'.")
        elif len(candidates) > 1:
            raise ValueError(
                f"Multiple .vec files found in '{weights_dir}': {candidates}"
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
        self, X: List[str], y: Optional[List[str]], input_path: Path
    ) -> List[str]:
        """
        Write the given input and labels (if any) into the format expected by fastText.
        Make sure the given directory exists first.

        Returns:
          A sorted list of unique labels found in the dataset.
        """
        label_set = set()
        with open(input_path, "w") as f:
            if y is not None:
                for text, label in zip(X, y):
                    f.write(f"__label__{label} {_fasttext_preprocess(text)}\n")
                    label_set.add(label)
            elif y is None:
                for text in X:
                    f.write(f"{_fasttext_preprocess(text)}\n")

        return list(sorted(label_set))

    def _run_supervised(
        self,
        user_checkpoint: Optional[Path],
        container_input_path: Path,
        container_output_path: Path,
        context: ContainerTaskContext,
        num_epochs: int,
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
          freeze_vectors: If true, use 0 learning rate; train solely for
            the purpose of calculating loss.

        Returns:
          A 2-tuple: container logs and loss.
        """
        host_checkpoint, container_checkpoint = self._get_checkpoint(
            user_checkpoint, context
        )

        lr = self.lr
        if freeze_vectors:
            lr = 0.0

        cmd = (
            "supervised"
            f" -input {container_input_path}"
            f" -output {container_output_path}"
            f" -wordNgrams {self.word_ngrams}"
            f" -lr {lr}"
            f" -dim {self.dim}"
            f" -epoch {num_epochs}"
            f" -ws {self.ws}"
        )

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
        last_loss_ndx = container_logs.rfind("loss:")
        failed_parse_msg = (
            "Failed to parse loss information from fastText container logs."
            " Run with debug logging to"
            " see why this might have happened."
        )
        if last_loss_ndx == -1:
            raise ValueError(failed_parse_msg)

        # Skip over the word "loss:" - next field in the output is "eta:"
        loss_start_ndx = last_loss_ndx + 5
        loss_end_ndx = container_logs.find("eta:", loss_start_ndx)
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
                row_data = {}
                for raw_label, prob in zip(tokens[0::2], tokens[1::2]):
                    # Strip the "__label__" prefix
                    label = raw_label[9:]
                    row_data[label] = float(prob)
                pred_prob_data.append(row_data)

        return (container_logs, pd.DataFrame(pred_prob_data))

    def _train(
        self, train_input: gobbli.io.TrainInput, context: ContainerTaskContext
    ) -> gobbli.io.TrainOutput:
        labels = self._write_input(
            train_input.X_train,
            train_input.y_train,
            context.host_input_dir / FastText._TRAIN_INPUT_FILE,
        )
        self._write_input(
            train_input.X_valid,
            train_input.y_valid,
            context.host_input_dir / FastText._VALID_INPUT_FILE,
        )

        train_logs, train_loss = self._run_supervised(
            train_input.checkpoint,
            context.container_input_dir / FastText._TRAIN_INPUT_FILE,
            context.container_output_dir / FastText._CHECKPOINT_BASE,
            context,
            train_input.num_train_epochs,
        )

        host_checkpoint_path = context.host_output_dir / f"{FastText._CHECKPOINT_BASE}"

        # Calculate validation accuracy on our own, since the CLI only provides
        # precision/recall
        container_input_path = context.container_input_dir / FastText._VALID_INPUT_FILE
        predict_logs, pred_prob_df = self._run_predict_prob(
            host_checkpoint_path, labels, container_input_path, context
        )
        pred_labels = pred_prob_to_pred_label(pred_prob_df)

        # Not ideal, but fastText doesn't provide a way to get validation loss;
        # Negate the validation accuracy instead
        valid_accuracy = accuracy_score(train_input.y_valid, pred_labels)
        valid_loss = -valid_accuracy

        return gobbli.io.TrainOutput(
            train_loss=train_loss,
            valid_loss=valid_loss,
            valid_accuracy=valid_accuracy,
            labels=labels,
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
