import json
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import gobbli.io
from gobbli.docker import maybe_mount, run_container
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.model.mixin import EmbedMixin, PredictMixin, TrainMixin
from gobbli.util import assert_type, escape_line_delimited_texts


class SpaCyModel(BaseModel, TrainMixin, PredictMixin, EmbedMixin):
    """
    gobbli interface for spaCy language models which allows for training
    and prediction via the
    `TextCategorizer pipeline component <https://spacy.io/api/textcategorizer>`__
    and static embeddings via `Vectors <https://spacy.io/api/vectors>`__.
    """

    _BUILD_PATH = Path(__file__).parent

    _TRAIN_INPUT_FILE = "train.tsv"
    _VALID_INPUT_FILE = "dev.tsv"
    _TEST_INPUT_FILE = "test.tsv"
    _LABELS_INPUT_FILE = "labels.tsv"

    _TRAIN_OUTPUT_CHECKPOINT = "checkpoint"
    _VALID_OUTPUT_FILE = "valid_results.json"
    _TEST_OUTPUT_FILE = "test_results.tsv"

    _EMBEDDING_INPUT_FILE = "input.tsv"
    _EMBEDDING_OUTPUT_FILE = "embeddings.jsonl"

    _CONTAINER_CACHE_DIR = Path("/cache")

    def init(self, params: Dict[str, Any]):
        """
        See :meth:`gobbli.model.base.BaseModel.init`.

        spaCy parameters:

        - ``model`` (:obj:`str`): Name of a spaCy model to use.
          Available values are in `the spaCy model docs <https://spacy.io/models>`__ and
          `the spacy-transformers docs <https://github.com/explosion/spacy-transformers>`__.
        - ``architecture`` (:obj:`str`): Model architecture to use.
          Available values are in `the spaCy API docs <https://spacy.io/api/textcategorizer#architectures>`__.
          This is ignored if using a spacy-transformers model.
        - ``dropout`` (:obj:`float`): Dropout proportion for training.
        - ``full_pipeline`` (:obj:`bool`): If True, enable the full spaCy language pipeline
          (including tagging, parsing, and named entity recognition) for the TextCategorizer
          model used in training and prediction.  This makes training/prediction much slower
          but theoretically provides more information to the model.  This is ignored if using a
          spacy-transformers model.

        Note that gobbli relies on spaCy to perform validation on these parameters,
        so initialization errors may not be caught until model runtime.
        """
        self.model = "en_core_web_lg"
        self.architecture = "ensemble"
        self.dropout = 0.2
        self.full_pipeline = False

        for name, value in params.items():
            if name == "model":
                self.model = value
            elif name == "architecture":
                self.architecture = value
            elif name == "dropout":
                assert_type(name, value, float)
                self.dropout = value
            elif name == "full_pipeline":
                assert_type(name, value, bool)
                self.full_pipeline = value
            else:
                raise ValueError(f"Unknown param '{name}'")

    @property
    def image_tag(self) -> str:
        """
        Returns:
          The Docker image tag to be used for the spaCy container.
        """
        return "gobbli-spacy"

    def _build(self):
        # Add the spaCy model to the image build so it's properly installed
        base_build_kwargs = deepcopy(self._base_docker_build_kwargs)
        if "buildargs" not in base_build_kwargs:
            base_build_kwargs["buildargs"] = {}
        base_build_kwargs["buildargs"]["model"] = self.model

        self.docker_client.images.build(
            path=str(SpaCyModel._BUILD_PATH), tag=self.image_tag, **base_build_kwargs
        )

    @staticmethod
    def _get_checkpoint(
        user_checkpoint: Optional[Path], context: ContainerTaskContext
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Determines the host checkpoint directory and container checkpoint directory
        using the user-requested checkpoint (if any) and the container context.

        Args:
          user_checkpoint: An optional checkpoint passed in by the user.  If the user doesn't
            pass one, use the default pretrained checkpoint.
          context: The container context to create the checkpoint in.

        Returns:
          A 2-tuple: the host checkpoint directory (if any) and
            the container checkpoint directory (if any)
        """
        if user_checkpoint is None:
            host_checkpoint_dir = None
            container_checkpoint_dir = None
        else:
            host_checkpoint_dir = user_checkpoint
            container_checkpoint_dir = context.container_root_dir / "checkpoint"

        return host_checkpoint_dir, container_checkpoint_dir

    def _get_model(self, container_checkpoint_dir: Optional[Path]) -> Union[str, Path]:
        """
        Determine the model to pass to the run_spacy script.  If we don't have a
        checkpoint, we'll use our stock model.  Otherwise, we should use the
        checkpoint.
        """
        if container_checkpoint_dir is None:
            return self.model
        else:
            return container_checkpoint_dir

    @property
    def host_cache_dir(self):
        """
        Directory to be used for downloaded spaCy files.
        Should be the same across all instances of the class, since these are
        generally static model weights that can be reused.
        """
        cache_dir = SpaCyModel.model_class_dir() / "cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir

    def _write_input(
        self, X: List[str], labels: Optional[List[List[str]]], input_path: Path
    ):
        """
        Write the given input texts and (optionally) labels to the file pointed to by
        ``input_path``.
        """
        df = pd.DataFrame({"Text": X})

        if labels is not None:
            df["Label"] = labels

        df.to_csv(input_path, sep="\t", index=False)

    def _write_labels(self, labels: List[str], labels_path: Path):
        """
        Write the given labels to the file pointed at by ``labels_path``.
        """
        labels_path.write_text(escape_line_delimited_texts(labels))

    def _train(
        self, train_input: gobbli.io.TrainInput, context: ContainerTaskContext
    ) -> gobbli.io.TrainOutput:

        if train_input.valid_batch_size != gobbli.io.TrainInput.valid_batch_size:
            warnings.warn(
                "The spaCy model doesn't batch validation data, so the validation "
                "batch size parameter will be ignored."
            )

        self._write_input(
            train_input.X_train,
            train_input.y_train_multilabel,
            context.host_input_dir / SpaCyModel._TRAIN_INPUT_FILE,
        )
        self._write_input(
            train_input.X_valid,
            train_input.y_valid_multilabel,
            context.host_input_dir / SpaCyModel._VALID_INPUT_FILE,
        )

        labels = train_input.labels()
        self._write_labels(
            labels, context.host_input_dir / SpaCyModel._LABELS_INPUT_FILE
        )

        # Determine checkpoint to use
        host_checkpoint_dir, container_checkpoint_dir = self._get_checkpoint(
            train_input.checkpoint, context
        )

        cmd = (
            "python3 run_spacy.py"
            " train"
            f" --input-dir {context.container_input_dir}"
            f" --output-dir {context.container_output_dir}"
            f" --model {self._get_model(container_checkpoint_dir)}"
            f" --architecture {self.architecture}"
            f" --cache-dir {SpaCyModel._CONTAINER_CACHE_DIR}"
            f" --train-batch-size {train_input.train_batch_size}"
            f" --num-train-epochs {train_input.num_train_epochs}"
            f" --dropout {self.dropout}"
        )

        if self.full_pipeline:
            cmd += " --full-pipeline"
        if train_input.multilabel:
            cmd += " --multilabel"

        run_kwargs = self._base_docker_run_kwargs(context)

        # Mount the checkpoint in the container if needed
        maybe_mount(
            run_kwargs["volumes"], host_checkpoint_dir, container_checkpoint_dir
        )

        # Mount the cache directory
        maybe_mount(
            run_kwargs["volumes"], self.host_cache_dir, SpaCyModel._CONTAINER_CACHE_DIR
        )

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        # Read in the generated evaluation results
        with open(context.host_output_dir / SpaCyModel._VALID_OUTPUT_FILE, "r") as f:
            results = json.load(f)

        return gobbli.io.TrainOutput(
            valid_loss=results["mean_valid_loss"],
            valid_accuracy=results["valid_accuracy"],
            train_loss=results["mean_train_loss"],
            labels=labels,
            multilabel=train_input.multilabel,
            checkpoint=context.host_output_dir / SpaCyModel._TRAIN_OUTPUT_CHECKPOINT,
            _console_output=container_logs,
        )

    def _read_predictions(self, predict_path: Path):
        return pd.read_csv(predict_path, sep="\t")

    def _predict(
        self, predict_input: gobbli.io.PredictInput, context: ContainerTaskContext
    ) -> gobbli.io.PredictOutput:

        if (
            predict_input.predict_batch_size
            != gobbli.io.PredictInput.predict_batch_size
        ):
            warnings.warn(
                "The spaCy model doesn't batch prediction data, so the prediction "
                "batch size parameter will be ignored."
            )

        self._write_input(
            predict_input.X, None, context.host_input_dir / SpaCyModel._TEST_INPUT_FILE
        )

        labels = predict_input.labels
        self._write_labels(
            labels, context.host_input_dir / SpaCyModel._LABELS_INPUT_FILE
        )

        host_checkpoint_dir, container_checkpoint_dir = self._get_checkpoint(
            predict_input.checkpoint, context
        )

        cmd = (
            "python3 run_spacy.py"
            " predict"
            f" --input-dir {context.container_input_dir}"
            f" --output-dir {context.container_output_dir}"
            f" --model {self._get_model(container_checkpoint_dir)}"
            f" --architecture {self.architecture}"
            f" --cache-dir {SpaCyModel._CONTAINER_CACHE_DIR}"
        )

        run_kwargs = self._base_docker_run_kwargs(context)

        # Mount the checkpoint in the container if needed
        maybe_mount(
            run_kwargs["volumes"], host_checkpoint_dir, container_checkpoint_dir
        )

        # Mount the cache directory
        maybe_mount(
            run_kwargs["volumes"], self.host_cache_dir, SpaCyModel._CONTAINER_CACHE_DIR
        )

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        return gobbli.io.PredictOutput(
            y_pred_proba=self._read_predictions(
                context.host_output_dir / SpaCyModel._TEST_OUTPUT_FILE
            ),
            _console_output=container_logs,
        )

    def _read_embeddings(
        self, embed_path: Path, pooling: gobbli.io.EmbedPooling
    ) -> Tuple[List[np.ndarray], Optional[List[List[str]]]]:

        embeddings = []  # type: List[np.ndarray]
        doc_tokens = []  # type: List[List[str]]
        with open(embed_path, "r") as f:
            for line in f:
                line_json = json.loads(line)
                embeddings.append(np.array(line_json["embedding"]))
                if pooling == gobbli.io.EmbedPooling.NONE:
                    doc_tokens.append(line_json["tokens"])

        tokens = None
        if pooling == gobbli.io.EmbedPooling.NONE:
            tokens = doc_tokens

        return embeddings, tokens

    def _embed(
        self, embed_input: gobbli.io.EmbedInput, context: ContainerTaskContext
    ) -> gobbli.io.EmbedOutput:
        self._write_input(
            embed_input.X,
            None,
            context.host_input_dir / SpaCyModel._EMBEDDING_INPUT_FILE,
        )

        if embed_input.checkpoint is not None:
            warnings.warn(
                "The spaCy model vectors can't be fine-tuned, so custom "
                "checkpoints are ignored when generating embeddings."
            )

        cmd = (
            "python3 run_spacy.py"
            " embed"
            f" --input-dir {context.container_input_dir}"
            f" --output-dir {context.container_output_dir}"
            f" --model {self.model}"
            f" --architecture {self.architecture}"
            f" --cache-dir {SpaCyModel._CONTAINER_CACHE_DIR}"
            f" --embed-pooling {embed_input.pooling.value}"
        )

        run_kwargs = self._base_docker_run_kwargs(context)

        # Mount the cache directory
        maybe_mount(
            run_kwargs["volumes"], self.host_cache_dir, SpaCyModel._CONTAINER_CACHE_DIR
        )

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        X_embedded, embed_tokens = self._read_embeddings(
            context.host_output_dir / SpaCyModel._EMBEDDING_OUTPUT_FILE,
            embed_input.pooling,
        )

        return gobbli.io.EmbedOutput(
            X_embedded=X_embedded,
            embed_tokens=embed_tokens,
            _console_output=container_logs,
        )
