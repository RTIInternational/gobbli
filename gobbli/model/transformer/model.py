import json
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


class Transformer(BaseModel, TrainMixin, PredictMixin, EmbedMixin):
    """
    Classifier/embedding wrapper for any of the Transformers from
    `transformers <https://github.com/huggingface/transformers>`__.
    """

    _BUILD_PATH = Path(__file__).parent

    _TRAIN_INPUT_FILE = "train.tsv"
    _VALID_INPUT_FILE = "dev.tsv"
    _TEST_INPUT_FILE = "test.tsv"
    _LABELS_INPUT_FILE = "labels.tsv"
    _CONFIG_OVERRIDE_FILE = "config.json"

    _TRAIN_OUTPUT_CHECKPOINT = "checkpoint"
    _VALID_OUTPUT_FILE = "valid_results.json"
    _TEST_OUTPUT_FILE = "test_results.tsv"

    _EMBEDDING_INPUT_FILE = "input.tsv"
    _EMBEDDING_OUTPUT_FILE = "embeddings.jsonl"

    _CONTAINER_CACHE_DIR = Path("/cache")

    def init(self, params: Dict[str, Any]):
        """
        See :meth:`gobbli.model.base.BaseModel.init`.

        Transformer parameters:

        - ``transformer_model`` (:obj:`str`): Name of a transformer model architecture to use.
          For training/prediction, the value should be one such that
          ``from transformers import <value>ForSequenceClassification`` is
          a valid import.  ex value = "Bert" ->
          ``from transformers import BertForSequenceClassification``.  Note this means
          only a subset of the transformers models are supported for these tasks -- search
          `the docs <https://huggingface.co/transformers/search.html?q=forsequenceclassification&check_keywords=yes&area=default>`__ to see which ones you can use.
          For embedding generation, the import is ``<value>Model``, so any transformer
          model is supported.
        - ``transformer_weights`` (:obj:`str`): Name of the pretrained weights to use.
          See the `transformers docs <https://huggingface.co/transformers/pretrained_models.html>`__
          for supported values.  These depend on the ``transformer_model`` chosen.
        - ``config_overrides`` (:obj:`dict`): Dictionary of keys and values that will
          override config for the model.
        - ``max_seq_length``: Truncate all sequences to this length after tokenization.
          Used to save memory.
        - ``lr``: Learning rate for the AdamW optimizer.
        - ``adam_eps``: Epsilon value for the AdamW optimizer.
        - ``gradient_accumulation_steps``: Number of iterations to accumulate gradients before
          updating the model.  Used to allow larger effective batch sizes for models too big to
          fit a large batch on the GPU.  The "effective batch size" is
          ``gradient_accumulation_steps`` * :paramref:`TrainInput.params.train_batch_size`.
          If you encounter memory errors while training, try decreasing the batch size and
          increasing ``gradient_accumulation_steps``. For example, if a training batch size of
          32 causes memory errors, try decreasing batch size to 16 and increasing
          ``gradient_accumulation_steps`` to 2.  If you still have problems with memory, you can
          drop batch size to 8 and ``gradient_accumulation_steps`` to 4, and so on.

        Note that gobbli relies on transformers to perform validation on these parameters,
        so initialization errors may not be caught until model runtime.
        """
        self.transformer_model = "Bert"
        self.transformer_weights = "bert-base-uncased"
        self.config_overrides = {}  # type: Dict[str, Any]
        self.max_seq_length = 128
        self.lr = 5e-5
        self.adam_eps = 1e-8
        self.gradient_accumulation_steps = 1

        for name, value in params.items():
            if name == "transformer_model":
                self.transformer_model = value
            elif name == "transformer_weights":
                self.transformer_weights = value
            elif name == "config_overrides":
                assert_type(name, value, dict)
                self.config_overrides = value
            elif name == "max_seq_length":
                assert_type(name, value, int)
                self.max_seq_length = value
            elif name == "lr":
                assert_type(name, value, float)
                self.lr = value
            elif name == "adam_eps":
                assert_type(name, value, float)
                self.adam_eps = value
            elif name == "gradient_accumulation_steps":
                assert_type(name, value, int)
                self.gradient_accumulation_steps = value
            else:
                raise ValueError(f"Unknown param '{name}'")

    @property
    def image_tag(self) -> str:
        """
        Returns:
          The Docker image tag to be used for the transformer container.
        """
        return "gobbli-transformer"

    def _build(self):
        self.docker_client.images.build(
            path=str(Transformer._BUILD_PATH),
            tag=self.image_tag,
            **self._base_docker_build_kwargs,
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

    def _get_weights(
        self, container_checkpoint_dir: Optional[Path]
    ) -> Union[str, Path]:
        """
        Determine the weights to pass to the run_model script.  If we don't have a
        checkpoint, we'll use the pretrained weights.  Otherwise, we should use the
        checkpoint weights.
        """
        if container_checkpoint_dir is None:
            return self.transformer_weights
        else:
            return container_checkpoint_dir

    @property
    def host_cache_dir(self):
        """
        Directory to be used for downloaded transformers files.
        Should be the same across all instances of the class, since these are
        generally static model weights/config files that can be reused.
        """
        cache_dir = Transformer.model_class_dir() / "cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir

    def _write_input(
        self,
        X: List[str],
        labels: Optional[Union[List[str], List[List[str]]]],
        input_path: Path,
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

    def _write_config(self, config_path: Path):
        """
        Write our model configuration overrides to the given path.
        """
        with open(config_path, "w") as f:
            json.dump(self.config_overrides, f)

    def _train(
        self, train_input: gobbli.io.TrainInput, context: ContainerTaskContext
    ) -> gobbli.io.TrainOutput:

        self._write_input(
            train_input.X_train,
            train_input.y_train,
            context.host_input_dir / Transformer._TRAIN_INPUT_FILE,
        )
        self._write_input(
            train_input.X_valid,
            train_input.y_valid,
            context.host_input_dir / Transformer._VALID_INPUT_FILE,
        )
        self._write_config(context.host_input_dir / Transformer._CONFIG_OVERRIDE_FILE)

        labels = train_input.labels()
        self._write_labels(
            labels, context.host_input_dir / Transformer._LABELS_INPUT_FILE
        )

        # Determine checkpoint to use
        host_checkpoint_dir, container_checkpoint_dir = self._get_checkpoint(
            train_input.checkpoint, context
        )

        cmd = (
            "python3 run_model.py"
            " train"
            f" --input-dir {context.container_input_dir}"
            f" --output-dir {context.container_output_dir}"
            f" --config-overrides {context.container_input_dir / Transformer._CONFIG_OVERRIDE_FILE}"
            f" --model {self.transformer_model}"
            f" --weights {self._get_weights(container_checkpoint_dir)}"
            f" --cache-dir {Transformer._CONTAINER_CACHE_DIR}"
            f" --max-seq-length {self.max_seq_length}"
            f" --train-batch-size {train_input.train_batch_size}"
            f" --valid-batch-size {train_input.valid_batch_size}"
            f" --num-train-epochs {train_input.num_train_epochs}"
            f" --lr {self.lr}"
            f" --adam-eps {self.adam_eps}"
            f" --gradient-accumulation-steps {self.gradient_accumulation_steps}"
        )

        if train_input.multilabel:
            cmd += " --multilabel"

        run_kwargs = self._base_docker_run_kwargs(context)

        # Mount the checkpoint in the container if needed
        maybe_mount(
            run_kwargs["volumes"], host_checkpoint_dir, container_checkpoint_dir
        )

        # Mount the cache directory
        maybe_mount(
            run_kwargs["volumes"], self.host_cache_dir, Transformer._CONTAINER_CACHE_DIR
        )

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        # Read in the generated evaluation results
        with open(context.host_output_dir / Transformer._VALID_OUTPUT_FILE, "r") as f:
            results = json.load(f)

        return gobbli.io.TrainOutput(
            valid_loss=results["mean_valid_loss"],
            valid_accuracy=results["valid_accuracy"],
            train_loss=results["mean_train_loss"],
            multilabel=train_input.multilabel,
            labels=labels,
            checkpoint=context.host_output_dir / Transformer._TRAIN_OUTPUT_CHECKPOINT,
            _console_output=container_logs,
        )

    def _read_predictions(self, predict_path: Path):
        return pd.read_csv(predict_path, sep="\t")

    def _predict(
        self, predict_input: gobbli.io.PredictInput, context: ContainerTaskContext
    ) -> gobbli.io.PredictOutput:

        self._write_input(
            predict_input.X, None, context.host_input_dir / Transformer._TEST_INPUT_FILE
        )
        self._write_config(context.host_input_dir / Transformer._CONFIG_OVERRIDE_FILE)

        labels = predict_input.labels
        self._write_labels(
            labels, context.host_input_dir / Transformer._LABELS_INPUT_FILE
        )

        host_checkpoint_dir, container_checkpoint_dir = self._get_checkpoint(
            predict_input.checkpoint, context
        )

        cmd = (
            "python3 run_model.py"
            " predict"
            f" --input-dir {context.container_input_dir}"
            f" --output-dir {context.container_output_dir}"
            f" --config-overrides {context.container_input_dir / Transformer._CONFIG_OVERRIDE_FILE}"
            f" --model {self.transformer_model}"
            f" --weights {self._get_weights(container_checkpoint_dir)}"
            f" --cache-dir {Transformer._CONTAINER_CACHE_DIR}"
            f" --max-seq-length {self.max_seq_length}"
            f" --predict-batch-size {predict_input.predict_batch_size}"
        )

        if predict_input.multilabel:
            cmd += " --multilabel"

        run_kwargs = self._base_docker_run_kwargs(context)

        # Mount the checkpoint in the container if needed
        maybe_mount(
            run_kwargs["volumes"], host_checkpoint_dir, container_checkpoint_dir
        )

        # Mount the cache directory
        maybe_mount(
            run_kwargs["volumes"], self.host_cache_dir, Transformer._CONTAINER_CACHE_DIR
        )

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        return gobbli.io.PredictOutput(
            y_pred_proba=self._read_predictions(
                context.host_output_dir / Transformer._TEST_OUTPUT_FILE
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
            context.host_input_dir / Transformer._EMBEDDING_INPUT_FILE,
        )
        self._write_config(context.host_input_dir / Transformer._CONFIG_OVERRIDE_FILE)

        host_checkpoint_dir, container_checkpoint_dir = self._get_checkpoint(
            embed_input.checkpoint, context
        )

        cmd = (
            "python3 run_model.py"
            " embed"
            f" --input-dir {context.container_input_dir}"
            f" --output-dir {context.container_output_dir}"
            f" --config-overrides {context.container_input_dir / Transformer._CONFIG_OVERRIDE_FILE}"
            f" --model {self.transformer_model}"
            f" --weights {self._get_weights(container_checkpoint_dir)}"
            f" --cache-dir {Transformer._CONTAINER_CACHE_DIR}"
            f" --max-seq-length {self.max_seq_length}"
            f" --embed-batch-size {embed_input.embed_batch_size}"
            f" --embed-pooling {embed_input.pooling.value}"
            f" --embed-layer -2"
        )

        run_kwargs = self._base_docker_run_kwargs(context)

        # Mount the checkpoint in the container if needed
        maybe_mount(
            run_kwargs["volumes"], host_checkpoint_dir, container_checkpoint_dir
        )

        # Mount the cache directory
        maybe_mount(
            run_kwargs["volumes"], self.host_cache_dir, Transformer._CONTAINER_CACHE_DIR
        )

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        X_embedded, embed_tokens = self._read_embeddings(
            context.host_output_dir / Transformer._EMBEDDING_OUTPUT_FILE,
            embed_input.pooling,
        )

        return gobbli.io.EmbedOutput(
            X_embedded=X_embedded,
            embed_tokens=embed_tokens,
            _console_output=container_logs,
        )
