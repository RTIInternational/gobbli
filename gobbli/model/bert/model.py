import json
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import gobbli.io
from gobbli.docker import maybe_mount, run_container
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.model.mixin import EmbedMixin, PredictMixin, TrainMixin
from gobbli.util import (
    assert_in,
    assert_type,
    download_archive,
    escape_line_delimited_texts,
)


def _preprocess_text(text_series: pd.Series) -> pd.Series:
    """
    Preprocess a Series of text for a BERT model.
    """
    return text_series.str.replace(r"\s", " ", regex=True)


def _df_to_train_tsv(df: pd.DataFrame, output_file: Path):
    """
    Write a dataframe with "Text" and "Label" columns to the given
    output file in the format expected by BERT for a TSV file for
    training/validation.
    """
    ordered_df = pd.DataFrame(
        OrderedDict(
            (
                ("Label", df["Label"]),
                ("a", np.repeat("a", df.shape[0])),
                ("Text", _preprocess_text(df["Text"])),
            )
        )
    )
    ordered_df.to_csv(output_file, sep="\t", index=True, header=False)


def _write_labels(labels: List[Any], output_file: Path):
    """
    Write the given set of labels to the given file.
    """
    output_file.write_text(escape_line_delimited_texts(labels))


def _read_predictions(labels: List[Any], output_file: Path) -> pd.DataFrame:
    """
    Read predictions from the BERT model into a dataframe containing the predicted
    probability for each label for each observation.
    """
    return pd.read_csv(output_file, sep="\t", names=labels)


def _read_embeddings(
    output_file: Path, pooling: gobbli.io.EmbedPooling
) -> Tuple[List[np.ndarray], Optional[List[List[str]]]]:
    """
    Read embeddings from the BERT model into a ndarray with the embedding values.  Also return
    a corresponding list of tokens if no pooling was applied.
    """
    embeddings = []  # type: List[np.ndarray]
    doc_tokens = []  # type: List[List[str]]
    with open(output_file, "r") as f:
        for line in f:
            line_json = json.loads(line)
            line_tokens = []  # type: List[str]
            line_layers = []  # type: List[List[float]]
            for token_info in line_json["features"]:
                line_tokens.append(token_info["token"])
                # Only take the first layer output; we don't currently support a
                # way to combine multiple layers
                line_layers.append(token_info["layers"][0]["values"])

            # Apply pooling, if necessary
            line_embedding = np.array(line_layers)
            if pooling == gobbli.io.EmbedPooling.MEAN:
                line_embedding = np.mean(line_embedding, axis=0)

            embeddings.append(line_embedding)
            doc_tokens.append(line_tokens)

    # Don't return tokens if we're doing any pooling, since
    # the pooled results combine all the tokens
    tokens = None
    if pooling == gobbli.io.EmbedPooling.NONE:
        tokens = doc_tokens

    return embeddings, tokens


BERT_MODEL_ARCHIVES = {
    "bert-base-uncased": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
    "bert-base-cased": "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip",
    "bert-large-uncased": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip",
    "bert-large-cased": "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip",
    "bert-large-whole-word-masking-uncased": "https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip",
    "bert-large-whole-word-masking-cased": "https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip",
    "bert-base-multilingual-cased": "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip",
    "bert-base-chinese": "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip",
    "clinical-bert-cased": "https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1",
    "biobert-cased": "https://github.com/naver/biobert-pretrained/releases/download/v1.1-pubmed/biobert_v1.1_pubmed.tar.gz",
    "scibert-uncased": "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz",
    "scibert-cased": "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_cased.tar.gz",
    "ncbi-bert-base-pubmed-uncased": "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12.zip",
    "ncbi-bert-base-pubmed-mimic-uncased": "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12.zip",
    "ncbi-bert-large-pubmed-uncased": "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_uncased_L-24_H-1024_A-16.zip",
    "ncbi-bert-large-pubmed-mimic-uncased": "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_mimic_uncased_L-24_H-1024_A-16.zip",
}  # type: Dict[str, str]
"""
A mapping from model names to archives.
See `the BERT repo <https://github.com/google-research/bert>`__ for guidelines on when
to use which model.  "bert-base-uncased" is a safe default for most situations.
Larger models require more time and GPU memory to run.
"""


class BERT(BaseModel, TrainMixin, PredictMixin, EmbedMixin):
    """
    Classifier/embedding wrapper for Google Research's BERT:
    https://github.com/google-research/bert
    """

    _BUILD_PATH = Path(__file__).parent

    _TRAIN_INPUT_FILE = "train.tsv"
    _VALID_INPUT_FILE = "dev.tsv"
    _TEST_INPUT_FILE = "test.tsv"
    _LABELS_INPUT_FILE = "labels.tsv"

    _TEST_OUTPUT_FILE = "test_results.tsv"

    _EMBEDDING_INPUT_FILE = "input.txt"
    _EMBEDDING_OUTPUT_FILE = "embeddings.jsonl"

    def init(self, params: Dict[str, Any]):
        """
        See :meth:`gobbli.model.base.BaseModel.init`.

        BERT parameters:

        - ``max_seq_length`` (:obj:`int`): The maximum total input sequence length after
          WordPiece tokenization.  Sequences longer than this will be truncated,
          and sequences shorter than this will be padded.  Default: 128
        - ``bert_model`` (:obj:`str`): Name of a pretrained BERT model to use.
          See :obj:`BERT_MODEL_ARCHIVES` for a listing of available BERT models.
        """
        self.max_seq_length = 128
        self.bert_model = "bert-base-uncased"

        for name, value in params.items():
            if name == "max_seq_length":
                assert_type(name, value, int)
                self.max_seq_length = value
            elif name == "bert_model":
                assert_in(name, value, set(BERT_MODEL_ARCHIVES.keys()))
                self.bert_model = value
            else:
                raise ValueError(f"Unknown param '{name}'")

    @property
    def weights_dir(self) -> Path:
        """
        Returns:
          Directory containing pretrained weights for this instance.
        """
        return self.class_weights_dir / self.bert_model

    @property
    def image_tag(self) -> str:
        """
        Returns:
          The Docker image tag to be used for the BERT container.
        """
        device = "gpu" if self.use_gpu else "cpu"
        return f"gobbli-bert-classifier-{device}"

    @property
    def do_lower_case(self) -> bool:
        """
        Returns:
          Whether the BERT tokenizer should lowercase its input.
        """
        return "uncased" in self.bert_model

    def _build(self):
        # Download data if we don't already have it
        # Download into a temp dir and move the result into the destination dir
        # to ensure partial downloads don't leave corrupted state
        if not self.weights_dir.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_weights_dir = Path(tmpdir) / self.weights_dir.name
                tmp_weights_dir.mkdir()
                self.logger.info("Downloading pre-trained weights.")
                download_archive(
                    BERT_MODEL_ARCHIVES[self.bert_model],
                    tmp_weights_dir,
                    junk_paths=True,
                )
                shutil.move(tmp_weights_dir, self.weights_dir)
                self.logger.info("Weights downloaded.")

        # Build the docker image
        self.docker_client.images.build(
            path=str(BERT._BUILD_PATH),
            tag=self.image_tag,
            **self._base_docker_build_kwargs,
        )

    @staticmethod
    def _get_checkpoint(
        user_checkpoint: Optional[Path], context: ContainerTaskContext
    ) -> Tuple[Optional[Path], Path, str]:
        """
        Determines the host checkpoint directory, container checkpoint directory, and
        checkpoint name using the user-requested checkpoint (if any) and the container context.

        Args:
          user_checkpoint: An optional checkpoint passed in by the user.  If the user doesn't
            pass one, use the default pretrained checkpoint.
          context: The container context to create the checkpoint in.

        Returns:
          A 3-tuple: the host checkpoint directory (if any),
            the container checkpoint directory, and the checkpoint name.
        """
        if user_checkpoint is None:
            # Default BERT weights
            host_checkpoint_dir = None
            checkpoint_name = "bert_model.ckpt"
            container_checkpoint_dir = BaseModel._CONTAINER_WEIGHTS_PATH
        else:
            # Trained weights, which will be mounted in the container
            host_checkpoint_dir = user_checkpoint.parent
            checkpoint_name = user_checkpoint.name
            container_checkpoint_dir = context.container_root_dir / "checkpoint"

        return host_checkpoint_dir, container_checkpoint_dir, checkpoint_name

    def _write_train_input(self, train_input: gobbli.io.TrainInput, input_dir: Path):
        """
        Write the given gobbli input into the format expected by BERT.
        Make sure the given directory exists first.
        """
        train_path = input_dir / BERT._TRAIN_INPUT_FILE
        valid_path = input_dir / BERT._VALID_INPUT_FILE

        train_df = pd.DataFrame(
            {"Text": train_input.X_train, "Label": train_input.y_train_multiclass}
        )
        valid_df = pd.DataFrame(
            {"Text": train_input.X_valid, "Label": train_input.y_valid_multiclass}
        )

        _df_to_train_tsv(train_df, train_path)
        _df_to_train_tsv(valid_df, valid_path)

        labels_path = input_dir / BERT._LABELS_INPUT_FILE
        _write_labels(train_input.labels(), labels_path)

    def _train(
        self, train_input: gobbli.io.TrainInput, context: ContainerTaskContext
    ) -> gobbli.io.TrainOutput:

        if train_input.multilabel:
            raise ValueError(
                "gobbli BERT model doesn't support multilabel classification."
            )

        self._write_train_input(train_input, context.host_input_dir)

        # Determine checkpoint to use
        host_checkpoint_dir, container_checkpoint_dir, checkpoint_name = self._get_checkpoint(
            train_input.checkpoint, context
        )

        labels = train_input.labels()

        cmd = (
            "bash -c 'python run_classifier.py"
            " --task_name=cola"
            " --do_train=true"
            " --do_eval=true"
            f" --data_dir={context.container_input_dir}"
            f" --vocab_file={BaseModel._CONTAINER_WEIGHTS_PATH}/vocab.txt"
            f" --bert_config_file={BaseModel._CONTAINER_WEIGHTS_PATH}/bert_config.json"
            f" --init_checkpoint={BaseModel._CONTAINER_WEIGHTS_PATH}/bert_model.ckpt"
            f" --max_seq_length={self.max_seq_length}"
            f" --train_batch_size={train_input.train_batch_size}"
            f" --eval_batch_size={train_input.valid_batch_size}"
            f" --learning_rate=2e-5"
            f" --do_lower_case={self.do_lower_case}"
            f" --num_train_epochs={train_input.num_train_epochs}"
            f" --output_dir={context.container_output_dir}'"
        )

        run_kwargs = self._base_docker_run_kwargs(context)

        # Mount the checkpoint in the container if needed
        maybe_mount(
            run_kwargs["volumes"], host_checkpoint_dir, container_checkpoint_dir
        )

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        # Parse the generated evaluation results file
        results_file = context.host_output_dir / "eval_results.txt"
        eval_results = {}  # type: Dict[str, Union[int, float]]
        with open(results_file, "r") as f:
            for line in f:
                key, str_val = line.split(" = ")
                if key == "global_step":
                    val: Union[int, float] = int(str_val)
                else:
                    val = float(str_val)
                eval_results[key] = val

        return gobbli.io.TrainOutput(
            valid_loss=eval_results["eval_loss"],
            valid_accuracy=eval_results["eval_accuracy"],
            train_loss=eval_results["loss"],
            labels=labels,
            multilabel=False,
            checkpoint=context.host_output_dir
            / f"model.ckpt-{eval_results['global_step']}",
            _console_output=container_logs,
        )

    def _write_predict_input(
        self, predict_input: gobbli.io.PredictInput, input_dir: Path
    ):
        """
        Write the given gobbli prediction input into the format expected by BERT.
        Make sure the given directory exists first.
        """
        test_path = input_dir / BERT._TEST_INPUT_FILE

        test_df = pd.DataFrame({"sentence": predict_input.X})
        test_df["sentence"] = _preprocess_text(test_df["sentence"])
        test_df.index.name = "id"

        test_df.to_csv(test_path, sep="\t", index=True, header=True)

        labels_path = input_dir / BERT._LABELS_INPUT_FILE
        _write_labels(predict_input.labels, labels_path)

    def _predict(
        self, predict_input: gobbli.io.PredictInput, context: ContainerTaskContext
    ) -> gobbli.io.PredictOutput:

        self._write_predict_input(predict_input, context.host_input_dir)

        # Determine checkpoint to use
        host_checkpoint_dir, container_checkpoint_dir, checkpoint_name = self._get_checkpoint(
            predict_input.checkpoint, context
        )

        cmd = (
            "bash -c 'python run_classifier.py"
            " --task_name=cola"
            " --do_predict=true"
            f" --data_dir={context.container_input_dir}"
            f" --vocab_file={BaseModel._CONTAINER_WEIGHTS_PATH}/vocab.txt"
            f" --bert_config_file={BaseModel._CONTAINER_WEIGHTS_PATH}/bert_config.json"
            f" --predict-batch-size={predict_input.predict_batch_size}"
            f" --do_lower_case={self.do_lower_case}"
            f" --init_checkpoint={container_checkpoint_dir / checkpoint_name}"
            f" --max_seq_length={self.max_seq_length}"
            f" --output_dir={context.container_output_dir}'"
        )

        run_kwargs = self._base_docker_run_kwargs(context)

        # Mount the checkpoint in the container if needed
        maybe_mount(
            run_kwargs["volumes"], host_checkpoint_dir, container_checkpoint_dir
        )

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        return gobbli.io.PredictOutput(
            y_pred_proba=_read_predictions(
                predict_input.labels, context.host_output_dir / BERT._TEST_OUTPUT_FILE
            ),
            _console_output=container_logs,
        )

    def _write_embed_input(self, embed_input: gobbli.io.EmbedInput, input_dir: Path):
        """
        Write the given gobbli embedding input into the format expected by BERT.
        Make sure the given directory exists first.
        """
        input_dir.mkdir(exist_ok=True, parents=True)
        input_path = input_dir / BERT._EMBEDDING_INPUT_FILE
        input_path.write_text(escape_line_delimited_texts(embed_input.X))

    def _embed(
        self, embed_input: gobbli.io.EmbedInput, context: ContainerTaskContext
    ) -> gobbli.io.EmbedOutput:

        self._write_embed_input(embed_input, context.host_input_dir)

        # Determine checkpoint to use
        host_checkpoint_dir, container_checkpoint_dir, checkpoint_name = self._get_checkpoint(
            embed_input.checkpoint, context
        )

        # Use the second-to-last layer for embeddings as suggested:
        # https://github.com/hanxiao/bert-as-service#q-why-not-the-last-hidden-layer-why-second-to-last
        cmd = (
            "bash -c 'python extract_features.py"
            f" --input_file={context.container_input_dir / BERT._EMBEDDING_INPUT_FILE}"
            f" --output_file={context.container_output_dir / BERT._EMBEDDING_OUTPUT_FILE}"
            f" --vocab_file={BaseModel._CONTAINER_WEIGHTS_PATH}/vocab.txt"
            f" --bert_config_file={BaseModel._CONTAINER_WEIGHTS_PATH}/bert_config.json"
            f" --init_checkpoint={container_checkpoint_dir / checkpoint_name}"
            f" --do_lower_case={self.do_lower_case}"
            f" --layers=-2"
            f" --max_seq_length={self.max_seq_length}"
            f" --batch_size={embed_input.embed_batch_size}'"
        )

        run_kwargs = self._base_docker_run_kwargs(context)

        # Mount the checkpoint in the container if needed
        maybe_mount(
            run_kwargs["volumes"], host_checkpoint_dir, container_checkpoint_dir
        )

        container_logs = run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        X_embedded, embed_tokens = _read_embeddings(
            context.host_output_dir / BERT._EMBEDDING_OUTPUT_FILE, embed_input.pooling
        )

        return gobbli.io.EmbedOutput(
            X_embedded=X_embedded,
            embed_tokens=embed_tokens,
            _console_output=container_logs,
        )
