import enum
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import pandas as pd

from gobbli.util import (
    TokenizeMethod,
    as_multiclass,
    as_multilabel,
    collect_labels,
    detokenize,
    is_multilabel,
    multiclass_to_multilabel_target,
    pred_prob_to_pred_label,
    pred_prob_to_pred_multilabel,
    tokenize,
)

T = TypeVar("T")


def _check_string_list(obj: Any):
    """
    Verify a given object is a list containing strings.
    """
    if not isinstance(obj, list):
        raise TypeError(f"obj must be a list, got '{type(obj)}'")
    if len(obj) > 0 and not isinstance(obj[0], str):
        raise TypeError(f"obj must contain strings, got '{type(obj[0])}'")


def _check_multilabel_list(obj: Any):
    """
    Verify a given object is a list containing lists of strings (labels).
    """
    if not isinstance(obj, list):
        raise TypeError(f"obj must be a list, got '{type(obj)}'")

    if len(obj) > 0:
        if not isinstance(obj[0], list):
            raise TypeError(f"obj must contain lists, got '{type(obj[0])}'")

        if len(obj[0]) > 0:
            if not isinstance(obj[0][0], str):
                raise TypeError(
                    f"obj must contain lists of strings, got lists of '{type(obj[0][0])}'"
                )


def validate_X(X: List[str]):
    """
    Confirm a given array matches the expected type for input.

    Args:
      X: Something that should be valid model input.
    """
    _check_string_list(X)


def validate_multilabel_y(y: Union[List[str], List[List[str]]], multilabel: bool):
    """
    Confirm an array is typed appropriately for the value of ``multilabel``.

    Args:
      y: Something that should be valid multiclass or multilabel output.
      multilabel: True if y should be formatted for a multilabel problem
        and False otherwise (for a multiclass problem).
    """
    if multilabel:
        _check_multilabel_list(y)
    else:
        _check_string_list(y)


def validate_X_y(X: List[str], y: List[Any]):
    """
    Assuming X is valid input and y is valid output, ensure they match sizes.

    Args:
      X: Something that should be valid model input.
      y: Something that should be valid model output.
    """
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length; X has length {len(X)}, and y has length {len(y)}"
        )


class TaskIO(ABC):
    """
    Base class for classes used for task input/output.
    """

    _METADATA_FILENAME = "gobbli-task-meta.json"

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Returns:
          The task information that constitutes its metadata --
          generally parameters of an input task and/or summarized results
          of an output task.
        """
        raise NotImplementedError


@dataclass
class TrainInput(TaskIO):
    """
    Input for training a model.  See :meth:`gobbli.model.mixin.TrainMixin.train`.

    For usage specific to a multiclass or multilabel paradigm, consider using the
    more specifically checked and typed properties: ``y_{train,valid}_{multiclass,multilabel}``
    as opposed to the more generically typed ``y_{train,valid}`` attributes.

    Args:
      X_train: Documents used for training.
      y_train: Labels for training documents.
      X_valid: Documents used for validation.
      y_valid: Labels for validation documents.
      train_batch_size: Number of observations per batch on the training dataset.
      valid_batch_size: Number of observations per batch on the validation dataset.
      num_train_epochs: Number of epochs to use for training.
      checkpoint: Checkpoint containing trained weights for the model.  If passed,
        training will continue from the checkpoint instead of starting from scratch.
        See :paramref:`TrainOutput.params.checkpoint`.
    """

    X_train: List[str]
    y_train: Union[List[str], List[List[str]]]
    X_valid: List[str]
    y_valid: Union[List[str], List[List[str]]]
    train_batch_size: int = 32
    valid_batch_size: int = 8
    num_train_epochs: int = 3
    checkpoint: Optional[Path] = None

    @property
    def y_train_multiclass(self) -> List[str]:
        return as_multiclass(self.y_train, self.multilabel)

    @property
    def y_train_multilabel(self) -> List[List[str]]:
        return as_multilabel(self.y_train, self.multilabel)

    @property
    def y_valid_multiclass(self) -> List[str]:
        if self.multilabel:
            raise ValueError(
                "Multilabel training input can't be converted to multiclass."
            )
        return cast(List[str], self.y_valid)

    @property
    def y_valid_multilabel(self) -> List[List[str]]:
        if self.multilabel:
            return cast(List[List[str]], self.y_valid)
        return multiclass_to_multilabel_target(cast(List[str], self.y_valid))

    def labels(self) -> List[str]:
        """
        Returns:
          The set of unique labels in the data.
          Sort and return a list for consistent ordering, in case that matters.
        """
        # We verify these types are compatible during initialization, so ignore
        # mypy warning about a possible mismatch due to the Union
        return collect_labels(self.y_train + self.y_valid)  # type: ignore

    def __post_init__(self):
        self.multilabel = is_multilabel(self.y_train)

        for X in (self.X_train, self.X_valid):
            validate_X(X)

        for y in self.y_train, self.y_valid:
            validate_multilabel_y(y, self.multilabel)

        for X, y in ((self.X_train, self.y_train), (self.X_valid, self.y_valid)):
            validate_X_y(X, y)

    def metadata(self) -> Dict[str, Any]:
        return {
            "train_batch_size": self.train_batch_size,
            "valid_batch_size": self.valid_batch_size,
            "num_train_epochs": self.num_train_epochs,
            "len_X_train": len(self.X_train),
            "len_y_train": len(self.y_train),
            "len_X_valid": len(self.X_valid),
            "len_y_valid": len(self.y_valid),
            "multilabel": self.multilabel,
            "checkpoint": self.checkpoint,
        }


@dataclass
class TrainOutput(TaskIO):
    """
    Output from model training.  See :meth:`gobbli.model.mixin.TrainMixin.train`.

    Args:
      valid_loss: Loss on the validation dataset.
      valid_accuracy: Accuracy on the validation dataset.
      train_loss:  Loss on the training dataset.
      labels: List of labels present in the training data.
        Used to initialize the model for prediction.
      multilabel: True if the model was trained in a multilabel context,
        otherwise False (indicating a multiclass context).
      checkpoint: Path to the best checkpoint from training.
        This may not be a literal filepath in the case of ex. TensorFlow,
        but it should give the user everything they need to run prediction
        using the results of training.
      _console_output: Raw console output from the container used to train the model.
    """

    valid_loss: float
    valid_accuracy: float
    train_loss: float
    labels: List[str]
    multilabel: bool
    checkpoint: Optional[Path] = None
    _console_output: str = ""

    def metadata(self) -> Dict[str, Any]:
        return {
            "valid_loss": float(self.valid_loss),
            "valid_accuracy": float(self.valid_accuracy),
            "train_loss": float(self.train_loss),
            "multilabel": self.multilabel,
            "labels": self.labels,
            "checkpoint": str(self.checkpoint),
        }


@dataclass
class PredictInput(TaskIO):
    """
    Input for generating predictions using a model.  See :meth:`gobbli.model.mixin.PredictMixin.predict`.

    Args:
      X: Documents to have labels predicted for.
      labels: See :paramref:`TrainOutput.params.labels`.
      multilabel: True if the model was trained in a multilabel context,
        otherwise False (indicating a multiclass context).
      predict_batch_size: Number of documents to predict in each batch.
      checkpoint: Checkpoint containing trained weights for the model.
        See :paramref:`TrainOutput.params.checkpoint`.
    """

    X: List[str]
    labels: List[str]
    multilabel: bool = False
    predict_batch_size: int = 32
    checkpoint: Optional[Path] = None

    def __post_init__(self):
        validate_X(self.X)

    def metadata(self) -> Dict[str, Any]:
        return {
            "predict_batch_size": self.predict_batch_size,
            "labels": self.labels,
            "checkpoint": str(self.checkpoint),
            "multilabel": self.multilabel,
            "len_X": len(self.X),
        }


@dataclass
class PredictOutput(TaskIO):
    """
    Output from generating predictions using a model.  See :meth:`gobbli.model.mixin.PredictMixin.predict`.

    Args:
      y_pred_proba: A dataframe containing the predicted probablity for each class.
        There is a row for each observation and a column for each class.
      _console_output: Raw console output from the container used to generate predictions.
    """

    y_pred_proba: pd.DataFrame
    _console_output: str = ""

    @property
    def y_pred(self) -> List[str]:
        """
        Returns:
          The most likely predicted label for each observation.
        """
        return pred_prob_to_pred_label(self.y_pred_proba)

    def y_pred_multilabel(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Returns:
          Indicator matrix representing the predicted labels for each observation
          using the given (optional) threshold.
        """
        return pred_prob_to_pred_multilabel(self.y_pred_proba, threshold)

    def __post_init__(self):
        validate_multilabel_y(self.y_pred, False)

    def metadata(self) -> Dict[str, Any]:
        return {"len_y_pred": self.y_pred_proba.shape[0]}


@enum.unique
class EmbedPooling(enum.Enum):
    """
    Enum describing all the different pooling methods that can be used
    when generating embeddings.

    Attributes:
      MEAN: Take the mean across all tokens as the embedding for the document.
      NONE: Return the token-wise embeddings for each document.
    """

    MEAN = "mean"
    NONE = "none"


@dataclass
class EmbedInput(TaskIO):
    """
    Input for generating embeddings using a model.  See :meth:`gobbli.model.mixin.EmbedMixin.embed`.

    Args:
      X: Documents to generate embeddings for.
      embed_batch_size: Number of documents to embed at a time.
      pooling: Pooling method to use for resulting embeddings.
      checkpoint: Checkpoint containing trained weights for the model.
        See :paramref:`TrainOutput.params.checkpoint`.
    """

    X: List[str]

    # Number of documents to embed at a time
    embed_batch_size: int = 32

    # Pooling method to use for resulting embeddings
    pooling: EmbedPooling = EmbedPooling.MEAN

    # Checkpoint containing trained weights for the model
    # See TrainOutput.checkpoint
    checkpoint: Optional[Path] = None

    def __post_init__(self):
        validate_X(self.X)

    def metadata(self) -> Dict[str, Any]:
        return {
            "embed_batch_size": self.embed_batch_size,
            "pooling": self.pooling.value,
            "checkpoint": str(self.checkpoint),
            "len_X": len(self.X),
        }


@dataclass
class EmbedOutput(TaskIO):
    """
    Output from generating embeddings.  See :meth:`gobbli.model.mixin.EmbedMixin.embed`.

    Args:
      X_embedded: A list of ndarrays representing the embedding for each document.
        The shape of each array depends on pooling method. ``l`` = length of the document, and
        ``d`` = dimensionality of embedding.

        - Mean pooling (default): ``(d,)``
        - No pooling: ``(l, d)``
      embed_tokens: If pooling strategy is "NONE", this is the list of tokens
        corresponding to each embedding for each document.  Otherwise, it's :obj:`None`.
      _console_output: Raw console output from the container used to generate the embeddings.
    """

    X_embedded: List[np.ndarray]
    embed_tokens: Optional[List[List[str]]] = None
    _console_output: str = ""

    def metadata(self) -> Dict[str, Any]:
        return {"len_X_embedded": len(self.X_embedded)}


def _chunk_tokens(tokens: List[str], window_len: int) -> Iterator[List[str]]:
    for i in range(0, len(tokens), window_len):
        yield tokens[i : i + window_len]


def make_document_windows(
    X: List[str],
    window_len: int,
    y: Optional[List[T]] = None,
    tokenize_method: TokenizeMethod = TokenizeMethod.SPLIT,
    model_path: Optional[Path] = None,
    vocab_size: Optional[int] = None,
) -> Tuple[List[str], List[int], Optional[List[T]]]:
    """
    This is a helper for when you have a dataset with long documents which is going to be
    passed through a model with a fixed max sequence length.  If you don't have enough
    memory to raise the max sequence length, but you don't want to miss out on the information
    in longer documents, you can use this helper to generate a dataset that splits each
    document into windows roughly the size of your ``max_seq_len``.  The resulting dataset can
    then be used to train your model.  You should then use :func:`pool_document_windows` to
    pool the results from downstream tasks (ex. predictions, embeddings).

    Note there may still be some mismatch between the window size and the size as tokenized
    by your model, since some models use custom tokenization methods.

    Args:
      X: List of texts to make windows out of.
      window_len: The maximum length of each window.  This should roughly correspond to
        the ``max_seq_len`` of your model.
      y: Optional list of classes (or list of list of labels).  If passed, a corresponding
        list of targets for each window (the target(s) associated with the window's document)
        will be returned.
      tokenize_method: :class:`gobbli.util.TokenizeMethod` corresponding to the tokenization
        method to use for determining windows.
      model_path: This argument is used if the tokenization method requires
        training a model; otherwise, it's ignored.  Path for a tokenization model.
        If it doesn't exist, a new tokenization model will be trained and saved at
        the given path.  If it does exist, the existing model will be used.  If no path
        is given, a temporary directory will be created/used and discarded
      vocab_size: Number of terms in the vocabulary for tokenization. May be ignored depending
        on the tokenization method and whether a model is already trained.

    Returns:
      A 3-tuple containing a new list of texts split into windows, a corresponding list
      containing the index of each original document for each window, and (optionally)
      a list containing a target per window.  The index should
      be used to pool the output from the windowed text (see :func:`pool_document_windows`).
    """
    X_windowed: List[str] = []
    X_windowed_indices: List[int] = []
    y_windowed: List[T] = []

    # Create a temp dir in case it's needed
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenize_kwargs: Dict[str, Any] = {}

        if model_path is None:
            model_path = Path(tmpdir) / "tokenizer"

        tokenize_kwargs["model_path"] = model_path

        detokenize_kwargs = tokenize_kwargs.copy()

        if vocab_size is not None:
            tokenize_kwargs["vocab_size"] = vocab_size

        for i, tokens in enumerate(tokenize(tokenize_method, X, **tokenize_kwargs)):
            for window in detokenize(
                tokenize_method, _chunk_tokens(tokens, window_len), **detokenize_kwargs
            ):
                X_windowed.append(window)
                X_windowed_indices.append(i)
                if y is not None:
                    y_windowed.append(y[i])

    if y is not None:
        return X_windowed, X_windowed_indices, y_windowed
    else:
        return X_windowed, X_windowed_indices, None


@enum.unique
class WindowPooling(enum.Enum):
    """
    Enum describing all the different pooling methods that can be used
    when pooling model output from windowed documents.

    Attributes:
      MEAN: Take the mean across all dimensions/classes as the output for the document.
      MAX: Take the max across all dimensions/classes as the output for the document.
      MIN: Take the min across all dimensions/classes as the output for the document.
    """

    MEAN = "mean"
    MAX = "max"
    MIN = "min"


def pool_document_windows(
    unpooled_output: Union[PredictOutput, EmbedOutput],
    window_indices: List[int],
    pooling: WindowPooling = WindowPooling.MEAN,
):
    """
    This helper pools output from a model whose input was document windows generated by
    :func:`make_document_windows`.  The output can be pooled in multiple ways.  See
    :class:`WindowPooling` for more info.

    This function mutates the passed output object to preserve other information in the
    output object.

    Args:
      unpooled_output: The output from the model to be pooled.
      window_indices: A list (size = number of rows in ``unpooled_output``) of integers corresponding
        to the index of the original document for each window.  These are used to group the window
        output appropriately.
      pooling: The method to use for pooling.
    """
    if isinstance(unpooled_output, PredictOutput):
        unpooled_df = unpooled_output.y_pred_proba
    elif isinstance(unpooled_output, EmbedOutput):
        if unpooled_output.embed_tokens is not None:
            raise ValueError(
                "Embedding output must be pooled when pooling document windows."
            )
        unpooled_df = pd.DataFrame(unpooled_output.X_embedded)
    else:
        raise TypeError(
            f"Unsupported type for unpooled_output: '{type(unpooled_output)}'"
        )

    if not unpooled_df.shape[0] == len(window_indices):
        raise ValueError(f"Unpooled output and window indices must have same length")

    unpooled_df.index = window_indices
    unpooled_grp = unpooled_df.groupby(unpooled_df.index)

    if pooling == WindowPooling.MEAN:
        pooled_df = unpooled_grp.mean()
    elif pooling == WindowPooling.MAX:
        pooled_df = unpooled_grp.max()
    elif pooling == WindowPooling.MIN:
        pooled_df = unpooled_grp.min()
    else:
        raise ValueError(f"Unsupported pooling value: {pooling}")

    if isinstance(unpooled_output, PredictOutput):
        unpooled_output.y_pred_proba = pooled_df
    elif isinstance(unpooled_output, EmbedOutput):
        unpooled_output.X_embedded = [arr for arr in pooled_df.values]
