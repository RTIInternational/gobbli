"""
Mixins which can be applied to classes derived from BaseModel.
"""
from abc import ABCMeta, abstractmethod
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Callable, Optional, cast

import gobbli.io
from gobbli.model.context import ContainerTaskContext
from gobbli.util import format_duration, generate_uuid, write_metadata


def _run_task(
    task_func: Callable[[Any, ContainerTaskContext], Any],
    task_input: gobbli.io.TaskIO,
    root_dir: Path,
    dir_name: Optional[str] = None,
) -> gobbli.io.TaskIO:
    """
    Run a task function that generates some output.  Can create a unique id
    to name the directory storing the input/output or use a user-provided name.
    Generate a context object to pass to the task.
    """
    if dir_name is None:
        task_id = generate_uuid()
        task_root_dir = root_dir / task_id
    else:
        task_root_dir = root_dir / dir_name

    if task_root_dir.exists():
        raise ValueError(
            f"Directory '{task_root_dir}' already exists.  Supply a different `dir_name`."
        )
    context = ContainerTaskContext(task_root_dir=task_root_dir)

    write_metadata(
        task_input.metadata(),
        context.host_input_dir / gobbli.io.TaskIO._METADATA_FILENAME,
    )

    task_output = cast(gobbli.io.TaskIO, task_func(task_input, context))

    write_metadata(
        task_output.metadata(),
        context.host_output_dir / gobbli.io.TaskIO._METADATA_FILENAME,
    )

    return task_output


class TrainMixin(metaclass=ABCMeta):
    """
    Apply to a model which can be trained in some way.
    """

    @abstractmethod
    def data_dir(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def logger(self):
        raise NotImplementedError

    def train_dir(self) -> Path:
        """
        The directory to be used for data related to training (data files, etc).

        Returns:
          Path to the training data directory.
        """
        return self.data_dir() / "train"

    def train(
        self, train_input: gobbli.io.TrainInput, train_dir_name: Optional[str] = None
    ) -> gobbli.io.TrainOutput:
        """
        Trains a model using params in the given :obj:`gobbli.io.TrainInput`.
        The training process varies depending on the model, but in general, it includes
        the following steps:

        - Update weights using the training dataset
        - Evaluate performance on the validation dataset.
        - Repeat for a number of epochs.
        - When finished, report loss/accuracy and return the trained weights.

        Args:
          train_input: Contains various parameters needed to determine
            how to train and what data to train on.
          train_dir_name: Optional name to store training input and output
            under.  The directory is always created under the model's
            :meth:`data_dir<gobbli.model.base.BaseModel.data_dir>`.  If a name is not given,
            a unique name is generated via a UUID.  If a name is given, that
            directory must not already exist.

        Returns:
          Output of training.
        """
        self.logger.info("Starting training.")
        start = timer()
        train_output = cast(
            gobbli.io.TrainOutput,
            _run_task(self._train, train_input, self.train_dir(), train_dir_name),
        )
        end = timer()
        for log_line in (
            f"Training finished in {format_duration(end - start)}.",
            "RESULTS:",
            f"  Validation loss: {train_output.valid_loss}",
            f"  Validation accuracy: {train_output.valid_accuracy}",
            f"  Training loss: {train_output.train_loss}",
        ):
            self.logger.info(log_line)

        return train_output

    @abstractmethod
    def _train(
        self, train_input: gobbli.io.TrainInput, context: ContainerTaskContext
    ) -> gobbli.io.TrainOutput:
        raise NotImplementedError


class PredictMixin(metaclass=ABCMeta):
    """
    Apply to a model which can be used to predict on new data.
    """

    @abstractmethod
    def data_dir(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def logger(self):
        raise NotImplementedError

    def predict_dir(self) -> Path:
        """
        The directory to be used for data related to prediction (weights, predictions, etc)

        Returns:
          Path to the prediction data directory.
        """
        return self.data_dir() / "predict"

    def predict(
        self,
        predict_input: gobbli.io.PredictInput,
        predict_dir_name: Optional[str] = None,
    ) -> gobbli.io.PredictOutput:
        """
        Runs prediction on new data using params containing in the given :obj:`gobbli.io.PredictInput`.

        Args:
          predict_input:  Contains various parameters needed to determine how to run prediction
            and what data to predict for.
          predict_dir_name: Optional name to store prediction input and output
            under.  The directory is always created under the model's
            :meth:`data_dir<gobbli.model.base.BaseModel.data_dir>`.  If a name is not given,
            a unique name is generated via a UUID.  If a name is given, that
            directory must not already exist.
        """
        self.logger.info("Starting prediction.")
        start = timer()
        predict_output = cast(
            gobbli.io.PredictOutput,
            _run_task(
                self._predict, predict_input, self.predict_dir(), predict_dir_name
            ),
        )
        end = timer()

        self.logger.info(f"Prediction finished in {format_duration(end - start)}.")

        return predict_output

    @abstractmethod
    def _predict(
        self, predict_input: gobbli.io.PredictInput, context: ContainerTaskContext
    ) -> gobbli.io.PredictOutput:
        raise NotImplementedError


class EmbedMixin(metaclass=ABCMeta):
    """
    Apply to a model which can be used to generate embeddings from data.
    """

    @abstractmethod
    def data_dir(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def logger(self):
        raise NotImplementedError

    def embed_dir(self) -> Path:
        """
        The directory to be used for data related to embedding (weights, embeddings, etc)

        Returns:
          Path to the embedding data directory.
        """
        return self.data_dir() / "embed"

    def embed(
        self, embed_input: gobbli.io.EmbedInput, embed_dir_name: Optional[str] = None
    ) -> gobbli.io.EmbedOutput:
        """
        Generates embeddings using a model and the params in the given :obj:`gobbli.io.EmbedInput`.

        Args:
          embed_input: Contains various parameters needed to determine
            how to generate embeddings and what data to generate embeddings for.
          embed_dir_name: Optional name to store embedding input and output
            under.  The directory is always created under the model's
            :meth:`data_dir<gobbli.model.base.BaseModel.data_dir>`.  If a name is not given,
            a unique name is generated via a UUID.  If a name is given, that
            directory must not already exist.

        Returns:
          Output of training.
        """
        self.logger.info("Generating embeddings.")
        start = timer()
        embed_output = cast(
            gobbli.io.EmbedOutput,
            _run_task(self._embed, embed_input, self.embed_dir(), embed_dir_name),
        )
        end = timer()

        self.logger.info(
            f"Embedding generation finished in {format_duration(end - start)}."
        )

        return embed_output

    @abstractmethod
    def _embed(
        self, embed_input: gobbli.io.EmbedInput, context: ContainerTaskContext
    ) -> gobbli.io.EmbedOutput:
        raise NotImplementedError
