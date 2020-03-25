import logging
from abc import ABC, abstractmethod
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, List, Optional, Tuple

from sklearn.model_selection import train_test_split

import gobbli.io
from gobbli.util import collect_labels, format_duration, gobbli_dir, shuffle_together

LOGGER = logging.getLogger(__name__)


def dataset_dir() -> Path:
    return gobbli_dir() / "dataset"


class BaseDataset(ABC):
    """
    Abstract base class for datasets used for benchmarking and testing.

    Derived classes should account for the following:
     - Dataset order should be consistent so limiting can work correctly
    """

    def __init__(self, *args, **kwargs):
        """
        Blank constructor needed to satisfy mypy
        """

    @classmethod
    def data_dir(cls) -> Path:
        return dataset_dir() / cls.__name__

    @classmethod
    def load(cls, *args, **kwargs) -> "BaseDataset":
        ds = cls(*args, **kwargs)

        if not ds._is_built():
            LOGGER.info("Dataset %s hasn't been built; building.", cls.__name__)
            start = timer()
            ds._build()
            end = timer()

            LOGGER.info(f"Dataset building finished in {format_duration(end - start)}.")

        return ds

    @abstractmethod
    def _is_built(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _build(self):
        raise NotImplementedError

    @abstractmethod
    def X_train(self):
        raise NotImplementedError

    @abstractmethod
    def y_train(self):
        raise NotImplementedError

    @abstractmethod
    def X_test(self):
        raise NotImplementedError

    @abstractmethod
    def y_test(self):
        raise NotImplementedError

    def _get_train_valid(
        self, limit: Optional[int] = None, shuffle_seed: int = 1234
    ) -> Tuple[List[str], List[Any]]:
        """
        Return the X and y used for training and validation with the
        appropriate limit applied.  Shuffle first to minimize the possibility of
        getting only a single label in a small/limited dataset if it happens to be ordered
        by label.
        """
        X_train_valid = self.X_train()
        y_train_valid = self.y_train()

        # Shuffle the two simultaneously so text and label stay together
        shuffle_together(X_train_valid, y_train_valid, shuffle_seed)

        if limit is not None:
            X_train_valid = X_train_valid[:limit]
            y_train_valid = y_train_valid[:limit]

        return X_train_valid, y_train_valid

    def train_input(
        self,
        train_batch_size: int = 32,
        valid_batch_size: int = 8,
        num_train_epochs: int = 3,
        valid_proportion: float = 0.2,
        split_seed: int = 1234,
        shuffle_seed: int = 1234,
        limit: Optional[int] = None,
    ) -> gobbli.io.TrainInput:
        if not self._is_built():
            raise ValueError("Dataset must be built before accessing train_input.")

        X_train_valid, y_train_valid = self._get_train_valid(
            limit=limit, shuffle_seed=shuffle_seed
        )

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_valid,
            y_train_valid,
            test_size=valid_proportion,
            random_state=split_seed,
        )

        return gobbli.io.TrainInput(
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            num_train_epochs=num_train_epochs,
        )

    def embed_input(
        self,
        embed_batch_size: int = 32,
        pooling: gobbli.io.EmbedPooling = gobbli.io.EmbedPooling.MEAN,
        limit: Optional[int] = None,
    ) -> gobbli.io.EmbedInput:
        if not self._is_built():
            raise ValueError("Dataset must be built before accessing embed_input.")

        X_test = self.X_test()
        if limit is not None:
            X_test = X_test[:limit]

        return gobbli.io.EmbedInput(
            X=X_test, embed_batch_size=embed_batch_size, pooling=pooling
        )

    def predict_input(
        self, predict_batch_size: int = 32, limit: Optional[int] = None
    ) -> gobbli.io.PredictInput:
        if not self._is_built():
            raise ValueError("Dataset must be built before accessing predict_input.")

        _, y_train_valid = self._get_train_valid(limit=limit)

        X_test = self.X_test()
        if limit is not None:
            X_test = X_test[:limit]

        labels = collect_labels(y_train_valid)

        return gobbli.io.PredictInput(
            X=X_test, predict_batch_size=predict_batch_size, labels=labels
        )
