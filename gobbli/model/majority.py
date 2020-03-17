import itertools
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import gobbli.io
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.model.mixin import PredictMixin, TrainMixin
from gobbli.util import multilabel_to_indicator_df


class MajorityClassifier(BaseModel, TrainMixin, PredictMixin):
    """
    Simple classifier that returns the majority class from the training set.

    Useful for ensuring user code works with the gobbli input/output format
    without having to build a time-consuming model.
    """

    def init(self, params: Dict[str, Any]):
        self.majority_class: Any = None

    def _build(self):
        """
        No build step required for this model.
        """

    def _make_pred_df(self, labels: List[str], size: int) -> pd.DataFrame:
        return pd.DataFrame(
            {label: 1 if label == self.majority_class else 0 for label in labels},
            index=range(size),
        )

    def _train(
        self, train_input: gobbli.io.TrainInput, context: ContainerTaskContext
    ) -> gobbli.io.TrainOutput:
        """
        Determine the majority class.
        """
        unique_values, value_counts = np.unique(
            list(itertools.chain(train_input.y_train_multilabel)), return_counts=True
        )
        self.majority_class = unique_values[value_counts.argmax(axis=0)]

        labels = train_input.labels()
        y_train_pred = self._make_pred_df(labels, len(train_input.y_train))
        y_train_indicator = multilabel_to_indicator_df(
            train_input.y_train_multilabel, labels
        )
        train_loss = (y_train_pred.subtract(y_train_indicator)).abs().to_numpy().sum()

        y_valid_pred = self._make_pred_df(labels, len(train_input.y_valid))
        y_valid_indicator = multilabel_to_indicator_df(
            train_input.y_valid_multilabel, labels
        )
        valid_loss = (y_valid_pred.subtract(y_valid_indicator)).abs().to_numpy().sum()
        valid_accuracy = valid_loss / (y_valid_pred.shape[0] * y_valid_pred.shape[1])

        return gobbli.io.TrainOutput(
            valid_loss=valid_loss,
            valid_accuracy=valid_accuracy,
            train_loss=train_loss,
            labels=train_input.labels(),
        )

    def _predict(
        self, predict_input: gobbli.io.PredictInput, context: ContainerTaskContext
    ) -> gobbli.io.PredictOutput:
        """
        Predict based on our learned majority class.
        """
        pred_proba_df = self._make_pred_df(predict_input.labels, len(predict_input.X))

        return gobbli.io.PredictOutput(y_pred_proba=pred_proba_df)
