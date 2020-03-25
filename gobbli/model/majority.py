import itertools
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import gobbli.io
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.model.mixin import PredictMixin, TrainMixin
from gobbli.util import multilabel_to_indicator_df, pred_prob_to_pred_label


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
        if train_input.multilabel:
            train_labels: List[str] = list(
                itertools.chain.from_iterable(train_input.y_train_multilabel)
            )
        else:
            train_labels = train_input.y_train_multiclass

        unique_values, value_counts = np.unique(train_labels, return_counts=True)
        self.majority_class = unique_values[value_counts.argmax(axis=0)]

        labels = train_input.labels()
        y_train_pred_proba = self._make_pred_df(labels, len(train_input.y_train))
        y_valid_pred_proba = self._make_pred_df(labels, len(train_input.y_valid))

        if train_input.multilabel:
            y_train_indicator = multilabel_to_indicator_df(
                train_input.y_train_multilabel, labels
            )
            train_loss = (
                (y_train_pred_proba.subtract(y_train_indicator)).abs().to_numpy().sum()
            )

            y_valid_indicator = multilabel_to_indicator_df(
                train_input.y_valid_multilabel, labels
            )
            valid_loss = (
                (y_valid_pred_proba.subtract(y_valid_indicator)).abs().to_numpy().sum()
            )
            valid_accuracy = valid_loss / (
                y_valid_pred_proba.shape[0] * y_valid_pred_proba.shape[1]
            )
        else:
            y_train_pred = pred_prob_to_pred_label(y_train_pred_proba)
            train_loss = np.sum(y_train_pred != train_input.y_train_multiclass)

            y_valid_pred = pred_prob_to_pred_label(y_valid_pred_proba)
            valid_loss = np.sum(y_valid_pred != train_input.y_valid_multiclass)
            valid_accuracy = valid_loss / len(y_valid_pred)

        return gobbli.io.TrainOutput(
            valid_loss=valid_loss,
            valid_accuracy=valid_accuracy,
            train_loss=train_loss,
            labels=train_input.labels(),
            multilabel=train_input.multilabel,
        )

    def _predict(
        self, predict_input: gobbli.io.PredictInput, context: ContainerTaskContext
    ) -> gobbli.io.PredictOutput:
        """
        Predict based on our learned majority class.
        """
        pred_proba_df = self._make_pred_df(predict_input.labels, len(predict_input.X))

        return gobbli.io.PredictOutput(y_pred_proba=pred_proba_df)
