from typing import Any

import numpy as np
import pandas as pd

import gobbli.io
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.model.mixin import PredictMixin, TrainMixin


class MajorityClassifier(BaseModel, TrainMixin, PredictMixin):
    """
    Simple classifier that returns the majority class from the training set.

    Useful for ensuring user code works with the gobbli input/output format
    without having to build a time-consuming model.
    """

    def init(self, params):
        self.majority_class: Any = None

    def _build(self):
        """
        No build step required for this model.
        """

    def _train(
        self, train_input: gobbli.io.TrainInput, context: ContainerTaskContext
    ) -> gobbli.io.TrainOutput:
        """
        Determine the majority class.
        """
        unique_values, value_counts = np.unique(train_input.y_train, return_counts=True)
        self.majority_class = unique_values[value_counts.argmax(axis=0)]

        y_train_pred = np.full_like(train_input.y_train, self.majority_class)
        train_loss = np.sum(y_train_pred != train_input.y_train)

        y_valid_pred = np.full_like(train_input.y_valid, self.majority_class)
        valid_loss = np.sum(y_valid_pred != train_input.y_valid)
        valid_accuracy = valid_loss / y_valid_pred.shape[0]

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
        pred_proba_df = pd.DataFrame(
            {
                label: 1 if label == self.majority_class else 0
                for label in predict_input.labels
            },
            index=range(len(predict_input.X)),
        )

        return gobbli.io.PredictOutput(y_pred_proba=pred_proba_df)
