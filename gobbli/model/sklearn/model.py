import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import gobbli.io
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.model.mixin import PredictMixin, TrainMixin
from gobbli.util import assert_type, generate_uuid


def persist_estimator(estimator: BaseEstimator) -> Path:
    """
    Saves the given estimator to a gobbli-managed filepath, where it can be loaded from
    disk by the SKLearnClassifier.  This is useful if you want to use an estimator but
    don't want to bother with saving it to disk on your own.

    Args:
      estimator: The estimator to load.

    Returns:
      The path where the estimator was saved.
    """
    estimator_dir = (
        SKLearnClassifier.model_class_dir() / "user_estimators" / generate_uuid()
    )
    estimator_dir.mkdir(exist_ok=True, parents=True)

    estimator_path = estimator_dir / SKLearnClassifier._TRAIN_OUTPUT_CHECKPOINT
    SKLearnClassifier._dump_estimator(estimator, estimator_path)

    return estimator_path


def make_default_tfidf_logistic_regression() -> BaseEstimator:
    """
    Returns:
      A pipeline composing a TF-IDF vectorizer and a logistic regression model using
      default parameters.
    """
    return Pipeline(
        [("tfidf", TfidfVectorizer()), ("logreg", LogisticRegression(random_state=1))]
    )


def make_cv_tfidf_logistic_regression(
    grid_params: Optional[Dict[str, Any]] = None
) -> BaseEstimator:
    """
    Args:
      grid_params: Grid search parameters for the pipeline.  Passed directly to
        :class:`sklearn.model_selection.GridSearchCV`.  See
        :func:`make_default_tfidf_logistic_regression` for the names of the pipeline
        components.  If not given, will use a somewhat reasonable default.

    Returns:
      A cross-validated pipeline combining a TF-IDF vectorizer and logistic regression model
      with the specified grid parameters.
    """
    if grid_params is None:
        grid_params = {
            "tfidf__ngram_range": [(1, 2)],
            "tfidf__min_df": [0.01],
            "tfidf__max_df": [0.95],
            "logreg__C": [0.1, 0.5, 1],
            "logreg__l1_ratio": [0, 0.25, 0.5, 0.75, 1],
            "logreg__solver": ["saga"],
            "logreg__penalty": ["elasticnet"],
        }
    return GridSearchCV(
        make_default_tfidf_logistic_regression(),
        grid_params,
        cv=5,
        return_train_score=True,
        verbose=10,
    )


_AT_LEAST_TWO_CLASSES_ERR_MSG = (
    "This solver needs samples of at least 2 classes in the data"
)


class _SafeEstimator(BaseEstimator, ClassifierMixin):
    """
    Wrap an arbitrary classifier estimator to catch errors when fitting
    models that require more than 1 class in the data.
    """

    def __init__(self, base_estimator: BaseEstimator):
        self.base_estimator = base_estimator
        self.classes_: Optional[np.ndarray] = None

        if hasattr(base_estimator, "classes_"):
            self.classes_ = self.base_estimator.classes_

    def fit(self, *args, **kwargs):
        try:
            return self.base_estimator.fit(*args, **kwargs)
        except ValueError as e:
            if _AT_LEAST_TWO_CLASSES_ERR_MSG not in str(e):
                raise
        finally:
            self.classes_ = self.base_estimator.classes_

    def predict_proba(self, X):
        if self.classes_ is None:
            raise ValueError(
                "Can't predict without knowing what the estimator's classes are."
            )

        if len(self.classes_) == 1:
            return np.ones((len(X), 1))
        return self.base_estimator.predict_proba(X)

    def predict(self, X):
        if self.classes_ is None:
            raise ValueError(
                "Can't predict without knowing what the estimator's classes are."
            )

        if len(self.classes_) == 1:
            return np.full_like(X, self.classes_[0])
        return self.base_estimator.predict(X)


class SKLearnClassifier(BaseModel, TrainMixin, PredictMixin):
    """
    Classifier wrapper for `scikit-learn <https://scikit-learn.org/stable/>`__ classifiers.
    Wraps a :class:`sklearn.base.BaseEstimator` which accepts text input and outputs
    predictions.

    Creating an estimator that meets those conditions will generally require
    some use of :class:`sklearn.pipeline.Pipeline` to compose a transform (e.g. a vectorizer
    to vectorize text) and an estimator (e.g. logistic regression).  See the helper functions
    in this module for some examples.  You may also consider wrapping the pipeline with
    :class:`sklearn.model_selection.GridSearchCV` to tune hyperparameters.
    """

    _TRAIN_OUTPUT_CHECKPOINT = "estimator.joblib"

    def init(self, params: Dict[str, Any]):
        """
        See :meth:`gobbli.model.base.BaseModel.init`.

        SKLearnClassifier parameters:

        - ``estimator_path`` (:obj:`str`): Path to an estimator pickled by joblib.
          The pickle will be loaded, and the resulting object will be used as the estimator.
        """
        estimator = None

        for name, value in params.items():
            if name == "estimator_path":
                assert_type(name, value, str)
                estimator = SKLearnClassifier._load_estimator(Path(value))
                SKLearnClassifier._validate_estimator(estimator)
            else:
                raise ValueError(f"Unknown param '{name}'")

        if estimator is None:
            self.estimator = _SafeEstimator(make_default_tfidf_logistic_regression())
        else:
            self.estimator = _SafeEstimator(estimator)

    @staticmethod
    def _load_estimator(estimator_path: Path) -> BaseEstimator:
        return joblib.load(estimator_path)

    @staticmethod
    def _dump_estimator(estimator: BaseEstimator, estimator_path: Path):
        joblib.dump(estimator, estimator_path)

    @staticmethod
    def _validate_estimator(estimator: BaseEstimator):
        """
        Run some checks on the given object to determine if it's an estimator which is
        valid for our purposes.
        """
        # sklearn has a function that does a lot more intensive checking regarding
        # the interface of a candidate Estimator
        # (sklearn.utils.estimator_checks.check_estimator), but the function
        # doesn't work well for our use case as of version 0.22.  It doesn't properly
        # detect Pipeline X_types based on the first pipeline component and won't
        # test anything that doesn't accept a 2-D numpy array as input.  We'll settle
        # for lax checks here until sklearn has something that works better for us.
        if not is_classifier(estimator):
            raise ValueError(
                "Estimator must be a classifier according to sklearn.base.is_classifier()"
            )

        if not hasattr(estimator, "predict_proba"):
            raise ValueError(
                "Estimator must support the predict_proba() method to fulfill gobbli's "
                "interface requirements for a prediction model."
            )

    def _build(self):
        """
        No build step required for this model.
        """

    def _train(
        self, train_input: gobbli.io.TrainInput, context: ContainerTaskContext
    ) -> gobbli.io.TrainOutput:
        if train_input.checkpoint is not None:
            warnings.warn(
                "SKLearnClassifier does not support training from an existing "
                "checkpoint, so the passed checkpoint will be ignored."
            )
        self.estimator.fit(train_input.X_train, train_input.y_train)

        y_train_pred = self.estimator.predict(train_input.X_train)
        train_loss = -f1_score(
            train_input.y_train, y_train_pred, zero_division="warn", average="weighted"
        )

        y_valid_pred = self.estimator.predict(train_input.X_valid)
        valid_loss = -f1_score(
            train_input.y_valid, y_valid_pred, zero_division="warn", average="weighted"
        )
        valid_accuracy = accuracy_score(train_input.y_valid, y_valid_pred)

        checkpoint_path = (
            context.host_output_dir / SKLearnClassifier._TRAIN_OUTPUT_CHECKPOINT
        )
        self._dump_estimator(self.estimator.base_estimator, checkpoint_path)

        return gobbli.io.TrainOutput(
            valid_loss=valid_loss,
            valid_accuracy=valid_accuracy,
            train_loss=train_loss,
            labels=train_input.labels(),
            checkpoint=checkpoint_path,
        )

    def _predict(
        self, predict_input: gobbli.io.PredictInput, context: ContainerTaskContext
    ) -> gobbli.io.PredictOutput:

        if predict_input.checkpoint is not None:
            self.estimator = _SafeEstimator(
                self._load_estimator(predict_input.checkpoint)
            )

        pred_proba_df = pd.DataFrame(self.estimator.predict_proba(predict_input.X))
        if self.estimator.classes_ is None:
            raise ValueError(
                "Can't determine column names for predicted probabilities."
            )
        pred_proba_df.columns = self.estimator.classes_.astype("str")

        labels = predict_input.labels
        for label in labels:
            if label not in pred_proba_df.columns:
                pred_proba_df[label] = 0.0

        return gobbli.io.PredictOutput(y_pred_proba=pred_proba_df)
