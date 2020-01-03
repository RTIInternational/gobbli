from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from gobbli.model.sklearn import (
    SKLearnClassifier,
    make_cv_tfidf_logistic_regression,
    make_default_tfidf_logistic_regression,
    persist_estimator,
)
from gobbli.model.sklearn.model import _AT_LEAST_TWO_CLASSES_ERR_MSG, _SafeEstimator


def _make_test_estimator():
    # Don't use anything complicated, like a Pipeline or GridSearchCV,
    # since those may have estimator params which don't compare cleanly
    return LogisticRegression()


def _assert_estimators_equal(clf1, clf2):
    assert type(clf1) is type(clf2)
    assert clf1.get_params() == clf2.get_params()


def test_dump_estimator(tmpdir):
    tempdir_path = Path(tmpdir)
    clf = _make_test_estimator()

    dump_path = tempdir_path / "test.joblib"
    SKLearnClassifier._dump_estimator(clf, dump_path)
    loaded_clf = joblib.load(dump_path)

    _assert_estimators_equal(clf, loaded_clf)


def test_load_estimator(tmpdir):
    tempdir_path = Path(tmpdir)
    clf = _make_test_estimator()

    dump_path = tempdir_path / "test.joblib"
    joblib.dump(clf, dump_path)
    loaded_clf = SKLearnClassifier._load_estimator(dump_path)

    _assert_estimators_equal(clf, loaded_clf)


def test_persist_estimator(tmp_gobbli_dir):
    clf = _make_test_estimator()
    clf_path = persist_estimator(clf)

    assert tmp_gobbli_dir in clf_path.parents

    loaded_clf = SKLearnClassifier._load_estimator(clf_path)

    _assert_estimators_equal(clf, loaded_clf)


@pytest.mark.parametrize(
    "clf,err",
    [
        # Not an estimator
        (None, ValueError),
        # Also not an estimator
        (1, ValueError),
        # Estimator but not a classifier
        (LinearRegression(), ValueError),
        # Estimator with no predict_proba
        (LinearSVC(), ValueError),
        # Valid estimator
        (LogisticRegression(), None),
        # Invalid pipeline (not a classifier)
        (Pipeline([("linreg", LinearRegression())]), ValueError),
        # Invalid pipeline (no predict_proba)
        (Pipeline([("svc", LinearSVC())]), ValueError),
        # Valid pipeline
        (Pipeline([("logreg", LogisticRegression())]), None),
        # Invalid grid search (not a classifier)
        (GridSearchCV(LinearRegression(), {}), ValueError),
        # Invalid grid search (no predict_proba)
        (GridSearchCV(LinearSVC(), {}), ValueError),
        # Valid grid search
        (GridSearchCV(LogisticRegression(), {}), None),
        # Our helpers should both be valid
        (make_cv_tfidf_logistic_regression(), None),
        (make_default_tfidf_logistic_regression(), None),
    ],
)
def test_validate_estimator(clf, err):
    if err is not None:
        with pytest.raises(err):
            SKLearnClassifier._validate_estimator(clf)
    else:
        SKLearnClassifier._validate_estimator(clf)


def test_safe_estimator():
    clf = make_default_tfidf_logistic_regression()

    X_train = ["test", "test2"]
    y_train = ["a", "a"]

    with pytest.raises(ValueError) as e:
        clf.fit(X_train, y_train)
    assert _AT_LEAST_TWO_CLASSES_ERR_MSG in str(e.value)

    safe_clf = _SafeEstimator(clf)
    safe_clf.fit(X_train, y_train)
    assert safe_clf.classes_.tolist() == ["a"]

    y_pred = safe_clf.predict(X_train)
    assert y_pred.tolist() == ["a", "a"]

    y_pred_proba = safe_clf.predict_proba(X_train)
    assert y_pred_proba.tolist() == [[1], [1]]


@pytest.mark.parametrize(
    "params,exception",
    [
        # Unknown param
        ({"unknown": None}, ValueError),
        # Bad type (estimator_path)
        ({"estimator_path": 1}, TypeError),
        # init loads the model path, and there isn't a good way to
        # reference a temp path from this param list, so we'll assume
        # other tests will catch a failure to initialize from a good
        # estimator path
    ],
)
def test_init(params, exception):
    with pytest.raises(exception):
        SKLearnClassifier(**params)
