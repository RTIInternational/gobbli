import pandas as pd

from gobbli.inspect.evaluate import ClassificationError, ClassificationEvaluation
from gobbli.util import multilabel_to_indicator_df


def test_classification_evaluation_multiclass():
    results = ClassificationEvaluation(
        labels=["a", "b"],
        X=["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4", "b5"],
        y_true=["a", "a", "a", "a", "b", "b", "b", "b", "b"],
        y_pred_proba=pd.DataFrame(
            {
                "a": [0.7, 0.3, 0.4, 0.6, 0.9, 0.3, 0.6, 0.4, 0.7],
                "b": [0.3, 0.7, 0.6, 0.4, 0.1, 0.7, 0.4, 0.6, 0.3],
            }
        ),
    )

    # Ensure predicted labels are calculated correctly
    expected_y_pred = ["a", "b", "b", "a", "a", "b", "a", "b", "a"]
    assert results.y_pred_multiclass == expected_y_pred

    # Ensure errors are calculated correctly
    # Error observations are a2, a3, b1, b3, b5
    ea2 = ClassificationError(X="a2", y_true="a", y_pred_proba={"a": 0.3, "b": 0.7})
    ea3 = ClassificationError(X="a3", y_true="a", y_pred_proba={"a": 0.4, "b": 0.6})
    eb1 = ClassificationError(X="b1", y_true="b", y_pred_proba={"a": 0.9, "b": 0.1})
    eb3 = ClassificationError(  # noqa: F841
        X="b3", y_true="b", y_pred_proba={"a": 0.6, "b": 0.4}
    )
    eb5 = ClassificationError(X="b5", y_true="b", y_pred_proba={"a": 0.7, "b": 0.3})

    # Cut off at k=2 and ensure the correct errors are in there and ordered correctly
    errors = results.errors(k=2)

    # Expected errors
    a_false_positives = [eb1, eb5]
    a_false_negatives = [ea2, ea3]
    b_false_positives = a_false_negatives
    b_false_negatives = a_false_positives

    a_errors = errors["a"]
    b_errors = errors["b"]
    assert a_errors[0] == a_false_positives
    assert a_errors[1] == a_false_negatives
    assert b_errors[0] == b_false_positives
    assert b_errors[1] == b_false_negatives


def test_classification_evaluation_multilabel():
    labels = ["a", "b"]
    results = ClassificationEvaluation(
        labels=labels,
        X=["a1", "a2", "ab1", "ab2", "b1", "b2", "01", "02", "03"],
        y_true=[["a"], ["a"], ["a", "b"], ["a", "b"], ["b"], ["b"], [], [], []],
        y_pred_proba=pd.DataFrame(
            {
                "a": [0.7, 0.3, 0.4, 0.6, 0.9, 0.3, 0.6, 0.4, 0.3],
                "b": [0.3, 0.7, 0.6, 0.6, 0.1, 0.7, 0.4, 0.6, 0.3],
            }
        ),
    )

    # Ensure predicted labels are calculated correctly
    expected_y_pred = multilabel_to_indicator_df(
        [["a"], ["b"], ["b"], ["a", "b"], ["a"], ["b"], ["a"], ["b"], []], labels
    )
    pd.testing.assert_frame_equal(results.y_pred_multilabel, expected_y_pred)

    # Ensure errors are calculated correctly
    # Error observations are a2, ab1, b1, 01, 02
    ea2 = ClassificationError(X="a2", y_true=["a"], y_pred_proba={"a": 0.3, "b": 0.7})
    eab1 = ClassificationError(
        X="ab1", y_true=["a", "b"], y_pred_proba={"a": 0.4, "b": 0.6}
    )
    eb1 = ClassificationError(X="b1", y_true=["b"], y_pred_proba={"a": 0.9, "b": 0.1})
    e01 = ClassificationError(X="01", y_true=[], y_pred_proba={"a": 0.6, "b": 0.4})
    e02 = ClassificationError(X="02", y_true=[], y_pred_proba={"a": 0.4, "b": 0.6})

    # Cut off at k=2 and ensure the correct errors are in there and ordered correctly
    errors = results.errors(k=2)

    # Expected errors
    a_false_positives = [eb1, e01]
    a_false_negatives = [ea2, eab1]
    b_false_positives = [ea2, e02]
    b_false_negatives = [eb1]

    a_errors = errors["a"]
    b_errors = errors["b"]
    assert a_errors[0] == a_false_positives
    assert a_errors[1] == a_false_negatives
    assert b_errors[0] == b_false_positives
    assert b_errors[1] == b_false_negatives
