import numpy as np
import pandas as pd
import pytest

from gobbli.io import (
    EmbedOutput,
    PredictOutput,
    TrainOutput,
    WindowPooling,
    make_document_windows,
    pool_document_windows,
    validate_multilabel_y,
    validate_X,
    validate_X_y,
)


@pytest.mark.parametrize(
    "obj,err_class",
    [
        # Wrong collection type (series)
        (pd.Series("a"), TypeError),
        # Wrong collection type (numpy array)
        (np.array(["a"]), TypeError),
        # Wrong type (float)
        ([1.0], TypeError),
        # Wrong type (int)
        ([1], TypeError),
        # Correct type
        (["a"], None),
        # Empty
        ([], None),
    ],
)
def test_validate_X(obj, err_class):
    if err_class is None:
        validate_X(obj)
    else:
        with pytest.raises(err_class):
            validate_X(obj)


@pytest.mark.parametrize(
    "multilabel,obj,err_class",
    [
        # Wrong collection type (series)
        (False, pd.Series(["a"]), TypeError),
        # Wrong collection type (numpy array)
        (False, np.array([["a"]]), TypeError),
        # Wrong type (list of str)
        (False, [["a"]], TypeError),
        # Wrong type (float)
        (False, [1.0], TypeError),
        # Wrong type (list of float)
        (False, [[1.0]], TypeError),
        # Wrong type (empty list)
        (False, [[]], TypeError),
        # Correct type (str)
        (False, ["1"], None),
        # Empty
        (False, [], None),
        # Wrong collection type (series)
        (True, pd.Series(["a"]), TypeError),
        # Wrong collection type (numpy array)
        (True, np.array([["a"]]), TypeError),
        # Wrong type (float)
        (True, [1.0], TypeError),
        # Wrong type (str)
        (True, ["1"], TypeError),
        # Wrong type (list of float)
        (True, [[1.0]], TypeError),
        # Correct type (list of str)
        (True, [["a"]], None),
        # Empty label
        (True, [[]], None),
        # Empty
        (True, [], None),
    ],
)
def test_validate_multilabel_y(multilabel, obj, err_class):
    if err_class is None:
        validate_multilabel_y(obj, multilabel)
    else:
        with pytest.raises(err_class):
            validate_multilabel_y(obj, multilabel)


@pytest.mark.parametrize(
    "X,y,err_class",
    [
        # Same size, empty
        ([], [], None),
        # X is empty
        (["a"], [], ValueError),
        # y is empty
        ([], [1], ValueError),
        # X is longer
        (["a", "b"], [1], ValueError),
        # y is longer
        (["a"], [1, 2], ValueError),
        # Same size, non-empty
        (["a"], [1], None),
    ],
)
def test_validate_X_y(X, y, err_class):
    if err_class is None:
        validate_X_y(X, y)
    else:
        with pytest.raises(err_class):
            validate_X_y(X, y)


@pytest.mark.parametrize(
    "df,y_pred",
    [
        (pd.DataFrame({"a": [], "b": []}, dtype="float"), []),
        (pd.DataFrame({"a": [1], "b": [0]}), ["a"]),
        (pd.DataFrame({"a": [0.7, 0.3], "b": [0.3, 0.7]}), ["a", "b"]),
    ],
)
def test_predict_output(df, y_pred):
    predict_output = PredictOutput(y_pred_proba=df)
    assert y_pred == predict_output.y_pred


@pytest.mark.parametrize("has_y", [True, False])
@pytest.mark.parametrize(
    "docs,window_len,expected_windowed",
    [
        # One doc, one window
        (["a"], 1, (["a"], [0])),
        # One doc, two windows
        (["a b"], 1, (["a", "b"], [0, 0])),
        # One doc, partial window
        (["a b c"], 2, (["a b", "c"], [0, 0])),
        # Two docs, one window
        (["a", "b"], 1, (["a", "b"], [0, 1])),
        # Two docs, two windows
        (["a b", "c d"], 1, (["a", "b", "c", "d"], [0, 0, 1, 1])),
        # Two docs, partial windows
        (["a b c", "d e f"], 2, (["a b", "c", "d e", "f"], [0, 0, 1, 1])),
    ],
)
def test_make_document_windows(docs, window_len, expected_windowed, has_y):
    kwargs = {}
    if has_y:
        y = [str(i) for i in range(len(docs))]
        kwargs = {"y": y}

    windowed, indices, windowed_y = make_document_windows(docs, window_len, **kwargs)

    assert (windowed, indices) == expected_windowed

    if has_y:
        assert [str(i) for i in indices] == windowed_y
    else:
        assert windowed_y is None


@pytest.mark.parametrize("output_cls", [PredictOutput, EmbedOutput])
@pytest.mark.parametrize(
    "unpooled_data,indices,pooling,pooled_data",
    [
        # One window, mean
        ([[0, 1]], [0], WindowPooling.MEAN, [[0, 1]]),
        # One window, max
        ([[0, 1]], [0], WindowPooling.MAX, [[0, 1]]),
        # One window, min
        ([[0, 1]], [0], WindowPooling.MIN, [[0, 1]]),
        # Two windows, same doc, mean
        ([[0, 1], [1, 0]], [0, 0], WindowPooling.MEAN, [[0.5, 0.5]]),
        # Two windows, same doc, max
        ([[0, 1], [1, 0]], [0, 0], WindowPooling.MAX, [[1, 1]]),
        # Two windows, same doc, min
        ([[0, 1], [1, 0]], [0, 0], WindowPooling.MIN, [[0, 0]]),
        # Two windows, different docs, mean
        ([[0, 1], [1, 0]], [0, 1], WindowPooling.MEAN, [[0, 1], [1, 0]]),
        # Two windows, different docs, max
        ([[0, 1], [1, 0]], [0, 1], WindowPooling.MAX, [[0, 1], [1, 0]]),
        # Two windows, different docs, min
        ([[0, 1], [1, 0]], [0, 1], WindowPooling.MIN, [[0, 1], [1, 0]]),
    ],
)
def test_pool_document_windows(
    output_cls, unpooled_data, indices, pooling, pooled_data
):
    unpooled_df = pd.DataFrame(unpooled_data).rename(str, axis=1)
    pooled_df = pd.DataFrame(pooled_data).rename(str, axis=1)

    if output_cls == PredictOutput:
        actual_output = PredictOutput(y_pred_proba=unpooled_df)
    elif output_cls == EmbedOutput:
        actual_output = EmbedOutput(X_embedded=unpooled_df)
    else:
        raise TypeError(
            f"Unknown class for document window pooling: '{output_cls.__name__}'"
        )

    pool_document_windows(actual_output, indices, pooling=pooling)

    if output_cls == PredictOutput:
        pd.testing.assert_frame_equal(actual_output.y_pred_proba, pooled_df)
    elif output_cls == EmbedOutput:
        for expected, actual in zip(np.array(pooled_data), actual_output.X_embedded):
            np.testing.assert_array_equal(expected, actual)


def test_pool_document_windows_validation():
    # Embedding output without pooling should throw an error
    embed_output = EmbedOutput(
        X_embedded=[np.array([[0, 1], [1, 0]])], embed_tokens=[["a", "b"]]
    )
    with pytest.raises(ValueError):
        pool_document_windows(embed_output, [])

    # Train output is unsupported
    train_output = TrainOutput(
        valid_loss=0.0, valid_accuracy=0.0, train_loss=0.0, labels=[], multilabel=False
    )
    with pytest.raises(TypeError):
        pool_document_windows(train_output, [])

    # Output and indices length must match
    embed_output = EmbedOutput(X_embedded=[np.array([0, 1]), np.array([1, 0])])

    for bad_indices_len in (0, 1, 3):
        with pytest.raises(ValueError):
            pool_document_windows(embed_output, list(range(bad_indices_len)))

    # Pooling value must be valid
    with pytest.raises(ValueError):
        pool_document_windows(embed_output, [0, 1], pooling="bad")
