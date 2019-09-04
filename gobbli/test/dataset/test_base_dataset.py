import pytest

from gobbli.test.util import MockDataset


def test_base_dataset_load():
    ds = MockDataset()

    # Dataset should be unbuilt after default initialization
    assert ds._build_count == 0

    ds = MockDataset.load()

    # Dataset should now be built
    assert ds._build_count == 1

    ds.load()

    # Dataset shouldn't have been built again
    assert ds._build_count == 1


def test_base_dataset_train_input():
    # Need to build first
    with pytest.raises(ValueError):
        MockDataset().train_input()

    ds = MockDataset.load()

    # No limit
    train_input = ds.train_input(valid_proportion=0.5)

    X_len = len(MockDataset.X_TRAIN_VALID)

    assert len(train_input.X_train) == X_len / 2
    assert len(train_input.y_train) == X_len / 2
    assert len(train_input.X_valid) == X_len / 2
    assert len(train_input.y_valid) == X_len / 2

    # Limit
    train_input = ds.train_input(valid_proportion=0.5, limit=2)

    assert len(train_input.X_train) == 1
    assert len(train_input.y_train) == 1
    assert len(train_input.X_valid) == 1
    assert len(train_input.y_valid) == 1


def test_base_dataset_predict_input():
    # Need to build first
    with pytest.raises(ValueError):
        MockDataset().train_input()

    ds = MockDataset.load()

    # No limit
    predict_input = ds.predict_input()

    X_len = len(MockDataset.X_TEST)

    assert len(predict_input.X) == X_len
    assert set(predict_input.labels) == set(MockDataset.Y_TEST)

    # Limit applied
    predict_input = ds.predict_input(limit=1)

    assert len(predict_input.X) == 1

    # Make sure we only have the labels from the limited subset
    assert set(predict_input.labels) < set(MockDataset.Y_TEST)
