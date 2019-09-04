import pandas as pd

from gobbli.dataset.newsgroups import NewsgroupsDataset


def test_load_newsgroups(tmp_gobbli_dir):
    ds = NewsgroupsDataset.load()

    X_train = ds.X_train()
    X_test = ds.X_test()

    y_train = ds.y_train()
    y_test = ds.y_test()

    assert len(X_train) == 11314
    assert len(y_train) == 11314
    assert len(X_test) == 7532
    assert len(y_test) == 7532

    assert len(pd.unique(y_train)) == 20
    assert len(pd.unique(y_test)) == 20

    # Ensure these objects pass validation
    ds.train_input()
    ds.predict_input()
