import pandas as pd

from gobbli.dataset.imdb import IMDBDataset


def test_load_imdb(tmp_gobbli_dir):
    ds = IMDBDataset.load()

    X_train = ds.X_train()
    X_test = ds.X_test()

    y_train = ds.y_train()
    y_test = ds.y_test()

    assert len(X_train) == 25000
    assert len(y_train) == 25000
    assert len(X_test) == 25000
    assert len(y_test) == 25000

    assert len(pd.unique(y_train)) == 2
    assert len(pd.unique(y_test)) == 2

    # Ensure these objects pass validation
    ds.train_input()
    ds.predict_input()
