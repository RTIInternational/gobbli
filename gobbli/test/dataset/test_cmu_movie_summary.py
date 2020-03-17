from gobbli.dataset.cmu_movie_summary import MovieSummaryDataset


def test_load_cmu_movie_summary(tmp_gobbli_dir):
    ds = MovieSummaryDataset.load()

    X_train = ds.X_train()
    X_test = ds.X_test()

    y_train = ds.y_train()
    y_test = ds.y_test()

    assert len(X_train) == 33763
    assert len(y_train) == 33763
    assert len(X_test) == 8441
    assert len(y_test) == 8441

    # Ensure these objects pass validation
    train_input = ds.train_input()
    ds.predict_input()

    assert len(train_input.labels()) == 357
