from gobbli.dataset.base import BaseDataset


class TrivialDataset(BaseDataset):
    """
    gobbli Dataset containing only a few observations.
    Useful for verifying a model runs without waiting for an
    actual dataset to process.
    """

    DATASET = ["This is positive.", "This, although, is negative."]
    LABELS = ["1", "0"]

    def _is_built(self) -> bool:
        return True

    def _build(self):
        pass

    def X_train(self):
        return TrivialDataset.DATASET

    def y_train(self):
        return TrivialDataset.LABELS

    def X_test(self):
        return TrivialDataset.DATASET

    def y_test(self):
        return TrivialDataset.LABELS
