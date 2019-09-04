from pathlib import Path
from typing import Any

import pytest

from gobbli.dataset.base import BaseDataset
from gobbli.experiment.base import BaseExperiment
from gobbli.model.base import BaseModel
from gobbli.model.bert import BERT
from gobbli.model.fasttext import FastText, FastTextCheckpoint
from gobbli.model.mtdnn import MTDNN
from gobbli.model.transformer import Transformer
from gobbli.util import gobbli_dir


def skip_if_no_gpu(config):
    if not config.option.use_gpu:
        pytest.skip("needs --use-gpu option to run")


def skip_if_low_resource(config):
    """
    Used when a test involves a large amount of CPU, memory, etc,
    and the user has indicated we're running in a resource-limited
    environment.
    """
    if config.option.low_resource:
        pytest.skip("skipping large test due to --low-resource option")


# TODO can we write a type declaration to indicate that args
# should be classes derived from BaseModel?
def model_test_dir(model_cls: Any) -> Path:
    """
    Return a directory to be used for models of the passed type.
    Helpful when the user wants data to be persisted so weights don't
    have to be reloaded for each test run.
    """
    return gobbli_dir() / "model_test" / model_cls.__name__


def validate_checkpoint(model_cls: Any, checkpoint: Path):
    """
    Use assertions to validate a given checkpoint depending on which kind of
    model created it.
    """
    if model_cls == BERT:
        # The checkpoint isn't an actual file, but it should have an associated metadata file
        assert Path(f"{str(checkpoint)}.meta").is_file()
    elif model_cls == MTDNN:
        # The checkpoint is a single file
        assert checkpoint.is_file()
    elif model_cls == FastText:
        # The checkpoint has a couple components
        fasttext_checkpoint = FastTextCheckpoint(checkpoint)
        assert fasttext_checkpoint.model.exists()
        assert fasttext_checkpoint.vectors.exists()
    elif model_cls == Transformer:
        assert checkpoint.is_dir()


class MockDataset(BaseDataset):
    """
    A minimal dataset derived from BaseDataset for testing the
    ABC's logic.
    """

    X_TRAIN_VALID = ["train1", "train2", "train3", "train4"]
    Y_TRAIN_VALID = ["0", "1", "0", "1"]

    X_TEST = ["test1", "test2"]
    Y_TEST = ["1", "0"]

    def __init__(self, *args, **kwargs):
        self._built = False
        self._build_count = 0

    def _is_built(self) -> bool:
        return self._built

    def _build(self):
        self._build_count += 1
        self._built = True

    def X_train(self):
        return MockDataset.X_TRAIN_VALID

    def y_train(self):
        return MockDataset.Y_TRAIN_VALID

    def X_test(self):
        return MockDataset.X_TEST

    def y_test(self):
        return MockDataset.Y_TEST


class MockModel(BaseModel):
    """
    A minimal model derived from BaseModel for testing the
    ABC's logic.
    """

    def init(self, params):
        self.params = params

    def _build(self):
        pass


class MockExperiment(BaseExperiment):
    """
    A minimal experiment derived from BaseExperiment for testing the
    ABC's logic.
    """
