from pathlib import Path

import pytest

from gobbli.test.util import MockModel


def test_base_model_init(tmpdir):
    tmpdir_path = Path(tmpdir)
    MODEL_PARAMS = {"param": "a"}

    # Create model
    m = MockModel(data_dir=tmpdir_path, **MODEL_PARAMS)

    # Metadata path for the model should now exist
    assert m.metadata_path.exists()

    # We shouldn't be able to create a new model in the same directory
    # with load_existing=False
    with pytest.raises(ValueError):
        MockModel(data_dir=tmpdir_path, load_existing=False, **MODEL_PARAMS)

    # We should be able to load the existing model, and it should
    # have the same param values without being passed explicitly
    m2 = MockModel(data_dir=tmpdir_path, load_existing=True)
    assert m2.params == MODEL_PARAMS
