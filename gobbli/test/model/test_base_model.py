from pathlib import Path

import pytest

from gobbli.test.util import MockModel
from gobbli.util import gobbli_version, read_metadata, write_metadata


def test_base_model_init(tmpdir):
    tmpdir_path = Path(tmpdir)
    MODEL_PARAMS = {"param": "a"}

    # Create model
    m = MockModel(data_dir=tmpdir_path, **MODEL_PARAMS)

    # Metadata path for the model should now exist
    assert m.metadata_path.exists()

    # So should the info path
    assert m.info_path.exists()

    # Info should be populated appropriately
    info = read_metadata(m.info_path)
    assert info["class"] == m.__class__.__name__
    assert info["gobbli_version"] == gobbli_version()

    # We shouldn't be able to create a new model in the same directory
    # with load_existing=False
    with pytest.raises(ValueError):
        MockModel(data_dir=tmpdir_path, load_existing=False, **MODEL_PARAMS)

    # We should be able to load the existing model, and it should
    # have the same param values without being passed explicitly
    m2 = MockModel(data_dir=tmpdir_path, load_existing=True)
    assert m2.params == MODEL_PARAMS


def test_base_model_init_warnings_errors(tmpdir):
    tmpdir_path = Path(tmpdir)

    # Create model to bootstrap the metadata/info files
    m = MockModel(data_dir=tmpdir_path, param="a")

    # Make sure we get a warning if user passes params that are ignored
    with pytest.warns(UserWarning):
        MockModel(data_dir=tmpdir_path, load_existing=True, param="b")

    # Make sure we get a warning if the gobbli version in the info file changes
    info = read_metadata(m.info_path)
    bad_version_info = info.copy()
    bad_version_info["gobbli_version"] = "not a real version"
    write_metadata(bad_version_info, m.info_path)
    with pytest.warns(UserWarning):
        MockModel(data_dir=tmpdir_path, load_existing=True)

    # Make sure we get an error if the class in the info file changes
    bad_class_info = info.copy()
    bad_class_info["class"] = "not a real model"
    write_metadata(bad_class_info, m.info_path)
    with pytest.raises(ValueError):
        MockModel(data_dir=tmpdir_path, load_existing=True)
