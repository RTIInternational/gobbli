from pathlib import Path

import pytest
import ray

from gobbli.test.util import MockDataset, MockExperiment, MockModel, skip_if_no_gpu


def test_base_experiment_init(tmpdir):
    tmpdir_path = Path(tmpdir)
    ds = MockDataset.load()

    # Create experiment
    exp = MockExperiment(MockModel, ds, data_dir=tmpdir_path / "test")
    assert exp.metadata_path.exists()

    # Shouldn't be able to create another experiment without ignoring the ray initialization error
    with pytest.raises(RuntimeError):
        MockExperiment(MockModel, ds, data_dir=tmpdir_path / "test2")
    MockExperiment(
        MockModel, ds, data_dir=tmpdir_path / "test3", ignore_ray_initialized_error=True
    )

    # Shouldn't be able to create another experiment in the same path
    with pytest.raises(ValueError):
        MockExperiment(
            MockModel,
            ds,
            data_dir=tmpdir_path / "test",
            ignore_ray_initialized_error=True,
        )

    # ...unless we pass overwrite_existing = True
    MockExperiment(
        MockModel,
        ds,
        data_dir=tmpdir_path / "test",
        ignore_ray_initialized_error=True,
        overwrite_existing=True,
    )

    # Limit should be obeyed
    assert len(exp.X) == len(MockDataset.X_TRAIN_VALID) + len(MockDataset.X_TEST)
    assert len(exp.y) == len(MockDataset.Y_TRAIN_VALID) + len(MockDataset.Y_TEST)

    exp_limit = MockExperiment(
        MockModel,
        ds,
        limit=1,
        data_dir=tmpdir_path / "test_limit",
        ignore_ray_initialized_error=True,
    )
    assert len(exp_limit.X) == 1
    assert len(exp_limit.y) == 1


def test_base_experiment_gpu(tmpdir, request):
    skip_if_no_gpu(request.config)

    tmpdir_path = Path(tmpdir)
    ds = MockDataset.load()

    MockExperiment(
        MockModel,
        ds,
        data_dir=tmpdir_path / "test",
        ray_kwargs={"num_gpus": 1},
        ignore_ray_initialized_error=True,
    )

    # Make sure GPUs are available
    # in a mock remote function
    # They won't necessarily be available on the master process
    @ray.remote(num_gpus=1)
    def find_gpus():
        return ray.get_gpu_ids()

    assert len(ray.get(find_gpus.remote())) > 0
