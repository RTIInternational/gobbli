import logging
import os
import tempfile
from pathlib import Path

import pytest

# Reinitialize test cache on disk
TEST_CACHE_DIR = Path(os.getenv("GOBBLI_TEST_CACHE_DIR", Path(__file__).parent))
TEST_CACHE_PATH = TEST_CACHE_DIR / ".test_cache"
TEST_CACHE_PATH.mkdir(exist_ok=True, parents=True)

# Directory for persisted data that can remain across test runs
PERSIST_DIR = TEST_CACHE_PATH / "persisted_data"
PERSIST_DIR.mkdir(exist_ok=True, parents=True)


def pytest_addoption(parser):
    parser.addoption(
        "--persist-data",
        action="store_true",
        help="Persist session-scoped "
        "test data between runs.  This is helpful if you're confident any "
        "dataset generation code is working as expected and doesn't need "
        "to be rerun between test sessions. Persisted data is stored in '"
        f"{PERSIST_DIR}'.  Note this directory will fill up very quickly with "
        "model weights.",
    )
    parser.addoption(
        "--use-gpu",
        action="store_true",
        help="Use a GPU where applicable for running models "
        "in tests.  Limit available GPUs via --nvidia-visible-devices.",
    )
    parser.addoption(
        "--nvidia-visible-devices",
        default="all",
        help="Which GPUs to make available for testing, if applicable. "
        "'all' or a comma-separated string of GPU IDs (ex '1,3').",
    )
    parser.addoption(
        "--worker-log-level",
        default=logging.WARN,
        help="Log level for containers run in worker processes as part of tests. "
        "Combine with --log-cli-level to see logs from workers.",
    )


def make_temp_dir() -> Path:
    # Can't use the pytest fixture because we need this to be in a location
    # known to Docker on OS X
    with tempfile.TemporaryDirectory(dir=TEST_CACHE_PATH) as d:
        old_val = os.getenv("GOBBLI_DIR", "")
        os.environ["GOBBLI_DIR"] = d

        yield Path(d)

        os.environ["GOBBLI_DIR"] = old_val


@pytest.yield_fixture(scope="function")
def tmp_gobbli_dir() -> Path:
    """
    Sets the environment such that the :func:`gobbli.util.gobbli_dir` is a temporary directory that
    won't be shared with any other tests.
    """
    for temp_dir in make_temp_dir():
        yield temp_dir


@pytest.yield_fixture(scope="session")
def gobbli_dir(request) -> Path:
    """
    Sets the environment such that the :func:`gobbli.util.gobbli_dir` is a directory that can be shared
    by other tests in the same session (i.e., use the same dataset as other tests
    to train a model, etc).

    Use a regular tempdir instead of the pytest fixture so it stays the same across the
    session.  Use a persistent directory instead of a tempdir if the user passed the appropriate
    command line option.
    """
    if request.config.getoption("persist_data"):
        old_val = os.getenv("GOBBLI_DIR", "")
        os.environ["GOBBLI_DIR"] = str(PERSIST_DIR)
        yield PERSIST_DIR
        os.environ["GOBBLI_DIR"] = old_val
    else:
        for temp_dir in make_temp_dir():
            yield temp_dir


@pytest.fixture
def model_gpu_config(request):
    """
    Evaluate the test run parameters to build config for models that may or may
    not use the GPU.
    """
    gpu_config = {}
    if request.config.getoption("use_gpu"):
        gpu_config["use_gpu"] = True
        gpu_config["nvidia_visible_devices"] = request.config.getoption(
            "nvidia_visible_devices"
        )

    return gpu_config
