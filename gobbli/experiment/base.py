import logging
import os
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ray

from gobbli.dataset.base import BaseDataset
from gobbli.model.sklearn import SKLearnClassifier
from gobbli.util import generate_uuid, gobbli_dir, is_dir_empty, write_metadata


def experiment_dir() -> Path:
    return gobbli_dir() / "experiment"


def init_worker_env(
    gobbli_dir: Optional[Path] = None, log_level: Union[int, str] = logging.WARNING
) -> logging.Logger:
    """
    Initialize environment on a ray worker.

    Args:
        gobbli_dir: Used as the value of the GOBBLI_DIR environment variable;
        determines where gobbli data is stored on the worker's filesystem.
        log_level: Level for logging coming from the worker.
    """
    if gobbli_dir is not None:
        os.environ["GOBBLI_DIR"] = str(gobbli_dir)

    logging.basicConfig(level=log_level)

    # Provide a logger for our workers
    # Workers should only log using loggers they've created to avoid
    # logger pickling, which generally doesn't work
    # https://stackoverflow.com/questions/55272066/how-can-i-use-the-python-logging-in-ray
    return logging.getLogger(__name__)


def init_gpu_config() -> Tuple[bool, str]:
    """
    Determine the GPU configuration from the current ray environment
    on a worker.

    Returns:
      2-tuple: whether GPU should be used and a comma-separated string
      containing the ids of the GPUs that should be used
    """
    try:
        gpu_ids = ray.get_gpu_ids()
    except Exception as e:
        # This message is either 'ray.get_gpu_ids() currently does not work in PYTHON MODE'
        # or '... in LOCAL MODE' depending on the version of ray installed
        if "ray.get_gpu_ids() currently does not work in" in str(e):
            gpu_ids = []
        else:
            raise
    use_gpu = len(gpu_ids) > 0
    nvidia_visible_devices = ",".join(str(i) for i in gpu_ids)
    return use_gpu, nvidia_visible_devices


def get_worker_ip() -> str:
    """
    Determine the IP address of the current ray worker.
    Returns:
      A string containing the IP address.
    """
    global_worker = ray.worker.global_worker
    return getattr(global_worker, "node_ip_address", "127.0.0.1")


class BaseExperiment(ABC):
    """
    Base class for all derived Experiments.
    """

    _METADATA_FILENAME = "gobbli-experiment-meta.json"

    def __init__(
        self,
        model_cls: Any,
        dataset: Union[Tuple[List[str], List[str]], BaseDataset],
        test_dataset: Optional[Tuple[List[str], List[str]]] = None,
        data_dir: Optional[Path] = None,
        name: Optional[str] = None,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        task_num_cpus: int = 1,
        task_num_gpus: int = 0,
        worker_gobbli_dir: Optional[Path] = None,
        worker_log_level: Union[int, str] = logging.WARNING,
        limit: Optional[int] = None,
        overwrite_existing: bool = False,
        ignore_ray_initialized_error: bool = False,
        distributed: bool = False,
        ray_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Construct an experiment.

        Args:
          model_cls: The class of model to be used for the experiment.
          dataset: Dataset to be used for the experiment.  Can be either a 2-tuple
            containing a list of texts and a corresponding list of labels
            or a :class:`gobbli.dataset.base.BaseDataset`.
          test_dataset: An optional separate dataset to be used for calculating test metrics.
            If passed, should be a 2-tuple containing a list of texts and corresponding list
            of labels.  If not passed, a test dataset will be automatically split out of
            the `dataset`.
          data_dir: Optional path to a directory used to store data for the experiment.
            If not given, a directory under GOBBLI_DIR will be created and used.
          name: A descriptive name for the experiment, used to label directories
            in the filesystem.  If not passed, a random name will be generated and used.
            The name must be unique (i.e., there should not be another experiment with the
            same name).
          param_grid: Optional grid of parameters.  If passed, it should be a dictionary
            with keys being valid parameter names for the passed model and values being lists
            of parameter values.  Every combination of parameter values will be tried in the
            experiment, and the results for the best combination will be returned.  If not passed,
            only the model's default parameters will be used.
          task_num_cpus: Number of CPUs to reserve per task.
          task_num_gpus: Number of GPUs to reserve per task.
          worker_gobbli_dir: Directory to use for gobbli file storage by workers.
          worker_log_level: Logging level to use for logs output by workers running
            training tasks.
          limit: Read up to this many rows from the passed dataset.  Useful for debugging.
          overwrite_existing: If True, don't fail if there's an existing experiment in
            the same directory.
          ignore_ray_initialized_error: If True, don't error when a ray connection is already
            initialized; instead, shut it down and restart it with the passed `ray_kwargs`.
          distributed: If True, run the ray cluster assuming workers are distributed over
            multiple nodes.  This requires model weights for all trials to fit in the ray
            object store, which requires a lot of memory.  If False, run the ray cluster
            assuming all workers are on the master node, and weights will be passed around as
            filepaths; an error will be thrown if a remote worker tries to run a task.
          ray_kwargs: Dictionary containing keyword arguments to be passed directly to
            :func:`ray.init`.  By default, a new ray cluster will be initialized on the current
            node using all available CPUs and no GPUs, but these arguments can be used to connect
            to a remote cluster, limit resource usage, and much more.
        """
        self.model_cls = model_cls
        self.worker_gobbli_dir = worker_gobbli_dir

        self.name = name
        if self.name is None:
            self.name = generate_uuid()

        if data_dir is None:
            self._data_dir = experiment_dir() / self.__class__.__name__ / self.name
        else:
            self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite_existing and not is_dir_empty(self._data_dir):
            raise ValueError(f"Experiment already exists for name '{self.name}'")

        if isinstance(dataset, BaseDataset):
            self.X = dataset.X_train() + dataset.X_test()
            self.y = dataset.y_train() + dataset.y_test()
        elif isinstance(dataset, tuple):
            if len(dataset) != 2:
                raise ValueError(
                    f"`dataset` must be a 2-tuple, got length {len(dataset)}"
                )
            self.X, self.y = dataset
        else:
            raise TypeError(f"Invalid type for dataset: {type(dataset)}")

        self.X_test = None  # type: Optional[List[str]]
        self.y_test = None  # type: Optional[List[str]]
        if test_dataset is not None:
            if not (isinstance(dataset, tuple) and len(dataset) == 2):
                raise ValueError(f"`test_dataset` must be a 2-tuple")
            self.X_test, self.y_test = test_dataset

        if limit is not None:
            self.X = self.X[:limit]
            self.y = self.y[:limit]

        self.param_grid = param_grid
        if param_grid is None:
            self.param_grid = {}

        self.task_num_cpus = task_num_cpus
        self.task_num_gpus = task_num_gpus
        self.worker_log_level = worker_log_level
        self.distributed = distributed

        if self.model_cls is SKLearnClassifier and distributed:
            raise ValueError(
                "The scikit-learn classifier is not supported for distributed "
                "experiments, since it needs to load a pickle from a file path "
                "which may not be on a given worker node."
            )

        _ray_kwargs = ray_kwargs
        if _ray_kwargs is None:
            _ray_kwargs = {}

        self.is_ray_local_mode = _ray_kwargs.get("local_mode", False)

        # We may have an existing ray connection active -- throw an error or
        # clear it out to ensure it's re-initialized with the passed params
        if ray.is_initialized():
            if ignore_ray_initialized_error:
                ray.shutdown()
            else:
                raise RuntimeError(
                    "A ray connection is already initialized. To ignore this error"
                    " and shut down the existing connection, pass"
                    " `ignore_ray_initialized_error=True`."
                )

        ray.init(**_ray_kwargs)

        metadata = {
            "model": model_cls.__name__,
            "len_X": len(self.X),
            "len_y": len(self.y),
            "param_grid": self.param_grid,
        }

        write_metadata(metadata, self.metadata_path)

    @property
    def metadata_path(self) -> Path:
        """
        Returns:
          The path to the experiment's metadata file containing information about the
          experiment parameters.
        """
        return self.data_dir() / BaseExperiment._METADATA_FILENAME

    def data_dir(self) -> Path:
        """
        Returns:
          The main data directory unique to this experiment.
        """
        return self._data_dir
