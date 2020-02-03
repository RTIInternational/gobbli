import json
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
from matplotlib import pyplot as plt

import gobbli.model
from benchmark_util import (
    PREPROCESS_FUNCS,
    format_exception,
    init_benchmark_env,
    run_benchmark_experiment,
)
from gobbli.dataset.newsgroups import NewsgroupsDataset
from tabulate import tabulate

LOGGER = logging.getLogger(__name__)


class BaseRun(ABC):
    """
    Base class for a single run within a benchmark scenario.
    """

    @property
    @abstractmethod
    def key(self):
        raise NotImplementedError


@dataclass
class ModelRun(BaseRun):
    """
    Parameters for a benchmark scenario run for a single model.
    """

    model_name: str
    param_grid: Dict[str, List[Any]]
    preprocess_func: str
    run_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self):
        return self.model_name


@dataclass
class AugmentRun(BaseRun):
    """
    Parameters for a benchmark scenario run for a single augmentation method.
    """

    augment_name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self):
        return self.augment_name


# ABCs and dataclasses can't be mixed until this is fixed:
# https://github.com/python/mypy/issues/5374
@dataclass
class BaseScenario(ABC):  # type: ignore
    """
    Base class for benchmark scenarios.
    """

    output_dir: Path
    params: Dict[str, Any]
    runs: Sequence[BaseRun]
    force: bool = False

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self._validate_params()
        self._gather_existing_runs()

    def _gather_existing_runs(self):
        self.existing_runs: Dict[str, Dict[str, Any]] = {}
        for run_dir in self.output_dir.iterdir():
            key = run_dir.name
            for meta_file in run_dir.glob(self.metadata_filename):
                try:
                    with open(meta_file, "r") as f:
                        self.existing_runs[key] = json.load(f)
                        LOGGER.debug(f"Found existing benchmark run for key '{key}'")
                except Exception:
                    warnings.warn(f"Failed to load run metadata from file {meta_file}.")

    def run(self, raise_exceptions: bool = False):
        all_outputs = []
        for run in self.runs:
            run_output_dir = self.output_dir / run.key
            run_output_dir.mkdir(exist_ok=True, parents=True)
            run_output_file = run_output_dir / self.output_filename

            if self.force or self.should_run(run):
                LOGGER.debug(f"Running benchmark for key '{run.key}'")
                init_benchmark_env()

                try:
                    output = self._do_run(run, run_output_dir)
                except Exception as e:
                    if raise_exceptions:
                        raise
                    else:
                        err_msg = (
                            f"ERROR: Exception during run '{run.key}'.\n\n"
                            f"```\n{format_exception(e)}\n```"
                        )
                        LOGGER.warn(err_msg)
                        output = f"# {err_msg}"

                with open(run_output_dir / self.metadata_filename, "w") as f:
                    json.dump(self.to_metadata(run), f)
                run_output_file.write_text(output)
            else:
                LOGGER.debug(f"Using existing output for benchmark '{run.key}'")
                output = run_output_file.read_text()

            all_outputs.append(output)

        (self.output_dir / self.all_output_filename).write_text("\n".join(all_outputs))

    def to_metadata(self, run: BaseRun) -> Dict[str, Any]:
        return {**self.params, **asdict(run)}

    def should_run(self, run: BaseRun) -> bool:
        return (
            run.key not in self.existing_runs
            or self.to_metadata(run) != self.existing_runs[run.key]
        )

    def get_markdown_relative_path(self, p: Path) -> Path:
        """
        Args:
          p: A path which may or may not be relative.

        Returns:
          A path relative to the output dir, suitable for embedding links/images/etc
          in the Markdown output.
        """
        return p.relative_to(self.output_dir)

    @property
    def metadata_filename(self) -> str:
        return "run-meta.json"

    @property
    def output_filename(self) -> str:
        return "output.md"

    @property
    def all_output_filename(self) -> str:
        return f"{self.name}.md"

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def _validate_params(self):
        raise NotImplementedError

    @abstractmethod
    def _do_run(self, run: Any, run_output_dir: Path) -> str:
        raise NotImplementedError


@dataclass
class ModelScenario(BaseScenario):  # type: ignore
    """
    Base class for scenarios whose primary purpose is to benchmark a model.
    """

    runs: Sequence[ModelRun]

    @abstractmethod
    def _do_run(self, run: ModelRun, run_output_dir: Path) -> str:
        raise NotImplementedError


@dataclass
class AugmentScenario(BaseScenario):  # type: ignore
    """
    Base class for scenarios whose primary purpose is to benchmark a data augmentation method.
    """

    runs: Sequence[AugmentRun]

    @abstractmethod
    def _do_run(self, run: AugmentRun, run_output_dir: Path) -> str:
        raise NotImplementedError


@dataclass
class NewsgroupsScenario(ModelScenario):
    """
    Benchmarking model performance on the 20 Newsgroups dataset.
    """

    @property
    def name(self):
        return "newsgroups"

    def _validate_params(self):
        for p in self.params:
            raise ValueError(f"Unexpected parameter: {p}")

    def _do_run(self, run: ModelRun, run_output_dir: Path) -> str:
        ds = NewsgroupsDataset.load()
        X_train_valid, X_test = ds.X_train(), ds.X_test()
        y_train_valid, y_test = ds.y_train(), ds.y_test()

        preprocess_func = PREPROCESS_FUNCS[run.preprocess_func]
        X_train_valid_preprocessed = preprocess_func(X_train_valid)
        X_test_preprocessed = preprocess_func(X_test)

        model_cls = getattr(gobbli.model, run.model_name)

        results = run_benchmark_experiment(
            f"{self.name}_{run.key}",
            X_train_valid_preprocessed,
            y_train_valid,
            model_cls,
            run.param_grid,
            test_dataset=(X_test_preprocessed, y_test),
            worker_log_level=logging.INFO,
            run_kwargs=run.run_kwargs,
        )

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot()
        ax = results.plot(ax=ax)
        plot_path = run_output_dir / "plot.png"
        fig.savefig(plot_path)

        md = f"# Results: {run.key}\n"
        md += tabulate(
            pd.DataFrame(results.training_results), tablefmt="pipe", headers="keys"
        )
        md += f"\n```\n{results.metrics_report()}\n```\n"
        md += f"\n![Results]({self.get_markdown_relative_path(plot_path)})\n---"

        return md


def load_scenario(
    scenario_cls: Any,
    output_dir: Path,
    params: Dict[str, Any],
    run_dicts: List[Dict[str, Any]],
    force: bool = False,
) -> BaseScenario:
    if issubclass(scenario_cls, ModelScenario):
        run_cls: Any = ModelRun
    elif issubclass(scenario_cls, AugmentScenario):
        run_cls = AugmentRun
    else:
        raise TypeError(f"Unsupported scenario class: {scenario_cls}")

    runs = [run_cls(**d) for d in run_dicts]

    return scenario_cls(output_dir=output_dir, params=params, runs=runs, force=force)
