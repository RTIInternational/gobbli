import datetime as dt
import json
import logging
import os
import tempfile
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from umap import UMAP

import gobbli.model
from benchmark_util import (
    PREPROCESS_FUNCS,
    StdoutCatcher,
    assert_param_required,
    assert_proportion,
    assert_valid_augment,
    assert_valid_model,
    format_exception,
    init_benchmark_env,
    maybe_limit,
    run_benchmark_experiment,
)
from gobbli.dataset.base import BaseDataset
from gobbli.dataset.imdb import IMDBDataset
from gobbli.dataset.newsgroups import NewsgroupsDataset
from gobbli.io import (
    EmbedInput,
    PredictOutput,
    WindowPooling,
    make_document_windows,
    pool_document_windows,
)
from gobbli.util import TokenizeMethod, assert_in, assert_type, pred_prob_to_pred_label
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
class ModelClassificationRun(BaseRun):
    """
    Parameters for a benchmark scenario run for a single model in a classification context.
    """

    name: str
    model_name: str
    param_grid: Dict[str, List[Any]]
    preprocess_func: Optional[str] = None
    run_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self):
        return self.name


@dataclass
class ModelEmbeddingRun(BaseRun):
    """
    Parameters for a benchmark scenario run for a single model in an embedding context.
    """

    name: str
    model_name: str
    model_params: Dict[str, Any] = field(default_factory=dict)
    preprocess_func: Optional[str] = None
    batch_size: int = 32

    @property
    def key(self):
        return self.name


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

    Args:
      output_dir: Directory to write combined output for the whole scenario.
        Individual run output should be created in subdirectories.
      params: Parameters for the scenario.  Each scenario implementation should
        validate these as thoroughly as possible at load time to prevent errors
        during time-consuming scenario runs.
      runs: A list of runs (different configurations) to apply in the scenario.
        Each scenario will specify a type of run to use (ex. one run per model type
        or one run per augmentation type).
      force: If True, rerun all runs even if they've already completed with the same
        parameters.  Otherwise, only rerun for which output already exists.
      dataset_limit: If given, limit the size of the individual (train + valid, test)
        datasets used during the scenario to decrease runtime for easier debugging.
        Scenario implementations are expected to respect this limit.
    """

    output_dir: Path
    params: Dict[str, Any]
    runs: Sequence[BaseRun]
    force: bool = False
    dataset_limit: Optional[int] = None

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
            run_error_file = run_output_dir / self.error_filename
            last_run_errored = run_error_file.exists()

            if self.force or last_run_errored or self.should_run(run):
                if last_run_errored:
                    LOGGER.debug(
                        f"Last run errored; removing existing error file at {run_error_file}"
                    )
                    run_error_file.unlink()

                LOGGER.debug(f"Running benchmark for key '{run.key}'")
                init_benchmark_env()

                try:
                    output = self._do_run(run, run_output_dir)
                except Exception as e:
                    if raise_exceptions:
                        raise
                    else:
                        run_error_file.touch()
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
    def error_filename(self) -> str:
        return "ERROR"

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
class ModelClassificationScenario(BaseScenario):  # type: ignore
    runs: Sequence[ModelClassificationRun]

    @abstractmethod
    def _do_run(self, run: ModelClassificationRun, run_output_dir: Path) -> str:
        raise NotImplementedError


@dataclass
class ModelEmbeddingScenario(BaseScenario):  # type: ignore
    runs: Sequence[ModelEmbeddingRun]

    @abstractmethod
    def _do_run(self, run: ModelEmbeddingRun, run_output_dir: Path) -> str:
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
class DatasetClassificationScenario(ModelClassificationScenario):  # type: ignore
    """
    Base class for scenarios which run a model on a gobbli Dataset and evaluate its
    performance in a classification context.
    """

    @property
    @abstractmethod
    def dataset(self) -> BaseDataset:
        raise NotImplementedError

    def _validate_params(self):
        for p in self.params:
            raise ValueError(f"Unexpected parameter: {p}")

    def _do_run(self, run: ModelClassificationRun, run_output_dir: Path) -> str:
        ds = self.dataset.load()
        X_train_valid, y_train_valid, X_test, y_test = maybe_limit(
            ds.X_train(), ds.y_train(), ds.X_test(), ds.y_test(), self.dataset_limit
        )

        assert_in("preprocess_func", run.preprocess_func, PREPROCESS_FUNCS)
        preprocess_func = PREPROCESS_FUNCS[run.preprocess_func]
        X_train_valid_preprocessed = preprocess_func(X_train_valid)
        X_test_preprocessed = preprocess_func(X_test)

        assert_valid_model(run.model_name)
        model_cls = getattr(gobbli.model, run.model_name)

        stdout_catcher = StdoutCatcher()
        with stdout_catcher:
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
            # Sleep a few seconds to let logs from the worker catch up
            time.sleep(3)

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot()
        ax = results.plot(ax=ax)
        plot_path = run_output_dir / "plot.png"
        fig.savefig(plot_path)

        md = f"# Results: {run.key}\n"
        md += f"```\n{stdout_catcher.get_logs()}\n```\n"
        md += tabulate(
            pd.DataFrame(results.training_results), tablefmt="pipe", headers="keys"
        )
        md += f"\n```\n{results.metrics_report()}\n```\n"
        md += f"\n![Results]({self.get_markdown_relative_path(plot_path)})\n---"

        return md


@dataclass
class NewsgroupsClassificationScenario(DatasetClassificationScenario):
    """
    Benchmarking model classification performance on the 20 Newsgroups dataset.
    """

    @property
    def name(self):
        return "newsgroups"

    @property
    def dataset(self):
        return NewsgroupsDataset


@dataclass
class IMDBClassificationScenario(DatasetClassificationScenario):
    """
    Benchmarking model classification performance on the IMDB dataset.
    """

    @property
    def name(self):
        return "imdb"

    @property
    def dataset(self):
        return IMDBDataset


@dataclass
class DatasetEmbeddingScenario(ModelEmbeddingScenario):  # type: ignore
    """
    Base class for scenarios which run a model on a gobbli Dataset and evaluate its
    performance in an embedding context.  The evaluation will be performed qualitatively
    by plotting the embeddings after reducing dimensionality.
    """

    runs: Sequence[ModelEmbeddingRun]

    @property
    @abstractmethod
    def dataset(self) -> BaseDataset:
        raise NotImplementedError

    def _validate_params(self):
        for p in self.params:
            raise ValueError(f"Unexpected parameter: {p}")

    def _do_run(self, run: ModelEmbeddingRun, run_output_dir: Path) -> str:
        ds = self.dataset.load()
        X_train_valid, y_train_valid, X_test, y_test = maybe_limit(
            ds.X_train(), ds.y_train(), ds.X_test(), ds.y_test(), self.dataset_limit
        )
        X_embed = X_train_valid + X_test
        labels = y_train_valid + y_test

        assert_in("preprocess_func", run.preprocess_func, PREPROCESS_FUNCS)
        preprocess_func = PREPROCESS_FUNCS[run.preprocess_func]
        X_embed_preprocessed = preprocess_func(X_embed)

        assert_valid_model(run.model_name)
        model_cls = getattr(gobbli.model, run.model_name)

        stdout_catcher = StdoutCatcher()
        with stdout_catcher:
            # Construct the dict of kwargs up-front so each run can override the "use_gpu"
            # option if necessary -- ex. for models like spaCy which have trouble controlling
            # memory usage on the GPU and don't gain much benefit from it
            model_kwargs = {
                "use_gpu": os.getenv("GOBBLI_USE_GPU") is not None,
                "nvidia_visible_devices": os.getenv("NVIDIA_VISIBLE_DEVICES", ""),
                **run.model_params,
            }
            model = model_cls(**model_kwargs)
            model.build()

            embed_input = EmbedInput(
                X=X_embed_preprocessed, embed_batch_size=run.batch_size
            )
            embed_output = model.embed(embed_input)

        X_embedded = pd.DataFrame(embed_output.X_embedded)
        umap = UMAP(random_state=1)
        umap_data = umap.fit_transform(X_embedded)
        umap_df = pd.DataFrame(
            umap_data, columns=["UMAP Component 1", "UMAP Component 2"]
        )
        umap_df["Label"] = labels
        groups = umap_df.groupby("Label")

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot()
        cmap = plt.cm.get_cmap("tab20")

        for (name, group), c in zip(groups, cmap.colors):
            ax.plot(
                group["UMAP Component 1"],
                group["UMAP Component 2"],
                marker="o",
                linestyle="",
                ms=6,
                label=name,
                color=c,
                alpha=0.5,
            )
        ax.legend()
        ax.axis("off")
        ax.set_title(f"Embeddings by Label - {run.key}")
        plot_path = run_output_dir / "plot.png"
        fig.savefig(plot_path)

        md = f"# Results: {run.key}\n"
        md += f"```\n{stdout_catcher.get_logs()}\n```\n"
        md += f"\n![Results]({self.get_markdown_relative_path(plot_path)})\n---"

        return md


@dataclass
class NewsgroupsEmbeddingScenario(DatasetEmbeddingScenario):
    """
    Benchmarking model embedding performance on the 20 Newsgroups dataset.
    """

    @property
    def name(self):
        return "newsgroups_embed"

    @property
    def dataset(self):
        return NewsgroupsDataset


@dataclass
class IMDBEmbeddingScenario(DatasetEmbeddingScenario):
    """
    Benchmarking model embedding performance on the IMDB dataset.
    """

    @property
    def name(self):
        return "imdb_embed"

    @property
    def dataset(self):
        return IMDBDataset


@dataclass
class ClassImbalanceScenario(ModelClassificationScenario):
    """
    Benchmarking model performance when classes are imbalanced to various degrees.
    """

    @property
    def name(self):
        return "class_imbalance"

    def _validate_params(self):
        assert_param_required("imbalance_proportions", self.params)
        proportions = self.params["imbalance_proportions"]
        assert_type("imbalance_proportions", proportions, list)
        for p in proportions:
            assert_type("imbalance_proportion", p, float)
            assert_proportion("imbalance_proportion", p)

    @staticmethod
    def find_majority_minority_classes(y: List[str]) -> Tuple[str, str]:
        majority, minority = pd.Series(y).value_counts().index.tolist()[:2]
        return majority, minority

    @staticmethod
    def split_dataset(
        X: List[str], y: List[str], majority: str, minority: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame({"X": X, "y": y})

        minority_df = df[df["y"] == minority]

        # Downsample the majority so we start with a 50/50 split
        majority_df = (
            df[df["y"] == majority].sample(n=minority_df.shape[0]).reset_index()
        )
        return majority_df, minority_df

    def _do_run(self, run: ModelClassificationRun, run_output_dir: Path) -> str:
        ds = IMDBDataset.load()
        X_train_valid, y_train_valid, X_test, y_test = maybe_limit(
            ds.X_train(), ds.y_train(), ds.X_test(), ds.y_test(), self.dataset_limit
        )

        assert_in("preprocess_func", run.preprocess_func, PREPROCESS_FUNCS)
        preprocess_func = PREPROCESS_FUNCS[run.preprocess_func]
        X_train_valid_preprocessed = preprocess_func(X_train_valid)
        X_test_preprocessed = preprocess_func(X_test)

        assert_valid_model(run.model_name)
        model_cls = getattr(gobbli.model, run.model_name)

        all_results = []

        majority, minority = ClassImbalanceScenario.find_majority_minority_classes(
            y_test
        )
        majority_df, minority_df = ClassImbalanceScenario.split_dataset(
            X_train_valid_preprocessed, y_train_valid, majority, minority
        )

        for proportion in self.params["imbalance_proportions"]:
            # Downsample the minority class so the final dataset contains the desired
            # proportion of the minority
            orig_len = majority_df.shape[0]
            downsample_proportion = -orig_len / (orig_len - orig_len / proportion)
            minority_sample = minority_df.sample(
                frac=downsample_proportion
            ).reset_index()
            sampled_df = pd.concat([majority_df, minority_sample])

            X = sampled_df["X"].tolist()
            y = sampled_df["y"].tolist()

            LOGGER.info(
                f"{dt.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
                f"Evaluating proportion {round(proportion, 3)} ({len(X)} obs)"
            )

            results = run_benchmark_experiment(
                f"{self.name}_{run.key}",
                X,
                y,
                model_cls,
                run.param_grid,
                test_dataset=(X_test_preprocessed, y_test),
                run_kwargs=run.run_kwargs,
            )
            all_results.append(results)

        minority_f1_scores = []
        majority_f1_scores = []
        for result in all_results:
            majority_f1, minority_f1 = f1_score(
                result.y_true,
                pred_prob_to_pred_label(result.y_pred_proba),
                average=None,
                labels=[majority, minority],
            )
            minority_f1_scores.append(minority_f1)
            majority_f1_scores.append(majority_f1)

        all_metrics = pd.DataFrame(
            [
                {"imbalance_proportion": p, **r.metrics()}
                for p, r in zip(self.params["imbalance_proportions"], all_results)
            ]
        )

        all_metrics["Minority Class F1 Score"] = minority_f1_scores
        all_metrics["Majority Class F1 Score"] = majority_f1_scores

        fig = plt.figure(figsize=(10, 10))
        minority_ax = fig.add_subplot()
        all_metrics.plot(
            x="imbalance_proportion", y="Minority Class F1 Score", ax=minority_ax
        )

        majority_ax = fig.add_subplot()
        all_metrics.plot(
            x="imbalance_proportion", y="Majority Class F1 Score", ax=majority_ax
        )

        plt.xlabel("Prevalence of Minority Class")
        plt.title(
            f"Model Performance by Prevalence of Minority Class - {model_cls.__name__}"
        )
        plt.xlim(0, 0.5)
        plt.ylim(0, 1)

        plot_path = run_output_dir / "plot.png"
        fig.savefig(plot_path)

        md = f"# Results: {run.key}\n"
        md += tabulate(all_metrics, tablefmt="pipe", headers="keys")
        md += f"\n![Results]({self.get_markdown_relative_path(plot_path)})\n---"

        return md


class LowResourceScenario(ModelClassificationScenario):
    """
    Benchmarking model performance when only a small amount of data is available for
    training and evaluation.
    """

    @property
    def name(self):
        return "low_resource"

    def _validate_params(self):
        assert_param_required("data_proportions", self.params)
        proportions = self.params["data_proportions"]
        assert_type("data_proportions", proportions, list)
        for p in proportions:
            assert_type("data_proportion", p, float)
            assert_proportion("data_proportion", p)

    def _do_run(self, run: ModelClassificationRun, run_output_dir: Path) -> str:
        ds = IMDBDataset.load()
        X_train_valid, y_train_valid, X_test, y_test = maybe_limit(
            ds.X_train(), ds.y_train(), ds.X_test(), ds.y_test(), self.dataset_limit
        )

        assert_in("preprocess_func", run.preprocess_func, PREPROCESS_FUNCS)
        preprocess_func = PREPROCESS_FUNCS[run.preprocess_func]
        X_train_valid_preprocessed = preprocess_func(X_train_valid)
        X_test_preprocessed = preprocess_func(X_test)

        assert_valid_model(run.model_name)
        model_cls = getattr(gobbli.model, run.model_name)

        all_results = []

        # Finish linting, test
        for proportion in self.params["data_proportions"]:
            X_sampled, _, y_sampled, _ = train_test_split(
                X_train_valid_preprocessed,
                y_train_valid,
                train_size=proportion,
                random_state=1,
            )
            LOGGER.info(
                f"{dt.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
                f"Evaluating proportion {round(proportion, 3)} ({len(X_sampled)} obs)"
            )
            results = run_benchmark_experiment(
                f"{self.name}_{run.key}",
                X_sampled,
                y_sampled,
                model_cls,
                run.param_grid,
                test_dataset=(X_test_preprocessed, y_test),
                run_kwargs=run.run_kwargs,
            )
            all_results.append(results)

        all_metrics = pd.DataFrame(
            [
                {
                    "data_proportion": p,
                    "num_documents": int(p * len(X_train_valid)),
                    **r.metrics(),
                }
                for p, r in zip(self.params["data_proportions"], all_results)
            ]
        )

        fig = plt.figure(figsize=(10, 10))
        f1_ax = fig.add_subplot()
        all_metrics.plot(x="num_documents", y="Weighted F1 Score", ax=f1_ax)

        acc_ax = fig.add_subplot()
        all_metrics.plot(x="num_documents", y="Accuracy", ax=acc_ax)

        plt.xlabel("Number of Documents Used for Training/Validation")
        plt.title(
            f"Model Performance by Number of Documents Used for Training/Validation - {model_cls.__name__}"
        )
        plt.xlim(0, int(all_metrics["num_documents"].max() * 1.1))
        plt.ylim(0, 1)
        plot_path = run_output_dir / "plot.png"
        fig.savefig(plot_path)

        md = f"# Results: {run.key}\n"
        md += tabulate(all_metrics, tablefmt="pipe", headers="keys")
        md += f"\n![Results]({self.get_markdown_relative_path(plot_path)})\n---"

        return md


class DataAugmentationScenario(AugmentScenario):
    @property
    def name(self):
        return "data_augmentation"

    def _validate_params(self):
        assert_param_required("percent_multipliers", self.params)
        percent_multipliers = self.params["percent_multipliers"]
        assert_type("percent_multipliers", percent_multipliers, list)
        for (p, m) in percent_multipliers:
            assert_type("percent", p, float)
            assert_proportion("percent", p)
            assert_type("multiplier", m, (int, float))

        assert_type("param_grid", self.params.get("param_grid", {}), dict)

        assert_param_required("model_name", self.params)
        assert_type("model_name", self.params["model_name"], str)
        assert_valid_model(self.params["model_name"])

        assert_param_required("augment_probability", self.params)
        assert_type("augment_probability", p, float)
        assert_proportion("augment_probability", p)

        assert_param_required("preprocess_func", self.params)
        assert_in("preprocess_func", self.params["preprocess_func"], PREPROCESS_FUNCS)

    def _do_run(self, run: AugmentRun, run_output_dir: Path) -> str:
        ds = IMDBDataset.load()
        X_train_valid, y_train_valid, X_test, y_test = maybe_limit(
            ds.X_train(), ds.y_train(), ds.X_test(), ds.y_test(), self.dataset_limit
        )

        preprocess_func = PREPROCESS_FUNCS[self.params["preprocess_func"]]
        X_test_preprocessed = preprocess_func(X_test)

        model_cls = getattr(gobbli.model, self.params["model_name"])

        assert_valid_augment(run.augment_name)
        augment_cls = getattr(gobbli.augment, run.augment_name)
        augment_obj = augment_cls(**run.params)

        all_results = []

        for percent, multiplier in self.params["percent_multipliers"]:

            X_sampled, _, y_sampled, _ = train_test_split(
                X_train_valid, y_train_valid, train_size=percent, random_state=1
            )

            if multiplier == 0:
                X_augmented = X_sampled
                y_augmented = y_sampled
            else:
                X_augmented = X_sampled + augment_obj.augment(
                    X_sampled, times=multiplier, p=self.params["augment_probability"]
                )
                y_augmented = y_sampled + (y_sampled * multiplier)

            print(
                f"{dt.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
                f"Evaluating multiplier x{multiplier}, percent {percent} ({len(X_augmented)} obs)"
            )
            results = run_benchmark_experiment(
                f"{self.name}_{run.key}",
                preprocess_func(X_augmented),
                y_augmented,
                model_cls,
                self.params["param_grid"],
                test_dataset=(X_test_preprocessed, y_test),
            )
            all_results.append(results.metrics())

        all_metrics = pd.DataFrame(
            [
                {"percent": p, "multiplier": m, **r}
                for (p, m), r in zip(self.params["percent_multipliers"], all_results)
            ]
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        for key, grp in all_metrics.groupby("multiplier"):
            grp.plot(
                x="percent",
                y="Weighted F1 Score",
                kind="line",
                label=f"{key}x augmentation",
                ax=ax,
            )

        plt.xlabel("Proportion of Data Used")
        plt.ylabel("Weighted F1 Score")
        plt.title(
            f"Model Performance by Proportion of Data Used - {model_cls.__name__}"
        )
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plot_path = run_output_dir / "plot.png"
        fig.savefig(plot_path)

        md = f"# Results: {run.key}\n"
        md += tabulate(all_metrics, tablefmt="pipe", headers="keys")
        md += f"\n![Results]({self.get_markdown_relative_path(plot_path)})\n---"

        return md


class DocumentWindowingScenario(ModelClassificationScenario):
    @property
    def name(self):
        return "data_augmentation"

    def _validate_params(self):
        assert_param_required("vocab_size", self.params)
        assert_type("vocab_size", self.params["vocab_size"], int)

        assert_param_required("sample_size", self.params)
        assert_type("sample_size", self.params["sample_size"], float)

        assert_param_required("window_len_poolings", self.params)
        window_len_poolings = self.params["window_len_poolings"]
        assert_type("window_len_poolings", window_len_poolings, list)
        for w, p in window_len_poolings:
            assert_type("window_len", w, int)
            assert_type("pooling", p, str)
            # This raises an exception if p isn't a valid pooling method
            WindowPooling(p)

    def _do_run(self, run: ModelClassificationRun, run_output_dir: Path) -> str:
        ds = IMDBDataset.load()
        X_train_valid, y_train_valid, X_test, y_test = maybe_limit(
            ds.X_train(), ds.y_train(), ds.X_test(), ds.y_test(), self.dataset_limit
        )

        assert_in("preprocess_func", run.preprocess_func, PREPROCESS_FUNCS)
        preprocess_func = PREPROCESS_FUNCS[run.preprocess_func]
        X_train_valid_preprocessed = preprocess_func(X_train_valid)
        X_test_preprocessed = preprocess_func(X_test)

        assert_valid_model(run.model_name)
        model_cls = getattr(gobbli.model, run.model_name)

        all_results = []

        for window_len, pooling in self.params["window_len_poolings"]:

            if window_len is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tokenizer_path = Path(tmpdir) / "tokenizer"

                    X_windowed, _, y_windowed = make_document_windows(
                        X_train_valid_preprocessed,
                        window_len=window_len,
                        y=y_train_valid,
                        tokenize_method=TokenizeMethod.SENTENCEPIECE,
                        vocab_size=self.params["vocab_size"],
                        model_path=tokenizer_path,
                    )
                    (
                        X_test_windowed,
                        X_test_windowed_indices,
                        y_test_windowed,
                    ) = make_document_windows(
                        X_test_preprocessed,
                        window_len=window_len,
                        y=y_test,
                        tokenize_method=TokenizeMethod.SENTENCEPIECE,
                        vocab_size=self.params["vocab_size"],
                        model_path=tokenizer_path,
                    )
            else:
                X_windowed, y_windowed = X_train_valid_preprocessed, y_train_valid
                X_test_windowed, y_test_windowed = X_test_preprocessed, y_test

            print(
                f"{dt.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
                f"Evaluating window: Length {window_len}, pooling {pooling} ({len(X_windowed)} obs)"
            )
            results = run_benchmark_experiment(
                f"{self.name}_{run.key}",
                X_windowed,
                y_windowed,
                model_cls,
                run.param_grid,
                test_dataset=(X_test_windowed, y_test_windowed),
                run_kwargs=run.run_kwargs,
            )

            if window_len is not None:
                pooled_output = PredictOutput(y_pred_proba=results.y_pred_proba.copy())

                pool_document_windows(
                    pooled_output,
                    X_test_windowed_indices,
                    pooling=WindowPooling(pooling),
                )

            all_results.append(results.metrics())

        all_metrics = pd.DataFrame(
            [
                {"Window Config": f"Length {window_len}, pooling {pooling}", **r}
                for w, r in zip(self.params["window_len_poolings"], all_results)
            ]
        )

        fig = plt.figure(figsize=(10, 10))

        acc_ax = fig.add_subplot()
        all_metrics.plot(x="Window Config", y="Accuracy", ax=acc_ax, kind="bar")

        plt.xlabel("Document Windowing")
        plt.title(f"Model Performance by Document Windowing - {model_cls.__name__}")
        plt.ylim(0, 1)

        plot_path = run_output_dir / "plot.png"
        fig.savefig(plot_path)

        md = f"# Results: {run.key}\n"
        md += tabulate(all_metrics, tablefmt="pipe", headers="keys")
        md += f"\n![Results]({self.get_markdown_relative_path(plot_path)})\n---"

        return md


def load_scenario(
    scenario_cls: Any,
    output_dir: Path,
    params: Dict[str, Any],
    run_dicts: List[Dict[str, Any]],
    force: bool = False,
    dataset_limit: Optional[int] = None,
) -> BaseScenario:
    if issubclass(scenario_cls, ModelClassificationScenario):
        run_cls: Any = ModelClassificationRun
    elif issubclass(scenario_cls, ModelEmbeddingScenario):
        run_cls = ModelEmbeddingRun
    elif issubclass(scenario_cls, AugmentScenario):
        run_cls = AugmentRun
    else:
        raise TypeError(f"Unsupported scenario class: {scenario_cls}")

    runs = [run_cls(**d) for d in run_dicts]

    return scenario_cls(
        output_dir=output_dir,
        params=params,
        runs=runs,
        force=force,
        dataset_limit=dataset_limit,
    )
