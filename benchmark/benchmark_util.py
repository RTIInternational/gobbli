import logging
import multiprocessing
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from spacy.lang.en import English

import gobbli.model
from gobbli.experiment.classification import (
    ClassificationExperiment,
    ClassificationExperimentResults,
)
from gobbli.model.fasttext import FastText

LOGGER = logging.getLogger(__name__)


def format_exception(e: BaseException):
    """
    Generate an informative warning message for a given error.
    """
    return "\n".join(
        ("\n".join(traceback.format_tb(e.__traceback__)), f"{type(e).__name__}: {e}\n")
    )


def init_benchmark_env():
    """
    Initialize the environment for a benchmark experiment.
    """
    os.environ["GOBBLI_DIR"] = os.path.join(os.path.abspath("."), "benchmark_gobbli")


def assert_param_required(name: str, params: Dict[str, Any]):
    """
    Show a helpful error message if the given parameter wasn't provided.
    """
    if name not in params:
        raise ValueError(f"Missing required parameter '{name}'.")


def assert_proportion(name: str, p: Union[float, int]):
    """
    Show a helpful error message if the given number isn't a valid proportion.
    """
    if not 0 < p <= 1:
        raise ValueError(
            "{name} '{p}' must be greater than 0 and less than or equal to 1"
        )


def assert_valid_model(name: str):
    if getattr(gobbli.model, name, None) is None:
        raise ValueError(
            f"Invalid model name: {name}.  Must be an attribute of `gobbli.model`."
        )


def assert_valid_augment(name: str):
    if getattr(gobbli.augment, name, None) is None:
        raise ValueError(
            f"Invalid augmentation method name: {name}.  Must be an attribute of `gobbli.augment`."
        )


def fasttext_preprocess(texts: List[str]) -> List[str]:
    """
    Apply preprocessing appropriate for a fastText model to a set of texts.

    Args:
      texts: Texts to preprocess.

    Returns:
      List of preprocessed texts.
    """
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)

    processed_texts = []
    for doc in tokenizer.pipe(texts, batch_size=500):
        processed_texts.append(" ".join(tok.lower_ for tok in doc if tok.is_alpha))
    return processed_texts


def bert_preprocess(texts: List[str]) -> List[str]:
    """
    Apply preprocessing appropriate for a BERT (or BERT-based) model to a set of texts.

    Args:
      texts: Texts to preprocess.

    Returns:
      List of preprocessed texts.
    """
    # BERT truncates input, so don't pass in more than is needed
    return [text[:512] for text in texts]


PREPROCESS_FUNCS: Dict[str, Callable[[List[str]], List[str]]] = {
    "fasttext_preprocess": fasttext_preprocess,
    "bert_preprocess": bert_preprocess,
}


def run_benchmark_experiment(
    name: str,
    X: List[str],
    y: List[str],
    model_cls: Any,
    param_grid: Dict[str, List[Any]],
    ray_log_level: Union[int, str] = logging.ERROR,
    worker_log_level: Union[int, str] = logging.ERROR,
    test_dataset: Optional[Tuple[List[str], List[str]]] = None,
    run_kwargs: Optional[Dict[str, Any]] = None,
) -> ClassificationExperimentResults:
    """
    Run a gobbli experiment in the benchmark environment.

    Args:
      name: Name of the experiment
      X: List of texts to predict
      y: List of labels
      model_cls: Class for the model to be instantiated
      param_grid: Model parameters to search for the experiment
      ray_log_level: Log level for local logging (ray cluster)
      worker_log_level: Log level for workers (processes running Docker containers)
      test_dataset: Optional fixed test dataset
      run_kwargs: Additional kwargs passed to :meth:`ClassificationExperiment.run`

    Returns:
      Experiment results
    """
    if run_kwargs is None:
        run_kwargs = {}

    use_gpu = os.getenv("GOBBLI_USE_GPU") is not None

    # FastText doesn't need a GPU
    gpus_needed = 1 if use_gpu and model_cls not in (FastText,) else 0

    exp = ClassificationExperiment(
        model_cls=model_cls,
        dataset=(X, y),
        test_dataset=test_dataset,
        data_dir=Path("./benchmark_meta"),
        name=name,
        param_grid=param_grid,
        task_num_cpus=1,
        task_num_gpus=gpus_needed,
        worker_gobbli_dir=Path("./benchmark_data"),
        worker_log_level=worker_log_level,
        ignore_ray_initialized_error=True,
        overwrite_existing=True,
        ray_kwargs={
            "num_cpus": min(multiprocessing.cpu_count() - 1, 4),
            "num_gpus": 1 if use_gpu else 0,
            "logging_level": ray_log_level,
        },
    )
    return exp.run(**run_kwargs)
