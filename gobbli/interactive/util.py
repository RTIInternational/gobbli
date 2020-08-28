import csv
import datetime as dt
import inspect
import itertools
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import pandas as pd
import streamlit as st

import gobbli
import gobbli.model
from gobbli.dataset.base import BaseDataset
from gobbli.io import PredictInput, TaskIO
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.model.mixin import TrainMixin
from gobbli.util import is_multilabel, read_metadata, truncate_text

DEFAULT_PREDICT_BATCH_SIZE = PredictInput.predict_batch_size


@st.cache
def get_label_indices(y: Union[List[str], List[List[str]]]) -> Dict[str, List[int]]:
    label_indices = defaultdict(list)
    if is_multilabel(y):
        for i, labels in enumerate(y):
            for label in labels:
                label_indices[label].append(i)
    else:
        for i, cls in enumerate(y):
            label_indices[cls].append(i)
    return label_indices


def _read_delimited(
    data_file: Path,
    delimiter: str,
    n_rows: Optional[int] = None,
    multilabel: bool = False,
    multilabel_sep: str = ",",
) -> Tuple[List[str], Optional[Union[List[str], List[List[str]]]]]:
    """
    Read up to n_rows lines from the given delimited text file and return lists
    of the texts and labels.  Texts must be stored in a column named "text", and
    labels (if any) must be stored in a column named "label".

    Args:
      data_file: Data file containing one text per line.
      delimiter: Field delimiter for the data file.
      n_rows: The maximum number of rows to read.
      multilabel: If True, read multiple labels for each line.
      multilabel_sep: The separator splitting multiple labels in each line.

    Returns:
      2-tuple: list of read texts and corresponding list of read labels.
    """
    with open(data_file, "r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = set(reader.fieldnames)

        if "text" not in fieldnames:
            raise ValueError("Delimited text file doesn't contain a 'text' column.")
        has_labels = "label" in fieldnames

        rows = list(itertools.islice(reader, n_rows))

    texts: List[str] = []
    labels: List[str] = []

    for row in rows:
        texts.append(row["text"])
        if has_labels:
            label_str = row["label"]
            if multilabel:
                if len(row["label"]) == 0:
                    row_labels = []
                else:
                    row_labels = label_str.split(multilabel_sep)
                labels.append(row_labels)
            else:
                labels.append(label_str)

    return texts, labels if has_labels else None


def _read_lines(data_file: Path, n_rows: Optional[int] = None) -> List[str]:
    """
    Read up to n_rows lines from the given text file and return them in a list.

    Args:
      data_file: Data file containing one text per line.
      n_rows: The maximum number of rows to read.

    Returns:
      List of read lines.
    """
    with open(data_file, "r") as f:
        return list(itertools.islice((l.strip() for l in f), n_rows))


def read_data_file(
    data_file: Path,
    multilabel: bool,
    multilabel_sep: str = ",",
    n_rows: Optional[int] = None,
) -> Tuple[List[str], Optional[Union[List[str], List[List[str]]]]]:
    """
    Read data to explore from a file.  Rows may be sampled using the n_rows argument.

    Args:
      data_file: Path to a data file to read.
      multilabel: If True, read multiple labels for each line.
      multilabel_sep: Separator for multiple labels on the same line.
      n_rows: The maximum number of rows to read.

    Returns:
      2-tuple: list of read texts and a list of read labels (if any)
    """
    extension = data_file.suffix
    if extension == ".tsv":
        texts, labels = _read_delimited(
            data_file,
            "\t",
            n_rows=n_rows,
            multilabel=multilabel,
            multilabel_sep=multilabel_sep,
        )
    elif extension == ".csv":
        texts, labels = _read_delimited(
            data_file,
            ",",
            n_rows=n_rows,
            multilabel=multilabel,
            multilabel_sep=multilabel_sep,
        )
    elif extension == ".txt":
        labels = None
        texts = _read_lines(data_file, n_rows=n_rows)
    else:
        raise ValueError(f"Data file extension '{extension}' is unsupported.")

    return texts, labels


def sample_dataset(
    dataset: BaseDataset, n_rows: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Sample the given number of rows from the given dataset.

    Args:
      dataset: Loaded dataset to sample from.
      n_rows: Optional number of rows to sample.  If None, return all rows.

    Returns:
      2-tuple: a list of texts and a list of labels.  If n_rows was given, these will
      be no longer than n_rows.
    """
    # Apply limit to the dataset, if any
    if n_rows is None:
        texts = dataset.X_train() + dataset.X_test()
        labels = dataset.y_train() + dataset.y_test()
    else:
        # Try to reach the limit from the train split only first
        train_texts = dataset.X_train()[:n_rows]
        train_labels = dataset.y_train()[:n_rows]

        if len(train_texts) < n_rows:
            # If we need more rows to reach the limit, get them
            # from the test set
            test_limit = n_rows - len(train_texts)
            test_texts = dataset.X_test()[:test_limit]
            test_labels = dataset.y_test()[:test_limit]

            texts = train_texts + test_texts
            labels = train_labels + test_labels
        else:
            # Otherwise, just use the limited train data
            texts = train_texts
            labels = train_labels

    return texts, labels


@st.cache(show_spinner=True)
def read_data_file_cached(
    # Streamlit errors sometimes when hashing Path objects, so use a string.
    # https://github.com/streamlit/streamlit/issues/857
    data_file: str,
    multilabel: bool,
    n_rows: Optional[int] = None,
    multilabel_sep: str = ",",
) -> Tuple[List[str], Optional[List[str]]]:
    """
    Streamlit-cached wrapper around :func:`read_data_file` for performance.
    """
    return read_data_file(
        Path(data_file),
        n_rows=n_rows,
        multilabel=multilabel,
        multilabel_sep=multilabel_sep,
    )


def load_data(
    data: str, multilabel: bool, n_rows: Optional[int], multilabel_sep: str = ","
) -> Tuple[List[str], Optional[List[str]]]:
    """
    Load data according to the given 'data' string and row limit.

    Args:
      data: Could be either the name of a gobbli dataset class or a path
        to a data file in a supported format.
      n_rows: Optional limit on number of rows read from the data.
      multilabel: If the dataset is a file and the file has labels, this determines
        whether the labels are interpreted as multiclass (one label per row) or multilabel
        (multiple labels per row).
      multilabel_sep: Determines how the labels in the label column are separated for
        a multilabel dataset read from a file.

    Returns:
      2-tuple: List of texts and optional list of targets.
    """
    if os.path.exists(data):
        data_path = Path(data)
        texts, labels = read_data_file_cached(
            str(data_path), multilabel, n_rows=None if n_rows == -1 else n_rows
        )
    elif data in gobbli.dataset.__all__:
        dataset = getattr(gobbli.dataset, data).load()
        texts, labels = sample_dataset(dataset, None if n_rows == -1 else n_rows)
    else:
        raise ValueError(
            "data argument did not correspond to an existing data file in a "
            "supported format or a built-in gobbli dataset.  Available datasets: "
            f"{gobbli.dataset.__all__}"
        )

    return texts, labels


T = TypeVar("T")


@st.cache
def safe_sample(l: Sequence[T], n: int, seed: Optional[int] = None) -> List[T]:
    if seed is not None:
        random.seed(seed)

    # Prevent an error from trying to sample more than the population
    return list(random.sample(l, min(n, len(l))))


DEFAULT_SAMPLE_SIZE = 100


def st_sample_data(
    texts: List[str], labels: Optional[List[str]]
) -> Tuple[List[str], Optional[List[str]]]:
    """
    Generate streamlit sidebar widgets to facilitate sampling a dataset at runtime.

    Args:
      texts: Full list of texts to sample from.
      labels: Full list of labels to sample from.

    Returns:
      2-tuple: the list of sampled texts and list of sampled labels.
    """
    if len(texts) <= DEFAULT_SAMPLE_SIZE:
        return texts[:], labels[:]

    st.sidebar.header("Sample")

    if st.sidebar.button("Randomize Seed"):
        default_seed = random.randint(0, 1000000)
    else:
        default_seed = 1
    sample_seed = st.sidebar.number_input("Sample Seed", value=default_seed)

    sample_size = st.sidebar.slider(
        "Sample Size",
        min_value=1,
        max_value=len(texts),
        value=min(DEFAULT_SAMPLE_SIZE, len(texts)),
    )

    sample_indices = safe_sample(range(len(texts)), sample_size, seed=sample_seed)

    sampled_texts = [texts[i] for i in sample_indices]

    if labels is None:
        sampled_labels = None
    else:
        sampled_labels = [labels[i] for i in sample_indices]

    return sampled_texts, sampled_labels


def st_example_documents(
    texts: List[str], labels: Optional[List[str]], truncate_len: int
):
    """
    Generate streamlit elements showing example documents (and optionally labe
    """
    df = pd.DataFrame({"Document": [truncate_text(t, truncate_len) for t in texts]})
    if labels is not None:
        df["Label"] = labels
    st.table(df)


def format_task(task_dir: Path) -> str:
    """
    Format the given task for a human-readable dropdown.

    Args:
      task_dir: Directory where the task's data is stored.

    Returns:
      String-formatted, human-readable task metadata.
    """
    task_id = task_dir.name
    try:
        # Should work on OS X, Linux
        task_creation_ts = task_dir.stat().st_birthtime
    except AttributeError:
        # Should work on Windows
        task_creation_ts = task_dir.stat().st_ctime
    task_creation_time = dt.datetime.fromtimestamp(task_creation_ts)
    return f"{task_id[:5]} - Created {task_creation_time.strftime('%Y-%m-%d %H:%M:%S')}"


def st_select_model_checkpoint(
    model_data_path: Path, use_gpu: bool, nvidia_visible_devices: str
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Generate widgets allowing for users to select a checkpoint from a given model directory.

    Args:
      model_data_path: Path to the model data directory to search for checkpoints.
      use_gpu: If True, initialize the model using a GPU.
      nvidia_visible_devices: The list of devices to make available to the model container.
       Should be either "all" or a comma-separated list of device IDs (ex "1,2").

    Returns:
      A 3-tuple: the class of model corresponding to the checkpoint, the kwargs to initialize
      the model with, and the metadata for the checkpoint.
    """
    try:
        model_info = read_metadata(model_data_path / BaseModel._INFO_FILENAME)
    except FileNotFoundError:
        raise ValueError(
            "The passed model data directory does not appear to contain a saved gobbli model. "
            "Did you pass the right directory?"
        )

    model_cls_name = model_info["class"]
    if not hasattr(gobbli.model, model_cls_name):
        raise ValueError(f"Unknown model type: {model_cls_name}")
    model_cls = getattr(gobbli.model, model_cls_name)

    model_kwargs = {
        "data_dir": model_data_path,
        "load_existing": True,
        "use_gpu": use_gpu,
        "nvidia_visible_devices": nvidia_visible_devices,
    }
    model = model_cls(**model_kwargs)

    task_metadata = {}
    if isinstance(model, TrainMixin):
        # The model can be trained, so it may have some trained weights
        model_train_dir = model.train_dir()

        # See if any checkpoints are available for the given model
        for task_dir in model_train_dir.iterdir():
            task_context = ContainerTaskContext(task_dir)
            output_dir = task_context.host_output_dir

            if output_dir.exists():
                metadata_path = output_dir / TaskIO._METADATA_FILENAME
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                        if "checkpoint" in metadata:
                            task_formatted = format_task(task_dir)
                            task_metadata[task_formatted] = metadata

    if len(task_metadata) == 0:
        st.error("No trained checkpoints found for the given model.")
        return

    model_checkpoint = st.sidebar.selectbox(
        "Model Checkpoint", list(task_metadata.keys())
    )
    return model_cls, model_kwargs, task_metadata[model_checkpoint]


def st_select_untrained_model(
    use_gpu: bool,
    nvidia_visible_devices: str,
    predicate: Callable[[Any], bool] = lambda _: True,
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """
    Generate widgets allowing users to select an untrained model and apply arbitrary
    model parameters.

    Args:
      use_gpu: If True, initialize the model using a GPU.
      nvidia_visible_devices: The list of devices to make available to the model container.
       Should be either "all" or a comma-separated list of device IDs (ex "1,2").
      predicate: A predicate used to filter the avaliable model classes.

    Returns:
      A 2-tuple: the class of model and the kwargs to initialized the model with.
    """
    model_choices = [
        cls.__name__
        for name, cls in inspect.getmembers(gobbli.model)
        if inspect.isclass(cls) and issubclass(cls, BaseModel) and predicate(cls)
    ]

    model_cls_name = st.sidebar.selectbox("Model Class", model_choices)
    model_params_str = st.sidebar.text_area("Model Parameters (JSON)", value="{}")

    # Slight convenience if the user deletes the text area contents
    if model_params_str == "":
        model_params_str = "{}"

    model_cls = getattr(gobbli.model, model_cls_name)

    # Validate the model parameter JSON
    try:
        model_params = json.loads(model_params_str)
    except Exception:
        st.sidebar.error("Model parameters must be valid JSON.")
        return None

    model_kwargs = {
        "use_gpu": use_gpu,
        "nvidia_visible_devices": nvidia_visible_devices,
        **model_params,
    }

    # Validate the parameters using the model initialization function
    try:
        model_cls(**model_kwargs)
    except (TypeError, ValueError) as e:
        st.sidebar.error(f"Error validating model parameters: {e}")
        return None

    return model_cls, model_kwargs


def st_model_metadata(model: BaseModel):
    with open(model.metadata_path, "r") as f:
        model_metadata = json.load(f)

    st.header("Model Metadata")
    st.json(model_metadata)


@st.cache(show_spinner=True)
def get_predictions(
    model_cls: BaseModel,
    model_kwargs: Dict[str, Any],
    texts: List[str],
    unique_labels: List[str],
    checkpoint: str,
    batch_size: int = DEFAULT_PREDICT_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Run the given model on the given texts and return the probabilities.

    Args:
      model: Model to use for prediction.
      texts: List of texts to generate predictions for.
      unique_labels: Ordered list of unique labels.
      checkpoint: Model checkpoint to use for prediction.
      batch_size: Batch size for prediction.

    Returns:
      A dataframe containing the predicted probability for each text and label.
    """
    predict_input = PredictInput(
        X=texts,
        labels=unique_labels,
        checkpoint=checkpoint,
        predict_batch_size=batch_size,
    )
    model = model_cls(**model_kwargs)
    model.build()
    predict_output = model.predict(predict_input)
    return predict_output.y_pred_proba
