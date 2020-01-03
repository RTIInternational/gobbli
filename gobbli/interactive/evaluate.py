import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import pandas as pd
import streamlit as st

import gobbli
from gobbli.inspect.evaluate import ClassificationEvaluation
from gobbli.interactive.util import get_label_indices, load_data, st_sample_data
from gobbli.io import PredictInput, TaskIO
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.model.mixin import TrainMixin


def format_task(task_dir: Path) -> str:
    """
    Format the given task for a human-readable dropdown.

    Args:
      task_dir: Directory where the task's data is stored.

    Returns:
      String-formatted, human-readable task metadata.
    """
    task_id = task_dir.name
    task_creation_time = dt.datetime.fromtimestamp(task_dir.stat().st_birthtime)
    return f"{task_id[:5]} - Created {task_creation_time.strftime('%Y-%m-%d %H:%M:%S')}"


@st.cache(show_spinner=True)
def get_predictions(
    model_cls: BaseModel,
    model_kwargs: Dict[str, Any],
    texts: List[str],
    unique_labels: List[str],
    checkpoint: str,
    batch_size: int = 32,
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
    predict_output = model.predict(predict_input)
    return predict_output.y_pred_proba


@click.command()
@click.argument("model_data_dir", type=str)
@click.argument("data", type=str)
@click.option(
    "--n-rows",
    type=int,
    help="Number of rows to load from the data file.  If -1, load all rows.",
    default=-1,
    show_default=True,
)
@click.option(
    "--use-gpu/--use-cpu",
    default=False,
    help="Which device to run the model on. Defaults to CPU.",
)
@click.option(
    "--nvidia-visible-devices",
    default="all",
    help="Which GPUs to make available to the container; ignored if running on CPU. "
    "If not 'all', should be a comma-separated string: ex. ``1,2``.",
    show_default=True,
)
def run(
    model_data_dir: str,
    data: str,
    n_rows: int,
    use_gpu: bool,
    nvidia_visible_devices: str,
):
    model_data_path = Path(model_data_dir)
    model_cls_name = model_data_path.parent.name
    if not hasattr(gobbli.model, model_cls_name):
        raise ValueError(f"Unknown model type: {model_cls_name}")
    model_cls = getattr(gobbli.model, model_cls_name)

    st.title(f"Evaluating: {model_cls.__name__} on {data}")

    model_kwargs = {
        "data_dir": model_data_path,
        "load_existing": True,
        "use_gpu": use_gpu,
        "nvidia_visible_devices": nvidia_visible_devices,
    }
    model = model_cls(**model_kwargs)

    with open(model.metadata_path, "r") as f:
        model_metadata = json.load(f)

    st.header("Model Metadata")
    st.json(model_metadata)

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

    texts, labels = load_data(data, n_rows=None if n_rows == -1 else n_rows)
    if labels is not None:
        label_indices = get_label_indices(labels)

    st.sidebar.header("Model Parameters")
    model_checkpoint = st.sidebar.selectbox("Checkpoint", list(task_metadata.keys()))
    checkpoint_meta = task_metadata[model_checkpoint]
    checkpoint_labels = checkpoint_meta["labels"]

    if labels is not None:
        dataset_labels = set(label_indices.keys())
        if dataset_labels != set(checkpoint_labels):
            st.error(
                f"Labels for the dataset ({dataset_labels}) are different from the model checkpoint "
                f"labels ({checkpoint_labels}).  They must match to evaluate the model."
            )
            return

    sampled_texts, sampled_labels = st_sample_data(texts, labels)

    y_pred_proba = get_predictions(
        model_cls,
        model_kwargs,
        sampled_texts,
        checkpoint_labels,
        checkpoint_meta["checkpoint"],
    )

    if labels is None:
        st.warning(
            "No ground truth labels found for the passed dataset.  Evaluation/error "
            "metrics can't be calculated without ground truth labels."
        )
    else:
        st.header("Evaluation")
        evaluation = ClassificationEvaluation(
            labels=checkpoint_labels,
            X=sampled_texts,
            y_true=sampled_labels,
            y_pred_proba=y_pred_proba,
        )

        metrics_df = pd.DataFrame({"Metric": pd.Series(evaluation.metrics())})
        st.dataframe(metrics_df)


if __name__ == "__main__":
    # Streamlit doesn't like click exiting the script early after the function runs
    try:
        run()
    except SystemExit:
        pass
