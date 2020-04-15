import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import altair as alt
import click
import pandas as pd
import streamlit as st

from gobbli.inspect.evaluate import ClassificationError, ClassificationEvaluation
from gobbli.interactive.util import (
    DEFAULT_PREDICT_BATCH_SIZE,
    get_label_indices,
    get_predictions,
    load_data,
    safe_sample,
    st_model_metadata,
    st_sample_data,
    st_select_model_checkpoint,
)
from gobbli.util import truncate_text


def show_metrics(metrics: Dict[str, Any]):
    st.header("Metrics")
    md = ""
    for name, value in metrics.items():
        md += f"- **{name}:** {value:.4f}\n"
    st.markdown(md)


def show_plot(plot: alt.Chart):
    st.header("Model Predicted Probability by True Class")
    st.altair_chart(plot)


TRUE_LABEL_COLOR = "#1f78b4"
TRUE_LABEL_TEXT_COLOR = "white"

PRED_PROB_LABEL_RE = re.compile(r"^(.+) \((?:[0-9.]+)\)$")


def _show_example_predictions(
    texts: List[str],
    labels: Optional[List[str]],
    y_pred_proba: pd.DataFrame,
    truncate_len: int,
    top_k: int,
):
    def gather_predictions(row):
        ndx = row.name
        pred_prob_order = row.sort_values(ascending=False)[:top_k]
        data = {"Document": truncate_text(texts[ndx], truncate_len)}
        if labels is not None:
            y_true = labels[ndx]
            column_header = "True Labels" if isinstance(y_true, list) else "True Label"
            data[column_header] = y_true

        for i, (label, pred_prob) in enumerate(pred_prob_order.items()):
            data[f"Predicted Label {i+1}"] = f"{label} ({pred_prob:.3f})"

        return pd.Series(data)

    df = y_pred_proba.apply(gather_predictions, axis=1)

    def style_pred_prob(row, labels):
        ndx = row.name
        true_label_style = (
            f"background-color: {TRUE_LABEL_COLOR};color: {TRUE_LABEL_TEXT_COLOR}"
        )
        style = [
            # Text
            "",
            # True label
            true_label_style,
        ]

        pred_probs = row[2:]
        for p in pred_probs:
            match = re.match(PRED_PROB_LABEL_RE, p)
            if match is None:
                raise ValueError(f"Failed to parse predicted probability cell: {p}")

            y_true = labels[ndx]
            if isinstance(y_true, list):
                is_match = match.group(1) in y_true
            else:
                is_match = match.group(1) == y_true

            if is_match:
                # The cell corresponds to a correct label
                cell_style = true_label_style
            else:
                cell_style = ""
            style.append(cell_style)

        return style

    if labels is not None:
        df = df.style.apply(style_pred_prob, axis=1, labels=labels)

    st.table(df)


def show_example_predictions(
    texts: List[str],
    labels: Optional[List[str]],
    y_pred_proba: pd.DataFrame,
    example_truncate_len: int,
    example_num_docs: int,
    example_top_k: int,
):
    st.header("Example Predictions")

    example_indices = safe_sample(range(len(texts)), example_num_docs)
    _show_example_predictions(
        [texts[i] for i in example_indices],
        None if labels is None else [labels[i] for i in example_indices],
        y_pred_proba.iloc[example_indices, :].reset_index(drop=True),
        example_truncate_len,
        example_top_k,
    )


def show_errors(errors: List[ClassificationError], truncate_len: int = 500):
    df_data = []
    for e in errors:
        pred_class = max(e.y_pred_proba, key=e.y_pred_proba.get)
        pred_class_prob = e.y_pred_proba[pred_class]
        true_column_header = (
            "True Labels" if isinstance(e.y_true, list) else "True Label"
        )
        df_data.append(
            # Use OrderedDict to preserve column order
            OrderedDict(
                {
                    "Document": truncate_text(e.X, truncate_len),
                    true_column_header: e.y_true,
                    "Top Predicted Label": f"{pred_class} ({pred_class_prob:.4f})",
                }
            )
        )
    df = pd.DataFrame(df_data)
    st.table(df)


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
    "--multilabel/--multiclass",
    default=False,
    help="Only applies when reading user data from a file.  If --multilabel, each "
    "observation may have multiple associated labels.  If --multiclass, only one label "
    "is allowed per observation.  Defaults to --multiclass.",
)
@click.option(
    "--multilabel-sep",
    default=",",
    help="Determines how the labels in the label column are separated for a multilabel "
    "dataset read from a file.",
    show_default=True,
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
    multilabel: bool,
    multilabel_sep: str,
    nvidia_visible_devices: str,
):
    st.sidebar.header("Model")
    model_data_path = Path(model_data_dir)
    model_cls, model_kwargs, checkpoint_meta = st_select_model_checkpoint(
        model_data_path, use_gpu, nvidia_visible_devices
    )
    st.title(f"Evaluating: {model_cls.__name__} on {data}")

    model = model_cls(**model_kwargs)
    st_model_metadata(model)

    texts, labels = load_data(
        data,
        multilabel,
        n_rows=None if n_rows == -1 else n_rows,
        multilabel_sep=multilabel_sep,
    )
    if labels is not None:
        label_indices = get_label_indices(labels)

    checkpoint_labels = checkpoint_meta["labels"]
    num_labels = len(checkpoint_labels)

    if labels is not None:
        dataset_labels = set(label_indices.keys())
        if dataset_labels > set(checkpoint_labels):
            st.warning(
                "The dataset contains some labels that the model was not trained on.  No "
                "predictions will be made for these labels: "
                f"{dataset_labels - set(checkpoint_labels)}"
            )

    sampled_texts, sampled_labels = st_sample_data(texts, labels)

    st.sidebar.header("Predictions")
    predict_batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=1,
        max_value=len(sampled_texts),
        value=min(len(sampled_texts), DEFAULT_PREDICT_BATCH_SIZE),
    )

    y_pred_proba = get_predictions(
        model_cls,
        model_kwargs,
        sampled_texts,
        checkpoint_labels,
        checkpoint_meta["checkpoint"],
        batch_size=predict_batch_size,
    )

    evaluation = None

    if labels is not None:
        evaluation = ClassificationEvaluation(
            labels=checkpoint_labels,
            X=sampled_texts,
            y_true=sampled_labels,
            y_pred_proba=y_pred_proba,
        )

        show_metrics(evaluation.metrics())
        show_plot(evaluation.plot())

    st.sidebar.header("Examples")
    example_top_k = st.sidebar.number_input(
        "Top K Predictions to Show",
        min_value=1,
        max_value=num_labels,
        value=min(num_labels, 3),
    )
    example_num_docs = st.sidebar.number_input(
        "Number of Example Documents to Show", min_value=1, max_value=None, value=5
    )
    example_truncate_len = st.sidebar.number_input(
        "Example Document Truncate Length", min_value=1, max_value=None, value=500
    )

    show_example_predictions(
        sampled_texts,
        sampled_labels,
        y_pred_proba,
        example_truncate_len,
        example_num_docs,
        example_top_k,
    )

    if labels is not None and evaluation is not None:
        st.sidebar.header("Errors")
        errors_label = st.sidebar.selectbox(
            "Label to Show Errors For", options=checkpoint_labels
        )
        errors_num_docs = st.sidebar.number_input(
            "Number of Error Documents to Show", min_value=1, max_value=None, value=5
        )
        errors_truncate_len = st.sidebar.number_input(
            "Error Document Truncate Length", min_value=1, max_value=None, value=500
        )

        false_positives, false_negatives = evaluation.errors_for_label(
            errors_label, k=errors_num_docs
        )

        st.header(f"Top Model Errors: {errors_label}")

        st.subheader("False Positives")
        show_errors(false_positives, truncate_len=errors_truncate_len)

        st.subheader("False Negatives")
        show_errors(false_negatives, truncate_len=errors_truncate_len)

    if labels is None:
        st.warning(
            "No ground truth labels found for the passed dataset.  Evaluation "
            "metrics and errors can't be calculated without ground truth labels."
        )


if __name__ == "__main__":
    # Streamlit doesn't like click exiting the script early after the function runs
    try:
        run()
    except SystemExit:
        pass
