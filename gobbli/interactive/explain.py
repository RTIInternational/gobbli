from pathlib import Path
from typing import Any, Callable, Dict, List

import click
import eli5
import numpy as np
import streamlit as st
from eli5.lime import TextExplainer

from gobbli.interactive.util import (
    DEFAULT_PREDICT_BATCH_SIZE,
    get_predictions,
    load_data,
    st_model_metadata,
    st_select_model_checkpoint,
)


def make_predict_func(
    model_cls: Any,
    model_kwargs: Dict[str, Any],
    unique_labels: List[str],
    checkpoint: str,
    batch_size: int,
) -> Callable[[List[str]], np.ndarray]:
    def predict(texts: List[str]) -> np.ndarray:
        preds = get_predictions(
            model_cls,
            model_kwargs,
            texts,
            unique_labels,
            checkpoint,
            batch_size=batch_size,
        ).values
        # Ensure rows sum to 1 -- eli5 raises an error if this isn't true
        preds = preds / preds.sum(axis=1, keepdims=1)
        return preds

    return predict


def st_lime_explanation(
    text: str,
    predict_func: Callable[[List[str]], np.ndarray],
    unique_labels: List[str],
    n_samples: int,
    position_dependent: bool = True,
):
    # TODO just use ELI5's built-in visualization when streamlit supports it:
    # https://github.com/streamlit/streamlit/issues/779
    with st.spinner("Generating LIME explanations..."):
        te = TextExplainer(
            random_state=1, n_samples=n_samples, position_dependent=position_dependent
        )
        te.fit(text, predict_func)
    st.json(te.metrics_)
    explanation = te.explain_prediction()
    explanation_df = eli5.format_as_dataframe(explanation)
    for target_ndx, target in enumerate(
        sorted(explanation.targets, key=lambda t: -t.proba)
    ):
        target_explanation_df = explanation_df[
            explanation_df["target"] == target_ndx
        ].copy()

        target_explanation_df["contribution"] = (
            target_explanation_df["weight"] * target_explanation_df["value"]
        )
        target_explanation_df["abs_contribution"] = abs(
            target_explanation_df["contribution"]
        )
        target_explanation_df = (
            target_explanation_df.drop("target", axis=1)
            .sort_values(by="abs_contribution", ascending=False)
            .reset_index(drop=True)
        )
        st.subheader(
            f"Target: {unique_labels[target_ndx]} (probability {target.proba:.4f}, score {target.score:.4f})"
        )
        st.dataframe(target_explanation_df)


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
    st.title(f"Explaining: {model_cls.__name__} using {data}")

    model = model_cls(**model_kwargs)
    st_model_metadata(model)

    texts, labels = load_data(
        data,
        multilabel,
        n_rows=None if n_rows == -1 else n_rows,
        multilabel_sep=multilabel_sep,
    )

    st.sidebar.header("Example")
    example_ndx = st.sidebar.number_input(
        f"Index of Example (0 to {len(texts)-1})",
        min_value=0,
        max_value=len(texts) - 1,
        value=0,
    )

    if labels is None:
        header_text = "Example to Explain"
    else:
        y_true = labels[example_ndx]
        if isinstance(y_true, list):
            header_text = f"Example to Explain (labels: {', '.join(y_true)})"
        else:
            header_text = f"Example to Explain (class: {y_true})"
    st.header(header_text)
    st.text(texts[example_ndx])

    st.sidebar.header("Predictions")
    predict_batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=1,
        max_value=len(texts),
        value=min(len(texts), DEFAULT_PREDICT_BATCH_SIZE),
    )

    predict_func = make_predict_func(
        model_cls,
        model_kwargs,
        checkpoint_meta["labels"],
        checkpoint_meta["checkpoint"],
        predict_batch_size,
    )

    do_lime = st.sidebar.checkbox("Generate LIME explanation")

    do_run = st.sidebar.button("Run")

    if do_lime:
        st.sidebar.header("LIME Parameters")
        n_samples = st.sidebar.number_input(
            "Number of Samples", min_value=1, max_value=None, value=500
        )
        position_dependent = st.sidebar.checkbox(
            "Use position-dependent vectorizer", value=True
        )
        if do_run:
            st.header("LIME Explanation")
            st_lime_explanation(
                texts[example_ndx],
                predict_func,
                checkpoint_meta["labels"],
                n_samples,
                position_dependent=position_dependent,
            )

    if not do_lime:
        st.header("Explanation")
        st.markdown(
            "Select one or more explanation methods in the sidebar and click 'Run' to generate explanations."
        )


if __name__ == "__main__":
    # Streamlit doesn't like click exiting the script early after the function runs
    try:
        run()
    except SystemExit:
        pass
