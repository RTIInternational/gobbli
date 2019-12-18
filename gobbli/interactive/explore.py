import itertools
import logging
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import click
import pandas as pd
import streamlit as st
from streamlit.logger import set_log_level

import gobbli.dataset
from gobbli.interactive.util import read_data
from gobbli.util import TokenizeMethod, tokenize, truncate_text


@st.cache(show_spinner=True)
def get_data(
    # Streamlit errors sometimes when hashing Path objects, so use a string.
    # https://github.com/streamlit/streamlit/issues/857
    data_file: str,
    n_rows: Optional[int] = None,
) -> Tuple[List[str], Optional[List[str]]]:
    return read_data(Path(data_file), n_rows=n_rows)


def get_tokens(
    texts: List[str], tokenize_method: TokenizeMethod, vocab_size: int
) -> List[List[str]]:
    # Note: the list of tokens is very large and doesn't work well in the streamlit
    # cache.  Try to cache any results generated from it instead.
    return tokenize(tokenize_method, texts, vocab_size=vocab_size)


@st.cache(show_spinner=True)
def get_document_lengths(
    texts: List[str], tokenize_method: TokenizeMethod, vocab_size: int
) -> List[int]:
    tokens = get_tokens(texts, tokenize_method, vocab_size)
    return [len(d) for d in tokens]


def show_label_distribution(labels: List[str]):
    if labels is not None:
        st.header("Label Distribution")
        label_counts = pd.Series(labels).value_counts().to_frame(name="Count")
        label_counts.index.name = "Label"
        label_counts = label_counts.reset_index()
        label_chart = (
            alt.Chart(label_counts, height=500)
            .mark_bar()
            .encode(alt.X("Label", type="nominal"), alt.Y("Count", type="quantitative"))
        )
        # Hack needed to get streamlit to set the chart height
        # https://github.com/streamlit/streamlit/issues/542
        st.altair_chart(label_chart + label_chart)


def show_document_length_distribution(
    texts: List[str], tokenize_method: TokenizeMethod, vocab_size: int
):
    st.header("Document Length Distribution")
    try:
        document_lengths = get_document_lengths(texts, tokenize_method, vocab_size)
    except RuntimeError as e:
        str_e = str(e)
        if "vocab_size()" in str_e and "pieces_size()" in str_e:
            st.error(
                "SentencePiece requires your texts to have at least as many different tokens "
                "as its vocabulary size.  Try a smaller vocabulary size."
            )
            return
        else:
            raise
    doc_lengths = pd.DataFrame({"Token Count": document_lengths})
    doc_length_chart = (
        alt.Chart(doc_lengths, height=500)
        .mark_bar()
        .encode(
            alt.X("Token Count", bin=alt.Bin(maxbins=30)),
            alt.Y("count()", type="quantitative"),
        )
    )
    # Hack needed to get streamlit to set the chart height
    # https://github.com/streamlit/streamlit/issues/542
    st.altair_chart(doc_length_chart + doc_length_chart)


def _random_documents(texts: List[str], n: int) -> List[str]:
    return random.sample(texts, min(n, len(texts)))


def _show_sample_documents(texts: List[str], truncate_len: int):
    for text in texts:
        st.markdown("---")
        st.text(truncate_text(text, truncate_len))


@st.cache
def get_label_indices(labels: List[str]) -> Dict[str, List[int]]:
    label_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_indices[label].append(i)
    return label_indices


def show_sample_documents(
    texts: List[str],
    label: Optional[str],
    label_indices: Dict[str, List[int]],
    sample_truncate_len: int,
    sample_num_docs: int,
):
    st.header("Sample Documents")

    if label is not None:
        st.subheader(f"Label: {label}")
        label_documents = [texts[i] for i in label_indices[label]]
        sample_population = label_documents
    else:
        sample_population = texts

    _show_sample_documents(
        _random_documents(sample_population, sample_num_docs), sample_truncate_len
    )


@click.command()
@click.argument("data", type=str)
@click.option(
    "--n-rows",
    type=int,
    help="Number of rows to load from the data file.  If -1, load all rows.",
    default=-1,
    show_default=True,
)
def run(data: str, n_rows: int):
    if os.path.exists(data):
        data_path = Path(data)
        st.title(f"Exploring: {data_path}")
        texts, labels = get_data(data_path, n_rows=None if n_rows == -1 else n_rows)
    elif data in gobbli.dataset.__all__:
        dataset = getattr(gobbli.dataset, data).load()
        texts = dataset.X_train() + dataset.X_test()
        labels = dataset.y_train() + dataset.y_test()
    else:
        raise ValueError(
            "data argument did not correspond to an existing data file in a "
            "supported format or a built-in gobbli dataset.  Available datasets: "
            f"{gobbli.dataset.__all__}"
        )

    st.sidebar.header("Tokenization Parameters")
    tokenize_method = TokenizeMethod[
        st.sidebar.selectbox("Method", tuple(tm.name for tm in TokenizeMethod))
    ]
    vocab_size_input = st.sidebar.empty()
    vocab_size = None
    if tokenize_method == TokenizeMethod.SENTENCEPIECE:
        vocab_size = vocab_size_input.number_input(
            "Vocabulary Size", min_value=1, max_value=None, value=20000
        )

    st.sidebar.header("Sample Parameters")
    # Clicking the button reruns the app, which automatically
    # resamples everything.  We may need to rework this if there are performance
    # implications of rerunning the whole app.
    do_refresh = st.sidebar.button("Resample")
    label_indices = get_label_indices(labels)
    label_input = st.sidebar.empty()
    label = None
    if st.sidebar.checkbox("By Label"):
        label = st.sidebar.selectbox("Label", list(sorted(label_indices.keys())))
    sample_truncate_len = st.sidebar.number_input(
        "Sample Truncate Length", min_value=1, max_value=None, value=500
    )
    sample_num_docs = st.sidebar.number_input(
        "Number of Documents Per Label", min_value=1, max_value=None, value=10
    )

    show_label_distribution(labels)

    show_document_length_distribution(texts, tokenize_method, vocab_size)

    show_sample_documents(
        texts, label, label_indices, sample_truncate_len, sample_num_docs
    )


if __name__ == "__main__":
    # Streamlit doesn't like click exiting the script early after the function runs
    try:
        run()
    except SystemExit:
        pass
