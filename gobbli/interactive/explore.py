import copy
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import altair as alt
import click
import pandas as pd
import streamlit as st

import gobbli.dataset
from gobbli.interactive.util import (
    get_label_indices,
    load_data,
    safe_sample,
    st_sample_data,
)
from gobbli.util import TokenizeMethod, tokenize, truncate_text


# For performance, don't let streamlit try to hash the tokens.  It takes forever
@st.cache(allow_output_mutation=True)
def get_tokens(
    texts: List[str], tokenize_method: TokenizeMethod, vocab_size: int
) -> List[List[str]]:
    try:
        return tokenize(tokenize_method, texts, vocab_size=vocab_size)
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


def _show_example_documents(
    texts: List[str], labels: Optional[List[str]], truncate_len: int
):
    df = pd.DataFrame({"Document": [truncate_text(t, truncate_len) for t in texts]})
    if labels is not None:
        df["Label"] = labels
    st.table(df)


def show_example_documents(
    texts: List[str],
    labels: List[str],
    filter_label: Optional[str],
    example_truncate_len: int,
    example_num_docs: int,
):
    st.header("Example Documents")

    # If we're filtered to a specific label,
    # just show it once at the top -- otherwise, show the label
    # with each example
    if filter_label is not None:
        st.subheader(f"Label: {filter_label}")
        example_labels = None
    else:
        example_labels = labels

    example_indices = safe_sample(range(len(texts)), example_num_docs)
    _show_example_documents(
        [texts[i] for i in example_indices],
        [example_labels[i] for i in example_indices]
        if example_labels is not None
        else None,
        example_truncate_len,
    )


def get_document_lengths(tokens: List[List[str]]) -> List[int]:
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


def show_document_length_distribution(tokens: List[List[str]]):
    st.header("Document Length Distribution")
    document_lengths = get_document_lengths(tokens)
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


@st.cache(show_spinner=True)
def get_topics(
    tokens: List[List[str]],
    num_topics: int = 10,
    train_chunksize: int = 2000,
    train_passes: int = 3,
    train_iterations: int = 100,
    do_bigrams: bool = True,
    bigram_min_count: int = 20,
    min_frequency: int = 20,
    max_proportion: float = 0.5,
) -> List[Tuple[List[Tuple[float, str]], float]]:
    """
    Calculate topics for the given tokens.

    Args:
      tokens: Tokenized documents.
      num_topics: Number of topics to pull out of the trained model.
      train_chunksize: Number of documents to process at a time during training.
      train_passes: Number of full passes to make over the dataset.
      train_iterations: Maximum number of iterations through the corpus.
      do_bigrams: If True, use bigram features in addition to unigram tokens.
      bigram_min_count: Minimum number of times 2 tokens need to be colocated to
        be considered a bigram.
      min_frequency: Minimum frequency across all documents for a token to be included
        in the vocabulary.
      max_proportion: Maximum proportion of documents containing a token for it to be
        included in the vocabulary.

    Returns:
      The calculated topic structure as returned by gensim.
    """
    try:
        import gensim
    except ImportError:
        raise ImportError("Topic modeling requires gensim to be installed.")

    # We need to be able to mutate this list
    gensim_tokens = copy.deepcopy(tokens)

    if do_bigrams:
        bigram = gensim.models.Phrases(gensim_tokens, min_count=bigram_min_count)
        for ndx in range(len(gensim_tokens)):
            for token in bigram[gensim_tokens[ndx]]:
                if "_" in token:
                    # Token is a bigram -- add to document.
                    gensim_tokens[ndx].append(token)

    dictionary = gensim.corpora.Dictionary(tokens)
    dictionary.filter_extremes(no_below=min_frequency, no_above=max_proportion)
    corpus = [dictionary.doc2bow(doc) for doc in gensim_tokens]

    _ = dictionary[0]  # "Load" the dictionary
    id2word = dictionary.id2token

    model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=train_chunksize,
        alpha="auto",
        eta="auto",
        iterations=train_iterations,
        num_topics=num_topics,
        passes=train_passes,
        eval_every=None,
    )

    return model.top_topics(corpus)


def show_topic_model(
    run_topic_model: bool, tokens: List[List[str]], label: Optional[str], **model_kwargs
):
    st.header("Topic Model")

    if not run_topic_model:
        st.markdown(
            "Click the 'Train Model' button in the sidebar to train a topic model."
        )
    else:
        topics = get_topics(tokens, **model_kwargs)

        if label is not None:
            st.subheader(f"Label: {label}")

        for i, (topic, coherence) in enumerate(topics):
            st.subheader(f"Topic {i} (coherence: {coherence:.4f})")

            md = ""
            for probability, word in topic:
                md += f"- {word} ({probability:.4f})\n"

            st.markdown(md)


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
    st.title(f"Exploring: {data}")
    texts, labels = load_data(data, None if n_rows == -1 else n_rows)
    if labels is not None:
        label_indices = get_label_indices(labels)

    #
    # Sidebar
    #

    st.sidebar.header("Filter Parameters")

    filter_label = None
    if labels is not None and st.sidebar.checkbox(
        "Filter By Label", key="sample_by_label"
    ):
        filter_label = st.sidebar.selectbox("Label", list(sorted(label_indices.keys())))

    if filter_label is None:
        filtered_texts = texts
        filtered_labels = labels
    else:
        filtered_texts = [texts[i] for i in label_indices[filter_label]]
        filtered_labels = [labels[i] for i in label_indices[filter_label]]

    sampled_texts, sampled_labels = st_sample_data(filtered_texts, filtered_labels)

    st.sidebar.header("Example Parameters")
    example_truncate_len = st.sidebar.number_input(
        "Example Truncate Length", min_value=1, max_value=None, value=500
    )
    example_num_docs = st.sidebar.number_input(
        "Number of Example Documents", min_value=1, max_value=None, value=5
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

    st.sidebar.header("Topic Model Parameters")
    run_topic_model = st.sidebar.button("Train Model")
    num_topics = st.sidebar.number_input(
        "Number of Topics", min_value=2, max_value=None, value=10
    )
    train_chunksize = st.sidebar.number_input(
        "Training Chunk Size", min_value=1, max_value=None, value=2000
    )
    train_passes = st.sidebar.number_input(
        "Number of Training Passes", min_value=1, max_value=None, value=3
    )
    train_iterations = st.sidebar.number_input(
        "Number of Training Iterations", min_value=1, max_value=None, value=100
    )
    do_bigrams = st.sidebar.checkbox("Use Bigrams?", value=True)
    if do_bigrams:
        bigram_min_count = st.sidebar.number_input(
            "Minimum Count for Bigrams", min_value=1, max_value=None, value=20
        )
    min_frequency = st.sidebar.number_input(
        "Minimum Vocabulary Frequency", min_value=1, max_value=None, value=20
    )
    max_proportion = st.sidebar.number_input(
        "Maximum Vocabulary Proportion", min_value=0.0, max_value=1.0, value=0.5
    )

    #
    # Main section
    #

    show_example_documents(
        sampled_texts,
        sampled_labels,
        filter_label,
        example_truncate_len,
        example_num_docs,
    )

    if filter_label is None:
        show_label_distribution(sampled_labels)

    tokens = get_tokens(sampled_texts, tokenize_method, vocab_size)

    show_document_length_distribution(tokens)

    try:
        show_topic_model(
            run_topic_model,
            tokens,
            filter_label,
            num_topics=num_topics,
            train_chunksize=train_chunksize,
            train_passes=train_passes,
            train_iterations=train_iterations,
            do_bigrams=do_bigrams,
            bigram_min_count=bigram_min_count,
            min_frequency=min_frequency,
            max_proportion=max_proportion,
        )
    except ImportError as e:
        st.error(str(e))


if __name__ == "__main__":
    # Streamlit doesn't like click exiting the script early after the function runs
    try:
        run()
    except SystemExit:
        pass
