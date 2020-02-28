import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import click
import hdbscan
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.neighbors import VALID_METRICS as SKLEARN_DISTANCE_METRICS
from umap import UMAP
from umap.distances import named_distances

from gobbli.interactive.util import (
    get_label_indices,
    load_data,
    safe_sample,
    st_sample_data,
    st_select_model_checkpoint,
    st_select_untrained_model,
)
from gobbli.io import EmbedInput
from gobbli.model import FastText
from gobbli.model.mixin import EmbedMixin
from gobbli.util import TokenizeMethod, tokenize, truncate_text

DEFAULT_EMBED_BATCH_SIZE = EmbedInput.embed_batch_size

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


def _collect_label_counts(labels: List[str]) -> pd.DataFrame:
    label_counts = (
        pd.Series(labels).value_counts(normalize=True).to_frame(name="Proportion")
    )
    label_counts.index.name = "Label"
    return label_counts.reset_index()


def show_label_distribution(
    sample_labels: List[str], all_labels: Optional[List[str]] = None
):
    if sample_labels is not None:
        st.header("Label Distribution")
        label_counts = _collect_label_counts(sample_labels)

        if all_labels is None:
            label_chart = (
                alt.Chart(label_counts, height=500)
                .mark_bar()
                .encode(
                    alt.X("Label", type="nominal"),
                    alt.Y("Proportion", type="quantitative"),
                )
            )
            # Hack needed to get streamlit to set the chart height
            # https://github.com/streamlit/streamlit/issues/542
            label_chart += label_chart
        else:
            label_counts["Label Set"] = "Sample"
            all_label_counts = _collect_label_counts(all_labels)
            all_label_counts["Label Set"] = "All Documents"
            label_counts = pd.concat([label_counts, all_label_counts])

            label_chart = (
                alt.Chart(label_counts, width=100)
                .mark_bar()
                .encode(
                    alt.X(
                        "Label Set",
                        type="nominal",
                        title=None,
                        sort=["Sample", "All Documents"],
                    ),
                    alt.Y("Proportion", type="quantitative"),
                    alt.Column(
                        "Label", type="nominal", header=alt.Header(labelAngle=0)
                    ),
                    alt.Color("Label Set", type="nominal", legend=None),
                )
            )

        st.altair_chart(label_chart)


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


def show_topic_model(tokens: List[List[str]], label: Optional[str], **model_kwargs):
    topics = get_topics(tokens, **model_kwargs)

    if label is not None:
        st.subheader(f"Label: {label}")

    topic_df_data = []
    for i, (topic, coherence) in enumerate(topics):
        topic_data = {"Coherence": coherence}

        for j, (probability, word) in enumerate(topic):
            topic_data[j + 1] = f"{word} ({probability:.4f})"

        topic_df_data.append(topic_data)

    topic_df = pd.DataFrame(topic_df_data).T
    topic_df.columns = [f"Topic {i+1}" for i in range(len(topic_df.columns))]

    # This is enough to show the full df, but streamlit renders it with a scrollbar still,
    # so there's still a little bit of scroll
    st.dataframe(topic_df, height=750)


@st.cache(show_spinner=True)
def get_embeddings(
    model_cls: Any,
    model_kwargs: Dict[str, Any],
    texts: List[str],
    checkpoint_meta: Optional[Dict[str, Any]] = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> List[np.ndarray]:
    """
    Generate embeddings using the given model and optional trained checkpoint.

    Args:
      model_cls: Class of model to use.
      model_kwargs: Parameters to pass when initializing the model.
      texts: Texts to generate embeddings for.
      checkpoint_meta: Optional metadata for a trained checkpoint.  If passed, a trained model
        will be used.
      batch_size: Batch size to use when generating embeddings.

    Returns:
      The generated embeddings - a numpy array for each passed document.
    """
    embed_input = EmbedInput(
        X=texts,
        checkpoint=None
        if checkpoint_meta is None
        else Path(checkpoint_meta["checkpoint"]),
        embed_batch_size=batch_size,
    )
    model = model_cls(**model_kwargs)
    model.build()
    embed_output = model.embed(embed_input)
    return embed_output.X_embedded


def show_embeddings(
    model_cls: Any,
    model_kwargs: Dict[str, Any],
    texts: List[str],
    labels: Optional[List[str]],
    checkpoint_meta: Optional[Dict[str, Any]] = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    umap_seed: int = 1,
    umap_n_neighbors: int = 15,
    umap_metric: str = "euclidean",
    umap_min_dist: float = 0.1,
    cluster_when: str = "before",
    clusterer: Optional[Any] = None,
):
    if cluster_when not in ("before", "after"):
        raise ValueError(f"Unexpected cluster_when value: '{cluster_when}'")

    X_embedded = pd.DataFrame(
        get_embeddings(
            model_cls,
            model_kwargs,
            texts,
            checkpoint_meta=checkpoint_meta,
            batch_size=batch_size,
        )
    )

    umap = UMAP(
        n_neighbors=umap_n_neighbors,
        metric=umap_metric,
        min_dist=umap_min_dist,
        random_state=umap_seed,
    )

    clusters = None
    if clusterer is not None and cluster_when == "before":
        clusterer.fit(X_embedded)
        clusters = clusterer.labels_

    umap_data = umap.fit_transform(X_embedded)

    if clusterer is not None and cluster_when == "after":
        clusterer.fit(umap_data)
        clusters = clusterer.labels_

    umap_df = pd.DataFrame(umap_data, columns=["UMAP Component 1", "UMAP Component 2"])
    tooltip_attrs = ["Text"]

    if labels is not None:
        tooltip_attrs.append("Label")
        umap_df["Label"] = labels

    if clusters is not None:
        color_attr = "Cluster"
        tooltip_attrs.append("Cluster")
        umap_df["Cluster"] = clusters
        umap_df["Cluster"] = umap_df["Cluster"].astype(str)
    elif labels is not None:
        color_attr = "Label"

    # NOTE: Altair (or the underlying charting library, vega-lite) will
    # truncate these texts before being displayed
    umap_df["Text"] = texts

    umap_chart = (
        alt.Chart(umap_df, height=700)
        .mark_circle(size=60)
        .encode(
            alt.X("UMAP Component 1", scale=alt.Scale(zero=False), axis=None),
            alt.Y("UMAP Component 2", scale=alt.Scale(zero=False), axis=None),
            alt.Color(color_attr),
            tooltip=alt.Tooltip(tooltip_attrs),
        )
    )

    # Hack needed to get streamlit to set the chart height
    # https://github.com/streamlit/streamlit/issues/542
    st.altair_chart(umap_chart + umap_chart)


@click.command()
@click.argument("data", type=str)
@click.option(
    "--n-rows",
    type=int,
    help="Number of rows to load from the data file.  If -1, load all rows.",
    default=-1,
    show_default=True,
)
@click.option(
    "--model-data-dir",
    type=str,
    help="Optional data directory for a model.  If given, the model checkpoints "
    "will become available to use for embedding generation, if the model supports it.",
    default=None,
)
@click.option(
    "--use-gpu/--use-cpu",
    default=False,
    help="Which device to run the model on, if any. Defaults to CPU.",
)
@click.option(
    "--nvidia-visible-devices",
    default="all",
    help="Which GPUs to make available to the container; ignored if running on CPU. "
    "If not 'all', should be a comma-separated string: ex. ``1,2``.",
    show_default=True,
)
def run(
    data: str,
    n_rows: int,
    model_data_dir: Optional[str],
    use_gpu: bool,
    nvidia_visible_devices: str,
):
    st.title(f"Exploring: {data}")
    texts, labels = load_data(data, None if n_rows == -1 else n_rows)
    if labels is not None:
        label_indices = get_label_indices(labels)

    #
    # Sidebar
    #

    st.sidebar.header("Filter")

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

    st.sidebar.header("Examples")
    example_truncate_len = st.sidebar.number_input(
        "Example Truncate Length", min_value=1, max_value=None, value=500
    )
    example_num_docs = st.sidebar.number_input(
        "Number of Example Documents", min_value=1, max_value=None, value=5
    )

    st.sidebar.header("Labels")
    show_full_label_distribution = st.sidebar.checkbox("Show Full Label Distribution")

    st.sidebar.header("Tokenization")
    tokenize_method = TokenizeMethod[
        st.sidebar.selectbox("Method", tuple(tm.name for tm in TokenizeMethod))
    ]
    vocab_size_input = st.sidebar.empty()
    vocab_size = None
    if tokenize_method == TokenizeMethod.SENTENCEPIECE:
        vocab_size = vocab_size_input.number_input(
            "Vocabulary Size", min_value=1, max_value=None, value=20000
        )

    st.sidebar.header("Topic Model")
    run_topic_model = False
    if st.sidebar.checkbox("Enable Topic Model"):
        run_topic_model = st.sidebar.button("Train Topic Model")
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

    st.sidebar.header("Embeddings")
    run_embeddings = False
    embeddings_error = False
    embed_batch_size = DEFAULT_EMBED_BATCH_SIZE
    if st.sidebar.checkbox("Enable Embeddings"):
        run_embeddings = st.sidebar.button("Generate Embeddings")
        model_type_options = ["Untrained"]
        if model_data_dir is not None:
            model_type_options.insert(0, "Trained")
        else:
            st.sidebar.markdown(
                "**Note**: Using a trained model requires passing a model to this app "
                "via the --model-data-dir command line argument."
            )
        model_type = st.sidebar.selectbox("Model Type", model_type_options)

        if model_type == "Trained" and model_data_dir is not None:
            result = st_select_model_checkpoint(
                Path(model_data_dir), use_gpu, nvidia_visible_devices
            )
            if result is None:
                embeddings_error = True
                st.sidebar.error("Error selecting model checkpoint.")
            else:
                model_cls, model_kwargs, checkpoint_meta = result
        else:
            checkpoint_meta = None
            model_result = st_select_untrained_model(
                use_gpu,
                nvidia_visible_devices,
                # We want only models that support embeddings and aren't
                # fastText, since fastText requires training before generating embeddings
                predicate=lambda m: issubclass(m, EmbedMixin) and m is not FastText,
            )

            if model_result is None:
                run_embeddings = False
                embeddings_error = True
            else:
                model_cls, model_kwargs = model_result

        embed_batch_size = st.sidebar.number_input(
            "Batch Size",
            min_value=1,
            max_value=len(sampled_texts),
            value=DEFAULT_EMBED_BATCH_SIZE,
        )

        umap_seed = st.sidebar.number_input(
            "UMAP Random Seed", min_value=1, max_value=None, value=1
        )
        umap_n_neighbors = st.sidebar.number_input(
            "UMAP Number of Neigbors", min_value=1, max_value=None, value=15
        )
        umap_metric = st.sidebar.selectbox(
            "UMAP Distance Metric", list(named_distances.keys())
        )
        umap_min_dist = st.sidebar.number_input(
            "UMAP Minimum Distance", min_value=0.0, max_value=1.0, value=0.1
        )

        cluster_when = "before"
        clusterer = None
        if st.sidebar.checkbox("Cluster Embeddings"):
            clustering_algorithm = st.sidebar.selectbox(
                "Clustering Algorithm", ["K-Means", "HDBSCAN"]
            )
            cluster_when_labels = {
                "before": "Before Dimensionality Reduction",
                "after": "After Dimensionality Reduction",
            }
            cluster_when = st.sidebar.selectbox(
                "When to Apply Clusters",
                ["before", "after"],
                format_func=lambda l: cluster_when_labels[l],
            )
            if clustering_algorithm == "K-Means":
                n_clusters = st.sidebar.number_input(
                    "K-Means Number of Clusters", min_value=1, max_value=None, value=10
                )
                max_iter = st.sidebar.number_input(
                    "K-Means Max Iterations", min_value=1, max_value=None, value=300
                )

                kmeans_seed = st.sidebar.number_input(
                    "K-Means Random Seed", min_value=1, max_value=None, value=1
                )
                clusterer = KMeans(
                    n_clusters=n_clusters, max_iter=max_iter, random_state=kmeans_seed
                )
            else:
                hdbscan_min_cluster_size = st.sidebar.number_input(
                    "HDBSCAN Minimum Cluster Size", min_value=1, max_value=None, value=2
                )
                hdbscan_min_samples = st.sidebar.number_input(
                    "HDBSCAN Minimum Samples", min_value=1, max_value=None, value=2
                )
                hdbscan_cluster_selection_epsilon = st.sidebar.number_input(
                    "HDBSCAN Cluster Selection Epsilon",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                )
                # There's a list of these available directly in HDBSCAN, but it's not
                # necessarily up-to-date with the distance metrics that are actually
                # implemented in sklearn
                hdbscan_metric_options = SKLEARN_DISTANCE_METRICS["ball_tree"]
                hdbscan_metric = st.sidebar.selectbox(
                    "HDBSCAN Distance Metric",
                    hdbscan_metric_options,
                    index=hdbscan_metric_options.index("euclidean"),
                )

                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=hdbscan_min_cluster_size,
                    min_samples=hdbscan_min_samples,
                    cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
                    metric=hdbscan_metric,
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
        show_label_distribution(
            sampled_labels,
            all_labels=filtered_labels if show_full_label_distribution else None,
        )

    tokens = get_tokens(sampled_texts, tokenize_method, vocab_size)

    show_document_length_distribution(tokens)

    st.header("Topic Model")
    if not run_topic_model:
        st.markdown(
            "Enable topic modeling in the sidebar and click the 'Train Topic Model' button to train a topic model."
        )
    else:
        try:
            show_topic_model(
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

    st.header("Embeddings")

    if embeddings_error:
        st.error(
            "Embeddings could not be generated due to bad parameters.  Look for errors in the 'Embeddings' section of the sidebar."
        )
    elif not run_embeddings:
        st.markdown(
            "Enable embeddings in the sidebar and click the 'Generate Embeddings' button to generate embeddings for the dataset."
        )
    else:
        show_embeddings(
            model_cls,
            model_kwargs,
            sampled_texts,
            sampled_labels,
            checkpoint_meta,
            batch_size=embed_batch_size,
            umap_seed=umap_seed,
            umap_n_neighbors=umap_n_neighbors,
            umap_metric=umap_metric,
            umap_min_dist=umap_min_dist,
            clusterer=clusterer,
        )


if __name__ == "__main__":
    # Streamlit doesn't like click exiting the script early after the function runs
    try:
        run()
    except SystemExit:
        pass
