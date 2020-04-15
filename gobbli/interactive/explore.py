import copy
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
from gobbli.io import EmbedInput, EmbedPooling
from gobbli.model import FastText
from gobbli.model.mixin import EmbedMixin
from gobbli.util import TokenizeMethod, is_multilabel, tokenize, truncate_text

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
    texts: List[str],
    labels: Optional[Union[List[str], List[List[str]]]],
    truncate_len: int,
):
    df = pd.DataFrame({"Document": [truncate_text(t, truncate_len) for t in texts]})
    if labels is not None:
        if is_multilabel(labels):
            label_col = "Labels"
        else:
            label_col = "Label"
        df[label_col] = labels
    st.table(df)


def show_example_documents(
    texts: List[str],
    labels: Union[List[str], List[List[str]]],
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


def _collect_label_counts(labels: Union[List[str], List[List[str]]]) -> pd.DataFrame:
    if is_multilabel(labels):
        counts = defaultdict(int)
        for row_labels in labels:
            for label in row_labels:
                counts[label] += 1
        label_counts = pd.Series(counts) / len(labels)
    else:
        label_counts = pd.Series(labels).value_counts(normalize=True)
    label_counts = label_counts.to_frame(name="Proportion")
    label_counts.index.name = "Label"
    return label_counts.reset_index()


def show_label_distribution(
    sample_labels: Union[List[str], List[List[str]]],
    all_labels: Optional[Union[List[str], List[List[str]]]] = None,
):
    if sample_labels is not None:
        st.header("Label Distribution")
        label_counts = _collect_label_counts(sample_labels)

        if all_labels is None:
            label_chart = (
                alt.Chart(label_counts, height=500, width=700)
                .mark_bar()
                .encode(
                    alt.X("Label", type="nominal"),
                    alt.Y("Proportion", type="quantitative"),
                )
            )
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
        alt.Chart(doc_lengths, height=500, width=700)
        .mark_bar()
        .encode(
            alt.X("Token Count", bin=alt.Bin(maxbins=30)),
            alt.Y("count()", type="quantitative"),
        )
    )

    st.altair_chart(doc_length_chart)


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
    Calculate topics and topic assignments for the given tokens.

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
      2-tuple: The calculated topic structure as returned by gensim and a list containing,
      for each document, the predicted topic ID for that document.
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

    topic_assignments = []
    for doc_bow in corpus:
        topic_probs = model.get_document_topics(doc_bow)
        assigned_topic, _ = max(topic_probs, key=lambda t: t[1])
        topic_assignments.append(assigned_topic)

    return model.top_topics(corpus), topic_assignments


def corr_df_to_heatmap_df(
    corr_df: pd.DataFrame, index_col_name: str, columns_col_name: str
) -> pd.DataFrame:
    """
    Convert a dataframe consisting of a correlation matrix (index values x column values)
    into a long dataframe which can drive an Altair heatmap.
    """
    heatmap_df = corr_df.copy()
    heatmap_df.index.name = index_col_name
    heatmap_df = heatmap_df.reset_index()
    return heatmap_df.melt(
        id_vars=index_col_name, var_name=columns_col_name, value_name="Correlation"
    )


def st_heatmap(
    heatmap_df: pd.DataFrame, x_col_name: str, y_col_name: str, color_col_name: str
):
    heatmap = (
        alt.Chart(heatmap_df, height=700, width=700)
        .mark_rect()
        .encode(alt.X(x_col_name), alt.Y(y_col_name), alt.Color(color_col_name))
    )
    st.altair_chart(heatmap)


def show_topic_model(
    tokens: List[List[str]],
    labels: Optional[List[str]],
    filter_label: Optional[str],
    **model_kwargs,
):
    topics, topic_assignments = get_topics(tokens, **model_kwargs)

    if filter_label is not None:
        st.subheader(f"Label: {filter_label}")

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

    # Combine all topics and labels together to get topic-label, topic-topic, and
    # label-label correlations
    topic_onehot = pd.get_dummies([str(t) for t in topic_assignments])
    label_onehot = pd.get_dummies(labels)
    heatmap_data = pd.concat(
        # Convert topic IDs to strings so Altair doesn't try to make them numeric
        [topic_onehot, label_onehot],
        axis=1,
    )
    heatmap_corr = heatmap_data.corr()

    # Pull out topic-topic correlations and display
    all_topics = topic_onehot.columns
    topic_topic_matrix = heatmap_corr.loc[all_topics, all_topics]

    st.subheader("Topic - Topic Correlation")
    st_heatmap(
        corr_df_to_heatmap_df(topic_topic_matrix, "Topic 1", "Topic 2"),
        "Topic 1",
        "Topic 2",
        "Correlation",
    )

    if labels is not None:
        # Pull out topic-label correlations and display
        all_labels = label_onehot.columns
        topic_label_matrix = heatmap_corr.loc[all_topics, all_labels]

        st.subheader("Topic - Label Correlation")
        st_heatmap(
            corr_df_to_heatmap_df(topic_label_matrix, "Topic", "Label"),
            "Topic",
            "Label",
            "Correlation",
        )


@st.cache(show_spinner=True)
def get_embeddings(
    model_cls: Any,
    model_kwargs: Dict[str, Any],
    texts: List[str],
    checkpoint_meta: Optional[Dict[str, Any]] = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    pooling: EmbedPooling = EmbedPooling.MEAN,
) -> Tuple[List[np.ndarray], Optional[List[List[str]]]]:
    """
    Generate embeddings using the given model and optional trained checkpoint.

    Args:
      model_cls: Class of model to use.
      model_kwargs: Parameters to pass when initializing the model.
      texts: Texts to generate embeddings for.
      checkpoint_meta: Optional metadata for a trained checkpoint.  If passed, a trained model
        will be used.
      batch_size: Batch size to use when generating embeddings.
      pooling: Pooling method for combining embeddings across documents.

    Returns:
      A 2-tuple: First element is the generated embeddings - a numpy array for each passed
      document.  If pooling method is NONE, the second element is the corresponding nested list of
      tokens; otherwise, the second element is None.
    """
    embed_input = EmbedInput(
        X=texts,
        checkpoint=None
        if checkpoint_meta is None
        else Path(checkpoint_meta["checkpoint"]),
        embed_batch_size=batch_size,
        pooling=pooling,
    )
    model = model_cls(**model_kwargs)
    model.build()
    embed_output = model.embed(embed_input)
    if pooling == EmbedPooling.NONE:
        return embed_output.X_embedded, embed_output.embed_tokens
    else:
        return embed_output.X_embedded, None


def show_embeddings(
    model_cls: Any,
    model_kwargs: Dict[str, Any],
    texts: List[str],
    labels: Optional[Union[List[str], List[List[str]]]],
    checkpoint_meta: Optional[Dict[str, Any]] = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    umap_seed: int = 1,
    umap_n_neighbors: int = 15,
    umap_metric: str = "euclidean",
    umap_min_dist: float = 0.1,
    cluster_when: str = "before",
    clusterer: Optional[Any] = None,
    show_vocab_overlap: bool = False,
):
    if cluster_when not in ("before", "after"):
        raise ValueError(f"Unexpected cluster_when value: '{cluster_when}'")

    embeddings, _ = get_embeddings(
        model_cls,
        model_kwargs,
        texts,
        checkpoint_meta=checkpoint_meta,
        batch_size=batch_size,
    )
    X_embedded = pd.DataFrame(embeddings)

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

    dataset_is_multilabel = labels is not None and is_multilabel(labels)
    label_col_name = "Labels" if dataset_is_multilabel else "Label"

    if labels is not None:
        tooltip_attrs.append(label_col_name)
        umap_df[label_col_name] = labels

    if clusters is not None:
        color_attr = "Cluster"
        tooltip_attrs.append("Cluster")
        umap_df["Cluster"] = clusters
        umap_df["Cluster"] = umap_df["Cluster"].astype(str)
    elif labels is not None and not dataset_is_multilabel:
        # Coloring by label doesn't make sense for a multilabel dataset
        color_attr = label_col_name
    else:
        color_attr = None

    # NOTE: Altair (or the underlying charting library, vega-lite) will
    # truncate these texts before being displayed
    umap_df["Text"] = texts

    umap_chart = (
        alt.Chart(umap_df, height=700, width=700)
        .mark_circle(size=60)
        .encode(
            alt.X("UMAP Component 1", scale=alt.Scale(zero=False), axis=None),
            alt.Y("UMAP Component 2", scale=alt.Scale(zero=False), axis=None),
            tooltip=alt.Tooltip(tooltip_attrs),
        )
    )
    if color_attr is not None:
        umap_chart = umap_chart.encode(alt.Color(color_attr))

    st.altair_chart(umap_chart)

    if show_vocab_overlap:
        missing_token_counts = defaultdict(int)
        oov_count = 0
        total_count = 0
        try:
            embeddings, embed_tokens = get_embeddings(
                model_cls,
                model_kwargs,
                texts,
                checkpoint_meta=checkpoint_meta,
                batch_size=batch_size,
                pooling=EmbedPooling.NONE,
            )
        except ValueError:
            st.error(
                "This model doesn't support generating token-level embeddings, "
                "so the vocabulary overlap report can't be calculated."
            )
            return

        for doc_embedding, doc_tokens in zip(embeddings, embed_tokens):
            oov_embeddings = np.abs(doc_embedding).sum(axis=1) == 0
            oov_count += oov_embeddings.sum()
            total_count += len(doc_tokens)
            for oov_token_ndx in np.argwhere(oov_embeddings).ravel():
                missing_token_counts[doc_tokens[oov_token_ndx]] += 1

        st.subheader("Vocabulary Overlap")
        num_oov_tokens_show = 20
        st.markdown(
            f"""
            - Total number of tokens: {total_count:,}
            - Number of out-of-vocabulary tokens: {oov_count:,} ({(oov_count / total_count) * 100:.2f}%)
            """
        )
        if oov_count > 0:
            st.markdown(
                f"""
                - Top {num_oov_tokens_show} most frequent out-of-vocabulary tokens:
                """
            )
            oov_df = pd.DataFrame.from_dict(
                {
                    tok: count
                    for tok, count in sorted(
                        missing_token_counts.items(), key=lambda kv: -kv[1]
                    )
                },
                orient="index",
            ).tail(num_oov_tokens_show)
            oov_df.columns = ["Out-of-Vocabulary Token Count"]
            st.dataframe(oov_df)


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
    multilabel: bool,
    multilabel_sep: str,
    use_gpu: bool,
    nvidia_visible_devices: str,
):
    st.title(f"Exploring: {data}")
    texts, labels = load_data(
        data,
        multilabel,
        None if n_rows == -1 else n_rows,
        multilabel_sep=multilabel_sep,
    )
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
            value=min(len(sampled_texts), DEFAULT_EMBED_BATCH_SIZE),
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

        show_vocab_overlap = st.sidebar.checkbox("Calculate Vocabulary Overlap")

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
                sampled_labels,
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
            cluster_when=cluster_when,
            show_vocab_overlap=show_vocab_overlap,
        )


if __name__ == "__main__":
    # Streamlit doesn't like click exiting the script early after the function runs
    try:
        run()
    except SystemExit:
        pass
