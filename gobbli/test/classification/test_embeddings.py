import pytest

import gobbli.io
from gobbli.dataset.newsgroups import NewsgroupsDataset
from gobbli.dataset.trivial import TrivialDataset
from gobbli.model.bert import BERT
from gobbli.model.fasttext import FastText
from gobbli.model.random import RandomEmbedder
from gobbli.model.sklearn import TfidfEmbedder
from gobbli.model.spacy import SpaCyModel
from gobbli.model.transformer import Transformer
from gobbli.model.use import USE
from gobbli.test.util import model_test_dir, skip_if_low_resource, validate_checkpoint


def check_embed_output(
    embed_input, embed_output, expected_dimensionality=None, max_seq_length=None
):
    """
    Verify some information regarding the embedding output.
    User can optionally pass expected values of the max sequence
    length and dimensionality, if they're known.
    If not, we'll just verify the dimensionality is larger than 0.
    """
    embeddings = embed_output.X_embedded
    assert len(embeddings) == len(embed_input.X)

    if embed_input.pooling == gobbli.io.EmbedPooling.NONE:
        for embedding, tokens in zip(embeddings, embed_output.embed_tokens):
            assert embedding.ndim == 2

            expected_length = len(tokens)
            if max_seq_length is not None and len(tokens) > max_seq_length:
                expected_length = max_seq_length

            assert embedding.shape[0] == expected_length
            if expected_dimensionality is not None:
                assert embedding.shape[1] == expected_dimensionality
            else:
                assert embedding.shape[1] > 0
    else:
        for embedding in embeddings:
            assert embedding.ndim == 1
            if expected_dimensionality is not None:
                assert embedding.shape[0] == expected_dimensionality
            else:
                assert embedding.shape[0] > 0


@pytest.mark.parametrize("pooling", list(gobbli.io.EmbedPooling))
@pytest.mark.parametrize(
    "model_cls,dataset_cls,model_kwargs,train_kwargs,embed_kwargs",
    [
        (RandomEmbedder, TrivialDataset, {}, {}, {}),
        (RandomEmbedder, NewsgroupsDataset, {}, {}, {}),
        (
            BERT,
            TrivialDataset,
            {},
            {"num_train_epochs": 1, "train_batch_size": 1, "valid_batch_size": 1},
            {"embed_batch_size": 1},
        ),
        (
            BERT,
            NewsgroupsDataset,
            {},
            {"num_train_epochs": 1, "train_batch_size": 32, "valid_batch_size": 8},
            {"embed_batch_size": 32},
        ),
        (USE, TrivialDataset, {}, {}, {"embed_batch_size": 1}),
        (USE, NewsgroupsDataset, {}, {}, {"embed_batch_size": 32}),
        (
            FastText,
            TrivialDataset,
            {},
            {"num_train_epochs": 1, "train_batch_size": 1, "valid_batch_size": 1},
            {"embed_batch_size": 1},
        ),
        (
            FastText,
            NewsgroupsDataset,
            {},
            {"num_train_epochs": 1, "train_batch_size": 32, "valid_batch_size": 8},
            {"embed_batch_size": 32},
        ),
        (
            Transformer,
            TrivialDataset,
            {"max_seq_length": 128},
            {"num_train_epochs": 1, "train_batch_size": 1, "valid_batch_size": 1},
            {"embed_batch_size": 1},
        ),
        (
            Transformer,
            NewsgroupsDataset,
            {"max_seq_length": 128},
            {"num_train_epochs": 1, "train_batch_size": 16, "valid_batch_size": 8},
            {"embed_batch_size": 32},
        ),
        (SpaCyModel, TrivialDataset, {"model": "en_core_web_sm"}, {}, {}),
        (SpaCyModel, NewsgroupsDataset, {"model": "en_core_web_sm"}, {}, {}),
        (TfidfEmbedder, TrivialDataset, {}, {}, {}),
        (TfidfEmbedder, NewsgroupsDataset, {}, {}, {}),
    ],
)
def test_embeddings(
    model_cls,
    dataset_cls,
    model_kwargs,
    train_kwargs,
    embed_kwargs,
    model_gpu_config,
    pooling,
    gobbli_dir,
    request,
):
    """
    Ensure embedding models train and generate embeddings appropriately
    across a few example datasets.
    """
    if (
        model_cls in (USE, FastText, TfidfEmbedder)
        and pooling == gobbli.io.EmbedPooling.NONE
    ):
        pytest.xfail(f"pooling is required for {model_cls.__name__}")

    # These combinations of model and dataset require a lot of memory
    if model_cls in (BERT, Transformer) and dataset_cls in (NewsgroupsDataset,):
        skip_if_low_resource(request.config)

    model = model_cls(
        data_dir=model_test_dir(model_cls),
        load_existing=True,
        **model_gpu_config,
        **model_kwargs,
    )

    model.build()
    ds = dataset_cls.load()

    embed_input = ds.embed_input(limit=50, pooling=pooling, **embed_kwargs)
    check_kwargs = {}
    if isinstance(model, RandomEmbedder):
        check_kwargs["expected_dimensionality"] = RandomEmbedder.DIMENSIONALITY
    if isinstance(model, Transformer):
        check_kwargs["max_seq_length"] = model_kwargs.get("max_seq_length")

    # For models which support generating embeddings without training
    if model_cls not in (FastText,):
        # Verify we can generate embeddings without a trained checkpoint
        embed_output = model.embed(embed_input)
        check_embed_output(embed_input, embed_output, **check_kwargs)

    # Only these models support training for embeddings
    if model_cls in (BERT, FastText, Transformer):
        # Verify embedding runs with a trained checkpoint
        train_output = model.train(ds.train_input(limit=50, **train_kwargs))
        assert train_output.valid_loss is not None
        assert train_output.train_loss is not None
        assert 0 <= train_output.valid_accuracy <= 1

        validate_checkpoint(model_cls, train_output.checkpoint)

        embed_input.checkpoint = train_output.checkpoint
        embed_output = model.embed(embed_input)
        check_embed_output(embed_input, embed_output, **check_kwargs)
