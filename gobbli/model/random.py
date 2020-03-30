from typing import List, Optional

import numpy as np

import gobbli.io
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.model.mixin import EmbedMixin, TrainMixin


class RandomEmbedder(BaseModel, TrainMixin, EmbedMixin):
    """
    Dummy embeddings generator that returns random numbers as embeddings
    and has a stub training method to create a uniform API with other embedding
    models.

    Useful for ensuring user code works with the gobbli input/output format
    without having to build a time-consuming model.
    """

    SEED = 1234
    DIMENSIONALITY = 32

    def init(self, params):
        pass

    def _build(self):
        """
        No build step required for this model.
        """

    def tokenize(self, X: List[str]) -> List[List[str]]:
        """
        Return a tokenized list of documents.
        """
        return [x.split() for x in X]

    def _train(
        self, train_input: gobbli.io.TrainInput, context: ContainerTaskContext
    ) -> gobbli.io.TrainOutput:
        """
        No training needed -- stubbed for API uniformity.
        """
        return gobbli.io.TrainOutput(
            valid_loss=0, valid_accuracy=0, train_loss=0, labels=[], multilabel=False
        )

    def _embed(
        self, embed_input: gobbli.io.EmbedInput, context: ContainerTaskContext
    ) -> gobbli.io.EmbedOutput:
        """
        Generate random embeddings.
        """
        np.random.seed(RandomEmbedder.SEED)
        X_tokenized = self.tokenize(embed_input.X)

        embeddings = []
        for tokens in X_tokenized:
            token_embeddings = np.random.rand(
                # sequence length
                len(tokens),
                # dimensionality
                RandomEmbedder.DIMENSIONALITY,
            )

            if embed_input.pooling == gobbli.io.EmbedPooling.MEAN:
                token_embeddings = np.mean(token_embeddings, axis=0)

            embeddings.append(token_embeddings)

        embed_tokens = None  # type: Optional[List[List[str]]]
        if embed_input.pooling == gobbli.io.EmbedPooling.NONE:
            embed_tokens = X_tokenized[:]

        return gobbli.io.EmbedOutput(X_embedded=embeddings, embed_tokens=embed_tokens)
