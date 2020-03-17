import json
from typing import List, Tuple

import pandas as pd

from gobbli.dataset.base import BaseDataset
from gobbli.util import download_archive


class MovieSummaryDataset(BaseDataset):
    """
    gobbli Dataset for the CMU Movie Summary dataset, framed as a multilabel
    classification problem predicting movie genres from plot summaries.

    http://www.cs.cmu.edu/~ark/personas/
    """

    PLOT_SUMMARIES_FILE = "MovieSummaries/plot_summaries.txt"
    METADATA_FILE = "MovieSummaries/movie.metadata.tsv"
    TRAIN_PCT = 0.8

    def _build(self):
        data_dir = self.data_dir()
        data_dir.mkdir(exist_ok=True, parents=True)

        download_archive(
            "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz", data_dir
        )

    @staticmethod
    def _make_multilabels(genres: pd.Series) -> List[List[str]]:
        return [list(json.loads(g).values()) for g in genres]

    def _is_built(self) -> bool:
        data_dir = self.data_dir()
        return (data_dir / MovieSummaryDataset.PLOT_SUMMARIES_FILE).exists() and (
            data_dir / MovieSummaryDataset.METADATA_FILE
        ).exists()

    def _get_source_df_split(self) -> Tuple[pd.DataFrame, int]:
        if not hasattr(self, "_source_df"):
            data_dir = self.data_dir()
            plot_df = pd.read_csv(
                data_dir / MovieSummaryDataset.PLOT_SUMMARIES_FILE,
                delimiter="\t",
                index_col=0,
                header=None,
                names=["wiki_id", "plot"],
            )

            meta_df = pd.read_csv(
                data_dir / MovieSummaryDataset.METADATA_FILE,
                delimiter="\t",
                index_col=0,
                header=None,
                names=[
                    "wiki_id",
                    "freebase_id",
                    "name",
                    "release_date",
                    "revenue",
                    "runtime",
                    "languages",
                    "countries",
                    "genres",
                ],
            )

            self._source_df = plot_df.join(meta_df, how="inner")[
                ["plot", "genres"]
            ].sort_index()

        return (
            self._source_df,
            int(len(self._source_df) * MovieSummaryDataset.TRAIN_PCT),
        )

    def X_train(self):
        source_df, split_ndx = self._get_source_df_split()
        return source_df["plot"].tolist()[:split_ndx]

    def y_train(self):
        source_df, split_ndx = self._get_source_df_split()
        return MovieSummaryDataset._make_multilabels(source_df["genres"][:split_ndx])

    def X_test(self):
        source_df, split_ndx = self._get_source_df_split()
        return source_df["plot"].tolist()[split_ndx:]

    def y_test(self):
        source_df, split_ndx = self._get_source_df_split()
        return MovieSummaryDataset._make_multilabels(source_df["genres"][split_ndx:])
