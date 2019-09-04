from pathlib import Path
from typing import Set, Tuple

from gobbli.dataset.nested_file import NestedFileDataset
from gobbli.util import download_archive


class IMDBDataset(NestedFileDataset):
    """
    gobbli Dataset for the IMDB sentiment analysis problem.

    https://ai.stanford.edu/~amaas/data/sentiment/
    """

    def labels(self) -> Set[str]:
        return {"pos", "neg"}

    def download(self, data_dir: Path) -> Path:
        return download_archive(
            "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", data_dir
        )

    def folders(self) -> Tuple[Path, Path]:
        return Path("aclImdb/train"), Path("aclImdb/test")

    def read_source_file(self, file_path: Path) -> str:
        return file_path.read_text()
