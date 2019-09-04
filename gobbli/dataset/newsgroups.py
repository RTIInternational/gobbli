from pathlib import Path
from typing import Set, Tuple

from gobbli.dataset.nested_file import NestedFileDataset
from gobbli.util import download_archive


class NewsgroupsDataset(NestedFileDataset):
    """
    gobbli Dataset for the 20 Newsgroups problem.

    http://qwone.com/~jason/20Newsgroups/
    """

    def labels(self) -> Set[str]:
        return {
            "alt.atheism",
            "comp.graphics",
            "comp.os.ms-windows.misc",
            "comp.sys.ibm.pc.hardware",
            "comp.sys.mac.hardware",
            "comp.windows.x",
            "misc.forsale",
            "rec.autos",
            "rec.motorcycles",
            "rec.sport.baseball",
            "rec.sport.hockey",
            "sci.crypt",
            "sci.electronics",
            "sci.med",
            "sci.space",
            "soc.religion.christian",
            "talk.politics.guns",
            "talk.politics.mideast",
            "talk.politics.misc",
            "talk.religion.misc",
        }

    def download(self, data_dir: Path) -> Path:
        return download_archive(
            "https://ndownloader.figshare.com/files/5975967",
            data_dir,
            filename="20news-bydate.tar.gz",
        )

    def folders(self) -> Tuple[Path, Path]:
        return Path("20news-bydate-train"), Path("20news-bydate-test")

    def read_source_file(self, file_path: Path) -> str:
        return file_path.read_text(encoding="latin-1")
