from abc import abstractmethod
from pathlib import Path
from typing import List, Set, Tuple

from gobbli.dataset.base import BaseDataset


class NestedFileDataset(BaseDataset):
    """
    A dataset downloaded as an archive from some URL
    and composed of the following directory structure:

    <train_folder>/
      <label1>/
        data1
        data2
      <label2>/
        data1
        data2
    <test_folder>/
      <label1>/
        data1
        data2
      <label2>/
        data1
        data2
    """

    TRAIN_X_FILE = "train_X.csv"
    TRAIN_Y_FILE = "train_y.csv"

    TEST_X_FILE = "test_X.csv"
    TEST_Y_FILE = "test_Y.csv"

    # Null byte _shouldn't_ be embedded in any data files...
    DELIMITER = "\x00"

    @abstractmethod
    def labels(self) -> Set[str]:
        """
        Return the set of folder names that should be considered
        labels in each directory.
        """
        raise NotImplementedError

    @abstractmethod
    def download(self, data_dir: Path) -> Path:
        """
        Download and extract the dataset archive into the given data dir.
        Return the resulting path.
        """
        raise NotImplementedError

    @abstractmethod
    def folders(self) -> Tuple[Path, Path]:
        """
        Return relative paths to the train and test folders, respectively, from the
        top level of the extracted archive.
        """
        raise NotImplementedError

    @abstractmethod
    def read_source_file(self, file_path: Path) -> str:
        """
        Read the text from a source file.  Used to account for per-dataset encodings
        and other format differences.
        """
        raise NotImplementedError

    def _build(self):
        data_dir = self.data_dir()
        data_dir.mkdir(exist_ok=True, parents=True)

        self.download(data_dir)

        train_folder, test_folder = self.folders()

        self._load_folder(
            data_dir / train_folder,
            data_dir / self.TRAIN_X_FILE,
            data_dir / self.TRAIN_Y_FILE,
        )

        self._load_folder(
            data_dir / test_folder,
            data_dir / self.TEST_X_FILE,
            data_dir / self.TEST_Y_FILE,
        )

    def _load_folder(self, folder, X_file, y_file):
        """
        Combine a nested directory structure into a single output file.
        Assume the directory names are category names, and each file in
        the directory is a separate row assigned to that category.
        Write output to the given output file.
        """
        X = []
        y = []

        labels = self.labels()

        for category_dir in folder.iterdir():
            category_name = category_dir.name

            # Skip folders not named for labels
            if category_name not in labels:
                continue

            for data_file in category_dir.iterdir():
                X.append(self.read_source_file(data_file))
                y.append(category_name)

        X_file.write_text(NestedFileDataset.DELIMITER.join(X))
        y_file.write_text(NestedFileDataset.DELIMITER.join(y))

    def _read_data_file(self, filepath: Path) -> List[str]:
        return filepath.read_text().split(NestedFileDataset.DELIMITER)

    def _is_built(self) -> bool:
        data_files = (
            self.TRAIN_X_FILE,
            self.TRAIN_Y_FILE,
            self.TEST_X_FILE,
            self.TEST_Y_FILE,
        )
        return all((self.data_dir() / data_file).exists() for data_file in data_files)

    def X_train(self):
        return self._read_data_file(self.data_dir() / self.TRAIN_X_FILE)

    def y_train(self):
        return self._read_data_file(self.data_dir() / self.TRAIN_Y_FILE)

    def X_test(self):
        return self._read_data_file(self.data_dir() / self.TEST_X_FILE)

    def y_test(self):
        return self._read_data_file(self.data_dir() / self.TEST_Y_FILE)
