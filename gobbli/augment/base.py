from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from gobbli.util import gobbli_dir


def augment_dir() -> Path:
    return gobbli_dir() / "augment"


class BaseAugment(ABC):
    """
    Base class for data augmentation methods.
    """

    @abstractmethod
    def augment(self, X: List[str], times: int = 5, p: float = 0.1) -> List[str]:
        """
        Return additional texts for each text in the passed array.

        Args:
          X: Input texts.
          times: How many texts to generate per text in the input.
          p: Probability of considering each token in the input for replacement.
            Note that some tokens aren't able to be replaced by a given augmentation
            method and will be ignored, so the actual proportion of replaced tokens
            in your input may be much lower than this number.
        Returns:
          Generated texts (length = ``times * len(X)``).
        """
        raise NotImplementedError

    @classmethod
    def data_dir(cls) -> Path:
        """
        Returns:
          The data directory used for this class of augmentation model.
        """
        return augment_dir() / cls.__name__
