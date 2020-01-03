import csv
import itertools
from pathlib import Path
from typing import List, Optional, Tuple


def _read_delimited(
    data_file: Path, delimiter: str, n_rows: Optional[int] = None
) -> Tuple[List[str], Optional[List[str]]]:
    """
    Read up to n_rows lines from the given delimited text file and return lists
    of the texts and labels.  Texts must be stored in a column named "text", and
    labels (if any) must be stored in a column named "label".

    Args:
      data_file: Data file containing one text per line.
      delimiter: Field delimiter for the data file.
      n_rows: The maximum number of rows to read.

    Returns:
      2-tuple: list of read texts and corresponding list of read labels.
    """
    with open(data_file, "r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = set(reader.fieldnames)

        if "text" not in fieldnames:
            raise ValueError("Delimited text file doesn't contain a 'text' column.")
        has_labels = "label" in fieldnames

        rows = list(itertools.islice(reader, n_rows))

    texts: List[str] = []
    labels: List[str] = []

    for row in rows:
        texts.append(row["text"])
        if has_labels:
            labels.append(row["label"])

    return texts, labels if has_labels else None


def _read_lines(data_file: Path, n_rows: Optional[int] = None) -> List[str]:
    """
    Read up to n_rows lines from the given text file and return them in a list.

    Args:
      data_file: Data file containing one text per line.
      n_rows: The maximum number of rows to read.

    Returns:
      List of read lines.
    """
    with open(data_file, "r") as f:
        return list(itertools.islice((l.strip() for l in f), n_rows))


def read_data(
    data_file: Path, n_rows: Optional[int] = None
) -> Tuple[List[str], Optional[List[str]]]:
    """
    Read data to explore from a file.  Rows may be sampled using the n_rows argument.

    Args:
      data_file: Path to a data file to read.
      n_rows: The maximum number of rows to read.

    Returns:
      2-tuple: list of read texts and a list of read labels (if any)
    """
    extension = data_file.suffix
    if extension == ".tsv":
        texts, labels = _read_delimited(data_file, "\t", n_rows=n_rows)
    elif extension == ".csv":
        texts, labels = _read_delimited(data_file, ",", n_rows=n_rows)
    elif extension == ".txt":
        labels = None
        texts = _read_lines(data_file, n_rows=n_rows)
    else:
        raise ValueError(f"Data file extension '{extension}' is unsupported.")

    return texts, labels
