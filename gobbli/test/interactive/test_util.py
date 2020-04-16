import inspect
from pathlib import Path

import pytest

from gobbli.interactive.util import read_data_file


@pytest.mark.parametrize(
    "filename,contents,multilabel,limit,expected",
    [
        # CSV, no text column
        ("test.csv", "a,b", False, None, ValueError),
        # CSV, no label column
        (
            "test.csv",
            "text,b\ntest1,1\ntest2,2",
            False,
            None,
            (["test1", "test2"], None),
        ),
        # CSV
        (
            "test.csv",
            "text,label\ntest1,1\ntest2,2",
            False,
            None,
            (["test1", "test2"], ["1", "2"]),
        ),
        # CSV, limit
        ("test.csv", "text,label\ntest1,1\ntest2,2", False, 1, (["test1"], ["1"])),
        # CSV, multilabel
        (
            "test.csv",
            'text,label\ntest1,"1,2"\ntest2,',
            True,
            None,
            (["test1", "test2"], [["1", "2"], []]),
        ),
        # TSV, no text column
        ("test.tsv", "a\tb", False, None, ValueError),
        # TSV, no label column
        (
            "test.tsv",
            "text\tb\ntest1\t1\ntest2\t2",
            False,
            None,
            (["test1", "test2"], None),
        ),
        # TSV
        (
            "test.tsv",
            "text\tlabel\ntest1\t1\ntest2\t2",
            False,
            None,
            (["test1", "test2"], ["1", "2"]),
        ),
        # TSV, limit
        ("test.tsv", "text\tlabel\ntest1\t1\ntest2\t2", False, 1, (["test1"], ["1"])),
        # TSV, multilabel
        (
            "test.tsv",
            "text\tlabel\ntest1\t1,2\ntest2\t",
            True,
            None,
            (["test1", "test2"], [["1", "2"], []]),
        ),
        # TXT
        ("test.txt", "test1\ntest2", False, None, (["test1", "test2"], None)),
        # TXT, limit
        ("test.txt", "test1\ntest2", False, 1, (["test1"], None)),
    ],
)
def test_read_data_file(tmpdir, filename, contents, multilabel, limit, expected):
    filepath = Path(tmpdir) / filename
    filepath.write_text(contents)

    if inspect.isclass(expected) and issubclass(expected, Exception):
        with pytest.raises(expected):
            read_data_file(filepath, multilabel, n_rows=limit)
    else:
        texts, labels = read_data_file(filepath, multilabel, n_rows=limit)
        expected_texts, expected_labels = expected

        assert texts == expected_texts
        assert labels == expected_labels
