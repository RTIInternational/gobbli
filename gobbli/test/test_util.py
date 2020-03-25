import gzip
import io
import tarfile
import zipfile
from pathlib import Path
from typing import List

import pytest

from gobbli.util import (
    TokenizeMethod,
    blob_to_dir,
    detokenize,
    dir_to_blob,
    extract_archive,
    is_archive,
    shuffle_together,
    tokenize,
)


def make_zip(tmpdir: Path, relative_paths: List[Path]) -> Path:
    """
    Make a zip archive from a list of relative paths.
    Create empty files at each path and add them to the archive.
    """
    zip_path = tmpdir / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for relative_path in relative_paths:
            full_path = tmpdir / relative_path
            full_path.parent.mkdir(exist_ok=True, parents=True)
            full_path.touch()
            z.write(full_path, arcname=relative_path)

    return zip_path


def make_tar_gz(tmpdir: Path, relative_paths: List[Path]) -> Path:
    """
    Make a .tar.gz archive from a list of relative paths.
    Create empty files at each path and add them to the archive.
    """
    tar_path = tmpdir / "test.tar.gz"
    with tarfile.open(tar_path, "w:gz") as z:
        for relative_path in relative_paths:
            full_path = tmpdir / relative_path
            full_path.parent.mkdir(exist_ok=True, parents=True)
            full_path.touch()
            z.add(str(full_path), arcname=str(relative_path), recursive=False)

    return tar_path


def make_gz(tmpdir: Path, name: str) -> Path:
    """
    Create a gzip-compressed file with the given name under the given temp directory.
    Return the path to the compressed file.
    """
    gzip_path = tmpdir / f"{name}.gz"
    with gzip.open(gzip_path, "wb") as z:
        z.write(b"Test")

    return gzip_path


TEST_ARCHIVE_DATA = ["./a", "./b/c"]


@pytest.mark.parametrize(
    "archive_func,junk,expected_paths",
    [
        (make_zip, False, [Path("a"), Path("b") / "c"]),
        (make_zip, True, [Path("a"), Path("c")]),
        (make_tar_gz, False, [Path("a"), Path("b") / "c"]),
        (make_tar_gz, True, [Path("a"), Path("c")]),
    ],
)
def test_extract_archive(tmpdir, archive_func, junk, expected_paths):
    tmpdir_path = Path(tmpdir)
    archive_path = archive_func(tmpdir_path, TEST_ARCHIVE_DATA)

    archive_extract_dir = tmpdir_path / "extract"
    extract_archive(archive_path, archive_extract_dir, junk_paths=junk)

    for relative_path in expected_paths:
        assert (archive_extract_dir / relative_path).exists()


def test_extract_gz(tmpdir):
    tmpdir_path = Path(tmpdir)
    filename = "test.txt"
    archive_path = make_gz(tmpdir_path, "test.txt")

    archive_extract_dir = tmpdir_path / "extract"
    extract_archive(archive_path, archive_extract_dir)

    assert (archive_extract_dir / filename).exists()


@pytest.mark.parametrize(
    "name,expected_is_archive",
    [
        ("test.tar.gz", True),
        ("test.gz", True),
        ("test.txt.gz", True),
        ("test.zip", True),
        ("test.xz", False),
        ("test.txt", False),
        ("test.vec", False),
        ("test.bin", False),
    ],
)
def test_is_archive(name, expected_is_archive):
    assert is_archive(Path(name)) == expected_is_archive


def test_dir_to_blob(tmpdir):
    test_dir = Path(tmpdir) / "test"
    test_dir.mkdir()
    test_file_name = "test.txt"
    test_file = test_dir / test_file_name
    file_contents = "test"
    test_file.write_text(file_contents)

    blob = dir_to_blob(test_dir)
    fileobj = io.BytesIO(blob)
    fileobj.seek(0)
    extract_path = test_dir / "test2"
    with tarfile.open(fileobj=fileobj, mode="r:gz") as archive:
        archive.extractall(extract_path)

    extracted_file = extract_path / test_file_name
    assert extracted_file.exists()
    assert extracted_file.read_text() == file_contents


def test_blob_to_dir(tmpdir):
    test_dir = Path(tmpdir) / "test"
    test_dir.mkdir()
    test_file_name = "test.txt"
    test_file = test_dir / test_file_name
    file_contents = "test"
    test_file.write_text(file_contents)

    blob = dir_to_blob(test_dir)
    extract_path = test_dir / "test2"
    blob_to_dir(blob, extract_path)

    extracted_file = extract_path / test_file_name
    assert extracted_file.exists()
    assert extracted_file.read_text() == file_contents


@pytest.mark.parametrize(
    "l1,l2,err",
    [
        ([], [], None),
        (["a"], [1], None),
        (["a", "b"], [1], ValueError),
        (["a", "b"], [1, 2], None),
        (["a", "b", "c"], [1, 2, 3], None),
        (["a", "b", "c", "d"], [1, 2, 3, 4], None),
    ],
)
def test_shuffle_together(l1, l2, err):
    seed = 1

    if err is not None:
        with pytest.raises(err):
            shuffle_together(l1, l2, seed=seed)
    else:
        original_rows = set(zip(l1, l2))

        shuffle_together(l1, l2, seed=seed)
        for row in zip(l1, l2):
            assert tuple(row) in original_rows


@pytest.mark.parametrize(
    "text,tokens",
    [
        ("This is a test.", ["this", "is", "a", "test."]),
        ("Two  spaces", ["two", "spaces"]),
        ("Hyphenated-word", ["hyphenated-word"]),
        ("Numbers 1 and 2", ["numbers", "1", "and", "2"]),
    ],
)
def test_tokenize_split(text, tokens):
    # Whitespace tokenization just splits on whitespace
    assert tokenize(TokenizeMethod.SPLIT, [text]) == [tokens]


@pytest.mark.parametrize(
    "text,tokens",
    [
        ("This is a test.", ["this", "is", "a", "test"]),
        ("Two  spaces", ["two", "spaces"]),
        ("Hyphenated-word", ["hyphenated", "word"]),
        ("Numbers 1 and 2", ["numbers", "and"]),
    ],
)
def test_tokenize_spacy(text, tokens):
    # Spacy tokenization lowercases and removes non-alphabetic tokens
    assert tokenize(TokenizeMethod.SPACY, [text]) == [tokens]


@pytest.mark.parametrize(
    "tokenize_method", [TokenizeMethod.SPLIT, TokenizeMethod.SPACY]
)
@pytest.mark.parametrize(
    "tokens,text",
    [
        (["this", "is", "a", "test"], "this is a test"),
        (["hyphenated-word"], "hyphenated-word"),
        (["try", ",", "punctuation", "."], "try , punctuation ."),
    ],
)
def test_detokenize_split_spacy(text, tokens, tokenize_method):
    assert detokenize(tokenize_method, [tokens]) == [text]


@pytest.mark.parametrize("model_path", [Path("spm"), None])
def test_tokenize_detokenize_sentencepiece(tmpdir, model_path):
    texts = ["a b c", "a ab c", "a b ac"]

    # Model should be trained
    if model_path is not None:
        model_path = Path(tmpdir) / model_path
    tokens = tokenize(
        TokenizeMethod.SENTENCEPIECE, texts, model_path=model_path, vocab_size=7
    )

    # Control sequence indicating whitespace
    _ = "▁"
    expected_tokens = [
        [_, "a", _, "b", _, "c"],
        [_, "a", _, "a", "b", _, "c"],
        [_, "a", _, "b", _, "a", "c"],
    ]
    assert tokens == expected_tokens

    # Can't detokenize if we didn't give a persistent model path to the tokenize
    # function
    if model_path is not None:
        assert detokenize(TokenizeMethod.SENTENCEPIECE, tokens, model_path) == texts

        # Previously should be reused with the old vocab size, and a new model
        # shouldn't be trained
        tokens = tokenize(TokenizeMethod.SENTENCEPIECE, texts, model_path=model_path)
        assert tokens == expected_tokens
