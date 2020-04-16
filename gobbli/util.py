import distutils.file_util
import enum
import gzip
import io
import json
import logging
import os
import random
import shutil
import tarfile
import tempfile
import uuid
import warnings
import zipfile
from distutils.util import strtobool
from pathlib import Path
from typing import (
    Any,
    Container,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
    cast,
)

import humanize
import pandas as pd
import pkg_resources
import requests
from sklearn.preprocessing import MultiLabelBinarizer

LOGGER = logging.getLogger(__name__)


def default_gobbli_dir() -> Path:
    """
    Returns:
      The default directory to be used to store gobbli data if there's no user-specified
      default.
    """
    return Path.home() / ".gobbli"


def gobbli_dir() -> Path:
    """
    Returns:
      The directory used to store gobbli data.  Can be overridden using the `GOBBLI_DIR`
      environment variable.
    """
    gobbli_dir = Path(os.getenv("GOBBLI_DIR", str(default_gobbli_dir())))
    gobbli_dir.mkdir(parents=True, exist_ok=True)
    return gobbli_dir


def model_dir() -> Path:
    """
    Returns:
      The subdirectory storing all model data and task run input/output.
    """
    return gobbli_dir() / "model"


def gobbli_version() -> str:
    """
    Returns:
      The version of gobbli installed.
    """
    return pkg_resources.get_distribution("gobbli").version


def disk_usage() -> int:
    """
    Returns:
      Disk usage (in bytes) of all gobbli data in the current gobbli directory.
    """
    return sum(f.stat().st_size for f in gobbli_dir().glob("**/*") if f.is_file())


def human_disk_usage() -> str:
    """
    Returns:
      Human-readable string representing disk usage of all gobbli data in the current gobbli
      directory.
    """
    return humanize.naturalsize(disk_usage())


def cleanup(force: bool = False, full: bool = False):
    """
    Cleans up the gobbli directory, removing files and directories.  By default, removes
    only task input/output.  If the ``full`` argument is True, removes all data,
    including downloaded model weights and datasets.

    Args:
      force: If True, don't prompt for user confirmation before cleaning up.
      full: If True, remove all data (including downloaded model weights and datasets).
    """
    if not force:
        msg = "Cleanup will remove all task input/output, including trained models.  Are you sure? [Y/n]"
        confirmation = bool(strtobool(input(msg)))
        if not confirmation:
            return

        if full:
            full_msg = "Full cleanup will additionally remove all downloaded model weights; they'll need to be redownloaded to be used again.  Are you sure? [Y/n]"
            confirmation = bool(strtobool(input(full_msg)))
            if not confirmation:
                return

    if full:
        shutil.rmtree(gobbli_dir())
    else:
        base_model_dir = gobbli_dir() / "model"
        if base_model_dir.exists():
            for model_dir in (gobbli_dir() / "model").iterdir():
                for instance_dir in model_dir.iterdir():
                    # Special directories containing cached weights or models
                    # which are reusable -- don't delete them here
                    if instance_dir.name not in ("cache", "weights"):
                        shutil.rmtree(instance_dir)


def pred_prob_to_pred_label(y_pred_proba: pd.DataFrame) -> List[str]:
    """
    Convert a dataframe of predicted probabilities (shape (n_samples, n_classes)) to
    a list of predicted classes.
    """
    if len(y_pred_proba) == 0:
        return []

    return y_pred_proba.idxmax(axis=1).tolist()


def pred_prob_to_pred_multilabel(
    y_pred_proba: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    """
    Convert a dataframe of predicted probabilities (shape (n_samples, n_classes)) to
    a dataframe of predicted label indicators (shape (n_samples, n_classes)).
    """
    return y_pred_proba > threshold


def is_multilabel(y: Union[List[str], List[List[str]]]) -> bool:
    """
    Returns:
      True if the passed dataset is formatted as a multilabel problem, otherwise false
      (it's multiclass).
    """
    return len(y) > 0 and isinstance(y[0], list)


def as_multilabel(
    y: Union[List[str], List[List[str]]], is_multilabel: bool
) -> List[List[str]]:
    """
    Returns:
      Labels in multilabel format, validated against
      possible mismatch between the multilabel attribute and data format.
    """
    if is_multilabel:
        return cast(List[List[str]], y)
    return multiclass_to_multilabel_target(cast(List[str], y))


def as_multiclass(
    y: Union[List[str], List[List[str]]], is_multilabel: bool
) -> List[str]:
    """
    Returns:
      Labels in multiclass format, validated against
      possible mismatch between the multilabel attribute and data format.
    """
    if is_multilabel:
        raise ValueError("Multilabel training input can't be converted to multiclass.")
    return cast(List[str], y)


def multiclass_to_multilabel_target(y: List[str]) -> List[List[str]]:
    """
    Args:
      y: Multiclass-formatted target variable

    Returns:
      The multiclass-formatted variable generalized to a multilabel context
    """
    return [[l] for l in y]


def multilabel_to_indicator_df(y: List[List[str]], labels: List[str]) -> pd.DataFrame:
    """
    Convert a list of label lists to a 0/1 indicator dataframe.

    Args:
      y: List of label lists
      labels: List of all unique labels found in y

    Returns:
      The dataframe will have a column for each label and a row for each observation,
      with a 1 if the observation has that label or a 0 if not.
    """
    mlb = MultiLabelBinarizer(classes=labels)
    return pd.DataFrame(mlb.fit_transform(y), columns=mlb.classes_)


def collect_labels(y: Any) -> List[str]:
    """
    Collect the unique labels in the given target variable, which could be in
    multiclass or multilabel format.

    Args:
      y: Target variable

    Returns:
      An ordered list of all labels in y.
    """
    if is_multilabel(y):
        label_set: Set[str] = set()
        for labels in y:
            label_set.update(labels)
    else:
        label_set = set(y)

    return list(sorted(label_set))


def truncate_text(text: str, length: int) -> str:
    """
    Truncate the text to the given length.  Append an ellipsis to make it clear the
    text was truncated.

    Args:
      text: The text to truncate.
      length: Maximum length of the truncated text.

    Returns:
      The truncated text.
    """
    truncated = text[:length]
    if len(truncated) < len(text):
        truncated += "..."
    return truncated


def assert_type(name: str, val: Any, cls: Any):
    """
    Raise an informative error if the passed value isn't of the given type.

    Args:
      name: User-friendly name of the value to be printed in an exception if raised.
      val: The value to check type of.
      cls: The type or tuple of types to compare the value's type against.
    """
    if not isinstance(val, cls):
        raise TypeError(f"{name} must be of type(s) '{cls}'; got '{type(val)}'")


def assert_in(name: str, val: Any, container: Container):
    """
    Raise an informative error if the passed value isn't in the given container.

    Args:
      name: User-friendly name of the value to be printed in an exception if raised.
      val: The value to check for membership in the container.
      container: The container that should contain the value.
    """
    if val not in container:
        raise ValueError(f"invalid {name} '{val}'; valid values are {container}")


def generate_uuid() -> str:
    """
    Generate a universally unique ID to be used for randomly naming directories,
    models, tasks, etc.
    """
    return uuid.uuid4().hex


def download_dir() -> Path:
    """
    Returns:
      The directory used to store gobbli downloaded files.
    """
    download_dir = gobbli_dir() / "download"
    download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir


def escape_line_delimited_text(text: str) -> str:
    """
    Convert a single text possibly containing newlines and other troublesome whitespace
    into a string suitable for writing and reading to a file where newlines will
    divide up the texts.

    Args:
      text: The text to convert.

    Returns:
      The text with newlines and whitespace taken care of.
    """
    return text.replace("\n", " ").strip()


def escape_line_delimited_texts(texts: List[str]) -> str:
    """
    Convert a list of texts possibly containing newlines and other troublesome whitespace
    into a string suitable for writing and reading to a file where newlines
    will divide up the texts.

    Args:
      texts: The list of texts to convert.

    Returns:
      The newline-delimited string.
    """
    clean_texts = [escape_line_delimited_text(text) for text in texts]
    return "\n".join(clean_texts)


def is_dir_empty(dir_path: Path) -> bool:
    """
    Determine whether a given directory is empty.  Assumes the directory exists.

    Args:
      dir_path: The directory to check for emptiness.

    Returns:
      :obj:`True` if the directory is empty, otherwise :obj:`False`.
    """
    empty = True
    for child in dir_path.iterdir():
        empty = False
        break
    return empty


def write_metadata(metadata: Dict[str, Any], file_path: Path):
    """
    Write some JSON-formatted metadata to the given file path,
    formatted appropriately for human consumption.

    Args:
      metadata: The valid JSON metadata to write.  Nesting is allowed, but
        ensure all types in the structure are JSON-serializable.
      file_path: The path to write the metadata to.
    """
    with file_path.open("w") as f:
        json.dump(metadata, f, indent=4)


def read_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Read JSON-formatted metadata from the given file path.

    Args:
      file_path: The path to read JSON metadata from.

    Returns:
      The JSON object read from the file.
    """
    with file_path.open("r") as f:
        return json.load(f)


def format_duration(seconds: float) -> str:
    """
    Nicely format a given duration in seconds.

    Args:
      seconds: The duration to format, in seconds.

    Returns:
      The duration formatted as a string with unit of measurement appended.
    """
    return f"{seconds:.2f} sec"


def copy_file(src_path: Path, dest_path: Path) -> bool:
    """
    Copy a file from the source to destination.  Be smart -- only copy if
    the source is newer than the destination.

    Args:
      src_path: The path to the source file.
      dest_path: The path to the destination where the source should be copied.

    Returns:
      True if the file was copied, otherwise False if the destination was already
      up to date.
    """
    _, copied = distutils.file_util.copy_file(
        str(src_path), str(dest_path), update=True
    )
    return copied == 1


def download_file(url: str, filename: Optional[str] = None) -> Path:
    """
    Save a file in the gobbli download directory if it doesn't already exist there.
    Stream the download to avoid running out of memory.

    Args:
      url: URL for the file.
      filename: If passed, use this as the filename instead of the best-effort one
        determined from the URL.

    Returns:
      The path to the downloaded file.
    """
    if filename is None:
        # Kind of hacky... pull out the last path component as the filename and strip
        # a trailing query, if any
        local_filename = url.split("/")[-1]

        try:
            query_start_ndx = local_filename.index("?")
        except ValueError:
            query_start_ndx = -1

        if query_start_ndx != -1:
            local_filename = local_filename[:query_start_ndx]
    else:
        local_filename = filename

    local_filepath = download_dir() / local_filename

    if local_filepath.exists():
        LOGGER.debug(f"Download for URL '{url}' already exists at '{local_filepath}'")
        return local_filepath

    LOGGER.debug(f"Downloading URL '{url}' to '{local_filepath}'")
    try:
        with requests.get(url, stream=True) as r:
            with open(local_filepath, "wb") as f:
                shutil.copyfileobj(r.raw, f)
    except Exception as e:
        # Don't leave the file in a partially downloaded state
        try:
            local_filepath.unlink()
            LOGGER.debug(f"Removed partially downloaded file at '{local_filepath}'")
        except Exception:
            warnings.warn(
                "Failed to remove partially downloaded file at '{local_filepath}'."
            )
        raise e

    return local_filepath


def _extract_tar_junk_path(tarfile_obj: tarfile.TarFile, archive_extract_dir: Path):
    """
    Extract a tarfile while flattening any directory hierarchy
    in the archive.
    """
    for member in tarfile_obj.getmembers():
        if member.isdir():
            # Skip directories
            continue
        # Remove the directory hierarchy from the file
        member.name = Path(member.name).name
        output_file = archive_extract_dir / member.name
        LOGGER.debug(f"Extracting member '{member.name}' to '{output_file}'")
        tarfile_obj.extract(member, path=archive_extract_dir)


def _extract_zip_junk_path(zipfile_obj: zipfile.ZipFile, archive_extract_dir: Path):
    """
    Extract a zip file while flattening any directory hierarchy in the archive.
    """
    for member in zipfile_obj.infolist():
        if member.is_dir():
            # Skip directories
            continue
        member_name = Path(member.filename).name
        output_file = archive_extract_dir / member_name
        LOGGER.debug(f"Extracting member '{member_name}' to '{output_file}'")
        with zipfile_obj.open(member, "r") as f:
            with output_file.open("wb") as f_out:
                shutil.copyfileobj(f, f_out)


_SUPPORTED_ARCHIVE_EXTENSIONS = (".gz", ".zip")


def is_archive(filepath: Path) -> bool:
    """
    Args:
      filepath: Path to a file.

    Returns:
      Whether the file is an archive supported by :func:`extract_archive`.
    """
    for ext in _SUPPORTED_ARCHIVE_EXTENSIONS:
        if filepath.name.endswith(ext):
            return True
    return False


def extract_archive(
    archive_path: Path, archive_extract_dir: Path, junk_paths: bool = False
):
    """
    Extract an archive to the given directory.

    Args:
      archive_path: Path to the archive file.
      archive_extract_dir: Extract the archive to this directory.
      junk_paths: If True, disregard the archive's internal directory hierarchy
        and extract all files directly to the output directory.

    Returns:
      Path to the directory containing the extracted file contents.
    """
    LOGGER.debug(f"Extracting archive '{archive_path}'")
    archive_extract_dir.mkdir(exist_ok=True, parents=True)

    if archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as archive_tar:
            if junk_paths:
                _extract_tar_junk_path(archive_tar, archive_extract_dir)
            else:
                LOGGER.debug(f"Extracting all members to '{archive_extract_dir}'")
                archive_tar.extractall(archive_extract_dir)

    elif archive_path.name.endswith(".gz"):
        LOGGER.debug(f"Extracting gzipped file to '{archive_extract_dir}'")
        with gzip.open(archive_path, "rb") as archive_gz:
            # Strip the trailing '.gz' and use the original filename as the new filename
            with open(archive_extract_dir / archive_path.name[:-3], "wb") as f:
                shutil.copyfileobj(archive_gz, f)

    elif archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as archive_zip:
            if junk_paths:
                _extract_zip_junk_path(archive_zip, archive_extract_dir)
            else:
                LOGGER.debug(f"Extracting all members to '{archive_extract_dir}'")
                archive_zip.extractall(archive_extract_dir)
    else:
        raise ValueError(f"Unsupported archive file: {archive_path}")

    return archive_extract_dir


def download_archive(
    archive_url: str,
    archive_extract_dir: Path,
    junk_paths: bool = False,
    filename: Optional[str] = None,
) -> Path:
    """
    Save an archive in the given directory and extract its contents.  Automatically
    retry the download once if the file comes back corrupted (in case it's left over
    from a partial download that was cancelled before).

    Args:
      archive_url: URL for the archive.
      archive_extract_dir: Download the archive and extract it to this directory.
      junk_paths: If True, disregard the archive's internal directory hierarchy
        and extract all files directly to the output directory.
      filename: If given, store the downloaded file under this name instead of
        one automatically inferred from the URL.

    Returns:
      Path to the directory containing the extracted file contents.
    """
    download_path = download_file(archive_url, filename=filename)
    try:
        return extract_archive(
            download_path, archive_extract_dir, junk_paths=junk_paths
        )
    except (zipfile.BadZipFile, tarfile.ReadError, OSError, EOFError):
        LOGGER.warning(
            f"Downloaded archive at '{download_path}' is corrupted.  Retrying..."
        )
        download_path.unlink()
        download_path = download_file(archive_url, filename=filename)
        return extract_archive(
            download_path, archive_extract_dir, junk_paths=junk_paths
        )


def dir_to_blob(dir_path: Path) -> bytes:
    """
    Archive a directory and save it as a blob in-memory.
    Useful for storing a directory's contents in an in-memory object store.
    Use compression to reduce file size.  Extract with :func:`blob_to_dir`.

    Args:
      dir_path: Path to the directory to be archived.

    Returns:
      The compressed directory as a binary buffer.
    """
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as archive:
        # Set arcname=. to ensure the files will be extracted properly
        # using relative paths
        archive.add(str(dir_path), arcname=".", recursive=True)

    # Seek the beginning of the buffer so we read the whole thing
    blob.seek(0)
    return blob.getvalue()


def blob_to_dir(blob: bytes, dir_path: Path):
    """
    Extract the given blob (assumed to be a compressed directory
    created by :func:`dir_to_blob`) to the given directory.

    Args:
      blob: The compressed directory as a binary buffer.
      dir_path: Path to extract the directory to.
    """
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        archive.extractall(dir_path)


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def shuffle_together(l1: List[T1], l2: List[T2], seed: Optional[int] = None):
    """
    Shuffle two lists together, so their order is random but the individual
    elements still correspond.  Ex. shuffle a list of texts and a list of labels
    so the labels still correspond correctly to the texts.  The lists are shuffled in-place.

    The lists must be the same length for this to make sense.

    Args:
      l1: The first list to be shuffled.
      l2: The second list to be shuffled.
      seed: Seed for the random number generator, if any
    """
    if not len(l1) == len(l2):
        raise ValueError("Lists have unequal length.")
    if len(l1) == 0:
        return

    zipped = list(zip(l1, l2))
    if seed is not None:
        random.seed(seed)
    random.shuffle(zipped)
    l1[:], l2[:] = zip(*zipped)


def _train_sentencepiece(spm: Any, texts: List[str], model_path: Path, vocab_size: int):
    """
    Train a Sentencepiece model on the given data and save it to the given
    location.

    Args:
      spm: Imported sentencepiece module
      texts: Data to train on.
      model_path: Path to write the trained model to.
      vocab_size: Size of the vocabulary for the model to train.
    """
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write("\n".join(texts))
        f.flush()

        # log levels:
        # https://github.com/google/sentencepiece/blob/d4dd947fe71c4fa4ee24ad8297beee32887d8828/src/common.h#L132
        # Default is waaay too chatty
        spm.SentencePieceTrainer.train(
            f"--input={f.name} --model_prefix={model_path} --vocab_size={vocab_size} --minloglevel=2"
        )


@enum.unique
class TokenizeMethod(enum.Enum):
    """
    Enum describing the different canned tokenization methods gobbli supports.
    Processes requiring tokenization should generally allow a user to pass in
    a custom tokenization function if their needs aren't met by one of these.

    Attributes:
      SPLIT: Naive tokenization based on whitespace.  Probably only useful for testing.
        Tokens will be lowercased.
      SPACY: Simple tokenization using spaCy's English language model.
        Tokens will be lowercased, and non-alphabetic tokens will be filtered out.
      SENTENCEPIECE: `SentencePiece <https://github.com/google/sentencepiece>`__-based tokenization.
    """

    SPLIT = "split"
    SPACY = "spacy"
    SENTENCEPIECE = "sentencepiece"


def tokenize(
    method: TokenizeMethod,
    texts: List[str],
    model_path: Optional[Path] = None,
    vocab_size: int = 2000,
) -> List[List[str]]:
    """
    Tokenize a list of texts using a predefined method.

    Args:
      texts: Texts to tokenize.
      method: The type of tokenization to apply.
      model_path: Path to save a trained model to.  Required if the tokenization method
        requires training a model; otherwise ignored.  If the model doesn't exist, it will
        be trained; if it does, the trained model will be reused.
      vocab_size: Number of terms in the vocabulary for tokenization methods with a fixed
        vocabulary size. You may need to lower this if you get tokenization errors or
        raise it if your texts have a very diverse vocabulary.
    Returns:
      List of tokenized texts.
    """
    if method == TokenizeMethod.SPLIT:
        return [[tok.lower() for tok in text.split()] for text in texts]
    elif method == TokenizeMethod.SPACY:
        try:
            from spacy.lang.en import English
        except ImportError:
            raise ImportError(
                "spaCy tokenization method requires spaCy and an english language "
                "model to be installed."
            )
        nlp = English()
        tokenizer = nlp.Defaults.create_tokenizer(nlp)
        processed_texts = []
        for doc in tokenizer.pipe(texts):
            processed_texts.append([tok.lower_ for tok in doc if tok.is_alpha])
        return processed_texts
    elif method == TokenizeMethod.SENTENCEPIECE:
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "SentencePiece tokenization requires the sentencepiece module "
                "to be installed."
            )
        # Train only if the model file path doesn't already exist
        # If we weren't given a path to save to, just make a temp file which will
        # be discarded after the run
        sp = spm.SentencePieceProcessor()
        if model_path is None:
            with tempfile.TemporaryDirectory() as temp_model_dir:
                temp_model_path = Path(temp_model_dir) / "temp"
                _train_sentencepiece(spm, texts, temp_model_path, vocab_size)
                # Have to add the extension here for sentencepiece to find the model
                sp.load(f"{temp_model_path}.model")
                return [sp.encode_as_pieces(text) for text in texts]
        else:
            model_file_path = Path(f"{model_path}.model")

            if not model_file_path.exists():
                _train_sentencepiece(spm, texts, model_path, vocab_size)
            sp.load(str(model_file_path))
            return [sp.encode_as_pieces(text) for text in texts]
    else:
        raise ValueError(f"Unsupported tokenization method '{method}'.")


def detokenize(
    method: TokenizeMethod,
    all_tokens: Iterator[List[str]],
    model_path: Optional[Path] = None,
) -> List[str]:
    """
    Detokenize a nested list of tokens into a list of strings, assuming
    they were created using the given predefined method.

    Args:
      method: The type of tokenization to reverse.
      tokens: The nested list of tokens to detokenize.
      model_path: Path to load a trained model from.  Required if the tokenization method
        requires training a model; otherwise ignored.
    Returns:
      List of texts.
    """
    if method in (TokenizeMethod.SPLIT, TokenizeMethod.SPACY):
        # TODO can this be better?
        return [" ".join(tokens) for tokens in all_tokens]
    elif method == TokenizeMethod.SENTENCEPIECE:
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "SentencePiece detokenization requires the sentencepiece module "
                "to be installed."
            )
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_path}.model")
        return [sp.decode_pieces(tokens) for tokens in all_tokens]
    else:
        raise ValueError(f"Unsupported tokenization method '{method}'.")
