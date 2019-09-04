import json
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np

import gobbli.io
from gobbli.docker import run_container
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.model.mixin import EmbedMixin
from gobbli.util import assert_in, download_archive, escape_line_delimited_texts


def _read_embeddings(output_file: Path) -> np.ndarray:
    embeddings = []
    with open(output_file, "r") as f:
        for line in f:
            embeddings.append(json.loads(line))
    return np.vstack(embeddings)


USE_MODEL_ARCHIVES = {
    "universal-sentence-encoder": "https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed",
    "universal-sentence-encoder-large": "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed",
}
"""
A mapping from model names to TFHub URLs.
"universal-sentence-encoder" is a safe default for most situations.
Larger models require more time and GPU memory to run.
"""


class USE(BaseModel, EmbedMixin):
    """
    Wrapper for Universal Sentence Encoder embeddings:
    https://tfhub.dev/google/universal-sentence-encoder/2
    """

    _BUILD_PATH = Path(__file__).parent

    _INPUT_FILE = "input.txt"
    _OUTPUT_FILE = "output.jsonl"

    def init(self, params: Dict[str, Any]):
        """
        See :meth:`gobbli.model.base.BaseModel.init`.

        USE parameters:

        - ``use_model`` (:obj:`str`): Name of a USE model to use.
          See :obj:`USE_MODEL_ARCHIVES` for a listing of available USE models.
        """
        self.use_model = "universal-sentence-encoder"

        for name, value in params.items():
            if name == "use_model":
                assert_in(name, value, set(USE_MODEL_ARCHIVES.keys()))
                self.use_model = value
            else:
                raise ValueError(f"Unknown param '{name}'")

    @property
    def image_tag(self) -> str:
        """
        Returns:
          The Docker image tag to be used for the USE container.
        """
        device = "gpu" if self.use_gpu else "cpu"
        return f"gobbli-use-embeddings-{device}"

    def _build(self):
        # Download data if we don't already have it
        if not self.weights_dir.exists():
            self.weights_dir.mkdir(parents=True)
            try:
                self.logger.info("Downloading model.")
                download_archive(
                    USE_MODEL_ARCHIVES[self.use_model],
                    self.weights_dir,
                    filename=f"{self.use_model}.tar.gz",
                )
                self.logger.info("Weights downloaded.")
            except Exception:
                # Don't leave the weights directory in a partially downloaded state
                if self.weights_dir.exists():
                    shutil.rmtree(self.weights_dir)
                raise

        # Build the docker image
        self.docker_client.images.build(
            path=str(USE._BUILD_PATH),
            tag=self.image_tag,
            **self._base_docker_build_kwargs,
        )

    def _embed(
        self, embed_input: gobbli.io.EmbedInput, context: ContainerTaskContext
    ) -> gobbli.io.EmbedOutput:

        if embed_input.pooling == gobbli.io.EmbedPooling.NONE:
            raise ValueError(
                "Universal Sentence Encoder does sentence encoding, so pooling is required."
            )

        (context.host_input_dir / USE._INPUT_FILE).write_text(
            escape_line_delimited_texts(embed_input.X)
        )

        cmd = (
            "python use.py"
            f" --input-file={context.container_input_dir / USE._INPUT_FILE}"
            f" --output-file={context.container_output_dir / USE._OUTPUT_FILE}"
            f" --module-dir={BaseModel._CONTAINER_WEIGHTS_PATH}"
        )

        container_logs = run_container(
            self.docker_client,
            self.image_tag,
            cmd,
            self.logger,
            **self._base_docker_run_kwargs(context),
        )

        return gobbli.io.EmbedOutput(
            X_embedded=_read_embeddings(context.host_output_dir / USE._OUTPUT_FILE),
            _console_output=container_logs,
        )
