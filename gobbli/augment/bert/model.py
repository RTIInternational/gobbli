from pathlib import Path
from typing import Any, Dict, List

from gobbli.augment.base import BaseAugment
from gobbli.docker import maybe_mount, run_container
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.util import assert_type, escape_line_delimited_texts


class BERTMaskedLM(BaseModel, BaseAugment):
    """
    BERT-based data augmenter.  Applies masked language modeling to generate
    predictions for missing tokens using a trained BERT model.
    """

    _BUILD_PATH = Path(__file__).parent

    _INPUT_FILE = "input.txt"
    _OUTPUT_FILE = "output.txt"

    _CONTAINER_CACHE_DIR = Path("/cache")

    def init(self, params: Dict[str, Any]):
        """
        See :meth:`gobbli.model.base.BaseModel.init`.

        BERTMaskedLM parameters:

        - ``bert_model`` (:obj:`str`): Name of a pretrained BERT model to use.
          See the `transformers <https://huggingface.co/transformers/pretrained_models.html>`__
          docs for supported values.
        - ``diversity``: 0 < diversity <= 1; determines the likelihood of selecting replacement words
          based on their predicted probability.
          At 1, the most probable words are most likely to be selected
          as replacements.  As diversity decreases, likelihood of selection becomes less
          dependent on predicted probability.
        - ``n_probable``: The number of probable tokens to consider for replacement.
        - ``batch_size``: Number of documents to run through the BERT model at once.
        """
        self.bert_model = "bert-base-uncased"
        self.diversity = 0.8
        self.batch_size = 32
        self.n_probable = 5

        for name, value in params.items():
            if name == "bert_model":
                self.bert_model = value
            elif name == "diversity":
                assert_type(name, value, float)
                if not 0 < value <= 1:
                    raise ValueError("diversity must be > 0 and <= 1")
                self.diversity = value
            elif name == "batch_size":
                assert_type(name, value, int)
                if value < 1:
                    raise ValueError("batch_size must be >= 1")
                self.batch_size = value
            elif name == "n_probable":
                assert_type(name, value, int)
                if value < 1:
                    raise ValueError("n_probable must be >= 1")
                self.n_probable = value
            else:
                raise ValueError(f"Unknown param '{name}'")

    @property
    def image_tag(self) -> str:
        """
        Returns:
          The Docker image tag to be used for the BERT container.
        """
        return f"gobbli-bert-maskedlm"

    def _build(self):
        self.docker_client.images.build(
            path=str(BERTMaskedLM._BUILD_PATH),
            tag=self.image_tag,
            **self._base_docker_build_kwargs,
        )

    def _write_input(self, X: List[str], context: ContainerTaskContext):
        """
        Write the user input to a file for the container to read.
        """
        input_path = context.host_input_dir / BERTMaskedLM._INPUT_FILE
        input_path.write_text(escape_line_delimited_texts(X))

    def _read_output(self, context: ContainerTaskContext) -> List[str]:
        """
        Read generated text output to a file from the container.
        """
        output_path = context.host_output_dir / BERTMaskedLM._OUTPUT_FILE
        return output_path.read_text().split("\n")

    @property
    def host_cache_dir(self):
        """
        Directory to be used for downloaded transformers files.
        Should be the same across all instances of the class, since these are
        generally static model weights/config files that can be reused.
        """
        cache_dir = BERTMaskedLM.model_class_dir() / "cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir

    def augment(self, X: List[str], times: int = 5, p: float = 0.1) -> List[str]:
        context = ContainerTaskContext(self.data_dir())

        self._write_input(X, context)

        # Determine which device to use for augmentation
        device = "cpu"
        if self.use_gpu:
            if self.nvidia_visible_devices == "all":
                device = "cuda"
            else:
                device_num = self.nvidia_visible_devices.split(",")[0]
                device = f"cuda:{device_num}"

        cmd = (
            "python3 augment_text.py"
            f" {context.container_input_dir / BERTMaskedLM._INPUT_FILE}"
            f" {context.container_output_dir / BERTMaskedLM._OUTPUT_FILE}"
            f" --probability {p}"
            f" --times {times}"
            f" --diversity {self.diversity}"
            f" --bert-model {self.bert_model}"
            f" --batch-size {self.batch_size}"
            f" --n-probable {self.n_probable}"
            f" --cache-dir {BERTMaskedLM._CONTAINER_CACHE_DIR}"
            f" --device {device}"
        )

        run_kwargs = self._base_docker_run_kwargs(context)

        maybe_mount(
            run_kwargs["volumes"],
            self.host_cache_dir,
            BERTMaskedLM._CONTAINER_CACHE_DIR,
        )

        run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        return self._read_output(context)
