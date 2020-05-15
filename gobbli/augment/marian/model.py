import warnings
from pathlib import Path
from typing import Any, Dict, List

from gobbli.augment.base import BaseAugment
from gobbli.docker import maybe_mount, run_container
from gobbli.model.base import BaseModel
from gobbli.model.context import ContainerTaskContext
from gobbli.util import assert_type, escape_line_delimited_texts


class MarianMT(BaseModel, BaseAugment):
    """
    Backtranslation-based data augmenter using the Marian Neural Machine Translation
    model.  Translates English text to one of several languages and back to obtain
    similar texts for training.
    """

    _BUILD_PATH = Path(__file__).parent

    _INPUT_FILE = "input.txt"
    _OUTPUT_FILE = "output.txt"

    _CONTAINER_CACHE_DIR = Path("/cache")

    MARIAN_MODEL = "Helsinki-NLP/opus-mt-en-ROMANCE"
    MARIAN_INVERSE_MODEL = "Helsinki-NLP/opus-mt-ROMANCE-en"

    # Hardcoded list based on the languages available in the en-ROMANCE model
    # https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE
    ALL_TARGET_LANGUAGES = set(
        (
            "fr",
            "fr_BE",
            "fr_CA",
            "fr_FR",
            "wa",
            "frp",
            "oc",
            "ca",
            "rm",
            "lld",
            "fur",
            "lij",
            "lmo",
            "es",
            "es_AR",
            "es_CL",
            "es_CO",
            "es_CR",
            "es_DO",
            "es_EC",
            "es_ES",
            "es_GT",
            "es_HN",
            "es_MX",
            "es_NI",
            "es_PA",
            "es_PE",
            "es_PR",
            "es_SV",
            "es_UY",
            "es_VE",
            "pt",
            "pt_br",
            "pt_BR",
            "pt_PT",
            "gl",
            "lad",
            "an",
            "mwl",
            "it",
            "it_IT",
            "co",
            "nap",
            "scn",
            "vec",
            "sc",
            "ro",
            "la",
        )
    )

    def init(self, params: Dict[str, Any]):
        """
        See :meth:`gobbli.model.base.BaseModel.init`.

        MarianMT parameters:

        - ``batch_size``: Number of documents to run through the Marian model at once.
        - ``target_languages``: List of target languages to translate texts to and back.
          One copy of each text will be generated for each target language.
          See :attr:`MarianMT.ALL_TARGET_LANGUAGES` for a full list of possible values.  See
          https://github.com/Helsinki-NLP/OPUS-MT-train/blame/198c779e91555594d76484109aaccee344b85aec/NOTES.md
          and https://github.com/Helsinki-NLP/OPUS-MT-train/blob/37a83a9eba4fdb73d0311356eadd9e0610139970/backtranslate/Makefile#L647
          for documentation about the abbreviations.
        """
        self.batch_size = 32
        self.target_languages = ["es", "fr", "it", "pt", "ro"]

        for name, value in params.items():
            if name == "batch_size":
                assert_type(name, value, int)
                if value < 1:
                    raise ValueError("batch_size must be >= 1")
                self.batch_size = value
            elif name == "target_languages":
                assert_type(name, value, list)
                for target in value:
                    if target not in self.ALL_TARGET_LANGUAGES:
                        raise ValueError(
                            f"invalid target language '{value}'. Valid values are "
                            f"{self.ALL_TARGET_LANGUAGES}"
                        )
                self.target_languages = value
            else:
                raise ValueError(f"Unknown praam '{name}'")

    @property
    def image_tag(self) -> str:
        """
        Returns:
          The Docker image tag to be used for the Marian container.
        """
        return f"gobbli-marian-nmt"

    def _build(self):
        self.docker_client.images.build(
            path=str(MarianMT._BUILD_PATH),
            tag=self.image_tag,
            **self._base_docker_build_kwargs,
        )

    def _write_input(self, X: List[str], context: ContainerTaskContext):
        """
        Write the user input to a file for the container to read.
        """
        input_path = context.host_input_dir / MarianMT._INPUT_FILE
        expanded_texts = []
        for language in self.target_languages:
            for text in X:
                expanded_texts.append(f">>{language}<< {text}")

        input_path.write_text(escape_line_delimited_texts(expanded_texts))

    def _read_output(self, context: ContainerTaskContext) -> List[str]:
        """
        Read generated text output to a file from the container.
        """
        output_path = context.host_output_dir / MarianMT._OUTPUT_FILE
        return output_path.read_text().split("\n")

    @property
    def host_cache_dir(self):
        """
        Directory to be used for downloaded transformers files.
        Should be the same across all instances of the class, since these are
        generally static model weights/config files that can be reused.
        """
        cache_dir = MarianMT.model_class_dir() / "cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir

    def augment(self, X: List[str], times: int = None, p: float = None) -> List[str]:
        if times is not None:
            warnings.warn(
                "MarianMT generates a number of times based on the target_languages parameter, "
                "so the 'times' parameter will be ignored."
            )
        if p is not None:
            warnings.warn(
                "MarianMT doesn't replace text at the token level, so the 'p' parameter "
                "will be ignored."
            )

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
            "python3 backtranslate_text.py"
            f" {context.container_input_dir / MarianMT._INPUT_FILE}"
            f" {context.container_output_dir / MarianMT._OUTPUT_FILE}"
            f" --batch-size {self.batch_size}"
            f" --cache-dir {MarianMT._CONTAINER_CACHE_DIR}"
            f" --device {device}"
            f" --marian-model {MarianMT.MARIAN_MODEL}"
            f" --marian-inverse-model {MarianMT.MARIAN_INVERSE_MODEL}"
        )

        run_kwargs = self._base_docker_run_kwargs(context)

        maybe_mount(
            run_kwargs["volumes"], self.host_cache_dir, MarianMT._CONTAINER_CACHE_DIR
        )

        run_container(
            self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
        )

        return self._read_output(context)
