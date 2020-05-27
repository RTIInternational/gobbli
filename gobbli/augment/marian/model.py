import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    # Hardcoded list based on the languages available in various models
    # Ignore models that translate multiple languages because they have
    # a different API (texts require a >>lang_code<< prelude describing
    # which language to translate to)
    # https://huggingface.co/models?search=opus-mt-en&sort=alphabetical
    # The below mapping was reconstructed manually using various resources
    # including ISO language codes, Google Translate auto-detect,
    # and the JWS docs from Opus: http://opus.nlpl.eu/JW300.php (see languages.json)
    LANGUAGE_CODE_MAPPING = {
        "afrikaans": "af",
        "central-bikol": "bcl",
        "bemba": "bem",
        "berber": "ber",
        "bulgarian": "bg",
        "bislama": "bi",
        # BZS stands for Brazilian Sign Language, but the text
        # looks like Portugese, and there's no other model for Portugese
        "portugese": "bzs",
        "catalan": "ca",
        "cebuano": "ceb",
        "chuukese": "chk",
        "seychelles-creole": "crs",
        "czech": "cs",
        "welsh": "cy",
        "danish": "da",
        "german": "de",
        "ewe": "ee",
        "efik": "efi",
        "esperanto": "eo",
        "estonian": "et",
        "basque": "eu",
        "finnish": "fi",
        "fijian": "fj",
        "french": "fr",
        "irish": "ga",
        "ga": "gaa",
        "gilbertese": "gil",
        "galician": "gl",
        "gun": "guw",
        "manx": "gv",
        "hausa": "ha",
        "hiligaynon": "hil",
        "hiri-motu": "ho",
        "haitian": "ht",
        "hungarian": "hu",
        "indonesian": "id",
        "igbo": "ig",
        "iloko": "ilo",
        "icelandic": "is",
        "isoko": "iso",
        "italian": "it",
        "japanese": "jap",
        "kongo": "kg",
        "kuanyama": "kj",
        "kikaonde": "kqn",
        "kwangali": "kwn",
        "kikongo": "kwy",
        "luganda": "lg",
        "lingala": "ln",
        "silozi": "loz",
        "kiluba": "lu",
        "tshiluba": "lua",
        "luvale": "lue",
        "lunda": "lun",
        "luo": "luo",
        "mizo": "lus",
        "mauritian-creole": "mfe",
        "malagasy": "mg",
        "marshallese": "mh",
        "macedonian": "mk",
        "malayalam": "ml",
        "moore": "mos",
        "marathi": "mr",
        "maltese": "mt",
        "ndonga": "ng",
        "niuean": "niu",
        "dutch": "nl",
        "sepedi": "nso",
        "chichewa": "ny",
        "nyaneka": "nyk",
        "oromo": "om",
        "pangasinan": "pag",
        "papiamento": "pap",
        "solomon-islands-pidgin": "pis",
        "ponapean": "pon",
        "uruund": "rnd",
        "russian": "ru",
        "kirundi": "run",
        "kinyarwanda": "rw",
        "sango": "sg",
        "slovak": "sk",
        "samoan": "sm",
        "shona": "sn",
        "albanian": "sq",
        "swati": "ss",
        "sesotho-lesotho": "st",
        "swedish": "sv",
        "swahili-congo": "swc",
        "tigrinya": "ti",
        "tiv": "tiv",
        "tagalog": "tl",
        "otetela": "tll",
        "setswana": "tn",
        "tongan": "to",
        "chitonga": "toi",
        "tok-pisin": "tpi",
        "tsonga": "ts",
        "tuvaluan": "tvl",
        "ukrainian": "uk",
        "umbundu": "umb",
        "xhosa": "xh",
    }

    def init(self, params: Dict[str, Any]):
        """
        See :meth:`gobbli.model.base.BaseModel.init`.

        MarianMT parameters:

        - ``batch_size``: Number of documents to run through the Marian model at once.
        - ``target_languages``: List of target languages to translate texts to and back.
          See :attr:`MarianMT.ALL_TARGET_LANGUAGES` for a full list of possible values. You may
          only augment texts up to the number of languages specified, since each language
          will be used at most once.  So if you want to augment 5 times, you need to specify
          at least 5 languages when initializing the model.
        """
        self.batch_size = 32
        # Current default - top 5 lanugages in Wikipedia which are also available
        # in the list of target languages
        # https://en.wikipedia.org/wiki/List_of_Wikipedias#List
        self.target_languages = ["french", "german", "japanese", "russian", "italian"]

        for name, value in params.items():
            if name == "batch_size":
                assert_type(name, value, int)
                if value < 1:
                    raise ValueError("batch_size must be >= 1")
                self.batch_size = value
            elif name == "target_languages":
                assert_type(name, value, list)
                for target in value:
                    if target not in MarianMT.LANGUAGE_CODE_MAPPING:
                        raise ValueError(
                            f"invalid target language '{target}'. Valid values are "
                            f"{list(MarianMT.LANGUAGE_CODE_MAPPING.keys())}"
                        )
                self.target_languages = value
            else:
                raise ValueError(f"Unknown param '{name}'")

    @property
    def image_tag(self) -> str:
        """
        Returns:
          The Docker image tag to be used for the Marian container.
        """
        return f"gobbli-marian-nmt"

    @classmethod
    def marian_model(cls, language: str) -> str:
        """
        Returns:
          Name of the Marian MT model to use to translate English
          to the passed language.
        """
        return f"Helsinki-NLP/opus-mt-en-{cls.LANGUAGE_CODE_MAPPING[language]}"

    @classmethod
    def marian_inverse_model(cls, language: str) -> str:
        """
        Returns:
          Name of the Marian MT model to use to translate the passed language
          back to English.
        """
        return f"Helsinki-NLP/opus-mt-{cls.LANGUAGE_CODE_MAPPING[language]}-en"

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
        input_path.write_text(escape_line_delimited_texts(X))

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

    def augment(
        self, X: List[str], times: Optional[int] = None, p: float = None
    ) -> List[str]:
        if times is None:
            times = len(self.target_languages)

        if times > len(self.target_languages):
            raise ValueError(
                "MarianMT was asked to augment {len(times)} times but was only initialized with "
                "{len(self.target_languages)} target languages.  You must specify at least as "
                "many target languages as the number of times you'd like to augment."
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

        augmented_texts = []
        for i in range(times):
            language = self.target_languages[i]
            cmd = (
                "python3 backtranslate_text.py"
                f" {context.container_input_dir / MarianMT._INPUT_FILE}"
                f" {context.container_output_dir / MarianMT._OUTPUT_FILE}"
                f" --batch-size {self.batch_size}"
                f" --cache-dir {MarianMT._CONTAINER_CACHE_DIR}"
                f" --device {device}"
                f" --marian-model {MarianMT.marian_model(language)}"
                f" --marian-inverse-model {MarianMT.marian_inverse_model(language)}"
            )

            run_kwargs = self._base_docker_run_kwargs(context)

            maybe_mount(
                run_kwargs["volumes"],
                self.host_cache_dir,
                MarianMT._CONTAINER_CACHE_DIR,
            )

            run_container(
                self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
            )

            augmented_texts.extend(self._read_output(context))

        return augmented_texts
