import argparse

from transformers import MarianMTModel, MarianTokenizer


def batch_list(l, batch_size):
    for i in range(0, len(l), batch_size):
        yield l[i : i + batch_size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file",
        help="Input file containing line-delimited texts. "
        "Note multi-language Marian models require each text to be prefixed with '>>id<<', "
        "where 'id' is the ID of a target language for translation.  This script assumes "
        "the >>id<< prefix is correctly placed (or not) depending on the type of model.",
    )
    parser.add_argument(
        "output_file",
        help="Output file, where line-delimited generated texts will be written.",
    )
    parser.add_argument(
        "--marian-model",
        help="Marian model to use. "
        "Can be one of the pretrained names supported by transformers, in which case "
        "the pretrained weights will be downloaded. "
        "Anything else supported by transformers should work as well. "
        "Note an inverse model is required, so a custom model will need a corresponding "
        "inverse to translate in the other direction.",
        required=True,
    )
    parser.add_argument(
        "--marian-inverse-model",
        help="Inverse Marian model to use. "
        "Should be the opposite of --marian-model.  Ex: "
        "--marian-model=Helsinki-NLP/opus-mt-en-ROMANCE "
        "--marian-inverse-model=Helsinki-NLP/opus-mt-ROMANCE-en",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        help="Number of documents to run through the model at once.",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory to use as the cache for transformers downloads.",
        default=None,
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch name for the device to use when running the model. Default: %(default)s",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    tokenizer = MarianTokenizer.from_pretrained(
        args.marian_model, cache_dir=args.cache_dir
    )
    if tokenizer is None:
        raise ValueError("Failed to acquire tokenizer")
    model = MarianMTModel.from_pretrained(args.marian_model, cache_dir=args.cache_dir)
    if model is None:
        raise ValueError("Failed to acquire model")

    model = model.to(args.device)
    model.eval()

    inv_tokenizer = MarianTokenizer.from_pretrained(
        args.marian_inverse_model, cache_dir=args.cache_dir
    )
    if inv_tokenizer is None:
        raise ValueError("Failed to acquire inverse tokenizer")
    inv_model = MarianMTModel.from_pretrained(
        args.marian_inverse_model, cache_dir=args.cache_dir
    )
    if inv_model is None:
        raise ValueError("Failed to acquire inverse model")

    inv_model = inv_model.to(args.device)
    inv_model.eval()

    with open(args.input_file, "r", encoding="utf-8") as f_in:
        with open(args.output_file, "w", encoding="utf-8") as f_out:
            batches = list(batch_list(f_in.readlines(), args.batch_size))
            for batch_id, batch in enumerate(batches):
                # Explicit max length needed to make the tokenizer cut off at the max length
                # Otherwise we get potential index errors running the texts through the models
                translated = model.generate(
                    **tokenizer.prepare_translation_batch(
                        batch, max_length=tokenizer.model_max_length
                    ).to(args.device)
                )
                translated_texts = [
                    tokenizer.decode(t, skip_special_tokens=True) for t in translated
                ]

                backtranslated = inv_model.generate(
                    **inv_tokenizer.prepare_translation_batch(
                        translated_texts, max_length=inv_tokenizer.model_max_length
                    ).to(args.device)
                )
                backtranslated_texts = [
                    inv_tokenizer.decode(t, skip_special_tokens=True)
                    for t in backtranslated
                ]
                # Write text to the output file
                # Escape any embedded newlines
                # Make sure not to write an extra newline after the last row
                for i, row in enumerate(backtranslated_texts):
                    f_out.write(row.replace("\n", " "))

                    # Write a newline except for the last row on the last
                    # time through.  We'd get an empty line at the end of the
                    # file otherwise
                    if not (
                        i == len(translated_texts) - 1 and batch_id == len(batches) - 1
                    ):
                        f_out.write("\n")
