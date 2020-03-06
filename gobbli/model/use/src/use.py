import argparse
import json

import tensorflow_hub as hub


def read_texts(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return f.readlines()


def make_batches(l, batch_size):
    for i in range(0, len(l), batch_size):
        yield l[i : i + batch_size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the file containing input texts, one per line.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to write computed embeddings to (JSON format).",
    )
    parser.add_argument(
        "--module-dir",
        required=True,
        help="Path to the downloaded/extracted TFHub Module for USE.",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Number of texts to embed at once. Default: %(default)s",
    )

    args = parser.parse_args()

    embed = hub.load(args.module_dir)
    texts = read_texts(args.input_file)

    with open(args.output_file, "w") as f:
        for batch in make_batches(texts, args.batch_size):
            embeddings = embed(batch).numpy()
            for embedding in embeddings.tolist():
                json.dump(embedding, f)
                f.write("\n")
