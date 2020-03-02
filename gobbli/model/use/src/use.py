import argparse
import json

import tensorflow_hub as hub


def read_texts(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return f.readlines()


def write_embeddings(embeddings, output_file):
    with open(output_file, "w") as f:
        for embedding in embeddings.tolist():
            json.dump(embedding, f)
            f.write("\n")


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

    args = parser.parse_args()

    embed = hub.load(args.module_dir)
    texts = read_texts(args.input_file)

    embeddings = embed(texts).numpy()
    write_embeddings(embeddings, args.output_file)
