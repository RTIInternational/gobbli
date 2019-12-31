import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from spacy.util import minibatch


def read_unique_labels(labels_path):
    """
    Read the list of unique labels from the given path.
    """
    return labels_path.read_text().split("\n")


def read_data(file_path, has_labels):
    """
    Read some text data with optional labels from the given path.
    Return a 2-tuple of texts and labels (if any).
    """
    df = pd.read_csv(file_path, sep="\t", dtype="str", keep_default_na=False)
    X = df["Text"].tolist()
    y = None
    if has_labels:
        y = df["Label"].tolist()
    return X, y


def spacy_format_labels(ys, labels):
    """Convert a list of labels to the format spaCy expects for model training."""
    return [{l: int(y == l) for l in labels} for y in ys]


def evaluate(tokenizer, nlp, valid_data, labels):
    """Evaluate model performance on a test dataset."""
    # Use small nonzero numbers for fp/fn to avoid division by zero
    tp = 0.0
    fp = 1e-8
    fn = 1e-8
    tn = 0.0

    texts, cats = zip(*valid_data)
    scores = np.zeros((len(cats), len(labels)), dtype="f")
    docs = []
    for i, doc in enumerate(nlp.pipe(texts)):
        gold = cats[i]["cats"]
        for j, (label, score) in enumerate(doc.cats.items()):
            if label not in gold:
                raise ValueError(f"Prediction for unexpected label: {label}")

            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score > +0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1

            scores[i, j] = score
        docs.append(doc)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)

    # The textcat loss calculation ignores the first parameter (documents)
    # and wants the second parameter to be gold standard documents, so pass
    # the texts twice
    # Reference: https://github.com/explosion/spaCy/blob/db9257559c0642262a46d7acb7855e1e23b50e56/spacy/pipeline/pipes.pyx#L1005
    loss, _ = nlp.get_pipe("textcat").get_loss(texts, docs, scores)

    return accuracy, precision, recall, f_score, loss


def train(
    *,
    input_dir,
    output_dir,
    nlp,
    architecture,
    train_batch_size,
    num_train_epochs,
    labels,
    dropout,
):
    """
    Train the TextCategorizer component of the passed pipeline on the given data.
    Return training/validation metrics and save the model to the specified output directory.
    """
    textcat = nlp.create_pipe(
        "textcat", config={"exclusive_classes": True, "architecture": architecture}
    )
    nlp.add_pipe(textcat, last=True)

    for label in labels:
        textcat.add_label(label)

    X_train, y_train = read_data(input_dir / "train.tsv", True)
    X_valid, y_valid = read_data(input_dir / "dev.tsv", True)

    train_labels = spacy_format_labels(y_train, labels)
    valid_labels = spacy_format_labels(y_valid, labels)

    train_data = list(zip(X_train, [{"cats": cats} for cats in train_labels]))
    valid_data = list(zip(X_valid, [{"cats": cats} for cats in valid_labels]))

    optimizer = nlp.begin_training()
    for i in range(num_train_epochs):
        losses = {}
        random.shuffle(train_data)
        batches = minibatch(train_data, train_batch_size)
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=dropout, losses=losses)

        with textcat.model.use_params(optimizer.averages):
            accuracy, precision, recall, f_score, valid_loss = evaluate(
                nlp.tokenizer, nlp, valid_data, labels
            )
            print(
                f"Iter {i}\tTrain Loss: {losses['textcat']:.3f}\tValid Loss: {valid_loss:.3f}\tAccuracy: {accuracy:.3f}\tPrecision: {precision:.3f}\tRecall: {recall:.3f}\tF: {f_score:.3f}"
            )

    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    with nlp.use_params(optimizer.averages):
        nlp.to_disk(checkpoint_dir)

    metrics = {
        "valid_accuracy": accuracy,
        "mean_train_loss": losses["textcat"] / len(X_train),
        "mean_valid_loss": valid_loss / len(X_valid),
    }

    with open(output_dir / "valid_results.json", "w") as f:
        json.dump(metrics, f)


def predict(*, input_dir, output_dir, nlp, labels):
    """
    Generate predictions for the given dataset using the TextCategorizer component
    of the passed pipeline.
    """
    X_test, _ = read_data(input_dir / "test.tsv", False)

    pred_probas = []
    for doc in nlp.pipe(X_test):
        pred_probas.append({label: doc.cats.get(label, 0.0) for label in labels})
    df = pd.DataFrame(pred_probas)
    df.to_csv(output_dir / "test_results.tsv", index=False, sep="\t")


def embed(*, input_dir, output_dir, nlp, embed_batch_size):
    """
    Generate embeddings for the given dataset using the vectors from the
    passed pipeline.
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=["train", "predict", "embed"],
        help="Action to perform with the model. Train: read in a training "
        "and validation dataset, train on training and evaluate the model "
        "on validation dataset. Predict: output predictions on a test dataset. "
        "Embed: return embeddings for a test dataset.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input files.  The exact files needed vary "
        "depending on the mode.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing output files.  The exact files created vary "
        "depending on the mode.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory to use for caching spaCy downloads.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="SpaCy language model to use.  This could be "
        "either the name of a stock spaCy model (in which case it's assumed to already "
        "be installed via pip) or a path to a custom spaCy model on disk.",
    )
    parser.add_argument(
        "--architecture",
        required=True,
        help="Architecture for the spaCy TextCategorizer.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        help="Per-GPU batch size to use for training.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=32,
        help="Per-GPU batch size to use for embedding.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Number of epochs to run training on.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout proportion for training."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    using_gpu = spacy.prefer_gpu()

    print(f"Using GPU: {using_gpu}")

    print("Initializing spaCy model...")
    print(f"  Language Model: {args.model}")
    print(f"  TextCategorizer Architecture: {args.architecture}")

    nlp = spacy.load(args.model, disable=["tagger", "parser", "ner"])

    # We need a list of labels for training or prediction so we know what the
    # output shape of the model should be
    labels = None
    if args.mode in ("train", "predict"):
        labels_file = input_dir / "labels.tsv"
        labels = read_unique_labels(input_dir / "labels.tsv")
        num_labels = len(labels)
        print(f"Inferred number of labels: {num_labels}")

    if args.mode == "train":
        train(
            input_dir=input_dir,
            output_dir=output_dir,
            nlp=nlp,
            architecture=args.architecture,
            labels=labels,
            train_batch_size=args.train_batch_size,
            num_train_epochs=args.num_train_epochs,
            dropout=args.dropout,
        )

    elif args.mode == "predict":
        predict(input_dir=input_dir, output_dir=output_dir, nlp=nlp, labels=labels)

    elif args.mode == "embed":
        embed(
            input_dir=input_dir,
            output_dir=output_dir,
            nlp=nlp,
            embed_batch_size=args.embed_batch_size,
        )

    else:
        raise ValueError(f"invalid mode: {args.mode}")
