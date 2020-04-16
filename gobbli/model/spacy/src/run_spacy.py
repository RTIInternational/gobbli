import argparse
import ast
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from spacy.gold import GoldParse
from spacy.util import minibatch

import torch
from spacy_transformers import TransformersLanguage
from spacy_transformers.util import PIPES, cyclic_triangular_rate


def is_transformer(nlp):
    """
    Determine whether the given spacy language instance is backed
    by a transformer model or a regular spaCy model.
    """
    return isinstance(nlp, TransformersLanguage)


def read_unique_labels(labels_path):
    """
    Read the list of unique labels from the given path.
    """
    return labels_path.read_text(encoding="utf-8").split("\n")


def read_data(file_path, has_labels):
    """
    Read some text data with optional labels from the given path.
    Return a 2-tuple of texts and labels (if any).
    """
    df = pd.read_csv(file_path, sep="\t", dtype="str", keep_default_na=False)
    X = df["Text"].tolist()
    y = None
    if has_labels:
        # Since spaCy defaults to multilabel paradigm, read in all data in that
        # format -- assume labels are a list
        y = df["Label"].apply(lambda labels: set(ast.literal_eval(labels))).tolist()
    return X, y


def spacy_format_labels(ys, labels):
    """Convert a list of labels to the format spaCy expects for model training."""
    return [{l: int(l in y) for l in labels} for y in ys]


def evaluate(tokenizer, nlp, valid_data, labels):
    """Evaluate model performance on a test dataset."""
    texts, cats = zip(*valid_data)

    golds = []
    # Use the model's ops module
    # to make sure this is compatible with GPU (cupy array)
    # or without (numpy array)
    scores = np.zeros((len(cats), len(labels)), dtype="f")
    if is_transformer(nlp):
        textcat = nlp.get_pipe(PIPES.textcat)
    else:
        textcat = nlp.get_pipe("textcat")
    scores = textcat.model.ops.asarray(scores)

    num_correct = 0
    for i, doc in enumerate(nlp.pipe(texts)):
        gold_cats = cats[i]["cats"]
        for j, (label, score) in enumerate(doc.cats.items()):
            if label not in gold_cats:
                raise ValueError(f"Prediction for unexpected label: {label}")

            scores[i, j] = score

            doc_prediction = score > 0.5
            if doc_prediction == bool(gold_cats[label]):
                num_correct += 1

        golds.append(GoldParse(doc, cats=gold_cats))

    accuracy = num_correct / ((len(texts) * len(labels)) + 1e-8)
    loss, _ = textcat.get_loss(texts, golds, scores)

    return accuracy, loss


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
    disabled_components,
    multilabel,
):
    """
    Train the TextCategorizer component of the passed pipeline on the given data.
    Return training/validation metrics and save the model to the specified output directory.
    Make sure to restore any disabled pipeline components before saving so we can reuse the
    saved checkpoint however we need to.
    """
    if is_transformer(nlp):
        textcat_pipe_name = PIPES.textcat
        textcat = nlp.create_pipe(
            textcat_pipe_name,
            config={
                "architecture": "softmax_last_hidden",
                "exclusive_classes": not multilabel,
                # We get an error about token_vector_width being unset if it isn't set
                # explicitly here.  We can't set it to an arbitrary value, either.  It must
                # be set based on the model
                "token_vector_width": nlp.get_pipe(PIPES.tok2vec).model.nO,
            },
        )
    else:
        textcat_pipe_name = "textcat"
        textcat = nlp.create_pipe(
            textcat_pipe_name,
            config={"exclusive_classes": not multilabel, "architecture": architecture},
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

    with nlp.disable_pipes(*disabled_components):
        if is_transformer(nlp):
            optimizer = nlp.resume_training()
            optimizer.alpha = 0.001
            optimizer.trf_weight_decay = 0.005
            optimizer.L2 = 0.0
            learn_rate = 2e-5
            learn_rates = cyclic_triangular_rate(
                learn_rate / 3, learn_rate * 3, 2 * len(train_data) // train_batch_size
            )
        else:
            optimizer = nlp.begin_training()
        for i in range(num_train_epochs):
            losses = {}
            random.shuffle(train_data)
            batches = minibatch(train_data, train_batch_size)
            for batch in batches:
                texts, annotations = zip(*batch)
                if is_transformer(nlp):
                    optimizer.trf_lr = next(learn_rates)

                nlp.update(
                    texts, annotations, sgd=optimizer, drop=dropout, losses=losses
                )

            with textcat.model.use_params(optimizer.averages):
                accuracy, valid_loss = evaluate(nlp.tokenizer, nlp, valid_data, labels)
                train_loss = losses[textcat_pipe_name]
                print(
                    f"Iter {i}\tTrain Loss: {train_loss:.3f}\tValid Loss: {valid_loss:.3f}\tAccuracy: {accuracy:.3f}"
                )

    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    with nlp.use_params(optimizer.averages):
        nlp.to_disk(checkpoint_dir)

    metrics = {
        "valid_accuracy": accuracy,
        "mean_train_loss": losses[textcat_pipe_name] / len(X_train),
        "mean_valid_loss": valid_loss / len(X_valid),
    }

    with open(output_dir / "valid_results.json", "w") as f:
        json.dump(metrics, f)


def predict(*, input_dir, output_dir, nlp, labels, disabled_components):
    """
    Generate predictions for the given dataset using the TextCategorizer component
    of the passed pipeline.
    """
    X_test, _ = read_data(input_dir / "test.tsv", False)

    pred_probas = []
    with nlp.disable_pipes(*disabled_components):
        for doc in nlp.pipe(X_test):
            pred_probas.append({label: doc.cats.get(label, 0.0) for label in labels})
    df = pd.DataFrame(pred_probas)
    df.to_csv(output_dir / "test_results.tsv", index=False, sep="\t")


def embed(*, input_dir, output_dir, nlp, embed_pooling, disabled_components):
    """
    Generate embeddings for the given dataset using the vectors from the
    passed pipeline.
    """
    embeddings = []
    X_embed, _ = read_data(input_dir / "input.tsv", False)

    with nlp.disable_pipes(*disabled_components):
        with open(output_dir / "embeddings.jsonl", "w") as f:
            for doc in nlp.pipe(X_embed):
                if embed_pooling == "mean":
                    row_json = {"embedding": doc.vector.tolist()}
                elif embed_pooling == "none":
                    embeddings = []
                    tokens = []
                    for tok in doc:
                        embeddings.append(tok.vector.tolist())
                        tokens.append(tok.text)
                    row_json = {"embedding": embeddings, "tokens": tokens}
                else:
                    raise ValueError(f"Unsupported pooling type: {embed_pooling}")

                f.write(f"{json.dumps(row_json)}\n")


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
        "--multilabel",
        action="store_true",
        help="If True, model will train in a multilabel context (i.e. it's allowed to make multiple "
        "class predictions for each observation).",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="If passed, use the full spaCy language pipeline (including tagging, "
        "parsing, and named entity recognition) for the TextCategorizer model used in "
        "training and prediction.  This makes training/prediction much slower but theoretically "
        "provides more information to the model.",
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
    parser.add_argument(
        "--embed-pooling",
        choices=["mean", "none"],
        default="mean",
        help="Pooling strategy to use for combining embeddings.  If None, embeddings for "
        "each token will be returned along with the token.  Ignored if method != 'embed'.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    using_gpu = spacy.prefer_gpu()
    if using_gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    device = "gpu" if using_gpu else "cpu"
    print(f"Using device: {device}")

    print("Initializing spaCy model...")
    print(f"  Language Model: {args.model}")

    nlp = spacy.load(args.model)
    if not is_transformer(nlp):
        print(f"  TextCategorizer Architecture: {args.architecture}")

    model_name = nlp.meta.get("name", "")

    print(f"Model '{model_name}' loaded.")

    disabled_components = set()

    if args.mode == "embed":
        # Don't need the text categorizer (if present) for embeddings
        for textcat_pipe in ("textcat", "trf_textcat"):
            if nlp.has_pipe(textcat_pipe):
                disabled_components.add(textcat_pipe)

        if model_name.endswith("sm"):
            # No vectors available for small models -- we need to enable the
            # other pipeline components to provide tensors, which aren't as good as vectors
            # but will suffice in a pinch
            pass
        else:
            # If vectors are available, disable everything, since we just want the vectors
            for component in ("tagger", "parser", "ner"):
                if nlp.has_pipe(component):
                    disabled_components.add(component)

    elif args.mode in ("train", "predict"):
        if args.full_pipeline:
            # Enable all parsing components to provide maximum information to the
            # text categorization model
            pass
        else:
            # Otherwise, disable everything that isn't part of the text categorizer model
            for component in ("tagger", "parser", "ner"):
                if nlp.has_pipe(component):
                    disabled_components.add(component)

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
            disabled_components=disabled_components,
            multilabel=args.multilabel,
        )

    elif args.mode == "predict":
        predict(
            input_dir=input_dir,
            output_dir=output_dir,
            nlp=nlp,
            labels=labels,
            disabled_components=disabled_components,
        )

    elif args.mode == "embed":
        embed(
            input_dir=input_dir,
            output_dir=output_dir,
            nlp=nlp,
            embed_pooling=args.embed_pooling,
            disabled_components=disabled_components,
        )

    else:
        raise ValueError(f"invalid mode: {args.mode}")
