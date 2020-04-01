import argparse
import ast
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup


def read_unique_labels(labels_path):
    """
    Read the list of unique labels from the given path.
    """
    return labels_path.read_text(encoding="utf-8").split("\n")


def read_data(file_path, labels, multilabel):
    """
    Read some text data with optional labels from the given path.
    Return a 2-tuple of texts and labels (if any).
    """
    df = pd.read_csv(file_path, sep="\t", dtype="str", keep_default_na=False)
    X = df["Text"].tolist()
    y = None
    if labels is not None:
        if multilabel:
            y = (
                df["Label"]
                .apply(
                    lambda str_labels: [
                        1 if l in set(ast.literal_eval(str_labels)) else 0
                        for l in labels
                    ]
                )
                .tolist()
            )
        else:
            y = df["Label"].tolist()
    return X, y


def batch_gen(l, batch_size):
    for i in range(0, len(l), batch_size):
        yield l[i : i + batch_size]


def encode_batch(batch, tokenizer, max_seq_length, return_tokens=False):
    """
    Tokenize and encode the given input; stack it in a batch with fixed length
    according to the longest sequence in the batch.

    Return a tuple.  If asked to return tokens, return the encoded batch and a nested
    list of the tokenized texts.  Otherwise, just return the batch.
    """
    # Truncate input based on the sequence length
    # before stacking as a batch
    # Also return a boolean array indicating which tokens are masked
    encoded_texts = []

    for text in batch:
        encoded_text = torch.tensor(tokenizer.encode(text))[:max_seq_length]
        encoded_texts.append(encoded_text)

    encoded_batch = torch.nn.utils.rnn.pad_sequence(encoded_texts, batch_first=True)

    if return_tokens:
        tokens = [tokenizer.tokenize(text) for text in batch]
        return encoded_batch, tokens
    else:
        return (encoded_batch,)


def encode_labels(y, labels, multilabel):
    """
    Convert string labels to autoincrementing integers.
    """
    if multilabel:
        # Float type needed for proper loss computation
        return torch.as_tensor(y).float()
    else:
        mapping = {l: i for i, l in enumerate(labels)}
        return torch.as_tensor([mapping[l] for l in y])


def decode_labels(y_encoded, labels):
    """
    Convert autoincrementing integers back to strings.
    """
    return [labels[i] for i in y_encoded]


def tsv_to_encoded_batches(
    input_path,
    tokenizer,
    labels,
    batch_size,
    max_seq_length,
    multilabel,
    return_tokens=False,
):
    """
    Convert data in a csv file with optional labels to a list
    of encoded batches.  Use a generator to guard against memory errors.

    If `return_tokens` is True, add a nested list containing the tokenized
    text for each row.

    Just return encoded texts and optional tokens if no labels,
    else also return encoded labels.
    """
    has_labels = labels is not None
    X, y = read_data(input_path, labels, multilabel)

    if y is None:
        y_batch_gen = itertools.repeat(None)
    else:
        y_batch_gen = batch_gen(y, batch_size)

    for X_batch, y_batch in zip(batch_gen(X, batch_size), y_batch_gen):
        X_encoded_batch = encode_batch(
            X_batch, tokenizer, max_seq_length, return_tokens=return_tokens
        )

        if has_labels:
            y_encoded_batch = encode_labels(y_batch, labels, multilabel)
            yield (*X_encoded_batch, y_encoded_batch)
        else:
            yield X_encoded_batch


def num_batches(data_path, batch_size):
    """
    Count the number of batches contained in the given TSV data file assuming
    the given batch size.
    """
    X, _ = read_data(data_path, False, False)
    return (len(X) - 1) // batch_size + 1


def get_loss_preds(model, X, y, num_labels, multilabel):
    """
    Return the loss and predicted class(es) for each observation in the passed
    dataset.  The multilabel boolean determines how loss will be calculated and
    whether multiple predicted classes can be returned for each input observation.
    """
    if multilabel:
        outputs = model(input_ids=X)
        logits = outputs[0].view(-1, num_labels)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        preds = F.sigmoid(logits.detach())
    else:
        outputs = model(input_ids=X, labels=y)
        loss = outputs[0]
        logits = outputs[1]
        # Keep preds on the device rather than moving to CPU, since we'll compare to y,
        # which is on the device
        preds = torch.argmax(logits.detach(), 1)

    return loss, preds


def train(
    *,
    input_dir,
    output_dir,
    model,
    config,
    tokenizer,
    train_batch_size,
    valid_batch_size,
    num_train_epochs,
    device,
    n_gpu,
    max_seq_length,
    labels,
    lr,
    adam_eps,
    gradient_accumulation_steps,
    multilabel,
):
    """
    Train the passed model on the given data.  Return training/validation metrics
    and save the model to the specified output directory.
    """
    # Prepare optimizer and scheduler
    # Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_squad.py
    optimizer = AdamW(model.parameters(), lr=lr, eps=adam_eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_epochs
        * num_batches(input_dir / "train.tsv", train_batch_size)
        // gradient_accumulation_steps,
    )

    model.zero_grad()
    num_labels = len(labels)

    for epoch in range(num_train_epochs):
        print(f"Start epoch {epoch}")
        model.train()
        train_loss = 0.0
        train_count = 0

        for i, (X, y) in enumerate(
            tsv_to_encoded_batches(
                input_dir / "train.tsv",
                tokenizer,
                labels,
                train_batch_size,
                max_seq_length,
                multilabel,
            )
        ):
            X = X.to(device)
            y = y.to(device)

            loss, _ = get_loss_preds(model, X, y, num_labels, multilabel)

            if gradient_accumulation_steps > 1:
                loss /= gradient_accumulation_steps

            if n_gpu > 1:
                loss = loss.mean()  # average loss on parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            train_loss += loss.detach().item()
            train_count += X.size(0)

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        print(f"Epoch {epoch} train loss: {train_loss / train_count}")

        model.eval()
        valid_loss = 0.0
        valid_count = 0
        valid_correct = 0

        for X, y in tsv_to_encoded_batches(
            input_dir / "dev.tsv",
            tokenizer,
            labels,
            valid_batch_size,
            max_seq_length,
            multilabel,
        ):
            X = X.to(device)
            y = y.to(device)

            with torch.no_grad():
                loss, preds = get_loss_preds(model, X, y, num_labels, multilabel)

            if n_gpu > 1:
                loss = loss.mean()  # average loss on parallel training

            valid_loss += loss.item()
            valid_count += X.size(0)
            valid_correct += (y == preds).sum().item()

        print(f"Epoch {epoch} validation loss: {valid_loss / valid_count}")

    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    if hasattr(model, "module"):
        # DataParallel object -- unpack the module before saving
        model.module.save_pretrained(checkpoint_dir)
    else:
        # Plain transformers model
        model.save_pretrained(checkpoint_dir)

    tokenizer.save_pretrained(checkpoint_dir)
    config.save_pretrained(checkpoint_dir)

    if multilabel:
        valid_accuracy = valid_correct / (valid_count * num_labels)
    else:
        valid_accuracy = valid_correct / valid_count

    return {
        "mean_train_loss": train_loss / train_count,
        "mean_valid_loss": valid_loss / valid_count,
        "valid_accuracy": valid_accuracy,
    }


def predict(
    *,
    input_dir,
    output_dir,
    model,
    tokenizer,
    predict_batch_size,
    device,
    max_seq_length,
    labels,
    multilabel,
):

    pred_probas = []

    for (X,) in tsv_to_encoded_batches(
        input_dir / "test.tsv",
        tokenizer,
        None,
        predict_batch_size,
        max_seq_length,
        multilabel,
    ):
        X = X.to(device)

        with torch.no_grad():
            outputs = model(input_ids=X)
            logits = outputs[0].detach().cpu()
            if multilabel:
                pred_proba = F.sigmoid(logits)
            else:
                pred_proba = torch.softmax(logits, 1)
            pred_probas.append(pred_proba.numpy())

    df = pd.DataFrame(np.vstack(pred_probas))
    df.columns = labels
    df.to_csv(output_dir / "test_results.tsv", index=False, sep="\t")


def embed(
    *,
    input_dir,
    output_dir,
    model,
    tokenizer,
    embed_batch_size,
    device,
    max_seq_length,
    embed_pooling,
    embed_layer,
):
    embeddings = []

    with open(output_dir / "embeddings.jsonl", "w") as f:
        for X, tokens in tsv_to_encoded_batches(
            input_dir / "input.tsv",
            tokenizer,
            None,
            embed_batch_size,
            max_seq_length,
            False,
            return_tokens=True,
        ):
            X = X.to(device)

            with torch.no_grad():
                outputs = model(input_ids=X)
                hidden_states = outputs[-1]
                hidden_state = hidden_states[embed_layer]

                if embed_pooling == "mean":
                    embeddings = hidden_state.mean(dim=1).cpu().numpy().tolist()
                    for row in embeddings:
                        row_json = {"embedding": row}
                        f.write(f"{json.dumps(row_json)}\n")
                elif embed_pooling == "none":
                    embeddings = hidden_state.cpu().numpy().tolist()
                    for row, tokens in zip(embeddings, tokens):
                        # Truncate padding embeddings if needed
                        row_json = {"embedding": row[: len(tokens)], "tokens": tokens}
                        f.write(f"{json.dumps(row_json)}\n")
                else:
                    raise ValueError(f"Unsupported pooling type: {embed_pooling}")


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
        help="Directory to use for caching transformers downloads.",
    )
    parser.add_argument(
        "--config-overrides",
        help="Path to a JSON file containing config overrides for the given model.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Name of the transformers model to use.  If `mode` is 'train' or 'predict', "
        "this value should ensure `from transformers import <model>ModelForSequenceClassification` "
        "is a valid import. For example, model = 'Bert' -> from transformers import "
        "BertForSequenceClassification.  If `mode` is 'embed', the import will be `<model>Model`.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Name of the pretrained weights to use or path to a file containing "
        "pretrained weights.  If a name, must correspond to a valid weights name "
        "for the chosen model.",
    )
    parser.add_argument(
        "--multilabel",
        action="store_true",
        help="If True, model will train in a multilabel context (i.e. it's allowed to make multiple "
        "class predictions for each observation).",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        help="Per-GPU batch size to use for training.",
    )
    parser.add_argument(
        "--valid-batch-size",
        type=int,
        default=32,
        help="Per-GPU batch size to use for validation.",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=32,
        help="Per-GPU batch size to use for prediction.",
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
        "--max-seq-length",
        type=int,
        default=128,
        help="Truncate sequences to this length after tokenization.  If set too large for "
        "the model used, this may cause errors.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate for the AdamW optimizer."
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
        help="Epsilon value for the AdamW optimizer.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of training steps to accumulate gradients before updating model weights.",
    )
    parser.add_argument(
        "--embed-pooling",
        choices=["mean", "none"],
        default="mean",
        help="Pooling strategy to use for combining embeddings.  If None, embeddings for "
        "each token will be returned along with the token.  Ignored if method != 'embed'.",
    )
    parser.add_argument(
        "--embed-layer",
        type=int,
        default=-2,
        help="Index of the model layer to use for generating embeddings.",
    )

    args = parser.parse_args()

    if args.gradient_accumulation_steps <= 0:
        raise ValueError(
            f"Gradient accumulation steps must be >0, got '{args.gradient_accumulation_steps}'"
        )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    print(f"Using device: {device}")
    print(f"Number of GPUs: {n_gpu}")

    # Which model class we import depends on the task
    if args.mode in ("train", "predict"):
        model_name = f"{args.model}ForSequenceClassification"
    elif args.mode == "embed":
        model_name = f"{args.model}Model"

    tokenizer_name = f"{args.model}Tokenizer"
    config_name = f"{args.model}Config"

    print("Initializing transformer...")
    print(f"  Model: {model_name}")
    print(f"    Weights: {args.weights}")
    print(f"  Tokenizer: {tokenizer_name}")
    print(f"  Config: {config_name}")

    model_cls = getattr(transformers, model_name)
    tokenizer_cls = getattr(transformers, tokenizer_name)
    config_cls = getattr(transformers, config_name)

    tokenizer = tokenizer_cls.from_pretrained(args.weights, cache_dir=args.cache_dir)
    if tokenizer is None:
        raise ValueError("Failed to acquire tokenizer")
    config = config_cls.from_pretrained(args.weights, cache_dir=args.cache_dir)
    if config is None:
        raise ValueError("Failed to acquire config")

    # Apply config overrides specified by the user
    if args.config_overrides is not None:
        with open(args.config_overrides, "r") as f:
            config_overrides = json.load(f)
            for key, val in config_overrides.items():
                print(
                    f"Overriding config key '{key}' with value '{val}' "
                    f"(old value: '{getattr(config, key)}"
                )
                setattr(config, key, val)

    # Set the number of labels on the config object appropriately before we
    # initialize the model using the config
    # We need a list of labels for training or prediction so we know what the
    # output shape of the model should be
    labels = None
    if args.mode in ("train", "predict"):
        labels_file = input_dir / "labels.tsv"
        labels = read_unique_labels(input_dir / "labels.tsv")
        config.num_labels = len(labels)
        if config.num_labels == 1:
            raise ValueError(
                "transformers calculates regression loss when only one "
                "label is given, so it doesn't support classification when only "
                "a single label is available."
            )

        print(f"Inferred number of labels: {config.num_labels}")

    # We need to set the config to output hidden states if we're generating embeddings
    # DON'T output attentions so we can rely on hidden states being the last value in
    # the model output
    if args.mode == "embed":
        config.output_hidden_states = True
        config.output_attentions = False
        print("Outputting hidden states for embeddings")

    model = model_cls.from_pretrained(
        args.weights, cache_dir=args.cache_dir, config=config
    )
    if model is None:
        raise ValueError("Failed to acquire model")

    # If we have multiple GPUs, size the batches appropriately so DataParallel
    # distributes them correctly
    batch_multiplier = max(1, n_gpu)
    train_batch_size = args.train_batch_size * batch_multiplier
    valid_batch_size = args.valid_batch_size * batch_multiplier
    predict_batch_size = args.predict_batch_size * batch_multiplier

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.mode == "train":
        metrics = train(
            input_dir=input_dir,
            output_dir=output_dir,
            model=model,
            config=config,
            tokenizer=tokenizer,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            num_train_epochs=args.num_train_epochs,
            device=device,
            n_gpu=n_gpu,
            max_seq_length=args.max_seq_length,
            labels=labels,
            lr=args.lr,
            adam_eps=args.adam_eps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            multilabel=args.multilabel,
        )
        with open(output_dir / "valid_results.json", "w") as f:
            json.dump(metrics, f)

    elif args.mode == "predict":
        predict(
            input_dir=input_dir,
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            predict_batch_size=args.predict_batch_size,
            device=device,
            max_seq_length=args.max_seq_length,
            labels=labels,
            multilabel=args.multilabel,
        )
    elif args.mode == "embed":
        embed(
            input_dir=input_dir,
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            embed_batch_size=args.embed_batch_size,
            device=device,
            max_seq_length=args.max_seq_length,
            embed_pooling=args.embed_pooling,
            embed_layer=args.embed_layer,
        )
    else:
        raise ValueError(f"invalid mode: {args.mode}")
