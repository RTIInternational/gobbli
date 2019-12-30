import argparse

import torch
import torch.nn.functional as F
from transformers import BertConfig, BertForMaskedLM, BertTokenizer


def batch_list(l, batch_size):
    for i in range(0, len(l), batch_size):
        yield l[i : i + batch_size]


def encode_batch(batch, tokenizer, config):
    # Truncate input based on the sequence length
    # before stacking as a batch
    # Also return a boolean array indicating which tokens are masked
    encoded_texts = []
    for text in batch:
        encoded_text = torch.tensor(tokenizer.encode(text))[
            : config.max_position_embeddings
        ]
        encoded_texts.append(encoded_text)

    return torch.nn.utils.rnn.pad_sequence(encoded_texts, batch_first=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file", help="Input file containing line-delimited texts."
    )
    parser.add_argument(
        "output_file",
        help="Output file, where line-delimited generated texts will be written.",
    )
    parser.add_argument(
        "--bert-model",
        help="BERT model to use. "
        "Can be one of the pretrained names supported by transformers, in which case "
        "the pretrained weights will be downloaded. "
        "Anything else supported by transformers should work as well. ",
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--n-probable",
        help="Number of probable tokens to consider for replacement.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--diversity",
        help="Inverse dependence of selection likelihood on predicted probability.",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--probability",
        help="Probability of masking each token.",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--times",
        help="Number of new documents to generate for each existing document.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--batch-size",
        help="Number of documents to run through the BERT model at once.",
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

    if args.times < 0:
        raise ValueError("times must be >= 0")

    if not 0 <= args.probability <= 1:
        raise ValueError("probability must be >= 0 and <= 1")

    if not 0 < args.diversity <= 1:
        raise ValueError("diversity must be > 0 and <= 1")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, cache_dir=args.cache_dir)
    if tokenizer is None:
        raise ValueError("Failed to acquire tokenizer")
    config = BertConfig.from_pretrained(args.bert_model, cache_dir=args.cache_dir)
    if config is None:
        raise ValueError("Failed to acquire config")
    model = BertForMaskedLM.from_pretrained(
        args.bert_model, config=config, cache_dir=args.cache_dir
    )
    if model is None:
        raise ValueError("Failed to acquire model")

    model = model.to(args.device)
    model.eval()

    with open(args.input_file, "r", encoding="utf-8") as f_in:
        with open(args.output_file, "w", encoding="utf-8") as f_out:
            batches = batch_list(f_in.readlines(), args.batch_size)
            input_id_batches = [
                encode_batch(batch, tokenizer, config) for batch in batches
            ]

            for time in range(args.times):
                for batch_id, input_ids in enumerate(input_id_batches):
                    # Generate a replacement mask for the batch.  Do this each time we
                    # run the batch to get different results
                    # Don't mask any padding tokens, so we don't get predictions for them
                    # and can easily remove them at the end
                    should_replace = (
                        torch.rand_like(input_ids, dtype=torch.float) < args.probability
                    ) & (input_ids != tokenizer.vocab.get(tokenizer.pad_token))

                    masked_ids = input_ids.clone().detach()
                    masked_ids[should_replace] = tokenizer.vocab.get(
                        tokenizer.mask_token
                    )
                    masked_ids = masked_ids.to(args.device)

                    # These are created implicitly, but we need to explicitly
                    # initialize them to their defaults because they won't
                    # be on the chosen device otherwise
                    attention_mask = torch.ones_like(masked_ids).to(args.device)
                    token_type_ids = torch.zeros_like(masked_ids).to(args.device)

                    with torch.no_grad():
                        (output,) = model(
                            masked_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                        )

                    # Bring model results back to the CPU for processing
                    masked_ids = masked_ids.cpu()
                    output = output.cpu()

                    # Exponentiate according to inverse diversity to set level of
                    # dependence on predicted probability
                    prediction_scores = torch.pow(
                        F.softmax(output, dim=2), 1.0 / args.diversity
                    )

                    max_seq_len = prediction_scores.size(1)

                    output_ids = []
                    for i, row in enumerate(prediction_scores):
                        # size: (seq_len, vocab_len)
                        # Generate probabilities for each token.  Ultimately, we only
                        # care about the ones that were masked
                        # First generate the indices of all candidates, i.e.
                        # tokens in the top n_probable when sorted by descending predicted
                        # probability
                        # size: (seq_len, n_probable)
                        candidate_ndxs = torch.argsort(row, dim=1, descending=True)[
                            :, : args.n_probable
                        ]

                        candidates = torch.stack(
                            [tok[candidate_ndxs[i]] for i, tok in enumerate(row)]
                        )

                        # Determine a replacement among the candidates for each token
                        # in the row
                        # size: (seq_len)
                        replacement_sorted_ndxs = torch.multinomial(
                            candidates, 1
                        ).squeeze()

                        # Map the candidate indices back to original row indices
                        # size: (seq_len)
                        replacements = torch.tensor(
                            [
                                candidate_ndxs[i][ndx]
                                for i, ndx in enumerate(replacement_sorted_ndxs)
                            ]
                        )

                        # Perform the replacement
                        output_ids.append(
                            torch.where(
                                should_replace[i], replacements, masked_ids[i]
                            ).tolist()
                        )

                    # Convert the token IDs back to text; filter out padding tokens
                    output_texts = [
                        tokenizer.decode(
                            [
                                tok
                                for tok in row
                                if not tok == tokenizer.vocab.get(tokenizer.pad_token)
                            ]
                        )
                        for row in output_ids
                    ]

                    # Write text to the output file
                    # Escape any embedded newlines
                    # Make sure not to write an extra newline after the last row
                    for i, row in enumerate(output_texts):
                        f_out.write(row.replace("\n", " "))

                        # Write a newline except for the last row on the last
                        # time through.  We'd get an empty line at the end of the
                        # file otherwise
                        if not (
                            i == len(output_texts) - 1
                            and batch_id == len(input_id_batches) - 1
                            and time == args.times - 1
                        ):
                            f_out.write("\n")
