import argparse
import csv
import json
import logging
import os
from datetime import datetime

import torch
from data_utils.log_wrapper import create_logger
from data_utils.metrics import compute_acc, compute_cross_entropy
from data_utils.utils import set_environment
from mt_dnn.gobbli_batcher import GobbliBatchGen
from mt_dnn.gobbli_model import GobbliMTDNNModel

logging.basicConfig(level=logging.INFO)


def model_config(parser):
    parser.add_argument("--update_bert_opt", default=0, type=int)
    parser.add_argument("--multi_gpu_on", action="store_true")
    parser.add_argument(
        "--mem_cum_type", type=str, default="simple", help="bilinear/simple/defualt"
    )
    parser.add_argument("--answer_num_turn", type=int, default=5)
    parser.add_argument("--answer_mem_drop_p", type=float, default=0.1)
    parser.add_argument("--answer_att_hidden_size", type=int, default=128)
    parser.add_argument(
        "--answer_att_type",
        type=str,
        default="bilinear",
        help="bilinear/simple/defualt",
    )
    parser.add_argument(
        "--answer_rnn_type", type=str, default="gru", help="rnn/gru/lstm"
    )
    parser.add_argument(
        "--answer_sum_att_type",
        type=str,
        default="bilinear",
        help="bilinear/simple/defualt",
    )
    parser.add_argument("--answer_merge_opt", type=int, default=1)
    parser.add_argument("--answer_mem_type", type=int, default=1)
    parser.add_argument("--answer_dropout_p", type=float, default=0.1)
    parser.add_argument("--answer_weight_norm_on", action="store_true")
    parser.add_argument("--dump_state_on", action="store_true")
    parser.add_argument("--answer_opt", type=int, default=0, help="0,1")
    parser.add_argument("--mtl_opt", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0)
    parser.add_argument("--mix_opt", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--init_ratio", type=float, default=1)
    return parser


def data_config(parser):
    parser.add_argument(
        "--log_file", default="mt-dnn-train.log", help="path for log file."
    )
    parser.add_argument(
        "--init_checkpoint", default="mt_dnn/bert_model_base.pt", type=str
    )
    parser.add_argument("--data_dir", default="data/mt_dnn")
    parser.add_argument("--data_sort_on", action="store_true")
    parser.add_argument("--name", default="farmer")
    parser.add_argument("--train_file")
    parser.add_argument("--valid_file")
    parser.add_argument("--test_file")
    parser.add_argument("--label_file", required=True)
    return parser


def train_config(parser):
    parser.add_argument(
        "--cuda",
        type=bool,
        default=torch.cuda.is_available(),
        help="whether to use GPU acceleration.",
    )
    parser.add_argument("--log_per_updates", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batch_size_eval", type=int, default=8)
    parser.add_argument(
        "--optimizer",
        default="adamax",
        help="supported optimizer: adamax, sgd, adadelta, adam",
    )
    parser.add_argument("--grad_clipping", type=float, default=0)
    parser.add_argument("--global_grad_clipping", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--warmup_schedule", type=str, default="warmup_linear")

    parser.add_argument("--vb_dropout", action="store_false")
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--dropout_w", type=float, default=0.000)
    parser.add_argument("--bert_dropout_p", type=float, default=0.1)

    # EMA
    parser.add_argument("--ema_opt", type=int, default=0)
    parser.add_argument("--ema_gamma", type=float, default=0.995)

    # scheduler
    parser.add_argument(
        "--have_lr_scheduler", dest="have_lr_scheduler", action="store_false"
    )
    parser.add_argument("--multi_step_lr", type=str, default="10,20,30")
    parser.add_argument("--freeze_layers", type=int, default=-1)
    parser.add_argument("--embedding_opt", type=int, default=0)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--bert_l2norm", type=float, default=0.0)
    parser.add_argument("--scheduler_type", type=str, default="ms", help="ms/rop/exp")
    parser.add_argument("--output_dir", default="checkpoint")
    parser.add_argument(
        "--seed",
        type=int,
        default=2018,
        help="random seed for data shuffling, embedding init, etc.",
    )
    parser.add_argument(
        "--task_config_path", type=str, default="configs/tasks_config.json"
    )

    return parser


def dump(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def dump_predictions(scores, labels, output_file):
    # Map label indices (used internally for predictions) to label names
    # (which the user will be familiar with)
    labels_inverse = {ndx: label for label, ndx in labels.items()}
    label_header = [labels_inverse[i] for i in range(len(labels))]
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        # Write the header
        writer.writerow(label_header)

        # Write the predictions
        for row in scores.tolist():
            writer.writerow(row)


def read_labels(label_file):
    with open(label_file, "r") as f:
        return {l.strip(): i for i, l in enumerate(f.readlines())}


def eval_model(model, dataset, use_cuda=True, with_label=True):
    dataset.reset()
    if use_cuda:
        model.cuda()

    predictions = []
    scores = []
    golds = []
    metrics = {}
    for batch in dataset:
        if with_label:
            input_ids, token_type_ids, attention_mask, labels = batch
            golds.extend(labels)
        else:
            input_ids, token_type_ids, attention_mask = batch

        score, pred = model.predict(input_ids, token_type_ids, attention_mask)
        predictions.extend(pred)
        scores.append(score)

    scores = torch.cat(scores, 0)

    if not with_label:
        return predictions, scores

    metrics["accuracy"] = compute_acc(predictions, golds)
    metrics["loss"] = compute_cross_entropy(scores, torch.LongTensor(golds)).item()

    return metrics, predictions, scores


parser = argparse.ArgumentParser()
parser = data_config(parser)
parser = model_config(parser)
parser = train_config(parser)
args = parser.parse_args()

given_train = args.train_file is not None
given_valid = args.valid_file is not None
given_test = args.test_file is not None

if given_train and not given_valid or given_valid and not given_train:
    raise ValueError("Must have both a train and valid dataset for training.")

output_dir = args.output_dir
data_dir = args.data_dir

os.makedirs(output_dir, exist_ok=True)
output_dir = os.path.abspath(output_dir)

set_environment(args.seed, args.cuda)
log_path = args.log_file
logger = create_logger(__name__, to_disk=True, log_file=log_path)


def main():
    logger.info("Launching the MT-DNN training")
    opt = vars(args)
    # update data dir
    opt["data_dir"] = data_dir

    labels = read_labels(args.label_file)
    # The original code expects this to be a comma-separated list of ints in string format
    # to account for multiple tasks having different label sizes
    # It works out if we just use our single label size as a string
    opt["label_size"] = str(len(labels))
    # This option was also assigned per-task in the original code, but we'll just take the default
    opt["tasks_dropout_p"] = [args.dropout_p]
    # This was also per-task, and the default command line arg is a scalar instead of a list,
    # so set it correctly to be a list
    opt["answer_opt"] = [0]
    batch_size = args.batch_size

    batch_gen_kwargs = {
        "maxlen": args.max_seq_len,
        "batch_size": batch_size,
        "dropout_w": args.dropout_w,
        "gpu": args.cuda,
    }

    if given_train:
        train_data = GobbliBatchGen(
            args.train_file,
            has_labels=True,
            labels=labels,
            is_train=True,
            **batch_gen_kwargs,
        )
        num_all_batches = args.epochs * len(train_data)
    else:
        num_all_batches = 0

    if given_valid:
        valid_data = GobbliBatchGen(
            args.valid_file,
            has_labels=True,
            labels=labels,
            is_train=False,
            **batch_gen_kwargs,
        )

    if given_test:
        test_data = GobbliBatchGen(
            args.test_file, has_labels=False, is_train=False, **batch_gen_kwargs
        )

    model_path = args.init_checkpoint

    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        config = state_dict["config"]
        config["attention_probs_dropout_prob"] = args.bert_dropout_p
        config["hidden_dropout_prob"] = args.bert_dropout_p
        opt.update(config)
    else:
        raise ValueError("Could not find the init model!")

    model = GobbliMTDNNModel(opt, state_dict=state_dict, num_train_step=num_all_batches)

    ####model meta str
    headline = "############# Model Arch of MT-DNN #############"
    ###print network
    logger.info("\n{}\n{}\n".format(headline, model.network))

    # dump config
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, "w", encoding="utf-8") as writer:
        writer.write("{}\n".format(json.dumps(opt)))
        writer.write("\n{}\n{}\n".format(headline, model.network))

    logger.info("Total number of params: {}".format(model.total_param))

    if args.freeze_layers > 0:
        model.network.freeze_layers(args.freeze_layers)

    if args.cuda:
        model.cuda()

    if given_train and given_valid:
        for epoch in range(0, args.epochs):
            logger.warning("At epoch {}".format(epoch))
            start = datetime.now()

            # Training
            train_data.reset()
            for i, (input_ids, token_type_ids, attention_mask, labels) in enumerate(
                train_data
            ):
                model.update(input_ids, token_type_ids, attention_mask, labels)
                if (model.updates) % args.log_per_updates == 0 or model.updates == 1:
                    logger.info(
                        "updates[{0:6}] train loss[{1:.5f}] remaining[{2}]".format(
                            model.updates,
                            model.train_loss.avg,
                            str(
                                (datetime.now() - start)
                                / (i + 1)
                                * (len(train_data) - i - 1)
                            ).split(".")[0],
                        )
                    )

            # Training and validation metrics
            for dataset, name in ((train_data, "train"), (valid_data, "valid")):
                metrics, _, _ = eval_model(
                    model, dataset, use_cuda=args.cuda, with_label=True
                )

                for key, val in metrics.items():
                    logger.warning(
                        "Epoch {0} -- [{1}] {2}: {3:.3f}".format(epoch, name, key, val)
                    )

                score_file = os.path.join(
                    output_dir, "{}_scores_{}.json".format(name, epoch)
                )
                results = {"metrics": metrics}
                dump(score_file, results)
        model_file = os.path.join(output_dir, "model_{}.pt".format(epoch))
        model.save(model_file)

    if given_test:
        _, scores = eval_model(model, test_data, use_cuda=args.cuda, with_label=False)

        predict_file = os.path.join(output_dir, "predict.csv")
        dump_predictions(scores, labels, predict_file)
        logger.info("[new predictions saved.]")


if __name__ == "__main__":
    main()
