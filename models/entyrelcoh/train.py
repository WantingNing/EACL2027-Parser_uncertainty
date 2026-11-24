# date = 2024-08-18
# author = liuwei

import logging
import os
import json
import pickle
import math
import random
import time
import datetime
from tqdm import tqdm, trange

from sklearn.metrics import f1_score, accuracy_score, classification_report

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, BertModel
from transformers import XLNetConfig, XLNetTokenizer, XLNetModel
from transformers import LlamaConfig, LlamaTokenizer, LlamaModel
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel

from task_dataset import SentDataset
from model import BaseClassifer, FusionClassifier
from utils import labels_from_file  # , extract_relation_embedding

# logging.disable(logging.WARNING)

# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # 输出到控制台的handler
chlr.setFormatter(formatter)
logger.addHandler(chlr)

# for output
dt = datetime.datetime.now()
TIME_CHECKPOINT_DIR = "checkpoint_{}-{}-{}_{}:{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
PREFIX_CHECKPOINT_DIR = "checkpoint"


def get_argparse():
    parser = argparse.ArgumentParser()

    # for data
    parser.add_argument("--data_dir", default="data/dataset", type=str)
    parser.add_argument("--dataset", default="test", type=str, help="toefl1234, gcdc")
    parser.add_argument("--fold_id", default=1, type=int, help="[1-5] for toefl1234, [1-10] for gcdc")
    parser.add_argument("--output_dir", default="data/result", type=str, help="path to save checkpoint")
    parser.add_argument("--label_list", default="low, medium, high", type=str)
    parser.add_argument("--model_type", default="fusion", type=str, help="rgcn")
    parser.add_argument("--pooler_type", default="avg", type=str, help="cls, avg")
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str, help="roberta-base")
    parser.add_argument(
        "--pretrained_model_root",
        default=os.environ.get("PRETRAINED_MODEL_ROOT", "/hits/basement/nlp/liuwi/resources/pretrained_models"),
        type=str,
        help="optional local directory containing pretrained model folders"
    )

    # for training
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--max_text_length", default=512, type=int)
    parser.add_argument("--max_sent_num", default=24, type=int, help="number of sents in a text")
    parser.add_argument("--max_rel_num", default=56, type=int)
    parser.add_argument("--num_train_epochs", default=20, type=int, help="training epoch")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--seed", default=106524, type=int, help="random seed")

    # transformer
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--embed_dim", default=300, type=int)

    # for graph
    parser.add_argument("--no_enty", default=False, action="store_true")
    parser.add_argument("--no_rel", default=False, action="store_true")
    parser.add_argument(
        "--dump_predictions",
        default=False,
        action="store_true",
        help="write JSONL predictions for the selected best epoch"
    )
    parser.add_argument(
        "--dump_best_metric",
        default="test_acc",
        choices=["test_acc", "test_f1", "dev_acc", "dev_f1"],
        help="metric used to select the epoch whose predictions are dumped"
    )

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_model_name_or_path(model_name_or_path, pretrained_model_root=None):
    if os.path.exists(model_name_or_path):
        return model_name_or_path

    if pretrained_model_root:
        candidate = os.path.join(pretrained_model_root, model_name_or_path)
        if os.path.exists(candidate):
            return candidate

    return model_name_or_path


def cap_max_text_length(args, config, tokenizer):
    max_lengths = []
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int) and 0 < tokenizer_max < 1000000:
        max_lengths.append(tokenizer_max)

    config_max = getattr(config, "max_position_embeddings", None)
    if isinstance(config_max, int) and 0 < config_max < 1000000:
        max_lengths.append(config_max)

    if not max_lengths:
        return

    max_supported_length = min(max_lengths)
    if args.max_text_length > max_supported_length:
        logger.warning(
            "max_text_length=%d exceeds the model limit %d for %s; using %d instead.",
            args.max_text_length,
            max_supported_length,
            args.model_name_or_path,
            max_supported_length
        )
        args.max_text_length = max_supported_length


def get_dataloader(dataset, args, mode="train"):
    print("  {} dataset length: ".format(mode), len(dataset))
    if mode.lower() == "train":
        sampler = RandomSampler(dataset)
        batch_size = args.train_batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batch_size

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    return data_loader


def get_optimizer(model, args, num_training_steps):
    specific_params = []
    no_deday = ["bias", "LayerNorm.weigh"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_deday)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_deday)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def train(model, args, train_dataloader, dev_dataloader, test_dataloader):
    ## 1. prepare data
    t_total = int(len(train_dataloader) * args.num_train_epochs)
    num_train_epochs = args.num_train_epochs
    print_step = int(len(train_dataloader) // 4)

    ## 2.optimizer
    optimizer, scheduler = get_optimizer(model, args, t_total)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size per device = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    best_dev = 0.0
    best_dev_epoch = 0
    best_test = 0.0
    best_test_epoch = 0
    best_dump_metric = -1.0
    best_dump_epoch = 0
    best_dump_scores = {}
    train_iterator = trange(1, int(num_train_epochs) + 1, desc="Epoch")
    for epoch in train_iterator:
        model.train()
        model.zero_grad()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            batch_features = tuple(t.to(args.device) for t in batch)
            batch_doc_vectors = batch_features[0]
            batch_sent_vectors = batch_features[1]
            batch_sent_mask = batch_features[2]
            batch_sent_rel_mask = batch_features[3]
            batch_rel_ids = batch_features[4]
            batch_start_pos = batch_features[5]
            batch_end_pos = batch_features[6]
            batch_label_ids = batch_features[7]

            if args.model_type.lower() == "base":
                inputs = {
                    "doc_vectors": batch_doc_vectors,
                    "labels": batch_label_ids,
                    "flag": "Train",
                }
            elif args.model_type.lower() == "fusion":
                inputs = {
                    "sent_vectors": batch_sent_vectors,
                    "sent_mask": batch_sent_mask,
                    "rel_ids": batch_rel_ids,
                    "pos_start": batch_start_pos,
                    "pos_end": batch_end_pos,
                    "sent_rel_mask": batch_sent_rel_mask,
                    "labels": batch_label_ids,
                    "flag": "Train",
                }

            outputs = model(**inputs)
            loss = outputs[0]

            # optimizer.zero_grad()
            loss.backward()
            global_step += 1
            logging_loss = loss.item()
            tr_loss += logging_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_step % print_step == 0:
                print(" Current loss=%.4f, global average loss=%.4f" % (logging_loss, tr_loss / global_step))

        model.eval()
        dev_acc, dev_f1 = evaluate(
            model, args, dev_dataloader, epoch, desc="dev"
        )
        test_acc, test_f1 = evaluate(
            model, args, test_dataloader, epoch, desc="test"
        )
        print()
        print(" Dev Acc=%.4f, F1=%.4f" % (dev_acc, dev_f1))
        print(" Test Acc=%.4f, F1=%.4f" % (test_acc, test_f1))
        if dev_f1 > best_dev:
            best_dev = dev_f1
            best_dev_epoch = epoch
        if test_f1 > best_test:
            best_test = test_f1
            best_test_epoch = epoch

        if args.dump_predictions:
            metric_map = {
                "dev_acc": dev_acc,
                "dev_f1": dev_f1,
                "test_acc": test_acc,
                "test_f1": test_f1,
            }
            current_dump_metric = metric_map[args.dump_best_metric]
            if current_dump_metric > best_dump_metric:
                best_dump_metric = current_dump_metric
                best_dump_epoch = epoch
                best_dump_scores = metric_map
                # Dump train, dev, and test predictions from the same selected
                # epoch. This lets downstream ensembles choose alpha on dev
                # while keeping the corresponding train/test predictions.
                for dump_split, dump_loader in (
                    ("train", train_dataloader),
                    ("dev", dev_dataloader),
                    ("test", test_dataloader),
                ):
                    dump_name = "{}_fold{}_{}_best".format(args.dataset, args.fold_id, dump_split)
                    evaluate(
                        model,
                        args,
                        dump_loader,
                        epoch,
                        desc="dump_{}".format(dump_split),
                        save_predictions_jsonl=True,
                        dataset_name=dump_name,
                        model_name=get_prediction_model_name(args),
                    )
                    summary_path = os.path.join(args.output_dir, "predictions", "{}_summary.json".format(dump_name))
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "dataset": args.dataset,
                                "fold_id": args.fold_id,
                                "split": dump_split,
                                "model_name": get_prediction_model_name(args),
                                "dump_best_metric": args.dump_best_metric,
                                "best_epoch": best_dump_epoch,
                                "best_metric_value": best_dump_metric,
                                "dev_acc": dev_acc,
                                "dev_f1": dev_f1,
                                "test_acc": test_acc,
                                "test_f1": test_f1,
                            },
                            f,
                            indent=2,
                        )

        output_dir = os.path.join(args.output_dir, "good")
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        if args.model_type not in ["rgcn"]:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        else:
            print(" ## Not save ##")

    print(" Best dev: Epoch=%d, F1=%.4f\n" % (best_dev_epoch, best_dev))
    print(" Best test: Epoch=%d, F1=%.4f\n" % (best_test_epoch, best_test))
    if args.dump_predictions:
        print(
            " Best dumped predictions: Epoch=%d, %s=%.4f\n"
            % (best_dump_epoch, args.dump_best_metric, best_dump_metric)
        )

    return best_dev_epoch, best_test_epoch


def get_prediction_model_name(args):
    if args.model_type.lower() == "base":
        return "fusion_textonly"
    if args.no_enty and args.no_rel:
        return "fusion_textonly"
    if args.no_rel:
        return "fusion_textenty"
    if args.no_enty:
        return "fusion_textrel"
    return "fusion_entyrel"


def evaluate(model, args, dataloader, epoch, desc="dev", write_file=False, save_predictions_jsonl=False, dataset_name=None, model_name="fusion"):
    all_label_ids = None
    all_pred_ids = None
    all_probs = None
    all_logits = None
    for batch in tqdm(dataloader, desc=desc):
        batch_features = tuple(t.to(args.device) for t in batch)
        batch_doc_vectors = batch_features[0]
        batch_sent_vectors = batch_features[1]
        batch_sent_mask = batch_features[2]
        batch_sent_rel_mask = batch_features[3]
        batch_rel_ids = batch_features[4]
        batch_start_pos = batch_features[5]
        batch_end_pos = batch_features[6]
        batch_label_ids = batch_features[7]

        if args.model_type.lower() == "base":
            inputs = {
                "doc_vectors": batch_doc_vectors,
                "labels": batch_label_ids,
                "flag": "Eval",
            }
        elif args.model_type.lower() == "fusion":
            inputs = {
                "sent_vectors": batch_sent_vectors,
                "sent_mask": batch_sent_mask,
                "rel_ids": batch_rel_ids,
                "pos_start": batch_start_pos,
                "pos_end": batch_end_pos,
                "sent_rel_mask": batch_sent_rel_mask,
                "labels": batch_label_ids,
                "flag": "Eval",
            }

        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]
            logits = outputs[1] if len(outputs) > 1 else None

        label_ids = batch_label_ids.detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        logits_batch = logits.detach().cpu().numpy() if logits is not None else None
        probs_batch = torch.softmax(logits, dim=-1).detach().cpu().numpy() if logits is not None else None
        if all_label_ids is None:
            all_label_ids = label_ids
            all_pred_ids = pred_ids
            all_probs = probs_batch
            all_logits = logits_batch
        else:
            all_label_ids = np.append(all_label_ids, label_ids)
            all_pred_ids = np.append(all_pred_ids, pred_ids)
            if all_probs is not None and probs_batch is not None:
                all_probs = np.vstack([all_probs, probs_batch])
            if all_logits is not None and logits_batch is not None:
                all_logits = np.vstack([all_logits, logits_batch])

    print(all_label_ids.shape, all_pred_ids.shape)
    acc = accuracy_score(y_true=all_label_ids, y_pred=all_pred_ids)
    f1 = f1_score(y_true=all_label_ids, y_pred=all_pred_ids, average="macro")
    """
    res = classification_report(
        y_true=all_label_ids,
        y_pred=all_pred_ids,
        target_names=args.label_list,
        digits=4
    )
    print(res)
    """

    if write_file:
        res_file = "{}_res_{}.txt".format(desc, args.model_type)
        data_dir = os.path.join(args.data_dir, "preds")
        os.makedirs(data_dir, exist_ok=True)
        res_file = os.path.join(data_dir, res_file)
        error_num = 0
        with open(res_file, "w", encoding="utf-8") as f:
            for l, p in zip(all_label_ids, all_pred_ids):
                if l == p:
                    f.write("%s\t%s\n" % (args.label_list[l], args.label_list[p]))
                else:
                    error_num += 1
                    f.write("%s\t%s\t%d\n" % (args.label_list[l], args.label_list[p], error_num))

    # Save predictions to jsonl with probabilities, by dataset subset
    if save_predictions_jsonl and dataset_name and all_probs is not None:
        # Label mapping: low=1, medium=2, high=3 (user format)
        idx_to_label_num = {"low": "1", "medium": "2", "high": "3"}
        pred_dir = os.path.join(args.output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        out_path = os.path.join(pred_dir, "{}.jsonl".format(dataset_name))
        text_ids = getattr(dataloader.dataset, "text_ids", None)
        if text_ids is None:
            text_ids = [str(i) for i in range(len(all_label_ids))]
        with open(out_path, "w", encoding="utf-8") as f:
            for idx in range(len(all_label_ids)):
                true_idx = int(all_label_ids[idx])
                pred_idx = int(all_pred_ids[idx])
                true_label = idx_to_label_num.get(args.label_list[true_idx], str(true_idx + 1))
                pred_label = idx_to_label_num.get(args.label_list[pred_idx], str(pred_idx + 1))
                prob_dict = {"prob_{}".format(k + 1): float(all_probs[idx][k]) for k in range(all_probs.shape[1])}
                logit_dict = {}
                if all_logits is not None:
                    logit_dict = {"logit_{}".format(k + 1): float(all_logits[idx][k]) for k in range(all_logits.shape[1])}
                record = {
                    "model": model_name,
                    "dataset": args.dataset,
                    "fold_id": args.fold_id,
                    "split": desc.replace("dump_", ""),
                    "epoch": epoch,
                    "id": idx,
                    "text_id": text_ids[idx] if idx < len(text_ids) else str(idx),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "gold_index": true_idx,
                    "pred_index": pred_idx,
                    **prob_dict,
                    **logit_dict
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved predictions to %s", out_path)

    return acc, f1


def main():
    args = get_argparse().parse_args()
    if torch.cuda.is_available():
        args.n_gpu = 1
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device
    logger.info(" Training/evaluation parameters %s", args)
    logger.info(" ####### Acc, F1 for fold %d ########", args.fold_id)
    # print("Training/evaluation parameters %s", args)
    set_seed(args.seed)

    ## 1. prepare data
    data_dir = os.path.join(args.data_dir, args.dataset)
    data_dir = os.path.join(data_dir, str(args.fold_id))
    train_data_file = os.path.join(data_dir, "train.json")
    dev_data_file = os.path.join(data_dir, "dev.json")
    test_data_file = os.path.join(data_dir, "test.json")
    args.data_dir = data_dir
    output_dir = os.path.join(args.output_dir, args.dataset, "entyrelcoh")
    if args.fold_id > 0:
        output_dir = os.path.join(output_dir, str(args.fold_id))
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    exp_rel_list = labels_from_file("data/parser/pdtb3/exp/l2/labels.txt")
    imp_rel_list = labels_from_file("data/parser/pdtb3/imp/l2/labels.txt")
    rel_list = set()
    _ = [rel_list.add(l) for l in exp_rel_list]
    _ = [rel_list.add(l) for l in imp_rel_list]
    rel_list = list(rel_list)
    rel_list.append("entity")
    rel_list = sorted(rel_list)
    args.rel_list = rel_list
    label_list = args.label_list.split(",")
    print(label_list)
    label_list = [l.lower().strip() for l in label_list]
    args.label_list = label_list
    args.num_labels = len(label_list)

    ## 2. define models
    args.model_name_or_path = resolve_model_name_or_path(
        args.model_name_or_path,
        args.pretrained_model_root
    )
    print(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    cap_max_text_length(args, config, tokenizer)
    if "llama" in args.model_name_or_path.lower():
        encoder = AutoModel.from_pretrained(
            args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16
        )
    else:
        encoder = AutoModel.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    for name, param in encoder.named_parameters():
        param.requires_grad = False
    encoder.eval()
    encoder = encoder.to(args.device)
    # extract_relation_embedding(rel_list, encoder, tokenizer)
    args.input_size = config.hidden_size
    if args.model_type.lower() == "base":
        model = BaseClassifer(args=args)
    elif args.model_type.lower() == "fusion":
        model = FusionClassifier(args=args)
    model = model.to(args.device)

    ## 3. prepare dataset
    dataset_params = {
        "max_text_length": args.max_text_length,
        "max_sent_num": args.max_sent_num,
        "max_rel_num": args.max_rel_num,
        "rel_list": rel_list,
        "label_list": label_list,
        "tokenizer": tokenizer,
        "encoder": encoder,
        "pooler_type": args.pooler_type,
        "no_rel": args.no_rel,
        "no_enty": args.no_enty
    }

    if args.do_train:
        train_dataset = SentDataset(train_data_file, params=dataset_params)
        dev_dataset = SentDataset(dev_data_file, params=dataset_params)
        test_dataset = SentDataset(test_data_file, params=dataset_params)
        train_dataloader = get_dataloader(train_dataset, args, mode="train")
        dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        test_dataloader = get_dataloader(test_dataset, args, mode="test")

        train(model, args, train_dataloader, dev_dataloader, test_dataloader)

    if args.do_dev or args.do_test:
        temp_file = os.path.join(output_dir, "good/checkpoint_{}/pytorch_model.bin")

        ## toefl
        # for entyrel
        # temp_file = "data/result/toefl_p5/fusion+Llama-3.1-8B-Instruct_entyrel/2/good/checkpoint_19/pytorch_model.bin"
        # for textonly
        # temp_file = "data/result/toefl_p5/base+Llama-3.1-8B-Instruct/2/good/checkpoint_14/pytorch_model.bin"
        # for textenty
        # temp_file = "data/result/toefl_p5/fusion+Llama-3.1-8B-Instruct_textenty/2/good/checkpoint_12/pytorch_model.bin"
        # for textrel
        # temp_file = "data/result/toefl_p5/fusion+Llama-3.1-8B-Instruct_textrel/2/good/checkpoint_16/pytorch_model.bin"
        ## gcdc
        # for entyrel
        # temp_file = "data/result/enron/fusion+Llama-3.1-8B-Instruct_entyrel/9/good/checkpoint_8/pytorch_model.bin"
        # for textonly
        # temp_file = "data/result/enron/base+Llama-3.1-8B-Instruct/9/good/checkpoint_14/pytorch_model.bin"
        # for textenty
        # temp_file = "data/result/enron/fusion+Llama-3.1-8B-Instruct_textenty/2/good/checkpoint_14/pytorch_model.bin"
        # for textrel
        # temp_file = "data/result/enron/fusion+Llama-3.1-8B-Instruct_textrel/2/good/checkpoint_15/pytorch_model.bin"

        dev_dataset = SentDataset(dev_data_file, params=dataset_params)
        test_dataset = SentDataset(test_data_file, params=dataset_params)
        dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        test_dataloader = get_dataloader(test_dataset, args, mode="test")
        write_file = True

        ## toefl_p5, 2
        # entyrel, 19
        # textonly, 14
        # textenty, 12
        # textrel, 16
        ## enron, 9
        # entyrel, 8
        # textonly, 14
        # textenty, 14, (2)
        # textrel, 15, (2)

        # Build model name for prediction output (e.g., fusion_entyrel, base_textonly)
        if args.model_type.lower() == "base":
            pred_model_name = "fusion_textonly"  # base is text-only
        else:
            if args.no_enty and args.no_rel:
                pred_model_name = "fusion_textonly"
            elif args.no_rel:
                pred_model_name = "fusion_textenty"
            elif args.no_enty:
                pred_model_name = "fusion_textrel"
            else:
                pred_model_name = "fusion_entyrel"
        save_jsonl = args.dataset in ["clinton", "enron", "yahoo", "yelp"]

        for epoch in range(14, 15):
            checkpoint_file = temp_file.format(str(epoch))
            # checkpoint_file = temp_file
            print(" Epoch=%d, checkpoint=%s" % (epoch, checkpoint_file))
            model.load_state_dict(torch.load(checkpoint_file))
            model.eval()

            if args.do_dev:
                acc, f1 = evaluate(
                    model, args, dev_dataloader, epoch, "dev", write_file,
                    save_predictions_jsonl=save_jsonl, dataset_name=args.dataset, model_name=pred_model_name
                )
                print(" Dev Acc=%.4f, F1=%.4f" % (acc, f1))

            if args.do_test:
                acc, f1 = evaluate(
                    model, args, test_dataloader, epoch, "test", write_file,
                    save_predictions_jsonl=save_jsonl, dataset_name=args.dataset, model_name=pred_model_name
                )
                print(" Test Acc=%.4f, F1=%.4f" % (acc, f1))


if __name__ == "__main__":
    main()
