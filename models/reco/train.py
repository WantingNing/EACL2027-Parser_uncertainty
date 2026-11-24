#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from data import (
    BatchCollator,
    PreparedDataset,
    attach_rst_tensor_cache,
    attach_sota_cached_representations,
    build_feature_store,
    load_sota_texts_for_prefix,
)
from graph_cache import PREFIXES, get_fold_paths, get_split_root, load_jsonl, load_or_build_graph_cache
from model import ReCo


MODEL_NAME_MAP = {
    "bert-large": "bert-large-uncased",
    "roberta-large": "roberta-large",
    "roberta-base": "roberta-base",
}
RAW_PREFIX_NAMES = {
    "clinton": "Clinton",
    "enron": "Enron",
    "yahoo": "Yahoo",
    "yelp": "Yelp",
}
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/reco/gcdc.yaml"
DEFAULT_ABLATIONS_PATH = Path(__file__).resolve().parents[2] / "configs/reco/ablations.yaml"
RESULT_SCHEMA_VERSION = 2


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got `{value}`.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ReCo or a paper component ablation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ablations-config", type=Path, default=DEFAULT_ABLATIONS_PATH)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--prefix", nargs="+", choices=PREFIXES, default=None)
    parser.add_argument("--source-prefix", nargs="+", choices=PREFIXES, default=None)
    parser.add_argument("--fold", type=int, choices=tuple(range(1, 11)), default=None)
    parser.add_argument("--model-name", choices=tuple(MODEL_NAME_MAP), default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--raw-data-root", type=Path, default=None)
    parser.add_argument("--parser-root", type=Path, default=None)
    parser.add_argument("--sota-data-root", type=Path, default=None)
    parser.add_argument("--graph-cache-dir", type=Path, default=None)
    parser.add_argument("--text-cache-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--other-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--use-precomputed-text-features", type=parse_bool, default=None)
    parser.add_argument("--text-cache-batch-size", type=int, default=None)
    parser.add_argument("--rebuild-text-cache", type=parse_bool, default=None)
    parser.add_argument("--use-rst-tensor-cache", type=parse_bool, default=None)
    parser.add_argument("--rst-tensor-cache-dir", type=Path, default=None)
    parser.add_argument("--classifier-hidden-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument("--use-amp", type=parse_bool, default=None)
    parser.add_argument("--eval-every-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--save-best-model", type=parse_bool, default=None)
    parser.add_argument("--dump-predictions", type=parse_bool, default=None)
    parser.add_argument(
        "--selection-metric",
        choices=("qwk", "macro_f1", "micro_f1", "accuracy", "argmax_accuracy", "argmax_macro_f1"),
        default=None,
    )
    parser.add_argument("--dataset-name", choices=("gcdc", "cohesentia"), default=None)
    parser.add_argument("--cohesentia-tau-c", type=float, default=None)
    parser.add_argument("--uncertainty-lambda-max", type=float, default=None)
    parser.add_argument("--uncertainty-warmup-epochs", type=int, default=None)
    parser.add_argument("--uncertainty-anneal-epochs", type=int, default=None)
    parser.add_argument("--cuda-id", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return config


def finalize_args(merged: dict[str, Any], config_path: Path, ablations_config_path: Path) -> argparse.Namespace:
    # Backward-compatible defaults for feature-gating ablations.  Older
    # dataset-specific configs may not list these keys explicitly.
    merged.setdefault("use_span_states", True)
    merged.setdefault("use_nuclearity_embedding", True)
    merged.setdefault("use_path_features", True)
    merged.setdefault("use_topology_features", True)
    merged.setdefault("use_doc_relation_histogram", True)
    merged.setdefault("use_parser_profile", True)
    merged.setdefault("parser_confidence_mode", "original")
    merged.setdefault("parser_profile_mode", "all")

    required = (
        "prefix",
        "source_prefix",
        "experiment_mode",
        "model_name",
        "data_root",
        "raw_data_root",
        "parser_root",
        "sota_data_root",
        "graph_cache_dir",
        "text_cache_dir",
        "output_root",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "encoder_lr",
        "other_lr",
        "head_lr",
        "weight_decay",
        "warmup_ratio",
        "max_length",
        "window_stride",
        "freeze_encoder",
        "text_pooling",
        "use_precomputed_text_features",
        "text_cache_batch_size",
        "rebuild_text_cache",
        "use_rst_tensor_cache",
        "rst_tensor_cache_dir",
        "seed",
        "num_workers",
        "gradient_accumulation_steps",
        "gradient_clip",
        "use_amp",
        "eval_every_epochs",
        "patience",
        "selection_metric",
        "dataset_name",
        "cuda_id",
        "dropout",
        "d_model",
        "relation_embedding_dim",
        "nuclearity_embedding_dim",
        "text_hidden_dim",
        "ablation",
        "relation_label_dropout",
        "relation_dist_dropout",
        "use_text_branch",
        "use_treeformer_branch",
        "use_comp_branch",
        "use_relation_distribution",
        "use_relation_embedding",
        "use_parser_dropout",
        "use_distance_bias",
        "use_ancestor_bias",
        "use_subtree_bias",
        "use_source_relation_bias",
        "use_relation_uncertainty_bias",
        "unordered_tree",
        "node_entropy_mode",
        "node_entropy_source",
        "parser_confidence_mode",
        "use_doc_entropy_features",
        "use_node_non_entropy_parser_stats",
        "use_span_states",
        "use_nuclearity_embedding",
        "use_path_features",
        "use_topology_features",
        "use_doc_relation_histogram",
        "use_parser_profile",
        "parser_profile_mode",
        "path_embedding_dim",
        "treeformer_layers",
        "treeformer_heads",
        "treeformer_bias_dim",
        "treeformer_ffn_multiplier",
        "tree_dist_cap",
        "max_path_depth",
        "use_label_au",
        "use_boundary_au",
        "parser_lambda",
        "beta_h_rel",
        "beta_confidence",
        "beta_margin",
        "beta_ambiguity",
        "beta_rri",
        "boundary_tau",
        "boundary_rho",
        "uncertainty_warmup_epochs",
        "uncertainty_anneal_epochs",
        "uncertainty_lambda_max",
        "classifier_hidden_size",
    )
    missing = [key for key in required if key not in merged]
    if missing:
        raise ValueError(f"Missing required config values: {', '.join(missing)}")

    prefixes = merged["prefix"]
    source_prefixes = merged["source_prefix"]
    if not isinstance(prefixes, list) or not prefixes:
        raise ValueError("`prefix` must be a non-empty list.")
    if not isinstance(source_prefixes, list):
        raise ValueError("`source_prefix` must be a list.")
    invalid = [prefix for prefix in prefixes + source_prefixes if prefix not in PREFIXES]
    if invalid:
        raise ValueError(f"Unsupported prefixes: {', '.join(sorted(set(invalid)))}")
    if merged["dataset_name"] not in {"gcdc", "cohesentia"}:
        raise ValueError(f"Unsupported dataset_name: {merged['dataset_name']}")
    if merged["dataset_name"] == "gcdc" and "cohe" in prefixes + source_prefixes:
        raise ValueError("dataset_name=gcdc cannot be used with prefix `cohe`.")
    if merged["dataset_name"] == "cohesentia" and any(prefix != "cohe" for prefix in prefixes + source_prefixes):
        raise ValueError("dataset_name=cohesentia currently supports only prefix `cohe`.")
    if merged["experiment_mode"] not in {"in_domain", "cross_domain", "multi_domain"}:
        raise ValueError(f"Unsupported experiment_mode: {merged['experiment_mode']}")
    if merged["experiment_mode"] == "cross_domain" and len(source_prefixes) != 1:
        raise ValueError("cross_domain requires exactly one source prefix.")
    if merged["model_name"] not in MODEL_NAME_MAP:
        raise ValueError(f"Unsupported model_name: {merged['model_name']}")
    if merged["text_pooling"] not in {"cls", "mean"}:
        raise ValueError(f"Unsupported text_pooling: {merged['text_pooling']}")
    if merged["use_precomputed_text_features"] and merged["text_pooling"] != "mean":
        raise ValueError("ReCo precomputed SOTA-style features require text_pooling=mean.")
    if merged["use_precomputed_text_features"] and not bool(merged["freeze_encoder"]):
        raise ValueError("ReCo precomputed SOTA-style features require freeze_encoder=true.")
    if merged["node_entropy_mode"] not in {"original", "zero", "shuffle"}:
        raise ValueError(f"Unsupported node_entropy_mode: {merged['node_entropy_mode']}")
    if merged["node_entropy_source"] not in {"relation", "nuclearity", "split"}:
        raise ValueError(f"Unsupported node_entropy_source: {merged['node_entropy_source']}")
    if merged["parser_confidence_mode"] not in {"original", "zero", "uniform", "shuffle"}:
        raise ValueError(f"Unsupported parser_confidence_mode: {merged['parser_confidence_mode']}")
    if merged["parser_profile_mode"] not in {"all", "confidence", "size"}:
        raise ValueError(f"Unsupported parser_profile_mode: {merged['parser_profile_mode']}")
    if int(merged["gradient_accumulation_steps"]) < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")
    if int(merged["eval_every_epochs"]) < 1:
        raise ValueError("eval_every_epochs must be >= 1.")
    merged["use_amp"] = bool(merged["use_amp"])
    merged["save_best_model"] = bool(merged.get("save_best_model", True))
    merged["dump_predictions"] = bool(merged.get("dump_predictions", False))
    merged["eval_every_epochs"] = int(merged["eval_every_epochs"])

    fold = merged.get("fold")
    if fold is not None and int(fold) not in range(1, 11):
        raise ValueError(f"Unsupported fold: {fold}. Expected 1-10.")
    merged["fold"] = fold

    merged["data_root"] = Path(merged["data_root"])
    merged["raw_data_root"] = Path(merged["raw_data_root"])
    merged["parser_root"] = Path(merged["parser_root"])
    merged["sota_data_root"] = Path(merged["sota_data_root"])
    merged["graph_cache_dir"] = Path(merged["graph_cache_dir"])
    merged["text_cache_dir"] = Path(merged["text_cache_dir"])
    merged["rst_tensor_cache_dir"] = Path(merged["rst_tensor_cache_dir"])
    merged["output_root"] = Path(merged["output_root"])

    if merged.get("device"):
        device = merged["device"]
    elif torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        cuda_id = int(merged["cuda_id"])
        if cuda_id >= device_count:
            raise ValueError(f"Requested cuda_id={cuda_id}, but only {device_count} CUDA device(s) exist.")
        device = f"cuda:{cuda_id}"
    else:
        device = "cpu"
    merged["device"] = device
    merged["config"] = config_path
    merged["ablations_config"] = ablations_config_path
    return argparse.Namespace(**merged)


def resolve_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    merged = load_yaml_config(cli_args.config)
    ablations = load_yaml_config(cli_args.ablations_config)
    ablation_name = cli_args.ablation or merged.get("ablation", "reco")
    if ablation_name not in ablations:
        raise ValueError(f"Unknown ablation `{ablation_name}`. Available: {', '.join(sorted(ablations))}")
    merged.update(ablations[ablation_name] or {})
    merged["ablation"] = ablation_name
    for key, value in vars(cli_args).items():
        if key not in {"config", "ablations_config", "ablation"} and value is not None:
            merged[key] = value
    return finalize_args(merged, cli_args.config, cli_args.ablations_config)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def make_checkpoint_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_checkpoint_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_checkpoint_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def format_metric_block(metrics: dict[str, float]) -> str:
    parts = []
    for key in (
        "loss",
        "ce_loss",
        "uncertainty_loss",
        "uncertainty_weight",
        "mean_target_uncertainty",
        "mean_pred_uncertainty",
        "corr_uncertainty_target",
        "macro_f1",
        "accuracy",
        "qwk",
        "ece",
        "brier",
        "nll",
    ):
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " ".join(parts)


def load_rating_map(prefix: str, raw_data_root: Path) -> dict[str, dict[str, Any]]:
    raw_name = RAW_PREFIX_NAMES[prefix]
    merged: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "test"):
        path = raw_data_root / f"{raw_name}_{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw CSV for {prefix}: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                text_id = str(row["text_id"])
                ratings = [int(row["ratingA1"]), int(row["ratingA2"]), int(row["ratingA3"])]
                mean_rating = sum(ratings) / 3.0
                recomputed = 1 if mean_rating <= 1.8 else 2 if mean_rating <= 2.2 else 3
                label_a = int(row["labelA"])
                if recomputed != label_a:
                    raise ValueError(
                        f"labelA sanity check failed for {prefix}_{text_id}: "
                        f"ratings={ratings}, recomputed={recomputed}, csv_labelA={label_a}"
                    )
                merged[text_id] = {
                    "text_id": text_id,
                    "labelA": label_a,
                    "ratings": ratings,
                }
    return merged


def build_relation_vocab(graph_bundles: dict[str, dict[str, Any]]) -> dict[str, int]:
    labels = set()
    for graphs in graph_bundles.values():
        for graph in graphs.values():
            for node_index in graph["internal_indices"]:
                label = graph["nodes"][node_index].get("relation_label")
                if label:
                    labels.add(str(label))
    ordered = ["<UNK>", "<ROOT_PATH>", *sorted(labels)]
    relation_vocab = {}
    for index, label in enumerate(ordered):
        relation_vocab[label] = index
    return relation_vocab


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return 0.0
    corr = np.corrcoef(x_arr, y_arr)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def spearman_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    return pearson_corr(rankdata(np.asarray(x, dtype=np.float64)).tolist(), rankdata(np.asarray(y, dtype=np.float64)).tolist())


def expected_calibration_error(confidences: list[float], correctness: list[int], num_bins: int = 15) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    total = len(conf)
    ece = 0.0
    for idx in range(num_bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == num_bins - 1:
            mask = (conf >= left) & (conf <= right)
        else:
            mask = (conf >= left) & (conf < right)
        if not np.any(mask):
            continue
        bin_acc = corr[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return float(ece)


def risk_coverage_auc(uncertainty: list[float], errors: list[int]) -> float:
    if not uncertainty:
        return 0.0
    order = np.argsort(np.asarray(uncertainty, dtype=np.float64))
    sorted_errors = np.asarray(errors, dtype=np.float64)[order]
    coverage = np.arange(1, len(sorted_errors) + 1, dtype=np.float64) / len(sorted_errors)
    cumulative_risk = np.cumsum(sorted_errors) / np.arange(1, len(sorted_errors) + 1, dtype=np.float64)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(cumulative_risk, coverage))
    return float(np.trapz(cumulative_risk, coverage))


def decode_gcdc_threshold_predictions(
    probs: list[list[float]] | np.ndarray,
    low_threshold: float = 1.8,
    high_threshold: float = 2.2,
) -> tuple[list[int], list[float]]:
    probs_arr = np.asarray(probs, dtype=np.float64)
    expected_scores = (probs_arr * np.arange(1, probs_arr.shape[1] + 1, dtype=np.float64)).sum(axis=1)
    pred_labels: list[int] = []
    for score in expected_scores.tolist():
        if score <= low_threshold:
            pred_labels.append(0)
        elif score <= high_threshold:
            pred_labels.append(1)
        else:
            pred_labels.append(2)
    return pred_labels, expected_scores.tolist()


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    num_labels: int,
    prefix: str = "",
) -> dict[str, float]:
    labels = list(range(num_labels))
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="micro",
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    class_f1 = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )[2].tolist()
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    if math.isnan(qwk):
        qwk = 0.0
    return {
        f"{prefix}micro_precision": float(micro_p),
        f"{prefix}micro_recall": float(micro_r),
        f"{prefix}micro_f1": float(micro_f1),
        f"{prefix}macro_precision": float(macro_p),
        f"{prefix}macro_recall": float(macro_r),
        f"{prefix}macro_f1": float(macro_f1),
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}qwk": qwk,
        f"{prefix}f1_low": float(class_f1[0]) if len(class_f1) > 0 else 0.0,
        f"{prefix}f1_medium": float(class_f1[1]) if len(class_f1) > 1 else 0.0,
        f"{prefix}f1_high": float(class_f1[2]) if len(class_f1) > 2 else 0.0,
    }


def compute_metrics(
    y_true: list[int],
    y_pred_argmax: list[int],
    probs: list[list[float]],
    mu_ratings: list[float],
    au_coh_values: list[float],
    target_uncertainty_values: list[float],
    uncertainty_mass: list[float],
    doc_parser_eu: list[float],
    num_labels: int,
    dataset_name: str,
) -> dict[str, float]:
    probs_arr = np.asarray(probs, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.int64)
    _, expected_scores = decode_gcdc_threshold_predictions(probs_arr)
    y_pred, expected_scores = decode_gcdc_threshold_predictions(probs_arr)
    confidences = probs_arr.max(axis=1).tolist()
    correctness = (np.asarray(y_pred) == y_arr).astype(np.int64).tolist()
    argmax_correctness = (np.asarray(y_pred_argmax) == y_arr).astype(np.int64).tolist()
    one_hot = np.zeros_like(probs_arr)
    one_hot[np.arange(len(y_arr)), y_arr] = 1.0
    nll = float((-np.log(np.clip(probs_arr[np.arange(len(y_arr)), y_arr], 1e-8, 1.0))).mean())
    brier = float(np.mean(np.sum((probs_arr - one_hot) ** 2, axis=1)))
    mae = float(np.mean(np.abs(expected_scores - np.asarray(mu_ratings, dtype=np.float64))))
    error_list = [1 - value for value in correctness]
    metrics = {
        "mae": mae,
        "spearman": spearman_corr(expected_scores, mu_ratings),
        "ece": expected_calibration_error(confidences, correctness, num_bins=15),
        "brier": brier,
        "nll": nll,
        "mean_target_uncertainty": float(np.mean(target_uncertainty_values)) if target_uncertainty_values else 0.0,
        "mean_pred_uncertainty": float(np.mean(uncertainty_mass)) if uncertainty_mass else 0.0,
        "corr_uncertainty_error": pearson_corr(uncertainty_mass, [float(value) for value in error_list]),
        "corr_uncertainty_target": pearson_corr(uncertainty_mass, target_uncertainty_values),
        "corr_uncertainty_au": pearson_corr(uncertainty_mass, au_coh_values),
        "risk_coverage_auc": risk_coverage_auc(uncertainty_mass, error_list),
        "corr_parser_eu_target": pearson_corr(doc_parser_eu, target_uncertainty_values),
        "corr_parser_eu_au": pearson_corr(doc_parser_eu, au_coh_values),
    }
    metrics.update(compute_classification_metrics(y_true, y_pred, num_labels))
    metrics.update(compute_classification_metrics(y_true, y_pred_argmax, num_labels, prefix="argmax_"))
    metrics["argmax_corr_uncertainty_error"] = pearson_corr(
        uncertainty_mass,
        [float(value) for value in (1 - np.asarray(argmax_correctness)).tolist()],
    )
    return metrics


def is_better(current: dict[str, float], best: dict[str, float] | None, metric: str) -> bool:
    if best is None:
        return True
    if not math.isclose(current[metric], best[metric]):
        return current[metric] > best[metric]
    if "ece" in current and "ece" in best and not math.isclose(current["ece"], best["ece"]):
        return current["ece"] < best["ece"]
    if not math.isclose(current["accuracy"], best["accuracy"]):
        return current["accuracy"] > best["accuracy"]
    return current["loss"] < best["loss"]


def label_mapping() -> dict[str, int]:
    return {"1": 0, "2": 1, "3": 2}


def cohesentia_label_from_raw_score(raw_score_5: int) -> int:
    score = int(raw_score_5)
    if score in {1, 2}:
        return 0
    if score in {3, 4}:
        return 1
    if score == 5:
        return 2
    raise ValueError(f"COHESENTIA raw_score_5 must be in {{1,2,3,4,5}}, got {score}.")


def sanity_check_cohesentia_record(record: dict[str, Any], label: int) -> None:
    raw_score_5 = int(record["raw_score_5"])
    expected_label = cohesentia_label_from_raw_score(raw_score_5)
    if label != expected_label:
        raise ValueError(
            f"COHESENTIA label mapping mismatch for {record.get('doc_id', record.get('text_id'))}: "
            f"raw_score_5={raw_score_5}, label={label}, expected={expected_label}"
        )
    score_id = record.get("score_id")
    if score_id is not None and str(score_id) in {"1", "2", "3"}:
        score_label = int(score_id) - 1
        if score_label != expected_label:
            raise ValueError(
                f"COHESENTIA score_id mismatch for {record.get('doc_id', record.get('text_id'))}: "
                f"raw_score_5={raw_score_5}, score_id={score_id}, expected_label={expected_label}"
            )


def build_examples(
    records: list[dict[str, Any]],
    prefix: str,
    feature_store: dict[str, dict[str, Any]],
    rating_map: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    labels = label_mapping()
    for record in records:
        text_id = str(record["text_id"])
        doc_id = f"{prefix}_{text_id}"
        if args.dataset_name == "cohesentia":
            raw_score_5 = int(record["raw_score_5"])
            label = cohesentia_label_from_raw_score(raw_score_5)
            sanity_check_cohesentia_record(record=record, label=label)
            q_emp = np.zeros(3, dtype=np.float64)
            q_emp[label] = 1.0
            examples.append(
                {
                    **feature_store[doc_id],
                    "label": label,
                    "label_a": label,
                    "ratings": [label + 1],
                    "mu_rating": float(label + 1),
                    "q_emp": [float(value) for value in q_emp.tolist()],
                    "au_coh": 0.0,
                    "raw_score_5": raw_score_5,
                    "prefix": prefix,
                }
            )
            continue
        ratings_payload = rating_map[text_id]
        if int(record["score"]) != int(ratings_payload["labelA"]):
            raise ValueError(
                f"Split label mismatch for {doc_id}: split score={record['score']}, "
                f"raw labelA={ratings_payload['labelA']}"
            )
        q_emp = np.zeros(3, dtype=np.float64)
        for rating in ratings_payload["ratings"]:
            q_emp[rating - 1] += 1.0 / 3.0
        non_zero = q_emp[q_emp > 0]
        au_rating = 0.0
        if non_zero.size > 0:
            au_rating = float(-(non_zero * np.log(non_zero)).sum() / math.log(3.0))
        mu_rating = float(sum(ratings_payload["ratings"]) / 3.0)
        examples.append(
            {
                **feature_store[doc_id],
                "label": labels[str(record["score"])],
                "label_a": int(ratings_payload["labelA"]) - 1,
                "ratings": list(ratings_payload["ratings"]),
                "mu_rating": mu_rating,
                "q_emp": [float(value) for value in q_emp.tolist()],
                "au_coh": au_rating,
                "raw_score_5": 0,
                "prefix": prefix,
            }
        )
    return examples


def threshold_boundary_au(mu_rating: torch.Tensor, boundary_tau: float) -> torch.Tensor:
    lower = torch.abs(mu_rating - 1.8)
    upper = torch.abs(mu_rating - 2.2)
    distance = torch.minimum(lower, upper)
    return torch.clamp(1.0 - distance / max(float(boundary_tau), 1e-8), min=0.0, max=1.0)


def combined_label_au(au_rating: torch.Tensor, mu_rating: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if not args.use_label_au:
        return torch.zeros_like(au_rating)
    if args.use_boundary_au:
        au_boundary = threshold_boundary_au(mu_rating=mu_rating, boundary_tau=float(args.boundary_tau))
    else:
        au_boundary = torch.zeros_like(au_rating)
    au = au_rating + (1.0 - au_rating) * float(args.boundary_rho) * au_boundary
    return torch.clamp(au, min=0.0, max=1.0)


def build_gcdc_target_uncertainty(
    q_emp: torch.Tensor,
    mu_rating: torch.Tensor,
    au_coh: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    del q_emp
    return combined_label_au(au_rating=au_coh, mu_rating=mu_rating, args=args)


def build_target_uncertainty(
    batch_payload: dict[str, Any],
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.dataset_name == "cohesentia":
        return torch.zeros_like(batch_payload["mu_rating"])
    return build_gcdc_target_uncertainty(
        q_emp=batch_payload["q_emp"],
        mu_rating=batch_payload["mu_rating"],
        au_coh=batch_payload["au_coh"],
        args=args,
    )


def uncertainty_regularizer_weight(epoch: int | None, args: argparse.Namespace) -> float:
    # ReCo is intentionally a CE-only comparison against SOTA text-only:
    # the only added signal is RST-derived features in the classifier input.
    # Keep this at zero even if legacy uncertainty config keys are present.
    return 0.0


def compute_training_loss(
    probs: torch.Tensor,
    pred_uncertainty: torch.Tensor,
    labels: torch.Tensor,
    target_uncertainty: torch.Tensor,
    uncertainty_weight: float,
    logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits is not None:
        ce_loss = F.cross_entropy(logits, labels)
    else:
        ce_loss = F.nll_loss(torch.log(probs.clamp_min(1e-8)), labels)
    uncertainty_loss = F.mse_loss(pred_uncertainty, target_uncertainty)
    total = ce_loss
    return total, ce_loss, uncertainty_loss


def move_batch_to_device(batch, device: torch.device) -> dict[str, Any]:
    moved = batch.to(device)
    return {
        "batch": moved,
        "labels": moved.labels,
        "q_emp": moved.q_emp,
        "mu_rating": moved.mu_ratings,
        "au_coh": moved.au_coh,
        "raw_score_5": moved.raw_score_5,
    }


def print_batch_sanity(ablation: str, batch, dataset_size: int, rst_windows_enabled: bool) -> None:
    input_ids = batch.doc_input_ids
    attention_mask = batch.doc_attention_mask
    print(f"=== Final / {ablation} batch sanity ===", flush=True)
    print(f"batch size:           {input_ids.shape[0]}", flush=True)
    print(f"seq length (padded):  {input_ids.shape[1]}", flush=True)
    print(f"actual tokens:        {attention_mask.sum().item()}", flush=True)
    print(f"avg tokens/sample:    {attention_mask.sum().item() / input_ids.shape[0]:.1f}", flush=True)
    print(f"#docs in dataset:     {dataset_size}", flush=True)
    print(f"cached doc repr:      {tuple(batch.doc_repr.shape) if batch.doc_repr.numel() else 'disabled'}", flush=True)
    print(
        f"cached subword states:{tuple(batch.global_states.shape) if batch.global_states.numel() else 'disabled'}",
        flush=True,
    )
    print(
        f"cached RST tensors:   {'enabled' if getattr(batch, 'rst_tensors', None) and batch.rst_tensors[0] is not None else 'disabled'}",
        flush=True,
    )
    print(f"RST window encoder:   {'enabled' if rst_windows_enabled else 'skipped'}", flush=True)
    print(f"window batch shape:   {tuple(batch.window_input_ids.shape)}", flush=True)
    print(f"window actual tokens: {batch.window_attention_mask.sum().item()}", flush=True)


def build_model(args: argparse.Namespace, relation_vocab: dict[str, int], num_labels: int):
    return ReCo(
        hf_model_name=MODEL_NAME_MAP[args.model_name],
        num_labels=num_labels,
        relation_vocab=relation_vocab,
        hidden_size=args.d_model,
        dropout=args.dropout,
        relation_embedding_dim=args.relation_embedding_dim,
        nuclearity_embedding_dim=args.nuclearity_embedding_dim,
        path_embedding_dim=args.path_embedding_dim,
        text_hidden_dim=args.text_hidden_dim,
        treeformer_layers=args.treeformer_layers,
        treeformer_heads=args.treeformer_heads,
        treeformer_bias_dim=args.treeformer_bias_dim,
        treeformer_ffn_multiplier=args.treeformer_ffn_multiplier,
        tree_dist_cap=args.tree_dist_cap,
        max_path_depth=args.max_path_depth,
        relation_label_dropout=args.relation_label_dropout,
        relation_dist_dropout=args.relation_dist_dropout,
        use_text_branch=bool(args.use_text_branch),
        use_treeformer_branch=bool(args.use_treeformer_branch),
        use_comp_branch=bool(args.use_comp_branch),
        use_relation_distribution=bool(args.use_relation_distribution),
        use_relation_embedding=bool(args.use_relation_embedding),
        use_parser_dropout=bool(args.use_parser_dropout),
        use_distance_bias=bool(args.use_distance_bias),
        use_ancestor_bias=bool(args.use_ancestor_bias),
        use_subtree_bias=bool(args.use_subtree_bias),
        use_source_relation_bias=bool(args.use_source_relation_bias),
        use_relation_uncertainty_bias=bool(args.use_relation_uncertainty_bias),
        unordered_tree=bool(args.unordered_tree),
        node_entropy_mode=args.node_entropy_mode,
        node_entropy_source=args.node_entropy_source,
        parser_confidence_mode=args.parser_confidence_mode,
        use_doc_entropy_features=bool(args.use_doc_entropy_features),
        use_node_non_entropy_parser_stats=bool(args.use_node_non_entropy_parser_stats),
        use_span_states=bool(args.use_span_states),
        use_nuclearity_embedding=bool(args.use_nuclearity_embedding),
        use_path_features=bool(args.use_path_features),
        use_topology_features=bool(args.use_topology_features),
        use_doc_relation_histogram=bool(args.use_doc_relation_histogram),
        use_parser_profile=bool(args.use_parser_profile),
        parser_profile_mode=args.parser_profile_mode,
        freeze_encoder=bool(args.freeze_encoder),
        text_pooling=args.text_pooling,
        use_precomputed_text_features=bool(args.use_precomputed_text_features),
        classifier_hidden_size=int(args.classifier_hidden_size),
    )


def build_optimizer(model: ReCo, encoder_lr: float, other_lr: float, head_lr: float, weight_decay: float):
    param_groups = [
        {"params": list(model.encoder_parameters()), "lr": encoder_lr},
        {"params": list(model.other_parameters()), "lr": other_lr},
        {"params": list(model.head_parameters()), "lr": head_lr},
    ]
    param_groups = [group for group in param_groups if group["params"]]
    return torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
    )


def get_sota_fusion_dir(args: argparse.Namespace | None = None) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if args is not None:
        data_root = Path(args.sota_data_root).resolve()
        for parent in (data_root, *data_root.parents):
            if parent.name == "fusion" and (parent / "task_dataset.py").exists():
                return parent
    candidates = [repo_root / "models" / "entyrelcoh"]
    for candidate in candidates:
        if (candidate / "task_dataset.py").exists():
            return candidate
    return candidates[0]


def candidate_sota_text_roots(args: argparse.Namespace | None = None) -> list[Path]:
    """Return text-only split roots, preferring the configured path."""
    repo_root = Path(__file__).resolve().parents[2]
    roots: list[Path] = []
    if args is not None:
        roots.append(Path(args.sota_data_root))
    roots.append(repo_root / "data" / "prepared" / "text_only")
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def resolve_sota_text_root(args: argparse.Namespace, prefix: str, fold_index: int | None = None) -> Path:
    """Find a SOTA text-only root containing the requested prefix/fold."""
    missing: list[str] = []
    for root in candidate_sota_text_roots(args):
        dataset_dir = root / f"gcdc_{prefix}"
        if fold_index is None:
            if dataset_dir.exists():
                return root
            missing.append(str(dataset_dir))
            continue
        fold_dir = dataset_dir / str(fold_index)
        paths = [fold_dir / f"{split}.json" for split in ("train", "dev", "test")]
        if all(path.exists() for path in paths):
            return root
        missing.extend(str(path) for path in paths if not path.exists())
    raise FileNotFoundError(
        "Missing SOTA-compatible text split files under all candidate roots. "
        f"Missing examples: {missing[:9]}"
    )


def load_sota_sent_dataset_cls(args: argparse.Namespace | None = None):
    fusion_dir = get_sota_fusion_dir(args)
    module_path = fusion_dir / "task_dataset.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find SOTA task_dataset.py: {module_path}")
    if str(fusion_dir) not in sys.path:
        sys.path.insert(0, str(fusion_dir))
    spec = importlib.util.spec_from_file_location("entyrelcoh_task_dataset", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import SOTA SentDataset from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SentDataset


def load_sota_relation_labels(args: argparse.Namespace | None = None) -> list[str]:
    fusion_dir = get_sota_fusion_dir(args)
    paths = [
        (
            fusion_dir / "data" / "parser" / "pdtb3" / "exp" / "l2" / "labels.txt",
            fusion_dir / "data" / "parser" / "pdtb3" / "exp" / "labels.txt",
        ),
        (
            fusion_dir / "data" / "parser" / "pdtb3" / "imp" / "l2" / "labels.txt",
            fusion_dir / "data" / "parser" / "pdtb3" / "imp" / "labels.txt",
        ),
    ]
    labels: set[str] = set()
    for primary_path, fallback_path in paths:
        path = primary_path if primary_path.exists() else fallback_path
        if not path.exists():
            raise FileNotFoundError(f"Missing SOTA relation label file: {primary_path} or {fallback_path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                label = line.strip()
                if label:
                    labels.add(label.lower())
    labels.add("entity")
    return sorted(labels)


class SotaIndexedDataset(Dataset):
    """A thin wrapper preserving EntyRelCoh SentDataset samples plus indices."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.text_ids = [str(text_id) for text_id in getattr(base_dataset, "text_ids", [])]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        return (*self.base_dataset[index], torch.tensor(index, dtype=torch.long))


def get_sota_compatible_dataloader(dataset: Dataset, args: argparse.Namespace, mode: str = "train") -> DataLoader:
    print("  {} dataset length: ".format(mode), len(dataset))
    if mode.lower() == "train":
        sampler = RandomSampler(dataset)
        batch_size = args.batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batch_size
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
    )


def build_sota_sent_dataset(split_path: Path, args: argparse.Namespace, tokenizer, encoder):
    SentDataset = load_sota_sent_dataset_cls(args)
    params = {
        "max_text_length": int(args.max_length),
        "max_sent_num": 15,
        "max_rel_num": 36,
        "rel_list": load_sota_relation_labels(args),
        "label_list": ["1", "2", "3"],
        "tokenizer": tokenizer,
        "encoder": encoder,
        "pooler_type": "avg",
        "no_rel": False,
        "no_enty": False,
    }
    return SotaIndexedDataset(SentDataset(str(split_path), params=params))


def build_sota_text_encoder(args: argparse.Namespace):
    model_name_or_path = MODEL_NAME_MAP[args.model_name]
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoder = AutoModel.from_pretrained(model_name_or_path, config=config).to(torch.device(args.device))
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    return tokenizer, encoder


def sota_dataset_split_paths(args: argparse.Namespace, prefix: str, fold_index: int) -> dict[str, Path]:
    root = resolve_sota_text_root(args, prefix, fold_index) / f"gcdc_{prefix}" / str(fold_index)
    paths = {split: root / f"{split}.json" for split in ("train", "dev", "test")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing SOTA-compatible text split files under `sota_data_root`. "
            f"Missing: {missing}"
        )
    return paths


def basic_sota_metrics(
    y_true: list[int],
    y_pred: list[int],
    probs: list[list[float]],
    loss: float,
) -> dict[str, float]:
    metrics = compute_classification_metrics(y_true, y_pred, num_labels=3)
    metrics.update(compute_classification_metrics(y_true, y_pred, num_labels=3, prefix="argmax_"))
    probs_arr = np.asarray(probs, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.int64)
    one_hot = np.zeros_like(probs_arr)
    if len(y_arr):
        one_hot[np.arange(len(y_arr)), y_arr] = 1.0
        nll = float((-np.log(np.clip(probs_arr[np.arange(len(y_arr)), y_arr], 1e-8, 1.0))).mean())
        brier = float(np.mean(np.sum((probs_arr - one_hot) ** 2, axis=1)))
    else:
        nll = 0.0
        brier = 0.0
    metrics.update(
        {
            "loss": float(loss),
            "ce_loss": float(loss),
            "uncertainty_loss": 0.0,
            "uncertainty_weight": 0.0,
            "mae": 0.0,
            "spearman": 0.0,
            "ece": 0.0,
            "brier": brier,
            "nll": nll,
            "corr_uncertainty_error": 0.0,
            "argmax_corr_uncertainty_error": 0.0,
            "corr_uncertainty_target": 0.0,
            "corr_uncertainty_au": 0.0,
            "risk_coverage_auc": 0.0,
            "corr_parser_eu_target": 0.0,
            "corr_parser_eu_au": 0.0,
            "mean_target_uncertainty": 0.0,
            "mean_pred_uncertainty": 0.0,
        }
    )
    return metrics


def sota_batch_to_inputs(
    batch,
    dataset: SotaIndexedDataset,
    structural_lookup: dict[str, dict[str, Any]] | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]] | None, list[dict[str, Any] | None] | None]:
    batch_features = tuple(t.to(device) for t in batch)
    doc_vectors = batch_features[0]
    labels = batch_features[7]
    indices = batch_features[8].detach().cpu().tolist()
    if structural_lookup is None:
        return doc_vectors, labels, None, None
    graphs = []
    rst_tensors = []
    for index in indices:
        text_id = dataset.text_ids[int(index)]
        item = structural_lookup[text_id]
        graphs.append(item["graph"])
        rst_tensors.append(item["rst_tensors"])
    return doc_vectors, labels, graphs, rst_tensors


def sota_compatible_evaluate(
    model: ReCo,
    args: argparse.Namespace,
    loader: DataLoader,
    structural_lookup: dict[str, dict[str, Any]] | None,
    desc: str,
) -> dict[str, float]:
    model.eval()
    device = torch.device(args.device)
    y_true: list[int] = []
    y_pred: list[int] = []
    probs_all: list[list[float]] = []
    total_loss = 0.0
    total_examples = 0
    for batch in tqdm(loader, desc=desc):
        doc_vectors, labels, graphs, rst_tensors = sota_batch_to_inputs(
            batch=batch,
            dataset=loader.dataset,
            structural_lookup=structural_lookup,
            device=device,
        )
        with torch.no_grad():
            outputs = model.forward_sota(doc_vectors=doc_vectors, graphs=graphs, rst_tensors=rst_tensors)
            loss = F.cross_entropy(outputs["logits"], labels)
        batch_size = int(labels.size(0))
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(outputs["preds"].detach().cpu().tolist())
        probs_all.extend(outputs["probs"].detach().cpu().tolist())
    print(np.asarray(y_true).shape, np.asarray(y_pred).shape)
    return basic_sota_metrics(y_true, y_pred, probs_all, total_loss / max(total_examples, 1))


@torch.no_grad()
def sota_compatible_prediction_records(
    model: ReCo,
    args: argparse.Namespace,
    loader: DataLoader,
    structural_lookup: dict[str, dict[str, Any]] | None,
    prefix: str,
    fold_index: int,
    split: str,
) -> list[dict[str, Any]]:
    model.eval()
    device = torch.device(args.device)
    records: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=f"dump {split}", leave=False):
        doc_vectors, labels, graphs, rst_tensors = sota_batch_to_inputs(
            batch=batch,
            dataset=loader.dataset,
            structural_lookup=structural_lookup,
            device=device,
        )
        indices = batch[-1].detach().cpu().tolist()
        outputs = model.forward_sota(doc_vectors=doc_vectors, graphs=graphs, rst_tensors=rst_tensors)
        logits = outputs["logits"].detach().cpu().tolist()
        probs = outputs["probs"].detach().cpu().tolist()
        preds = outputs["preds"].detach().cpu().tolist()
        gold = labels.detach().cpu().tolist()
        for row, index in enumerate(indices):
            text_id = str(loader.dataset.text_ids[int(index)])
            records.append(
                {
                    "dataset_name": str(args.dataset_name),
                    "prefix": prefix,
                    "fold": int(fold_index),
                    "split": split,
                    "doc_id": f"{prefix}_{text_id}",
                    "text_id": text_id,
                    "gold": int(gold[row]),
                    "pred": int(preds[row]),
                    "logits": [float(value) for value in logits[row]],
                    "probs": [float(value) for value in probs[row]],
                }
            )
    return records


def dump_prediction_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_sota_optimizer(model: ReCo, args: argparse.Namespace, num_training_steps: int):
    no_decay = ["bias", "LayerNorm.weigh"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=float(args.head_lr))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * float(args.warmup_ratio)),
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def sota_compatible_train_fold(
    model: ReCo,
    args: argparse.Namespace,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    structural_lookup: dict[str, dict[str, Any]] | None,
    fold_dir: Path,
    prefix: str,
    fold_index: int,
) -> dict[str, Any]:
    device = torch.device(args.device)
    optimizer, scheduler = build_sota_optimizer(
        model=model,
        args=args,
        num_training_steps=int(len(train_loader) * int(args.epochs)),
    )
    history: list[dict[str, Any]] = []
    best_dev_metrics: dict[str, float] | None = None
    best_test_metrics: dict[str, float] | None = None
    best_selection_metrics: dict[str, float] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    tr_loss = 0.0
    print_step = max(int(len(train_loader) // 4), 1)
    log_path = fold_dir / "training.log"
    epoch_metrics_path = fold_dir / "epoch_metrics.jsonl"

    for epoch in tqdm(range(1, int(args.epochs) + 1), desc="Epoch"):
        model.train()
        model.zero_grad()
        for batch in tqdm(train_loader, desc="Iteration"):
            doc_vectors, labels, graphs, rst_tensors = sota_batch_to_inputs(
                batch=batch,
                dataset=train_loader.dataset,
                structural_lookup=structural_lookup,
                device=device,
            )
            outputs = model.forward_sota(doc_vectors=doc_vectors, graphs=graphs, rst_tensors=rst_tensors)
            loss = F.cross_entropy(outputs["logits"], labels)
            loss.backward()
            global_step += 1
            tr_loss += float(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if global_step % print_step == 0:
                print(" Current loss=%.4f, global average loss=%.4f" % (loss.item(), tr_loss / global_step))

        dev_metrics = sota_compatible_evaluate(
            model=model,
            args=args,
            loader=dev_loader,
            structural_lookup=structural_lookup,
            desc="dev",
        )
        test_metrics = sota_compatible_evaluate(
            model=model,
            args=args,
            loader=test_loader,
            structural_lookup=structural_lookup,
            desc="test",
        )
        print()
        print(" Dev Acc=%.4f, F1=%.4f" % (dev_metrics["accuracy"], dev_metrics["macro_f1"]))
        print(" Test Acc=%.4f, F1=%.4f" % (test_metrics["accuracy"], test_metrics["macro_f1"]))
        epoch_record = {
            "epoch": epoch,
            "train_metrics": {
                "loss": tr_loss / max(global_step, 1),
                "ce_loss": tr_loss / max(global_step, 1),
                "uncertainty_loss": 0.0,
                "uncertainty_weight": 0.0,
            },
            "dev_metrics": dev_metrics,
            "test_metrics": test_metrics,
        }
        history.append(epoch_record)
        save_json(fold_dir / "history.json", history)
        append_jsonl(epoch_metrics_path, epoch_record)
        append_log_line(
            log_path,
            f"epoch={epoch} train[{format_metric_block(epoch_record['train_metrics'])}] "
            f"dev[{format_metric_block(dev_metrics)}] test[{format_metric_block(test_metrics)}]",
        )
        if is_better(dev_metrics, best_selection_metrics, metric=args.selection_metric):
            best_selection_metrics = dev_metrics
            best_dev_metrics = dev_metrics
            best_test_metrics = test_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            if args.dump_predictions:
                pred_dir = fold_dir / "predictions"
                dump_prediction_records(
                    pred_dir / "train_predictions.jsonl",
                    sota_compatible_prediction_records(
                        model=model,
                        args=args,
                        loader=train_loader,
                        structural_lookup=structural_lookup,
                        prefix=prefix,
                        fold_index=fold_index,
                        split="train",
                    ),
                )
                dump_prediction_records(
                    pred_dir / "dev_predictions.jsonl",
                    sota_compatible_prediction_records(
                        model=model,
                        args=args,
                        loader=dev_loader,
                        structural_lookup=structural_lookup,
                        prefix=prefix,
                        fold_index=fold_index,
                        split="dev",
                    ),
                )
                dump_prediction_records(
                    pred_dir / "test_predictions.jsonl",
                    sota_compatible_prediction_records(
                        model=model,
                        args=args,
                        loader=test_loader,
                        structural_lookup=structural_lookup,
                        prefix=prefix,
                        fold_index=fold_index,
                        split="test",
                    ),
                )
                save_json(pred_dir / "prediction_dump_info.json", {"best_epoch": best_epoch})

            if args.save_best_model:
                checkpoint_dir = fold_dir / "good" / f"checkpoint-{epoch}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(args.patience):
                append_log_line(
                    log_path,
                    f"early_stop epoch={epoch} selection_metric={args.selection_metric} "
                    f"best_epoch={best_epoch}",
                )
                break

    print(
        " Best dev: Epoch=%d, %s=%.4f\n"
        % (best_epoch, args.selection_metric, (best_dev_metrics or {}).get(args.selection_metric, 0.0))
    )
    print(
        " Best test: Epoch=%d, %s=%.4f\n"
        % (best_epoch, args.selection_metric, (best_test_metrics or {}).get(args.selection_metric, 0.0))
    )
    if best_test_metrics is None:
        raise RuntimeError("No SOTA-compatible evaluation was completed.")
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "fold": None,
        "best_epoch": best_epoch,
        "selection_split": "dev",
        "selection_metric": args.selection_metric,
        "best_selection_metrics": best_selection_metrics,
        "best_dev_metrics": best_dev_metrics,
        "test_metrics": best_test_metrics,
    }


def train_one_epoch(
    model: ReCo,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    gradient_clip: float,
    args: argparse.Namespace,
    epoch: int,
    progress_desc: str,
    amp_enabled: bool,
    scaler: torch.cuda.amp.GradScaler,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_uncertainty_loss = 0.0
    total_target_uncertainty = 0.0
    total_pred_uncertainty = 0.0
    total_examples = 0
    uncertainty_weight = uncertainty_regularizer_weight(epoch=epoch, args=args)
    progress = tqdm(loader, desc=progress_desc, leave=False)
    printed_sanity = False
    accumulation_steps = max(int(args.gradient_accumulation_steps), 1)
    optimizer.zero_grad(set_to_none=True)
    for step_index, batch in enumerate(progress, start=1):
        if not printed_sanity:
            print_batch_sanity(
                ablation=str(args.ablation),
                batch=batch,
                dataset_size=len(loader.dataset),
                rst_windows_enabled=bool(args.use_treeformer_branch or args.use_comp_branch),
            )
            print(
                f"gradient accumulation: {accumulation_steps} "
                f"(effective train batch ~= {int(batch.doc_input_ids.shape[0]) * accumulation_steps})",
                flush=True,
            )
            print(f"mixed precision AMP: {'enabled' if amp_enabled else 'disabled'}", flush=True)
            printed_sanity = True
        moved = move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(moved["batch"])
            target_uncertainty = build_target_uncertainty(moved, args=args)
            loss, ce_loss, uncertainty_loss = compute_training_loss(
                probs=outputs["probs"],
                pred_uncertainty=outputs["uncertainty_mass"],
                labels=moved["labels"],
                target_uncertainty=target_uncertainty,
                uncertainty_weight=uncertainty_weight,
                logits=outputs.get("logits"),
            )
        scaler.scale(loss / accumulation_steps).backward()
        should_step = step_index % accumulation_steps == 0 or step_index == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = int(moved["labels"].size(0))
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_ce_loss += float(ce_loss.item()) * batch_size
        total_uncertainty_loss += float(uncertainty_loss.item()) * batch_size
        total_target_uncertainty += float(target_uncertainty.mean().item()) * batch_size
        total_pred_uncertainty += float(outputs["uncertainty_mass"].mean().item()) * batch_size
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{(total_loss / max(total_examples, 1)):.4f}",
            lr=f"{max(scheduler.get_last_lr()):.2e}",
        )
    denom = max(total_examples, 1)
    return {
        "loss": total_loss / denom,
        "ce_loss": total_ce_loss / denom,
        "uncertainty_loss": total_uncertainty_loss / denom,
        "uncertainty_weight": uncertainty_weight,
        "mean_target_uncertainty": total_target_uncertainty / denom,
        "mean_pred_uncertainty": total_pred_uncertainty / denom,
    }


@torch.no_grad()
def evaluate(
    model: ReCo,
    loader: DataLoader,
    device: torch.device,
    num_labels: int,
    args: argparse.Namespace,
    epoch: int | None = None,
    amp_enabled: bool = False,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_uncertainty_loss = 0.0
    total_examples = 0
    y_true: list[int] = []
    y_pred_argmax: list[int] = []
    probs_all: list[list[float]] = []
    mu_ratings: list[float] = []
    au_coh_values: list[float] = []
    target_uncertainty_values: list[float] = []
    uncertainty_mass: list[float] = []
    doc_parser_eu: list[float] = []
    uncertainty_weight = uncertainty_regularizer_weight(epoch=epoch, args=args)

    for batch in loader:
        moved = move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(moved["batch"])
            target_uncertainty = build_target_uncertainty(moved, args=args)
            loss, ce_loss, uncertainty_loss = compute_training_loss(
                probs=outputs["probs"],
                pred_uncertainty=outputs["uncertainty_mass"],
                labels=moved["labels"],
                target_uncertainty=target_uncertainty,
                uncertainty_weight=uncertainty_weight,
                logits=outputs.get("logits"),
            )
        batch_size = int(moved["labels"].size(0))
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_ce_loss += float(ce_loss.item()) * batch_size
        total_uncertainty_loss += float(uncertainty_loss.item()) * batch_size
        y_true.extend(moved["labels"].detach().cpu().tolist())
        y_pred_argmax.extend(outputs["preds"].detach().cpu().tolist())
        probs_all.extend(outputs["probs"].detach().cpu().tolist())
        mu_ratings.extend(moved["mu_rating"].detach().cpu().tolist())
        au_coh_values.extend(target_uncertainty.detach().cpu().tolist())
        target_uncertainty_values.extend(target_uncertainty.detach().cpu().tolist())
        uncertainty_mass.extend(outputs["uncertainty_mass"].detach().cpu().tolist())
        doc_parser_eu.extend(outputs["doc_parser_eu"].detach().cpu().tolist())

    metrics = compute_metrics(
        y_true=y_true,
        y_pred_argmax=y_pred_argmax,
        probs=probs_all,
        mu_ratings=mu_ratings,
        au_coh_values=au_coh_values,
        target_uncertainty_values=target_uncertainty_values,
        uncertainty_mass=uncertainty_mass,
        doc_parser_eu=doc_parser_eu,
        num_labels=num_labels,
        dataset_name=args.dataset_name,
    )
    denom = max(total_examples, 1)
    metrics["loss"] = total_loss / denom
    metrics["ce_loss"] = total_ce_loss / denom
    metrics["uncertainty_loss"] = total_uncertainty_loss / denom
    metrics["uncertainty_weight"] = uncertainty_weight
    return metrics


@torch.no_grad()
def prediction_records(
    model: ReCo,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    prefix: str,
    fold_index: int,
    split: str,
    amp_enabled: bool = False,
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=f"dump {split}", leave=False):
        moved = move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(moved["batch"])
        logits = outputs["logits"].detach().cpu().tolist()
        probs = outputs["probs"].detach().cpu().tolist()
        preds = outputs["preds"].detach().cpu().tolist()
        gold = moved["labels"].detach().cpu().tolist()
        for row, doc_id in enumerate(batch.doc_ids):
            records.append(
                {
                    "dataset_name": str(args.dataset_name),
                    "prefix": str(prefix),
                    "fold": int(fold_index),
                    "split": split,
                    "doc_id": str(doc_id),
                    "text_id": str(batch.text_ids[row]),
                    "gold": int(gold[row]),
                    "pred": int(preds[row]),
                    "logits": [float(value) for value in logits[row]],
                    "probs": [float(value) for value in probs[row]],
                }
            )
    return records


def save_training_curve(path: Path, history: list[dict[str, Any]], title: str) -> None:
    if not history:
        return
    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_metrics"]["loss"] for record in history]
    dev_loss = [record["dev_metrics"]["loss"] for record in history]
    dev_macro_f1 = [record["dev_metrics"]["macro_f1"] for record in history]
    dev_acc = [record["dev_metrics"]["accuracy"] for record in history]
    dev_ece = [record["dev_metrics"]["ece"] for record in history]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(epochs, train_loss, marker="o", label="train_loss")
    axes[0].plot(epochs, dev_loss, marker="o", label="dev_loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    axes[1].plot(epochs, dev_macro_f1, marker="o", label="dev_macro_f1")
    axes[1].plot(epochs, dev_acc, marker="o", label="dev_accuracy")
    axes[1].plot(epochs, dev_ece, marker="o", label="dev_ece")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_metric(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def summarize_fold_results(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "accuracy",
        "qwk",
        "argmax_accuracy",
        "argmax_macro_f1",
        "argmax_qwk",
        "mae",
        "spearman",
        "ece",
        "brier",
        "nll",
        "f1_low",
        "f1_medium",
        "f1_high",
        "argmax_f1_low",
        "argmax_f1_medium",
        "argmax_f1_high",
        "corr_uncertainty_error",
        "corr_uncertainty_target",
        "corr_uncertainty_au",
        "risk_coverage_auc",
        "corr_parser_eu_target",
        "corr_parser_eu_au",
        "mean_target_uncertainty",
        "mean_pred_uncertainty",
        "ce_loss",
        "uncertainty_loss",
        "loss",
    ]
    summary = {"num_folds": len(fold_results), "test_metrics": {}}
    for metric in metric_names:
        summary["test_metrics"][metric] = summarize_metric([result["test_metrics"][metric] for result in fold_results])
    return summary


def write_fold_results_csv(path: Path, fold_results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "fold",
        "best_epoch",
        "selection_split",
        "selection_metric",
        "test_macro_f1",
        "test_accuracy",
        "test_qwk",
        "test_argmax_accuracy",
        "test_argmax_macro_f1",
        "test_argmax_qwk",
        "test_mae",
        "test_spearman",
        "test_ece",
        "test_brier",
        "test_nll",
        "test_f1_low",
        "test_f1_medium",
        "test_f1_high",
        "test_argmax_f1_low",
        "test_argmax_f1_medium",
        "test_argmax_f1_high",
        "test_corr_uncertainty_error",
        "test_corr_uncertainty_target",
        "test_corr_uncertainty_au",
        "test_risk_coverage_auc",
        "test_corr_parser_eu_target",
        "test_corr_parser_eu_au",
        "test_mean_target_uncertainty",
        "test_mean_pred_uncertainty",
        "test_ce_loss",
        "test_uncertainty_loss",
        "test_loss",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in fold_results:
            metrics = result["test_metrics"]
            writer.writerow(
                {
                    "fold": result["fold"],
                    "best_epoch": result["best_epoch"],
                    "selection_split": result.get("selection_split", ""),
                    "selection_metric": result.get("selection_metric", ""),
                    "test_macro_f1": metrics["macro_f1"],
                    "test_accuracy": metrics["accuracy"],
                    "test_qwk": metrics["qwk"],
                    "test_argmax_accuracy": metrics["argmax_accuracy"],
                    "test_argmax_macro_f1": metrics["argmax_macro_f1"],
                    "test_argmax_qwk": metrics["argmax_qwk"],
                    "test_mae": metrics["mae"],
                    "test_spearman": metrics["spearman"],
                    "test_ece": metrics["ece"],
                    "test_brier": metrics["brier"],
                    "test_nll": metrics["nll"],
                    "test_f1_low": metrics["f1_low"],
                    "test_f1_medium": metrics["f1_medium"],
                    "test_f1_high": metrics["f1_high"],
                    "test_argmax_f1_low": metrics["argmax_f1_low"],
                    "test_argmax_f1_medium": metrics["argmax_f1_medium"],
                    "test_argmax_f1_high": metrics["argmax_f1_high"],
                    "test_corr_uncertainty_error": metrics["corr_uncertainty_error"],
                    "test_corr_uncertainty_target": metrics["corr_uncertainty_target"],
                    "test_corr_uncertainty_au": metrics["corr_uncertainty_au"],
                    "test_risk_coverage_auc": metrics["risk_coverage_auc"],
                    "test_corr_parser_eu_target": metrics["corr_parser_eu_target"],
                    "test_corr_parser_eu_au": metrics["corr_parser_eu_au"],
                    "test_mean_target_uncertainty": metrics["mean_target_uncertainty"],
                    "test_mean_pred_uncertainty": metrics["mean_pred_uncertainty"],
                    "test_ce_loss": metrics["ce_loss"],
                    "test_uncertainty_loss": metrics["uncertainty_loss"],
                    "test_loss": metrics["loss"],
                }
            )


def save_best_checkpoint(
    checkpoint_dir: Path,
    model: ReCo,
    tokenizer,
    relation_vocab: dict[str, int],
    args: argparse.Namespace,
    best_epoch: int,
    best_selection_metrics: dict[str, float],
    best_dev_metrics: dict[str, float],
    best_test_metrics: dict[str, float],
    selection_split: str,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "relation_vocab": relation_vocab,
            "model_name": args.model_name,
            "model_config": {
                "hidden_size": args.d_model,
                "dropout": args.dropout,
                "relation_embedding_dim": args.relation_embedding_dim,
                "nuclearity_embedding_dim": args.nuclearity_embedding_dim,
                "text_hidden_dim": args.text_hidden_dim,
                "relation_label_dropout": args.relation_label_dropout,
                "relation_dist_dropout": args.relation_dist_dropout,
                "use_text_branch": args.use_text_branch,
                "use_treeformer_branch": args.use_treeformer_branch,
                "use_comp_branch": args.use_comp_branch,
                "use_relation_distribution": args.use_relation_distribution,
                "use_relation_embedding": args.use_relation_embedding,
                "use_parser_dropout": args.use_parser_dropout,
                "use_distance_bias": args.use_distance_bias,
                "use_ancestor_bias": args.use_ancestor_bias,
                "use_subtree_bias": args.use_subtree_bias,
                "use_source_relation_bias": args.use_source_relation_bias,
                "use_relation_uncertainty_bias": args.use_relation_uncertainty_bias,
                "unordered_tree": args.unordered_tree,
                "node_entropy_mode": args.node_entropy_mode,
                "node_entropy_source": args.node_entropy_source,
                "parser_confidence_mode": args.parser_confidence_mode,
                "use_doc_entropy_features": args.use_doc_entropy_features,
                "use_node_non_entropy_parser_stats": args.use_node_non_entropy_parser_stats,
                "use_span_states": args.use_span_states,
                "use_nuclearity_embedding": args.use_nuclearity_embedding,
                "use_path_features": args.use_path_features,
                "use_topology_features": args.use_topology_features,
                "use_doc_relation_histogram": args.use_doc_relation_histogram,
                "use_parser_profile": args.use_parser_profile,
                "parser_profile_mode": args.parser_profile_mode,
                "path_embedding_dim": args.path_embedding_dim,
                "treeformer_layers": args.treeformer_layers,
                "treeformer_heads": args.treeformer_heads,
                "treeformer_bias_dim": args.treeformer_bias_dim,
                "treeformer_ffn_multiplier": args.treeformer_ffn_multiplier,
                "tree_dist_cap": args.tree_dist_cap,
                "max_path_depth": args.max_path_depth,
                "freeze_encoder": args.freeze_encoder,
                "text_pooling": args.text_pooling,
                "use_precomputed_text_features": args.use_precomputed_text_features,
                "classifier_hidden_size": args.classifier_hidden_size,
            },
            "best_epoch": best_epoch,
            "selection_split": selection_split,
            "selection_metric": args.selection_metric,
            "best_selection_metrics": best_selection_metrics,
            "best_dev_metrics": best_dev_metrics,
            "best_test_metrics": best_test_metrics,
        },
        checkpoint_dir / "model.pt",
    )
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    save_json(
        checkpoint_dir / "training_state.json",
        {
            "best_epoch": best_epoch,
            "selection_split": selection_split,
            "selection_metric": args.selection_metric,
            "best_selection_metrics": best_selection_metrics,
            "best_dev_metrics": best_dev_metrics,
            "best_test_metrics": best_test_metrics,
            "ablation": args.ablation,
            "experiment_mode": args.experiment_mode,
        },
    )


def load_best_model(checkpoint_dir: Path, device: torch.device) -> tuple[ReCo, dict[str, int], str]:
    checkpoint = torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    relation_vocab = checkpoint["relation_vocab"]
    model = ReCo(
        hf_model_name=MODEL_NAME_MAP[checkpoint["model_name"]],
        num_labels=3,
        relation_vocab=relation_vocab,
        hidden_size=model_config["hidden_size"],
        dropout=model_config["dropout"],
        relation_embedding_dim=model_config["relation_embedding_dim"],
        nuclearity_embedding_dim=model_config["nuclearity_embedding_dim"],
        path_embedding_dim=model_config["path_embedding_dim"],
        text_hidden_dim=model_config["text_hidden_dim"],
        treeformer_layers=model_config["treeformer_layers"],
        treeformer_heads=model_config["treeformer_heads"],
        treeformer_bias_dim=model_config["treeformer_bias_dim"],
        treeformer_ffn_multiplier=model_config["treeformer_ffn_multiplier"],
        tree_dist_cap=model_config["tree_dist_cap"],
        max_path_depth=model_config["max_path_depth"],
        relation_label_dropout=model_config["relation_label_dropout"],
        relation_dist_dropout=model_config["relation_dist_dropout"],
        use_text_branch=model_config["use_text_branch"],
        use_treeformer_branch=model_config["use_treeformer_branch"],
        use_comp_branch=model_config["use_comp_branch"],
        use_relation_distribution=model_config["use_relation_distribution"],
        use_relation_embedding=model_config["use_relation_embedding"],
        use_parser_dropout=model_config["use_parser_dropout"],
        use_distance_bias=model_config["use_distance_bias"],
        use_ancestor_bias=model_config["use_ancestor_bias"],
        use_subtree_bias=model_config["use_subtree_bias"],
        use_source_relation_bias=model_config["use_source_relation_bias"],
        use_relation_uncertainty_bias=model_config["use_relation_uncertainty_bias"],
        unordered_tree=model_config["unordered_tree"],
        node_entropy_mode=model_config.get("node_entropy_mode", "original"),
        node_entropy_source=model_config.get("node_entropy_source", "relation"),
        parser_confidence_mode=model_config.get("parser_confidence_mode", "original"),
        use_doc_entropy_features=model_config.get("use_doc_entropy_features", True),
        use_node_non_entropy_parser_stats=model_config.get("use_node_non_entropy_parser_stats", True),
        use_span_states=model_config.get("use_span_states", True),
        use_nuclearity_embedding=model_config.get("use_nuclearity_embedding", True),
        use_path_features=model_config.get("use_path_features", True),
        use_topology_features=model_config.get("use_topology_features", True),
        use_doc_relation_histogram=model_config.get("use_doc_relation_histogram", True),
        use_parser_profile=model_config.get("use_parser_profile", True),
        parser_profile_mode=model_config.get("parser_profile_mode", "all"),
        freeze_encoder=model_config.get("freeze_encoder", True),
        text_pooling=model_config.get("text_pooling", "mean"),
        use_precomputed_text_features=model_config.get("use_precomputed_text_features", False),
        classifier_hidden_size=model_config.get("classifier_hidden_size", 256),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, relation_vocab, checkpoint["model_name"]


def load_prefix_resources(args: argparse.Namespace, prefixes: list[str]):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_MAP[args.model_name])
    parser_beta = [
        float(args.beta_h_rel),
        float(args.beta_confidence),
        float(args.beta_margin),
        float(args.beta_ambiguity),
        float(args.beta_rri),
    ]
    graph_bundles: dict[str, dict[str, Any]] = {}
    feature_stores: dict[str, dict[str, dict[str, Any]]] = {}
    rating_maps: dict[str, dict[str, dict[str, Any]]] = {}
    split_roots: dict[str, Path] = {}
    for prefix in prefixes:
        split_root = get_split_root(data_root=args.data_root, prefix=prefix)
        split_roots[prefix] = split_root
        representation_texts = None
        sota_text_root = args.sota_data_root
        if args.dataset_name == "gcdc":
            sota_text_root = resolve_sota_text_root(args, prefix)
            representation_texts = load_sota_texts_for_prefix(sota_text_root, prefix)
        graph_bundles[prefix] = load_or_build_graph_cache(
            prefix=prefix,
            data_root=args.data_root,
            parser_root=args.parser_root,
            cache_dir=args.graph_cache_dir,
            parser_lambda=float(args.parser_lambda),
            parser_beta=parser_beta,
        )
        feature_stores[prefix] = build_feature_store(
            prefix_root=split_root,
            prefix=prefix,
            graph_cache=graph_bundles[prefix],
            tokenizer=tokenizer,
            max_length=args.max_length,
            window_stride=args.window_stride,
            representation_texts=representation_texts,
        )
        if bool(args.use_precomputed_text_features):
            attach_sota_cached_representations(
                feature_store=feature_stores[prefix],
                encoder_name=MODEL_NAME_MAP[args.model_name],
                tokenizer=tokenizer,
                cache_dir=args.text_cache_dir,
                prefix=prefix,
                max_length=args.max_length,
                device=torch.device(args.device),
                batch_size=int(args.text_cache_batch_size),
                rebuild=bool(args.rebuild_text_cache),
                sota_data_root=sota_text_root if args.dataset_name == "gcdc" else None,
            )
        if bool(args.use_rst_tensor_cache) and (bool(args.use_treeformer_branch) or bool(args.use_comp_branch)):
            if not bool(args.use_precomputed_text_features):
                raise ValueError("use_rst_tensor_cache requires use_precomputed_text_features=true.")
            attach_rst_tensor_cache(
                feature_store=feature_stores[prefix],
                cache_dir=args.rst_tensor_cache_dir,
                prefix=prefix,
                encoder_name=MODEL_NAME_MAP[args.model_name],
                max_length=args.max_length,
                rebuild=bool(args.rebuild_text_cache),
            )
        rating_maps[prefix] = (
            load_rating_map(prefix=prefix, raw_data_root=args.raw_data_root)
            if args.dataset_name == "gcdc"
            else {}
        )
    return tokenizer, graph_bundles, feature_stores, rating_maps, split_roots


def build_split_records(prefix: str, split_path: Path) -> list[dict[str, Any]]:
    return load_jsonl(split_path)


def experiment_tag(args: argparse.Namespace, target_prefix: str) -> str:
    if args.experiment_mode == "in_domain":
        return target_prefix
    if args.experiment_mode == "cross_domain":
        return f"{args.source_prefix[0]}_to_{target_prefix}"
    default_sources = ["cohe"] if args.dataset_name == "cohesentia" else [prefix for prefix in PREFIXES if prefix != "cohe"]
    sources = args.source_prefix if args.source_prefix else default_sources
    return f"{'-'.join(sources)}_to_{target_prefix}"


def build_fold_specs(args: argparse.Namespace, target_prefix: str, split_roots: dict[str, Path], fold_index: int):
    target_paths = dict(get_fold_paths(split_roots[target_prefix]))[fold_index]
    if args.experiment_mode == "in_domain":
        train_specs = [(target_prefix, target_paths["train"])]
        dev_specs = [(target_prefix, target_paths["dev"])]
        test_specs = [(target_prefix, target_paths["test"])]
    elif args.experiment_mode == "cross_domain":
        source_prefix = args.source_prefix[0]
        source_paths = dict(get_fold_paths(split_roots[source_prefix]))[fold_index]
        train_specs = [(source_prefix, source_paths["train"])]
        dev_specs = [(source_prefix, source_paths["dev"])]
        test_specs = [(target_prefix, target_paths["test"])]
    else:
        default_prefixes = ["cohe"] if args.dataset_name == "cohesentia" else [prefix for prefix in PREFIXES if prefix != "cohe"]
        train_prefixes = args.source_prefix if args.source_prefix else default_prefixes
        train_specs = []
        dev_specs = []
        for prefix in train_prefixes:
            source_paths = dict(get_fold_paths(split_roots[prefix]))[fold_index]
            train_specs.append((prefix, source_paths["train"]))
            dev_specs.append((prefix, source_paths["dev"]))
        test_specs = [(target_prefix, target_paths["test"])]
    return train_specs, dev_specs, test_specs


def examples_from_specs(
    specs: list[tuple[str, Path]],
    feature_stores: dict[str, dict[str, dict[str, Any]]],
    rating_maps: dict[str, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for prefix, split_path in specs:
        records = build_split_records(prefix, split_path)
        if args.dataset_name == "gcdc" and bool(args.use_precomputed_text_features):
            records = [
                record
                for record in records
                if feature_stores[prefix].get(f"{prefix}_{record['text_id']}", {}).get("sota_text_available", True)
            ]
        examples.extend(
            build_examples(
                records=records,
                prefix=prefix,
                feature_store=feature_stores[prefix],
                rating_map=rating_maps[prefix],
                args=args,
            )
        )
    return examples


def build_structural_lookup_for_prefix(
    prefix: str,
    feature_store: dict[str, dict[str, Any]],
    use_structure: bool,
    require_rst_tensors: bool = True,
) -> dict[str, dict[str, Any]] | None:
    if not use_structure:
        return None
    lookup: dict[str, dict[str, Any]] = {}
    for doc_id, item in feature_store.items():
        text_id = str(item["text_id"])
        if require_rst_tensors and "rst_tensors" not in item:
            raise ValueError(f"Missing cached RST tensors for {doc_id}; enable use_rst_tensor_cache.")
        lookup[text_id] = {
            "graph": item["graph"],
            "rst_tensors": item.get("rst_tensors"),
        }
    return lookup


def train_target_prefix_sota_compatible(
    args: argparse.Namespace,
    target_prefix: str,
    feature_stores: dict[str, dict[str, dict[str, Any]]],
    relation_vocab: dict[str, int],
) -> None:
    """GCDC runner whose data/training protocol follows EntyRelCoh text-only.

    The final method looks up fixed RST uncertainty tensors by `text_id` and
    concatenates the resulting lightweight RST feature vector to the frozen
    document vector before the classifier head.
    """
    if args.experiment_mode != "in_domain":
        raise ValueError("SOTA-compatible runner currently supports GCDC in_domain experiments only.")

    tag = experiment_tag(args, target_prefix)
    model_root = args.output_root / args.ablation
    model_root.mkdir(parents=True, exist_ok=True)
    save_json(model_root / "run_config.json", make_checkpoint_safe(vars(args)))

    requested_folds = [args.fold] if args.fold is not None else list(range(1, 11))
    structural_lookup = build_structural_lookup_for_prefix(
        prefix=target_prefix,
        feature_store=feature_stores[target_prefix],
        use_structure=bool(args.use_treeformer_branch or args.use_comp_branch or args.use_parser_profile),
        require_rst_tensors=bool(args.use_treeformer_branch or args.use_comp_branch),
    )
    fold_results: list[dict[str, Any]] = []

    for fold_index in requested_folds:
        fold_dir = model_root / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_result_path = fold_dir / "fold_result.json"
        if args.fold is None and fold_result_path.exists():
            with fold_result_path.open("r", encoding="utf-8") as handle:
                existing_result = json.load(handle)
            if int(existing_result.get("result_schema_version", 0)) == RESULT_SCHEMA_VERSION:
                fold_results.append(existing_result)
                print(f"skip existing fold {fold_index}: {fold_result_path}", flush=True)
                continue

        set_seed(args.seed)
        tokenizer, encoder = build_sota_text_encoder(args)
        split_paths = sota_dataset_split_paths(args, target_prefix, int(fold_index))
        train_dataset = build_sota_sent_dataset(split_paths["train"], args=args, tokenizer=tokenizer, encoder=encoder)
        dev_dataset = build_sota_sent_dataset(split_paths["dev"], args=args, tokenizer=tokenizer, encoder=encoder)
        test_dataset = build_sota_sent_dataset(split_paths["test"], args=args, tokenizer=tokenizer, encoder=encoder)
        train_loader = get_sota_compatible_dataloader(train_dataset, args, mode="train")
        dev_loader = get_sota_compatible_dataloader(dev_dataset, args, mode="dev")
        test_loader = get_sota_compatible_dataloader(test_dataset, args, mode="test")

        model = build_model(args=args, relation_vocab=relation_vocab, num_labels=3).to(torch.device(args.device))
        fold_result = sota_compatible_train_fold(
            model=model,
            args=args,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            structural_lookup=structural_lookup,
            fold_dir=fold_dir,
            prefix=target_prefix,
            fold_index=fold_index,
        )
        fold_result["fold"] = fold_index
        fold_results.append(fold_result)
        save_json(fold_result_path, fold_result)

    write_fold_results_csv(model_root / "fold_results.csv", fold_results)
    summary = summarize_fold_results(fold_results)
    summary["ablation"] = args.ablation
    summary["experiment_mode"] = args.experiment_mode
    summary["selection_split"] = "dev"
    summary["selection_metric"] = args.selection_metric
    summary["target_prefix"] = target_prefix
    summary["tag"] = tag
    summary["runner"] = "sota_compatible"
    save_json(model_root / "summary.json", summary)


def train_target_prefix(
    args: argparse.Namespace,
    target_prefix: str,
    tokenizer,
    graph_bundles: dict[str, dict[str, Any]],
    feature_stores: dict[str, dict[str, dict[str, Any]]],
    rating_maps: dict[str, dict[str, dict[str, Any]]],
    split_roots: dict[str, Path],
    relation_vocab: dict[str, int],
) -> None:
    tag = experiment_tag(args, target_prefix)
    model_root = args.output_root / args.ablation
    model_root.mkdir(parents=True, exist_ok=True)
    save_json(model_root / "run_config.json", make_checkpoint_safe(vars(args)))

    fold_paths = get_fold_paths(split_roots[target_prefix])
    if args.fold is not None:
        fold_paths = [(fold_idx, split_paths) for fold_idx, split_paths in fold_paths if fold_idx == args.fold]
        if not fold_paths:
            raise ValueError(f"Fold {args.fold} not found for prefix {target_prefix}.")

    fold_results: list[dict[str, Any]] = []
    for fold_index, _ in fold_paths:
        fold_dir = model_root / f"fold_{fold_index}"
        fold_result_path = fold_dir / "fold_result.json"
        if args.fold is None and fold_result_path.exists():
            with fold_result_path.open("r", encoding="utf-8") as handle:
                existing_result = json.load(handle)
            if int(existing_result.get("result_schema_version", 0)) == RESULT_SCHEMA_VERSION:
                fold_results.append(existing_result)
                print(f"skip existing fold {fold_index}: {fold_result_path}", flush=True)
                continue
            print(f"rerun fold {fold_index}: outdated result schema at {fold_result_path}", flush=True)

        train_specs, dev_specs, test_specs = build_fold_specs(args, target_prefix, split_roots, fold_index)
        train_examples = examples_from_specs(train_specs, feature_stores, rating_maps, args=args)
        dev_examples = examples_from_specs(dev_specs, feature_stores, rating_maps, args=args)
        test_examples = examples_from_specs(test_specs, feature_stores, rating_maps, args=args)

        train_loader = DataLoader(
            PreparedDataset(train_examples),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=BatchCollator(),
        )
        dev_loader = DataLoader(
            PreparedDataset(dev_examples),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=BatchCollator(),
        )
        test_loader = DataLoader(
            PreparedDataset(test_examples),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=BatchCollator(),
        )

        device = torch.device(args.device)
        model = build_model(args=args, relation_vocab=relation_vocab, num_labels=3).to(device)
        optimizer = build_optimizer(
            model=model,
            encoder_lr=args.encoder_lr,
            other_lr=args.other_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
        update_steps_per_epoch = max(math.ceil(len(train_loader) / max(int(args.gradient_accumulation_steps), 1)), 1)
        total_steps = max(update_steps_per_epoch * args.epochs, 1)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        amp_enabled = bool(args.use_amp) and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        history_path = fold_dir / "history.json"
        log_path = fold_dir / "training.log"
        epoch_metrics_path = fold_dir / "epoch_metrics.jsonl"
        best_checkpoint_dir = fold_dir / "best_model"
        save_json(
            fold_dir / "split_plan.json",
            {
                "train": [(prefix, str(path)) for prefix, path in train_specs],
                "dev": [(prefix, str(path)) for prefix, path in dev_specs],
                "test": [(prefix, str(path)) for prefix, path in test_specs],
            },
        )

        history: list[dict[str, Any]] = []
        selection_split = "dev"
        best_selection_metrics: dict[str, float] | None = None
        best_dev_metrics: dict[str, float] | None = None
        best_test_metrics: dict[str, float] | None = None
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                gradient_clip=args.gradient_clip,
                args=args,
                epoch=epoch,
                progress_desc=f"{tag} fold {fold_index} epoch {epoch}",
                amp_enabled=amp_enabled,
                scaler=scaler,
            )
            should_evaluate = (
                epoch == 1
                or epoch == args.epochs
                or epoch % int(args.eval_every_epochs) == 0
            )
            if not should_evaluate:
                append_log_line(
                    log_path,
                    f"epoch={epoch} train[{format_metric_block(train_metrics)}] "
                    f"eval[skipped eval_every_epochs={args.eval_every_epochs}]",
                )
                continue
            dev_metrics = evaluate(
                model=model,
                loader=dev_loader,
                device=device,
                num_labels=3,
                args=args,
                epoch=epoch,
                amp_enabled=amp_enabled,
            )
            test_metrics = evaluate(
                model=model,
                loader=test_loader,
                device=device,
                num_labels=3,
                args=args,
                epoch=epoch,
                amp_enabled=amp_enabled,
            )
            epoch_record = {
                "epoch": epoch,
                "train_metrics": train_metrics,
                "dev_metrics": dev_metrics,
                "test_metrics": test_metrics,
            }
            history.append(epoch_record)
            save_json(history_path, history)
            append_jsonl(epoch_metrics_path, epoch_record)
            append_log_line(
                log_path,
                f"epoch={epoch} train[{format_metric_block(train_metrics)}] "
                f"dev[{format_metric_block(dev_metrics)}] test[{format_metric_block(test_metrics)}]",
            )

            if is_better(dev_metrics, best_selection_metrics, metric=args.selection_metric):
                best_selection_metrics = dev_metrics
                best_dev_metrics = dev_metrics
                best_test_metrics = test_metrics
                best_epoch = epoch
                epochs_without_improvement = 0
                if args.dump_predictions:
                    pred_dir = fold_dir / "predictions"
                    dump_prediction_records(
                        pred_dir / "train_predictions.jsonl",
                        prediction_records(
                            model=model,
                            loader=train_loader,
                            device=device,
                            args=args,
                            prefix=target_prefix,
                            fold_index=fold_index,
                            split="train",
                            amp_enabled=amp_enabled,
                        ),
                    )
                    dump_prediction_records(
                        pred_dir / "dev_predictions.jsonl",
                        prediction_records(
                            model=model,
                            loader=dev_loader,
                            device=device,
                            args=args,
                            prefix=target_prefix,
                            fold_index=fold_index,
                            split="dev",
                            amp_enabled=amp_enabled,
                        ),
                    )
                    dump_prediction_records(
                        pred_dir / "test_predictions.jsonl",
                        prediction_records(
                            model=model,
                            loader=test_loader,
                            device=device,
                            args=args,
                            prefix=target_prefix,
                            fold_index=fold_index,
                            split="test",
                            amp_enabled=amp_enabled,
                        ),
                    )
                    save_json(pred_dir / "prediction_dump_info.json", {"best_epoch": best_epoch})
                if args.save_best_model:
                    save_best_checkpoint(
                        checkpoint_dir=best_checkpoint_dir,
                        model=model,
                        tokenizer=tokenizer,
                        relation_vocab=relation_vocab,
                        args=args,
                        best_epoch=best_epoch,
                        best_selection_metrics=best_selection_metrics,
                        best_dev_metrics=best_dev_metrics,
                        best_test_metrics=best_test_metrics,
                        selection_split=selection_split,
                    )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    append_log_line(log_path, f"early_stop epoch={epoch}")
                    break

        save_training_curve(
            fold_dir / "training_curve.png",
            history,
            title=f"{tag} fold {fold_index} ReCo training curves",
        )
        if best_test_metrics is None and args.save_best_model:
            best_model, _, _ = load_best_model(best_checkpoint_dir, device=device)
            best_test_metrics = evaluate(
                model=best_model,
                loader=test_loader,
                device=device,
                num_labels=3,
                args=args,
                epoch=None,
                amp_enabled=amp_enabled,
            )
        if best_test_metrics is None:
            raise RuntimeError(f"No evaluated checkpoint was selected for fold {fold_index}.")
        fold_result = {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "fold": fold_index,
            "best_epoch": best_epoch,
            "selection_split": selection_split,
            "selection_metric": args.selection_metric,
            "best_selection_metrics": best_selection_metrics,
            "best_dev_metrics": best_dev_metrics,
            "test_metrics": best_test_metrics,
        }
        fold_results.append(fold_result)
        save_json(fold_dir / "fold_result.json", fold_result)

    write_fold_results_csv(model_root / "fold_results.csv", fold_results)
    summary = summarize_fold_results(fold_results)
    summary["ablation"] = args.ablation
    summary["experiment_mode"] = args.experiment_mode
    summary["selection_split"] = "dev"
    summary["selection_metric"] = args.selection_metric
    summary["target_prefix"] = target_prefix
    summary["tag"] = tag
    save_json(model_root / "summary.json", summary)


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    default_prefixes = ["cohe"] if args.dataset_name == "cohesentia" else [prefix for prefix in PREFIXES if prefix != "cohe"]
    involved_prefixes = sorted(
        set(args.prefix + args.source_prefix + list(default_prefixes if args.experiment_mode == "multi_domain" and not args.source_prefix else []))
    )

    if args.dataset_name == "gcdc":
        use_structure = bool(args.use_treeformer_branch or args.use_comp_branch or args.use_parser_profile)
        if use_structure:
            tokenizer, graph_bundles, feature_stores, rating_maps, split_roots = load_prefix_resources(args, involved_prefixes)
            relation_vocab = build_relation_vocab(graph_bundles)
        else:
            feature_stores = {prefix: {} for prefix in involved_prefixes}
            relation_vocab = {"<UNK>": 0, "<ROOT_PATH>": 1}
        for target_prefix in args.prefix:
            train_target_prefix_sota_compatible(
                args=args,
                target_prefix=target_prefix,
                feature_stores=feature_stores,
                relation_vocab=relation_vocab,
            )
        return

    tokenizer, graph_bundles, feature_stores, rating_maps, split_roots = load_prefix_resources(args, involved_prefixes)
    relation_vocab = build_relation_vocab(graph_bundles)
    for target_prefix in args.prefix:
        train_target_prefix(
            args=args,
            target_prefix=target_prefix,
            tokenizer=tokenizer,
            graph_bundles=graph_bundles,
            feature_stores=feature_stores,
            rating_maps=rating_maps,
            split_roots=split_roots,
            relation_vocab=relation_vocab,
        )


def main() -> None:
    args = resolve_args(parse_args())
    run_experiment(args)


if __name__ == "__main__":
    main()
