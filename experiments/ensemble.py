import csv
import json

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_recall_fscore_support

BASELINES = ("entyrelcoh", "relcoh")
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "qwk", "brier", "nll")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_path(train_dir, model, dataset, fold, split):
    # The two baseline trainers and the ReCo trainer use different folder and file names.
    if model in BASELINES:
        return train_dir / model / str(fold) / "predictions" / f"{dataset}_fold{fold}_{split}_best.jsonl"
    return train_dir / model / f"fold_{fold}" / "predictions" / f"{split}_predictions.jsonl"


def read_predictions(path, model):
    predictions = {}
    for row in read_jsonl(path):
        if model == "entyrelcoh":
            probs = [row["prob_1"], row["prob_2"], row["prob_3"]]
            gold = row["gold_index"]
        elif model == "relcoh":
            probs = row["probabilities"]
            gold = row["gold_id"]
        else:
            probs = row["probs"]
            gold = row["gold"]
        key = (str(row["text_id"]), int(gold))
        if key in predictions:
            raise ValueError(f"Duplicate document {key} in {path}")
        predictions[key] = np.array(probs, dtype=np.float64)
    return predictions


def align(left, right, fill_from_left):
    extra = [key for key in right if key not in left]
    missing = [key for key in left if key not in right]
    if extra or (missing and not fill_from_left):
        raise ValueError(f"Predictions do not align: {len(missing)} missing on the right, {len(extra)} extra")
    keys = sorted(left)
    right_rows = []
    for key in keys:
        # A document missing on the right side keeps the left model's prediction.
        right_rows.append(right[key] if key in right else left[key])
    gold = np.array([key[1] for key in keys])
    return keys, gold, np.stack([left[key] for key in keys]), np.stack(right_rows)


def mix(left_probs, right_probs, alpha):
    probs = (1.0 - alpha) * left_probs + alpha * right_probs
    return probs / probs.sum(axis=1, keepdims=True).clip(min=1e-12)


def compute_metrics(gold, probs):
    pred = probs.argmax(axis=1)
    one_hot = np.eye(probs.shape[1])[gold]
    gold_probs = np.clip(probs[np.arange(len(gold)), gold], 1e-8, 1.0)
    precision, recall, _, _ = precision_recall_fscore_support(gold, pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(gold, pred, weights="quadratic")),
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "nll": float(np.mean(-np.log(gold_probs))),
    }


def select_alpha(gold, left_probs, right_probs, step, objective):
    best_alpha = 0.0
    best_score = -1.0
    curve = []
    for index in range(int(round(1.0 / step)) + 1):
        alpha = round(index * step, 6)
        scores = compute_metrics(gold, mix(left_probs, right_probs, alpha))
        curve.append({"alpha": alpha, **scores})
        # Strictly better only, so the smallest alpha wins ties.
        if scores[objective] > best_score + 1e-12:
            best_alpha = alpha
            best_score = scores[objective]
    return best_alpha, curve


def load_split(train_dir, dataset, left_model, right_model, fold, split):
    left = read_predictions(prediction_path(train_dir, left_model, dataset, fold, split), left_model)
    right = read_predictions(prediction_path(train_dir, right_model, dataset, fold, split), right_model)
    fill_from_left = left_model in BASELINES and right_model not in BASELINES
    return align(left, right, fill_from_left)


def summarize(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def run_ensemble(config, train_dir, out_dir, dataset, pair, mode):
    left_model, right_model = config["ensemble"]["pairs"][pair]
    step = config["ensemble"]["alpha_step"]
    objective = config["ensemble"]["objective"]
    fold_rows = []
    curve_rows = []
    for fold in range(1, config["datasets"][dataset]["folds"] + 1):
        alpha = 0.5
        if mode == "weighted":
            _, dev_gold, dev_left, dev_right = load_split(train_dir, dataset, left_model, right_model, fold, "dev")
            alpha, curve = select_alpha(dev_gold, dev_left, dev_right, step, objective)
            curve_rows.extend({"fold": fold, **row} for row in curve)
        keys, gold, left_probs, right_probs = load_split(train_dir, dataset, left_model, right_model, fold, "test")
        probs = mix(left_probs, right_probs, alpha)
        fold_rows.append({"fold": fold, "alpha": alpha, "n": len(gold), **compute_metrics(gold, probs)})
        rows = []
        for index, key in enumerate(keys):
            rows.append({"text_id": key[0], "gold": key[1], "pred": int(probs[index].argmax()),
                         "probs": probs[index].tolist(), "left_probs": left_probs[index].tolist(),
                         "right_probs": right_probs[index].tolist(), "alpha": alpha})
        write_jsonl(out_dir / f"fold_{fold}" / "test_predictions.jsonl", rows)
        print(f"{dataset} {pair} {mode} fold {fold}: alpha={alpha:.3f} accuracy={fold_rows[-1]['accuracy']:.4f}", flush=True)

    write_csv(out_dir / "fold_metrics.csv", fold_rows)
    if curve_rows:
        write_csv(out_dir / "alpha_search.csv", curve_rows)
    summary = {
        "dataset": dataset,
        "pair": pair,
        "left": left_model,
        "right": right_model,
        "mode": mode,
        "selection_split": "dev" if mode == "weighted" else "none (alpha fixed at 0.5)",
        "alpha_step": step if mode == "weighted" else None,
        "objective": objective if mode == "weighted" else None,
        "folds": len(fold_rows),
        "alpha": summarize([row["alpha"] for row in fold_rows]),
        "metrics": {name: summarize([row[name] for row in fold_rows]) for name in METRICS},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
