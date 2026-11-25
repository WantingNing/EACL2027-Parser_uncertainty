import json
import os
import shlex
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import yaml

import llm_inference
import llm_inputs
import text_difficulty
from ensemble import (BASELINES, METRICS, compute_metrics, prediction_path, read_predictions, run_ensemble,
                      summarize, write_csv)

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def pick(names, available):
    if names == "all":
        return list(available)
    return names


def train_dir(config, dataset, seed):
    return ROOT / config["paths"]["runs"] / "train" / f"seed_{seed}" / dataset


def data_roots(config, dataset):
    # The trainer reads the fold splits and the raw annotation files of the dataset.
    if dataset == "cohesentia":
        data_root = ROOT / config["paths"]["cohesentia"]
        return "cohesentia", data_root, data_root / "raw" / "cohe"
    data_root = ROOT / config["paths"]["gcdc"]
    return "gcdc", data_root, data_root / "raw" / "gcdc"


def reco_params(config, model, dataset, seed):
    name, data_root, raw_data_root = data_roots(config, dataset)
    cache = ROOT / config["paths"]["cache"]
    params = {}
    for group in config["reco_trainer"].values():
        params.update(group)
    params.update(config["datasets"][dataset]["reco"])
    params.update(config["variants"][model])
    params.update({
        "prefix": [config["datasets"][dataset]["prefix"]],
        "dataset_name": name,
        "seed": seed,
        "ablation": model,
        "data_root": str(data_root),
        "raw_data_root": str(raw_data_root),
        "parser_root": str(ROOT / config["paths"]["rst_parser"]),
        "sota_data_root": str(ROOT / config["paths"]["prepared"] / "text_only"),
        "graph_cache_dir": str(cache / name),
        "text_cache_dir": str(cache / f"{name}_text_vectors"),
        "rst_tensor_cache_dir": str(cache / f"{name}_rst_tensors"),
        "output_root": str(train_dir(config, dataset, seed)),
    })
    return params


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def reco_command(config, model, dataset, seed, fold):
    model_dir = train_dir(config, dataset, seed) / model
    params_path = model_dir / "run_params.yaml"
    variant_path = model_dir / "variant.yaml"
    write_yaml(params_path, reco_params(config, model, dataset, seed))
    # The trainer always reads an ablation file; every setting is already in run_params.yaml.
    write_yaml(variant_path, {model: {}})
    command = [sys.executable, str(ROOT / "models" / "reco" / "train.py"),
               "--config", str(params_path), "--ablations-config", str(variant_path),
               "--ablation", model, "--fold", str(fold)]
    return command, ROOT


def baseline_command(config, model, dataset, seed, fold):
    params = dict(config[f"{model}_trainer"])
    if model == "entyrelcoh":
        params.update(config["datasets"][dataset]["entyrelcoh"])
    params.update({
        "data_dir": str(ROOT / config["paths"]["prepared"] / model),
        "dataset": dataset,
        "fold_id": fold,
        "seed": seed,
        "output_dir": str(ROOT / config["paths"]["runs"] / "train" / f"seed_{seed}"),
    })
    command = [sys.executable, "train.py"]
    for key, value in params.items():
        if value is True:
            command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])
    return command, ROOT / "models" / model


def trainer_env(config):
    env = dict(os.environ)
    env["GLOVE_EMBED_PATH"] = str(ROOT / config["paths"]["glove"])
    env["REL_EMBED_CACHE"] = str(ROOT / config["paths"]["rel_embed"])
    env["ALLOW_RANDOM_REL_EMBED"] = "0"
    return env


def train_fold(config, model, dataset, seed, fold):
    test_file = prediction_path(train_dir(config, dataset, seed), model, dataset, fold, "test")
    if test_file.exists():
        print(f"skip {model} {dataset} seed {seed} fold {fold}: {test_file} exists", flush=True)
        return
    if model in BASELINES:
        command, cwd = baseline_command(config, model, dataset, seed, fold)
    else:
        command, cwd = reco_command(config, model, dataset, seed, fold)
    print(f"cd {cwd} && {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=trainer_env(config), check=True)


def write_cv_summary(config, model, dataset, seed):
    # The trainers run one fold per process, so their own summary files cover only the last fold.
    # This recomputes the test metrics of every fold from the saved predictions and averages them.
    model_dir = train_dir(config, dataset, seed) / model
    fold_rows = []
    for fold in range(1, config["datasets"][dataset]["folds"] + 1):
        predictions = read_predictions(prediction_path(train_dir(config, dataset, seed), model, dataset, fold, "test"), model)
        keys = sorted(predictions)
        gold = np.array([key[1] for key in keys])
        probs = np.stack([predictions[key] for key in keys])
        fold_rows.append({"fold": fold, "n": len(gold), **compute_metrics(gold, probs)})
    write_csv(model_dir / "cv_fold_metrics.csv", fold_rows)
    summary = {"dataset": dataset, "model": model, "seed": seed, "folds": len(fold_rows),
               "metrics": {name: summarize([row[name] for row in fold_rows]) for name in METRICS}}
    with open(model_dir / "cv_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"{dataset} {model} seed {seed}: mean test accuracy {summary['metrics']['accuracy']['mean']:.4f}", flush=True)


def build_cache(config, dataset):
    # Builds the graph, text-vector and RST-tensor caches with full-ReCo settings, without training,
    # so that every training run starts from existing caches.
    sys.path.insert(0, str(ROOT / "models" / "reco"))
    import train as reco_train

    params = reco_params(config, "reco", dataset, config["seeds"][0])
    args = reco_train.finalize_args(params, ROOT / "config.yaml", ROOT / "config.yaml")
    print(f"building caches for {dataset}", flush=True)
    reco_train.load_prefix_resources(args, args.prefix)


def run_cache(config, recipe):
    for dataset in pick(recipe["datasets"], config["datasets"]):
        build_cache(config, dataset)


def run_train(config, recipe):
    datasets = pick(recipe["datasets"], config["datasets"])
    for seed, dataset, model in product(config["seeds"], datasets, recipe["models"]):
        for fold in range(1, config["datasets"][dataset]["folds"] + 1):
            train_fold(config, model, dataset, seed, fold)
        write_cv_summary(config, model, dataset, seed)


def run_ensembles(config, recipe):
    datasets = pick(recipe["datasets"], config["datasets"])
    pairs = pick(recipe["pairs"], config["ensemble"]["pairs"])
    for seed, dataset, pair, mode in product(config["seeds"], datasets, pairs, recipe["modes"]):
        out_dir = ROOT / config["paths"]["runs"] / "ensembles" / f"seed_{seed}" / dataset / pair / mode
        run_ensemble(config, train_dir(config, dataset, seed), out_dir, dataset, pair, mode)


def run_llm(config, recipe):
    models = pick(recipe["models"], config["llm"]["models"])
    for model, prompt in product(models, recipe["prompts"]):
        llm_inference.run(config, ROOT, model, prompt)


def main():
    if len(sys.argv) != 2:
        print("usage: python experiments/run.py <recipe>")
        return
    config = load_config()
    recipe = config["recipes"][sys.argv[1]]
    step = recipe["step"]
    if step == "cache":
        run_cache(config, recipe)
    if step == "train":
        run_train(config, recipe)
    if step == "ensemble":
        run_ensembles(config, recipe)
    if step == "text_difficulty":
        text_difficulty.run(config, ROOT)
    if step == "llm_inputs":
        llm_inputs.run(config, ROOT)
    if step == "llm_inference":
        run_llm(config, recipe)


if __name__ == "__main__":
    main()
