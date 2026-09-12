from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import filter_by_episode_ids, load_shards, topk_match, validation_mask
from .config import ROOT, load_config


def git_state() -> tuple[str, bool]:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    ).strip())
    return sha, dirty


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return float("nan")
    return float(
        (ranks[positives].sum() - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def decision_metrics(
    scores: np.ndarray,
    data: dict,
    decision_mask: np.ndarray,
    row_mask: np.ndarray,
    *,
    classifier: bool,
) -> dict:
    sizes = data["dec_sizes"][decision_mask]
    labels = data["y"][row_mask]
    counts = np.ones(len(sizes), dtype=int)
    matches = topk_match(scores, labels, sizes, counts)
    teacher_counts = data["dec_k"][decision_mask]
    max_count = data["dec_maxcount"][decision_mask]
    set_matches = topk_match(scores, labels, sizes, teacher_counts)
    policy_matches = topk_match(scores, labels, sizes, max_count)
    single = max_count == 1
    rule = data["dec_rule_match"][decision_mask]
    known_single = single & (rule >= 0)
    metrics = {
        "top1_all": float(matches.mean()),
        "top1_max_count_1": float(matches[single].mean()),
        "teacher_set_all": float(set_matches.mean()),
        "policy_set_all": float(policy_matches.mean()),
        "rule_max_count_1": float(rule[known_single].mean()),
        "contexts": {},
    }
    if classifier:
        metrics["auc"] = binary_auc(scores, labels)
    for context in np.unique(data["dec_ctx"][decision_mask]):
        selected = data["dec_ctx"][decision_mask] == context
        known_rule = selected & (rule >= 0)
        metrics["contexts"][str(int(context))] = {
            "decisions": int(selected.sum()),
            "model_top1": float(matches[selected].mean()),
            "model_teacher_set": float(set_matches[selected].mean()),
            "model_policy_set": float(policy_matches[selected].mean()),
            "rule_policy_set": (
                float(rule[known_rule].mean()) if known_rule.any() else None
            ),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deck action scorer with CatBoost.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--runs-root", default="data/catboost")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--random-strength", type=float, default=None)
    parser.add_argument("--positive-class-weight", type=float, default=0.0,
                        help="positive class weight; 0 keeps auto Balanced weights")
    parser.add_argument("--subsample", type=float, default=0.0,
                        help="Bernoulli row fraction; 0 keeps CatBoost default bootstrap")
    parser.add_argument("--rsm", type=float, default=0.0,
                        help="feature fraction per split; 0 keeps all features")
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=0,
                        help="CatBoost worker threads; 0 uses all detected CPUs")
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument(
        "--objective", choices=("classifier", "ranker"), default="classifier",
        help="binary option classification or within-decision ranking",
    )
    parser.add_argument(
        "--ranking-loss", default="PairLogit",
        help="CatBoost ranking loss specification, including loss-specific parameters",
    )
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="limit episodes for a smoke run; 0 uses all data")
    parser.add_argument("--keep-episodes", default="",
                        help="json list of episode_ids; instant filter on dec_episode_id (no re-parse)")
    parser.add_argument("--validation-episodes", default="",
                        help="json list of episode_ids reserved for validation")
    parser.add_argument(
        "--final-fit", action="store_true",
        help="fit all data for exactly --iterations trees; reported metrics are in-sample",
    )
    parser.add_argument(
        "--snapshot-file", default="",
        help="CatBoost checkpoint path; an existing compatible snapshot is resumed",
    )
    parser.add_argument(
        "--snapshot-interval", type=int, default=60,
        help="seconds between CatBoost checkpoints (default: 60)",
    )
    parser.add_argument(
        "--drop-features", default="",
        help="JSON list of feature names to physically remove before training",
    )
    args = parser.parse_args()
    started = time.perf_counter()

    from catboost import CatBoostClassifier, CatBoostRanker, Pool

    config = load_config(args.config)
    data = load_shards(args.data, max_episodes=args.max_episodes)
    if args.keep_episodes:
        keep = {int(value) for value in json.load(open(args.keep_episodes))}
        decisions_before, rows_before = len(data["dec_sizes"]), len(data["y"])
        data = filter_by_episode_ids(data, keep)
        print(f"keep-episodes: {args.keep_episodes} | "
              f"decisions {decisions_before:,}->{len(data['dec_sizes']):,} | "
              f"rows {rows_before:,}->{len(data['y']):,}")
    with open(os.path.join(args.data, "features.json"), encoding="utf-8") as file:
        feature_metadata = json.load(file)
    names = feature_metadata["names"]
    dropped_features: list[str] = []
    if args.drop_features:
        dropped_features = list(json.loads(
            Path(args.drop_features).read_text(encoding="utf-8")
        ))
        dropped_set = set(dropped_features)
        unknown = sorted(dropped_set - set(names))
        if unknown:
            raise SystemExit(f"unknown --drop-features names: {unknown[:10]}")
        keep_indices = [
            index for index, name in enumerate(names) if name not in dropped_set
        ]
        data["X"] = data["X"][:, keep_indices]
        names = [names[index] for index in keep_indices]
        print(f"drop-features: {args.drop_features} "
              f"({len(dropped_set):,} dropped, {len(names):,} remaining)")
    if args.final_fit and args.validation_episodes:
        raise SystemExit("--final-fit cannot be combined with --validation-episodes")
    if args.final_fit:
        decision_validation = np.ones(len(data["dec_sizes"]), dtype=bool)
        row_validation = np.ones(len(data["y"]), dtype=bool)
        decision_train = decision_validation
        row_train = row_validation
    elif args.validation_episodes:
        validation_ids = {
            int(value) for value in json.load(open(args.validation_episodes))
        }
        decision_validation = np.isin(data["dec_episode_id"], list(validation_ids))
        row_validation = decision_validation[data["group"]]
        if not decision_validation.any() or decision_validation.all():
            raise SystemExit("explicit validation split must leave both train and validation decisions")
        print(f"validation-episodes: {args.validation_episodes} "
              f"({decision_validation.sum():,} decisions)")
        decision_train = ~decision_validation
        row_train = ~row_validation
    else:
        decision_validation, row_validation = validation_mask(
            data["dec_episode"], data["group"], args.valid_frac, args.seed
        )
        decision_train = ~decision_validation
        row_train = ~row_validation
    valid_decisions = 0 if args.final_fit else decision_validation.sum()
    valid_rows = 0 if args.final_fit else row_validation.sum()
    print(f"episodes={len(np.unique(data['dec_episode'])):,} "
          f"decisions train={decision_train.sum():,} valid={valid_decisions:,} "
          f"rows train={row_train.sum():,} valid={valid_rows:,}")

    model_args = dict(
        iterations=args.iterations, depth=args.depth, learning_rate=args.lr,
        l2_leaf_reg=args.l2_leaf_reg, random_seed=args.seed,
        verbose=100,
        thread_count=args.threads or (os.cpu_count() or 4),
        allow_writing_files=bool(args.snapshot_file),
    )
    if not args.final_fit:
        model_args["early_stopping_rounds"] = args.early_stopping_rounds
    if args.snapshot_file:
        model_args["train_dir"] = str(
            Path(args.snapshot_file).resolve().parent / "catboost_info"
    )
    if args.random_strength is not None:
        model_args["random_strength"] = args.random_strength
    if args.subsample > 0:
        if not 0 < args.subsample <= 1:
            raise SystemExit("--subsample must be in (0, 1]")
        model_args["bootstrap_type"] = "Bernoulli"
        model_args["subsample"] = args.subsample
    if args.rsm > 0:
        if not 0 < args.rsm <= 1:
            raise SystemExit("--rsm must be in (0, 1]")
        model_args["rsm"] = args.rsm
    if args.objective == "ranker":
        if args.positive_class_weight > 0:
            raise SystemExit("--positive-class-weight is classifier-only")
        model = CatBoostRanker(
            loss_function=args.ranking_loss, eval_metric="NDCG:top=1", **model_args
        )
        train_pool = Pool(
            data["X"][row_train], data["y"][row_train],
            group_id=data["group"][row_train], feature_names=names,
        )
        validation_pool = Pool(
            data["X"][row_validation], data["y"][row_validation],
            group_id=data["group"][row_validation], feature_names=names,
        )
    else:
        if args.positive_class_weight > 0:
            model_args["class_weights"] = [1.0, args.positive_class_weight]
        else:
            model_args["auto_class_weights"] = "Balanced"
        model = CatBoostClassifier(
            loss_function="Logloss", eval_metric="AUC", **model_args
        )
        train_pool = Pool(data["X"][row_train], data["y"][row_train],
                          feature_names=names)
        validation_pool = Pool(data["X"][row_validation], data["y"][row_validation],
                               feature_names=names)
    snapshot_options = {}
    if args.snapshot_file:
        snapshot_path = Path(args.snapshot_file).resolve()
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_options = {
            "save_snapshot": True,
            "snapshot_file": str(snapshot_path),
            "snapshot_interval": args.snapshot_interval,
        }
        print(f"snapshot -> {snapshot_path} (every {args.snapshot_interval}s)")
    fit_started = time.perf_counter()
    model.fit(train_pool, eval_set=None if args.final_fit else validation_pool,
              use_best_model=not args.final_fit,
              **snapshot_options)

    fit_seconds = time.perf_counter() - fit_started

    validation_scores = (
        model.predict(validation_pool) if args.objective == "ranker"
        else model.predict_proba(validation_pool)[:, 1]
    )
    train_scores = (
        model.predict(train_pool) if args.objective == "ranker"
        else model.predict_proba(train_pool)[:, 1]
    )
    metrics = decision_metrics(
        validation_scores, data, decision_validation, row_validation,
        classifier=args.objective == "classifier",
    )
    training_metrics = decision_metrics(
        train_scores, data, decision_train, row_train,
        classifier=args.objective == "classifier",
    )

    sha, dirty = git_state()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{sha[:7]}-s{args.seed}"
    run_dir = Path(args.runs_root) / config["name"] / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_path = run_dir / "model.cbm"
    model.save_model(model_path)
    importance = model.get_feature_importance(type="PredictionValuesChange")
    feature_importance = [
        {"name": name, "importance": float(value)}
        for name, value in sorted(zip(names, importance), key=lambda item: item[1], reverse=True)
    ]
    metrics["scope"] = "training" if args.final_fit else "validation"
    training_metrics["scope"] = "training"
    if not args.final_fit:
        best_scores = model.get_best_score()["validation"]
        if args.objective == "ranker":
            ndcg_name = next(name for name in best_scores if name.startswith("NDCG"))
            metrics["ndcg_top1"] = float(best_scores[ndcg_name])
    manifest = {
        "run_id": run_id,
        "git_sha": sha,
        "git_dirty": dirty,
        "config": os.path.relpath(config["_path"], ROOT),
        "dataset": os.path.relpath(os.path.abspath(args.data), ROOT),
        "feature_version": feature_metadata["version"],
        "feature_count": len(names),
        "feature_names": names,
        "dropped_features": dropped_features,
        "max_episodes": args.max_episodes,
        "objective": args.objective,
        "ranking_loss": args.ranking_loss if args.objective == "ranker" else None,
        "parameters": {
            "iterations": args.iterations, "depth": args.depth,
            "learning_rate": args.lr, "l2_leaf_reg": args.l2_leaf_reg,
            "random_strength": args.random_strength,
            "positive_class_weight": args.positive_class_weight or None,
            "subsample": args.subsample or None,
            "rsm": args.rsm or None,
            "early_stopping_rounds": args.early_stopping_rounds,
            "threads": args.threads or (os.cpu_count() or 4),
            "valid_fraction": args.valid_frac,
            "seed": args.seed,
            "validation_episodes": args.validation_episodes or None,
            "final_fit": args.final_fit,
        },
        "timing_seconds": {"fit": fit_seconds, "elapsed_before_artifact_write": time.perf_counter() - started},
        "tree_count": int(model.tree_count_),
        "metrics": metrics,
        "training_metrics": training_metrics,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "training_metrics.json").write_text(
        json.dumps(training_metrics, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "feature_importance.json").write_text(
        json.dumps(feature_importance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"fit: {fit_seconds:.2f}s")
    print(f"run -> {run_dir}")
    print("training metrics")
    print(json.dumps(training_metrics, indent=2))
    print("training metrics (final fit)" if args.final_fit else "validation metrics")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
