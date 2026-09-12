from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "extracted" / "sample_submission"))

from scripts.replay_stream import (  # noqa: E402
    DECK_MIN_LEN,
    decks_from_steps,
    download_day,
    iter_raw_episodes,
    kaggle_api,
    select_days,
)

from .config import import_module, load_config

WORKER: dict = {}


def initialize_worker(config_path: str, with_rule: bool) -> None:
    import cg.api as cg_api
    import cg.utils as cg_utils
    import orjson
    from scripts.fastparse import fast_to_dataclass

    config = load_config(config_path)
    cg_utils.to_dataclass = fast_to_dataclass
    cg_api.to_dataclass = fast_to_dataclass
    WORKER.update({
        "config": config,
        "training": import_module(config, "training_module"),
        "features": import_module(config, "feature_module"),
        "orjson": orjson,
        "to_observation": cg_api.to_observation_class,
    })
    if with_rule:
        WORKER["rule"] = import_module(config, "rule_policy_module")


def parse_episode(item):
    episode_id, raw = item
    stats = collections.Counter()
    try:
        episode = WORKER["orjson"].loads(raw)
    except Exception:  # noqa: BLE001
        stats["json_error"] += 1
        return None, stats
    steps = episode.get("steps") or []
    rewards = episode.get("rewards") or []
    winner = rewards.index(1) if 1 in rewards else -1
    if winner < 0:
        return None, stats
    deck = decks_from_steps(steps).get(winner, [])
    if not WORKER["training"].matches_deck(deck, WORKER["config"]):
        return None, stats

    rule = WORKER.get("rule")
    if rule is not None:
        WORKER["training"].reset_rule_policy(rule)
    features_module = WORKER["features"]
    rows: list[list[float]] = []
    labels: list[int] = []
    decision: dict[str, list[int]] = {
        "sizes": [], "context": [], "max_count": [], "turn": [],
        "teacher_count": [], "rule_match": [],
    }
    for step_index in range(len(steps) - 1):
        current, following = steps[step_index], steps[step_index + 1]
        if not isinstance(current, list) or not isinstance(following, list) or winner >= len(current):
            continue
        agent = current[winner]
        if agent.get("status") != "ACTIVE":
            continue
        action = following[winner].get("action")
        if not isinstance(action, list) or not action or len(action) >= DECK_MIN_LEN:
            continue
        observation_dict = agent.get("observation")
        if not observation_dict:
            continue
        try:
            observation = WORKER["to_observation"](observation_dict)
        except Exception:  # noqa: BLE001
            stats["parse_error"] += 1
            continue
        if observation.select is None or not observation.select.option:
            continue
        rule_choice = None
        if rule is not None:
            try:
                rule_choice = rule.agent(observation_dict)
            except Exception:  # noqa: BLE001
                stats["rule_error"] += 1
        if not features_module.is_own_turn(observation.current):
            continue
        option_count = len(observation.select.option)
        if option_count < 2:
            continue
        if min(action) < 0 or max(action) >= option_count:
            stats["action_range_error"] += 1
            continue
        try:
            feature_builder = features_module.Features(observation, deck)
            option_rows = [feature_builder.state + feature_builder.option(index)
                           for index in range(option_count)]
        except Exception:  # noqa: BLE001
            stats["encode_error"] += 1
            continue
        chosen = set(action)
        rows.extend(option_rows)
        labels.extend(int(index in chosen) for index in range(option_count))
        decision["sizes"].append(option_count)
        decision["context"].append(int(observation.select.context))
        decision["max_count"].append(int(observation.select.maxCount))
        decision["turn"].append(int(observation.current.turn))
        decision["teacher_count"].append(len(action))
        decision["rule_match"].append(
            -1 if rule_choice is None else int(sorted(rule_choice) == sorted(action))
        )
    if not rows:
        return None, stats
    return {
        "X": np.asarray(rows, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.uint8),
        "episode_id": int(episode_id) if episode_id is not None else -1,
        **{key: np.asarray(value, dtype=np.int32) for key, value in decision.items()},
    }, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate option-level CatBoost rows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--from", dest="date_from", default="")
    parser.add_argument("--to", dest="date_to", default="")
    parser.add_argument("--replays", default="data/replays")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-per-day", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--keep-zip", action="store_true")
    parser.add_argument("--keep-episodes", default="",
                        help="json list of episode_ids to parse (e.g. LB top-N winners)")
    parser.add_argument("--no-rule", action="store_true")
    parser.add_argument("--offline", action="store_true",
                        help="use an existing replay index and zip files without Kaggle auth")
    args = parser.parse_args()

    keep = None
    if args.keep_episodes:
        keep = {int(value) for value in json.load(open(args.keep_episodes))}
        print(f"keep-set: {len(keep)} episode ids from {args.keep_episodes}", flush=True)

    config = load_config(args.config)
    features_module = import_module(config, "feature_module")
    workers = args.workers or (os.cpu_count() or 1)
    os.makedirs(args.out, exist_ok=True)
    api = None if args.offline else kaggle_api()
    days = select_days(api, args.replays, args.days, args.date_from, args.date_to)
    if not days:
        raise SystemExit("no replay days selected")
    feature_metadata = {
        "version": features_module.FEATURE_VERSION,
        "n_state": features_module.N_STATE,
        "n_option": features_module.N_OPTION,
        "names": features_module.FEATURE_NAMES,
        "own_cards": features_module.OWN_CARDS,
        "opp_poke": features_module.OPP_POKE,
    }
    with open(os.path.join(args.out, "features.json"), "w", encoding="utf-8") as file:
        json.dump(feature_metadata, file, ensure_ascii=False, indent=1)

    context = multiprocessing.get_context("spawn")
    pool = context.Pool(workers, initializer=initialize_worker,
                        initargs=(config["_path"], not args.no_rule))
    generated = []
    try:
        for metadata in days:
            day, slug = metadata["date"], metadata["daily_dataset_slug"]
            shard_path = os.path.join(args.out, f"rows_{day}.npz")
            if os.path.exists(shard_path):
                print(f"[{day}] shard exists, skip")
                generated.append(shard_path)
                continue
            started = time.perf_counter()
            zip_path = download_day(api, slug, os.path.join(args.replays, "_zip"))
            parts, total = [], collections.Counter()
            episodes = iter_raw_episodes(zip_path, args.max_per_day, keep, with_id=True)
            for output, stats in pool.imap_unordered(parse_episode, episodes, chunksize=4):
                total.update(stats)
                if output is not None:
                    parts.append(output)
            if not parts:
                print(f"[{day}] no matching rows")
                continue
            sizes = np.concatenate([part["sizes"] for part in parts])
            labels = np.concatenate([part["y"] for part in parts])
            np.savez(
                shard_path,
                X=np.concatenate([part["X"] for part in parts]),
                y=labels,
                group=np.repeat(np.arange(len(sizes), dtype=np.int32), sizes),
                dec_sizes=sizes,
                dec_episode=np.concatenate([
                    np.full(len(part["sizes"]), index, dtype=np.int32)
                    for index, part in enumerate(parts)
                ]),
                dec_episode_id=np.concatenate([
                    np.full(len(part["sizes"]), part["episode_id"], dtype=np.int64)
                    for part in parts
                ]),
                dec_ctx=np.concatenate([part["context"] for part in parts]),
                dec_maxcount=np.concatenate([part["max_count"] for part in parts]),
                dec_turn=np.concatenate([part["turn"] for part in parts]),
                dec_k=np.concatenate([part["teacher_count"] for part in parts]),
                dec_rule_match=np.concatenate([part["rule_match"] for part in parts]),
            )
            generated.append(shard_path)
            print(f"[{day}] episodes={len(parts)} decisions={len(sizes):,} rows={len(labels):,} "
                  f"seconds={time.perf_counter() - started:.0f} diagnostics={dict(total)}")
            if not args.keep_zip:
                os.remove(zip_path)
    finally:
        pool.close()
        pool.join()

    dataset_manifest = {
        "config": os.path.relpath(config["_path"], ROOT),
        "date_from": days[0]["date"],
        "date_to": days[-1]["date"],
        "feature_version": features_module.FEATURE_VERSION,
        "feature_count": features_module.N_FEATURES,
        "shards": [
            {"name": os.path.basename(path), "bytes": os.path.getsize(path)}
            for path in generated
        ],
    }
    with open(os.path.join(args.out, "dataset_manifest.json"), "w", encoding="utf-8") as file:
        json.dump(dataset_manifest, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
