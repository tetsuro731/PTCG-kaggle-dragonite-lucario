from __future__ import annotations

import glob
import os

import numpy as np


def load_shards(data_dir: str, max_episodes: int = 0) -> dict[str, np.ndarray]:
    paths = sorted(glob.glob(os.path.join(data_dir, "rows_*.npz")))
    if not paths:
        raise SystemExit(f"no shards under {data_dir}")
    accumulated: dict[str, list[np.ndarray]] = {}
    decision_offset = 0
    episode_offset = 0
    for path in paths:
        with np.load(path) as shard:
            part = {key: shard[key] for key in shard.files}
        if max_episodes:
            remaining = max_episodes - episode_offset
            if remaining <= 0:
                break
            part = limit_episodes(part, remaining)
        part["group"] = part["group"] + decision_offset
        part["dec_episode"] = part["dec_episode"] + episode_offset
        decision_offset += len(part["dec_sizes"])
        if len(part["dec_episode"]):
            episode_offset += int(part["dec_episode"].max()) + 1
        for key, value in part.items():
            accumulated.setdefault(key, []).append(value)
        if max_episodes and episode_offset >= max_episodes:
            break
    return {key: np.concatenate(values) for key, values in accumulated.items()}


def validation_mask(
    episodes: np.ndarray,
    groups: np.ndarray,
    valid_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_episodes = np.unique(episodes)
    rng.shuffle(unique_episodes)
    validation_count = max(1, int(len(unique_episodes) * valid_fraction))
    validation_episodes = set(unique_episodes[:validation_count].tolist())
    decision_mask = np.array([episode in validation_episodes for episode in episodes])
    return decision_mask, decision_mask[groups]


def filter_by_episode_ids(data: dict[str, np.ndarray], keep_ids: set[int]) -> dict[str, np.ndarray]:
    """Mask rows to decisions whose real ``dec_episode_id`` is in ``keep_ids``.

    Reindexes ``group`` so it stays 0-based contiguous over the kept decisions.
    Enables an instant LB-rank filter on a master dataset with no re-parse.
    """
    if "dec_episode_id" not in data:
        raise SystemExit("dataset has no dec_episode_id; regenerate with the updated generate_data")
    keep = np.fromiter((int(value) for value in keep_ids), dtype=np.int64)
    decision_mask = np.isin(data["dec_episode_id"], keep)
    row_mask = decision_mask[data["group"]]
    remapped_group = (np.cumsum(decision_mask) - 1).astype(np.int32)
    return {
        **data,
        "X": data["X"][row_mask],
        "y": data["y"][row_mask],
        "group": remapped_group[data["group"][row_mask]],
        **{key: value[decision_mask] for key, value in data.items() if key.startswith("dec_")},
    }


def limit_episodes(data: dict[str, np.ndarray], maximum: int) -> dict[str, np.ndarray]:
    if maximum <= 0:
        return data
    episodes = data["dec_episode"]
    keep_episodes = set(np.unique(episodes)[:maximum].tolist())
    decision_mask = np.array([episode in keep_episodes for episode in episodes])
    row_mask = decision_mask[data["group"]]
    decision_indexes = np.flatnonzero(decision_mask)
    remap = np.full(len(decision_mask), -1, dtype=np.int64)
    remap[decision_indexes] = np.arange(len(decision_indexes))
    limited: dict[str, np.ndarray] = {}
    for key, value in data.items():
        if key == "group":
            limited[key] = remap[value[row_mask]].astype(value.dtype)
        elif key in {"X", "y"}:
            limited[key] = value[row_mask]
        else:
            limited[key] = value[decision_mask]
    return limited


def topk_match(
    scores: np.ndarray,
    labels: np.ndarray,
    sizes: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Return one when each decision's top-k set equals the teacher's set."""
    matches = np.zeros(len(sizes), dtype=np.float64)
    offset = 0
    for index, size in enumerate(sizes):
        decision_slice = slice(offset, offset + size)
        offset += size
        count = int(counts[index])
        if count < size:
            selected = np.argpartition(-scores[decision_slice], count - 1)[:count]
        else:
            selected = np.arange(size)
        matches[index] = float(labels[decision_slice][selected].sum() == count)
    return matches