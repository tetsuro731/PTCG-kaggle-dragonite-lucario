#!/usr/bin/env python3
"""Shared plumbing for streaming Kaggle "Daily Top Episodes" replays.

Extracted from ``scripts/fetch_replays.py`` so that both the Transformer-BC
pipeline and the CatBoost pipeline can stream the same daily datasets without
duplicating the disk-safe download logic. ``fetch_replays.py`` itself is left
untouched to keep the existing BC results reproducible.

Disk contract: one daily zip (~20GB) at a time; callers decide whether to keep
it (``download_day(..., keep=True)``) so a second pass does not re-download.
"""
from __future__ import annotations

import csv
import os
import zipfile
from typing import Iterator

INDEX_DATASET = "kaggle/pokemon-tcg-ai-battle-episodes-index"
DECK_MIN_LEN = 55


def _episode_id(name: str) -> int | None:
    try:
        return int(os.path.splitext(os.path.basename(name))[0])
    except ValueError:
        return None


def kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def select_days(api, out_dir: str, days: int = 0, dfrom: str = "", dto: str = "") -> list[dict]:
    """Return manifest rows (one per day) sorted by date, filtered by range."""
    idx_dir = os.path.join(out_dir, "_index")
    os.makedirs(idx_dir, exist_ok=True)
    manifest = os.path.join(idx_dir, "manifest.csv")
    if not os.path.exists(manifest):
        if api is None:
            raise FileNotFoundError(f"offline replay index not found: {manifest}")
        api.dataset_download_files(INDEX_DATASET, path=idx_dir, unzip=True, quiet=True)
    with open(manifest, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: r["date"])
    if dfrom or dto:
        rows = [r for r in rows if (not dfrom or r["date"] >= dfrom) and (not dto or r["date"] <= dto)]
    elif days:
        rows = rows[-days:]
    return rows


def download_day(api, slug: str, tmp_dir: str) -> str:
    """Download one daily dataset zip (skipped if already present)."""
    os.makedirs(tmp_dir, exist_ok=True)
    zip_path = os.path.join(tmp_dir, f"{slug}.zip")
    if os.path.exists(zip_path):
        return zip_path
    if api is None:
        raise FileNotFoundError(f"offline replay archive not found: {zip_path}")
    api.dataset_download_files(f"kaggle/{slug}", path=tmp_dir, unzip=False, quiet=True)
    if os.path.exists(zip_path):
        return zip_path
    cands = [f for f in os.listdir(tmp_dir) if f.endswith(".zip")]
    return os.path.join(tmp_dir, cands[0]) if cands else zip_path


def iter_raw_episodes(zip_path: str, max_n: int = 0, keep: set | None = None,
                      with_id: bool = False) -> Iterator[bytes]:
    """Yield each ``<episodeId>.json`` as bytes, straight out of the zip.

    ``keep`` optionally restricts output to the given episode ids (filename stem).
    ``with_id`` yields ``(episode_id, bytes)`` tuples instead of bare bytes.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        if keep is not None:
            names = [n for n in names if _episode_id(n) in keep]
        if max_n:
            names = names[:max_n]
        for name in names:
            try:
                raw = zf.read(name)
            except Exception:  # noqa: BLE001 - a corrupt member must not stop the run
                continue
            yield (_episode_id(name), raw) if with_id else raw


def count_json_members(zip_path: str) -> int:
    with zipfile.ZipFile(zip_path) as zf:
        return sum(1 for n in zf.namelist() if n.endswith(".json"))


def decks_from_steps(steps) -> dict[int, list[int]]:
    """Each player's 60-card deck = their first action of length >= 55."""
    decks: dict[int, list[int]] = {}
    for st in steps:
        if not isinstance(st, list):
            continue
        for ai, ag in enumerate(st):
            a = ag.get("action")
            if ai not in decks and isinstance(a, list) and len(a) >= DECK_MIN_LEN:
                decks[ai] = [int(x) for x in a]
        if len(decks) >= 2:
            break
    return decks
