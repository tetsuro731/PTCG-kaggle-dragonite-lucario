from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "data", "extracted", "sample_submission"))

from cg.api import SelectContext  # noqa: E402

from .common import load_shards

CONTEXT_NAMES = {int(context): context.name for context in SelectContext}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a CatBoost dataset and rule baseline.")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    data = load_shards(args.data)
    with open(os.path.join(args.data, "features.json"), encoding="utf-8") as file:
        names = json.load(file)["names"]
    features, labels = data["X"], data["y"]

    print("=== data health ===")
    print(f"rows={len(labels):,} decisions={len(data['dec_sizes']):,} "
          f"episodes={len(np.unique(data['dec_episode'])):,}")
    print(f"features={features.shape[1]} expected={len(names)} positives={labels.mean():.1%}")
    print(f"NaN cells={int(np.isnan(features).sum())}")
    column_min, column_max = features.min(axis=0), features.max(axis=0)
    constants = [names[index] for index in range(features.shape[1])
                 if column_min[index] == column_max[index]]
    print(f"constant columns={len(constants)}")
    positives = np.bincount(data["group"], weights=labels,
                            minlength=len(data["dec_sizes"]))
    print(f"decisions with positive count != teacher k: "
          f"{int((positives != data['dec_k']).sum())}")

    print("\n=== rule-policy baseline ===")
    rule = data["dec_rule_match"]
    known = rule >= 0
    print(f"evaluated={known.sum():,} errors={(~known).sum():,} match={rule[known].mean():.1%}")
    context = data["dec_ctx"]
    rows = []
    for value in np.unique(context):
        selected = known & (context == value)
        if selected.any():
            rows.append((int(selected.sum()), CONTEXT_NAMES.get(int(value), str(value)),
                         float(rule[selected].mean())))
    print(f"{'context':<28}{'n':>8}{'match':>9}")
    for count, name, match in sorted(rows, reverse=True):
        print(f"{name:<28}{count:>8,}{match:>8.1%}")


if __name__ == "__main__":
    main()