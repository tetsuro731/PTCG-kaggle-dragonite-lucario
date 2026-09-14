"""Deck-specific hooks used by the CatBoost training pipeline."""

from __future__ import annotations

from collections import Counter


def matches_deck(deck: list[int], config: dict) -> bool:
    counts = Counter(deck)
    return all(
        counts[int(card_id)] >= minimum
        for card_id, minimum in config["required_cards"].items()
    )


def reset_rule_policy(policy) -> None:
    policy.pre_turn = -1
    policy.ability_used = False
    policy.power_pro_used = 0
    policy.plan = policy.AttackPlan()
