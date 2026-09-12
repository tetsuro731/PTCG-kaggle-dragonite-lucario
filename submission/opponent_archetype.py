"""Infer recent-meta opponent deck types from causally revealed cards."""
from __future__ import annotations

import collections
from dataclasses import dataclass

try:
    from . import opponent_archetype_data as data
except ImportError:  # flattened Kaggle submission layout
    import opponent_archetype_data as data  # type: ignore[no-redef]


# Missing cards are strong negative evidence but not an absolute rejection.
# This prevents a hard 49/50-card boundary for unseen nearby deck variants.
MISSING_CARD_WEIGHT = 0.05

ARCHETYPE_SLUGS = tuple(item[0] for item in data.ARCHETYPES)
ARCHETYPE_LABELS = tuple(item[1] for item in data.ARCHETYPES)
ARCHETYPE_COUNTS = tuple(float(item[2]) for item in data.ARCHETYPES)
_TEMPLATES = tuple(
    (float(count), int(archetype), dict(pairs))
    for count, archetype, pairs in data.TEMPLATES
)


@dataclass(frozen=True)
class ArchetypeBelief:
    support: tuple[float, ...]
    probability: tuple[float, ...]
    top1: tuple[float, ...]
    other_probability: float
    revealed_count: float
    top_probability: float
    top_margin: float
    strict_candidate_count: float


def _card_ids(cards) -> list[int]:
    result = []
    for card in cards or ():
        card_id = getattr(card, "id", None)
        if card_id is not None:
            result.append(int(card_id))
    return result


def revealed_opponent_cards(observation) -> list[int]:
    """Return opponent cards visible in the current observation only.

    The current stadium is excluded because the observation does not identify
    which player put it into play. Once discarded it is attributable through
    the opponent discard pile and is included.
    """
    state = observation.current
    opponent = state.players[1 - int(state.yourIndex)]
    revealed = _card_ids(getattr(opponent, "discard", None))
    board = list(getattr(opponent, "active", None) or ())
    board += list(getattr(opponent, "bench", None) or ())
    for pokemon in board:
        if pokemon is None:
            continue
        card_id = getattr(pokemon, "id", None)
        if card_id is not None:
            revealed.append(int(card_id))
        revealed.extend(_card_ids(getattr(pokemon, "energyCards", None)))
        revealed.extend(_card_ids(getattr(pokemon, "tools", None)))
        revealed.extend(_card_ids(getattr(pokemon, "preEvolution", None)))
    return revealed


def infer_from_revealed(revealed: list[int]) -> ArchetypeBelief:
    observed = collections.Counter(int(card) for card in revealed)
    major_mass = [0.0] * len(ARCHETYPE_SLUGS)
    strict_candidates = 0
    other_mass = 0.0
    for frequency, archetype, template in _TEMPLATES:
        missing = sum(
            max(0, count - template.get(card_id, 0))
            for card_id, count in observed.items()
        )
        if missing == 0:
            strict_candidates += 1
        mass = frequency * (MISSING_CARD_WEIGHT ** missing)
        if archetype >= 0:
            major_mass[archetype] += mass
        else:
            other_mass += mass

    total_mass = sum(major_mass) + other_mass
    if total_mass <= 0.0:  # generated data always contains templates
        probability = [0.0] * len(major_mass)
        other_probability = 1.0
    else:
        probability = [mass / total_mass for mass in major_mass]
        other_probability = other_mass / total_mass
    # Support removes the prior: 1 means every observed list assigned to this
    # archetype remains compatible with the revealed multiset.
    support = [
        mass / count if count > 0.0 else 0.0
        for mass, count in zip(major_mass, ARCHETYPE_COUNTS)
    ]
    choices = probability + [other_probability]
    ranked = sorted(choices, reverse=True)
    best = ranked[0] if ranked else 0.0
    second = ranked[1] if len(ranked) > 1 else 0.0
    top1 = [0.0] * len(probability)
    if probability:
        top1[probability.index(max(probability))] = 1.0
    return ArchetypeBelief(
        support=tuple(support),
        probability=tuple(probability),
        top1=tuple(top1),
        other_probability=other_probability,
        revealed_count=float(sum(observed.values())),
        top_probability=best,
        top_margin=best - second,
        strict_candidate_count=float(strict_candidates),
    )


def infer(observation) -> ArchetypeBelief:
    return infer_from_revealed(revealed_opponent_cards(observation))


def feature_names() -> list[str]:
    names = [f"opp_archetype_{slug}_support" for slug in ARCHETYPE_SLUGS]
    names += [f"opp_archetype_{slug}_probability" for slug in ARCHETYPE_SLUGS]
    names += [f"opp_archetype_{slug}_top1" for slug in ARCHETYPE_SLUGS]
    names += [
        "opp_archetype_other_probability",
        "opp_archetype_revealed_count",
        "opp_archetype_top_probability",
        "opp_archetype_top_margin",
        "opp_archetype_strict_candidate_count",
    ]
    return names


def feature_values(belief: ArchetypeBelief) -> list[float]:
    return [
        *belief.support,
        *belief.probability,
        *belief.top1,
        belief.other_probability,
        belief.revealed_count,
        belief.top_probability,
        belief.top_margin,
        belief.strict_candidate_count,
    ]
