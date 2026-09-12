"""Broad attack-text profiles and targeted current-meta damage overrides.

This module is deliberately independent from ``incoming_damage`` so its text
coverage can be audited without a rules-engine observation. Generic patterns
cover fixed Bench/any-Pokemon damage throughout the card pool; small explicit
overrides cover public next-turn conditions that text parsing alone cannot
infer accurately.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from cg.api import EnergyType


ALAKAZAM_POWERFUL_HAND = 1072
CHANDELURE_MIND_RULER = 123
KROOKODILE_CURSED_SLUG = 771
GOTHITELLE_SYNCHRO_SHOT = 857
MEGA_FROSLASS_RESENTFUL_REFRAIN = 1240
MEDICHAM_SEVENTH_KICK = 1275
IRON_BOULDER_ADJUSTED_HORN = 1396
FAN_ROTOM_ASSAULT_LANDING = 230
SOLROCK_COSMIC_BEAM = 980
MEGA_LOPUNNY_GALE_THRUST = 1225
TEAL_MASK_OGERPON_EX = 96
MARNIES_GRIMMSNARL_EX = 648

IGNORE_DEFENDER_POKEMON_EFFECTS = {479, 1226}  # Superb Scissors, Spiky Hopper

_ANY_DIRECT = re.compile(
    r"this attack (?:also )?does (\d+) damage to (?:(\d+) of )?"
    r"your opponent's pok"
)
_BENCH_DIRECT = re.compile(
    r"(?:also )?does (\d+) damage to (?:(\d+) of |1 of )?"
    r"your opponent's benched pok"
)
_BENCH_EACH = re.compile(
    r"(?:also )?does (\d+) damage to each of your opponent's benched pok"
)
_BENCH_COUNTER_POOL = re.compile(
    r"(?:put|place) (\d+) damage counters? on your opponent's benched pok"
    r"[^.]*any way"
)
_ANY_COUNTER_TARGET = re.compile(
    r"(?:put|place) (\d+) damage counters? on 1 of your opponent's pok"
)
_ANY_COUNTER_POOL = re.compile(
    r"(?:put|place) (\d+) damage counters? on your opponent's pok[^.]*any way"
)
_TARGET_PER_SELF_ENERGY = re.compile(
    r"does (\d+) damage to 1 of your opponent's pok[^.]*"
    r"for each (?:\{[a-z]+\} )?energy attached to this pok"
)


def normalize(text: str | None) -> str:
    return (text or "").replace("\u2019", "'").replace("\xa0", " ").lower()


@dataclass(frozen=True)
class ActiveOverride:
    damage: float
    known: bool = True
    apply_weakness_resistance: bool = True


@dataclass(frozen=True)
class BenchProfile:
    damage: float = 0.0
    total: float = 0.0
    targets: int = 0
    each: bool = False
    counter_pool: bool = False
    known: bool = True


def active_override(attack_id: int, text: str | None, board: dict) -> ActiveOverride | None:
    """Return active-target damage when generic ``Attack.damage`` is insufficient."""
    if attack_id == ALAKAZAM_POWERFUL_HAND:
        # The opponent draws once at the start of their turn. If Alakazam is
        # evolved during that turn, incoming_damage adds the net Psychic Draw
        # adjustment to ``attacker_hand_next`` before calling this function.
        return ActiveOverride(
            20.0 * float(board.get("attacker_hand_next", 0.0)),
            apply_weakness_resistance=False,
        )
    attacker_hand = float(board.get("attacker_hand_next", 0.0))
    defender_hand = float(board.get("defender_hand_next", 0.0))
    if attack_id == CHANDELURE_MIND_RULER:
        return ActiveOverride(30.0 * defender_hand)
    if attack_id == KROOKODILE_CURSED_SLUG:
        return ActiveOverride(150.0 if defender_hand <= 3 else 30.0)
    if attack_id == GOTHITELLE_SYNCHRO_SHOT:
        return ActiveOverride(180.0 if attacker_hand == defender_hand else 90.0)
    if attack_id == MEGA_FROSLASS_RESENTFUL_REFRAIN:
        return ActiveOverride(50.0 * defender_hand)
    if attack_id == MEDICHAM_SEVENTH_KICK:
        return ActiveOverride(150.0 if attacker_hand == 7 else 0.0)
    if attack_id == IRON_BOULDER_ADJUSTED_HORN:
        return ActiveOverride(170.0 if attacker_hand == defender_hand else 0.0)
    if attack_id == FAN_ROTOM_ASSAULT_LANDING:
        return ActiveOverride(70.0 if board.get("stadium_present") else 0.0)
    if attack_id == SOLROCK_COSMIC_BEAM:
        return ActiveOverride(
            70.0 if board.get("attacker_has_lunatone") else 0.0,
            apply_weakness_resistance=False,
        )
    if attack_id == MEGA_LOPUNNY_GALE_THRUST:
        # Conditional upper bound: the deck can move Lopunny from Bench to
        # Active, but the current observation does not prove that it will.
        return ActiveOverride(230.0, known=False)

    value = normalize(text)
    per_energy = _TARGET_PER_SELF_ENERGY.search(value)
    if per_energy:
        return ActiveOverride(
            float(per_energy.group(1)) * float(board.get("attacker_energy", 0.0)),
            apply_weakness_resistance=False,
        )
    target = _ANY_COUNTER_TARGET.search(value) or _ANY_COUNTER_POOL.search(value)
    if target:
        return ActiveOverride(
            10.0 * float(target.group(1)),
            apply_weakness_resistance=False,
        )
    direct = _ANY_DIRECT.search(value)
    if direct and "benched pok" not in direct.group(0):
        ignores_wr = "isn't affected by weakness or resistance" in value
        return ActiveOverride(float(direct.group(1)), apply_weakness_resistance=not ignores_wr)
    # Oil Salvo chooses the same Pokemon up to six times.
    if attack_id == 564:
        return ActiveOverride(120.0, apply_weakness_resistance=False)
    return None


def bench_profile(attack_id: int, text: str | None, board: dict) -> BenchProfile:
    """Parse damage that can reach the opponent's Bench.

    ``damage`` is the maximum for one Benched Pokemon; ``total`` is the total
    distributable damage. ``targets`` limits how many Pokemon can be selected.
    """
    value = normalize(text)
    match = _BENCH_EACH.search(value)
    if match:
        damage = float(match.group(1))
        return BenchProfile(damage=damage, total=damage, each=True)
    match = _BENCH_COUNTER_POOL.search(value)
    if match:
        total = 10.0 * float(match.group(1))
        return BenchProfile(damage=total, total=total, counter_pool=True)
    match = _BENCH_DIRECT.search(value)
    if match:
        damage = float(match.group(1))
        targets = int(match.group(2) or 1)
        return BenchProfile(damage=damage, total=damage * targets, targets=targets)

    per_energy = _TARGET_PER_SELF_ENERGY.search(value)
    if per_energy:
        damage = float(per_energy.group(1)) * float(board.get("attacker_energy", 0.0))
        return BenchProfile(damage=damage, total=damage, targets=1)
    match = _ANY_COUNTER_TARGET.search(value)
    if match:
        damage = 10.0 * float(match.group(1))
        return BenchProfile(
            damage=damage, total=damage, targets=1, counter_pool=True
        )
    match = _ANY_COUNTER_POOL.search(value)
    if match:
        total = 10.0 * float(match.group(1))
        return BenchProfile(damage=total, total=total, counter_pool=True)
    match = _ANY_DIRECT.search(value)
    if match and "benched pok" not in match.group(0):
        damage = float(match.group(1))
        targets = int(match.group(2) or 1)
        return BenchProfile(damage=damage, total=damage * targets, targets=targets)
    if attack_id == 564:  # Oil Salvo: six selections, repeatable target
        return BenchProfile(damage=120.0, total=120.0, counter_pool=True)
    return BenchProfile()


def bench_ko_count(profile: BenchProfile, bench) -> int:
    hp = sorted(float(pokemon.hp) for pokemon in (bench or ()) if pokemon is not None)
    if not hp or profile.damage <= 0.0:
        return 0
    if profile.each:
        return sum(value <= profile.damage for value in hp)
    if profile.counter_pool:
        remaining = profile.total
        knocked_out = 0
        for value in hp:
            if value > remaining:
                break
            remaining -= value
            knocked_out += 1
        return knocked_out
    return min(profile.targets, sum(value <= profile.damage for value in hp))


def extra_energy_bundles(card_id: int) -> tuple[tuple[int, ...], ...]:
    """Extra next-turn attachments beyond the generic one-card scenarios."""
    if card_id == TEAL_MASK_OGERPON_EX:
        grass = int(EnergyType.GRASS)
        # Teal Dance plus the normal attachment can add two Grass Energy.
        return ((grass, grass),)
    return ()


def evolution_energy_bundles(card_id: int) -> tuple[tuple[int, ...], ...]:
    if card_id == MARNIES_GRIMMSNARL_EX:
        # Punk Up can attach up to five Basic Darkness Energy on evolution.
        return ((int(EnergyType.DARKNESS),) * 5,)
    return ()
