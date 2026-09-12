"""Opponent next-turn damage projections for Lucario CatBoost features.

The simulator only exposes our legal actions.  Opponent attacks therefore have
to be reconstructed from public card definitions and attached Energy.  Future
scenarios in this module are conditional upper bounds, not predictions that the
opponent actually holds an Energy or evolution card.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from types import SimpleNamespace

from cg.api import CardType, EnergyType, all_attack, all_card_data

try:
    from . import damage
    from . import incoming_damage_profiles as profiles
except ImportError:  # flattened Kaggle submission layout
    import damage  # type: ignore[no-redef]
    import incoming_damage_profiles as profiles  # type: ignore[no-redef]


# Dynamic quantities are named from the attacker's point of view.
REF_NONE = 0
REF_UNKNOWN = 1
REF_COIN = 2
REF_DEF_ACTIVE_ENERGY = 3
REF_BOTH_ACTIVE_ENERGY = 4
REF_DEF_ACTIVE_COUNTERS = 5
REF_DEF_ALL_COUNTERS = 6
REF_ATK_SELF_COUNTERS = 7
REF_DEF_PRIZE_TAKEN = 8
REF_ATK_PRIZE_TAKEN = 9
REF_DEF_BENCH_COUNT = 10
REF_ATK_BENCH_COUNT = 11
REF_BOTH_BENCH_COUNT = 12
REF_DEF_HAND_COUNT = 13
REF_ATK_HAND_COUNT = 14

_CARD = {card.cardId: card for card in all_card_data()}
_ATTACK = {attack.attackId: attack for attack in all_attack()}

_EVOLUTIONS_BY_NAME: dict[str, tuple[int, ...]] = {}
_evolution_lists: dict[str, list[int]] = {}
for _card in _CARD.values():
    if _card.cardType == CardType.POKEMON and _card.evolvesFrom:
        _evolution_lists.setdefault(_card.evolvesFrom, []).append(_card.cardId)
for _name, _ids in _evolution_lists.items():
    _EVOLUTIONS_BY_NAME[_name] = tuple(sorted(_ids))

_BASIC_ENERGIES = tuple(int(energy) for energy in (
    EnergyType.GRASS,
    EnergyType.FIRE,
    EnergyType.WATER,
    EnergyType.LIGHTNING,
    EnergyType.PSYCHIC,
    EnergyType.FIGHTING,
    EnergyType.DARKNESS,
    EnergyType.METAL,
))

_RE_FOR_EACH = re.compile(r"does (\d+)(?: more)? damage[^.]*? for each (.{0,100})")
_RE_DOES = re.compile(r"this attack does (\d+) damage")
_RE_PUT_COUNTERS = re.compile(
    r"put (\d+) damage counters? on (?:1 of )?your opponent(?:'s|s')"
)
_RE_IF_HEADS_MORE = re.compile(r"if heads, this attack does (\d+) more damage")
_RE_TAILS_NOTHING = re.compile(r"if tails, this attack does nothing")
_RE_FLIP_N = re.compile(r"flip (\d+) coins")
_RE_FLIP_UNTIL_TAILS = re.compile(r"flip a coin until you get tails")
_RE_FLIP_ONE = re.compile(r"flip a coin")
_DYNAMIC_MARKERS = (
    "more damage",
    "less damage",
    "for each",
    "times the number",
    "damage instead",
    "damage equal to",
)


@dataclass(frozen=True)
class DamageEstimate:
    value: float = 0.0
    known: bool = False
    apply_weakness_resistance: bool = True


@dataclass(frozen=True)
class _ParsedDamage:
    base: float
    per_unit: float
    ref: int
    coin_scale: float
    expected_heads: float
    apply_weakness_resistance: bool
    unresolved: bool


@dataclass(frozen=True)
class IncomingThreat:
    damage: float = 0.0
    margin: float = 0.0
    ko: bool = False
    known: bool = False
    usable_attack_count: int = 0
    bench_damage: float = 0.0
    bench_ko_count: int = 0


@dataclass(frozen=True)
class IncomingScenarios:
    now: IncomingThreat = IncomingThreat()
    plus_energy: IncomingThreat = IncomingThreat()
    evolution: IncomingThreat = IncomingThreat()
    evolution_plus_energy: IncomingThreat = IncomingThreat()
    plus_energy_unlocks_attack: bool = False
    evolution_candidate_count: int = 0


def _normalize(text: str | None) -> str:
    return (text or "").replace("\u2019", "'").replace("\xa0", " ").lower()


def _expected_heads(text: str) -> float:
    if _RE_FLIP_UNTIL_TAILS.search(text):
        return 1.0
    match = _RE_FLIP_N.search(text)
    if match:
        return int(match.group(1)) / 2.0
    return 0.5 if _RE_FLIP_ONE.search(text) else 0.0


def _classify_ref(tail: str) -> int:
    if "heads" in tail:
        return REF_COIN
    if "energy attached to your opponent's active" in tail:
        return REF_DEF_ACTIVE_ENERGY
    if "energy attached to both active" in tail:
        return REF_BOTH_ACTIVE_ENERGY
    if "damage counter on your opponent's active" in tail:
        return REF_DEF_ACTIVE_COUNTERS
    if "damage counter on all of your opponent's" in tail:
        return REF_DEF_ALL_COUNTERS
    if "damage counter on this pok" in tail:
        return REF_ATK_SELF_COUNTERS
    if "prize card your opponent has taken" in tail:
        return REF_DEF_PRIZE_TAKEN
    if "prize card you have taken" in tail:
        return REF_ATK_PRIZE_TAKEN
    if "card in your opponent's hand" in tail:
        return REF_DEF_HAND_COUNT
    if "card in your hand" in tail:
        return REF_ATK_HAND_COUNT
    if "your opponent's benched" in tail:
        return REF_DEF_BENCH_COUNT
    if "both yours and your opponent's" in tail and "benched" in tail:
        return REF_BOTH_BENCH_COUNT
    if ("of your benched" in tail or "on your bench" in tail
            or "your benched pok" in tail):
        return REF_ATK_BENCH_COUNT
    return REF_UNKNOWN


def _parse_damage(attack) -> _ParsedDamage:
    text = _normalize(attack.text)
    base = float(attack.damage)
    per_unit = 0.0
    ref = REF_NONE
    apply_weakness = True
    unresolved = False

    match = _RE_FOR_EACH.search(text)
    if match:
        per_unit = float(match.group(1))
        ref = _classify_ref(match.group(2))
        unresolved = ref == REF_UNKNOWN
    elif base == 0.0:
        match = _RE_DOES.search(text)
        # Damage restricted to the Bench does not threaten the Active Pokemon.
        if match and "benched pok" not in text:
            base = float(match.group(1))
        else:
            match = _RE_PUT_COUNTERS.search(text)
            if match:
                base = 10.0 * int(match.group(1))
                apply_weakness = False

    if per_unit == 0.0:
        match = _RE_IF_HEADS_MORE.search(text)
        if match:
            per_unit = float(match.group(1))
            ref = REF_COIN

    coin_scale = 0.5 if _RE_TAILS_NOTHING.search(text) else 1.0
    if any(marker in text for marker in _DYNAMIC_MARKERS):
        handled = per_unit > 0.0 or coin_scale != 1.0
        unresolved = unresolved or not handled
    return _ParsedDamage(
        base=base,
        per_unit=per_unit,
        ref=ref,
        coin_scale=coin_scale,
        expected_heads=_expected_heads(text),
        apply_weakness_resistance=apply_weakness,
        unresolved=unresolved,
    )


_PARSED_DAMAGE = {attack_id: _parse_damage(attack)
                  for attack_id, attack in _ATTACK.items()}


def estimate_attack_damage(attack_id: int, board: dict[int, float] | None = None) -> DamageEstimate:
    """Estimate expected damage from public quantities in ``board``."""
    override = profiles.active_override(
        attack_id, getattr(_ATTACK.get(attack_id), "text", ""), board or {}
    )
    if override is not None:
        return DamageEstimate(
            value=override.damage, known=override.known,
            apply_weakness_resistance=override.apply_weakness_resistance,
        )
    parsed = _PARSED_DAMAGE.get(attack_id)
    if parsed is None:
        return DamageEstimate()
    value = parsed.base
    known = not parsed.unresolved
    if parsed.per_unit > 0.0:
        if parsed.ref == REF_COIN:
            value += parsed.per_unit * parsed.expected_heads
        elif board is not None and parsed.ref in board:
            value += parsed.per_unit * float(board[parsed.ref])
        else:
            known = False
    return DamageEstimate(
        value=value * parsed.coin_scale,
        known=known,
        apply_weakness_resistance=parsed.apply_weakness_resistance,
    )


def can_pay_energy_cost(attached, cost) -> bool:
    """Return whether attached Energy can pay a typed attack cost."""
    available = [int(energy) for energy in (attached or [])]
    required = [int(energy) for energy in (cost or [])]
    colorless = sum(energy == int(EnergyType.COLORLESS) for energy in required)
    colored = [energy for energy in required if energy != int(EnergyType.COLORLESS)]
    if len(available) < len(required):
        return False

    def matches(energy: int, requirement: int) -> bool:
        if energy == requirement or energy == int(EnergyType.RAINBOW):
            return True
        return (energy == int(EnergyType.TEAM_ROCKET)
                and requirement in {int(EnergyType.PSYCHIC), int(EnergyType.DARKNESS)})

    # Assign the least-flexible colored requirement first. Attack costs in the
    # current pool contain at most five Energy, so exhaustive matching is tiny.
    colored.sort(key=lambda requirement: sum(matches(e, requirement) for e in available))

    def assign(index: int, remaining: tuple[int, ...]) -> bool:
        if index == len(colored):
            return len(remaining) >= colorless
        requirement = colored[index]
        for position, energy in enumerate(remaining):
            if matches(energy, requirement) and assign(
                index + 1, remaining[:position] + remaining[position + 1:]
            ):
                return True
        return False

    return assign(0, tuple(available))


def evolution_candidates(card_id: int) -> tuple[int, ...]:
    card = _CARD.get(card_id)
    if card is None:
        return ()
    return _EVOLUTIONS_BY_NAME.get(card.name, ())


def _active(player):
    active = getattr(player, "active", None) or []
    return active[0] if active else None


def _bench(player):
    return getattr(player, "bench", None) or []


def _count(values) -> int:
    return len(values or [])


def _hand_count(player) -> int:
    count = getattr(player, "handCount", None)
    return int(count) if count is not None else _count(getattr(player, "hand", None))


def _bench_has(player, card_id: int) -> bool:
    return any(
        pokemon is not None and int(getattr(pokemon, "id", 0)) == card_id
        for pokemon in _bench(player)
    )


def _damage_taken(pokemon) -> float:
    if pokemon is None:
        return 0.0
    card = _CARD.get(pokemon.id)
    return max(0.0, float(card.hp) - float(pokemon.hp)) if card else 0.0


def _all_damage_counters(player) -> float:
    pokemon = (getattr(player, "active", None) or []) + _bench(player)
    return sum(_damage_taken(item) for item in pokemon) / 10.0


def _board_values(attacker_player, defender_player, attacker, defender,
                  stadium_id: int, hand_adjustment: int = 1,
                  attacker_hand_count: int | None = None,
                  defender_hand_count: int | None = None) -> dict[object, float]:
    attacker_hand = (
        _hand_count(attacker_player)
        if attacker_hand_count is None else attacker_hand_count
    ) + hand_adjustment
    defender_hand = (
        _hand_count(defender_player)
        if defender_hand_count is None else defender_hand_count
    )
    return {
        REF_DEF_ACTIVE_ENERGY: float(_count(getattr(defender, "energies", None))),
        REF_BOTH_ACTIVE_ENERGY: float(
            _count(getattr(attacker, "energies", None))
            + _count(getattr(defender, "energies", None))
        ),
        REF_DEF_ACTIVE_COUNTERS: _damage_taken(defender) / 10.0,
        REF_DEF_ALL_COUNTERS: _all_damage_counters(defender_player),
        REF_ATK_SELF_COUNTERS: _damage_taken(attacker) / 10.0,
        REF_DEF_PRIZE_TAKEN: float(max(0, 6 - _count(getattr(defender_player, "prize", None)))),
        REF_ATK_PRIZE_TAKEN: float(max(0, 6 - _count(getattr(attacker_player, "prize", None)))),
        REF_DEF_BENCH_COUNT: float(_count(_bench(defender_player))),
        REF_ATK_BENCH_COUNT: float(_count(_bench(attacker_player))),
        REF_BOTH_BENCH_COUNT: float(
            _count(_bench(attacker_player)) + _count(_bench(defender_player))
        ),
        REF_DEF_HAND_COUNT: float(defender_hand),
        REF_ATK_HAND_COUNT: float(attacker_hand),
        "attacker_hand_next": float(attacker_hand),
        "defender_hand_next": float(defender_hand),
        "attacker_energy": float(_count(getattr(attacker, "energies", None))),
        "stadium_present": float(bool(stadium_id)),
        "attacker_has_lunatone": float(_bench_has(attacker_player, 675)),
    }


def _attacker_view(source, card_id: int, energies: list[int]):
    old_damage = _damage_taken(source)
    card = _CARD.get(card_id)
    hp = max(0.0, float(card.hp) - old_damage) if card else float(source.hp)
    return SimpleNamespace(id=card_id, hp=hp, energies=energies)


def _estimated_breakdown(attacker, defender, attack_id: int, stadium_id: int,
                         board: dict[int, float], effects,
                         defender_player_index: int):
    estimate = estimate_attack_damage(attack_id, board)
    static = damage.calculate_attack_damage(
        attacker,
        defender,
        attack_id,
        stadium_id,
        effects,
        defender_player_index=defender_player_index,
    )
    before = estimate.value + static.bonus_before_weakness
    weakness = static.weakness_multiplier if estimate.apply_weakness_resistance else 1.0
    resistance = static.resistance_reduction if estimate.apply_weakness_resistance else 0.0
    after_weakness = before * weakness
    after_resistance = max(0.0, after_weakness - resistance)
    ignores_defender = attack_id in profiles.IGNORE_DEFENDER_POKEMON_EFFECTS
    reduction = (
        0.0 if ignores_defender else static.reduction_after_weakness_resistance
    )
    pokemon_prevented = static.prevented_by_pokemon and not ignores_defender
    after_reduction = max(0.0, after_resistance - reduction)
    final = 0.0 if pokemon_prevented or static.prevented_by_stadium else after_reduction
    hp = float(defender.hp)
    return replace(
        static,
        base=estimate.value,
        before_weakness_resistance=before,
        weakness_multiplier=weakness,
        resistance_reduction=resistance,
        after_weakness_resistance=after_resistance,
        reduction_after_weakness_resistance=reduction,
        prevented_by_pokemon=pokemon_prevented,
        final=final,
        hp_margin=final - hp,
        ko=bool(hp > 0 and final >= hp),
        known=estimate.known,
    )


def _threat_for_card(card_id: int, energies: list[int], source_attacker, defender,
                     attacker_player, defender_player, stadium_id: int, effects,
                     defender_player_index: int,
                     attacker_hand_count: int | None = None,
                     defender_hand_count: int | None = None) -> IncomingThreat:
    card = _CARD.get(card_id)
    if card is None or defender is None:
        return IncomingThreat(known=False)
    attacker = _attacker_view(source_attacker, card_id, energies)
    # Evolving Kadabra into Alakazam: start-turn draw +1, play Alakazam -1,
    # Psychic Draw +3 => current observed hand +3. Existing Alakazam gets +1.
    hand_adjustment = (
        3 if card_id == 743 and int(getattr(source_attacker, "id", 0)) != 743 else 1
    )
    board = _board_values(
        attacker_player, defender_player, attacker, defender, stadium_id,
        hand_adjustment, attacker_hand_count, defender_hand_count
    )
    breakdowns = []
    bench_profiles = []
    for attack_id in card.attacks:
        attack = _ATTACK.get(attack_id)
        if attack is None or not can_pay_energy_cost(energies, attack.energies):
            continue
        breakdowns.append(_estimated_breakdown(
            attacker, defender, attack_id, stadium_id, board, effects,
            defender_player_index,
        ))
        bench_profiles.append(profiles.bench_profile(attack_id, attack.text, board))
    hp = float(defender.hp)
    if not breakdowns:
        return IncomingThreat(margin=-hp, known=True)
    maximum = max(item.final for item in breakdowns)
    bench_damage = max((item.damage for item in bench_profiles), default=0.0)
    bench_kos = max(
        (profiles.bench_ko_count(item, _bench(defender_player))
         for item in bench_profiles),
        default=0,
    )
    return IncomingThreat(
        damage=maximum,
        margin=maximum - hp,
        ko=bool(hp > 0 and maximum >= hp),
        known=(
            all(item.known for item in breakdowns)
            and all(item.known for item in bench_profiles)
        ),
        usable_attack_count=len(breakdowns),
        bench_damage=bench_damage,
        bench_ko_count=bench_kos,
    )

def _combine(threats: list[IncomingThreat], hp: float) -> IncomingThreat:
    if not threats:
        return IncomingThreat(margin=-hp, known=False)
    maximum = max(threat.damage for threat in threats)
    return IncomingThreat(
        damage=maximum,
        margin=maximum - hp,
        ko=bool(hp > 0 and maximum >= hp),
        known=all(threat.known for threat in threats),
        usable_attack_count=max(threat.usable_attack_count for threat in threats),
        bench_damage=max(threat.bench_damage for threat in threats),
        bench_ko_count=max(threat.bench_ko_count for threat in threats),
    )


def evaluate(observation, effects=None, defender=None,
             attacker_hand_count: int | None = None,
             defender_hand_count: int | None = None) -> IncomingScenarios:
    """Calculate four opponent next-turn damage scenarios."""
    state = observation.current
    if state is None:
        return IncomingScenarios()
    mine = int(state.yourIndex)
    opponent = 1 - mine
    defender_player = state.players[mine]
    attacker_player = state.players[opponent]
    if defender is None:
        defender = _active(defender_player)
    attacker = _active(attacker_player)
    if attacker is None or defender is None or not getattr(attacker, "id", 0):
        return IncomingScenarios()

    stadium_id = state.stadium[0].id if state.stadium else 0
    projected_effects = effects
    if effects is not None:
        # Model decisions occur on our turn; incoming damage lands on the next
        # turn. This makes an Iron Defender played now apply to the projection.
        # Premium Power Pro uses belong to our current turn and must not be
        # transferred to the opponent's Fighting attacker.
        projected_effects = replace(
            effects, turn=int(state.turn) + 1, premium_power_pro_used=0
        )
    energies = [int(energy) for energy in (attacker.energies or [])]
    hp = float(defender.hp)

    def threat(card_id: int, attached: list[int]) -> IncomingThreat:
        return _threat_for_card(
            card_id,
            attached,
            attacker,
            defender,
            attacker_player,
            defender_player,
            stadium_id,
            projected_effects,
            mine,
            attacker_hand_count,
            defender_hand_count,
        )

    now = threat(attacker.id, energies)
    plus_energy_options = [now] + [
        threat(attacker.id, energies + [energy]) for energy in _BASIC_ENERGIES
    ]
    plus_energy_options.extend(
        threat(attacker.id, energies + list(bundle))
        for bundle in profiles.extra_energy_bundles(attacker.id)
    )
    plus_energy = _combine(plus_energy_options, hp)

    candidates = evolution_candidates(attacker.id)
    evolution_options = [now] + [threat(card_id, energies) for card_id in candidates]
    for card_id in candidates:
        evolution_options.extend(
            threat(card_id, energies + list(bundle))
            for bundle in profiles.evolution_energy_bundles(card_id)
        )
    evolution = _combine(evolution_options, hp)

    evolution_plus_options = list(plus_energy_options)
    for card_id in candidates:
        evolution_plus_options.append(threat(card_id, energies))
        evolution_plus_options.extend(
            threat(card_id, energies + [energy]) for energy in _BASIC_ENERGIES
        )
        for bundle in profiles.evolution_energy_bundles(card_id):
            boosted = energies + list(bundle)
            evolution_plus_options.append(threat(card_id, boosted))
            evolution_plus_options.extend(
                threat(card_id, boosted + [energy]) for energy in _BASIC_ENERGIES
            )
    evolution_plus_energy = _combine(evolution_plus_options, hp)
    return IncomingScenarios(
        now=now,
        plus_energy=plus_energy,
        evolution=evolution,
        evolution_plus_energy=evolution_plus_energy,
        plus_energy_unlocks_attack=(
            plus_energy.usable_attack_count > now.usable_attack_count
        ),
        evolution_candidate_count=len(candidates),
    )
