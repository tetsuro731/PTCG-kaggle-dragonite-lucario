"""Deterministic attack-damage features for the Lucario action scorer.

Card effects are kept in an explicit registry. Parsing arbitrary card text at
inference would make training and serving fragile, so unsupported effects are
left out instead of being guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from cg.api import CardType, EnergyType, LogType, OptionType, all_attack, all_card_data

PREMIUM_POWER_PRO = 1141
IRON_DEFENDER = 1140
NEUTRALIZATION_ZONE = 1247
FARIGIRAF_EX = 83
SYLVEON = 330
CRUSTLE = 345
_DYNAMIC_DAMAGE_MARKERS = (
    "more damage", "less damage", "for each", "times the number",
    "damage for every", "damage instead", "damage equal to",
)

_CARD = {card.cardId: card for card in all_card_data()}
_ATTACK = {attack.attackId: attack for attack in all_attack()}
STADIUM_IDS = tuple(sorted(
    card.cardId for card in _CARD.values() if card.cardType == CardType.STADIUM
))


@dataclass
class TurnEffects:
    """Hidden turn state that is not fully represented in ``State``."""

    turn: int = -1
    player_index: int = -1
    premium_power_pro_used: int = 0
    iron_defender_owner: int = -1
    iron_defender_active_turn: int = -1

    def sync(self, observation) -> None:
        state = observation.current
        if state is None:
            return
        new_game = self.turn >= 0 and int(state.turn) < self.turn
        if new_game:
            self.iron_defender_owner = -1
            self.iron_defender_active_turn = -1
        if self.turn != int(state.turn) or self.player_index != int(state.yourIndex):
            self.turn = int(state.turn)
            self.player_index = int(state.yourIndex)
            self.premium_power_pro_used = 0
        if self.iron_defender_active_turn < self.turn:
            self.iron_defender_owner = -1
            self.iron_defender_active_turn = -1
        for log in observation.logs or []:
            if (log.type == LogType.PLAY and log.cardId == IRON_DEFENDER
                    and log.playerIndex is not None):
                turn_player = (
                    state.firstPlayer if state.turn % 2 == 1
                    else 1 - state.firstPlayer
                )
                self.iron_defender_owner = int(log.playerIndex)
                # The log may arrive at our first selection of the following
                # turn. In that case Iron Defender already applies now.
                self.iron_defender_active_turn = int(state.turn) + int(
                    int(log.playerIndex) == turn_player
                )

    def with_extra_power_pro(self) -> "TurnEffects":
        return replace(self, premium_power_pro_used=self.premium_power_pro_used + 1)


@dataclass(frozen=True)
class DamageBreakdown:
    base: float = 0.0
    bonus_before_weakness: float = 0.0
    before_weakness_resistance: float = 0.0
    weakness_multiplier: float = 1.0
    resistance_reduction: float = 0.0
    after_weakness_resistance: float = 0.0
    reduction_after_weakness_resistance: float = 0.0
    prevented_by_pokemon: bool = False
    prevented_by_stadium: bool = False
    final: float = 0.0
    hp_margin: float = 0.0
    ko: bool = False
    known: bool = False
    attacker_rule_box: bool = False
    defender_rule_box: bool = False
    has_secondary_effect: bool = False
    has_energy_acceleration: bool = False

    @property
    def prevented(self) -> bool:
        return self.prevented_by_pokemon or self.prevented_by_stadium


def has_rule_box(card_id: int) -> bool:
    data = _CARD.get(card_id)
    return bool(data and (data.ex or data.megaEx))


def _pokemon_prevents_damage(attacker_id: int, defender_id: int) -> bool:
    attacker = _CARD.get(attacker_id)
    if attacker is None or not has_rule_box(attacker_id):
        return False
    if defender_id in {SYLVEON, CRUSTLE}:
        return True
    return defender_id == FARIGIRAF_EX and bool(attacker.basic)


def _stadium_prevents_damage(attacker_id: int, defender_id: int, stadium_id: int) -> bool:
    return (stadium_id == NEUTRALIZATION_ZONE
            and has_rule_box(attacker_id) and not has_rule_box(defender_id))


def calculate_attack_damage(attacker, defender, attack_id: int | None,
                            stadium_id: int = 0,
                            effects: TurnEffects | None = None,
                            defender_player_index: int = -1) -> DamageBreakdown:
    """Calculate deterministic active-to-active damage known from observation."""
    if attacker is None or defender is None or attack_id is None:
        return DamageBreakdown()
    attacker_data = _CARD.get(attacker.id)
    defender_data = _CARD.get(defender.id)
    attack = _ATTACK.get(attack_id)
    if attacker_data is None or defender_data is None or attack is None:
        return DamageBreakdown()

    effect_text = attack.text or ""
    lower_text = effect_text.lower()
    bonus = 0.0
    if effects is not None and attacker_data.energyType == EnergyType.FIGHTING:
        bonus = 30.0 * effects.premium_power_pro_used
    before = float(attack.damage) + bonus
    weakness_multiplier = 2.0 if defender_data.weakness == attacker_data.energyType else 1.0
    after_weakness = before * weakness_multiplier
    resistance = 30.0 if defender_data.resistance == attacker_data.energyType else 0.0
    after_resistance = max(0.0, after_weakness - resistance)
    reduction_after = 0.0
    if (effects is not None
            and effects.iron_defender_active_turn == effects.turn
            and effects.iron_defender_owner == defender_player_index
            and defender_data.energyType == EnergyType.METAL):
        reduction_after = 30.0
    after_reduction = max(0.0, after_resistance - reduction_after)

    pokemon_prevented = _pokemon_prevents_damage(attacker.id, defender.id)
    stadium_prevented = _stadium_prevents_damage(attacker.id, defender.id, stadium_id)
    final = 0.0 if pokemon_prevented or stadium_prevented else after_reduction
    hp = float(defender.hp)
    return DamageBreakdown(
        base=float(attack.damage), bonus_before_weakness=bonus,
        before_weakness_resistance=before, weakness_multiplier=weakness_multiplier,
        resistance_reduction=resistance, after_weakness_resistance=after_resistance,
        reduction_after_weakness_resistance=reduction_after,
        prevented_by_pokemon=pokemon_prevented,
        prevented_by_stadium=stadium_prevented, final=final,
        hp_margin=final - hp, ko=bool(hp > 0 and final >= hp),
        known=not any(marker in lower_text for marker in _DYNAMIC_DAMAGE_MARKERS),
        attacker_rule_box=has_rule_box(attacker.id),
        defender_rule_box=has_rule_box(defender.id),
        has_secondary_effect=bool(effect_text.strip()),
        has_energy_acceleration=("attach" in lower_text and "energy" in lower_text),
    )


def selected_play_card_id(observation, action: list[int]) -> list[int]:
    if observation.select is None:
        return []
    hand = observation.current.players[observation.current.yourIndex].hand or []
    result = []
    for index in action:
        if not 0 <= int(index) < len(observation.select.option):
            continue
        option = observation.select.option[int(index)]
        if (option.type == OptionType.PLAY and option.index is not None
                and 0 <= int(option.index) < len(hand)):
            result.append(int(hand[int(option.index)].id))
    return result


def observe_action(effects: TurnEffects, observation, action: list[int]) -> None:
    effects.sync(observation)
    effects.premium_power_pro_used += selected_play_card_id(
        observation, action
    ).count(PREMIUM_POWER_PRO)
