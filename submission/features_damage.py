"""Experimental v3 features with deterministic damage projections.

The adopted v2 module remains untouched. This module appends features to v2 so
it can be trained and evaluated without changing the current production model.
"""
from __future__ import annotations

from cg.api import OptionType

try:
    from . import damage
    from . import features as base
except ImportError:  # flattened Kaggle submission layout
    import damage  # type: ignore[no-redef]
    try:
        import features as base  # type: ignore[no-redef]
    except ImportError:
        import features_base as base  # type: ignore[no-redef]

FEATURE_VERSION = 3
OWN_CARDS = base.OWN_CARDS
OPP_POKE = base.OPP_POKE
is_own_turn = base.is_own_turn
_STADIUM_INDEX = {card_id: index for index, card_id in enumerate(damage.STADIUM_IDS)}

DAMAGE_STATE_NAMES = (
    [f"stadium_card_{card_id}" for card_id in damage.STADIUM_IDS]
    + ["stadium_card_other", "turn_premium_power_pro_used",
       "turn_damage_bonus_before_weakness",
       "turn_iron_defender_applies_to_opponent"]
)
DAMAGE_OPTION_NAMES = [
    "damage_base", "damage_bonus_before_weakness",
    "damage_before_weakness_resistance", "damage_weakness_multiplier",
    "damage_resistance_reduction", "damage_after_weakness_resistance",
    "damage_reduction_after_weakness_resistance",
    "damage_prevented_by_pokemon", "damage_prevented_by_stadium",
    "damage_final", "damage_hp_margin", "damage_ko",
    "damage_calculation_known", "damage_attacker_has_rule_box",
    "damage_defender_has_rule_box", "attack_has_secondary_effect",
    "attack_has_energy_acceleration", "projected_best_damage_after_action",
    "projected_damage_gain", "projected_best_hp_margin",
    "projected_ko_after_action", "projected_damage_known",
    "projected_action_is_damage_modifier",
]
STATE_NAMES = base.STATE_NAMES + DAMAGE_STATE_NAMES
OPTION_NAMES = base.OPTION_NAMES + DAMAGE_OPTION_NAMES
FEATURE_NAMES = STATE_NAMES + OPTION_NAMES
N_STATE = len(STATE_NAMES)
N_OPTION = len(OPTION_NAMES)
N_FEATURES = len(FEATURE_NAMES)


def new_episode_state() -> damage.TurnEffects:
    return damage.TurnEffects()


def build_features(observation, deck=None, runtime_state=None):
    return Features(observation, deck, runtime_state)


def update_episode_state(runtime_state, observation, action) -> None:
    damage.observe_action(runtime_state, observation, action)


class Features:
    def __init__(self, obs, deck: list[int] | None = None, runtime_state=None):
        self.obs = obs
        self.effects = runtime_state or damage.TurnEffects()
        self.effects.sync(obs)
        self.base = base.Features(obs, deck)
        self.state = self.base.state + self._damage_state()
        state = obs.current
        mine = state.yourIndex
        self.me = state.players[mine]
        self.opp = state.players[1 - mine]
        self.attacker = self.me.active[0] if self.me.active else None
        self.defender = self.opp.active[0] if self.opp.active else None
        self.stadium_id = state.stadium[0].id if state.stadium else 0
        self._current_attacks = self._attack_breakdowns(self.effects)
        self._current_best = max((item.final for item in self._current_attacks), default=0.0)

    def _damage_state(self) -> list[float]:
        state = self.obs.current
        stadium_id = state.stadium[0].id if state.stadium else 0
        hot = [0.0] * (len(damage.STADIUM_IDS) + 1)
        if stadium_id:
            hot[_STADIUM_INDEX.get(stadium_id, len(damage.STADIUM_IDS))] = 1.0
        opponent = 1 - state.yourIndex
        return hot + [
            float(self.effects.premium_power_pro_used),
            float(30 * self.effects.premium_power_pro_used),
            float(self.effects.iron_defender_active_turn == state.turn
                  and self.effects.iron_defender_owner == opponent),
        ]

    def _attack_breakdowns(self, effects):
        if self.attacker is None or self.defender is None:
            return []
        return [self._calculate(option.attackId, effects)
                for option in self.obs.select.option
                if option.type == OptionType.ATTACK]

    def _calculate(self, attack_id, effects):
        return damage.calculate_attack_damage(
            self.attacker, self.defender, attack_id, self.stadium_id, effects,
            defender_player_index=1 - self.obs.current.yourIndex,
        )

    def _source_card_id(self, option) -> int:
        if option.type != OptionType.PLAY or option.index is None:
            return -1
        hand = self.me.hand or []
        return hand[option.index].id if 0 <= option.index < len(hand) else -1

    @staticmethod
    def _breakdown_vector(item: damage.DamageBreakdown) -> list[float]:
        return [
            item.base, item.bonus_before_weakness,
            item.before_weakness_resistance, item.weakness_multiplier,
            item.resistance_reduction, item.after_weakness_resistance,
            item.reduction_after_weakness_resistance,
            float(item.prevented_by_pokemon), float(item.prevented_by_stadium),
            item.final, item.hp_margin, float(item.ko), float(item.known),
            float(item.attacker_rule_box), float(item.defender_rule_box),
            float(item.has_secondary_effect), float(item.has_energy_acceleration),
        ]

    def option(self, index: int) -> list[float]:
        option = self.obs.select.option[index]
        item = (self._calculate(option.attackId, self.effects)
                if option.type == OptionType.ATTACK else damage.DamageBreakdown())
        modifier = self._source_card_id(option) == damage.PREMIUM_POWER_PRO
        projected_effects = self.effects.with_extra_power_pro() if modifier else self.effects
        projected_attacks = self._attack_breakdowns(projected_effects)
        projected_best = max((attack.final for attack in projected_attacks),
                             default=self._current_best)
        hp = float(self.defender.hp) if self.defender is not None else 0.0
        projection_known = bool(self.defender is not None and projected_attacks
                                and (modifier or option.type == OptionType.ATTACK))
        projection = [
            projected_best, projected_best - self._current_best,
            projected_best - hp, float(hp > 0 and projected_best >= hp),
            float(projection_known), float(modifier),
        ]
        return self.base.option(index) + self._breakdown_vector(item) + projection
