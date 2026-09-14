"""Feature v5 with prize-loss risk for Active and switch candidates."""
from __future__ import annotations

from cg.api import AreaType, OptionType, SelectContext

try:
    from . import features_incoming_only as base
    from . import incoming_damage
except ImportError:  # flattened Kaggle submission layout
    import features_incoming_only as base  # type: ignore[no-redef]
    import incoming_damage  # type: ignore[no-redef]

FEATURE_VERSION = 5
OWN_CARDS = base.OWN_CARDS
OPP_POKE = base.OPP_POKE
is_own_turn = base.is_own_turn

LEGACY_ENERGY = 12
LILLIES_PEARL = 1172
SWITCH = 1123
JUDGE = 1213
LILLIES_DETERMINATION = 1227

PRIZE_RISK_STATE_NAMES = [
    "me_active_prize_value",
    "opp_prizes_after_active_ko",
    "active_ko_loses_game",
    "incoming_now_ko_and_loses_game",
    "incoming_plus_energy_ko_and_loses_game",
    "incoming_evolution_ko_and_loses_game",
    "incoming_evolution_plus_energy_ko_and_loses_game",
]
PRIZE_RISK_OPTION_NAMES = [
    "switch_risk_candidate",
    "switch_risk_start_action",
    "switch_target_prize_value",
    "switch_opp_prizes_after_ko",
    "switch_target_ko_loses_game",
    "switch_incoming_now_damage",
    "switch_incoming_now_margin",
    "switch_incoming_now_ko",
    "switch_incoming_now_ko_and_loses_game",
    "switch_incoming_evolution_plus_energy_damage",
    "switch_incoming_evolution_plus_energy_margin",
    "switch_incoming_evolution_plus_energy_ko",
    "switch_incoming_evolution_plus_energy_ko_and_loses_game",
]
HAND_PROJECTION_OPTION_NAMES = [
    "opt_shuffles_my_hand",
    "opt_shuffles_opp_hand",
    "opt_my_hand_after",
    "opt_opp_hand_after",
    "opt_my_hand_delta",
    "opt_opp_hand_delta",
    "opt_hand_gap_after",
] + [
    f"opt_incoming_{scenario}_{metric}_after"
    for scenario in ("now", "plus_energy", "evolution", "evolution_plus_energy")
    for metric in ("damage", "margin", "ko", "known")
]

STATE_NAMES = base.STATE_NAMES + PRIZE_RISK_STATE_NAMES
OPTION_NAMES = base.OPTION_NAMES + PRIZE_RISK_OPTION_NAMES + HAND_PROJECTION_OPTION_NAMES
FEATURE_NAMES = STATE_NAMES + OPTION_NAMES
N_STATE = len(STATE_NAMES)
N_OPTION = len(OPTION_NAMES)
N_FEATURES = len(FEATURE_NAMES)


def new_episode_state():
    return base.new_episode_state()


def build_features(observation, deck=None, runtime_state=None):
    return Features(observation, deck, runtime_state)


def update_episode_state(runtime_state, observation, action) -> None:
    base.update_episode_state(runtime_state, observation, action)


def prize_value(pokemon) -> int:
    data = incoming_damage._CARD.get(int(pokemon.id))
    value = 3 if data is not None and data.megaEx else 2 if data is not None and data.ex else 1
    if any(int(card.id) == LEGACY_ENERGY for card in getattr(pokemon, "energyCards", ())):
        value -= 1
    if data is not None and "Lillie" in data.name and any(
        int(card.id) == LILLIES_PEARL for card in getattr(pokemon, "tools", ())
    ):
        value -= 1
    return max(0, value)


def project_hands(card_id: int | None, my_hand: int, opp_hand: int,
                  my_prizes: int) -> tuple[bool, bool, int, int]:
    if card_id == JUDGE:
        return True, True, 4, 4
    if card_id == LILLIES_DETERMINATION:
        return True, False, 8 if my_prizes == 6 else 6, opp_hand
    return False, False, my_hand, opp_hand


class Features:
    def __init__(self, obs, deck: list[int] | None = None, runtime_state=None):
        self.obs = obs
        self.base = base.Features(obs, deck, runtime_state)
        self.effects = self.base.base.effects
        state = obs.current
        self.mine = int(state.yourIndex)
        self.me = state.players[self.mine]
        self.opp = state.players[1 - self.mine]
        self.opponent_prizes = len(self.opp.prize)
        self.active = self.me.active[0] if self.me.active else None
        self._risk_cache = {}
        self._hand_threat_cache = {}
        self.state = self.base.state + self._state_features()

    def _risk(self, pokemon):
        key = int(getattr(pokemon, "serial", id(pokemon)))
        if key in self._risk_cache:
            return self._risk_cache[key]
        value = prize_value(pokemon)
        remaining = max(0, self.opponent_prizes - value)
        loses = self.opponent_prizes <= value
        threats = (
            self.base.incoming
            if pokemon is self.active
            else incoming_damage.evaluate(self.obs, self.effects, defender=pokemon)
        )
        result = value, remaining, loses, threats
        self._risk_cache[key] = result
        return result

    def _state_features(self) -> list[float]:
        if self.active is None:
            return [0.0] * len(PRIZE_RISK_STATE_NAMES)
        value, remaining, loses, threats = self._risk(self.active)
        return [
            float(value),
            float(remaining),
            float(loses),
            float(loses and threats.now.ko),
            float(loses and threats.plus_energy.ko),
            float(loses and threats.evolution.ko),
            float(loses and threats.evolution_plus_energy.ko),
        ]

    def _card_at_bench(self, index):
        if index is None or index < 0 or index >= len(self.me.bench):
            return None
        return self.me.bench[index]

    def _switch_target(self, option):
        if self.obs.select.context != SelectContext.SWITCH:
            return None
        area = option.inPlayArea if option.inPlayArea is not None else option.area
        index = option.inPlayIndex if option.inPlayIndex is not None else option.index
        return self._card_at_bench(index) if area == AreaType.BENCH else None

    def _is_switch_start(self, option) -> bool:
        if option.type == OptionType.RETREAT:
            return True
        if option.type != OptionType.PLAY or option.index is None:
            return False
        hand = self.me.hand or []
        return 0 <= option.index < len(hand) and int(hand[option.index].id) == SWITCH

    def _best_bench(self):
        if not self.me.bench:
            return None
        risks = [(pokemon, self._risk(pokemon)) for pokemon in self.me.bench]
        return min(
            risks,
            key=lambda item: (
                item[1][2] and item[1][3].evolution_plus_energy.ko,
                item[1][3].evolution_plus_energy.ko,
                item[1][3].evolution_plus_energy.margin,
                item[1][0],
            ),
        )[0]

    @staticmethod
    def _hand_count(player) -> int:
        count = getattr(player, "handCount", None)
        return int(count) if count is not None else len(getattr(player, "hand", ()) or ())

    def _played_card_id(self, option) -> int | None:
        if (self.obs.select.context != SelectContext.MAIN
                or option.type != OptionType.PLAY or option.index is None):
            return None
        hand = self.me.hand or []
        if option.index < 0 or option.index >= len(hand):
            return None
        return int(hand[option.index].id)

    def _hand_projection(self, option) -> list[float]:
        my_hand = self._hand_count(self.me)
        opp_hand = self._hand_count(self.opp)
        shuffles_me, shuffles_opp, my_after, opp_after = project_hands(
            self._played_card_id(option), my_hand, opp_hand, len(self.me.prize)
        )
        key = my_after, opp_after
        threats = self._hand_threat_cache.get(key)
        if threats is None:
            threats = (
                self.base.incoming
                if key == (my_hand, opp_hand)
                else incoming_damage.evaluate(
                    self.obs,
                    self.effects,
                    attacker_hand_count=opp_after,
                    defender_hand_count=my_after,
                )
            )
            self._hand_threat_cache[key] = threats
        result = [
            float(shuffles_me),
            float(shuffles_opp),
            float(my_after),
            float(opp_after),
            float(my_after - my_hand),
            float(opp_after - opp_hand),
            float(my_after - opp_after),
        ]
        for scenario in ("now", "plus_energy", "evolution", "evolution_plus_energy"):
            threat = getattr(threats, scenario)
            result.extend([
                threat.damage,
                threat.margin,
                float(threat.ko),
                float(threat.known),
            ])
        return result

    def option(self, index: int) -> list[float]:
        option = self.obs.select.option[index]
        target = self._switch_target(option)
        start = self._is_switch_start(option)
        if target is None and start:
            target = self._best_bench()
        if target is None:
            extra = [0.0] * len(PRIZE_RISK_OPTION_NAMES)
        else:
            value, remaining, loses, threats = self._risk(target)
            now = threats.now
            future = threats.evolution_plus_energy
            extra = [
                float(not start),
                float(start),
                float(value),
                float(remaining),
                float(loses),
                now.damage,
                now.margin,
                float(now.ko),
                float(loses and now.ko),
                future.damage,
                future.margin,
                float(future.ko),
                float(loses and future.ko),
            ]
        return self.base.option(index) + extra + self._hand_projection(option)