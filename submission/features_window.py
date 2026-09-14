"""Feature v7 combining all previously evaluated Lucario feature groups."""

from __future__ import annotations

from cg.api import SelectContext

try:
    from . import features_draw_probability as draw
    from . import features_incoming as incoming
    from . import opponent_archetype_window as window_archetype
    from . import opponent_vocab_data
    from . import features_prize_risk as prize
except ImportError:  # flattened Kaggle submission layout
    import features_draw_probability as draw  # type: ignore[no-redef]
    import features_incoming as incoming  # type: ignore[no-redef]
    import opponent_archetype_window as window_archetype  # type: ignore[no-redef]
    import opponent_vocab_data  # type: ignore[no-redef]
    import features_prize_risk as prize  # type: ignore[no-redef]


FEATURE_VERSION = 8
OWN_CARDS = incoming.OWN_CARDS
OPP_POKE = list(opponent_vocab_data.OPP_POKE)
is_own_turn = incoming.is_own_turn

BASE_ARCHETYPE_NAMES = incoming.ARCHETYPE_STATE_NAMES
WINDOW_ARCHETYPE_NAMES = window_archetype.feature_names()
PRIZE_STATE_NAMES = prize.PRIZE_RISK_STATE_NAMES
PRIZE_OPTION_NAMES = prize.PRIZE_RISK_OPTION_NAMES
HAND_OPTION_NAMES = prize.HAND_PROJECTION_OPTION_NAMES
DRAW_OPTION_NAMES = [
    name
    for name in draw.DRAW_PROBABILITY_OPTION_NAMES
    if name != "opt_shuffles_my_hand"
]
HEAL_CONTEXT_NAME = f"ctx_{int(SelectContext.HEAL)}"
ADDITIONAL_OPP_POKE = [
    card_id for card_id in opponent_vocab_data.OPP_POKE
    if card_id not in incoming.OPP_POKE
]
ADDITIONAL_OPP_NAMES = (
    [f"opp_active_is_{card_id}" for card_id in ADDITIONAL_OPP_POKE]
    + [f"opp_bench_{card_id}_count" for card_id in ADDITIONAL_OPP_POKE]
)

STATE_NAMES = (
    incoming.STATE_NAMES[:-len(BASE_ARCHETYPE_NAMES)]
    + WINDOW_ARCHETYPE_NAMES
    + PRIZE_STATE_NAMES
)
OPTION_NAMES = (
    incoming.OPTION_NAMES
    + PRIZE_OPTION_NAMES
    + HAND_OPTION_NAMES
    + DRAW_OPTION_NAMES
    + ADDITIONAL_OPP_NAMES
    + [HEAL_CONTEXT_NAME]
)
FEATURE_NAMES = STATE_NAMES + OPTION_NAMES
N_STATE = len(STATE_NAMES)
N_OPTION = len(OPTION_NAMES)
N_FEATURES = len(FEATURE_NAMES)


def new_episode_state():
    return incoming.new_episode_state()


def build_features(observation, deck=None, runtime_state=None):
    return Features(observation, deck, runtime_state)


def update_episode_state(runtime_state, observation, action) -> None:
    incoming.update_episode_state(runtime_state, observation, action)


class Features:
    def __init__(self, obs, deck: list[int] | None = None, runtime_state=None):
        self.incoming = incoming.Features(obs, deck, runtime_state)
        self.prize = prize.Features(obs, deck, runtime_state)
        self.draw = draw.Features(obs, deck, runtime_state)
        self.window_archetype = window_archetype.infer(obs)
        self.is_heal_context = float(obs.select.context == SelectContext.HEAL)
        self.additional_opp = self._additional_opponent_values(obs)
        self.state = (
            self.incoming.state[:-len(BASE_ARCHETYPE_NAMES)]
            + window_archetype.feature_values(self.window_archetype)
            + self.prize.state[-len(PRIZE_STATE_NAMES):]
        )

    @staticmethod
    def _additional_opponent_values(obs) -> list[float]:
        active = [0.0] * len(ADDITIONAL_OPP_POKE)
        bench = [0.0] * len(ADDITIONAL_OPP_POKE)
        index = {card_id: offset for offset, card_id in enumerate(ADDITIONAL_OPP_POKE)}
        current = getattr(obs, "current", None)
        if current is None:
            return active + bench
        opponent = current.players[1 - current.yourIndex]
        if opponent.active and opponent.active[0] is not None:
            offset = index.get(opponent.active[0].id)
            if offset is not None:
                active[offset] = 1.0
        for pokemon in opponent.bench:
            if pokemon is None:
                continue
            offset = index.get(pokemon.id)
            if offset is not None:
                bench[offset] += 1.0
        return active + bench

    def option(self, index: int) -> list[float]:
        prize_extra_count = len(PRIZE_OPTION_NAMES) + len(HAND_OPTION_NAMES)
        prize_extra = self.prize.option(index)[-prize_extra_count:]
        draw_extra = self.draw.option(index)[-len(draw.DRAW_PROBABILITY_OPTION_NAMES):]
        draw_without_duplicate_shuffle = draw_extra[:1] + draw_extra[2:]
        return (
            self.incoming.option(index)
            + prize_extra
            + draw_without_duplicate_shuffle
            + self.additional_opp
            + [self.is_heal_context]
        )