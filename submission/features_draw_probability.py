"""Feature v6 with per-card probabilities for draw actions."""
from __future__ import annotations

from math import comb

from cg.api import AreaType, OptionType

try:
    from . import features_incoming_only as base
except ImportError:  # flattened Kaggle submission layout
    import features_incoming_only as base  # type: ignore[no-redef]


FEATURE_VERSION = 6
OWN_CARDS = base.OWN_CARDS
OPP_POKE = base.OPP_POKE
is_own_turn = base.is_own_turn

LUNATONE = 675
JUDGE = 1213
LILLIES_DETERMINATION = 1227

DRAW_PROBABILITY_OPTION_NAMES = [
    "opt_draw_count",
    "opt_shuffles_my_hand",
] + [f"opt_draw_prob_{card_id}" for card_id in OWN_CARDS]

STATE_NAMES = base.STATE_NAMES
OPTION_NAMES = base.OPTION_NAMES + DRAW_PROBABILITY_OPTION_NAMES
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


def draw_probability(
    deck_count: int,
    prize_count: int,
    hidden_target_count: int,
    draw_count: int,
    returned_count: int = 0,
    returned_target_count: int = 0,
) -> float:
    """Probability of drawing a target without observing prize identities."""
    deck_count = max(0, int(deck_count))
    prize_count = max(0, int(prize_count))
    hidden_target_count = max(0, int(hidden_target_count))
    returned_count = max(0, int(returned_count))
    returned_target_count = max(0, int(returned_target_count))
    final_deck_count = deck_count + returned_count
    draws = min(max(0, int(draw_count)), final_deck_count)
    if draws == 0:
        return 0.0

    hidden_count = deck_count + prize_count
    hidden_target_count = min(hidden_target_count, hidden_count)
    denominator = comb(hidden_count, deck_count) * comb(final_deck_count, draws)
    if denominator == 0:
        return 0.0

    no_target_weight = 0
    minimum = max(0, deck_count - (hidden_count - hidden_target_count))
    maximum = min(hidden_target_count, deck_count)
    for target_in_deck in range(minimum, maximum + 1):
        allocation_weight = (
            comb(hidden_target_count, target_in_deck)
            * comb(hidden_count - hidden_target_count, deck_count - target_in_deck)
        )
        non_targets = final_deck_count - target_in_deck - returned_target_count
        if non_targets >= draws:
            no_target_weight += allocation_weight * comb(non_targets, draws)
    return 1.0 - no_target_weight / denominator


def draw_action(card_id: int | None, option_type, prize_count: int) -> tuple[int, bool]:
    if option_type == OptionType.ABILITY and card_id == LUNATONE:
        return 3, False
    if option_type == OptionType.PLAY and card_id == JUDGE:
        return 4, True
    if option_type == OptionType.PLAY and card_id == LILLIES_DETERMINATION:
        return (8 if prize_count == 6 else 6), True
    return 0, False


class Features:
    def __init__(self, obs, deck: list[int] | None = None, runtime_state=None):
        self.obs = obs
        self.deck = deck or []
        self.base = base.Features(obs, deck, runtime_state)
        state = obs.current
        self.mine = int(state.yourIndex)
        self.me = state.players[self.mine]
        self.state = self.base.state
        self._hand_counts, self._hidden_counts = self._zone_counts()

    def _zone_counts(self) -> tuple[dict[int, int], dict[int, int]]:
        hand = {card_id: 0 for card_id in OWN_CARDS}
        discard = {card_id: 0 for card_id in OWN_CARDS}
        board = {card_id: 0 for card_id in OWN_CARDS}
        total = {card_id: 0 for card_id in OWN_CARDS}
        for card in self.me.hand or ():
            if int(card.id) in hand:
                hand[int(card.id)] += 1
        for card in self.me.discard:
            if int(card.id) in discard:
                discard[int(card.id)] += 1
        for pokemon in list(self.me.active) + list(self.me.bench):
            if pokemon is None:
                continue
            cards = [pokemon] + list(pokemon.tools) + list(pokemon.energyCards) \
                + list(pokemon.preEvolution)
            for card in cards:
                if int(card.id) in board:
                    board[int(card.id)] += 1
        for card_id in self.deck:
            if int(card_id) in total:
                total[int(card_id)] += 1
        hidden = {
            card_id: max(0, total[card_id] - hand[card_id]
                         - discard[card_id] - board[card_id])
            for card_id in OWN_CARDS
        }
        return hand, hidden

    def _source_card(self, option):
        if option.index is None or option.index < 0:
            return None
        if option.type == OptionType.PLAY:
            cards = self.me.hand or ()
        elif option.type == OptionType.ABILITY and option.area == AreaType.ACTIVE:
            cards = self.me.active
        elif option.type == OptionType.ABILITY and option.area == AreaType.BENCH:
            cards = self.me.bench
        else:
            return None
        return cards[option.index] if option.index < len(cards) else None

    def _draw_action(self, option) -> tuple[int, bool, int | None]:
        source = self._source_card(option)
        card_id = int(source.id) if source is not None else None
        draw_count, shuffles = draw_action(card_id, option.type, len(self.me.prize))
        return draw_count, shuffles, card_id

    def _draw_features(self, option) -> list[float]:
        draw_count, shuffles, played_card_id = self._draw_action(option)
        if draw_count == 0:
            return [0.0] * len(DRAW_PROBABILITY_OPTION_NAMES)

        returned_count = max(0, len(self.me.hand or ()) - 1) if shuffles else 0
        probabilities = []
        for card_id in OWN_CARDS:
            returned_target = self._hand_counts[card_id]
            if shuffles and card_id == played_card_id:
                returned_target -= 1
            probabilities.append(draw_probability(
                self.me.deckCount,
                len(self.me.prize),
                self._hidden_counts[card_id],
                draw_count,
                returned_count,
                max(0, returned_target) if shuffles else 0,
            ))
        return [float(draw_count), float(shuffles), *probabilities]

    def option(self, index: int) -> list[float]:
        option = self.obs.select.option[index]
        return self.base.option(index) + self._draw_features(option)