"""Third-generation Mega Lucario rule policy.

The deck and the new search/disruption rules are replay-derived from leading
teams that used the same 60 cards on 2026-08-04 through 2026-08-08.
"""

from __future__ import annotations

import os
from collections import defaultdict

from cg.api import (
    AreaType,
    Card,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    to_observation_class,
)


class C:
    KYOGRE = 721
    SNOVER = 722
    MEGA_ABOMASNOW_EX = 723

    MAKUHITA = 673
    HARIYAMA = 674
    LUNATONE = 675
    SOLROCK = 676
    RIOLU = 677
    MEGA_LUCARIO_EX = 678

    BASIC_FIGHTING_ENERGY = 6
    ULTRA_BALL = 1121
    SWITCH = 1123
    PREMIUM_POWER_PRO = 1141
    FIGHTING_GONG = 1142
    POKE_PAD = 1152
    HERO_CAPE = 1159
    BOSS_ORDERS = 1182
    JUDGE = 1213
    LILLIE_DETERMINATION = 1227
    WALLYS_COMPASSION = 1229

    LUMIOSE_CITY = 1267
    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


MEGA_BRAVE = 983
LOW_DECK_COUNT = 10
LUCARIO_HEAL_HP_THRESHOLD = 270
EARLY_SETUP_LAST_TURN = 4


_DECK_CANDIDATES = []
if "__file__" in globals():
    _DECK_CANDIDATES.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
    )
_DECK_CANDIDATES += ["deck.csv", "/kaggle_simulations/agent/deck.csv"]
DECK_PATH = next((p for p in _DECK_CANDIDATES if os.path.exists(p)), "deck.csv")
with open(DECK_PATH, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]
# Public alias so the battle harness / deck registry can load this deck list.
DECK = my_deck


all_card = all_card_data()
card_table = {card.cardId: card for card in all_card}


class AttackPlan:
    def __init__(
        self,
        attacker: int = -1,
        target: int = -1,
        attack_index: int = -1,
        remain_hp: int = -1,
        needs_energy: bool = False,
    ):
        self.attacker = attacker
        self.target = target
        self.attack_index = attack_index
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy


plan = AttackPlan()
pre_turn = -1
ability_used = False
power_pro_used = 0


def get_card(
    obs: Observation, area: AreaType, index: int, player_index: int
) -> Pokemon | Card | None:
    player = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return player.hand[index]
        case AreaType.DISCARD:
            return player.discard[index]
        case AreaType.ACTIVE:
            return player.active[index]
        case AreaType.BENCH:
            return player.bench[index]
        case AreaType.PRIZE:
            return player.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def prize_count(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == C.LEGACY_ENERGY:
            count -= 1
    for card in pokemon.tools:
        if card.id == C.LILLIES_PEARL and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def target_score(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130

    if pokemon.id in {144, 322, 323, 337}:  # low-value support Pokemon
        score -= 200
    if pokemon.id == C.SNOVER:
        score += 950
    elif pokemon.id == C.MEGA_ABOMASNOW_EX:
        score += 250
    if pokemon.id == C.RIOLU:
        score += 800
    elif pokemon.id == C.MEGA_LUCARIO_EX:
        score += 100
    if pokemon.id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score


class LucarioPolicy:
    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]
        self.my_prizes_left = len(self.me.prize)

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.has_ready_lucario_line = False
        self.has_ready_hariyama_line = False
        self.can_switch = False
        self.can_gust = False
        self.can_attack = False
        self.can_use_mega_brave = False
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0

        self._count_cards()
        self._scan_main_options()

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []

        if self.context == SelectContext.MAIN:
            self._plan_attack()

        scores = [self._score_option(option) for option in self.select.option]
        ranked = [
            i
            for i, _ in sorted(
                enumerate(scores), key=lambda item: item[1], reverse=True
            )
        ]
        self._remember_lunatone_ability(ranked)
        self._remember_power_pro(ranked)
        return ranked[: self.select.maxCount]

    def _count_cards(self) -> None:
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
            if pokemon.id in {C.MAKUHITA, C.HARIYAMA} and len(pokemon.energies) >= 3:
                self.has_ready_hariyama_line = True
            if (
                pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX}
                and len(pokemon.energies) >= 2
            ):
                self.has_ready_lucario_line = True

        for card in self.me.hand:
            self.hand_counts[card.id] += 1
        for card in self.me.discard:
            self.discard_counts[card.id] += 1

    def _scan_main_options(self) -> None:
        if self.context != SelectContext.MAIN:
            return
        for option in self.select.option:
            if option.type == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.SWITCH:
                    self.can_switch = True
                elif card.id == C.BOSS_ORDERS:
                    self.can_gust = True
            elif option.type == OptionType.EVOLVE:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.HARIYAMA:
                    self.can_gust = True
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True
                if option.attackId == MEGA_BRAVE:
                    self.can_use_mega_brave = True

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    def _opponent_has(self, ids: set[int]) -> bool:
        return any(
            pokemon is not None and pokemon.id in ids
            for pokemon in self._opponent_board()
        )

    def _opponent_is_water_deck(self) -> bool:
        return self._opponent_has({C.KYOGRE, C.SNOVER, C.MEGA_ABOMASNOW_EX})

    def _opponent_is_crustle_wall(self) -> bool:
        return self._opponent_has({344, 345})

    def _active_damaged_lucario(self) -> Pokemon | None:
        if not self.me.active:
            return None
        active = self.me.active[0]
        if (
            active is not None
            and active.id == C.MEGA_LUCARIO_EX
            and active.hp <= LUCARIO_HEAL_HP_THRESHOLD
        ):
            return active
        return None

    def _is_early_setup_phase(self) -> bool:
        return self.state.turn <= EARLY_SETUP_LAST_TURN

    def _planned_bench_lucario_needs_wally(self) -> bool:
        if (
            self.state.supporterPlayed
            or self.hand_counts[C.WALLYS_COMPASSION] == 0
            or plan.attacker < 1
        ):
            return False
        board = self._my_board()
        if plan.attacker >= len(board):
            return False
        attacker = board[plan.attacker]
        return (
            attacker is not None
            and attacker.id == C.MEGA_LUCARIO_EX
            and attacker.hp <= LUCARIO_HEAL_HP_THRESHOLD
        )

    def _can_evolve_board_index(self, board_index: int) -> bool:
        for option in self.select.option:
            if option.type != OptionType.EVOLVE:
                continue
            target_index = option.inPlayIndex
            if option.inPlayArea == AreaType.BENCH:
                target_index += 1
            if target_index == board_index:
                return True
        return False

    def _base_attack(
        self, pokemon: Pokemon, attack_index: int
    ) -> tuple[int, int, int] | None:
        energy_required = 0
        base_damage = 0
        base_score = 0

        if pokemon.id == C.MEGA_LUCARIO_EX:
            if attack_index == 0:
                energy_required = 1
                base_damage = 130
                base_score += 60 * min(3, self.discard_counts[C.BASIC_FIGHTING_ENERGY])
            else:
                energy_required = 2
                base_damage = 270
            if self._opponent_is_water_deck() and len(self.opponent.prize) <= 3:
                base_score -= 500
        elif attack_index == 1:
            return None
        elif pokemon.id == C.HARIYAMA:
            energy_required = 3
            base_damage = 210
        elif pokemon.id == C.MAKUHITA:
            return None
        elif pokemon.id == C.SOLROCK and self.field_counts[C.LUNATONE] >= 1:
            energy_required = 1
            base_damage = 70

        if base_damage <= 0:
            return None
        return energy_required, base_damage, base_score

    def _base_attack_after_evolution(
        self, pokemon: Pokemon, board_index: int, attack_index: int
    ):
        if (
            pokemon.id == C.MAKUHITA
            and attack_index == 0
            and self._can_evolve_board_index(board_index)
        ):
            return 3, 210, -100
        return self._base_attack(pokemon, attack_index)

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()

        if self.state.turn < 2:
            return

        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break

            for attack_index in range(2):
                attack = self._base_attack_after_evolution(
                    my_pokemon, attacker_index, attack_index
                )
                if attack is None:
                    continue
                energy_required, base_damage, base_score = attack

                energy_count = len(my_pokemon.energies)
                if (
                    attack_index == 1
                    and attacker_index == 0
                    and energy_count >= 2
                    and not self.can_use_mega_brave
                ):
                    break

                needs_energy = False
                if energy_count < energy_required:
                    if (
                        self.hand_counts[C.BASIC_FIGHTING_ENERGY] >= 1
                        and not self.state.energyAttached
                    ):
                        energy_count += 1
                        needs_energy = energy_count >= energy_required
                    if not needs_energy:
                        continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break
                    if (
                        self._opponent_is_crustle_wall()
                        and my_pokemon.id == C.MEGA_LUCARIO_EX
                        and op_pokemon.id == 345
                    ):
                        continue

                    damage = base_damage + 30 * power_pro_used
                    op_data = card_table[op_pokemon.id]
                    if op_data.weakness == EnergyType.FIGHTING:
                        damage *= 2
                    elif op_data.resistance == EnergyType.FIGHTING:
                        damage -= 30

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / op_pokemon.hp
                    if len(self.opponent.prize) <= prize:
                        score = 50000

                    score += base_score
                    score += 220 if attacker_index == 0 else 0
                    score += 300 if target_index == 0 else 0
                    score += energy_count

                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(
                            attacker=attacker_index,
                            target=target_index,
                            attack_index=attack_index,
                            remain_hp=op_pokemon.hp - damage,
                            needs_energy=needs_energy,
                        )

    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)

        if pokemon.id in {C.MAKUHITA, C.HARIYAMA}:
            score += 1 if pokemon.id == C.HARIYAMA else 0
            if self._opponent_is_crustle_wall():
                score += 260 if energy_count < 3 else 30
            else:
                score += 100 if energy_count < 3 else 0
                score -= 50 if self.has_ready_hariyama_line else 0
        elif pokemon.id == C.LUNATONE:
            score -= 100
        elif pokemon.id == C.SOLROCK:
            score += 20 if energy_count < 1 else -100
        elif pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX}:
            score += 1 if pokemon.id == C.MEGA_LUCARIO_EX else 0
            score += 220 if energy_count < 2 else 0
            score -= 50 if self.has_ready_lucario_line else 0
        return score

    def _score_option(self, option) -> float:
        if option.type == OptionType.NUMBER:
            return option.number
        if option.type == OptionType.YES:
            return 100 if self.context == SelectContext.IS_FIRST else 1
        if option.type == OptionType.NO:
            return 0
        if option.type == OptionType.CARD:
            return self._score_card_choice(option)
        if option.type == OptionType.PLAY:
            return self._score_play(option)
        if option.type == OptionType.ATTACH:
            return self._score_attach(option)
        if option.type == OptionType.EVOLVE:
            return self._score_evolve(option)
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            return 2000 if plan.attacker >= 1 else -1
        if option.type == OptionType.ATTACK:
            if (
                self._opponent_is_crustle_wall()
                and self.me.active
                and self.opponent.active
                and self.me.active[0].id == C.MEGA_LUCARIO_EX
                and self.opponent.active[0].id == 345
                and plan.target < 0
            ):
                return -1
            return (
                1100
                if (option.attackId == MEGA_BRAVE) == (plan.attack_index == 1)
                else 1000
            )
        return 0

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0

        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._score_setup_active(card)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context == SelectContext.DISCARD:
            return self._score_discard(card)
        if self.context == SelectContext.HEAL:
            return self._score_heal_target(option, card)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        return 0

    def _score_heal_target(self, option, card: Pokemon | Card) -> int:
        if not isinstance(card, Pokemon):
            return -1
        if option.playerIndex != self.my_index or option.area != AreaType.ACTIVE:
            return -1
        if (
            card.id == C.MEGA_LUCARIO_EX
            and card.hp <= LUCARIO_HEAL_HP_THRESHOLD
        ):
            return 100
        return -1

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0

        if option.playerIndex != self.my_index:
            return 100 if option.index == plan.target - 1 else 0

        score = len(card.energies) * 2
        if option.index == plan.attacker - 1:
            score += 100
        if card.id == C.MEGA_LUCARIO_EX:
            score += (
                8
                if self._opponent_is_water_deck() and len(self.opponent.prize) <= 3
                else 20
            )
        elif card.id == C.HARIYAMA and len(card.energies) >= 2:
            score += 45 if self._opponent_is_crustle_wall() else 15
        elif card.id == C.MAKUHITA and len(card.energies) >= 2:
            score += 35 if self._opponent_is_crustle_wall() else 10
        elif card.id == C.SOLROCK:
            score += 5
        elif card.id == C.RIOLU:
            score += 4
        return score

    def _score_setup_active(self, card: Pokemon | Card) -> int:
        if card.id == C.SOLROCK:
            return 2 if self.state.firstPlayer == self.my_index else 4
        if card.id == C.RIOLU:
            return 3
        if card.id == C.MAKUHITA:
            return 1
        return 0

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        effect_id = self._effect_id()
        if effect_id == C.ULTRA_BALL:
            return self._score_ultra_ball_target(card)
        if effect_id == C.POKE_PAD:
            return self._score_poke_pad_target(card)
        if effect_id == C.FIGHTING_GONG:
            return self._score_fighting_gong_target(card)

        score = 200 - self.hand_counts[card.id] * 100
        if card.id == C.MAKUHITA:
            if self._opponent_is_crustle_wall():
                score += 80 if self.field_counts[card.id] < 2 else -20
            else:
                score += -10 if self.field_counts[card.id] >= 1 else 10
        elif card.id == C.HARIYAMA:
            if self._opponent_is_crustle_wall():
                score += 120 if self.field_counts[C.MAKUHITA] >= 1 else -5
            else:
                score += 20 if self.field_counts[C.MAKUHITA] >= 1 else -20
        elif card.id == C.LUNATONE:
            score += -250 if self.field_counts[card.id] >= 1 else 60
        elif card.id == C.SOLROCK:
            score += -250 if self.field_counts[card.id] >= 1 else 50
        elif card.id == C.RIOLU:
            lucario_line = (
                self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX]
            )
            score += -150 if lucario_line >= 2 else -3 if lucario_line >= 1 else 40
        elif card.id == C.MEGA_LUCARIO_EX:
            score += 40 if self.field_counts[C.RIOLU] >= 1 else -15
        elif card.id == C.BASIC_FIGHTING_ENERGY:
            score += 30 if not ability_used or not self.state.energyAttached else -1
        return score

    def _effect_id(self) -> int | None:
        effect = self.select.effect or self.select.contextCard
        return None if effect is None else effect.id

    def _score_ultra_ball_target(self, card: Pokemon | Card) -> float:
        """Prioritize the search targets seen in top-50 winning replays."""
        lucario_line = (
            self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX]
        )
        opening = self.state.turn <= 2
        if card.id == C.MEGA_LUCARIO_EX:
            ready_to_evolve = self.field_counts[C.RIOLU] > 0
            return (1350 if ready_to_evolve and not opening else 500) \
                - 180 * self.hand_counts[card.id]
        if card.id == C.RIOLU:
            return (1450 if lucario_line == 0 else 750 if lucario_line == 1 else 250) \
                - 160 * self.hand_counts[card.id]
        if card.id == C.HARIYAMA:
            return 900 if self.field_counts[C.MAKUHITA] else 400
        if card.id == C.MAKUHITA:
            return 760 if self.field_counts[card.id] == 0 else 300
        if card.id == C.LUNATONE:
            return 1350 if self.field_counts[card.id] == 0 else 100
        if card.id == C.SOLROCK:
            return 1300 if self.field_counts[card.id] == 0 else 100
        return 0

    def _score_poke_pad_target(self, card: Pokemon | Card) -> float:
        lucario_line = (
            self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX]
        )
        if card.id == C.RIOLU:
            return (1400 if lucario_line == 0 else 1100 if lucario_line == 1 else 250) \
                - 160 * self.hand_counts[card.id]
        if card.id == C.HARIYAMA:
            return 1000 if self.field_counts[C.MAKUHITA] else 450
        if card.id == C.LUNATONE:
            return 1500 if self.field_counts[card.id] == 0 else 100
        if card.id == C.MAKUHITA:
            return 800 if self.field_counts[card.id] == 0 else 300
        if card.id == C.SOLROCK:
            return 1450 if self.field_counts[card.id] == 0 else 100
        return 0

    def _score_fighting_gong_target(self, card: Pokemon | Card) -> float:
        lucario_line = (
            self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX]
        )
        if card.id == C.RIOLU:
            return 1350 if lucario_line == 0 else 650 if lucario_line == 1 else 250
        if card.id == C.BASIC_FIGHTING_ENERGY:
            needs_energy = not ability_used or not self.state.energyAttached
            return 1250 if needs_energy and self.hand_counts[card.id] < 2 else 400
        if card.id == C.LUNATONE:
            return 1200 if self.field_counts[card.id] == 0 else 100
        if card.id == C.SOLROCK:
            return 1100 if self.field_counts[card.id] == 0 else 100
        if card.id == C.MAKUHITA:
            return 700 if self.field_counts[card.id] == 0 else 250
        return 0

    def _score_discard(self, card: Pokemon | Card) -> float:
        """Choose Ultra Ball costs without throwing away a live combo piece."""
        if self._effect_id() != C.ULTRA_BALL:
            return 0
        duplicates = self.hand_counts[card.id] - 1
        if card.id == C.HERO_CAPE:
            return -1000
        if card.id == C.BASIC_FIGHTING_ENERGY:
            return 600 if self.hand_counts[card.id] >= 3 else 180
        if card.id == C.JUDGE:
            opponent_hand = self.opponent.handCount or 0
            return 560 if duplicates > 0 or opponent_hand <= 4 else 120
        if card.id == C.ULTRA_BALL:
            return 520 if duplicates > 0 else 160
        if card.id in {C.SOLROCK, C.LUNATONE}:
            return 500 if self.field_counts[card.id] or duplicates > 0 else 120
        if card.id == C.HARIYAMA:
            return 460 if self.field_counts[C.MAKUHITA] == 0 else 80
        if card.id == C.PREMIUM_POWER_PRO:
            return 420 if duplicates > 0 or not self.can_attack else 60
        if card.id == C.LILLIE_DETERMINATION:
            return 400 if duplicates > 0 or (self.me.handCount or 0) >= 7 else 70
        if card.id == C.BOSS_ORDERS:
            return 380 if duplicates > 0 or plan.target < 1 else 40
        if card.id == C.SWITCH:
            return 360 if duplicates > 0 or plan.attacker <= 0 else 40
        if card.id == C.WALLYS_COMPASSION:
            return 340 if duplicates > 0 or self.field_counts[C.MEGA_LUCARIO_EX] == 0 else -200
        if card.id == C.MEGA_LUCARIO_EX:
            return 300 if duplicates > 0 or self.field_counts[C.RIOLU] == 0 else -100
        if card.id == C.MAKUHITA:
            return 260 if duplicates > 0 else 20
        if card.id == C.RIOLU:
            return 220 if duplicates > 0 else -150
        if card.id in {C.POKE_PAD, C.FIGHTING_GONG}:
            return 200 if duplicates > 0 else 30
        return 100

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card: Card) -> float:
        score = 20000
        if card.id in {C.LUNATONE, C.SOLROCK} and self.field_counts[card.id] >= 1:
            return -1
        if (
            card.id == C.RIOLU
            and self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2
        ):
            return -1
        return score

    def _score_play_trainer(self, card: Card) -> float:
        if card.id == C.SWITCH:
            return 6000 if plan.attacker > 0 else -1
        if card.id == C.WALLYS_COMPASSION:
            if self._active_damaged_lucario() is None:
                return -1
            return 6500 if self._is_early_setup_phase() else 40000
        if card.id == C.PREMIUM_POWER_PRO:
            if plan.target >= 0 and plan.remain_hp <= 0:
                return -1
            if not self.can_attack:
                return -1
            return 14000 if 0 < plan.remain_hp <= 30 else 5000
        if (
            card.id in {C.ULTRA_BALL, C.POKE_PAD, C.FIGHTING_GONG}
            and self._should_take_ko_now()
        ):
            return -1
        if card.id == C.ULTRA_BALL:
            lucario_line = (
                self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX]
            )
            urgent = (
                lucario_line == 0
                or (
                    self.field_counts[C.RIOLU] > 0
                    and self.hand_counts[C.MEGA_LUCARIO_EX] == 0
                )
            )
            if urgent:
                return 9500
            return 7500 if (self.me.handCount or 0) >= 6 else -1
        if card.id == C.POKE_PAD:
            return 12000 if self._needs_non_rule_pokemon() else -1
        if card.id == C.FIGHTING_GONG:
            return 10000
        if card.id == C.BOSS_ORDERS:
            if plan.target < 1:
                return -1
            return 7000 if plan.remain_hp <= 0 else 4300
        if card.id == C.JUDGE:
            own_hand = self.me.handCount or 0
            opponent_hand = self.opponent.handCount or 0
            hand_gap = opponent_hand - own_hand
            if hand_gap >= 3:
                return 4200
            return 3800 if hand_gap >= 1 and opponent_hand >= 5 else -1
        if card.id == C.LILLIE_DETERMINATION:
            if self._low_deck():
                return -1
            own_hand = self.me.handCount or 0
            opponent_hand = self.opponent.handCount or 0
            if self.state.turn <= 3:
                return 4100
            if own_hand <= 5:
                return 3900
            return 3500 if own_hand <= 6 and opponent_hand <= 7 else -1
        return 10000

    def _should_take_ko_now(self) -> bool:
        return (
            not self._is_early_setup_phase()
            and self.can_attack
            and plan.target >= 0
            and plan.remain_hp <= 0
        )

    def _needs_non_rule_pokemon(self) -> bool:
        lucario_line = (
            self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX]
        )
        if lucario_line < 2:
            return True
        if self.field_counts[C.LUNATONE] == 0 or self.field_counts[C.SOLROCK] == 0:
            return True
        return self.field_counts[C.MAKUHITA] == 0

    def _low_deck(self) -> bool:
        return self.me.deckCount <= LOW_DECK_COUNT

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(
            self.obs, option.inPlayArea, option.inPlayIndex, self.my_index
        )
        if not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.HERO_CAPE:
            score = 7000
            if self._opponent_is_water_deck():
                if pokemon.id == C.RIOLU:
                    return 12200
                if pokemon.id == C.MEGA_LUCARIO_EX:
                    return 12800
            if pokemon.id == C.RIOLU:
                score += 100
            elif pokemon.id == C.MEGA_LUCARIO_EX:
                score += 200
            return score

        if self._planned_bench_lucario_needs_wally():
            return -1

        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        if self._is_early_setup_phase():
            if pokemon.id == C.RIOLU:
                score += 200
            elif pokemon.id == C.SOLROCK:
                score += 200
        if (
            pokemon.id == C.MEGA_LUCARIO_EX
            and len(pokemon.energies) < 2
            and not self._opponent_is_crustle_wall()
        ):
            score += 400
        board_index = (
            option.inPlayIndex
            if option.inPlayArea == AreaType.ACTIVE
            else option.inPlayIndex + 1
        )
        if board_index == plan.attacker and plan.needs_energy:
            score += 600
        return score

    def _score_evolve(self, option) -> float:
        evolution = get_card(self.obs, option.area, option.index, self.my_index)
        pokemon = get_card(
            self.obs, option.inPlayArea, option.inPlayIndex, self.my_index
        )
        if not isinstance(pokemon, Pokemon):
            return 0
        if (
            pokemon.id == C.MAKUHITA
            and plan.target == 0
            and not self._opponent_is_crustle_wall()
        ):
            return -1
        score = 9000 + len(pokemon.energies)
        if evolution is not None:
            if evolution.id == C.HARIYAMA and self._opponent_is_crustle_wall():
                score += 700
            elif (
                evolution.id == C.MEGA_LUCARIO_EX
                and not self._opponent_is_crustle_wall()
            ):
                score += 500
        return score

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if option.area == AreaType.STADIUM:
            return 1
        if card.id == C.LUNATONE and self._low_deck():
            return -1
        if (
            card.id == C.LUNATONE
            and self._active_damaged_lucario() is not None
            and self.hand_counts[C.WALLYS_COMPASSION] == 0
        ):
            return 35000
        if card.id == C.LUNATONE:
            if not self.state.energyAttached and self.hand_counts[C.BASIC_FIGHTING_ENERGY] <= 1:
                return 7000
            return 13000
        return 1

    def _remember_lunatone_ability(self, ranked: list[int]) -> None:
        global ability_used
        if self.context != SelectContext.MAIN or not ranked:
            return
        option = self.select.option[ranked[0]]
        if option.type != OptionType.ABILITY:
            return
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is not None and card.id == C.LUNATONE:
            ability_used = True

    def _remember_power_pro(self, ranked: list[int]) -> None:
        global power_pro_used
        if self.context != SelectContext.MAIN or not ranked:
            return
        option = self.select.option[ranked[0]]
        if option.type != OptionType.PLAY:
            return
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        if card is not None and card.id == C.PREMIUM_POWER_PRO:
            power_pro_used += 1


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    global pre_turn
    global ability_used
    global power_pro_used
    global plan

    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        ability_used = False
        power_pro_used = 0
        plan = AttackPlan()

    return LucarioPolicy(obs).choose()
