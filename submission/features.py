"""Experimental Lucario feature builder (v2).

ONE implementation is used both offline (scripts.catboost.generate_data) and at
inference (decks/lucario_v3_cb/policy.py); building the vector twice
would silently create train/serve skew. Imports are restricted to ``cg.api`` +
stdlib so this file can be bundled into a Kaggle submission.

Layout: ``Features(obs).state`` is the per-decision block (A/B/C/E), and
``Features(obs).option(i)`` is the per-candidate block (D). A training row is
``state + option(i)``; the label is 1 when the teacher picked option ``i``.

Design notes (see docs/catboost-action-scoring-plan.md §5):
- card ids are never emitted as ordinals: own-deck cards become one-hots and
  every other card is reduced to its attributes, because a tree cannot learn an
  embedding from an id the way the Transformer encoder does;
- bench Pokemon are aggregated by role (card id), never by slot index, so the
  engine's bench ordering cannot leak in as noise;
- differences (a - b) are materialised because axis-aligned splits cannot form
  them.
"""
from __future__ import annotations

from cg.api import (
    AreaType,
    CardType,
    EnergyType,
    OptionType,
    SelectContext,
    all_attack,
    all_card_data,
)

FEATURE_VERSION = 2

# --- vocabularies -----------------------------------------------------------
# The 17 distinct cards in the current Mega Lucario ex v3 list.
OWN_CARDS = [
    6,  # Basic Fighting Energy
    673,  # Makuhita
    674,  # Hariyama
    675,  # Lunatone
    676,  # Solrock
    677,  # Riolu
    678,  # Mega Lucario ex
    1121,  # Ultra Ball
    1123,  # Switch
    1141,  # Premium Power Pro
    1142,  # Fighting Gong
    1152,  # Poké Pad
    1159,  # Hero's Cape
    1182,  # Boss’s Orders
    1213,  # Judge
    1227,  # Lillie's Determination
    1229,  # Wally's Compassion
]
OWN_POKE = [673, 674, 675, 676, 677, 678]
OWN_TOOL = 1159  # Hero's Cape
OWN_STADIUM = -1  # deck runs no stadium; sentinel keeps the 3-way one-hot valid
OWN_KEY_POKEMON = {677, 678}  # Riolu -> Mega Lucario ex attacker line

# Opponent Pokemon vocabulary covering the current top-episode meta. It will be
# regenerated from the Lucario replay cohort before the production model is
# published; the tail is cheap for a tree, so no cutoff is applied.
OPP_POKE = [
    24, 31, 42, 63, 65, 66, 88, 89, 90, 91,
    92, 93, 96, 100, 104, 108, 112, 115, 116, 117,
    119, 120, 121, 140, 144, 149, 150, 162, 163, 172,
    173, 174, 183, 184, 224, 235, 240, 247, 272, 292,
    293, 303, 305, 306, 322, 324, 325, 326, 341, 342,
    343, 344, 345, 346, 379, 380, 381, 387, 400, 401,
    402, 403, 404, 414, 431, 434, 484, 646, 647, 648,
    650, 651, 652, 655, 673, 674, 675, 676, 677, 678,
    708, 709, 710, 741, 742, 743, 756, 791, 848, 849,
    860, 861, 869, 906, 917, 918, 920, 978, 1051, 1052,
    1071,
]

_OWN_INDEX = {cid: i for i, cid in enumerate(OWN_CARDS)}
_OWN_POKE_INDEX = {cid: i for i, cid in enumerate(OWN_POKE)}
_OPP_POKE_INDEX = {cid: i for i, cid in enumerate(OPP_POKE)}
_OPP_OOV = len(OPP_POKE)

CONTEXTS = [
    SelectContext.MAIN,
    SelectContext.TO_HAND,
    SelectContext.DISCARD,
    SelectContext.TO_DECK,
    SelectContext.TO_BENCH,
    SelectContext.TO_ACTIVE,
    SelectContext.SWITCH,
    SelectContext.ATTACH_FROM,
    SelectContext.ATTACH_TO,
    SelectContext.EVOLVES_FROM,
    SelectContext.DAMAGE_COUNTER,
    SelectContext.REMOVE_DAMAGE_COUNTER,
    SelectContext.ACTIVATE,
]
_CONTEXT_INDEX = {c: i for i, c in enumerate(CONTEXTS)}

OPTION_TYPES = [
    OptionType.PLAY,
    OptionType.ATTACH,
    OptionType.EVOLVE,
    OptionType.ABILITY,
    OptionType.DISCARD,
    OptionType.RETREAT,
    OptionType.ATTACK,
    OptionType.END,
    OptionType.CARD,
    OptionType.ENERGY_CARD,
    OptionType.ENERGY,
    OptionType.TOOL_CARD,
    OptionType.NUMBER,
    OptionType.YES,
    OptionType.NO,
    OptionType.SPECIAL_CONDITION,
]
_OPTION_TYPE_INDEX = {t: i for i, t in enumerate(OPTION_TYPES)}

FIGHTING_LIKE = {EnergyType.FIGHTING, EnergyType.RAINBOW, EnergyType.TEAM_ROCKET}

# --- card tables ------------------------------------------------------------
_CARD = {c.cardId: c for c in all_card_data()}
_ATTACK = {a.attackId: a for a in all_attack()}


def _attack_stats(card_id: int, index: int) -> tuple[float, float, float]:
    """(damage, total cost, Fighting-compatible cost) of a card's Nth attack."""
    data = _CARD.get(card_id)
    if data is None or index >= len(data.attacks):
        return (0.0, 0.0, 0.0)
    atk = _ATTACK.get(data.attacks[index])
    if atk is None:
        return (0.0, 0.0, 0.0)
    fighting = sum(1 for e in atk.energies if e in FIGHTING_LIKE)
    return (float(atk.damage), float(len(atk.energies)), float(fighting))


def _card_attrs(card_id: int) -> dict:
    data = _CARD.get(card_id)
    if data is None:
        return {"hp": 0.0, "retreat": 0.0, "ex": 0.0, "mega": 0.0, "tera": 0.0,
                "stage": 0.0, "prize": 1.0, "n_attacks": 0.0,
                "max_dmg": 0.0, "min_cost": 0.0, "max_cost": 0.0}
    stage = 2.0 if data.stage2 else (1.0 if data.stage1 else 0.0)
    prize = 3.0 if data.megaEx else (2.0 if data.ex else 1.0)
    dmgs, costs = [], []
    for aid in data.attacks:
        atk = _ATTACK.get(aid)
        if atk is not None:
            dmgs.append(float(atk.damage))
            costs.append(float(len(atk.energies)))
    return {
        "hp": float(data.hp),
        "retreat": float(data.retreatCost),
        "ex": float(data.ex),
        "mega": float(data.megaEx),
        "tera": float(data.tera),
        "stage": stage,
        "prize": prize,
        "n_attacks": float(len(data.attacks)),
        "max_dmg": max(dmgs) if dmgs else 0.0,
        "min_cost": min(costs) if costs else 0.0,
        "max_cost": max(costs) if costs else 0.0,
    }


_ATTRS = {cid: _card_attrs(cid) for cid in _CARD}
_EMPTY_ATTRS = _card_attrs(-1)


def attrs(card_id: int) -> dict:
    return _ATTRS.get(card_id, _EMPTY_ATTRS)


def n_fighting(poke) -> float:
    return float(sum(1 for e in poke.energies if e in FIGHTING_LIKE))


def ready_energy_count(card_id: int) -> int:
    if card_id in {673, 674}:  # Makuhita / Hariyama
        return 3
    if card_id == 676:  # Solrock
        return 1
    return 2


# --- turn ownership ---------------------------------------------------------
def turn_player(state) -> int:
    """Seat whose turn it is, or -1 before the first turn / unknown."""
    if state is None or state.turn <= 0 or state.firstPlayer < 0:
        return -1
    return state.firstPlayer if state.turn % 2 == 1 else 1 - state.firstPlayer


def is_own_turn(state) -> bool:
    tp = turn_player(state)
    return tp >= 0 and tp == state.yourIndex


# --- feature names ----------------------------------------------------------
def _state_names() -> list[str]:
    n: list[str] = [
        "turn", "my_turn_no", "turn_action_count", "is_first_player",
        "supporter_played", "stadium_played", "energy_attached", "retreated",
        "stadium_none", "stadium_is_own", "stadium_is_other",
    ]
    n += [f"ctx_{int(c)}" for c in CONTEXTS] + ["ctx_other"]
    # Search targets differ materially between Ultra Ball, Poke Pad and
    # Fighting Gong even though all three use SelectContext.TO_HAND.
    n += ["effect_none"] + [f"effect_card_{c}" for c in OWN_CARDS] + ["effect_other"]
    n += ["min_count", "max_count", "n_options", "remain_damage_counter", "remain_energy_cost"]

    for side in ("me", "opp"):
        n += [f"{side}_deck_count", f"{side}_n_discard", f"{side}_hand_count",
              f"{side}_n_bench", f"{side}_bench_free", f"{side}_n_prize",
              f"{side}_n_board", f"{side}_poisoned", f"{side}_burned",
              f"{side}_asleep", f"{side}_paralyzed", f"{side}_confused"]

    n += ["me_active_empty"] + [f"me_active_is_{c}" for c in OWN_POKE]
    n += ["me_active_hp", "me_active_max_hp", "me_active_damage", "me_active_hp_ratio",
          "me_active_n_energy", "me_active_n_fighting", "me_active_n_tools", "me_active_has_tool",
          "me_active_appear_this_turn", "me_active_retreat_cost", "me_active_is_ex",
          "me_active_is_mega", "me_active_stage", "me_active_n_pre_evolution"]
    for k in (0, 1):
        n += [f"me_atk{k}_damage", f"me_atk{k}_cost", f"me_atk{k}_cost_fighting",
              f"me_atk{k}_usable", f"me_atk{k}_shortfall"]
    for c in OWN_POKE:
        n += [f"me_bench_{c}_count", f"me_bench_{c}_max_energy", f"me_bench_{c}_max_damage",
              f"me_bench_{c}_n_ready", f"me_bench_{c}_n_appear"]
    n += ["me_bench_energy_total", "me_bench_tools_total"]
    n += [f"me_hand_{c}" for c in OWN_CARDS]
    n += [f"me_discard_{c}" for c in OWN_CARDS]
    n += [f"me_hidden_{c}" for c in OWN_CARDS]

    n += ["opp_active_empty", "opp_active_facedown", "opp_active_hp", "opp_active_max_hp",
          "opp_active_damage", "opp_active_hp_ratio", "opp_active_n_energy",
          "opp_active_n_tools", "opp_active_retreat_cost", "opp_active_is_ex",
          "opp_active_is_mega", "opp_active_prize_value", "opp_active_stage",
          "opp_active_n_attacks", "opp_active_max_damage", "opp_active_min_cost",
          "opp_active_max_cost"]
    n += [f"opp_active_is_{c}" for c in OPP_POKE] + ["opp_active_is_other"]
    n += ["opp_bench_n_ex", "opp_bench_n_mega", "opp_bench_n_basic", "opp_bench_n_stage1",
          "opp_bench_n_stage2", "opp_bench_n_damaged", "opp_bench_max_damage",
          "opp_bench_min_hp", "opp_bench_min_hp_ex", "opp_bench_energy_total",
          "opp_bench_n_tera", "opp_bench_max_prize_value"]
    n += [f"opp_bench_{c}_count" for c in OPP_POKE] + ["opp_bench_other_count"]
    n += ["opp_discard_n_energy", "opp_discard_n_pokemon", "opp_discard_n_item",
          "opp_discard_n_supporter", "opp_discard_n_tool", "opp_discard_n_stadium",
          "opp_discard_n_ace_spec", "opp_prize_taken"]

    n += ["diff_prize", "diff_deck_count", "diff_hand_count", "diff_discard",
          "diff_my_damage_vs_opp_hp", "diff_opp_damage_vs_my_hp", "diff_my_energy_vs_cost"]
    # Compact Lucario-specific strategic state. These are deliberately
    # materialised instead of asking depth-6 trees to reconstruct them from
    # several active/bench columns before interacting with an option.
    n += ["me_lucario_line_count", "me_ready_lucario_count",
          "me_hariyama_line_count", "me_ready_hariyama_count",
          "me_has_lunatone_solrock_pair", "opp_active_weak_to_fighting",
          "opp_active_resists_fighting"]
    return n


def _option_names() -> list[str]:
    n = [f"opt_type_{int(t)}" for t in OPTION_TYPES]
    n += [f"opt_card_{c}" for c in OWN_CARDS]
    n += ["opt_target_my_active", "opt_target_my_bench", "opt_target_opp_active",
          "opt_target_opp_bench", "opt_src_hand", "opt_src_deck", "opt_src_discard"]
    n += ["opt_t_hp_ratio", "opt_t_damage", "opt_t_n_energy", "opt_t_n_fighting",
          "opt_t_is_ex", "opt_t_is_key_pokemon", "opt_t_appear_this_turn"]
    n += ["opt_atk_damage", "opt_atk_cost", "opt_atk_cost_fighting", "opt_atk_index"]
    n += ["opt_number", "opt_special_condition", "opt_same_card_in_hand", "opt_dup_rank"]
    n += ["opt_atk_damage_vs_opp_hp"]
    # Candidate-specific projections of the deck's actual setup goals.
    n += [f"opt_target_own_{c}" for c in OWN_POKE]
    n += ["opt_t_ready_now", "opt_t_energy_shortfall", "opt_t_shortfall_after",
          "opt_t_becomes_ready", "opt_card_missing_on_board",
          "opt_card_completes_line", "opt_card_completes_pair",
          "opt_atk_effective_damage", "opt_atk_effective_margin", "opt_atk_ko"]
    return n


STATE_NAMES = _state_names()
OPTION_NAMES = _option_names()
FEATURE_NAMES = STATE_NAMES + OPTION_NAMES
N_STATE = len(STATE_NAMES)
N_OPTION = len(OPTION_NAMES)
N_FEATURES = len(FEATURE_NAMES)
_V2_OPTION_BASE = N_OPTION - 16


class Features:
    """Builds the per-decision state vector once, then per-option vectors."""

    def __init__(self, obs, deck: list[int] | None = None):
        self.obs = obs
        self.st = obs.current
        self.sel = obs.select
        self.mine = self.st.yourIndex
        self.me = self.st.players[self.mine]
        self.opp = self.st.players[1 - self.mine]
        self.deck = deck or []
        self._dup_seen: dict[tuple, int] = {}
        self.state = self._build_state()

    # -- helpers -------------------------------------------------------------
    def _card_at(self, area, index, player_index):
        if area is None or index is None or index < 0:
            return None
        ps = self.st.players[player_index if player_index is not None else self.mine]
        try:
            if area == AreaType.DECK:
                return self.sel.deck[index] if self.sel.deck else None
            if area == AreaType.HAND:
                return ps.hand[index] if ps.hand else None
            if area == AreaType.DISCARD:
                return ps.discard[index]
            if area == AreaType.ACTIVE:
                return ps.active[index]
            if area == AreaType.BENCH:
                return ps.bench[index]
            if area == AreaType.PRIZE:
                return ps.prize[index]
            if area == AreaType.STADIUM:
                return self.st.stadium[index]
            if area == AreaType.LOOKING:
                return self.st.looking[index] if self.st.looking else None
        except (IndexError, TypeError):
            return None
        return None

    # -- state ---------------------------------------------------------------
    def _build_state(self) -> list[float]:
        st, sel, me, opp = self.st, self.sel, self.me, self.opp
        v: list[float] = []

        stadium_id = st.stadium[0].id if st.stadium else 0
        v += [
            float(st.turn), float((st.turn + 1) // 2), float(st.turnActionCount),
            float(st.firstPlayer == self.mine),
            float(st.supporterPlayed), float(st.stadiumPlayed),
            float(st.energyAttached), float(st.retreated),
            float(stadium_id == 0), float(stadium_id == OWN_STADIUM),
            float(stadium_id not in (0, OWN_STADIUM)),
        ]
        ctx_hot = [0.0] * (len(CONTEXTS) + 1)
        ctx_hot[_CONTEXT_INDEX.get(sel.context, len(CONTEXTS))] = 1.0
        v += ctx_hot
        effect = sel.effect or sel.contextCard
        effect_hot = [0.0] * (len(OWN_CARDS) + 2)
        if effect is None:
            effect_hot[0] = 1.0
        elif effect.id in _OWN_INDEX:
            effect_hot[1 + _OWN_INDEX[effect.id]] = 1.0
        else:
            effect_hot[-1] = 1.0
        v += effect_hot
        v += [float(sel.minCount), float(sel.maxCount), float(len(sel.option)),
              float(sel.remainDamageCounter), float(sel.remainEnergyCost)]

        for ps in (me, opp):
            board = [p for p in (list(ps.active) + list(ps.bench)) if p is not None]
            v += [
                float(ps.deckCount), float(len(ps.discard)), float(ps.handCount),
                float(len(ps.bench)), float(max(0, 5 - len(ps.bench))),
                float(len(ps.prize)), float(len(board)),
                float(ps.poisoned), float(ps.burned), float(ps.asleep),
                float(ps.paralyzed), float(ps.confused),
            ]

        # my active
        act = me.active[0] if me.active and me.active[0] is not None else None
        hot = [0.0] * len(OWN_POKE)
        if act is not None and act.id in _OWN_POKE_INDEX:
            hot[_OWN_POKE_INDEX[act.id]] = 1.0
        v += [float(act is None)] + hot
        if act is None:
            v += [0.0] * 14
            v += [0.0] * 10
            my_best_damage = 0.0
        else:
            a = attrs(act.id)
            v += [
                float(act.hp), float(act.maxHp), float(act.maxHp - act.hp),
                (act.hp / act.maxHp) if act.maxHp else 0.0,
                float(len(act.energies)), n_fighting(act), float(len(act.tools)),
                float(any(t.id == OWN_TOOL for t in act.tools)),
                float(act.appearThisTurn), a["retreat"], a["ex"], a["mega"],
                a["stage"], float(len(act.preEvolution)),
            ]
            n_e, n_f = len(act.energies), n_fighting(act)
            my_best_damage = 0.0
            for k in (0, 1):
                dmg, cost, cost_fighting = _attack_stats(act.id, k)
                usable = float(n_e >= cost and n_f >= cost_fighting)
                v += [dmg, cost, cost_fighting, usable, max(0.0, cost - n_e)]
                if usable:
                    my_best_damage = max(my_best_damage, dmg)

        # my bench, aggregated by role
        bench_energy = bench_tools = 0.0
        agg = {c: [0.0, 0.0, 0.0, 0.0, 0.0] for c in OWN_POKE}
        for p in me.bench:
            if p is None:
                continue
            bench_energy += len(p.energies)
            bench_tools += len(p.tools)
            slot = agg.get(p.id)
            if slot is None:
                continue
            slot[0] += 1
            slot[1] = max(slot[1], float(len(p.energies)))
            slot[2] = max(slot[2], float(p.maxHp - p.hp))
            slot[3] += float(n_fighting(p) >= ready_energy_count(p.id))
            slot[4] += float(p.appearThisTurn)
        for c in OWN_POKE:
            v += agg[c]
        v += [bench_energy, bench_tools]

        hand_counts = {c: 0.0 for c in OWN_CARDS}
        for card in me.hand or []:
            if card.id in hand_counts:
                hand_counts[card.id] += 1
        discard_counts = {c: 0.0 for c in OWN_CARDS}
        for card in me.discard:
            if card.id in discard_counts:
                discard_counts[card.id] += 1
        board_counts = {c: 0.0 for c in OWN_CARDS}
        for p in list(me.active) + list(me.bench):
            if p is None:
                continue
            for cid in [p.id] + [x.id for x in p.tools] + [x.id for x in p.energyCards] \
                    + [x.id for x in p.preEvolution]:
                if cid in board_counts:
                    board_counts[cid] += 1
        deck_total = {c: 0.0 for c in OWN_CARDS}
        for cid in self.deck:
            if cid in deck_total:
                deck_total[cid] += 1
        v += [hand_counts[c] for c in OWN_CARDS]
        v += [discard_counts[c] for c in OWN_CARDS]
        v += [max(0.0, deck_total[c] - hand_counts[c] - discard_counts[c] - board_counts[c])
              for c in OWN_CARDS]

        # opponent active
        oact = opp.active[0] if opp.active and opp.active[0] is not None else None
        facedown = bool(opp.active) and opp.active[0] is None
        if oact is None:
            v += [float(not opp.active), float(facedown)] + [0.0] * 15
            v += [0.0] * (len(OPP_POKE) + 1)
            opp_hp = 0.0
            opp_best_damage = 0.0
        else:
            a = attrs(oact.id)
            opp_hp = float(oact.hp)
            opp_best_damage = a["max_dmg"]
            v += [
                0.0, float(facedown), float(oact.hp), float(oact.maxHp),
                float(oact.maxHp - oact.hp), (oact.hp / oact.maxHp) if oact.maxHp else 0.0,
                float(len(oact.energies)), float(len(oact.tools)), a["retreat"],
                a["ex"], a["mega"], a["prize"], a["stage"], a["n_attacks"],
                a["max_dmg"], a["min_cost"], a["max_cost"],
            ]
            hot = [0.0] * (len(OPP_POKE) + 1)
            hot[_OPP_POKE_INDEX.get(oact.id, _OPP_OOV)] = 1.0
            v += hot

        n_ex = n_mega = n_basic = n_st1 = n_st2 = n_dmg = n_tera = 0.0
        max_dmg = max_prize = 0.0
        min_hp = min_hp_ex = 0.0
        energy_total = 0.0
        opp_bench_counts = [0.0] * (len(OPP_POKE) + 1)
        for p in opp.bench:
            if p is None:
                continue
            a = attrs(p.id)
            n_ex += a["ex"]
            n_mega += a["mega"]
            n_tera += a["tera"]
            n_basic += float(a["stage"] == 0)
            n_st1 += float(a["stage"] == 1)
            n_st2 += float(a["stage"] == 2)
            dmg = float(p.maxHp - p.hp)
            n_dmg += float(dmg > 0)
            max_dmg = max(max_dmg, dmg)
            max_prize = max(max_prize, a["prize"])
            energy_total += len(p.energies)
            min_hp = float(p.hp) if min_hp == 0.0 else min(min_hp, float(p.hp))
            if a["ex"]:
                min_hp_ex = float(p.hp) if min_hp_ex == 0.0 else min(min_hp_ex, float(p.hp))
            opp_bench_counts[_OPP_POKE_INDEX.get(p.id, _OPP_OOV)] += 1
        v += [n_ex, n_mega, n_basic, n_st1, n_st2, n_dmg, max_dmg, min_hp, min_hp_ex,
              energy_total, n_tera, max_prize]
        v += opp_bench_counts

        d_energy = d_poke = d_item = d_sup = d_tool = d_stadium = d_ace = 0.0
        for card in opp.discard:
            data = _CARD.get(card.id)
            if data is None:
                continue
            if data.cardType == CardType.POKEMON:
                d_poke += 1
            elif data.cardType == CardType.ITEM:
                d_item += 1
            elif data.cardType == CardType.SUPPORTER:
                d_sup += 1
            elif data.cardType == CardType.TOOL:
                d_tool += 1
            elif data.cardType == CardType.STADIUM:
                d_stadium += 1
            else:
                d_energy += 1
            d_ace += float(data.aceSpec)
        v += [d_energy, d_poke, d_item, d_sup, d_tool, d_stadium, d_ace,
              float(6 - len(opp.prize))]

        my_active_energy = float(len(act.energies)) if act is not None else 0.0
        my_best_cost = 0.0
        if act is not None:
            costs = [c for _, c, _ in (_attack_stats(act.id, k) for k in (0, 1)) if c > 0]
            my_best_cost = min(costs) if costs else 0.0
        my_hp = float(act.hp) if act is not None else 0.0
        v += [
            float(len(me.prize) - len(opp.prize)),
            float(me.deckCount - opp.deckCount),
            float(me.handCount - opp.handCount),
            float(len(me.discard) - len(opp.discard)),
            my_best_damage - opp_hp,
            opp_best_damage - my_hp,
            my_active_energy - my_best_cost,
        ]

        my_board = [p for p in list(me.active) + list(me.bench) if p is not None]
        field_counts = {c: 0 for c in OWN_POKE}
        ready_lucario = ready_hariyama = 0.0
        for p in my_board:
            if p.id in field_counts:
                field_counts[p.id] += 1
            if p.id in {677, 678} and n_fighting(p) >= 2:
                ready_lucario += 1.0
            if p.id in {673, 674} and n_fighting(p) >= 3:
                ready_hariyama += 1.0
        odata = _CARD.get(oact.id) if oact is not None else None
        v += [
            float(field_counts[677] + field_counts[678]), ready_lucario,
            float(field_counts[673] + field_counts[674]), ready_hariyama,
            float(field_counts[675] > 0 and field_counts[676] > 0),
            float(odata is not None and odata.weakness == EnergyType.FIGHTING),
            float(odata is not None and odata.resistance == EnergyType.FIGHTING),
        ]

        self._opp_active_hp = opp_hp
        self._opp_active = oact
        self._my_active = act
        self._hand_counts = hand_counts
        self._field_counts = field_counts
        return v

    # -- option --------------------------------------------------------------
    def option(self, i: int) -> list[float]:
        o = self.sel.option[i]
        v = [0.0] * N_OPTION
        idx = _OPTION_TYPE_INDEX.get(o.type)
        if idx is not None:
            v[idx] = 1.0
        base = len(OPTION_TYPES)

        src, target = self._resolve(o)

        if src is not None and src.id in _OWN_INDEX:
            v[base + _OWN_INDEX[src.id]] = 1.0
        base += len(OWN_CARDS)

        t_owner = o.playerIndex if o.playerIndex is not None else self.mine
        t_area = o.inPlayArea if o.inPlayArea is not None else o.area
        if target is not None:
            mine = t_owner == self.mine
            if t_area == AreaType.ACTIVE:
                v[base + (0 if mine else 2)] = 1.0
            elif t_area == AreaType.BENCH:
                v[base + (1 if mine else 3)] = 1.0
        if o.area == AreaType.HAND:
            v[base + 4] = 1.0
        elif o.area == AreaType.DECK:
            v[base + 5] = 1.0
        elif o.area == AreaType.DISCARD:
            v[base + 6] = 1.0
        base += 7

        if target is not None and hasattr(target, "maxHp"):
            v[base + 0] = (target.hp / target.maxHp) if target.maxHp else 0.0
            v[base + 1] = float(target.maxHp - target.hp)
            v[base + 2] = float(len(target.energies))
            v[base + 3] = n_fighting(target)
            v[base + 4] = attrs(target.id)["ex"]
            v[base + 5] = float(target.id in OWN_KEY_POKEMON)
            v[base + 6] = float(target.appearThisTurn)
        base += 7

        attack_damage = 0.0
        if o.type == OptionType.ATTACK and self._my_active is not None:
            data = _CARD.get(self._my_active.id)
            k = data.attacks.index(o.attackId) if data and o.attackId in data.attacks else 0
            dmg, cost, cost_fighting = _attack_stats(self._my_active.id, k)
            attack_damage = dmg
            v[base + 0] = dmg
            v[base + 1] = cost
            v[base + 2] = cost_fighting
            v[base + 3] = float(k)
            v[_V2_OPTION_BASE - 1] = dmg - self._opp_active_hp
        base += 4

        v[base + 0] = float(o.number) if o.number is not None else 0.0
        v[base + 1] = float(o.specialConditionType) if o.specialConditionType is not None else 0.0
        v[base + 2] = self._hand_counts.get(src.id, 0.0) if src is not None else 0.0
        key = (int(o.type), src.id if src is not None else -1,
               target.serial if target is not None else -1)
        v[base + 3] = float(self._dup_seen.get(key, 0))
        self._dup_seen[key] = self._dup_seen.get(key, 0) + 1

        # Keep the original final column (raw attack damage margin) in place.
        base = _V2_OPTION_BASE
        if target is not None and t_owner == self.mine and target.id in _OWN_POKE_INDEX:
            v[base + _OWN_POKE_INDEX[target.id]] = 1.0
            need = float(ready_energy_count(target.id))
            have = n_fighting(target)
            shortfall = max(0.0, need - have)
            will_attach = o.type == OptionType.ATTACH or self.sel.context == SelectContext.ATTACH_FROM
            shortfall_after = max(0.0, shortfall - float(will_attach))
            target_base = base + len(OWN_POKE)
            v[target_base + 0] = float(shortfall == 0)
            v[target_base + 1] = shortfall
            v[target_base + 2] = shortfall_after
            v[target_base + 3] = float(will_attach and shortfall > 0 and shortfall_after == 0)

        projected = base + len(OWN_POKE)
        if src is not None and src.id in _OWN_POKE_INDEX:
            v[projected + 4] = float(self._field_counts[src.id] == 0)
            v[projected + 5] = float(
                (src.id == 674 and self._field_counts[673] > 0)
                or (src.id == 678 and self._field_counts[677] > 0)
            )
            v[projected + 6] = float(
                (src.id == 675 and self._field_counts[675] == 0
                 and self._field_counts[676] > 0)
                or (src.id == 676 and self._field_counts[676] == 0
                    and self._field_counts[675] > 0)
            )

        if o.type == OptionType.ATTACK and self._opp_active is not None:
            effective = attack_damage
            data = _CARD.get(self._opp_active.id)
            if data is not None and data.weakness == EnergyType.FIGHTING:
                effective *= 2.0
            elif data is not None and data.resistance == EnergyType.FIGHTING:
                effective = max(0.0, effective - 30.0)
            v[projected + 7] = effective
            v[projected + 8] = effective - self._opp_active_hp
            v[projected + 9] = float(effective >= self._opp_active_hp > 0)
        return v

    def _resolve(self, o):
        """(source card, target Pokemon) for an option, or (None, None)."""
        mine = self.mine
        owner = o.playerIndex if o.playerIndex is not None else mine
        if o.type == OptionType.PLAY:
            return (self._card_at(AreaType.HAND, o.index, mine), None)
        if o.type in (OptionType.ATTACH, OptionType.EVOLVE):
            return (self._card_at(o.area, o.index, mine),
                    self._card_at(o.inPlayArea, o.inPlayIndex, mine))
        if o.type in (OptionType.ABILITY, OptionType.DISCARD):
            card = self._card_at(o.area, o.index, mine)
            target = card if o.area in (AreaType.ACTIVE, AreaType.BENCH) else None
            return (card, target)
        if o.type == OptionType.RETREAT:
            act = self.me.active[0] if self.me.active else None
            return (act, act)
        if o.type == OptionType.CARD:
            card = self._card_at(o.area, o.index, owner)
            target = card if o.area in (AreaType.ACTIVE, AreaType.BENCH) else None
            return (card, target)
        if o.type == OptionType.TOOL_CARD:
            poke = self._card_at(o.area, o.index, owner)
            tool = None
            if poke is not None and o.toolIndex is not None and o.toolIndex < len(poke.tools):
                tool = poke.tools[o.toolIndex]
            return (tool, poke)
        if o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
            poke = self._card_at(o.area, o.index, owner)
            energy = None
            if poke is not None and o.energyIndex is not None \
                    and o.energyIndex < len(poke.energyCards):
                energy = poke.energyCards[o.energyIndex]
            return (energy, poke)
        if o.type == OptionType.ATTACK:
            act = self.me.active[0] if self.me.active else None
            return (act, act)
        return (None, None)
