"""Experimental feature v4 with opponent next-turn damage projections."""
from __future__ import annotations

try:
    from . import features_damage as base
    from . import incoming_damage
    from . import opponent_archetype
except ImportError:  # flattened Kaggle submission layout
    try:
        import features_damage_base as base  # type: ignore[no-redef]
    except ImportError:
        import features_damage as base  # type: ignore[no-redef]
    import incoming_damage  # type: ignore[no-redef]
    import opponent_archetype  # type: ignore[no-redef]


FEATURE_VERSION = 4
OWN_CARDS = base.OWN_CARDS
OPP_POKE = base.OPP_POKE
is_own_turn = base.is_own_turn

_SCENARIOS = ("now", "plus_energy", "evolution", "evolution_plus_energy")
INCOMING_STATE_NAMES = [
    f"incoming_{scenario}_{metric}"
    for scenario in _SCENARIOS
    for metric in ("damage", "margin", "ko", "known", "bench_damage", "bench_ko_count")
] + [
    "incoming_now_usable_attack_count",
    "incoming_plus_energy_unlocks_attack",
    "incoming_evolution_candidate_count",
    "incoming_plus_energy_gain",
    "incoming_evolution_gain",
    "incoming_evolution_plus_energy_gain",
    "incoming_plus_energy_new_ko",
    "incoming_evolution_new_ko",
    "incoming_evolution_plus_energy_new_ko",
]

ARCHETYPE_STATE_NAMES = opponent_archetype.feature_names()

STATE_NAMES = base.STATE_NAMES + INCOMING_STATE_NAMES + ARCHETYPE_STATE_NAMES
OPTION_NAMES = base.OPTION_NAMES
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


class Features:
    def __init__(self, obs, deck: list[int] | None = None, runtime_state=None):
        self.base = base.Features(obs, deck, runtime_state)
        self.incoming = incoming_damage.evaluate(obs, self.base.effects)
        self.archetype = opponent_archetype.infer(obs)
        self.state = (
            self.base.state
            + self._incoming_state()
            + opponent_archetype.feature_values(self.archetype)
        )

    @staticmethod
    def _threat_vector(threat) -> list[float]:
        return [
            threat.damage,
            threat.margin,
            float(threat.ko),
            float(threat.known),
            threat.bench_damage,
            float(threat.bench_ko_count),
        ]

    def _incoming_state(self) -> list[float]:
        result = []
        for scenario in _SCENARIOS:
            result.extend(self._threat_vector(getattr(self.incoming, scenario)))
        now = self.incoming.now
        plus_energy = self.incoming.plus_energy
        evolution = self.incoming.evolution
        evolution_plus = self.incoming.evolution_plus_energy
        result.extend([
            float(now.usable_attack_count),
            float(self.incoming.plus_energy_unlocks_attack),
            float(self.incoming.evolution_candidate_count),
            plus_energy.damage - now.damage,
            evolution.damage - now.damage,
            evolution_plus.damage - now.damage,
            float(plus_energy.ko and not now.ko),
            float(evolution.ko and not now.ko),
            float(evolution_plus.ko and not now.ko),
        ])
        return result

    def option(self, index: int) -> list[float]:
        return self.base.option(index)
