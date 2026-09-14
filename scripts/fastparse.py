"""Faster drop-in replacement for cg.utils.to_dataclass.

The stock ``to_dataclass`` rebuilds per-class field/type metadata on every call.
This version caches the per-class plan once and specializes the observation
shapes used when generating CatBoost training data from replay JSON.
"""

from __future__ import annotations

import dataclasses

_PLAN_CACHE: dict[type, tuple] = {}


def _plan(cls: type):
    """Return (field_plan, defaults) for a class, cached."""
    cached = _PLAN_CACHE.get(cls)
    if cached is not None:
        return cached
    plan = {}
    defaults = {}
    for field in dataclasses.fields(cls):
        field_type = field.type
        dict_cls = field_type
        if hasattr(dict_cls, "__args__"):
            dict_cls = dict_cls.__args__[0]
        list_cls = None
        list_is_dc = False
        if hasattr(field_type, "__args__"):
            candidate = field_type.__args__[0]
            if hasattr(candidate, "__args__"):
                candidate = candidate.__args__[0]
                if hasattr(candidate, "__args__"):
                    candidate = candidate.__args__[0]
            list_cls = candidate
            list_is_dc = hasattr(candidate, "__dataclass_fields__")
        plan[field.name] = (dict_cls, list_cls, list_is_dc)
        if field.default is not dataclasses.MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            defaults[field.name] = field.default_factory()
    cached = (plan, defaults)
    _PLAN_CACHE[cls] = cached
    return cached


_SPEC: dict[type, object] = {}


def _build_spec() -> dict:
    from cg.api import (
        ApiResult,
        Card,
        Log,
        Observation,
        Option,
        PlayerState,
        Pokemon,
        SearchState,
        SelectData,
        State,
    )

    def card(data):
        if data is None:
            return None
        instance = object.__new__(Card)
        instance.__dict__ = data
        return instance

    def card_list(values):
        return [card(value) for value in values]

    def pokemon(data):
        if data is None:
            return None
        instance = object.__new__(Pokemon)
        converted = dict(data)
        converted["energyCards"] = card_list(data["energyCards"])
        converted["tools"] = card_list(data["tools"])
        converted["preEvolution"] = card_list(data["preEvolution"])
        instance.__dict__ = converted
        return instance

    def option(data):
        instance = object.__new__(Option)
        instance.__dict__ = dict(data)
        return instance

    def log(data):
        instance = object.__new__(Log)
        instance.__dict__ = dict(data)
        return instance

    def player(data):
        instance = object.__new__(PlayerState)
        converted = dict(data)
        converted["active"] = [pokemon(value) for value in data["active"]]
        converted["bench"] = [pokemon(value) for value in data["bench"]]
        converted["discard"] = card_list(data["discard"])
        converted["prize"] = card_list(data["prize"])
        converted["hand"] = None if data["hand"] is None else card_list(data["hand"])
        instance.__dict__ = converted
        return instance

    def state(data):
        if data is None:
            return None
        instance = object.__new__(State)
        converted = dict(data)
        converted["stadium"] = card_list(data["stadium"])
        looking = data["looking"]
        converted["looking"] = None if looking is None else [card(value) for value in looking]
        converted["players"] = [player(value) for value in data["players"]]
        instance.__dict__ = converted
        return instance

    def select(data):
        if data is None:
            return None
        instance = object.__new__(SelectData)
        converted = dict(data)
        converted["option"] = [option(value) for value in data["option"]]
        converted["deck"] = None if data["deck"] is None else card_list(data["deck"])
        converted["contextCard"] = card(data["contextCard"])
        converted["effect"] = card(data["effect"])
        instance.__dict__ = converted
        return instance

    def observation(data):
        if data is None:
            return None
        instance = object.__new__(Observation)
        converted = dict(data)
        converted["select"] = select(data["select"])
        converted["logs"] = [log(value) for value in data["logs"]]
        converted["current"] = state(data["current"])
        if "search_begin_input" not in converted:
            converted["search_begin_input"] = None
        instance.__dict__ = converted
        return instance

    def api_result(data):
        if data is None:
            return None
        instance = object.__new__(ApiResult)
        converted = dict(data)
        state_data = data["state"]
        if state_data is not None:
            search_state = object.__new__(SearchState)
            search_state.__dict__ = {
                "observation": observation(state_data["observation"]),
                "searchId": state_data["searchId"],
            }
            converted["state"] = search_state
        instance.__dict__ = converted
        return instance

    return {ApiResult: api_result, Observation: observation}


def fast_to_dataclass(data, cls):
    """Convert a dictionary to a dataclass using cached field metadata."""
    if data is None:
        return None
    spec = _SPEC
    if not spec:
        spec = _SPEC.update(_build_spec()) or _SPEC
    decoder = spec.get(cls)
    if decoder is not None:
        return decoder(data)
    cached = _PLAN_CACHE.get(cls)
    if cached is None:
        cached = _plan(cls)
    plan, defaults = cached
    converted = {}
    for key, value in data.items():
        info = plan.get(key)
        if info is None:
            continue
        value_type = type(value)
        if value_type is dict:
            converted[key] = fast_to_dataclass(value, info[0])
        elif value_type is list:
            if info[2]:
                element_type = info[1]
                converted[key] = [fast_to_dataclass(item, element_type) for item in value]
            else:
                converted[key] = value
        else:
            converted[key] = value
    for name, default in defaults.items():
        if name not in converted:
            converted[name] = default
    instance = object.__new__(cls)
    instance.__dict__ = converted
    return instance