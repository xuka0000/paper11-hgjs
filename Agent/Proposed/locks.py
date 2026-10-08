from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DecisionPartition:
    fixed_arc_ids: frozenset[str]
    fixed_tuple_ids: frozenset[str]
    emergency_tuple_ids: frozenset[str]


def partition_decisions(state, unsafe_tuple_ids: frozenset[str]) -> DecisionPartition:
    emergency = state.locked_tuple_ids & unsafe_tuple_ids
    return DecisionPartition(
        state.locked_arc_ids,
        state.locked_tuple_ids - emergency,
        emergency,
    )


def plan_change(current: Mapping[str, int], previous: Mapping[str, int]) -> int:
    keys = set(current) | set(previous)
    return sum(abs(int(current.get(key, 0)) - int(previous.get(key, 0))) for key in keys)
