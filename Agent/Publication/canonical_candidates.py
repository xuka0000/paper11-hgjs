"""Target-independent canonical repair neighborhoods for physical search states."""
from dataclasses import dataclass
from hashlib import sha256

from Agent.Proposed.regions import CoreRegion
from Agent.Publication.action_features import (
    build_observable_action_features,
    physical_opportunity_key,
)
from Agent.Publication.closure import build_closure_index, close_region


@dataclass(frozen=True)
class CanonicalCandidate:
    candidate_id: str
    family: str
    anchor_id: str
    core_node_ids: tuple[str, ...]
    region_node_ids: tuple[str, ...]
    closure: object
    aliases: tuple[str, ...] = ()


def closure_signature(closure):
    return (
        tuple(closure.task_ids),
        tuple(closure.tuple_ids),
        tuple(closure.vehicle_ids),
        tuple(closure.uav_ids),
        tuple(closure.arc_ids),
        tuple(closure.anchors),
    )


def _round_robin_unique(sequences, limit):
    result = []
    seen = set()
    positions = [0] * len(sequences)
    while len(result) < limit:
        added = False
        for index, sequence in enumerate(sequences):
            while positions[index] < len(sequence):
                value = sequence[positions[index]]
                positions[index] += 1
                if value in seen:
                    continue
                seen.add(value)
                result.append(value)
                added = True
                break
            if len(result) >= limit:
                break
        if not added:
            break
    return result


def _diverse_tuple_ids(items, features, node_by_id, limit):
    items = list(items)
    if not items:
        return []
    sequences = [
        [item.tuple_id for item in sorted(
            items, key=lambda item: physical_opportunity_key(
                node_by_id[item.tuple_id], features))],
        [item.tuple_id for item in sorted(
            items, key=lambda item: (
                features[item.tuple_id]["direct_cost"],
                item.launch_layer, item.recovery_layer, item.tuple_id))],
        [item.tuple_id for item in sorted(
            items, key=lambda item: (
                item.launch_layer, item.recovery_layer, item.vehicle_id,
                item.uav_id, item.tuple_id))],
        [item.tuple_id for item in sorted(
            items, key=lambda item: (
                -item.recovery_layer, -item.launch_layer, item.vehicle_id,
                item.uav_id, item.tuple_id))],
    ]
    return _round_robin_unique(sequences, limit)


def _core_region(identifier, node_ids, node_by_id):
    node_ids = tuple(sorted(set(node_ids)))
    nodes = [node_by_id[node_id] for node_id in node_ids]
    tasks = sum(node.node_type == "task" for node in nodes)
    tuples = sum(node.node_type == "tuple" for node in nodes)
    vehicles = len({node.vehicle_id for node in nodes if node.vehicle_id is not None})
    layers = [node.time_layer for node in nodes if node.time_layer is not None]
    span = max(layers) - min(layers) if layers else 0
    stable = sha256(repr((identifier, node_ids)).encode("utf-8")).hexdigest()[:16]
    return CoreRegion(f"canonical:{stable}", identifier, node_ids,
                      tasks, tuples, vehicles, span)


def _region_node_ids(closure, node_by_id):
    candidates = []
    candidates.extend(f"task:{task_id}" for task_id in closure.task_ids)
    candidates.extend(closure.tuple_ids)
    candidates.extend(f"vehicle:{vehicle_id}" for vehicle_id in closure.vehicle_ids)
    candidates.extend(f"uav:{uav_id}" for uav_id in closure.uav_ids)
    candidates.extend(f"arc:{arc_id}" for arc_id in closure.arc_ids)
    for vehicle_id, pre_anchor, post_anchor in closure.anchors:
        for node, layer in (pre_anchor, post_anchor):
            key = f"time:{vehicle_id}:{node}:{layer}"
            if key in node_by_id:
                candidates.append(key)
    return tuple(sorted({node_id for node_id in candidates if node_id in node_by_id}))


def _window_centers(arcs, count):
    layers = sorted({int(value) for arc in arcs
                     for value in (arc.departure_layer, arc.arrival_layer)})
    if not layers:
        return ()
    if len(layers) <= count:
        return tuple(layers)
    return tuple(layers[round((index + 1) * (len(layers) - 1) / (count + 1))]
                 for index in range(count))


def build_canonical_candidates(
    state,
    graph,
    task_weights,
    objective_weights,
    closure_budget,
    *,
    locked_tuple_ids=frozenset(),
    locked_arc_ids=frozenset(),
    max_tuples_per_anchor=16,
    windows_per_vehicle=3,
    window_span=28,
    feature_rows=None,
):
    """Build task, UAV-resource, and vehicle-window closures from current state."""
    if max_tuples_per_anchor <= 0 or windows_per_vehicle <= 0 or window_span <= 0:
        raise ValueError("Canonical candidate limits must be positive")
    node_by_id = {node.node_id: node for node in graph.nodes}
    features = (
        build_observable_action_features(
            state, graph, task_weights, objective_weights)
        if feature_rows is None else feature_rows
    )
    tuples = tuple(item for item in state.tuples if item.admitted)
    selected_tuple_ids = set(state.selected_tuple_ids)
    selected_arc_ids = set(state.selected_arc_ids)
    specifications = []
    closure_index = build_closure_index(state)

    task_ids = sorted({int(task_id) for item in tuples for task_id in item.task_ids})
    for task_id in task_ids:
        relevant = [item for item in tuples if task_id in item.task_ids]
        selected = [item.tuple_id for item in relevant
                    if item.tuple_id in selected_tuple_ids]
        alternatives = _diverse_tuple_ids(
            [item for item in relevant if item.tuple_id not in selected_tuple_ids],
            features, node_by_id,
            max(0, max_tuples_per_anchor - len(selected)))
        node_ids = [f"task:{task_id}"] + selected + alternatives
        specifications.append(("task_exchange", f"task:{task_id}", node_ids))

    uav_ids = sorted({int(item.uav_id) for item in tuples})
    for uav_id in uav_ids:
        relevant = [item for item in tuples if int(item.uav_id) == uav_id]
        selected = [item.tuple_id for item in relevant
                    if item.tuple_id in selected_tuple_ids]
        alternatives = _diverse_tuple_ids(
            [item for item in relevant if item.tuple_id not in selected_tuple_ids],
            features, node_by_id,
            max(0, max_tuples_per_anchor - len(selected)))
        specifications.append((
            "uav_resource", f"uav:{uav_id}",
            [f"uav:{uav_id}"] + selected + alternatives))

    vehicle_ids = sorted({int(arc.vehicle_id) for arc in state.arcs})
    half_span = max(1, window_span // 2)
    for vehicle_id in vehicle_ids:
        vehicle_arcs = [arc for arc in state.arcs if int(arc.vehicle_id) == vehicle_id]
        for center in _window_centers(vehicle_arcs, windows_per_vehicle):
            lower = center - half_span
            upper = center + half_span
            window_arcs = [
                arc for arc in vehicle_arcs
                if arc.departure_layer <= upper and arc.arrival_layer >= lower]
            overlapping = [
                item for item in tuples
                if int(item.vehicle_id) == vehicle_id
                and item.launch_layer <= upper and item.recovery_layer >= lower]
            selected = [item.tuple_id for item in overlapping
                        if item.tuple_id in selected_tuple_ids]
            alternatives = _diverse_tuple_ids(
                [item for item in overlapping if item.tuple_id not in selected_tuple_ids],
                features, node_by_id,
                max(0, max_tuples_per_anchor - len(selected)))
            arc_ids = [f"arc:{arc.arc_id}" for arc in window_arcs]
            anchor = f"vehicle:{vehicle_id}:window:{center}"
            specifications.append((
                "vehicle_window", anchor,
                [f"vehicle:{vehicle_id}"] + arc_ids + selected + alternatives))

    candidates = []
    signature_to_index = {}
    for family, anchor, node_ids in specifications:
        node_ids = [node_id for node_id in node_ids if node_id in node_by_id]
        if not node_ids:
            continue
        core = _core_region(anchor, node_ids, node_by_id)
        closure = close_region(
            state, core, closure_budget, closure_index=closure_index
        )
        if not closure.accepted:
            continue
        released = (
            (set(closure.tuple_ids) - set(locked_tuple_ids))
            | (set(closure.arc_ids) - set(locked_arc_ids)))
        if not released:
            continue
        signature = closure_signature(closure)
        alias = f"{family}:{anchor}"
        previous = signature_to_index.get(signature)
        if previous is not None:
            old = candidates[previous]
            candidates[previous] = CanonicalCandidate(
                old.candidate_id, old.family, old.anchor_id,
                old.core_node_ids, old.region_node_ids, old.closure,
                tuple(sorted(set(old.aliases) | {alias})))
            continue
        stable = sha256(repr(signature).encode("utf-8")).hexdigest()[:20]
        candidate = CanonicalCandidate(
            f"canonical-closure:{stable}", family, anchor,
            tuple(core.node_ids), _region_node_ids(closure, node_by_id), closure,
            (alias,))
        signature_to_index[signature] = len(candidates)
        candidates.append(candidate)
    return tuple(candidates)


__all__ = [
    "CanonicalCandidate",
    "build_canonical_candidates",
    "closure_signature",
]
