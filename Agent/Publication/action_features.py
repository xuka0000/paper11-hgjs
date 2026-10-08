"""Observed physical features for bounded action-candidate selection.

The extractor uses only the current search state, its constraint hypergraph,
and fixed objective weights.  It does not receive a target plan, target
objective, repair outcome, validation row, or test row.
"""
from collections import defaultdict

import numpy as np


FEATURE_NAMES = (
    "selected",
    "priority",
    "time_layer",
    "degree",
    "uncovered_priority",
    "covered_priority",
    "replacement_saving",
    "selected_removal_burden",
    "direct_cost",
    "secondary_cost",
    "risk_cost",
    "energy_cost",
    "energy_ratio",
    "duration",
    "required_arc_support",
    "task_count",
    "adjacent_tuple_count",
    "adjacent_selected_tuple_count",
    "adjacent_uncovered_priority",
    "adjacent_replacement_saving",
    "adjacent_selected_removal_burden",
    "adjacent_direct_cost",
    "arc_secondary_cost",
    "arc_duty_time",
)


def _direct_tuple_cost(item, weights):
    return (
        float(item.secondary_cost)
        + float(weights["risk_weight"]) * float(item.risk_exposure)
        + float(weights["energy_weight"]) * float(item.expected_energy)
    )


def build_observable_action_features(state, graph, task_weights, objective_weights):
    """Return raw candidate features keyed by graph node identifier."""
    task_weights = {int(key): float(value) for key, value in task_weights.items()}
    node_by_id = {node.node_id: node for node in graph.nodes}
    adjacency = graph.adjacency()
    degree = {node_id: len(neighbors) for node_id, neighbors in adjacency.items()}
    tuple_by_id = {item.tuple_id: item for item in state.tuples}
    arc_by_node = {f"arc:{arc.arc_id}": arc for arc in state.arcs}
    selected_tuple_ids = set(state.selected_tuple_ids)
    selected_arc_ids = set(state.selected_arc_ids)
    selected_by_task = defaultdict(list)
    for item in state.tuples:
        if item.tuple_id in selected_tuple_ids:
            for task_id in item.task_ids:
                selected_by_task[int(task_id)].append(item)
    covered_tasks = set(selected_by_task)
    selected_per_task_cost = {}
    for task_id, items in selected_by_task.items():
        selected_per_task_cost[task_id] = min(
            _direct_tuple_cost(item, objective_weights) / max(1, len(item.task_ids))
            for item in items)

    tuple_features = {}
    for item in state.tuples:
        tasks = tuple(map(int, item.task_ids))
        selected = item.tuple_id in selected_tuple_ids
        direct_cost = _direct_tuple_cost(item, objective_weights)
        per_task_cost = direct_cost / max(1, len(tasks))
        uncovered_priority = sum(
            task_weights.get(task_id, 0.0)
            for task_id in tasks if task_id not in covered_tasks)
        covered_priority = sum(
            task_weights.get(task_id, 0.0)
            for task_id in tasks if task_id in covered_tasks)
        replacement_saving = sum(
            max(0.0, selected_per_task_cost.get(task_id, per_task_cost) - per_task_cost)
            for task_id in tasks if task_id in covered_tasks)
        redundant = sum(
            task_weights.get(task_id, 0.0)
            for task_id in tasks if len(selected_by_task.get(task_id, ())) > 1)
        total_priority = sum(task_weights.get(task_id, 0.0) for task_id in tasks)
        removal_fraction = redundant / total_priority if total_priority > 0.0 else 0.0
        required_support = (
            sum(arc_id in selected_arc_ids for arc_id in item.required_arc_ids)
            / len(item.required_arc_ids)
            if item.required_arc_ids else 1.0)
        usable_energy = float(item.battery_capacity) * max(
            0.0, 1.0 - float(item.reserve_ratio))
        energy_ratio = (
            float(item.reliable_energy) / usable_energy if usable_energy > 0.0 else 0.0)
        tuple_features[item.tuple_id] = {
            "uncovered_priority": uncovered_priority,
            "covered_priority": covered_priority,
            "replacement_saving": replacement_saving,
            "selected_removal_burden": direct_cost * removal_fraction if selected else 0.0,
            "direct_cost": direct_cost,
            "secondary_cost": float(item.secondary_cost),
            "risk_cost": float(objective_weights["risk_weight"]) * float(item.risk_exposure),
            "energy_cost": float(objective_weights["energy_weight"]) * float(item.expected_energy),
            "energy_ratio": energy_ratio,
            "duration": float(item.recovery_layer - item.launch_layer),
            "required_arc_support": required_support,
            "task_count": float(len(tasks)),
        }

    result = {}
    for node_id, node in node_by_id.items():
        row = {name: 0.0 for name in FEATURE_NAMES}
        row.update(
            selected=float(node.selected),
            priority=float(node.priority),
            time_layer=float(node.time_layer or 0),
            degree=float(degree[node_id]),
        )
        own = tuple_features.get(node_id)
        if own is not None:
            row.update(own)
        if node.node_type == "task":
            task_id = int(node.entity_id)
            row["uncovered_priority"] = (
                task_weights.get(task_id, 0.0) if task_id not in covered_tasks else 0.0)
            row["covered_priority"] = (
                task_weights.get(task_id, 0.0) if task_id in covered_tasks else 0.0)
        arc = arc_by_node.get(node_id)
        if arc is not None:
            row["selected"] = float(arc.arc_id in selected_arc_ids)
            row["arc_secondary_cost"] = float(arc.secondary_cost)
            row["arc_duty_time"] = float(arc.duty_time)
            row["direct_cost"] = (
                float(arc.secondary_cost)
                + float(objective_weights["time_weight"]) * float(arc.duty_time))
        adjacent_tuples = [
            tuple_features[neighbor]
            for neighbor in adjacency[node_id] if neighbor in tuple_features]
        adjacent_ids = [
            neighbor for neighbor in adjacency[node_id] if neighbor in tuple_features]
        row["adjacent_tuple_count"] = float(len(adjacent_tuples))
        row["adjacent_selected_tuple_count"] = float(sum(
            tuple_id in selected_tuple_ids for tuple_id in adjacent_ids))
        if adjacent_tuples:
            row["adjacent_uncovered_priority"] = max(
                item["uncovered_priority"] for item in adjacent_tuples)
            row["adjacent_replacement_saving"] = max(
                item["replacement_saving"] for item in adjacent_tuples)
            row["adjacent_selected_removal_burden"] = max(
                item["selected_removal_burden"] for item in adjacent_tuples)
            row["adjacent_direct_cost"] = max(
                item["direct_cost"] for item in adjacent_tuples)
        result[node_id] = row
    return result


def physical_opportunity_key(node, features):
    """Lexicographic, scale-free ordering for deterministic preselection."""
    row = features[node.node_id]
    direct_uncovered = row["uncovered_priority"]
    direct_saving = row["replacement_saving"]
    direct_removal = row["selected_removal_burden"]
    propagated_uncovered = row["adjacent_uncovered_priority"]
    propagated_saving = row["adjacent_replacement_saving"]
    propagated_removal = row["adjacent_selected_removal_burden"]
    return (
        -max(direct_uncovered, propagated_uncovered),
        -max(direct_saving, propagated_saving),
        -max(direct_removal, propagated_removal),
        -row["selected"],
        -row["priority"],
        -row["adjacent_selected_tuple_count"],
        -row["direct_cost"],
        row["time_layer"],
        node.node_id,
    )


def normalized_feature_matrix(graph, feature_rows, node_ids):
    """Min-max normalize observed features within one current state."""
    if not node_ids:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    matrix = np.asarray([
        [float(feature_rows[node_id][name]) for name in FEATURE_NAMES]
        for node_id in node_ids
    ], dtype=np.float64)
    minima = matrix.min(axis=0)
    maxima = matrix.max(axis=0)
    ranges = maxima - minima
    nonconstant = ranges > 0.0
    normalized = np.zeros_like(matrix)
    normalized[:, nonconstant] = (
        matrix[:, nonconstant] - minima[nonconstant]
    ) / ranges[nonconstant]
    return normalized


__all__ = [
    "FEATURE_NAMES",
    "build_observable_action_features",
    "normalized_feature_matrix",
    "physical_opportunity_key",
]
