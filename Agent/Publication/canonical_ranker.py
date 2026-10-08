"""Trainable hypergraph ranker for target-independent canonical closures."""
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from Agent.Proposed.hgnn import RoleAwareHGNN, tensorize_graph
from Agent.Publication.action_features import FEATURE_NAMES, normalized_feature_matrix


FAMILIES = ("task_exchange", "uav_resource", "vehicle_window")
CORE_AGGREGATES = (
    "uncovered_priority",
    "replacement_saving",
    "selected_removal_burden",
    "direct_cost",
    "energy_ratio",
    "required_arc_support",
    "adjacent_selected_tuple_count",
    "arc_secondary_cost",
    "arc_duty_time",
)
REGION_AGGREGATES = (
    "selected",
    "priority",
    "degree",
    "direct_cost",
)
DESCRIPTOR_NAMES = (
    tuple("family_" + family for family in FAMILIES)
    + ("closure_task_fraction", "closure_tuple_fraction",
       "closure_vehicle_fraction", "closure_uav_fraction",
       "closure_arc_fraction", "core_node_fraction", "region_node_fraction")
    + tuple("core_mean_" + name for name in CORE_AGGREGATES)
    + tuple("core_max_" + name for name in CORE_AGGREGATES)
    + tuple("region_mean_" + name for name in REGION_AGGREGATES)
)


def candidate_feature_cache(graph, feature_rows):
    node_ids = tuple(node.node_id for node in graph.nodes)
    matrix = np.asarray(
        normalized_feature_matrix(graph, feature_rows, node_ids),
        dtype=np.float64,
    )
    if matrix.shape != (len(node_ids), len(FEATURE_NAMES)):
        raise ValueError("Normalized feature matrix has an invalid shape")
    return {
        "node_ids": node_ids,
        "node_index": {
            node_id: index for index, node_id in enumerate(node_ids)},
        "matrix": matrix,
        "feature_index": {
            name: index for index, name in enumerate(FEATURE_NAMES)
        },
        "uav_total": max(
            1,
            len({
                node.entity_id for node in graph.nodes
                if node.node_type == "uav"
            }),
        ),
        "arc_total": max(
            1, sum(node.node_type == "arc" for node in graph.nodes)),
    }


def tensorize_observable_graph(graph, feature_rows, feature_cache=None):
    base = tensorize_graph(graph)
    cache = feature_cache or candidate_feature_cache(graph, feature_rows)
    if tuple(base.node_ids) != tuple(cache["node_ids"]):
        raise ValueError("Cached node order differs from the tensor graph")
    extra = torch.tensor(
        cache["matrix"],
        dtype=base.node_features.dtype,
    )
    if extra.shape != (len(base.node_ids), len(FEATURE_NAMES)):
        raise ValueError("Observable feature matrix has an invalid shape")
    return replace(base, node_features=torch.cat((base.node_features, extra), dim=1))


def _aggregate(indices, matrix, feature_index, names, reducer):
    if not indices:
        return [0.0] * len(names)
    columns = np.asarray([feature_index[name] for name in names], dtype=int)
    values = matrix[np.ix_(np.asarray(indices, dtype=int), columns)]
    if reducer == "mean":
        return values.mean(axis=0).tolist()
    if reducer == "max":
        return values.max(axis=0).tolist()
    raise ValueError("unknown aggregate reducer")


def candidate_tensor(
    candidate,
    graph,
    feature_rows,
    closure_budget,
    feature_cache=None,
):
    cache = feature_cache or candidate_feature_cache(graph, feature_rows)
    node_ids = cache["node_ids"]
    node_index = cache["node_index"]
    matrix = cache["matrix"]
    feature_index = cache["feature_index"]
    core_ids = tuple(candidate.core_node_ids)
    region_ids = tuple(candidate.region_node_ids)
    if any(node_id not in node_index for node_id in core_ids + region_ids):
        raise ValueError("Canonical candidate references an unknown graph node")
    if not core_ids or not region_ids:
        raise ValueError("Canonical candidate has an empty tensor region")
    closure = candidate.closure
    uav_total = cache["uav_total"]
    arc_total = cache["arc_total"]
    family = [float(candidate.family == name) for name in FAMILIES]
    sizes = [
        len(closure.task_ids) / max(1, closure_budget.max_tasks),
        len(closure.tuple_ids) / max(1, closure_budget.max_tuples),
        len(closure.vehicle_ids) / max(1, closure_budget.max_vehicles),
        len(closure.uav_ids) / uav_total,
        len(closure.arc_ids) / arc_total,
        len(core_ids) / len(node_ids),
        len(region_ids) / len(node_ids),
    ]
    core_indices = [node_index[node_id] for node_id in core_ids]
    region_indices = [node_index[node_id] for node_id in region_ids]
    core_mean = _aggregate(
        core_indices, matrix, feature_index, CORE_AGGREGATES, "mean"
    )
    core_max = _aggregate(
        core_indices, matrix, feature_index, CORE_AGGREGATES, "max"
    )
    region_mean = _aggregate(
        region_indices, matrix, feature_index, REGION_AGGREGATES, "mean"
    )
    descriptor = family + sizes + core_mean + core_max + region_mean
    if len(descriptor) != len(DESCRIPTOR_NAMES):
        raise ValueError("Canonical descriptor dimension changed")
    opportunity = (
        core_max[CORE_AGGREGATES.index("uncovered_priority")]
        + core_max[CORE_AGGREGATES.index("replacement_saving")]
        + core_max[CORE_AGGREGATES.index("selected_removal_burden")]
    )
    return {
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "anchor_id": candidate.anchor_id,
        "core_indices": torch.tensor(core_indices, dtype=torch.long),
        "region_indices": torch.tensor(region_indices, dtype=torch.long),
        "descriptor": torch.tensor(descriptor, dtype=torch.float32),
        "heuristic_key": (
            float(opportunity),
            float(core_max[CORE_AGGREGATES.index("required_arc_support")]),
            -float(sizes[6]),
            candidate.candidate_id,
        ),
    }


def heuristic_candidate_index(candidates):
    if not candidates:
        raise ValueError("Candidate list is empty")
    return max(range(len(candidates)), key=lambda index: candidates[index]["heuristic_key"])


class CanonicalClosureRanker(nn.Module):
    def __init__(self, feature_dim, descriptor_dim, hidden_dim=24, layers=1):
        super().__init__()
        self.encoder = RoleAwareHGNN(feature_dim, hidden_dim, layers)
        self.scorer = nn.Sequential(
            nn.Linear(3 * hidden_dim + descriptor_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 1),
        )

    def forward(self, graph, candidates):
        if not candidates:
            raise ValueError("Candidate list is empty")
        node_state = self.encoder.encode(graph)
        global_state = node_state.mean(dim=0)
        pooled = []
        for candidate in candidates:
            core = candidate["core_indices"].to(node_state.device)
            region = candidate["region_indices"].to(node_state.device)
            descriptor = candidate["descriptor"].to(node_state.device)
            pooled.append(torch.cat((
                global_state,
                node_state[core].mean(dim=0),
                node_state[region].mean(dim=0),
                descriptor,
            )))
        return self.scorer(torch.stack(pooled)).squeeze(-1)


class DescriptorRanker(nn.Module):
    def __init__(self, descriptor_dim, hidden_dim=24):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(descriptor_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 1),
        )

    def forward(self, graph, candidates):
        del graph
        if not candidates:
            raise ValueError("Candidate list is empty")
        descriptors = torch.stack([candidate["descriptor"] for candidate in candidates])
        return self.scorer(descriptors).squeeze(-1)


__all__ = [
    "CORE_AGGREGATES",
    "DESCRIPTOR_NAMES",
    "FAMILIES",
    "CanonicalClosureRanker",
    "DescriptorRanker",
    "candidate_feature_cache",
    "candidate_tensor",
    "heuristic_candidate_index",
    "tensorize_observable_graph",
]
