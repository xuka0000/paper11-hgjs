from collections import deque
from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True, order=True)
class RegionBudget:
    max_tasks: int
    max_tuples: int
    max_vehicles: int
    max_span: int


@dataclass(frozen=True)
class CoreRegion:
    region_id: str
    seed_id: str
    node_ids: tuple[str, ...]
    task_count: int
    tuple_count: int
    vehicle_count: int
    time_span: int


def _region_counts(node_ids, node_by_id):
    selected = [node_by_id[node_id] for node_id in node_ids]
    task_count = sum(node.node_type == "task" for node in selected)
    tuple_count = sum(node.node_type == "tuple" for node in selected)
    vehicle_ids = {node.vehicle_id for node in selected if node.vehicle_id is not None}
    layers = [node.time_layer for node in selected if node.time_layer is not None]
    time_span = max(layers) - min(layers) if layers else 0
    return task_count, tuple_count, len(vehicle_ids), time_span


def _within(counts, budget):
    task_count, tuple_count, vehicle_count, time_span = counts
    return (
        task_count <= budget.max_tasks
        and tuple_count <= budget.max_tuples
        and vehicle_count <= budget.max_vehicles
        and time_span <= budget.max_span
    )


def generate_regions(
    graph,
    seed_ids,
    budgets,
    *,
    adjacency=None,
    node_by_id=None,
):
    adjacency = graph.adjacency() if adjacency is None else adjacency
    node_by_id = (
        {node.node_id: node for node in graph.nodes}
        if node_by_id is None
        else node_by_id
    )
    regions = []
    seen = set()
    for seed in sorted(seed_ids):
        if seed not in node_by_id:
            continue
        for budget in sorted(budgets):
            accepted = []
            examined = set()
            queued = {seed}
            queue = deque([seed])
            task_count = 0
            tuple_count = 0
            vehicle_ids = set()
            minimum_layer = None
            maximum_layer = None
            while queue:
                node_id = queue.popleft()
                queued.discard(node_id)
                if node_id in examined:
                    continue
                examined.add(node_id)
                node = node_by_id[node_id]
                next_task_count = task_count + int(node.node_type == "task")
                next_tuple_count = tuple_count + int(node.node_type == "tuple")
                next_vehicle_ids = vehicle_ids
                if node.vehicle_id is not None and node.vehicle_id not in vehicle_ids:
                    next_vehicle_ids = vehicle_ids | {node.vehicle_id}
                next_minimum_layer = minimum_layer
                next_maximum_layer = maximum_layer
                if node.time_layer is not None:
                    next_minimum_layer = (
                        node.time_layer
                        if minimum_layer is None
                        else min(minimum_layer, node.time_layer)
                    )
                    next_maximum_layer = (
                        node.time_layer
                        if maximum_layer is None
                        else max(maximum_layer, node.time_layer)
                    )
                next_span = (
                    next_maximum_layer - next_minimum_layer
                    if next_minimum_layer is not None
                    else 0
                )
                counts = (
                    next_task_count,
                    next_tuple_count,
                    len(next_vehicle_ids),
                    next_span,
                )
                if _within(counts, budget):
                    accepted.append(node_id)
                    task_count = next_task_count
                    tuple_count = next_tuple_count
                    vehicle_ids = next_vehicle_ids
                    minimum_layer = next_minimum_layer
                    maximum_layer = next_maximum_layer
                    for neighbor in adjacency[node_id]:
                        if neighbor not in examined and neighbor not in queued:
                            queue.append(neighbor)
                            queued.add(neighbor)
            signature = tuple(sorted(accepted))
            if not signature or signature in seen:
                continue
            seen.add(signature)
            digest = sha256(repr(signature).encode("utf-8")).hexdigest()[:16]
            counts = _region_counts(signature, node_by_id)
            regions.append(CoreRegion(f"region:{digest}", seed, signature, *counts))
    return tuple(sorted(regions, key=lambda item: item.region_id))
