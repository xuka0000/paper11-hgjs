from collections import deque
from dataclasses import dataclass, replace
from math import ceil
from random import Random

from .closure import ClosureBudget, close_region
from Agent.Proposed.regions import RegionBudget, generate_regions


NEIGHBORHOOD_TYPES = (
    "task",
    "tuple",
    "resource",
    "route_time",
)

SEED_NODE_TYPES = {
    "task": frozenset({"task"}),
    "tuple": frozenset({"tuple"}),
    "resource": frozenset(
        {"uav", "interval_resource", "operation_resource"}
    ),
    "route_time": frozenset({"arc", "time_node", "vehicle"}),
}


@dataclass(frozen=True, order=True)
class JointAction:
    neighborhood_type: str
    seed_id: str
    release_profile: int
    time_budget: int

    def __post_init__(self):
        if self.neighborhood_type not in NEIGHBORHOOD_TYPES:
            raise ValueError("unknown neighborhood_type")
        if not str(self.seed_id).strip():
            raise ValueError("seed_id must not be empty")
        if self.release_profile < 0:
            raise ValueError("release_profile must be nonnegative")
        if self.time_budget < 0:
            raise ValueError("time_budget must be nonnegative")


@dataclass(frozen=True)
class RelativeClosureBudget:
    task_fraction: float
    tuple_fraction: float
    vehicle_fraction: float
    span_fraction: float

    def __post_init__(self):
        for name, value in (
            ("task_fraction", self.task_fraction),
            ("tuple_fraction", self.tuple_fraction),
            ("vehicle_fraction", self.vehicle_fraction),
            ("span_fraction", self.span_fraction),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass(frozen=True)
class ReleaseProfile:
    region_budget: RegionBudget
    closure_budget: ClosureBudget
    relative_closure_budget: RelativeClosureBudget | None = None


@dataclass(frozen=True)
class ActionProjection:
    accepted: bool
    reason: str
    action: JointAction
    region: object | None = None
    closure: object | None = None
    time_limit_s: float = 0.0
    effective_closure_budget: ClosureBudget | None = None


def resolve_closure_budget(graph, profile):
    relative = profile.relative_closure_budget
    if relative is None:
        return profile.closure_budget
    task_count = sum(node.node_type == "task" for node in graph.nodes)
    tuple_count = sum(node.node_type == "tuple" for node in graph.nodes)
    vehicle_count = sum(node.node_type == "vehicle" for node in graph.nodes)
    layers = [
        node.time_layer
        for node in graph.nodes
        if node.time_layer is not None
    ]
    horizon = max(layers) - min(layers) if layers else 0
    base = profile.closure_budget
    return ClosureBudget(
        max(base.max_tasks, ceil(relative.task_fraction * task_count)),
        max(base.max_tuples, ceil(relative.tuple_fraction * tuple_count)),
        max(
            base.max_vehicles,
            ceil(relative.vehicle_fraction * vehicle_count),
        ),
        max(base.max_span, ceil(relative.span_fraction * horizon)),
    )


def _shortest_distances(adjacency, source):
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node_id = queue.popleft()
        for neighbor in adjacency[node_id]:
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node_id] + 1
            queue.append(neighbor)
    return distances


def _approximate_graph_diameter(
    adjacency,
    *,
    initial_source=None,
    initial_distances=None,
):
    """Return a deterministic linear-time lower bound on graph diameter.

    A two-sweep traversal is exact on trees and avoids the previous all-source
    traversal.  Disconnected components are handled independently.
    """
    remaining = set(adjacency)
    diameter = 1

    def visit_component(source, distances=None):
        nonlocal diameter
        distances = distances or _shortest_distances(adjacency, source)
        remaining.difference_update(distances)
        eccentricity = max(distances.values(), default=0)
        farthest = min(
            node_id
            for node_id, distance in distances.items()
            if distance == eccentricity
        )
        opposite = _shortest_distances(adjacency, farthest)
        diameter = max(
            diameter,
            eccentricity,
            max(opposite.values(), default=0),
        )

    if initial_source in remaining:
        visit_component(initial_source, initial_distances)
    while remaining:
        visit_component(min(remaining))
    return diameter


def _normalized_seed_distance(graph, left_seed, right_seed):
    adjacency = graph.adjacency()
    if left_seed not in adjacency or right_seed not in adjacency:
        raise ValueError("action seed is not present in the hypergraph")
    if left_seed == right_seed:
        return 0.0
    distances = _shortest_distances(adjacency, left_seed)
    if right_seed not in distances:
        return 1.0
    diameter = _approximate_graph_diameter(
        adjacency,
        initial_source=left_seed,
        initial_distances=distances,
    )
    return min(1.0, float(distances[right_seed]) / max(1, diameter))


def _ordinal_distance(left, right, levels, field_name):
    if levels <= 0:
        raise ValueError(f"{field_name} levels must be positive")
    if left >= levels or right >= levels:
        raise ValueError(f"{field_name} index exceeds configured levels")
    return abs(left - right) / max(1, levels - 1)


def mixed_action_distance(
    left,
    right,
    graph,
    *,
    release_levels,
    time_levels,
):
    return max(
        float(left.neighborhood_type != right.neighborhood_type),
        _normalized_seed_distance(graph, left.seed_id, right.seed_id),
        _ordinal_distance(
            left.release_profile,
            right.release_profile,
            release_levels,
            "release_profile",
        ),
        _ordinal_distance(
            left.time_budget,
            right.time_budget,
            time_levels,
            "time_budget",
        ),
    )


def _weighted_choice(rng, options, distances, radius):
    weights = [max(1e-6, radius - distance + 1e-3) for distance in distances]
    return rng.choices(options, weights=weights, k=1)[0]


def sample_dynamic_neighborhood(
    graph,
    center,
    *,
    release_levels,
    time_levels,
    sample_count,
    radius,
    rng,
    feasibility=None,
    adjacency=None,
    node_by_id=None,
    stop_requested=None,
):
    if sample_count <= 0:
        return ()
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    node_by_id = (
        {node.node_id: node for node in graph.nodes}
        if node_by_id is None
        else node_by_id
    )
    if center.seed_id not in node_by_id:
        raise ValueError("center seed is not present in the hypergraph")
    adjacency = graph.adjacency() if adjacency is None else adjacency
    center_distances = _shortest_distances(adjacency, center.seed_id)
    diameter = _approximate_graph_diameter(
        adjacency,
        initial_source=center.seed_id,
        initial_distances=center_distances,
    )

    def cached_seed_distance(seed_id):
        if seed_id == center.seed_id:
            return 0.0
        if seed_id not in center_distances:
            return 1.0
        return min(
            1.0,
            float(center_distances[seed_id]) / max(1, diameter),
        )

    def cached_action_distance(action):
        return max(
            float(action.neighborhood_type != center.neighborhood_type),
            cached_seed_distance(action.seed_id),
            _ordinal_distance(
                center.release_profile,
                action.release_profile,
                release_levels,
                "release_profile",
            ),
            _ordinal_distance(
                center.time_budget,
                action.time_budget,
                time_levels,
                "time_budget",
            ),
        )

    type_options = [
        name
        for name in NEIGHBORHOOD_TYPES
        if float(name != center.neighborhood_type) <= radius
    ]
    release_options = [
        index
        for index in range(release_levels)
        if _ordinal_distance(
            center.release_profile,
            index,
            release_levels,
            "release_profile",
        )
        <= radius
    ]
    time_options = [
        index
        for index in range(time_levels)
        if _ordinal_distance(
            center.time_budget,
            index,
            time_levels,
            "time_budget",
        )
        <= radius
    ]
    seed_options = {}
    seed_distances = {}
    for neighborhood_type in type_options:
        candidates = [
            node.node_id
            for node in graph.nodes
            if node.node_type in SEED_NODE_TYPES[neighborhood_type]
        ]
        distances = [
            cached_seed_distance(seed_id)
            for seed_id in candidates
        ]
        seed_options[neighborhood_type] = [
            seed_id
            for seed_id, distance in zip(candidates, distances)
            if distance <= radius
        ]
        seed_distances[neighborhood_type] = [
            distance for distance in distances if distance <= radius
        ]

    selected = []
    selected_set = set()
    examined_set = set()

    def accept(action):
        if action in examined_set:
            return
        examined_set.add(action)
        if cached_action_distance(action) > radius:
            return
        if feasibility is not None and not feasibility(action):
            return
        selected.append(action)
        selected_set.add(action)

    if stop_requested is not None and stop_requested():
        return ()
    accept(center)
    max_attempts = max(20, sample_count * 4)
    attempts = 0
    while len(selected) < sample_count and attempts < max_attempts:
        if stop_requested is not None and stop_requested():
            break
        attempts += 1
        neighborhood_type = _weighted_choice(
            rng,
            type_options,
            [float(name != center.neighborhood_type) for name in type_options],
            radius,
        )
        candidates = seed_options[neighborhood_type]
        if not candidates:
            continue
        seed_id = _weighted_choice(
            rng,
            candidates,
            seed_distances[neighborhood_type],
            radius,
        )
        release_profile = _weighted_choice(
            rng,
            release_options,
            [
                _ordinal_distance(
                    center.release_profile,
                    index,
                    release_levels,
                    "release_profile",
                )
                for index in release_options
            ],
            radius,
        )
        time_budget = _weighted_choice(
            rng,
            time_options,
            [
                _ordinal_distance(
                    center.time_budget,
                    index,
                    time_levels,
                    "time_budget",
                )
                for index in time_options
            ],
            radius,
        )
        accept(
            JointAction(
                neighborhood_type,
                seed_id,
                release_profile,
                time_budget,
            )
        )
    return tuple(selected)


def project_action(
    state,
    graph,
    action,
    release_profiles,
    *,
    time_budgets,
    remaining_budget_s,
    locked_tuple_ids=frozenset(),
    locked_arc_ids=frozenset(),
    adjacency=None,
    node_by_id=None,
):
    if action.release_profile >= len(release_profiles):
        return ActionProjection(False, "unknown_release_profile", action)
    if action.time_budget >= len(time_budgets):
        return ActionProjection(False, "unknown_time_budget", action)
    time_limit_s = float(time_budgets[action.time_budget])
    if time_limit_s <= 0:
        return ActionProjection(False, "nonpositive_time_budget", action)
    if time_limit_s > float(remaining_budget_s):
        return ActionProjection(False, "time_budget_exceeded", action)
    node_by_id = (
        {node.node_id: node for node in graph.nodes}
        if node_by_id is None
        else node_by_id
    )
    seed = node_by_id.get(action.seed_id)
    if seed is None:
        return ActionProjection(False, "unknown_seed", action)
    if seed.node_type not in SEED_NODE_TYPES[action.neighborhood_type]:
        return ActionProjection(False, "seed_type_mismatch", action)

    profile = release_profiles[action.release_profile]
    regions = generate_regions(
        graph,
        (action.seed_id,),
        (profile.region_budget,),
        adjacency=adjacency,
        node_by_id=node_by_id,
    )
    if not regions:
        return ActionProjection(False, "no_candidate_region", action)
    region = regions[0]
    effective_closure_budget = resolve_closure_budget(graph, profile)
    closure = close_region(state, region, effective_closure_budget)
    if not closure.accepted:
        return ActionProjection(
            False,
            closure.reason,
            action,
            region,
            closure,
            time_limit_s,
            effective_closure_budget,
        )
    released = (
        set(closure.tuple_ids) - set(locked_tuple_ids)
    ) | (set(closure.arc_ids) - set(locked_arc_ids))
    if not released:
        return ActionProjection(
            False,
            "empty_release",
            action,
            region,
            closure,
            time_limit_s,
            effective_closure_budget,
        )
    return ActionProjection(
        True,
        "accepted",
        action,
        region,
        closure,
        time_limit_s,
        effective_closure_budget,
    )


def _iter_stratified_actions(graph,release_levels,time_levels,seed,time_indices=None):
    if release_levels <= 0 or time_levels <= 0:
        raise ValueError("action levels must be positive")
    selected_time_indices = tuple(
        range(int(time_levels))
        if time_indices is None
        else dict.fromkeys(int(index) for index in time_indices)
    )
    if not selected_time_indices:
        raise ValueError("time_indices must not be empty")
    if any(index < 0 or index >= int(time_levels) for index in selected_time_indices):
        raise ValueError("time index exceeds configured levels")
    rng=Random(int(seed));queues={};strata_by_type={name: [] for name in NEIGHBORHOOD_TYPES}
    for neighborhood_type in NEIGHBORHOOD_TYPES:
        seeds=sorted(node.node_id for node in graph.nodes if node.node_type in SEED_NODE_TYPES[neighborhood_type])
        if not seeds:continue
        for release_index in range(int(release_levels)):
            for time_index in selected_time_indices:
                key=(neighborhood_type,release_index,time_index);order=list(seeds);rng.shuffle(order)
                queues[key]=deque(order);strata_by_type[neighborhood_type].append(key)
        rng.shuffle(strata_by_type[neighborhood_type]);strata_by_type[neighborhood_type]=deque(strata_by_type[neighborhood_type])
    while True:
        added=False
        for neighborhood_type in NEIGHBORHOOD_TYPES:
            strata=strata_by_type[neighborhood_type]
            if not strata:continue
            checked=0
            while strata and checked<len(strata):
                key=strata.popleft();seeds=queues[key];checked+=1
                if seeds:
                    yield JointAction(neighborhood_type,seeds.popleft(),key[1],key[2]);added=True
                    if seeds:strata.append(key)
                    break
        if not added:return


def build_independent_dbu_action_pool(
    *,
    state,
    graph,
    release_profiles,
    time_budgets,
    remaining_budget_s,
    locked_tuple_ids=frozenset(),
    locked_arc_ids=frozenset(),
    max_candidates,
    seed,
    projection_for=None,
    stop_requested=None,
):
    """Build a feasible DBU pool without consulting the actor output."""
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    # Filter the established stratified order instead of drawing a replacement
    # order.  Every feasible member of the original prefix is retained, then
    # later candidates fill slots that rejected projections would have used.
    selected = []
    if projection_for is None:
        projection_cache = {}
        structural_projection_cache = {}
        adjacency = graph.adjacency()
        node_by_id = {node.node_id: node for node in graph.nodes}

        def projection_for(action):
            if action not in projection_cache:
                time_limit_s = float(time_budgets[action.time_budget])
                if time_limit_s > float(remaining_budget_s):
                    projection_cache[action] = ActionProjection(
                        False,
                        "time_budget_exceeded",
                        action,
                    )
                else:
                    structural_key = (
                        action.neighborhood_type,
                        action.seed_id,
                        action.release_profile,
                    )
                    if structural_key not in structural_projection_cache:
                        structural_projection_cache[structural_key] = project_action(
                            state,
                            graph,
                            action,
                            release_profiles,
                            time_budgets=time_budgets,
                            remaining_budget_s=remaining_budget_s,
                            locked_tuple_ids=locked_tuple_ids,
                            locked_arc_ids=locked_arc_ids,
                            adjacency=adjacency,
                            node_by_id=node_by_id,
                        )
                    projection_cache[action] = replace(
                        structural_projection_cache[structural_key],
                        action=action,
                        time_limit_s=time_limit_s,
                    )
            return projection_cache[action]

    for action in _iter_stratified_actions(graph,len(release_profiles),len(time_budgets),seed):
        if stop_requested is not None and stop_requested():
            break
        projection = projection_for(action)
        if projection.accepted:
            selected.append(action)
            if len(selected) >= int(max_candidates):
                break
    return tuple(selected)


def build_stratified_action_pool(
    *,
    graph,
    release_levels,
    time_levels,
    max_candidates,
    seed,
    time_indices=None,
):
    """Build a deterministic policy-free pool balanced across action types."""
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    selected=[]
    for action in _iter_stratified_actions(graph,release_levels,time_levels,seed,time_indices):
        selected.append(action)
        if len(selected)>=int(max_candidates):break
    return tuple(selected)


__all__ = [
    "ActionProjection",
    "JointAction",
    "NEIGHBORHOOD_TYPES",
    "RelativeClosureBudget",
    "ReleaseProfile",
    "build_independent_dbu_action_pool",
    "build_stratified_action_pool",
    "mixed_action_distance",
    "project_action",
    "resolve_closure_budget",
    "sample_dynamic_neighborhood",
]
