from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from random import Random
from time import perf_counter

from Agent.Baseline.common import calculate_objective
from Agent.Proposed.dgrl_actions import (
    build_stratified_action_pool,
    project_action,
)
from Agent.Proposed.hypergraph import build_constraint_hypergraph
from Agent.Proposed.repair import repair_region
from Agent.Tool.Benchmark.formal_v3_loader import FormalV3InstanceLoader
from Agent.Tool.Training.chdgrl_config import load_chdgrl_experiment_config
from Mask.ServiceTuple import SearchState


@dataclass(frozen=True)
class ExpertActionLabel:
    action_id: str
    neighborhood_type: str
    seed_id: str
    release_profile: int
    time_budget: int
    time_limit_s: float
    projection_feasible: bool
    projection_reason: str
    effective_max_tasks: int
    effective_max_tuples: int
    effective_max_vehicles: int
    effective_max_span: int
    closure_task_count: int
    closure_tuple_count: int
    closure_vehicle_count: int
    closure_arc_count: int
    outcome_feasible: bool
    repair_accepted: bool
    strict_improvement: bool
    repair_reason: str
    validation_ok: bool
    primary_gain: float
    secondary_gain: float
    projection_runtime_s: float
    solver_runtime_s: float
    total_runtime_s: float
    primary_status: str
    secondary_status: str
    primary_bound: float
    secondary_bound: float
    repair_solver_backend: str = "CBC"
    repair_node_limit: int | None = None
    solver_nodes: int = 0
    repair_budget_fraction: float | None = None


@dataclass(frozen=True)
class ExpertActionBundle:
    instance_id: str
    instance_hash: str
    split: str
    state_index: int
    state_hash: str
    selected_arc_ids: tuple[str, ...]
    selected_tuple_ids: tuple[str, ...]
    initial_primary: float
    initial_secondary: float
    labels: tuple[ExpertActionLabel, ...]
    ordered_pairs: tuple[tuple[str, str], ...]


def _action_id(action):
    return (
        f"{action.neighborhood_type}|{action.seed_id}|"
        f"r{action.release_profile}|t{action.time_budget}"
    )


def _state_hash(instance_hash, state_index, state):
    payload = (
        str(instance_hash),
        int(state_index),
        tuple(sorted(state.selected_arc_ids)),
        tuple(sorted(state.selected_tuple_ids)),
    )
    return sha256(repr(payload).encode("utf-8")).hexdigest()


def _stable_seed(seed, instance_id, state_index):
    payload = f"{int(seed)}|{instance_id}|{int(state_index)}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def compare_expert_action_labels(
    left,
    right,
    *,
    epsilon_u=1e-8,
    primary_relative_tolerance=0.0,
    epsilon_f=1e-8,
    runtime_epsilon_s=None,
):
    if float(primary_relative_tolerance) < 0:
        raise ValueError("primary_relative_tolerance must be nonnegative")
    if left.projection_feasible != right.projection_feasible:
        return 1 if left.projection_feasible else -1
    if not left.projection_feasible:
        return 0
    if left.outcome_feasible != right.outcome_feasible:
        return 1 if left.outcome_feasible else -1
    if not left.outcome_feasible:
        return 0
    primary_tolerance = float(epsilon_u) + float(
        primary_relative_tolerance
    ) * max(1.0, abs(float(left.primary_gain)), abs(float(right.primary_gain)))
    if left.primary_gain > right.primary_gain + primary_tolerance:
        return 1
    if right.primary_gain > left.primary_gain + primary_tolerance:
        return -1
    if left.secondary_gain > right.secondary_gain + float(epsilon_f):
        return 1
    if right.secondary_gain > left.secondary_gain + float(epsilon_f):
        return -1
    if left.repair_accepted != right.repair_accepted:
        return 1 if left.repair_accepted else -1
    if runtime_epsilon_s is not None:
        if left.total_runtime_s + float(runtime_epsilon_s) < right.total_runtime_s:
            return 1
        if right.total_runtime_s + float(runtime_epsilon_s) < left.total_runtime_s:
            return -1
    return 0


def _label_expert_actions_and_repairs(
    instance,
    config,
    *,
    max_candidates,
    seed,
    time_indices=None,
    search_state=None,
    state_index=0,
    epsilon_u=1e-8,
    primary_relative_tolerance=0.0,
    epsilon_f=1e-8,
    runtime_epsilon_s=None,
    repair_solver_backend="CBC",
    repair_node_budgets=(),
):
    if float(primary_relative_tolerance) < 0:
        raise ValueError("primary_relative_tolerance must be nonnegative")
    repair_solver_backend = str(repair_solver_backend).upper()
    repair_node_budgets = tuple(int(value) for value in repair_node_budgets)
    if repair_solver_backend not in {"CBC", "SCIP"}:
        raise ValueError("repair_solver_backend must be CBC or SCIP")
    if any(value <= 0 for value in repair_node_budgets):
        raise ValueError("repair_node_budgets must be positive")
    if repair_node_budgets and repair_solver_backend != "SCIP":
        raise ValueError("deterministic repair node budgets require SCIP")
    if repair_node_budgets and len(repair_node_budgets) != len(
        config.search.time_budgets
    ):
        raise ValueError("repair node budgets must match action time levels")
    state = search_state or instance.search_state
    graph = build_constraint_hypergraph(
        state,
        instance.task_ids,
        task_weights=instance.miss_penalty,
    )
    actions = build_stratified_action_pool(
        graph=graph,
        release_levels=len(config.search.release_profiles),
        time_levels=len(config.search.time_budgets),
        max_candidates=max_candidates,
        seed=_stable_seed(seed, instance.instance_id, state_index),
        time_indices=time_indices,
    )
    search_config = config.search.to_search_config()
    initial_primary, initial_secondary = calculate_objective(
        state,
        instance.task_ids,
        instance.miss_penalty,
        search_config,
        validation_state=instance.rolling_state,
    )
    adjacency = graph.adjacency()
    node_by_id = {node.node_id: node for node in graph.nodes}
    remaining_budget_s = max(config.search.time_budgets)
    labels = []
    repairs = {}
    for action in actions:
        action_id = _action_id(action)
        action_started = perf_counter()
        projection_started = perf_counter()
        projection = project_action(
            state,
            graph,
            action,
            config.search.release_profiles,
            time_budgets=config.search.time_budgets,
            remaining_budget_s=remaining_budget_s,
            locked_tuple_ids=instance.rolling_state.locked_tuple_ids,
            locked_arc_ids=instance.rolling_state.locked_arc_ids,
            adjacency=adjacency,
            node_by_id=node_by_id,
        )
        projection_runtime_s = perf_counter() - projection_started
        effective = projection.effective_closure_budget
        closure = projection.closure
        repaired = None
        repair_node_limit = (
            repair_node_budgets[action.time_budget]
            if repair_node_budgets
            else None
        )
        repair_budget_fraction = (
            float(repair_node_limit) / float(max(repair_node_budgets))
            if repair_node_limit is not None
            else None
        )
        if projection.accepted:
            repaired = repair_region(
                state,
                closure,
                instance.task_ids,
                instance.miss_penalty,
                instance.rolling_state,
                instance.operation_capacity,
                projection.time_limit_s,
                epsilon_u,
                epsilon_f,
                makespan_weight=search_config.makespan_weight,
                time_weight=search_config.time_weight,
                risk_weight=search_config.risk_weight,
                energy_weight=search_config.energy_weight,
                plan_change_weight=search_config.plan_change_weight,
                solver_backend=repair_solver_backend,
                node_limit=repair_node_limit,
            )
            repairs[action_id] = repaired
        total_runtime_s = perf_counter() - action_started
        outcome_feasible = bool(
            repaired is not None
            and repaired.validation.ok
            and repaired.primary_status in {"optimal", "feasible"}
        )
        primary_gain = (
            float(initial_primary - repaired.primary_value)
            if outcome_feasible
            else 0.0
        )
        secondary_gain = (
            float(initial_secondary - repaired.secondary_value)
            if outcome_feasible
            else 0.0
        )
        strict_improvement = bool(
            repaired is not None
            and repaired.accepted
            and (
                primary_gain > float(epsilon_u)
                or (
                    abs(primary_gain) <= float(epsilon_u)
                    and secondary_gain > float(epsilon_f)
                )
            )
        )
        labels.append(
            ExpertActionLabel(
                action_id,
                action.neighborhood_type,
                action.seed_id,
                action.release_profile,
                action.time_budget,
                float(projection.time_limit_s),
                bool(projection.accepted),
                str(projection.reason),
                int(effective.max_tasks) if effective is not None else 0,
                int(effective.max_tuples) if effective is not None else 0,
                int(effective.max_vehicles) if effective is not None else 0,
                int(effective.max_span) if effective is not None else 0,
                len(closure.task_ids) if closure is not None else 0,
                len(closure.tuple_ids) if closure is not None else 0,
                len(closure.vehicle_ids) if closure is not None else 0,
                len(closure.arc_ids) if closure is not None else 0,
                outcome_feasible,
                bool(repaired.accepted) if repaired is not None else False,
                strict_improvement,
                str(repaired.reason) if repaired is not None else "not_attempted",
                bool(repaired.validation.ok) if repaired is not None else False,
                primary_gain,
                secondary_gain,
                projection_runtime_s,
                float(repaired.runtime_s) if repaired is not None else 0.0,
                total_runtime_s,
                str(repaired.primary_status) if repaired is not None else "not_solved",
                str(repaired.secondary_status) if repaired is not None else "not_solved",
                float(repaired.primary_bound)
                if repaired is not None
                else float("inf"),
                float(repaired.secondary_bound)
                if repaired is not None
                else float("inf"),
                repair_solver_backend,
                repair_node_limit,
                int(repaired.solver_nodes) if repaired is not None else 0,
                repair_budget_fraction,
            )
        )
    labels = tuple(labels)
    ordered_pairs = []
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            if left.time_budget != right.time_budget:
                continue
            comparison = compare_expert_action_labels(
                left,
                right,
                epsilon_u=epsilon_u,
                primary_relative_tolerance=primary_relative_tolerance,
                epsilon_f=epsilon_f,
                runtime_epsilon_s=runtime_epsilon_s,
            )
            if comparison > 0:
                ordered_pairs.append((left.action_id, right.action_id))
            elif comparison < 0:
                ordered_pairs.append((right.action_id, left.action_id))
    bundle = ExpertActionBundle(
        instance.instance_id,
        instance.instance_hash,
        instance.split,
        int(state_index),
        _state_hash(instance.instance_hash, state_index, state),
        tuple(sorted(state.selected_arc_ids)),
        tuple(sorted(state.selected_tuple_ids)),
        float(initial_primary),
        float(initial_secondary),
        labels,
        tuple(sorted(ordered_pairs)),
    )
    return bundle, repairs


def label_expert_actions(
    instance,
    config,
    *,
    max_candidates,
    seed,
    time_indices=None,
    search_state=None,
    state_index=0,
    epsilon_u=1e-8,
    primary_relative_tolerance=0.0,
    epsilon_f=1e-8,
    runtime_epsilon_s=None,
    repair_solver_backend="CBC",
    repair_node_budgets=(),
):
    bundle, _ = _label_expert_actions_and_repairs(
        instance,
        config,
        max_candidates=max_candidates,
        seed=seed,
        time_indices=time_indices,
        search_state=search_state,
        state_index=state_index,
        epsilon_u=epsilon_u,
        primary_relative_tolerance=primary_relative_tolerance,
        epsilon_f=epsilon_f,
        runtime_epsilon_s=runtime_epsilon_s,
        repair_solver_backend=repair_solver_backend,
        repair_node_budgets=repair_node_budgets,
    )
    return bundle


def generate_expert_action_trajectory(
    instance,
    config,
    *,
    max_states,
    max_candidates,
    seed,
    time_indices=None,
    epsilon_u=1e-8,
    primary_relative_tolerance=0.0,
    epsilon_f=1e-8,
    repair_solver_backend="CBC",
    repair_node_budgets=(),
):
    if max_states <= 0:
        raise ValueError("max_states must be positive")
    state = instance.search_state
    bundles = []
    seen = set()
    for state_index in range(int(max_states)):
        signature = (
            tuple(sorted(state.selected_arc_ids)),
            tuple(sorted(state.selected_tuple_ids)),
        )
        if signature in seen:
            break
        seen.add(signature)
        bundle, repairs = _label_expert_actions_and_repairs(
            instance,
            config,
            max_candidates=max_candidates,
            seed=seed,
            time_indices=time_indices,
            search_state=state,
            state_index=state_index,
            epsilon_u=epsilon_u,
            primary_relative_tolerance=primary_relative_tolerance,
            epsilon_f=epsilon_f,
            repair_solver_backend=repair_solver_backend,
            repair_node_budgets=repair_node_budgets,
        )
        bundles.append(bundle)
        improving = [
            label
            for label in bundle.labels
            if label.strict_improvement and label.action_id in repairs
        ]
        if not improving:
            break
        best = improving[0]
        for candidate in improving[1:]:
            comparison = compare_expert_action_labels(
                candidate,
                best,
                epsilon_u=epsilon_u,
                primary_relative_tolerance=primary_relative_tolerance,
                epsilon_f=epsilon_f,
            )
            if comparison > 0 or (
                comparison == 0 and candidate.action_id < best.action_id
            ):
                best = candidate
        repaired = repairs[best.action_id]
        state = SearchState(
            state.arcs,
            state.tuples,
            frozenset(repaired.selected_arc_ids),
            frozenset(repaired.selected_tuple_ids),
        )
    return tuple(bundles)


def _bundle_to_dict(bundle):
    row = asdict(bundle)
    labels = []
    for label in row["labels"]:
        for key in ("primary_bound", "secondary_bound"):
            if not isfinite(label[key]):
                label[key] = None
        labels.append(label)
    row["labels"] = labels
    return row


def _bundle_from_dict(row):
    labels = []
    for item in row["labels"]:
        item = dict(item)
        item.setdefault("repair_solver_backend", "CBC")
        item.setdefault("repair_node_limit", None)
        item.setdefault("solver_nodes", 0)
        item.setdefault("repair_budget_fraction", None)
        for key in ("primary_bound", "secondary_bound"):
            if item[key] is None:
                item[key] = float("inf")
        labels.append(ExpertActionLabel(**item))
    return ExpertActionBundle(
        row["instance_id"],
        row["instance_hash"],
        row["split"],
        int(row["state_index"]),
        row["state_hash"],
        tuple(row["selected_arc_ids"]),
        tuple(row["selected_tuple_ids"]),
        float(row["initial_primary"]),
        float(row["initial_secondary"]),
        tuple(labels),
        tuple(tuple(pair) for pair in row["ordered_pairs"]),
    )


def read_expert_action_bundles(path):
    path = Path(path)
    if not path.exists():
        return ()
    return tuple(
        _bundle_from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def write_expert_action_bundles(path, bundles, *, resume=False):
    path = Path(path)
    existing = read_expert_action_bundles(path) if resume else ()
    by_state = {bundle.state_hash: bundle for bundle in existing}
    for bundle in bundles:
        by_state.setdefault(bundle.state_hash, bundle)
    ordered = sorted(
        by_state.values(),
        key=lambda item: (item.instance_id, item.state_index, item.state_hash),
    )
    text = "".join(
        json.dumps(_bundle_to_dict(bundle), sort_keys=True) + "\n"
        for bundle in ordered
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return tuple(ordered)


def expert_action_semantic_sha256(bundles):
    normalized = []
    for bundle in sorted(
        tuple(bundles),
        key=lambda item: (item.instance_id, item.state_index, item.state_hash),
    ):
        row = _bundle_to_dict(bundle)
        for label in row["labels"]:
            for field in (
                "projection_runtime_s",
                "solver_runtime_s",
                "total_runtime_s",
                "solver_nodes",
                "primary_bound",
                "secondary_bound",
            ):
                label.pop(field, None)
        normalized.append(row)
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in normalized
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def summarize_expert_action_bundles(bundles):
    bundles = tuple(bundles)
    labels = tuple(label for bundle in bundles for label in bundle.labels)
    return {
        "state_count": len(bundles),
        "action_count": len(labels),
        "projection_feasible_count": sum(label.projection_feasible for label in labels),
        "outcome_feasible_count": sum(label.outcome_feasible for label in labels),
        "strict_improvement_count": sum(label.strict_improvement for label in labels),
        "positive_primary_count": sum(label.primary_gain > 1e-8 for label in labels),
        "ordered_pair_count": sum(len(bundle.ordered_pairs) for bundle in bundles),
        "zero_improvement_state_count": sum(
            not any(label.strict_improvement for label in bundle.labels)
            for bundle in bundles
        ),
        "test_instances_used": sum(bundle.split == "test" for bundle in bundles),
    }


def _stratified_instance_selection(loader, instance_ids, limit, seed):
    groups = {}
    for instance_id in instance_ids:
        row = loader.row_for(instance_id)
        key = (row.split, row.difficulty, row.topology_family)
        groups.setdefault(key, []).append(instance_id)
    rng = Random(int(seed))
    remaining_keys = set(groups)
    tie_order = {key: rng.random() for key in sorted(remaining_keys)}
    keys = []
    seen_splits = set()
    seen_difficulties = set()
    seen_topologies = set()
    while remaining_keys:
        key = max(
            remaining_keys,
            key=lambda item: (
                int(item[0] not in seen_splits)
                + int(item[1] not in seen_difficulties)
                + int(item[2] not in seen_topologies),
                int(item[1] not in seen_difficulties),
                int(item[2] not in seen_topologies),
                tie_order[item],
            ),
        )
        keys.append(key)
        remaining_keys.remove(key)
        seen_splits.add(key[0])
        seen_difficulties.add(key[1])
        seen_topologies.add(key[2])
    for values in groups.values():
        values.sort()
        rng.shuffle(values)
    selected = []
    while len(selected) < int(limit):
        added = False
        for key in keys:
            values = groups[key]
            if values:
                selected.append(values.pop())
                added = True
                if len(selected) >= int(limit):
                    break
        if not added:
            break
    return tuple(selected)


def generate_expert_action_dataset(
    manifest_path,
    config_path,
    output_path,
    *,
    instance_ids=(),
    splits=("train",),
    max_instances=None,
    max_candidates=12,
    max_states=1,
    seed=20260825,
    time_indices=None,
    repair_solver_backend="CBC",
    repair_node_budgets=(),
    primary_relative_tolerance=0.0,
    resume=False,
):
    if tuple(repair_node_budgets) and os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "deterministic repair node budgets require PYTHONHASHSEED=0 "
            "before the Python process starts"
        )
    loader = FormalV3InstanceLoader(manifest_path)
    allowed_splits = frozenset({"train", "validation"})
    requested_splits = frozenset(str(item) for item in splits)
    if not requested_splits or not requested_splits <= allowed_splits:
        raise ValueError("expert labels may use only train or validation splits")
    explicit_selection = bool(instance_ids)
    selected = tuple(
        dict.fromkeys(instance_ids)
        if instance_ids
        else (
            instance_id
            for instance_id in loader.instance_ids
            if loader.split_for(instance_id) in requested_splits
        )
    )
    if any(instance_id not in loader.instance_ids for instance_id in selected):
        raise ValueError("expert label instance selection is invalid")
    if any(loader.split_for(instance_id) not in requested_splits for instance_id in selected):
        raise ValueError("expert label instance is outside requested splits")
    if max_instances is not None:
        if int(max_instances) <= 0:
            raise ValueError("max_instances must be positive")
        selected = (
            selected[: int(max_instances)]
            if explicit_selection
            else _stratified_instance_selection(
                loader,
                selected,
                int(max_instances),
                seed,
            )
        )
    if not selected:
        raise ValueError("expert label instance selection is empty")
    if int(max_states) <= 0:
        raise ValueError("max_states must be positive")
    primary_relative_tolerance = float(primary_relative_tolerance)
    if primary_relative_tolerance < 0:
        raise ValueError("primary_relative_tolerance must be nonnegative")
    repair_solver_backend = str(repair_solver_backend).upper()
    repair_node_budgets = tuple(int(value) for value in repair_node_budgets)
    if repair_solver_backend not in {"CBC", "SCIP"}:
        raise ValueError("repair_solver_backend must be CBC or SCIP")
    if any(value <= 0 for value in repair_node_budgets):
        raise ValueError("repair_node_budgets must be positive")
    if repair_node_budgets and repair_solver_backend != "SCIP":
        raise ValueError("deterministic repair node budgets require SCIP")
    output_path = Path(output_path)
    if output_path.exists() and not resume:
        raise FileExistsError(output_path)
    config = load_chdgrl_experiment_config(config_path)
    if repair_node_budgets and len(repair_node_budgets) != len(
        config.search.time_budgets
    ):
        raise ValueError("repair node budgets must match action time levels")
    existing = read_expert_action_bundles(output_path) if resume else ()
    completed = {bundle.instance_id for bundle in existing}
    bundles = list(existing)
    for instance_id in selected:
        if instance_id in completed:
            continue
        bundles.extend(
            generate_expert_action_trajectory(
                loader(instance_id),
                config,
                max_states=max_states,
                max_candidates=max_candidates,
                seed=seed,
                time_indices=time_indices,
                repair_solver_backend=repair_solver_backend,
                repair_node_budgets=repair_node_budgets,
                primary_relative_tolerance=primary_relative_tolerance,
            )
        )
        bundles = list(
            write_expert_action_bundles(output_path, bundles, resume=False)
        )
    return tuple(bundles), summarize_expert_action_bundles(bundles)


__all__ = [
    "ExpertActionBundle",
    "ExpertActionLabel",
    "compare_expert_action_labels",
    "expert_action_semantic_sha256",
    "generate_expert_action_trajectory",
    "generate_expert_action_dataset",
    "label_expert_actions",
    "read_expert_action_bundles",
    "summarize_expert_action_bundles",
    "write_expert_action_bundles",
]
