from dataclasses import dataclass, field, replace
from time import perf_counter

import torch

from .closure import ClosureBudget, close_region
from .hgnn import tensorize_graph
from .hypergraph import build_constraint_hypergraph
from .regions import RegionBudget, generate_regions
from .repair import repair_region
from Mask.ServiceTuple import SearchState


ACCEPTANCE_MODES = frozenset({"first_improvement", "best_of_top_k"})


@dataclass(frozen=True)
class SearchConfig:
    max_iterations: int
    no_improve_limit: int
    top_k: int
    region_budgets: tuple[RegionBudget, ...] = field(
        default_factory=lambda: (RegionBudget(4, 8, 2, 8),)
    )
    closure_budget: ClosureBudget = field(
        default_factory=lambda: ClosureBudget(6, 12, 3, 12)
    )
    repair_time_limit_s: float = 5.0
    repair_solver_backend: str = "CBC"
    repair_node_limit: int | None = None
    repair_node_budgets: tuple[int, ...] = ()
    epsilon_u: float = 1e-8
    epsilon_f: float = 1e-8
    deterministic_guard_count: int = 0
    acceptance_mode: str = "first_improvement"
    time_weight: float = 0.0
    risk_weight: float = 0.0
    energy_weight: float = 0.0
    makespan_weight: float = 0.0
    plan_change_weight: float = 0.0
    total_time_limit_s: float | None = None


@dataclass(frozen=True)
class SearchRecord:
    iteration: int
    region_id: str
    closure_accepted: bool
    repair_accepted: bool
    improvement_accepted: bool
    validation_ok: bool
    primary_value: float
    secondary_value: float
    reason: str
    strict_improvement: bool = False
    selected: bool = False
    neighborhood_type: str | None = None
    release_profile: int | None = None
    time_budget_s: float = 0.0
    used_fallback: bool = False


@dataclass(frozen=True)
class SearchResult:
    initial_primary: float
    initial_secondary: float
    best_primary: float
    best_secondary: float
    selected_tuple_ids: tuple[str, ...]
    selected_arc_ids: tuple[str, ...]
    trace: tuple[SearchRecord, ...]
    accepted_records: tuple[SearchRecord, ...]
    stop_reason: str


@dataclass(frozen=True)
class DGRLStepEvent:
    iteration: int
    graph: object
    context: torch.Tensor
    action: object
    state_before: SearchState
    state_after: SearchState
    primary_gain: float
    secondary_gain: float
    failure_cost: float
    runtime_cost: float
    remaining_budget_s: float
    used_fallback: bool


def _objective(
    state,
    task_ids,
    miss_penalty,
    config=None,
    previous_arc_values=None,
    previous_tuple_values=None,
):
    selected = [item for item in state.tuples if item.tuple_id in state.selected_tuple_ids]
    covered = {task for item in selected for task in item.task_ids}
    primary = sum(miss_penalty[task] for task in task_ids if task not in covered)
    secondary = sum(item.secondary_cost for item in selected)
    secondary += sum(
        arc.secondary_cost for arc in state.arcs if arc.arc_id in state.selected_arc_ids
    )
    if config is not None:
        secondary += config.time_weight * sum(
            arc.duty_time
            for arc in state.arcs
            if arc.arc_id in state.selected_arc_ids
        )
        secondary += config.risk_weight * sum(
            item.risk_exposure for item in selected
        )
        secondary += config.energy_weight * sum(
            item.expected_energy for item in selected
        )
        completion_layers = [
            layer
            for item in selected
            for _, layer in item.completion_layers
        ]
        return_layers = [
            arc.arrival_layer
            for arc in state.arcs
            if arc.arc_id in state.selected_arc_ids and arc.head == "sink"
        ]
        secondary += config.makespan_weight * max(
            completion_layers + return_layers,
            default=0,
        )
        plan_change = sum(
            abs(int(tuple_id in state.selected_tuple_ids) - int(value))
            for tuple_id, value in (previous_tuple_values or {}).items()
        )
        plan_change += sum(
            abs(int(arc_id in state.selected_arc_ids) - int(value))
            for arc_id, value in (previous_arc_values or {}).items()
        )
        secondary += config.plan_change_weight * plan_change
    return float(primary), float(secondary), covered


def baseline_region_scores(graph, regions):
    return tuple(
        100.0 * region.task_count
        + 10.0 * region.tuple_count
        + region.vehicle_count
        - 0.01 * region.time_span
        for region in regions
    )


def risk_aware_region_scores(graph, regions):
    """Rank regions by their two strongest unselected repair opportunities."""
    node_by_id = {node.node_id: node for node in graph.nodes}
    scores = []
    for region in regions:
        candidate_values = sorted(
            (
                node_by_id[node_id].priority
                * (1.0 - node_by_id[node_id].selected)
                for node_id in region.node_ids
                if node_id in node_by_id
                and node_by_id[node_id].node_type == "tuple"
            ),
            reverse=True,
        )
        uncovered_task_risk = sum(
            node_by_id[node_id].priority
            * (1.0 - node_by_id[node_id].selected)
            for node_id in region.node_ids
            if node_id in node_by_id and node_by_id[node_id].node_type == "task"
        )
        scores.append(sum(candidate_values[:2]) + 1e-3 * uncovered_task_risk)
    return tuple(scores)


def _candidate_better(left, right, epsilon_u, epsilon_f):
    if right is None:
        return True
    left_record = left[0]
    right_record = right[0]
    if left_record.primary_value < right_record.primary_value - epsilon_u:
        return True
    if right_record.primary_value < left_record.primary_value - epsilon_u:
        return False
    if left_record.secondary_value < right_record.secondary_value - epsilon_f:
        return True
    if right_record.secondary_value < left_record.secondary_value - epsilon_f:
        return False
    return left_record.region_id < right_record.region_id


def _dgrl_context(
    *,
    initial_primary,
    initial_secondary,
    best_primary,
    best_secondary,
    iteration,
    max_iterations,
    no_improve,
    no_improve_limit,
    remaining_budget_s,
    total_budget_s,
    last_runtime_s,
    last_failure,
):
    primary_scale = max(1.0, abs(float(initial_primary)))
    secondary_scale = max(1.0, abs(float(initial_secondary)))
    if total_budget_s is None:
        remaining_ratio = 1.0
        runtime_ratio = 0.0
    else:
        total = max(float(total_budget_s), 1e-9)
        remaining_ratio = max(0.0, min(1.0, float(remaining_budget_s) / total))
        runtime_ratio = max(0.0, min(1.0, float(last_runtime_s) / total))
    return torch.tensor(
        [
            float(best_primary) / primary_scale,
            float(best_secondary) / secondary_scale,
            float(iteration) / max(1, int(max_iterations)),
            float(no_improve) / max(1, int(no_improve_limit)),
            remaining_ratio,
            runtime_ratio,
            float(bool(last_failure)),
        ],
        dtype=torch.float32,
    )


def rank_regions(
    regions,
    learned_scores,
    top_k,
    guard_count=0,
    deterministic_scores=None,
):
    if len(regions) != len(learned_scores):
        raise ValueError("regions and learned_scores must have equal length")
    if top_k <= 0:
        return ()

    learned_ranked = sorted(
        zip(regions, learned_scores),
        key=lambda pair: (-float(pair[1]), pair[0].region_id),
    )
    guard_count = min(max(int(guard_count), 0), top_k, len(regions))
    if guard_count == 0:
        return tuple(learned_ranked[:top_k])

    if deterministic_scores is None:
        deterministic_scores = baseline_region_scores(None, regions)
    if len(regions) != len(deterministic_scores):
        raise ValueError("regions and deterministic_scores must have equal length")
    deterministic_ranked = sorted(
        zip(regions, deterministic_scores),
        key=lambda pair: (-float(pair[1]), pair[0].region_id),
    )
    score_by_id = {
        region.region_id: score for region, score in zip(regions, learned_scores)
    }
    selected = []
    selected_ids = set()
    for region, _ in deterministic_ranked[:guard_count]:
        if region.region_id not in selected_ids:
            selected.append((region, score_by_id[region.region_id]))
            selected_ids.add(region.region_id)
    for region, score in learned_ranked:
        if len(selected) >= top_k:
            break
        if region.region_id not in selected_ids:
            selected.append((region, score))
            selected_ids.add(region.region_id)
    return tuple(selected[:top_k])


class HGNNRegionScorer:
    def __init__(self, model):
        self.model = model

    def __call__(self, graph, regions):
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        tensor_graph = tensorize_graph(graph)
        node_index = {
            node_id: index for index, node_id in enumerate(tensor_graph.node_ids)
        }
        region_indices = tuple(
            tuple(node_index[node_id] for node_id in region.node_ids)
            for region in regions
        )
        with torch.no_grad():
            scores = self.model(tensor_graph, region_indices)
        return tuple(float(value) for value in scores.cpu())


class OrdinaryGNNRegionScorer(HGNNRegionScorer):
    pass


def run_search(
    state,
    task_ids,
    miss_penalty,
    validation_state,
    operation_capacity,
    config: SearchConfig,
    scorer=None,
    tuple_expander=None,
    controller=None,
    controller_training=False,
    step_observer=None,
):
    repair_solver_backend = str(config.repair_solver_backend).upper()
    if repair_solver_backend not in {"CBC", "SCIP"}:
        raise ValueError("repair_solver_backend must be CBC or SCIP")
    if config.repair_node_limit is not None and config.repair_node_limit <= 0:
        raise ValueError("repair_node_limit must be positive")
    if any(limit <= 0 for limit in config.repair_node_budgets):
        raise ValueError("repair_node_budgets must be positive")
    if (config.repair_node_limit is not None or config.repair_node_budgets) and (
        repair_solver_backend != "SCIP"
    ):
        raise ValueError("deterministic repair node budgets require SCIP")
    if config.acceptance_mode not in ACCEPTANCE_MODES:
        raise ValueError(
            f"unknown acceptance_mode: {config.acceptance_mode!r}; "
            f"expected one of {sorted(ACCEPTANCE_MODES)}"
        )
    incumbent = state
    initial_primary, initial_secondary, _ = _objective(
        incumbent,
        task_ids,
        miss_penalty,
        config,
        validation_state.previous_arc_values,
        validation_state.previous_tuple_values,
    )
    best_primary = initial_primary
    best_secondary = initial_secondary
    trace = []
    accepted_records = []
    no_improve = 0
    stop_reason = "max_iterations"
    scorer = scorer or baseline_region_scores
    started = perf_counter()
    last_runtime_s = 0.0
    last_failure = False

    for iteration in range(config.max_iterations):
        iteration_incumbent = incumbent
        graph = build_constraint_hypergraph(
            iteration_incumbent,
            task_ids,
            task_weights=miss_penalty,
        )
        _, _, covered = _objective(
            iteration_incumbent,
            task_ids,
            miss_penalty,
            config,
            validation_state.previous_arc_values,
            validation_state.previous_tuple_values,
        )
        if controller is not None:
            state_before = incumbent
            primary_before = best_primary
            secondary_before = best_secondary
            if config.total_time_limit_s is None:
                remaining_budget_s = float("inf")
            else:
                remaining_budget_s = max(
                    0.0,
                    float(config.total_time_limit_s) - (perf_counter() - started),
                )
            if remaining_budget_s <= 0:
                stop_reason = "total_time_limit"
                break
            context = _dgrl_context(
                initial_primary=initial_primary,
                initial_secondary=initial_secondary,
                best_primary=best_primary,
                best_secondary=best_secondary,
                iteration=iteration,
                max_iterations=config.max_iterations,
                no_improve=no_improve,
                no_improve_limit=config.no_improve_limit,
                remaining_budget_s=remaining_budget_s,
                total_budget_s=config.total_time_limit_s,
                last_runtime_s=last_runtime_s,
                last_failure=last_failure,
            )
            decision = controller.select_action(
                state=incumbent,
                graph=graph,
                context=context,
                remaining_budget_s=remaining_budget_s,
                locked_tuple_ids=validation_state.locked_tuple_ids,
                locked_arc_ids=validation_state.locked_arc_ids,
                training=controller_training,
            )
            if decision.action is None or decision.projection is None:
                stop_reason = "no_feasible_dgrl_action"
                break
            projection = decision.projection
            repair_node_limit = config.repair_node_limit
            if config.repair_node_budgets:
                if decision.action.time_budget >= len(config.repair_node_budgets):
                    raise ValueError("repair_node_budgets do not cover DGRL action")
                repair_node_limit = config.repair_node_budgets[
                    decision.action.time_budget
                ]
            repaired = repair_region(
                incumbent,
                projection.closure,
                task_ids,
                miss_penalty,
                validation_state,
                operation_capacity,
                projection.time_limit_s,
                config.epsilon_u,
                config.epsilon_f,
                makespan_weight=config.makespan_weight,
                time_weight=config.time_weight,
                risk_weight=config.risk_weight,
                energy_weight=config.energy_weight,
                plan_change_weight=config.plan_change_weight,
                solver_backend=repair_solver_backend,
                node_limit=repair_node_limit,
            )
            strict = repaired.accepted and (
                repaired.primary_value < best_primary - config.epsilon_u
                or (
                    abs(repaired.primary_value - best_primary) <= config.epsilon_u
                    and repaired.secondary_value
                    < best_secondary - config.epsilon_f
                )
            )
            record = SearchRecord(
                iteration,
                projection.region.region_id,
                projection.closure.accepted,
                repaired.accepted,
                strict,
                repaired.validation.ok,
                repaired.primary_value,
                repaired.secondary_value,
                "improved" if strict else repaired.reason,
                strict,
                strict,
                decision.action.neighborhood_type,
                decision.action.release_profile,
                projection.time_limit_s,
                decision.used_fallback,
            )
            trace.append(record)
            last_runtime_s = repaired.runtime_s
            last_failure = not repaired.accepted
            effort_cost = float(repaired.runtime_s)
            if repair_node_limit is not None:
                effort_denominator = max(
                    config.repair_node_budgets or (repair_node_limit,)
                )
                effort_cost = float(repair_node_limit) / float(
                    effort_denominator
                )
            if strict:
                incumbent = SearchState(
                    incumbent.arcs,
                    incumbent.tuples,
                    frozenset(repaired.selected_arc_ids),
                    frozenset(repaired.selected_tuple_ids),
                )
                best_primary = repaired.primary_value
                best_secondary = repaired.secondary_value
                accepted_records.append(record)
                no_improve = 0
            else:
                no_improve += 1
            if step_observer is not None:
                step_observer(
                    DGRLStepEvent(
                        iteration,
                        graph,
                        context.detach().clone(),
                        decision.action,
                        state_before,
                        incumbent,
                        float(primary_before - best_primary),
                        float(secondary_before - best_secondary),
                        float(not (repaired.accepted and repaired.validation.ok)),
                        effort_cost,
                        float(remaining_budget_s),
                        bool(decision.used_fallback),
                    )
                )
            if not strict:
                if no_improve >= config.no_improve_limit:
                    stop_reason = "no_improve_limit"
                    break
            continue
        missed = sorted(set(task_ids) - covered)
        seed_tasks = missed if missed else sorted(task_ids)
        regions = generate_regions(
            graph,
            tuple(f"task:{task_id}" for task_id in seed_tasks),
            config.region_budgets,
        )
        if not regions:
            stop_reason = "no_candidate_region"
            break
        scores = scorer(graph, regions)
        deterministic_scores = (
            risk_aware_region_scores(graph, regions)
            if config.deterministic_guard_count
            else None
        )
        ranked = rank_regions(
            regions,
            scores,
            config.top_k,
            config.deterministic_guard_count,
            deterministic_scores,
        )
        improved_this_iteration = False
        eligible_candidates = []
        best_candidate = None
        for region, _ in ranked:
            candidate_state = (
                iteration_incumbent
                if config.acceptance_mode == "best_of_top_k"
                else incumbent
            )
            closure = close_region(candidate_state, region, config.closure_budget)
            if not closure.accepted:
                record = SearchRecord(
                    iteration,
                    region.region_id,
                    False,
                    False,
                    False,
                    False,
                    best_primary,
                    best_secondary,
                    closure.reason,
                )
                trace.append(record)
                continue
            if tuple_expander is not None:
                generated = tuple(tuple_expander(candidate_state, closure))
                invalid = [
                    item.tuple_id
                    for item in generated
                    if not item.admitted
                    or not set(item.task_ids) <= set(closure.task_ids)
                ]
                if invalid:
                    trace.append(
                        SearchRecord(
                            iteration,
                            region.region_id,
                            True,
                            False,
                            False,
                            False,
                            best_primary,
                            best_secondary,
                            "tuple_expansion_rejected",
                        )
                    )
                    continue
                if generated:
                    tuple_by_id = {
                        item.tuple_id: item for item in candidate_state.tuples
                    }
                    tuple_by_id.update({item.tuple_id: item for item in generated})
                    candidate_state = SearchState(
                        candidate_state.arcs,
                        tuple(sorted(tuple_by_id.values(), key=lambda item: item.tuple_id)),
                        candidate_state.selected_arc_ids,
                        candidate_state.selected_tuple_ids,
                    )
                    if config.acceptance_mode == "first_improvement":
                        incumbent = candidate_state
                    expanded_region = replace(
                        region,
                        node_ids=tuple(
                            sorted(
                                set(region.node_ids)
                                | {item.tuple_id for item in generated}
                            )
                        ),
                    )
                    closure = close_region(
                        candidate_state,
                        expanded_region,
                        config.closure_budget,
                    )
                    if not closure.accepted:
                        trace.append(
                            SearchRecord(
                                iteration,
                                region.region_id,
                                False,
                                False,
                                False,
                                False,
                                best_primary,
                                best_secondary,
                                closure.reason,
                            )
                        )
                        continue
            repaired = repair_region(
                candidate_state,
                closure,
                task_ids,
                miss_penalty,
                validation_state,
                operation_capacity,
                config.repair_time_limit_s,
                config.epsilon_u,
                config.epsilon_f,
                makespan_weight=config.makespan_weight,
                time_weight=config.time_weight,
                risk_weight=config.risk_weight,
                energy_weight=config.energy_weight,
                plan_change_weight=config.plan_change_weight,
                solver_backend=repair_solver_backend,
                node_limit=config.repair_node_limit,
            )
            strict = repaired.accepted and (
                repaired.primary_value < best_primary - config.epsilon_u
                or (
                    abs(repaired.primary_value - best_primary) <= config.epsilon_u
                    and repaired.secondary_value < best_secondary - config.epsilon_f
                )
            )
            selected_now = strict and config.acceptance_mode == "first_improvement"
            record = SearchRecord(
                iteration,
                region.region_id,
                True,
                repaired.accepted,
                selected_now,
                repaired.validation.ok,
                repaired.primary_value,
                repaired.secondary_value,
                (
                    "improved"
                    if selected_now
                    else "improved_candidate"
                    if strict
                    else repaired.reason
                ),
                strict,
                selected_now,
            )
            trace.append(record)
            if strict:
                if config.acceptance_mode == "first_improvement":
                    incumbent = SearchState(
                        candidate_state.arcs,
                        candidate_state.tuples,
                        frozenset(repaired.selected_arc_ids),
                        frozenset(repaired.selected_tuple_ids),
                    )
                    best_primary = repaired.primary_value
                    best_secondary = repaired.secondary_value
                    accepted_records.append(record)
                    improved_this_iteration = True
                    break
                candidate = (record, candidate_state, repaired, len(trace) - 1)
                eligible_candidates.append(candidate)
                if _candidate_better(
                    candidate,
                    best_candidate,
                    config.epsilon_u,
                    config.epsilon_f,
                ):
                    best_candidate = candidate
        if config.acceptance_mode == "best_of_top_k" and best_candidate is not None:
            for candidate in eligible_candidates:
                candidate_record, _, _, trace_index = candidate
                trace[trace_index] = replace(
                    candidate_record,
                    reason="improved_not_selected",
                )
            winner_record, winner_state, winner_repair, winner_trace_index = (
                best_candidate
            )
            winner_record = replace(
                winner_record,
                improvement_accepted=True,
                reason="improved",
                selected=True,
            )
            trace[winner_trace_index] = winner_record
            incumbent = SearchState(
                winner_state.arcs,
                winner_state.tuples,
                frozenset(winner_repair.selected_arc_ids),
                frozenset(winner_repair.selected_tuple_ids),
            )
            best_primary = winner_repair.primary_value
            best_secondary = winner_repair.secondary_value
            accepted_records.append(winner_record)
            improved_this_iteration = True
        if improved_this_iteration:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.no_improve_limit:
                stop_reason = "no_improve_limit"
                break

    return SearchResult(
        initial_primary,
        initial_secondary,
        best_primary,
        best_secondary,
        tuple(sorted(incumbent.selected_tuple_ids)),
        tuple(sorted(incumbent.selected_arc_ids)),
        tuple(trace),
        tuple(accepted_records),
        stop_reason,
    )
