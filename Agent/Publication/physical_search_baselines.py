"""Physical strong baselines for the canonical iterative-search study."""
from dataclasses import dataclass, replace
from math import isfinite
from random import Random
from time import perf_counter
from types import SimpleNamespace as NS

from Agent.Proposed.hypergraph import build_constraint_hypergraph
from Agent.Publication.action_features import build_observable_action_features
from Agent.Publication.actions import resolve_closure_budget
from Agent.Publication.budget import BudgetExpired, Deadline
from Agent.Publication.canonical_candidates import build_canonical_candidates
from Agent.Publication.canonical_iterative_search import strictly_better
from Agent.Publication.canonical_ranker import (
    FAMILIES,
    candidate_feature_cache,
    candidate_tensor,
)
from Agent.Publication.milp import solve_committed
from Agent.Publication.repair import repair_region


@dataclass(frozen=True)
class PhysicalAlnsConfig:
    seed: int = 0
    max_iterations: int = 12
    no_improve_limit: int = 6
    total_time_limit_s: float = 24.0
    repair_time_limit_s: float = 6.0
    final_validation_reserve_s: float = 0.30
    validation_reserve_s: float = 0.30
    minimum_repair_s: float = 0.50
    primary_time_fraction: float = 2.0 / 3.0
    reaction_factor: float = 0.20
    max_tuples_per_anchor: int = 40
    windows_per_vehicle: int = 3

    def __post_init__(self):
        if self.max_iterations <= 0 or self.no_improve_limit <= 0:
            raise ValueError("ALNS iteration limits must be positive")
        if self.no_improve_limit > self.max_iterations:
            raise ValueError("The ALNS no-improvement limit cannot exceed max iterations")
        for name in (
            "total_time_limit_s", "repair_time_limit_s",
            "final_validation_reserve_s", "validation_reserve_s",
            "minimum_repair_s",
        ):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.repair_time_limit_s > self.total_time_limit_s:
            raise ValueError("The ALNS repair limit cannot exceed the total limit")
        if not 0.0 < self.primary_time_fraction <= 1.0:
            raise ValueError("The primary-stage fraction must lie in (0, 1]")
        if not 0.0 < self.reaction_factor <= 1.0:
            raise ValueError("The reaction factor must lie in (0, 1]")


def _weighted_family(families, weights, rng):
    total = sum(max(1e-9, float(weights[family])) for family in families)
    threshold = rng.random() * total
    cumulative = 0.0
    for family in families:
        cumulative += max(1e-9, float(weights[family]))
        if threshold <= cumulative:
            return family
    return families[-1]


def select_alns_candidate(candidate_rows, weights, rng, iteration, attempted):
    available = {}
    for index, row in enumerate(candidate_rows):
        if row["candidate_id"] in attempted:
            continue
        available.setdefault(row["family"], []).append(index)
    families = tuple(family for family in FAMILIES if family in available)
    if not families:
        return None
    scheduled = FAMILIES[iteration] if iteration < len(FAMILIES) else None
    family = scheduled if scheduled in available else _weighted_family(
        families, weights, rng)
    return max(
        available[family],
        key=lambda index: candidate_rows[index]["heuristic_key"],
    )


def run_physical_adaptive_lns(
    *, problem, objective_weights, release_profile,
    config=PhysicalAlnsConfig(),
):
    """Run adaptive family selection with the same physical repair operator."""
    deadline = Deadline(config.total_time_limit_s)
    incumbent = problem.state
    initial = problem.check(incumbent, objective_weights)
    initial_objective = float(initial["primary"]), float(initial["secondary"])
    best_objective = initial_objective
    rng = Random(config.seed)
    weights = {family: 1.0 for family in FAMILIES}
    attempted_by_state = {}
    trace = []
    accepted = []
    no_improve = 0
    previous_construction_s = 0.0
    stop_reason = "max_iterations"

    for iteration in range(config.max_iterations):
        construction_guard_s = 1.25 * previous_construction_s
        if deadline.remaining() <= (
            config.final_validation_reserve_s + config.minimum_repair_s
            + construction_guard_s
        ):
            stop_reason = "total_time_limit"
            break
        iteration_tick = perf_counter()
        before = best_objective
        graph = build_constraint_hypergraph(
            incumbent, problem.domain.problem.task_ids,
            task_weights=problem.domain.problem.miss_penalty)
        feature_rows = build_observable_action_features(
            incumbent, graph, problem.domain.problem.miss_penalty,
            objective_weights)
        closure_budget = resolve_closure_budget(graph, release_profile)
        candidates = build_canonical_candidates(
            incumbent, graph, problem.domain.problem.miss_penalty,
            objective_weights, closure_budget,
            locked_tuple_ids=problem.validation_state.locked_tuple_ids,
            locked_arc_ids=problem.validation_state.locked_arc_ids,
            max_tuples_per_anchor=config.max_tuples_per_anchor,
            windows_per_vehicle=config.windows_per_vehicle,
            window_span=release_profile.region_budget.max_span,
            feature_rows=feature_rows)
        if not candidates:
            stop_reason = "no_candidate_region"
            break
        cache = candidate_feature_cache(graph, feature_rows)
        rows = [
            candidate_tensor(
                candidate, graph, feature_rows, closure_budget,
                feature_cache=cache)
            for candidate in candidates]
        previous_construction_s = perf_counter() - iteration_tick
        if deadline.remaining() <= (
            config.final_validation_reserve_s + config.minimum_repair_s
        ):
            stop_reason = "total_time_limit"
            break
        state_key = (
            tuple(sorted(incumbent.selected_arc_ids)),
            tuple(sorted(incumbent.selected_tuple_ids)),
        )
        attempted = attempted_by_state.setdefault(state_key, set())
        index = select_alns_candidate(
            rows, weights, rng, iteration, attempted)
        if index is None:
            stop_reason = "candidate_exhausted"
            break
        candidate = candidates[index]
        attempted.add(candidate.candidate_id)
        allocation_s = min(
            config.repair_time_limit_s,
            deadline.remaining() - config.final_validation_reserve_s)
        if allocation_s < config.minimum_repair_s:
            stop_reason = "total_time_limit"
            break
        problem.deadline = Deadline(allocation_s)
        repair_tick = perf_counter()
        repaired = repair_region(
            incumbent, candidate.closure,
            problem.domain.problem.task_ids,
            problem.domain.problem.miss_penalty,
            problem.validation_state, {}, allocation_s,
            solver_backend="SCIP",
            validation_reserve_s=config.validation_reserve_s,
            primary_time_fraction=config.primary_time_fraction,
            **objective_weights)
        proposed = incumbent
        if repaired.accepted and repaired.validation.ok:
            proposed = replace(
                incumbent,
                selected_arc_ids=frozenset(repaired.selected_arc_ids),
                selected_tuple_ids=frozenset(repaired.selected_tuple_ids))
        checked = problem.check(proposed, objective_weights)
        after = float(checked["primary"]), float(checked["secondary"])
        improved = strictly_better(after, best_objective)
        if improved:
            incumbent = proposed
            best_objective = after
            accepted.append({
                "iteration": iteration,
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "objective": after,
            })
            no_improve = 0
            reward = 5.0
        else:
            no_improve += 1
            reward = 1.0 if repaired.accepted and repaired.validation.ok else 0.0
        rate = config.reaction_factor
        weights[candidate.family] = (
            (1.0 - rate) * weights[candidate.family] + rate * reward)
        trace.append({
            "iteration": iteration,
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "objective_before": before,
            "objective_after": best_objective,
            "strict_improvement": improved,
            "repair_accepted": bool(repaired.accepted),
            "validation_ok": bool(repaired.validation.ok),
            "repair_reason": repaired.reason,
            "primary_solver_status": repaired.primary_status,
            "secondary_solver_status": repaired.secondary_status,
            "solver_nodes": int(repaired.solver_nodes),
            "incumbent_hint_accepted": repaired.incumbent_hint_accepted,
            "incumbent_hint_violation_count": (
                repaired.incumbent_hint_violation_count),
            "repair_budget_s": allocation_s,
            "repair_elapsed_s": perf_counter() - repair_tick,
            "construction_elapsed_s": previous_construction_s,
            "iteration_elapsed_s": perf_counter() - iteration_tick,
            "family_weights_after": dict(weights),
        })
        if no_improve >= config.no_improve_limit:
            stop_reason = "no_improve_limit"
            break

    final = problem.check(incumbent, objective_weights)
    final_objective = float(final["primary"]), float(final["secondary"])
    if final_objective != best_objective:
        raise ValueError("Physical ALNS returned a mismatched incumbent objective")
    return {
        "policy": "physical_adaptive_lns",
        "seed": config.seed,
        "initial_objective": initial_objective,
        "best_objective": best_objective,
        "selected_tuple_ids": tuple(sorted(incumbent.selected_tuple_ids)),
        "selected_arc_ids": tuple(sorted(incumbent.selected_arc_ids)),
        "iterations": len(trace),
        "accepted_iterations": len(accepted),
        "repair_observations": len(trace),
        "physical_checks_passed": len(trace) + 2,
        "elapsed_s": deadline.elapsed(),
        "stop_reason": stop_reason,
        "final_family_weights": weights,
        "accepted": accepted,
        "trace": trace,
    }


def _pool_domain(domain, copies):
    return NS(
        instance=domain.instance,
        context=domain.context,
        problem=domain.problem,
        templates=(),
        specials=tuple(copies),
        groups=domain.groups,
        required_arc_ids=domain.required_arc_ids,
        required_vehicle_ids=domain.required_vehicle_ids,
        fixed_busy=domain.fixed_busy,
        fixed_operations=domain.fixed_operations,
        previous_copies=domain.previous_copies,
        waiting=domain.waiting,
        objective_reference_domain=domain,
    )


def run_pooled_complete_milp(
    *, problem, objective_weights, total_time_limit_s=24.0,
    primary_time_fraction=2.0 / 3.0, validation_reserve_s=0.30,
):
    """Solve the complete bounded candidate pool under the matched wall clock."""
    if total_time_limit_s <= 0:
        raise ValueError("The complete-MILP time limit must be positive")
    if not 0.0 < primary_time_fraction <= 1.0:
        raise ValueError("The primary-stage fraction must lie in (0, 1]")
    deadline = Deadline(total_time_limit_s)
    incumbent = problem.plan(problem.state)
    domain = _pool_domain(problem.domain, problem.full_copies.values())
    try:
        raw = solve_committed(
            domain,
            NS(objective_weights=objective_weights),
            primary_seconds=total_time_limit_s * primary_time_fraction,
            secondary_seconds=total_time_limit_s,
            deadline=deadline,
            incumbent=incumbent,
            validation_reserve_s=validation_reserve_s,
        )
    except BudgetExpired:
        raw = {
            "selected": None,
            "budget_exhausted": True,
            "variables": 0,
            "constraints": 0,
            "stages": [],
        }
    selected = raw["selected"]
    final_plan = incumbent if selected is None else selected["plan"]
    initial_checked = problem.check(problem.state, objective_weights)
    initial_objective = (
        float(initial_checked["primary"]), float(initial_checked["secondary"]))
    final_state = replace(
        problem.state,
        selected_arc_ids=frozenset(arc.arc_id for arc in final_plan.arcs),
        selected_tuple_ids=frozenset(copy.tuple_id for copy in final_plan.copies),
    )
    final_checked = problem.check(final_state, objective_weights)
    best_objective = (
        float(final_checked["primary"]), float(final_checked["secondary"]))
    if strictly_better(initial_objective, best_objective):
        final_plan = incumbent
        final_state = problem.state
        best_objective = initial_objective
    elapsed_s = deadline.elapsed()
    return {
        "policy": "pooled_complete_milp",
        "initial_objective": initial_objective,
        "best_objective": best_objective,
        "selected_tuple_ids": tuple(sorted(final_state.selected_tuple_ids)),
        "selected_arc_ids": tuple(sorted(final_state.selected_arc_ids)),
        "elapsed_s": elapsed_s,
        "within_budget": elapsed_s <= total_time_limit_s,
        "solver_budget_exhausted": bool(raw["budget_exhausted"]),
        "solver_variables": int(raw["variables"]),
        "solver_constraints": int(raw["constraints"]),
        "solver_stages": raw["stages"],
        "physical_checks_passed": 2,
    }


__all__ = [
    "PhysicalAlnsConfig",
    "run_physical_adaptive_lns",
    "run_pooled_complete_milp",
    "select_alns_candidate",
]
