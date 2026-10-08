"""Iterative canonical-neighborhood search under one measured wall-clock budget."""
from collections import Counter
from dataclasses import dataclass, replace
from math import isfinite
from time import perf_counter

import torch

from Agent.Proposed.hypergraph import build_constraint_hypergraph
from Agent.Publication.action_features import build_observable_action_features
from Agent.Publication.actions import resolve_closure_budget
from Agent.Publication.budget import Deadline
from Agent.Publication.canonical_candidates import build_canonical_candidates
from Agent.Publication.canonical_ranker import (
    FAMILIES,
    candidate_feature_cache,
    candidate_tensor,
    heuristic_candidate_index,
    tensorize_observable_graph,
)
from Agent.Publication.canonical_residual_portfolio import (
    residual_portfolio_indices,
    residual_repair_budget,
)
from Agent.Publication.repair import repair_region


POLICIES = (
    "deterministic_residual_family_portfolio",
    "matched_primary_protected_single",
    "learned_family_cycle",
    "heuristic_family_cycle",
    "heuristic_family_balance",
    "learned_global_then_balanced",
    "learned_failure_triggered_coverage",
    "heuristic",
)


@dataclass(frozen=True)
class CanonicalIterativeConfig:
    max_iterations: int = 4
    no_improve_limit: int = 2
    total_time_limit_s: float = 24.0
    action_time_limit_s: float = 6.0
    action_reserve_s: float = 0.25
    final_validation_reserve_s: float = 0.30
    minimum_repair_s: float = 0.50
    primary_time_fraction: float = 2.0 / 3.0
    validation_reserve_s: float = 0.30
    max_tuples_per_anchor: int = 40
    windows_per_vehicle: int = 3
    deterministic_fallback_after_no_improve: bool = False

    def __post_init__(self):
        if self.max_iterations <= 0 or self.no_improve_limit <= 0:
            raise ValueError("Iteration limits must be positive")
        if self.no_improve_limit > self.max_iterations:
            raise ValueError("The no-improvement limit cannot exceed max iterations")
        for name in (
            "total_time_limit_s",
            "action_time_limit_s",
            "action_reserve_s",
            "final_validation_reserve_s",
            "minimum_repair_s",
            "validation_reserve_s",
        ):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.action_time_limit_s > self.total_time_limit_s:
            raise ValueError("The action limit cannot exceed the search limit")
        if self.action_reserve_s + self.minimum_repair_s >= self.action_time_limit_s:
            raise ValueError("The guarded action allowance leaves no repair time")
        if not 0.0 < self.primary_time_fraction <= 1.0:
            raise ValueError("The primary-stage fraction must lie in (0, 1]")
        if self.max_tuples_per_anchor <= 0 or self.windows_per_vehicle <= 0:
            raise ValueError("Canonical candidate limits must be positive")


def strictly_better(left, right, epsilon_u=1e-8, epsilon_f=1e-8):
    return (
        float(left[0]) < float(right[0]) - float(epsilon_u)
        or (
            abs(float(left[0]) - float(right[0])) <= float(epsilon_u)
            and float(left[1]) < float(right[1]) - float(epsilon_f)
        )
    )


def selected_candidate_indices(
    policy,
    scores,
    candidate_rows,
    excluded_candidate_ids=frozenset(),
    iteration=0,
    family_attempt_counts=None,
    consecutive_no_improve=0,
):
    if policy not in POLICIES:
        raise ValueError(f"Unknown canonical iterative policy {policy}")
    available = [
        index for index, row in enumerate(candidate_rows)
        if row["candidate_id"] not in excluded_candidate_ids
    ]
    if not available:
        return ()
    if policy == "heuristic":
        selected = heuristic_candidate_index(
            [candidate_rows[index] for index in available])
        return (available[selected],)
    if policy == "heuristic_family_cycle":
        start = int(iteration) % len(FAMILIES)
        for offset in range(len(FAMILIES)):
            family = FAMILIES[(start + offset) % len(FAMILIES)]
            eligible = [
                index for index in available
                if candidate_rows[index]["family"] == family
            ]
            if eligible:
                return (max(
                    eligible,
                    key=lambda index: candidate_rows[index]["heuristic_key"],
                ),)
        return ()
    if policy == "heuristic_family_balance":
        counts = Counter() if family_attempt_counts is None else Counter(
            family_attempt_counts)
        available_families = [
            family for family in FAMILIES
            if any(candidate_rows[index]["family"] == family for index in available)
        ]
        minimum = min(counts[family] for family in available_families)
        family = next(
            family for family in available_families if counts[family] == minimum)
        eligible = [
            index for index in available
            if candidate_rows[index]["family"] == family
        ]
        return (max(
            eligible,
            key=lambda index: candidate_rows[index]["heuristic_key"],
        ),)
    if policy == "matched_primary_protected_single":
        return (min(
            available,
            key=lambda index: (
                -float(scores[index]), candidate_rows[index]["candidate_id"]),
        ),)
    if policy == "learned_global_then_balanced":
        counts = Counter() if family_attempt_counts is None else Counter(
            family_attempt_counts)
        if not counts:
            return (min(
                available,
                key=lambda index: (
                    -float(scores[index]), candidate_rows[index]["candidate_id"]),
            ),)
        best_by_family = {}
        for family in FAMILIES:
            eligible = [
                index for index in available
                if candidate_rows[index]["family"] == family
            ]
            if eligible:
                best_by_family[family] = min(
                    eligible,
                    key=lambda index: (
                        -float(scores[index]),
                        candidate_rows[index]["candidate_id"],
                    ),
                )
        minimum = min(counts[family] for family in best_by_family)
        least_used = [
            best_by_family[family] for family in FAMILIES
            if family in best_by_family and counts[family] == minimum
        ]
        return (min(
            least_used,
            key=lambda index: (
                -float(scores[index]), candidate_rows[index]["candidate_id"]),
        ),)
    if policy == "learned_failure_triggered_coverage":
        if int(consecutive_no_improve) <= 0:
            return (min(
                available,
                key=lambda index: (
                    -float(scores[index]), candidate_rows[index]["candidate_id"]),
            ),)
        counts = Counter() if family_attempt_counts is None else Counter(
            family_attempt_counts)
        best_by_family = {}
        for family in FAMILIES:
            eligible = [
                index for index in available
                if candidate_rows[index]["family"] == family
            ]
            if eligible:
                best_by_family[family] = min(
                    eligible,
                    key=lambda index: (
                        -float(scores[index]),
                        candidate_rows[index]["candidate_id"],
                    ),
                )
        minimum = min(counts[family] for family in best_by_family)
        least_used = [
            best_by_family[family] for family in FAMILIES
            if family in best_by_family and counts[family] == minimum
        ]
        return (min(
            least_used,
            key=lambda index: (
                -float(scores[index]), candidate_rows[index]["candidate_id"]),
        ),)
    if policy == "learned_family_cycle":
        start = int(iteration) % len(FAMILIES)
        for offset in range(len(FAMILIES)):
            family = FAMILIES[(start + offset) % len(FAMILIES)]
            eligible = [
                index for index in available
                if candidate_rows[index]["family"] == family
            ]
            if eligible:
                return (min(
                    eligible,
                    key=lambda index: (
                        -float(scores[index]),
                        candidate_rows[index]["candidate_id"],
                    ),
                ),)
        return ()
    local_rows = [candidate_rows[index] for index in available]
    local_scores = scores[available]
    if all(any(row["family"] == family for row in local_rows) for family in FAMILIES):
        local_indices = residual_portfolio_indices(local_scores, local_rows)
    else:
        local_indices = []
        for family in FAMILIES:
            eligible = [
                index for index, row in enumerate(local_rows)
                if row["family"] == family]
            if eligible:
                local_indices.append(min(
                    eligible,
                    key=lambda index: (
                        -float(local_scores[index]),
                        local_rows[index]["candidate_id"],
                    ),
                ))
        local_indices = tuple(sorted(
            local_indices,
            key=lambda index: (
                -float(local_scores[index]),
                local_rows[index]["candidate_id"],
            ),
        ))
    return tuple(available[index] for index in local_indices)


def deterministic_fallback_index(
    candidate_rows,
    excluded_candidate_ids,
    consecutive_no_improve,
):
    if consecutive_no_improve <= 0:
        return None
    available = [
        index for index, row in enumerate(candidate_rows)
        if row["candidate_id"] not in excluded_candidate_ids
    ]
    if not available:
        return None
    start = (int(consecutive_no_improve) - 1) % len(FAMILIES)
    for offset in range(len(FAMILIES)):
        family = FAMILIES[(start + offset) % len(FAMILIES)]
        eligible = [
            index for index in available
            if candidate_rows[index]["family"] == family]
        if eligible:
            return max(
                eligible,
                key=lambda index: candidate_rows[index]["heuristic_key"],
            )
    return None


def candidate_attempt_is_complete(position, repair_reason):
    return int(position) == 0 or repair_reason != "complete_budget_exhausted"


def _objective(problem, state, weights):
    checked = problem.check(state, weights)
    return float(checked["primary"]), float(checked["secondary"])


def run_canonical_iterative_search(
    *,
    problem,
    model,
    objective_weights,
    release_profile,
    policy,
    config=CanonicalIterativeConfig(),
    forced_first_candidate_id=None,
):
    """Rebuild, rank and execute canonical closures after each accepted move.

    Candidate construction, scoring, repair and final physical validation share
    one wall-clock deadline. Candidates within one portfolio action are all
    repaired from the same incumbent, after which the best strict improvement
    becomes the next state.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown canonical iterative policy {policy}")
    deadline = Deadline(config.total_time_limit_s)
    incumbent = problem.state
    initial_objective = _objective(problem, incumbent, objective_weights)
    best_objective = initial_objective
    trace = []
    accepted = []
    no_improve = 0
    stop_reason = "max_iterations"
    previous_construction_s = 0.0
    attempted_by_state = {}
    family_attempt_counts = Counter()

    for iteration in range(config.max_iterations):
        construction_guard_s = 1.25 * previous_construction_s
        if deadline.remaining() <= (
            config.final_validation_reserve_s + config.minimum_repair_s
            + config.action_reserve_s + construction_guard_s
        ):
            stop_reason = "total_time_limit"
            break
        iteration_tick = perf_counter()
        objective_before = best_objective

        graph = build_constraint_hypergraph(
            incumbent,
            problem.domain.problem.task_ids,
            task_weights=problem.domain.problem.miss_penalty,
        )
        feature_rows = build_observable_action_features(
            incumbent,
            graph,
            problem.domain.problem.miss_penalty,
            objective_weights,
        )
        closure_budget = resolve_closure_budget(graph, release_profile)
        candidates = build_canonical_candidates(
            incumbent,
            graph,
            problem.domain.problem.miss_penalty,
            objective_weights,
            closure_budget,
            locked_tuple_ids=problem.validation_state.locked_tuple_ids,
            locked_arc_ids=problem.validation_state.locked_arc_ids,
            max_tuples_per_anchor=config.max_tuples_per_anchor,
            windows_per_vehicle=config.windows_per_vehicle,
            window_span=release_profile.region_budget.max_span,
            feature_rows=feature_rows,
        )
        if not candidates:
            stop_reason = "no_candidate_region"
            break
        feature_cache = candidate_feature_cache(graph, feature_rows)
        candidate_rows = [
            candidate_tensor(
                candidate,
                graph,
                feature_rows,
                closure_budget,
                feature_cache=feature_cache,
            )
            for candidate in candidates
        ]
        construction_elapsed_s = perf_counter() - iteration_tick
        previous_construction_s = construction_elapsed_s
        if deadline.remaining() <= (
            config.final_validation_reserve_s
            + config.action_reserve_s
            + config.minimum_repair_s
        ):
            stop_reason = "total_time_limit"
            break
        action_tick = perf_counter()
        action_limit_s = min(
            config.action_time_limit_s,
            deadline.remaining() - config.final_validation_reserve_s,
        )
        scores = None
        if policy not in (
            "heuristic", "heuristic_family_cycle", "heuristic_family_balance"
        ):
            tensor_graph = tensorize_observable_graph(
                graph, feature_rows, feature_cache=feature_cache)
            with torch.no_grad():
                scores = model(tensor_graph, candidate_rows)
        state_key = (
            tuple(sorted(incumbent.selected_arc_ids)),
            tuple(sorted(incumbent.selected_tuple_ids)),
        )
        attempted = attempted_by_state.setdefault(state_key, set())
        if iteration == 0 and forced_first_candidate_id is not None:
            indices = tuple(
                index for index, row in enumerate(candidate_rows)
                if row["candidate_id"] == forced_first_candidate_id
                and row["candidate_id"] not in attempted
            )
            if len(indices) != 1:
                raise ValueError(
                    "The forced first candidate is missing or duplicated")
        else:
            indices = selected_candidate_indices(
                policy,
                scores,
                candidate_rows,
                excluded_candidate_ids=attempted,
                iteration=iteration,
                family_attempt_counts=family_attempt_counts,
                consecutive_no_improve=no_improve,
            )
        if not indices:
            stop_reason = "candidate_exhausted"
            break
        fallback_index = None
        if (
            policy == "deterministic_residual_family_portfolio"
            and config.deterministic_fallback_after_no_improve
            and no_improve > 0
        ):
            fallback_index = deterministic_fallback_index(
                candidate_rows, attempted, no_improve)
            if fallback_index is not None:
                indices = (fallback_index,) + tuple(
                    index for index in indices if index != fallback_index)
        selection_elapsed_s = perf_counter() - action_tick
        observations = []

        for position, index in enumerate(indices):
            action_elapsed_s = perf_counter() - action_tick
            guarded_limit_s = action_limit_s - config.action_reserve_s
            if policy == "deterministic_residual_family_portfolio":
                allocation_s = residual_repair_budget(
                    guarded_limit_s,
                    action_elapsed_s,
                    len(indices) - position,
                    first_candidate=position == 0,
                )
            else:
                allocation_s = max(0.0, guarded_limit_s - action_elapsed_s)
            allocation_s = min(
                allocation_s,
                max(0.0, deadline.remaining() - config.final_validation_reserve_s),
            )
            if allocation_s < config.minimum_repair_s:
                break
            candidate = candidates[index]
            problem.deadline = Deadline(allocation_s)
            repair_tick = perf_counter()
            repaired = repair_region(
                incumbent,
                candidate.closure,
                problem.domain.problem.task_ids,
                problem.domain.problem.miss_penalty,
                problem.validation_state,
                {},
                allocation_s,
                solver_backend="SCIP",
                validation_reserve_s=config.validation_reserve_s,
                primary_time_fraction=config.primary_time_fraction,
                **objective_weights,
            )
            proposed = incumbent
            if repaired.accepted and repaired.validation.ok:
                proposed = replace(
                    incumbent,
                    selected_arc_ids=frozenset(repaired.selected_arc_ids),
                    selected_tuple_ids=frozenset(repaired.selected_tuple_ids),
                )
            after = _objective(problem, proposed, objective_weights)
            observations.append({
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "anchor_id": candidate.anchor_id,
                "model_score": (
                    None if scores is None else float(scores[index])
                ),
                "repair_budget_s": allocation_s,
                "repair_elapsed_s": perf_counter() - repair_tick,
                "repair_accepted": bool(repaired.accepted),
                "validation_ok": bool(repaired.validation.ok),
                "repair_reason": repaired.reason,
                "primary_solver_status": repaired.primary_status,
                "secondary_solver_status": repaired.secondary_status,
                "solver_nodes": int(repaired.solver_nodes),
                "incumbent_hint_accepted": repaired.incumbent_hint_accepted,
                "incumbent_hint_violation_count": (
                    repaired.incumbent_hint_violation_count),
                "objective": after,
                "strict_improvement": strictly_better(after, best_objective),
                "state": proposed,
            })
            if candidate_attempt_is_complete(position, repaired.reason):
                attempted.add(candidate.candidate_id)

        if not observations:
            stop_reason = "insufficient_repair_time"
            break
        if policy in (
            "learned_global_then_balanced",
            "learned_failure_triggered_coverage",
            "heuristic_family_balance",
        ):
            family_attempt_counts.update(
                row["family"] for row in observations)
        winner = min(observations, key=lambda row: tuple(row["objective"]))
        improved = strictly_better(winner["objective"], best_objective)
        if improved:
            incumbent = winner["state"]
            best_objective = winner["objective"]
            accepted.append({
                "iteration": iteration,
                "candidate_id": winner["candidate_id"],
                "family": winner["family"],
                "objective": winner["objective"],
            })
            no_improve = 0
        else:
            no_improve += 1
        trace.append({
            "iteration": iteration,
            "objective_before": objective_before,
            "objective_after": best_objective,
            "candidate_count": len(candidates),
            "selected_candidate_ids": [
                candidates[index].candidate_id for index in indices
            ],
            "deterministic_fallback_used": fallback_index is not None,
            "deterministic_fallback_candidate_id": (
                None if fallback_index is None
                else candidates[fallback_index].candidate_id
            ),
            "selection_elapsed_s": selection_elapsed_s,
            "construction_elapsed_s": construction_elapsed_s,
            "action_elapsed_s": perf_counter() - action_tick,
            "iteration_elapsed_s": perf_counter() - iteration_tick,
            "improved": improved,
            "winning_candidate_id": winner["candidate_id"],
            "winning_family": winner["family"],
            "candidate_observations": [
                {key: value for key, value in row.items() if key != "state"}
                for row in observations
            ],
        })
        if no_improve >= config.no_improve_limit:
            stop_reason = "no_improve_limit"
            break

    final_objective = _objective(problem, incumbent, objective_weights)
    if final_objective != best_objective:
        raise ValueError("Final physical objective differs from the accepted incumbent")
    elapsed_s = deadline.elapsed()
    return {
        "policy": policy,
        "forced_first_candidate_id": forced_first_candidate_id,
        "initial_objective": initial_objective,
        "best_objective": best_objective,
        "selected_tuple_ids": tuple(sorted(incumbent.selected_tuple_ids)),
        "selected_arc_ids": tuple(sorted(incumbent.selected_arc_ids)),
        "iterations": len(trace),
        "accepted_iterations": len(accepted),
        "repair_observations": sum(
            len(row["candidate_observations"]) for row in trace
        ),
        "physical_checks_passed": sum(
            len(row["candidate_observations"]) for row in trace
        ) + 2,
        "elapsed_s": elapsed_s,
        "stop_reason": stop_reason,
        "accepted": accepted,
        "trace": trace,
    }


__all__ = [
    "CanonicalIterativeConfig",
    "POLICIES",
    "candidate_attempt_is_complete",
    "run_canonical_iterative_search",
    "deterministic_fallback_index",
    "selected_candidate_indices",
    "strictly_better",
]
