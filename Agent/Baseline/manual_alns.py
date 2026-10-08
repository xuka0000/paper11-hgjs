from dataclasses import dataclass, replace
from math import ceil, exp
from random import Random
from time import perf_counter

from .common import BaselineResult, BaselineTraceRecord, project_tuple_selection
from .risk_greedy import (
    RiskPriorityGreedyBaseline,
    _mandatory_tuple_ids,
    _tuple_secondary_estimate,
)


DESTROY_OPERATORS = ("worst_cost", "related", "random")
REPAIR_OPERATORS = ("risk", "efficiency", "regret")


def _lexicographically_better(left, right, epsilon_u, epsilon_f):
    if left.best_primary < right.best_primary - epsilon_u:
        return True
    if right.best_primary < left.best_primary - epsilon_u:
        return False
    return left.best_secondary < right.best_secondary - epsilon_f


def _weighted_operator(names, weights, rng):
    total = sum(max(1e-9, weights[name]) for name in names)
    threshold = rng.random() * total
    cumulative = 0.0
    for name in names:
        cumulative += max(1e-9, weights[name])
        if threshold <= cumulative:
            return name
    return names[-1]


def _destroy_tuple_ids(
    name,
    selected,
    mandatory,
    tuple_by_id,
    remove_count,
    rng,
):
    pool = sorted(set(selected) - set(mandatory))
    if not pool or remove_count <= 0:
        return frozenset()
    remove_count = min(remove_count, len(pool))
    if name == "worst_cost":
        ranked = sorted(
            pool,
            key=lambda tuple_id: (
                -float(tuple_by_id[tuple_id].secondary_cost),
                -float(tuple_by_id[tuple_id].risk_exposure),
                tuple_id,
            ),
        )
        return frozenset(ranked[:remove_count])
    if name == "related":
        anchor_id = pool[0]
        anchor = tuple_by_id[anchor_id]
        ranked = sorted(
            pool,
            key=lambda tuple_id: (
                tuple_by_id[tuple_id].vehicle_id != anchor.vehicle_id,
                tuple_by_id[tuple_id].uav_id != anchor.uav_id,
                abs(tuple_by_id[tuple_id].launch_layer - anchor.launch_layer),
                tuple_id,
            ),
        )
        return frozenset(ranked[:remove_count])
    return frozenset(rng.sample(pool, remove_count))


def _marginal_risk(item, covered, miss_penalty):
    return sum(
        float(miss_penalty[task_id])
        for task_id in item.task_ids
        if task_id not in covered
    )


def _repair_order(name, candidates, selected, tuple_by_id, miss_penalty, config):
    covered = {
        task_id
        for tuple_id in selected
        for task_id in tuple_by_id[tuple_id].task_ids
    }

    def values(item):
        risk = _marginal_risk(item, covered, miss_penalty)
        cost = max(0.0, _tuple_secondary_estimate(item, config))
        efficiency = risk / (1.0 + cost)
        alternatives = [
            other
            for other in candidates
            if other.tuple_id != item.tuple_id
            and set(other.task_ids) & set(item.task_ids)
        ]
        alternative_efficiency = max(
            (
                _marginal_risk(other, covered, miss_penalty)
                / (1.0 + max(0.0, _tuple_secondary_estimate(other, config)))
                for other in alternatives
            ),
            default=0.0,
        )
        regret = efficiency - alternative_efficiency
        return risk, efficiency, regret, cost

    if name == "risk":
        return tuple(
            sorted(
                candidates,
                key=lambda item: (-values(item)[0], values(item)[3], item.tuple_id),
            )
        )
    if name == "efficiency":
        return tuple(
            sorted(
                candidates,
                key=lambda item: (-values(item)[1], -values(item)[0], item.tuple_id),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (-values(item)[2], -values(item)[0], item.tuple_id),
        )
    )


@dataclass(frozen=True)
class ManualAlnsBaseline:
    seed: int = 0
    max_iterations: int | None = 20
    projection_limit: int | None = 100
    time_limit_s: float = 1.0
    destroy_fraction: float = 0.25
    initial_temperature: float = 1.0
    cooling_rate: float = 0.95
    reaction_factor: float = 0.2
    total_time_limit_s: float | None = None

    def __post_init__(self):
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.projection_limit is not None and self.projection_limit <= 0:
            raise ValueError("projection_limit must be positive")
        if self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")
        if self.total_time_limit_s is not None and self.total_time_limit_s <= 0:
            raise ValueError("total_time_limit_s must be positive")
        if self.max_iterations is None and self.total_time_limit_s is None:
            raise ValueError("unbounded iterations require total_time_limit_s")
        if not 0 < self.destroy_fraction <= 1:
            raise ValueError("destroy_fraction must be in (0, 1]")
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if not 0 < self.cooling_rate <= 1:
            raise ValueError("cooling_rate must be in (0, 1]")
        if not 0 < self.reaction_factor <= 1:
            raise ValueError("reaction_factor must be in (0, 1]")

    def solve(
        self,
        *,
        state,
        task_ids,
        miss_penalty,
        validation_state,
        operation_capacity,
        config,
        incumbent_observer=None,
    ):
        started = perf_counter()
        def remaining_time():
            if self.total_time_limit_s is None:
                return self.time_limit_s
            return self.total_time_limit_s - (perf_counter() - started)

        def repair_limit():
            return max(0.001, min(self.time_limit_s, remaining_time()))

        def projection_exhausted(calls):
            return self.projection_limit is not None and calls >= self.projection_limit

        initializer_limit = max(
            1,
            min(
                len(state.tuples) + 1,
                max(1, self.projection_limit // 2)
                if self.projection_limit is not None else len(state.tuples) + 1,
            ),
        )
        initializer = RiskPriorityGreedyBaseline(
            time_limit_s=self.time_limit_s,
            projection_limit=initializer_limit,
            total_time_limit_s=(
                0.25 * self.total_time_limit_s
                if self.total_time_limit_s is not None else None
            ),
        ).solve(
            state=state,
            task_ids=task_ids,
            miss_penalty=miss_penalty,
            validation_state=validation_state,
            operation_capacity=operation_capacity,
            config=config,
            incumbent_observer=incumbent_observer,
        )
        if not initializer.validation_ok:
            return initializer
        rng = Random(self.seed)
        tuple_by_id = {item.tuple_id: item for item in state.tuples}
        admitted = tuple(item for item in state.tuples if item.admitted)
        mandatory = _mandatory_tuple_ids(state, validation_state)
        current = initializer
        best = initializer
        solver_calls = initializer.solver_calls
        trace = [
            BaselineTraceRecord(
                0,
                "initialize:greedy",
                True,
                True,
                initializer.best_primary,
                initializer.best_secondary,
                "greedy_initializer",
            )
        ]
        accepted = []
        temperature = self.initial_temperature
        destroy_weights = {name: 1.0 for name in DESTROY_OPERATORS}
        repair_weights = {name: 1.0 for name in REPAIR_OPERATORS}
        completed_iterations = 0
        stop_reason = "max_iterations"

        iteration = 0
        while self.max_iterations is None or iteration < self.max_iterations:
            if remaining_time() < 0.005:
                stop_reason = "total_time_limit"
                break
            if projection_exhausted(solver_calls):
                stop_reason = "projection_limit"
                break
            if iteration < max(len(DESTROY_OPERATORS), len(REPAIR_OPERATORS)):
                destroy_name = DESTROY_OPERATORS[iteration % len(DESTROY_OPERATORS)]
                repair_name = REPAIR_OPERATORS[iteration % len(REPAIR_OPERATORS)]
            else:
                destroy_name = _weighted_operator(
                    DESTROY_OPERATORS,
                    destroy_weights,
                    rng,
                )
                repair_name = _weighted_operator(
                    REPAIR_OPERATORS,
                    repair_weights,
                    rng,
                )
            selected = set(current.selected_tuple_ids)
            removable_count = len(selected - set(mandatory))
            remove_count = (
                max(1, ceil(removable_count * self.destroy_fraction))
                if removable_count
                else 0
            )
            removed = _destroy_tuple_ids(
                destroy_name,
                selected,
                mandatory,
                tuple_by_id,
                remove_count,
                rng,
            )
            partial_ids = selected - set(removed)
            if removed:
                candidate = project_tuple_selection(
                    state=state,
                    task_ids=task_ids,
                    miss_penalty=miss_penalty,
                    validation_state=validation_state,
                    operation_capacity=operation_capacity,
                    config=config,
                    selected_tuple_ids=partial_ids,
                    time_limit_s=repair_limit(),
                )
                solver_calls += candidate.solver_calls
            else:
                candidate = current

            if candidate.validation_ok:
                repair_candidates = [
                    item for item in admitted if item.tuple_id not in partial_ids
                ]
                for item in _repair_order(
                    repair_name,
                    repair_candidates,
                    partial_ids,
                    tuple_by_id,
                    miss_penalty,
                    config,
                ):
                    if projection_exhausted(solver_calls) or remaining_time() < 0.005:
                        break
                    projected = project_tuple_selection(
                        state=state,
                        task_ids=task_ids,
                        miss_penalty=miss_penalty,
                        validation_state=validation_state,
                        operation_capacity=operation_capacity,
                        config=config,
                        selected_tuple_ids=partial_ids | {item.tuple_id},
                        time_limit_s=repair_limit(),
                    )
                    solver_calls += projected.solver_calls
                    if projected.validation_ok and _lexicographically_better(
                        projected,
                        candidate,
                        float(config.epsilon_u),
                        float(config.epsilon_f),
                    ):
                        partial_ids.add(item.tuple_id)
                        candidate = projected
                    if incumbent_observer is not None:
                        observed = candidate if _lexicographically_better(
                            candidate, best, float(config.epsilon_u), float(config.epsilon_f)
                        ) else best
                        incumbent_observer(replace(observed, solver_calls=solver_calls))

            accepted_now = False
            if candidate.validation_ok:
                if _lexicographically_better(
                    candidate,
                    current,
                    float(config.epsilon_u),
                    float(config.epsilon_f),
                ):
                    accepted_now = True
                elif abs(candidate.best_primary - current.best_primary) <= float(
                    config.epsilon_u
                ):
                    delta = max(0.0, candidate.best_secondary - current.best_secondary)
                    accepted_now = delta <= float(config.epsilon_f) or rng.random() < exp(
                        -delta / max(temperature, 1e-12)
                    )
            if accepted_now:
                current = candidate
            improved_best = candidate.validation_ok and _lexicographically_better(
                candidate,
                best,
                float(config.epsilon_u),
                float(config.epsilon_f),
            )
            if improved_best:
                best = candidate
            reward = (
                5.0
                if improved_best
                else 2.0
                if accepted_now
                else 0.5
                if candidate.validation_ok
                else 0.0
            )
            destroy_weights[destroy_name] = (
                (1.0 - self.reaction_factor) * destroy_weights[destroy_name]
                + self.reaction_factor * reward
            )
            repair_weights[repair_name] = (
                (1.0 - self.reaction_factor) * repair_weights[repair_name]
                + self.reaction_factor * reward
            )
            completed_iterations += 1
            record = BaselineTraceRecord(
                completed_iterations,
                f"destroy={destroy_name};repair={repair_name}",
                accepted_now,
                candidate.validation_ok,
                candidate.best_primary,
                candidate.best_secondary,
                "accepted" if accepted_now else candidate.stop_reason,
            )
            trace.append(record)
            if accepted_now:
                accepted.append(record)
            if incumbent_observer is not None:
                incumbent_observer(replace(best, solver_calls=solver_calls))
            temperature *= self.cooling_rate
            iteration += 1

        return BaselineResult(
            initializer.initial_primary,
            initializer.initial_secondary,
            best.best_primary,
            best.best_secondary,
            best.selected_tuple_ids,
            best.selected_arc_ids,
            tuple(trace),
            tuple(accepted),
            stop_reason,
            best.status,
            best.validation_ok,
            solver_calls,
            completed_iterations,
            perf_counter() - started,
            best.primary_status,
            best.secondary_status,
            best.primary_bound,
            best.secondary_bound,
            tuple(
                sorted(
                    [(f"destroy:{name}", value) for name, value in destroy_weights.items()]
                    + [(f"repair:{name}", value) for name, value in repair_weights.items()]
                )
            ),
        )


__all__ = ["ManualAlnsBaseline"]
