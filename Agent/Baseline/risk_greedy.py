from dataclasses import dataclass, replace
from time import perf_counter

from .common import (
    BaselineResult,
    BaselineTraceRecord,
    calculate_objective,
    project_tuple_selection,
)


def _mandatory_tuple_ids(state, validation_state):
    tuple_ids = {item.tuple_id for item in state.tuples}
    locked = (
        frozenset(validation_state.locked_tuple_ids)
        & state.selected_tuple_ids
        & tuple_ids
    )
    residual = {
        item.tuple_id for item in state.tuples if item.is_air_residual and item.admitted
    }
    return frozenset(locked | residual)


def _tuple_secondary_estimate(item, config):
    completion = max(
        (layer for _, layer in item.completion_layers),
        default=item.recovery_layer,
    )
    return (
        float(item.secondary_cost)
        + float(config.risk_weight) * float(item.risk_exposure)
        + float(config.energy_weight) * float(item.expected_energy)
        + float(config.makespan_weight) * float(completion)
    )


def _candidate_key(item, covered, miss_penalty, config):
    marginal_risk = sum(
        float(miss_penalty[task_id])
        for task_id in item.task_ids
        if task_id not in covered
    )
    return -marginal_risk, _tuple_secondary_estimate(item, config), item.tuple_id


@dataclass(frozen=True)
class RiskPriorityGreedyBaseline:
    time_limit_s: float = 2.0
    projection_limit: int | None = None
    total_time_limit_s: float | None = None

    def __post_init__(self):
        if self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")
        if self.projection_limit is not None and self.projection_limit <= 0:
            raise ValueError("projection_limit must be positive")
        if self.total_time_limit_s is not None and self.total_time_limit_s <= 0:
            raise ValueError("total_time_limit_s must be positive")

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
        initial_primary, initial_secondary = calculate_objective(
            state,
            task_ids,
            miss_penalty,
            config,
            validation_state=validation_state,
        )
        selected = set(_mandatory_tuple_ids(state, validation_state))
        current = project_tuple_selection(
            state=state,
            task_ids=task_ids,
            miss_penalty=miss_penalty,
            validation_state=validation_state,
            operation_capacity=operation_capacity,
            config=config,
            selected_tuple_ids=selected,
            time_limit_s=repair_limit(),
        )
        if not current.validation_ok:
            return current

        trace = [
            BaselineTraceRecord(
                0,
                "initialize",
                True,
                True,
                current.best_primary,
                current.best_secondary,
                "mandatory_projection",
            )
        ]
        accepted = []
        solver_calls = 1
        if incumbent_observer is not None:
            incumbent_observer(current)
        attempts = 0
        limit = self.projection_limit or (len(state.tuples) + 1)
        remaining = {
            item.tuple_id: item
            for item in state.tuples
            if item.admitted and item.tuple_id not in selected
        }
        tuple_by_id = {item.tuple_id: item for item in state.tuples}
        stop_reason = "candidate_exhausted"
        while remaining and solver_calls < limit:
            if remaining_time() < 0.005:
                stop_reason = "total_time_limit"
                break
            covered = {
                task_id
                for tuple_id in selected
                for task_id in tuple_by_id[tuple_id].task_ids
            }
            candidate = min(
                remaining.values(),
                key=lambda item: _candidate_key(
                    item,
                    covered,
                    miss_penalty,
                    config,
                ),
            )
            remaining.pop(candidate.tuple_id)
            marginal_risk = sum(
                float(miss_penalty[task_id])
                for task_id in candidate.task_ids
                if task_id not in covered
            )
            if marginal_risk <= 0:
                continue
            attempts += 1
            projected = project_tuple_selection(
                state=state,
                task_ids=task_ids,
                miss_penalty=miss_penalty,
                validation_state=validation_state,
                operation_capacity=operation_capacity,
                config=config,
                selected_tuple_ids=selected | {candidate.tuple_id},
                time_limit_s=repair_limit(),
            )
            solver_calls += projected.solver_calls
            accepted_now = (
                projected.validation_ok
                and projected.best_primary
                < current.best_primary - float(config.epsilon_u)
            )
            record = BaselineTraceRecord(
                attempts,
                f"add:{candidate.tuple_id}",
                accepted_now,
                projected.validation_ok,
                projected.best_primary,
                projected.best_secondary,
                "risk_insertion" if accepted_now else projected.stop_reason,
            )
            trace.append(record)
            if accepted_now:
                selected.add(candidate.tuple_id)
                current = projected
                accepted.append(record)
            if incumbent_observer is not None:
                incumbent_observer(replace(current, solver_calls=solver_calls))
        if remaining and solver_calls >= limit:
            stop_reason = "projection_limit"
        return BaselineResult(
            initial_primary,
            initial_secondary,
            current.best_primary,
            current.best_secondary,
            current.selected_tuple_ids,
            current.selected_arc_ids,
            tuple(trace),
            tuple(accepted),
            stop_reason,
            current.status,
            current.validation_ok,
            solver_calls,
            attempts,
            perf_counter() - started,
            current.primary_status,
            current.secondary_status,
            current.primary_bound,
            current.secondary_bound,
        )


__all__ = ["RiskPriorityGreedyBaseline"]
