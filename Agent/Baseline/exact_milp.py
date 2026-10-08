from dataclasses import dataclass
from time import perf_counter

from Agent.Proposed.milp import RollingMilpInput, solve_lexicographic

from .common import (
    _infeasible_result,
    _milp_arcs,
    _milp_tuples,
    _result_from_solved,
    _rolling_vehicle_requirements,
    calculate_objective,
)


@dataclass(frozen=True)
class ExactMilpBaseline:
    max_tasks: int = 12
    time_limit_s: float = 5.0

    def __post_init__(self):
        if self.max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        if self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")

    def solve(
        self,
        *,
        state,
        task_ids,
        miss_penalty,
        validation_state,
        operation_capacity,
        config,
    ):
        if len(tuple(task_ids)) > self.max_tasks:
            raise ValueError(
                f"instance exceeds small-scale limit: "
                f"tasks={len(tuple(task_ids))} limit={self.max_tasks}"
            )
        started = perf_counter()
        initial_primary, initial_secondary = calculate_objective(
            state,
            task_ids,
            miss_penalty,
            config,
            validation_state=validation_state,
        )
        tuples = _milp_tuples(state)
        arcs = _milp_arcs(state)
        tuple_ids = {item.tuple_id for item in tuples}
        arc_ids = {arc.arc_id for arc in arcs}
        fixed_tuple_values = {
            tuple_id: int(tuple_id in state.selected_tuple_ids)
            for tuple_id in validation_state.locked_tuple_ids
            if tuple_id in tuple_ids
        }
        fixed_arc_values = {
            arc_id: int(arc_id in state.selected_arc_ids)
            for arc_id in validation_state.locked_arc_ids
            if arc_id in arc_ids
        }
        required_vehicle_ids, max_duty_by_vehicle = _rolling_vehicle_requirements(
            validation_state
        )
        solved = solve_lexicographic(
            RollingMilpInput(
                task_ids=tuple(task_ids),
                miss_penalty=dict(miss_penalty),
                tuples=tuples,
                fixed_tuple_values=fixed_tuple_values,
                vehicle_ids=tuple(sorted({arc.vehicle_id for arc in arcs})),
                required_vehicle_ids=required_vehicle_ids,
                max_duty_by_vehicle=max_duty_by_vehicle,
                arcs=arcs,
                operation_capacity=dict(operation_capacity),
                fixed_arc_values=fixed_arc_values,
                previous_tuple_values=dict(validation_state.previous_tuple_values),
                previous_arc_values=dict(validation_state.previous_arc_values),
                plan_change_weight=float(config.plan_change_weight),
                makespan_weight=float(config.makespan_weight),
                time_weight=float(config.time_weight),
                risk_weight=float(config.risk_weight),
                energy_weight=float(config.energy_weight),
            ),
            self.time_limit_s,
            float(config.epsilon_u),
        )
        if solved.status == "infeasible":
            return _infeasible_result(
                initial_primary,
                initial_secondary,
                (),
                "complete_milp_infeasible",
                perf_counter() - started,
                1,
            )
        return _result_from_solved(
            state=state,
            task_ids=task_ids,
            miss_penalty=miss_penalty,
            validation_state=validation_state,
            operation_capacity=operation_capacity,
            config=config,
            solved=solved,
            initial_primary=initial_primary,
            initial_secondary=initial_secondary,
            started=started,
            action="complete_milp",
            success_reason="complete_milp_solved",
        )


__all__ = ["ExactMilpBaseline"]
