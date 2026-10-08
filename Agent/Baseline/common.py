from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace

from Agent.Proposed.milp import (
    MilpArc,
    MilpTuple,
    RollingMilpInput,
    solve_lexicographic,
)
from Agent.Proposed.validator import ValidationReport, validate_schedule


@dataclass(frozen=True)
class BaselineTraceRecord:
    iteration: int
    action: str
    accepted: bool
    validation_ok: bool
    primary_value: float
    secondary_value: float
    reason: str


@dataclass(frozen=True)
class BaselineResult:
    initial_primary: float
    initial_secondary: float
    best_primary: float
    best_secondary: float
    selected_tuple_ids: tuple[str, ...]
    selected_arc_ids: tuple[str, ...]
    trace: tuple[BaselineTraceRecord, ...]
    accepted_records: tuple[BaselineTraceRecord, ...]
    stop_reason: str
    status: str
    validation_ok: bool
    solver_calls: int
    iterations: int
    runtime_s: float
    primary_status: str = "not_solved"
    secondary_status: str = "not_solved"
    primary_bound: float = float("inf")
    secondary_bound: float = float("inf")
    metadata: tuple[tuple[str, float], ...] = ()


def calculate_objective(
    state,
    task_ids,
    miss_penalty,
    config,
    selected_tuple_ids=None,
    selected_arc_ids=None,
    validation_state=None,
):
    tuple_ids = frozenset(
        state.selected_tuple_ids if selected_tuple_ids is None else selected_tuple_ids
    )
    arc_ids = frozenset(
        state.selected_arc_ids if selected_arc_ids is None else selected_arc_ids
    )
    selected = [item for item in state.tuples if item.tuple_id in tuple_ids]
    covered = {task_id for item in selected for task_id in item.task_ids}
    primary = sum(
        float(miss_penalty[task_id])
        for task_id in task_ids
        if task_id not in covered
    )
    secondary = sum(float(item.secondary_cost) for item in selected)
    secondary += sum(
        float(arc.secondary_cost) for arc in state.arcs if arc.arc_id in arc_ids
    )
    secondary += float(config.time_weight) * sum(
        float(arc.duty_time) for arc in state.arcs if arc.arc_id in arc_ids
    )
    secondary += float(config.risk_weight) * sum(
        float(item.risk_exposure) for item in selected
    )
    secondary += float(config.energy_weight) * sum(
        float(item.expected_energy) for item in selected
    )
    completion_layers = [
        layer for item in selected for _, layer in item.completion_layers
    ]
    return_layers = [
        arc.arrival_layer
        for arc in state.arcs
        if arc.arc_id in arc_ids and arc.head == "sink"
    ]
    secondary += float(config.makespan_weight) * max(
        completion_layers + return_layers,
        default=0,
    )
    if validation_state is not None:
        secondary += float(config.plan_change_weight) * sum(
            abs(int(tuple_id in tuple_ids) - int(value))
            for tuple_id, value in validation_state.previous_tuple_values.items()
        )
        secondary += float(config.plan_change_weight) * sum(
            abs(int(arc_id in arc_ids) - int(value))
            for arc_id, value in validation_state.previous_arc_values.items()
        )
    return float(primary), float(secondary)


def _milp_tuples(state):
    return tuple(
        MilpTuple(
            tuple_id=item.tuple_id,
            task_ids=item.task_ids,
            secondary_cost=item.secondary_cost,
            vehicle_id=item.vehicle_id,
            launch_node=item.launch_node,
            recovery_node=item.recovery_node,
            required_arc_ids=item.required_arc_ids,
            interval_keys=item.interval_keys,
            operation_keys=item.operation_keys,
            completion_layers=item.completion_layers,
            risk_exposure=item.risk_exposure,
            expected_energy=item.expected_energy,
            is_air_residual=item.is_air_residual,
            launch_layer=item.launch_layer,
            recovery_layer=item.recovery_layer,
        )
        for item in state.tuples
        if item.admitted
    )


def _milp_arcs(state):
    return tuple(
        MilpArc(
            arc_id=arc.arc_id,
            vehicle_id=arc.vehicle_id,
            tail=arc.tail,
            head=arc.head,
            secondary_cost=arc.secondary_cost,
            arrival_layer=arc.arrival_layer,
            duty_time=arc.duty_time,
            departure_layer=arc.departure_layer,
        )
        for arc in state.arcs
    )


def _rolling_vehicle_requirements(validation_state):
    required_vehicle_ids = tuple(
        sorted(
            vehicle_id
            for vehicle_id, vehicle in validation_state.vehicles.items()
            if vehicle.must_return
        )
    )
    max_duty_by_vehicle = {
        vehicle_id: float(vehicle.remaining_duty_time)
        for vehicle_id, vehicle in validation_state.vehicles.items()
        if vehicle.remaining_duty_time is not None
    }
    return required_vehicle_ids, max_duty_by_vehicle


def _validation_context(
    state,
    miss_penalty,
    operation_capacity,
    config,
    validation_state,
):
    return {
        "admitted_tuple_ids": {
            item.tuple_id for item in state.tuples if item.admitted
        },
        "arc_defs": {
            arc.arc_id: (arc.vehicle_id, arc.tail, arc.head) for arc in state.arcs
        },
        "arc_time_defs": {
            arc.arc_id: (
                arc.vehicle_id,
                arc.tail,
                arc.departure_layer,
                arc.head,
                arc.arrival_layer,
            )
            for arc in state.arcs
        },
        "operation_capacity": dict(operation_capacity),
        "miss_penalty": dict(miss_penalty),
        "arc_secondary_cost": {
            arc.arc_id: arc.secondary_cost for arc in state.arcs
        },
        "arc_duty_time": {arc.arc_id: arc.duty_time for arc in state.arcs},
        "time_weight": float(config.time_weight),
        "risk_weight": float(config.risk_weight),
        "energy_weight": float(config.energy_weight),
        "makespan_weight": float(config.makespan_weight),
        "plan_change_weight": float(config.plan_change_weight),
        "previous_tuple_values": dict(validation_state.previous_tuple_values),
        "previous_arc_values": dict(validation_state.previous_arc_values),
        "tolerance": max(float(config.epsilon_u), float(config.epsilon_f)),
    }


def _infeasible_result(
    initial_primary,
    initial_secondary,
    requested,
    reason,
    runtime_s,
    solver_calls=0,
):
    trace = (
        BaselineTraceRecord(
            0,
            "project",
            False,
            False,
            float("inf"),
            float("inf"),
            reason,
        ),
    )
    return BaselineResult(
        initial_primary,
        initial_secondary,
        float("inf"),
        float("inf"),
        tuple(sorted(requested)),
        (),
        trace,
        (),
        reason,
        "infeasible",
        False,
        solver_calls,
        1,
        runtime_s,
    )


def _result_from_solved(
    *,
    state,
    task_ids,
    miss_penalty,
    validation_state,
    operation_capacity,
    config,
    solved,
    initial_primary,
    initial_secondary,
    started,
    action,
    success_reason,
    expected_tuple_ids=None,
):
    tuple_by_id = {item.tuple_id: item for item in state.tuples}
    selected_tuples = tuple(
        tuple_by_id[tuple_id] for tuple_id in solved.selected_tuple_ids
    )
    covered = {task_id for item in selected_tuples for task_id in item.task_ids}
    missed = tuple(sorted(set(task_ids) - covered))
    completion_layers = {
        task_id: layer
        for item in selected_tuples
        for task_id, layer in item.completion_layers
    }
    arc_by_id = {arc.arc_id: arc for arc in state.arcs}
    active_vehicle_ids = {
        arc_by_id[arc_id].vehicle_id
        for arc_id in solved.selected_arc_ids
        if arc_by_id[arc_id].head == "sink"
    }
    return_layers = {
        vehicle_id: max(
            (
                arc_by_id[arc_id].arrival_layer
                for arc_id in solved.selected_arc_ids
                if arc_by_id[arc_id].vehicle_id == vehicle_id
                and arc_by_id[arc_id].head == "sink"
            ),
            default=0,
        )
        for vehicle_id in active_vehicle_ids
    }
    schedule = SimpleNamespace(
        selected_tuples=selected_tuples,
        selected_arc_ids=solved.selected_arc_ids,
        missed_task_ids=missed,
        completion_layers=completion_layers,
        vehicle_return_layers=return_layers,
        makespan=solved.makespan,
        primary_value=solved.missed_weight,
        secondary_value=solved.secondary_value,
    )
    validation = validate_schedule(
        schedule,
        validation_state,
        _validation_context(
            state,
            miss_penalty,
            operation_capacity,
            config,
            validation_state,
        ),
    )
    expected = (
        None if expected_tuple_ids is None else frozenset(expected_tuple_ids)
    )
    preserved = expected is None or frozenset(solved.selected_tuple_ids) == expected
    validation_ok = validation.ok and preserved
    reason = (
        success_reason
        if validation_ok
        else "tuple_selection_changed"
        if not preserved
        else validation.first_violation or "validation_failed"
    )
    record = BaselineTraceRecord(
        0,
        action,
        validation_ok,
        validation_ok,
        solved.missed_weight,
        solved.secondary_value,
        reason,
    )
    return BaselineResult(
        initial_primary,
        initial_secondary,
        solved.missed_weight,
        solved.secondary_value,
        solved.selected_tuple_ids,
        solved.selected_arc_ids,
        (record,),
        (record,) if validation_ok else (),
        reason,
        solved.status,
        validation_ok,
        1,
        1,
        perf_counter() - started,
        solved.primary_status,
        solved.secondary_status,
        solved.primary_bound,
        solved.secondary_bound,
    )


def project_tuple_selection(
    *,
    state,
    task_ids,
    miss_penalty,
    validation_state,
    operation_capacity,
    config,
    selected_tuple_ids,
    time_limit_s=None,
):
    started = perf_counter()
    requested = frozenset(selected_tuple_ids)
    initial_primary, initial_secondary = calculate_objective(
        state,
        task_ids,
        miss_penalty,
        config,
        validation_state=validation_state,
    )
    tuple_by_id = {item.tuple_id: item for item in state.tuples}
    unknown = requested - set(tuple_by_id)
    inadmissible = {
        tuple_id
        for tuple_id in requested
        if tuple_id in tuple_by_id and not tuple_by_id[tuple_id].admitted
    }
    locked_selected = (
        frozenset(validation_state.locked_tuple_ids) & state.selected_tuple_ids
    )
    if unknown:
        return _infeasible_result(
            initial_primary,
            initial_secondary,
            requested,
            "unknown_tuple",
            perf_counter() - started,
        )
    if inadmissible:
        return _infeasible_result(
            initial_primary,
            initial_secondary,
            requested,
            "tuple_not_admitted",
            perf_counter() - started,
        )
    if not locked_selected <= requested:
        return _infeasible_result(
            initial_primary,
            initial_secondary,
            requested,
            "locked_tuple_removed",
            perf_counter() - started,
        )

    tuples = _milp_tuples(state)
    arcs = _milp_arcs(state)
    fixed_tuple_values = {
        item.tuple_id: int(item.tuple_id in requested) for item in tuples
    }
    fixed_arc_values = {
        arc_id: int(arc_id in state.selected_arc_ids)
        for arc_id in validation_state.locked_arc_ids
        if arc_id in {arc.arc_id for arc in arcs}
    }
    required_vehicle_ids, max_duty_by_vehicle = _rolling_vehicle_requirements(
        validation_state
    )
    data = RollingMilpInput(
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
    )
    solved = solve_lexicographic(
        data,
        float(time_limit_s or config.repair_time_limit_s),
        float(config.epsilon_u),
    )
    if solved.status == "infeasible":
        return _infeasible_result(
            initial_primary,
            initial_secondary,
            requested,
            "projection_infeasible",
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
        action="project",
        success_reason="projected",
        expected_tuple_ids=requested,
    )


__all__ = [
    "BaselineResult",
    "BaselineTraceRecord",
    "calculate_objective",
    "project_tuple_selection",
]
