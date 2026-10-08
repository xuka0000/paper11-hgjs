from dataclasses import dataclass
from types import SimpleNamespace

from .locks import partition_decisions
from .milp import MilpArc, MilpTuple, RollingMilpInput, solve_lexicographic
from .validator import ValidationReport, validate_schedule


@dataclass(frozen=True)
class RepairResult:
    accepted: bool
    reason: str
    selected_tuple_ids: tuple[str, ...]
    selected_arc_ids: tuple[str, ...]
    primary_value: float
    secondary_value: float
    validation: ValidationReport
    primary_status: str
    secondary_status: str
    primary_bound: float = float("inf")
    secondary_bound: float = float("inf")
    runtime_s: float = 0.0
    primary_proven_optimal: bool = False
    solver_nodes: int = 0
    incumbent_hint_accepted: bool | None = None
    incumbent_hint_violation_count: int = 0


def _current_objective(
    state,
    task_ids,
    miss_penalty,
    makespan_weight=0.0,
    time_weight=0.0,
    risk_weight=0.0,
    energy_weight=0.0,
    plan_change_weight=0.0,
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
    secondary += float(makespan_weight) * max(
        completion_layers + return_layers,
        default=0,
    )
    secondary += float(time_weight) * sum(
        arc.duty_time
        for arc in state.arcs
        if arc.arc_id in state.selected_arc_ids
    )
    secondary += float(risk_weight) * sum(
        item.risk_exposure for item in selected
    )
    secondary += float(energy_weight) * sum(
        item.expected_energy for item in selected
    )
    current_tuple_ids = set(state.selected_tuple_ids)
    current_arc_ids = set(state.selected_arc_ids)
    plan_change = sum(
        abs(int(tuple_id in current_tuple_ids) - int(value))
        for tuple_id, value in (previous_tuple_values or {}).items()
    )
    plan_change += sum(
        abs(int(arc_id in current_arc_ids) - int(value))
        for arc_id, value in (previous_arc_values or {}).items()
    )
    secondary += float(plan_change_weight) * plan_change
    return float(primary), float(secondary)


def repair_region(
    state,
    closure,
    task_ids,
    miss_penalty,
    validation_state,
    operation_capacity,
    time_limit_s,
    epsilon_u=1e-8,
    epsilon_f=1e-8,
    makespan_weight=0.0,
    time_weight=0.0,
    risk_weight=0.0,
    energy_weight=0.0,
    plan_change_weight=0.0,
    unsafe_tuple_ids=frozenset(),
    solver_backend="CBC",
    node_limit=None,
):
    if not closure.accepted:
        report = ValidationReport(False, closure.reason, 1.0, closure.reason)
        return RepairResult(
            False,
            closure.reason,
            tuple(sorted(state.selected_tuple_ids)),
            tuple(sorted(state.selected_arc_ids)),
            float("inf"),
            float("inf"),
            report,
            "not_solved",
            "not_solved",
        )

    released_tuples = set(closure.tuple_ids)
    released_arcs = set(closure.arc_ids)
    partition = partition_decisions(
        validation_state,
        frozenset(unsafe_tuple_ids),
    )
    tuple_by_id = {item.tuple_id: item for item in state.tuples}
    arc_by_id = {arc.arc_id: arc for arc in state.arcs}
    tuple_ids = set(tuple_by_id)
    arc_ids = set(arc_by_id)
    previous_tuple_values = dict(validation_state.previous_tuple_values)
    previous_arc_values = dict(validation_state.previous_arc_values)

    active_tuple_ids = (
        (released_tuples & tuple_ids)
        | (set(state.selected_tuple_ids) & tuple_ids)
        | (set(partition.fixed_tuple_ids) & tuple_ids)
        | (set(previous_tuple_values) & tuple_ids)
    )
    active_arc_ids = (
        (released_arcs & arc_ids)
        | (set(state.selected_arc_ids) & arc_ids)
        | (set(partition.fixed_arc_ids) & arc_ids)
        | (set(previous_arc_values) & arc_ids)
    )
    active_arc_ids.update(
        arc_id
        for tuple_id in active_tuple_ids
        for arc_id in tuple_by_id[tuple_id].required_arc_ids
        if arc_id in arc_ids
    )

    milp_tuples = tuple(
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
        if item.admitted and item.tuple_id in active_tuple_ids
    )
    milp_arcs = tuple(
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
        if arc.arc_id in active_arc_ids
    )
    milp_tuple_ids = {item.tuple_id for item in milp_tuples}
    milp_arc_ids = {arc.arc_id for arc in milp_arcs}
    fixed_tuple_values = {
        tuple_id: int(tuple_id in state.selected_tuple_ids)
        for tuple_id in milp_tuple_ids
        if tuple_id not in released_tuples
    }
    fixed_arc_values = {
        arc_id: int(arc_id in state.selected_arc_ids)
        for arc_id in milp_arc_ids
        if arc_id not in released_arcs
    }
    for tuple_id in partition.fixed_tuple_ids & milp_tuple_ids:
        fixed_tuple_values[tuple_id] = int(tuple_id in state.selected_tuple_ids)
    for arc_id in partition.fixed_arc_ids & milp_arc_ids:
        fixed_arc_values[arc_id] = int(arc_id in state.selected_arc_ids)
    data = RollingMilpInput(
        task_ids=tuple(task_ids),
        miss_penalty=dict(miss_penalty),
        tuples=milp_tuples,
        fixed_tuple_values=fixed_tuple_values,
        vehicle_ids=tuple(sorted({arc.vehicle_id for arc in state.arcs})),
        required_vehicle_ids=tuple(
            sorted(
                vehicle_id
                for vehicle_id, vehicle in validation_state.vehicles.items()
                if vehicle.must_return
            )
        ),
        max_duty_by_vehicle={
            vehicle_id: float(vehicle.remaining_duty_time)
            for vehicle_id, vehicle in validation_state.vehicles.items()
            if vehicle.remaining_duty_time is not None
        },
        arcs=milp_arcs,
        operation_capacity=dict(operation_capacity),
        fixed_arc_values=fixed_arc_values,
        makespan_weight=float(makespan_weight),
        time_weight=float(time_weight),
        risk_weight=float(risk_weight),
        energy_weight=float(energy_weight),
        previous_tuple_values=previous_tuple_values,
        previous_arc_values=previous_arc_values,
        plan_change_weight=float(plan_change_weight),
    )
    solved = solve_lexicographic(
        data,
        time_limit_s,
        epsilon_u,
        solver_backend=solver_backend,
        node_limit=node_limit,
    )
    if solved.status == "infeasible":
        report = ValidationReport(False, "repair_infeasible", 1.0, closure.reason)
        return RepairResult(
            False,
            "repair_infeasible",
            tuple(sorted(state.selected_tuple_ids)),
            tuple(sorted(state.selected_arc_ids)),
            float("inf"),
            float("inf"),
            report,
            solved.primary_status,
            solved.secondary_status,
            solved.primary_bound,
            solved.secondary_bound,
            solved.runtime_s,
            solved.primary_proven_optimal,
            solved.solver_nodes,
        )

    selected_tuples = tuple(tuple_by_id[tuple_id] for tuple_id in solved.selected_tuple_ids)
    covered = {task for item in selected_tuples for task in item.task_ids}
    missed = tuple(sorted(set(task_ids) - covered))
    completion_layers = {
        task: layer
        for item in selected_tuples
        for task, layer in item.completion_layers
    }
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
    context = {
        "admitted_tuple_ids": {item.tuple_id for item in state.tuples if item.admitted},
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
        "tolerance": max(epsilon_u, epsilon_f),
        "makespan_weight": float(makespan_weight),
        "time_weight": float(time_weight),
        "risk_weight": float(risk_weight),
        "energy_weight": float(energy_weight),
        "arc_duty_time": {
            arc.arc_id: arc.duty_time for arc in state.arcs
        },
        "previous_tuple_values": previous_tuple_values,
        "previous_arc_values": previous_arc_values,
        "plan_change_weight": float(plan_change_weight),
    }
    report = validate_schedule(schedule, validation_state, context)
    current_primary, current_secondary = _current_objective(
        state,
        task_ids,
        miss_penalty,
        makespan_weight,
        time_weight,
        risk_weight,
        energy_weight,
        plan_change_weight,
        previous_arc_values,
        previous_tuple_values,
    )
    nonworse = solved.missed_weight <= current_primary + epsilon_u and (
        solved.missed_weight < current_primary - epsilon_u
        or solved.secondary_value <= current_secondary + epsilon_f
    )
    accepted = report.ok and nonworse
    reason = "accepted" if accepted else report.first_violation or "lexicographic_worse"
    return RepairResult(
        accepted,
        reason,
        solved.selected_tuple_ids if accepted else tuple(sorted(state.selected_tuple_ids)),
        solved.selected_arc_ids if accepted else tuple(sorted(state.selected_arc_ids)),
        solved.missed_weight if accepted else current_primary,
        solved.secondary_value if accepted else current_secondary,
        report,
        solved.primary_status,
        solved.secondary_status,
        solved.primary_bound,
        solved.secondary_bound,
        solved.runtime_s,
        solved.primary_proven_optimal,
        solved.solver_nodes,
    )
