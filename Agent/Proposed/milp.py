from dataclasses import dataclass
from time import perf_counter

from ortools.linear_solver import pywraplp


@dataclass(frozen=True)
class MilpTuple:
    tuple_id: str
    task_ids: tuple[int, ...]
    secondary_cost: float
    vehicle_id: int = 1
    launch_node: str = ""
    recovery_node: str = ""
    required_arc_ids: tuple[str, ...] = ()
    interval_keys: tuple[str, ...] = ()
    operation_keys: tuple[str, ...] = ()
    completion_layers: tuple[tuple[int, int], ...] = ()
    risk_exposure: float = 0.0
    expected_energy: float = 0.0
    is_air_residual: bool = False
    launch_layer: int = 0
    recovery_layer: int = 0


@dataclass(frozen=True)
class MilpArc:
    arc_id: str
    vehicle_id: int
    tail: str
    head: str
    secondary_cost: float = 0.0
    arrival_layer: int = 0
    duty_time: float = 0.0
    departure_layer: int = 0


@dataclass(frozen=True)
class RollingMilpInput:
    task_ids: tuple[int, ...]
    miss_penalty: dict[int, float]
    tuples: tuple[MilpTuple, ...]
    conflicts: tuple[tuple[str, str], ...] = ()
    fixed_tuple_values: dict[str, int] | None = None
    vehicle_ids: tuple[int, ...] = ()
    required_vehicle_ids: tuple[int, ...] = ()
    max_duty_by_vehicle: dict[int, float] | None = None
    arcs: tuple[MilpArc, ...] = ()
    source_node: str = "source"
    sink_node: str = "sink"
    operation_capacity: dict[str, int] | None = None
    fixed_arc_values: dict[str, int] | None = None
    previous_arc_values: dict[str, int] | None = None
    previous_tuple_values: dict[str, int] | None = None
    plan_change_weight: float = 0.0
    makespan_weight: float = 0.0
    risk_weight: float = 0.0
    energy_weight: float = 0.0
    time_weight: float = 0.0


@dataclass(frozen=True)
class RollingMilpResult:
    status: str
    selected_tuple_ids: tuple[str, ...]
    selected_arc_ids: tuple[str, ...]
    missed_weight: float
    secondary_value: float
    primary_status: str
    secondary_status: str
    primary_bound: float
    secondary_bound: float
    runtime_s: float
    makespan: float = float("inf")
    risk_exposure: float = 0.0
    expected_energy: float = 0.0
    duty_time: float = 0.0
    primary_proven_optimal: bool = False
    solver_nodes: int = 0


def _status_name(status: int) -> str:
    if status == pywraplp.Solver.OPTIMAL:
        return "optimal"
    if status == pywraplp.Solver.FEASIBLE:
        return "feasible"
    return "infeasible"


def solve_lexicographic(
    data: RollingMilpInput,
    time_limit_s: float | None,
    epsilon_u: float = 1e-7,
    *,
    solver_backend: str = "CBC",
    node_limit: int | None = None,
) -> RollingMilpResult:
    solver_backend = str(solver_backend).upper()
    if solver_backend not in {"CBC", "SCIP"}:
        raise ValueError("solver_backend must be CBC or SCIP")
    if node_limit is not None:
        if solver_backend != "SCIP":
            raise ValueError("deterministic node_limit requires SCIP")
        if int(node_limit) <= 0:
            raise ValueError("node_limit must be positive")
    elif time_limit_s is None or float(time_limit_s) <= 0:
        raise ValueError("time_limit_s must be positive without node_limit")
    started = perf_counter()
    solver = pywraplp.Solver.CreateSolver(solver_backend)
    if solver is None:
        raise RuntimeError(f"OR-Tools {solver_backend} is unavailable")
    solver.SetNumThreads(1)
    if node_limit is None:
        solver.SetTimeLimit(int(float(time_limit_s) * 1000))
    else:
        parameters = "\n".join(
            (
                f"limits/nodes = {int(node_limit)}",
                "parallel/maxnthreads = 1",
                "randomization/randomseedshift = 0",
                "randomization/permutationseed = 0",
                "randomization/lpseed = 0",
            )
        )
        if not solver.SetSolverSpecificParametersAsString(parameters):
            raise RuntimeError("failed to configure deterministic SCIP node limit")
    y = {item.tuple_id: solver.BoolVar(f"y[{item.tuple_id}]") for item in data.tuples}
    x = {arc.arc_id: solver.BoolVar(f"x[{arc.vehicle_id},{arc.arc_id}]") for arc in data.arcs}
    g = {vehicle_id: solver.BoolVar(f"g[{vehicle_id}]") for vehicle_id in data.vehicle_ids}
    c_max = solver.NumVar(0.0, solver.infinity(), "C_max")
    source_key = (data.source_node, None)
    sink_key = (data.sink_node, None)
    required_vehicle_ids = set(data.required_vehicle_ids)

    def tail_key(arc):
        if arc.tail == data.source_node:
            return source_key
        return arc.tail, arc.departure_layer

    def head_key(arc):
        if arc.head == data.sink_node:
            return sink_key
        return arc.head, arc.arrival_layer

    for vehicle_id in data.vehicle_ids:
        vehicle_arcs = [arc for arc in data.arcs if arc.vehicle_id == vehicle_id]
        vehicle_tuples = [item for item in data.tuples if item.vehicle_id == vehicle_id]
        if vehicle_id in required_vehicle_ids:
            solver.Add(g[vehicle_id] == 1)
        else:
            solver.Add(g[vehicle_id] <= sum(y[item.tuple_id] for item in vehicle_tuples))
        for item in vehicle_tuples:
            solver.Add(y[item.tuple_id] <= g[vehicle_id])
        if vehicle_id in (data.max_duty_by_vehicle or {}):
            solver.Add(
                sum(arc.duty_time * x[arc.arc_id] for arc in vehicle_arcs)
                <= float(data.max_duty_by_vehicle[vehicle_id])
            )
        nodes = {tail_key(arc) for arc in vehicle_arcs} | {
            head_key(arc) for arc in vehicle_arcs
        }
        solver.Add(
            sum(x[arc.arc_id] for arc in vehicle_arcs if tail_key(arc) == source_key)
            == g[vehicle_id]
        )
        solver.Add(
            sum(x[arc.arc_id] for arc in vehicle_arcs if head_key(arc) == sink_key)
            == g[vehicle_id]
        )
        for node in nodes - {source_key, sink_key}:
            solver.Add(
                sum(x[arc.arc_id] for arc in vehicle_arcs if head_key(arc) == node)
                == sum(x[arc.arc_id] for arc in vehicle_arcs if tail_key(arc) == node)
            )
    for item in data.tuples:
        incoming_launch = [
            arc
            for arc in data.arcs
            if arc.vehicle_id == item.vehicle_id
            and arc.head == item.launch_node
            and arc.arrival_layer == item.launch_layer
        ]
        incoming_recovery = [
            arc
            for arc in data.arcs
            if arc.vehicle_id == item.vehicle_id
            and arc.head == item.recovery_node
            and arc.arrival_layer == item.recovery_layer
        ]
        if not incoming_recovery or (
            not item.is_air_residual
            and item.launch_node != data.source_node
            and not incoming_launch
        ):
            solver.Add(y[item.tuple_id] == 0)
        else:
            if not item.is_air_residual and item.launch_node != data.source_node:
                solver.Add(
                    y[item.tuple_id]
                    <= sum(x[arc.arc_id] for arc in incoming_launch)
                )
            solver.Add(y[item.tuple_id] <= sum(x[arc.arc_id] for arc in incoming_recovery))
        for arc_id in item.required_arc_ids:
            solver.Add(y[item.tuple_id] <= x[arc_id])
    interval_keys = {key for item in data.tuples for key in item.interval_keys}
    for key in interval_keys:
        solver.Add(
            sum(y[item.tuple_id] for item in data.tuples if key in item.interval_keys) <= 1
        )
    for key, capacity in (data.operation_capacity or {}).items():
        solver.Add(
            sum(y[item.tuple_id] for item in data.tuples if key in item.operation_keys)
            <= capacity
        )
    for arc_id, value in (data.fixed_arc_values or {}).items():
        solver.Add(x[arc_id] == value)
    miss = {task_id: solver.BoolVar(f"miss[{task_id}]") for task_id in data.task_ids}
    for task_id in data.task_ids:
        solver.Add(
            sum(y[item.tuple_id] for item in data.tuples if task_id in item.task_ids)
            + miss[task_id]
            == 1
        )
    for left, right in data.conflicts:
        solver.Add(y[left] + y[right] <= 1)
    for tuple_id, value in (data.fixed_tuple_values or {}).items():
        solver.Add(y[tuple_id] == value)
    for arc in data.arcs:
        if arc.head == data.sink_node:
            solver.Add(c_max >= float(arc.arrival_layer) * x[arc.arc_id])
    for item in data.tuples:
        for _, layer in item.completion_layers:
            solver.Add(c_max >= float(layer) * y[item.tuple_id])
    primary = solver.Sum(data.miss_penalty[task_id] * miss[task_id] for task_id in data.task_ids)
    solver.Minimize(primary)
    status1 = solver.Solve()
    if status1 not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        return RollingMilpResult(
            "infeasible",
            (),
            (),
            float("inf"),
            float("inf"),
            _status_name(status1),
            "not_solved",
            float("inf"),
            float("inf"),
            perf_counter() - started,
            solver_nodes=int(solver.nodes()),
        )
    best_primary = primary.solution_value()
    primary_bound = solver.Objective().BestBound()
    stage1_tuples = tuple(sorted(key for key, variable in y.items() if variable.solution_value() > 0.5))
    stage1_arcs = tuple(sorted(key for key, variable in x.items() if variable.solution_value() > 0.5))
    solver.Add(primary <= best_primary + epsilon_u)
    tuple_cost = solver.Sum(item.secondary_cost * y[item.tuple_id] for item in data.tuples)
    arc_cost = solver.Sum(arc.secondary_cost * x[arc.arc_id] for arc in data.arcs)
    duty_time = solver.Sum(arc.duty_time * x[arc.arc_id] for arc in data.arcs)
    risk_exposure = solver.Sum(
        item.risk_exposure * y[item.tuple_id] for item in data.tuples
    )
    expected_energy = solver.Sum(
        item.expected_energy * y[item.tuple_id] for item in data.tuples
    )
    tuple_change = solver.Sum(
        (1 - y[key]) if int(value) == 1 else y[key]
        for key, value in (data.previous_tuple_values or {}).items()
        if key in y
    )
    arc_change = solver.Sum(
        (1 - x[key]) if int(value) == 1 else x[key]
        for key, value in (data.previous_arc_values or {}).items()
        if key in x
    )
    removed_tuple_change = sum(
        int(value) == 1
        for key, value in (data.previous_tuple_values or {}).items()
        if key not in y
    )
    removed_arc_change = sum(
        int(value) == 1
        for key, value in (data.previous_arc_values or {}).items()
        if key not in x
    )
    secondary = (
        tuple_cost
        + arc_cost
        + data.plan_change_weight
        * (
            tuple_change
            + arc_change
            + removed_tuple_change
            + removed_arc_change
        )
        + data.makespan_weight * c_max
        + data.risk_weight * risk_exposure
        + data.energy_weight * expected_energy
        + data.time_weight * duty_time
    )
    solver.Minimize(secondary)
    if node_limit is None:
        remaining_ms = max(
            1,
            int((float(time_limit_s) - (perf_counter() - started)) * 1000),
        )
        solver.SetTimeLimit(remaining_ms)
    elif not solver.SetSolverSpecificParametersAsString(parameters):
        raise RuntimeError("failed to reset deterministic SCIP node limit")
    status2 = solver.Solve()
    if status2 in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        selected = tuple(sorted(key for key, variable in y.items() if variable.solution_value() > 0.5))
        selected_arcs = tuple(sorted(key for key, variable in x.items() if variable.solution_value() > 0.5))
        missed_weight = sum(
            data.miss_penalty[task_id] * miss[task_id].solution_value()
            for task_id in data.task_ids
        )
        secondary_value = secondary.solution_value()
        secondary_bound = solver.Objective().BestBound()
        makespan = c_max.solution_value()
        risk_value = risk_exposure.solution_value()
        energy_value = expected_energy.solution_value()
        time_value = duty_time.solution_value()
        final_status = _status_name(status2)
    else:
        selected = stage1_tuples
        selected_arcs = stage1_arcs
        missed_weight = best_primary
        previous_tuples = data.previous_tuple_values or {}
        previous_arcs = data.previous_arc_values or {}
        secondary_value = sum(item.secondary_cost for item in data.tuples if item.tuple_id in selected)
        secondary_value += sum(arc.secondary_cost for arc in data.arcs if arc.arc_id in selected_arcs)
        secondary_value += data.plan_change_weight * sum(
            abs(int(key in selected) - int(value)) for key, value in previous_tuples.items()
        )
        secondary_value += data.plan_change_weight * sum(
            abs(int(key in selected_arcs) - int(value)) for key, value in previous_arcs.items()
        )
        makespan = max(
            [
                arc.arrival_layer
                for arc in data.arcs
                if arc.arc_id in selected_arcs and arc.head == data.sink_node
            ]
            + [
                layer
                for item in data.tuples
                if item.tuple_id in selected
                for _, layer in item.completion_layers
            ],
            default=0,
        )
        secondary_value += data.makespan_weight * makespan
        risk_value = sum(
            item.risk_exposure
            for item in data.tuples
            if item.tuple_id in selected
        )
        energy_value = sum(
            item.expected_energy
            for item in data.tuples
            if item.tuple_id in selected
        )
        secondary_value += data.risk_weight * risk_value
        secondary_value += data.energy_weight * energy_value
        time_value = sum(
            arc.duty_time
            for arc in data.arcs
            if arc.arc_id in selected_arcs
        )
        secondary_value += data.time_weight * time_value
        secondary_bound = float("inf")
        final_status = _status_name(status1)
    return RollingMilpResult(
        final_status,
        selected,
        selected_arcs,
        missed_weight,
        secondary_value,
        _status_name(status1),
        _status_name(status2) if status2 in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE} else "not_solved",
        primary_bound,
        secondary_bound,
        perf_counter() - started,
        float(makespan),
        float(risk_value),
        float(energy_value),
        float(time_value),
        status1 == pywraplp.Solver.OPTIMAL,
        int(solver.nodes()),
    )
