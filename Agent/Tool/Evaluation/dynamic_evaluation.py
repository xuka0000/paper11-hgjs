from dataclasses import asdict, dataclass, replace, fields, is_dataclass
from collections.abc import Mapping
import json
from hashlib import sha256
from math import ceil
from time import perf_counter
from types import SimpleNamespace
from Agent.Proposed.validator import validate_schedule, ValidationReport

from Agent.Tool.Evaluation.method_runner import method_family, method_spec, run_method
from Env.Execution import (
    PlannedLaunchEvent,
    PlannedRecoveryEvent,
    PlannedTaskEvent,
    PlannedVehicleArcEvent,
    advance_state,
    execute_due_events,
    record_plan,
)
from Env.Trigger import TriggerPolicy as EnvironmentTriggerPolicy
from Env.Runtime import EnergyObservation, apply_energy_observations
from Mask.ServiceTuple import SearchArc, SearchState, SearchTuple
from Mask.TupleFactory import (
    AirResidualTuple,
    EnergyMoments,
    build_air_residual,
    build_energy_certificate,
    build_service_resource_keys,
)


DISTURBANCE_KINDS = frozenset(
    {"weather", "road", "risk", "vehicle_delay", "uav_energy"}
)

LEGACY_PHYSICAL_DEFAULTS = {
    "energy_alpha": 0.9,
    "road_beta_w": 0.02,
    "road_beta_eta": 0.4,
    "preparation_layers": 1,
    "recovery_service_layers": 1,
    "turnaround_layers": 1,
}


@dataclass(frozen=True)
class DisturbanceEvent:
    kind: str
    layer: int
    magnitude: float
    target_id: int | None = None

    def __post_init__(self):
        if self.kind not in DISTURBANCE_KINDS:
            raise ValueError(f"unknown disturbance kind: {self.kind!r}")
        if self.layer < 1:
            raise ValueError("disturbance layer must be positive")
        if self.magnitude < 0:
            raise ValueError("disturbance magnitude must be nonnegative")


@dataclass(frozen=True)
class DynamicScenario:
    name: str
    events: tuple[DisturbanceEvent, ...]

    def __post_init__(self):
        object.__setattr__(
            self,
            "events",
            tuple(sorted(self.events, key=lambda item: (item.layer, item.kind, item.target_id or 0))),
        )


@dataclass(frozen=True)
class ReplanPolicy:
    name: str
    interval: int | None = None
    use_events: bool = False
    weather_threshold: float = 0.3
    risk_threshold: float = 0.3
    delay_threshold: float = 1.0
    energy_threshold: float = 0.2

    def __post_init__(self):
        if self.name not in {"one_shot", "fixed_period", "event_only", "combined"}:
            raise ValueError(f"unknown replan policy: {self.name!r}")
        if self.interval is not None and self.interval <= 0:
            raise ValueError("replan interval must be positive")

    @classmethod
    def one_shot(cls):
        return cls("one_shot")

    @classmethod
    def fixed_period(cls, interval=2):
        return cls("fixed_period", interval=interval)

    @classmethod
    def event_only(cls, **thresholds):
        return cls("event_only", use_events=True, **thresholds)

    @classmethod
    def combined(cls, interval=2, **thresholds):
        return cls("combined", interval=interval, use_events=True, **thresholds)


@dataclass(frozen=True)
class ReplanDecision:
    triggered: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DynamicProblem:
    task_ids: tuple[int, ...]
    miss_penalty: dict[int, float]
    operation_capacity: dict[str, int]
    search_state: SearchState
    rolling_state: object
    safety_status: str = "feasible"
    safety_reason: str = ""


@dataclass(frozen=True)
class DynamicTransition:
    layer: int
    event_kinds: tuple[str, ...]
    trigger_reasons: tuple[str, ...]
    replanned: bool
    state_before_hash: str
    state_after_hash: str
    completed_tasks: tuple[int, ...]
    done_tasks_after: tuple[int, ...]
    remaining_tasks_after: tuple[int, ...]
    plan_change: int
    primary_value: float
    secondary_value: float
    milp_calls: int
    runtime_s: float
    validation_ok: bool
    safety_status: str = "feasible"
    active_vehicle_arcs: tuple[tuple[int, str, int], ...] = ()
    airborne_uav_ids: tuple[int, ...] = ()
    remaining_energy: tuple[tuple[int, float | None], ...] = ()
    energy_consumed: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class DynamicTrajectoryResult:
    instance_id: str
    scenario: str
    policy: str
    transitions: tuple[DynamicTransition, ...]
    trigger_count: int
    replan_count: int
    final_weighted_misses: float
    final_secondary: float
    completion_rate: float
    total_runtime_s: float
    validation_ok: bool
    safety_status: str
    final_state: object


def should_replan(policy, last_replan_layer, current_layer, events):
    if policy.name == "one_shot":
        return ReplanDecision(False, ())
    trigger = EnvironmentTriggerPolicy(
        max_interval=policy.interval,
        weather_threshold=policy.weather_threshold,
        risk_threshold=policy.risk_threshold,
        delay_threshold=policy.delay_threshold,
        energy_threshold=policy.energy_threshold,
    )
    event_rows = (
        [
            (event.kind, event.layer, event.magnitude)
            for event in events
        ]
        if policy.use_events
        else []
    )
    decision = trigger.evaluate(
        last_replan_layer,
        current_layer,
        event_rows,
    )
    return ReplanDecision(decision.triggered, decision.reasons)


def materialize_dynamic_problem(instance, rolling_state, active_disturbances):
    remaining = tuple(sorted(rolling_state.remaining_tasks))
    current = rolling_state.current_layer
    energy_alpha = float(
        getattr(instance, "energy_alpha", LEGACY_PHYSICAL_DEFAULTS["energy_alpha"])
    )
    road_beta_w = float(
        getattr(instance, "road_beta_w", LEGACY_PHYSICAL_DEFAULTS["road_beta_w"])
    )
    road_beta_eta = float(
        getattr(instance, "road_beta_eta", LEGACY_PHYSICAL_DEFAULTS["road_beta_eta"])
    )
    preparation_layers = int(
        getattr(
            instance,
            "preparation_layers",
            LEGACY_PHYSICAL_DEFAULTS["preparation_layers"],
        )
    )
    recovery_service_layers = int(
        getattr(
            instance,
            "recovery_service_layers",
            LEGACY_PHYSICAL_DEFAULTS["recovery_service_layers"],
        )
    )
    turnaround_layers = int(
        getattr(
            instance,
            "turnaround_layers",
            LEGACY_PHYSICAL_DEFAULTS["turnaround_layers"],
        )
    )
    closed_vehicles = {
        event.target_id
        for event in active_disturbances
        if event.kind == "road" and event.target_id is not None
    }
    energy_reduction_by_uav = {
        event.target_id: min(float(event.magnitude), 1.0)
        for event in active_disturbances
        if event.kind == "uav_energy" and event.target_id is not None
    }
    weather_magnitude = sum(
        event.magnitude for event in active_disturbances if event.kind == "weather"
    )
    delay_by_vehicle = {
        event.target_id: event.magnitude
        for event in active_disturbances
        if event.kind == "vehicle_delay" and event.target_id is not None
    }

    vehicle_ids = sorted({arc.vehicle_id for arc in instance.search_state.arcs})
    arcs = []
    vehicle_arcs = {}
    arc_id_maps = {}
    node_layers = {}
    operation_capacity = {}
    default_operation_capacity = min(
        instance.operation_capacity.values(),
        default=1,
    )
    committed_arc_ids = set()
    state_failures = []
    for vehicle_id in vehicle_ids:
        if vehicle_id in closed_vehicles:
            continue
        delay = float(delay_by_vehicle.get(vehicle_id, 0.0))
        road_factor = 1.0 + (
            road_beta_w * weather_magnitude
            + road_beta_eta * weather_magnitude
        )
        base_route = sorted(
            (
                arc
                for arc in instance.search_state.arcs
                if arc.vehicle_id == vehicle_id
            ),
            key=lambda arc: (
                arc.departure_layer,
                arc.arrival_layer,
                arc.arc_id,
            ),
        )
        vehicle = rolling_state.vehicles[vehicle_id]
        committed = vehicle.active_arc is not None
        anchor_node = vehicle.active_arc[1] if committed else vehicle.node_id
        committed_base = None
        if committed:
            candidates = [arc for arc in base_route if (
                arc.arc_id == vehicle.active_arc_id if vehicle.active_arc_id else
                (arc.tail, arc.head) == vehicle.active_arc
            )]
            if len(candidates) != 1 or vehicle.active_arc_arrival_layer is None or vehicle.active_arc_arrival_layer < current:
                state_failures.append(f"vehicle={vehicle_id}:invalid in-transit commitment")
                continue
            committed_base = candidates[0]
            base_route = base_route[base_route.index(committed_base) + 1:]
        anchor_indices = [
            index
            for index, arc in enumerate(base_route)
            if arc.tail == anchor_node
        ]
        if anchor_indices and not committed:
            base_route = base_route[anchor_indices[0] :]
        elif anchor_node == "sink":
            base_route = []
        if not base_route and not committed:
            continue
        route = []
        departure = max(current, vehicle.available_layer)
        arc_id_map = {}
        if committed:
            departure = vehicle.active_arc_arrival_layer
            arc_id = committed_base.arc_id
            route.append(SearchArc(
                arc_id, vehicle_id, "source", anchor_node, current, departure,
                "committed", 0.0, duty_time=vehicle.active_arc_duty_time,
            ))
            arc_id_map[arc_id] = arc_id
            committed_arc_ids.add(arc_id)
        node_layer = {anchor_node if committed else base_route[0].tail: departure}
        if not committed and base_route[0].tail != "source":
            route.append(SearchArc(
                f"anchor:{vehicle_id}:{current}:{base_route[0].tail}",
                vehicle_id, "source", base_route[0].tail,
                current, departure, "anchor", 0.0, duty_time=0.0,
            ))
        delay_per_arc = delay / max(1, len(base_route))
        for index, base_arc in enumerate(base_route):
            base_layers = max(
                1,
                base_arc.arrival_layer - base_arc.departure_layer,
            )
            duration_layers = max(
                1,
                int(ceil(base_layers * road_factor + delay_per_arc)),
            )
            arrival = departure + duration_layers
            arc_id = base_arc.arc_id
            duty_time = (
                base_arc.duty_time
                if base_arc.duty_time > 0
                else float(base_layers)
            )
            duty_time = duty_time * road_factor + delay_per_arc
            rebuilt = SearchArc(
                arc_id,
                vehicle_id,
                base_arc.tail,
                base_arc.head,
                departure,
                arrival,
                base_arc.arc_type,
                float(base_arc.secondary_cost) * road_factor + delay_per_arc,
                duty_time=float(duty_time),
            )
            route.append(rebuilt)
            arc_id_map[base_arc.arc_id] = arc_id
            node_layer[base_arc.head] = arrival
            departure = arrival
        route = tuple(route)
        arcs.extend(route)
        vehicle_arcs[vehicle_id] = route
        arc_id_maps[vehicle_id] = arc_id_map
        node_layers[vehicle_id] = node_layer

    base_tuples = {
        (item.vehicle_id, item.uav_id, item.task_ids): item
        for item in instance.search_state.tuples
    }
    airborne_uav_ids = set(rolling_state.airborne_tasks) | {
        uav_id for uav_id, uav in rolling_state.uavs.items() if uav.is_airborne
    }
    tuples = []
    for (vehicle_id, uav_id, task_ids), base_tuple in base_tuples.items():
        if vehicle_id not in vehicle_arcs:
            continue
        if uav_id in airborne_uav_ids:
            continue
        if not set(task_ids) <= set(remaining):
            continue
        if (
            uav_id in energy_reduction_by_uav
            and base_tuple.battery_capacity <= 0
        ):
            continue
        launch = base_tuple.launch_node
        recovery = base_tuple.recovery_node
        node_layer = node_layers[vehicle_id]
        if launch not in node_layer or recovery not in node_layer:
            continue
        launch_layer = node_layer[launch]
        recovery_layer = node_layer[recovery]
        if launch_layer < rolling_state.uavs[uav_id].available_layer:
            continue
        if recovery_layer < launch_layer + (base_tuple.recovery_layer - base_tuple.launch_layer):
            continue
        completion_layers = tuple(
            (
                task_id,
                launch_layer + dict(base_tuple.completion_layers)[task_id] - base_tuple.launch_layer,
            )
            for index, task_id in enumerate(task_ids)
        )
        risk_multiplier = 1.0 + 0.15 * weather_magnitude
        for event in active_disturbances:
            if event.kind == "risk" and event.target_id in task_ids:
                risk_multiplier += 0.25 * event.magnitude
        energy_multiplier = 1.0 + 0.15 * weather_magnitude
        if base_tuple.battery_capacity > 0:
            capacity = base_tuple.battery_capacity * (
                1.0 - energy_reduction_by_uav.get(uav_id, 0.0)
            )
            certificate = build_energy_certificate(
                EnergyMoments(
                    base_tuple.expected_energy * energy_multiplier,
                    base_tuple.energy_std * energy_multiplier,
                ),
                alpha=energy_alpha,
                full_capacity=capacity,
                reserve_ratio=base_tuple.reserve_ratio,
            )
            if not certificate.admitted:
                continue
        else:
            certificate = None
        resources = build_service_resource_keys(
            vehicle_id,
            uav_id,
            launch,
            recovery,
            launch_layer,
            recovery_layer,
            preparation_layers,
            recovery_service_layers,
            turnaround_layers,
        )
        for key in resources.operation_keys:
            operation_capacity[key] = default_operation_capacity
        tuples.append(
            replace(
                base_tuple,
                launch_node=launch,
                recovery_node=recovery,
                launch_layer=launch_layer,
                recovery_layer=recovery_layer,
                required_arc_ids=tuple(
                    arc_id_maps[vehicle_id][arc_id]
                    for arc_id in base_tuple.required_arc_ids
                    if arc_id in arc_id_maps[vehicle_id]
                ),
                interval_keys=resources.interval_keys,
                operation_keys=resources.operation_keys,
                completion_layers=completion_layers,
                secondary_cost=float(base_tuple.secondary_cost) * road_factor,
                admitted=(
                    base_tuple.admitted
                    and base_tuple.admission_checks.all_passed
                    and (certificate.admitted if certificate else True)
                ),
                risk_exposure=float(base_tuple.risk_exposure) * risk_multiplier,
                expected_energy=(
                    certificate.mean if certificate else base_tuple.expected_energy
                ),
                energy_std=(certificate.std if certificate else base_tuple.energy_std),
                reliable_energy=(
                    certificate.reliable_bound
                    if certificate
                    else base_tuple.reliable_energy
                ),
                battery_capacity=(
                    certificate.full_capacity
                    if certificate
                    else base_tuple.battery_capacity
                ),
                reserve_ratio=base_tuple.reserve_ratio,
                admission_certificate=(
                    certificate.certificate_id
                    if certificate
                    else base_tuple.admission_certificate
                ),
            )
        )
    residual_ids = set()
    safety_failures = []
    for uav_id in sorted(airborne_uav_ids):
        uav = rolling_state.uavs[uav_id]
        flight = rolling_state.airborne_services.get(uav_id)
        residual_task_ids = tuple(
            task for task in rolling_state.airborne_tasks.get(uav_id, uav.remaining_tasks)
            if task in remaining
        )
        candidates = [
            item
            for item in base_tuples.values()
            if item.uav_id == uav_id
            and set(residual_task_ids) <= set(item.task_ids)
            and item.vehicle_id in vehicle_arcs
            and (
                not rolling_state.active_service_ids.get(uav_id)
                or item.tuple_id == rolling_state.active_service_ids[uav_id]
            )
        ]
        if not candidates:
            safety_failures.append(f"uav={uav_id}:airborne recovery tuple absent")
            continue
        base_tuple = min(
            candidates,
            key=lambda item: (
                item.tuple_id != rolling_state.active_service_ids.get(uav_id),
                item.tuple_id not in rolling_state.locked_tuple_ids,
                not rolling_state.previous_tuple_values.get(item.tuple_id, 0),
                len(item.task_ids),
                item.recovery_layer,
                item.tuple_id,
            ),
        )
        vehicle_id = base_tuple.vehicle_id
        recovery = flight.recovery_node if flight else base_tuple.recovery_node
        recovery_layer = node_layers[vehicle_id].get(recovery)
        if recovery_layer is None:
            safety_failures.append(f"uav={uav_id}:airborne recovery node unreachable")
            continue
        moments = EnergyMoments(*flight.remaining_moments(current, recovery_layer)) if flight else EnergyMoments(
            max(1e-6, base_tuple.expected_energy), max(0.0, base_tuple.energy_std),
        )
        try:
            if base_tuple.battery_capacity <= 0:
                # Legacy unit benchmarks omit an energy model entirely.
                residual = AirResidualTuple(
                    f"air:{uav_id}:{recovery}:{recovery_layer}", uav_id,
                    residual_task_ids, None, recovery, recovery_layer,
                    base_tuple.reliable_energy,
                )
            else:
                if uav.remaining_energy is None:
                    raise ValueError("remaining energy unavailable")
                residual = build_air_residual(
                    uav,
                    recovery_stop=recovery,
                    recovery_layer=recovery_layer,
                    moments=moments,
                    full_capacity=flight.battery_capacity if flight else base_tuple.battery_capacity,
                    reserve_ratio=flight.reserve_ratio if flight else base_tuple.reserve_ratio,
                    alpha=energy_alpha,
                )
        except ValueError as error:
            safety_failures.append(f"uav={uav_id}:{error}")
            continue
        resources = build_service_resource_keys(
            vehicle_id,
            uav_id,
            "",
            recovery,
            current,
            recovery_layer,
            0,
            recovery_service_layers,
            turnaround_layers,
        )
        for key in resources.operation_keys:
            operation_capacity[key] = default_operation_capacity
        required_arc_ids = tuple(
            arc_id_maps[vehicle_id][arc_id]
            for arc_id in base_tuple.required_arc_ids
            if arc_id in arc_id_maps[vehicle_id]
        )
        completion_layers = tuple(
            (
                task_id,
                max(current, dict(flight.completion_layers)[task_id]) if flight else min(recovery_layer, current + index + 1),
            )
            for index, task_id in enumerate(residual_task_ids)
        )
        residual_tuple = SearchTuple(
            tuple_id=residual.residual_id,
            vehicle_id=vehicle_id,
            uav_id=uav_id,
            task_ids=tuple(residual_task_ids),
            launch_node="",
            recovery_node=recovery,
            launch_layer=current,
            recovery_layer=recovery_layer,
            required_arc_ids=required_arc_ids,
            interval_keys=resources.interval_keys,
            operation_keys=resources.operation_keys,
            completion_layers=completion_layers,
            secondary_cost=0.0,
            admitted=True,
            risk_exposure=base_tuple.risk_exposure,
            expected_energy=moments.mean,
            energy_std=moments.std,
            reliable_energy=residual.q_bound,
            battery_capacity=base_tuple.battery_capacity,
            reserve_ratio=base_tuple.reserve_ratio,
            admission_certificate=residual.residual_id,
            is_air_residual=True,
            admission_checks=base_tuple.admission_checks,
        )
        tuples.append(residual_tuple)
        residual_ids.add(residual_tuple.tuple_id)
    tuples = tuple(sorted(tuples, key=lambda item: item.tuple_id))
    if tuples:
        tuple_ids = {item.tuple_id for item in tuples}
        retained_tuple_ids = {
            tuple_id
            for tuple_id, value in rolling_state.previous_tuple_values.items()
            if int(value) == 1 and tuple_id in tuple_ids
        }
        retained_tuple_ids.update(
            rolling_state.locked_tuple_ids & tuple_ids
        )
        retained_tuple_ids.update(residual_ids)
        if retained_tuple_ids:
            selected_tuple_ids = frozenset(retained_tuple_ids)
        else:
            selected_tuple_ids = frozenset({tuples[0].tuple_id})
        selected_vehicle_ids = {
            item.vehicle_id
            for item in tuples
            if item.tuple_id in selected_tuple_ids
        }
        available_arc_ids = {arc.arc_id for arc in arcs}
        retained_arc_ids = {
            arc_id
            for arc_id, value in rolling_state.previous_arc_values.items()
            if int(value) == 1 and arc_id in available_arc_ids
        }
        retained_arc_ids.update(
            rolling_state.locked_arc_ids & available_arc_ids
        )
        selected_arc_ids = frozenset(
            retained_arc_ids
            | {
                arc.arc_id
                for vehicle_id in selected_vehicle_ids
                for arc in vehicle_arcs[vehicle_id]
            }
        )
    else:
        selected_tuple_ids = frozenset()
        selected_arc_ids = frozenset()
    # A vehicle still needs its residual path when every inspection is complete.
    required_vehicle_ids = {
        vehicle_id for vehicle_id, vehicle in rolling_state.vehicles.items()
        if vehicle.must_return or vehicle.active_arc is not None
    }
    return_failures = state_failures + [
        f"vehicle={vehicle_id}:return route unavailable"
        for vehicle_id in sorted(required_vehicle_ids)
        if vehicle_id not in vehicle_arcs
        and rolling_state.vehicles[vehicle_id].node_id != "sink"
    ]
    selected_arc_ids = selected_arc_ids | frozenset(
        arc.arc_id for arc in arcs
        if arc.vehicle_id in required_vehicle_ids
        or arc.arc_id in rolling_state.locked_arc_ids
    )
    search_state = SearchState(
        tuple(sorted(arcs)),
        tuples,
        selected_arc_ids,
        selected_tuple_ids,
    )
    miss_penalty = {
        task_id: float(instance.miss_penalty[task_id]) for task_id in remaining
    }
    for event in active_disturbances:
        if (
            event.kind == "risk"
            and event.target_id in miss_penalty
        ):
            miss_penalty[event.target_id] *= 1.0 + event.magnitude
    problem_rolling_state = replace(
        rolling_state,
        locked_tuple_ids=rolling_state.locked_tuple_ids | frozenset(residual_ids),
        locked_arc_ids=rolling_state.locked_arc_ids | frozenset(committed_arc_ids),
    )
    return DynamicProblem(
        remaining,
        miss_penalty,
        operation_capacity,
        search_state,
        problem_rolling_state,
        (
            "unsafe_airborne_recovery"
            if safety_failures
            else "infeasible_vehicle_return"
            if return_failures
            else "feasible"
        ),
        ";".join(safety_failures + return_failures),
    )


def _state_hash(state):
    def canonical(value):
        if is_dataclass(value):
            return {field.name: canonical(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, Mapping):
            pairs = [(canonical(key), canonical(item)) for key, item in value.items()]
            return {"mapping": sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))}
        if isinstance(value, (set, frozenset)):
            return {"set": sorted((canonical(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))}
        if isinstance(value, (tuple, list)):
            return [canonical(item) for item in value]
        return value
    encoded = json.dumps(canonical(state), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _plan_entries(problem, result):
    tuple_by_id = {item.tuple_id: item for item in problem.search_state.tuples}
    arc_by_id = {arc.arc_id: arc for arc in problem.search_state.arcs}
    entries = []
    entries.extend(
        PlannedVehicleArcEvent(
            arc.arc_id,
            arc.vehicle_id,
            arc.head,
            arc.arrival_layer,
            arc.duty_time,
            arc.departure_layer,
            arc.tail,
        )
        for arc_id in getattr(result, "selected_arc_ids", ())
        for arc in (arc_by_id[arc_id],)
    )
    for tuple_id in result.selected_tuple_ids:
        item = tuple_by_id[tuple_id]
        if not item.is_air_residual:
            entries.append(PlannedLaunchEvent(
                item.vehicle_id, item.uav_id, item.tuple_id,
                item.launch_layer, item.task_ids,
                item.recovery_layer, item.recovery_node, item.completion_layers,
                item.expected_energy, item.energy_std, item.battery_capacity, item.reserve_ratio,
            ))
        completion_by_task = dict(item.completion_layers)
        entries.extend(
            PlannedTaskEvent(
                task_id,
                item.vehicle_id,
                item.uav_id,
                completion_by_task[task_id],
            )
            for task_id in item.task_ids
        )
        entries.append(
            PlannedRecoveryEvent(
                item.vehicle_id,
                item.uav_id,
                item.recovery_node,
                item.recovery_layer,
                max(
                    (int(key.rsplit(":", 1)[1]) + 1 for key in item.interval_keys),
                    default=item.recovery_layer,
                ),
            )
        )
    return tuple(sorted(set(entries), key=repr))


def validate_dynamic_plan(problem, result):
    """Check the returned plan itself, including an unchanged search incumbent."""
    state = problem.rolling_state
    tuple_ids = set(result.selected_tuple_ids)
    arc_ids = set(result.selected_arc_ids)
    tuple_by_id = {item.tuple_id: item for item in problem.search_state.tuples}
    arc_by_id = {arc.arc_id: arc for arc in problem.search_state.arcs}
    if tuple_ids - tuple_by_id.keys() or arc_ids - arc_by_id.keys():
        return ValidationReport(False, "unknown_plan_object", 1.0, "domain")
    selected = tuple(tuple_by_id[key] for key in sorted(tuple_ids))
    report = validate_schedule(SimpleNamespace(
        selected_tuples=selected, selected_arc_ids=tuple(sorted(arc_ids)),
        missed_task_ids=tuple(sorted(set(problem.task_ids) - {task for item in selected for task in item.task_ids})),
    ), state, {
        "admitted_tuple_ids": {item.tuple_id for item in tuple_by_id.values() if item.admitted},
        "arc_defs": {arc.arc_id: (arc.vehicle_id, arc.tail, arc.head) for arc in arc_by_id.values()},
        "arc_time_defs": {arc.arc_id: (arc.vehicle_id, arc.tail, arc.departure_layer, arc.head, arc.arrival_layer) for arc in arc_by_id.values()},
        "operation_capacity": problem.operation_capacity,
    })
    if not report.ok:
        return report
    if (state.locked_arc_ids & arc_by_id.keys()) - arc_ids:
        return ValidationReport(False, "committed_arc_removed", 1.0, "vehicle")
    if (state.locked_tuple_ids & tuple_by_id.keys()) - tuple_ids:
        return ValidationReport(False, "committed_service_removed", 1.0, "UAV")
    for vehicle_id, vehicle in state.vehicles.items():
        arcs = [arc_by_id[key] for key in arc_ids if arc_by_id[key].vehicle_id == vehicle_id]
        if (vehicle.must_return or vehicle.active_arc is not None) and not any(arc.head == 'sink' for arc in arcs):
            return ValidationReport(False, "return_obligation_removed", 1.0, str(vehicle_id))
        if vehicle.remaining_duty_time is not None:
            excess = sum(arc.duty_time for arc in arcs) - vehicle.remaining_duty_time
            if excess > 1e-8:
                return ValidationReport(False, "remaining_duty_exceeded", excess, str(vehicle_id))
    for uav_id, uav in state.uavs.items():
        if uav.is_airborne and sum(item.is_air_residual and item.uav_id == uav_id for item in selected) != 1:
            return ValidationReport(False, "airborne_recovery_removed", 1.0, str(uav_id))
    return report


def run_dynamic_trajectory(
    instance,
    scenario,
    policy,
    scorer,
    search_config,
    decision_layers=(1, 3, 5, 7),
    energy_observations=(),
    terminal_layer=None,
):
    if not decision_layers:
        raise ValueError("decision_layers must not be empty")
    if tuple(sorted(set(decision_layers))) != tuple(decision_layers):
        raise ValueError("decision_layers must be sorted and unique")
    if any(event.layer not in decision_layers for event in scenario.events):
        raise ValueError("disturbance layers must be present in the decision grid")
    if any(observation.layer not in decision_layers for observation in energy_observations):
        raise ValueError("energy observation layers must be present in the decision grid")
    if terminal_layer is not None and terminal_layer <= decision_layers[-1]:
        raise ValueError("terminal execution layer must follow the last decision layer")
    rolling_state = replace(instance.rolling_state, current_layer=decision_layers[0])
    last_replan_layer = decision_layers[0]
    plan_entries = ()
    previous_plan = frozenset()
    transitions = []
    trigger_count = 0
    replan_count = 0
    total_runtime = 0.0
    final_secondary = 0.0
    validation_ok = True
    safety_status = "feasible"

    for index, layer in enumerate(decision_layers):
        if rolling_state.current_layer != layer:
            rolling_state = replace(rolling_state, current_layer=layer)
        state_before_hash = _state_hash(rolling_state)
        current_events = tuple(event for event in scenario.events if event.layer == layer)
        active_events = tuple(event for event in scenario.events if event.layer <= layer)
        # The shock is applied once when revealed, never once per reconstruction.
        shock_observations = []
        for event in current_events:
            if event.kind == "uav_energy" and event.target_id is not None:
                uav = rolling_state.uavs[event.target_id]
                if uav.remaining_energy is not None:
                    shock_observations.append(EnergyObservation(
                        event.target_id, layer,
                        uav.remaining_energy * (1.0 - min(event.magnitude, 1.0)),
                        "synthetic_E5_energy_loss",
                    ))
        rolling_state = apply_energy_observations(rolling_state, shock_observations)
        rolling_state = apply_energy_observations(rolling_state, (
            observation for observation in energy_observations if observation.layer == layer
        ))
        decision = should_replan(
            policy,
            last_replan_layer,
            layer,
            current_events,
        )
        obligations_remain = bool(
            rolling_state.remaining_tasks
            or rolling_state.airborne_tasks
            or plan_entries
            or any(uav.is_airborne for uav in rolling_state.uavs.values())
            or any(
                vehicle.must_return or vehicle.active_arc is not None
                for vehicle in rolling_state.vehicles.values()
            )
        )
        replanned = (index == 0 or decision.triggered) and obligations_remain
        if index > 0 and decision.triggered:
            trigger_count += 1
        plan_change = 0
        primary_value = sum(
            instance.miss_penalty[task_id]
            for task_id in rolling_state.remaining_tasks
        )
        milp_calls = 0
        runtime_s = 0.0
        step_validation = True
        step_safety_status = "feasible"
        if replanned:
            committed_entries = tuple(
                entry for entry in plan_entries
                if isinstance(entry, PlannedVehicleArcEvent)
                and entry.arc_id == rolling_state.vehicles[entry.vehicle_id].active_arc_id
            )
            problem = materialize_dynamic_problem(
                instance,
                rolling_state,
                active_events,
            )
            replan_count += 1
            last_replan_layer = layer
            if problem.safety_status != "feasible":
                step_validation = False
                validation_ok = False
                step_safety_status = problem.safety_status
                safety_status = problem.safety_status
                plan_entries = committed_entries
                previous_plan = frozenset()
            else:
                started = perf_counter()
                execution = run_method(
                    scorer,
                    state=problem.search_state,
                    task_ids=problem.task_ids,
                    miss_penalty=problem.miss_penalty,
                    validation_state=problem.rolling_state,
                    operation_capacity=problem.operation_capacity,
                    config=search_config,
                )
                search_result = execution.result
                runtime_s = perf_counter() - started
                total_runtime += runtime_s
                returned_plan_report = validate_dynamic_plan(problem, search_result)
                new_plan = _plan_entries(problem, search_result) if returned_plan_report.ok else committed_entries
                new_plan_set = frozenset(new_plan)
                plan_change = len(previous_plan ^ new_plan_set)
                previous_plan = new_plan_set
                plan_entries = new_plan
                rolling_state = record_plan(
                    rolling_state,
                    problem.search_state.arcs,
                    problem.search_state.tuples,
                    search_result.selected_arc_ids,
                    search_result.selected_tuple_ids,
                )
                primary_value = search_result.best_primary
                final_secondary = search_result.best_secondary
                milp_calls = (
                    search_result.solver_calls
                    if hasattr(search_result, "solver_calls")
                    else sum(
                        record.closure_accepted for record in search_result.trace
                    )
                )
                step_validation = (
                    search_result.validation_ok
                    if hasattr(search_result, "validation_ok")
                    else all(
                        record.validation_ok
                        for record in search_result.accepted_records
                    )
                )
                step_validation = step_validation and returned_plan_report.ok
                if not step_validation:
                    step_safety_status = safety_status = "invalid_replanned_schedule"
                validation_ok = validation_ok and step_validation

        blocked_vehicles = {
            event.target_id
            for event in active_events
            if event.kind == "road" and event.target_id is not None
        }
        # Reduced energy changes admission; it does not freeze a flying vehicle.
        blocked_uavs = set()
        end_layer = (
            decision_layers[index + 1]
            if index + 1 < len(decision_layers)
            else terminal_layer if terminal_layer is not None else layer + 2
        )
        execution_result, plan_entries = execute_due_events(
            rolling_state,
            plan_entries,
            end_layer,
            blocked_vehicle_ids=frozenset(blocked_vehicles),
            blocked_uav_ids=frozenset(blocked_uavs),
        )
        completed = tuple(sorted(execution_result.completed_tasks))
        rolling_state = advance_state(rolling_state, execution_result)
        if execution_result.safety_status != "feasible":
            step_validation = validation_ok = False
            step_safety_status = safety_status = execution_result.safety_status
        transitions.append(
            DynamicTransition(
                layer,
                tuple(event.kind for event in current_events),
                decision.reasons,
                replanned,
                state_before_hash,
                _state_hash(rolling_state),
                tuple(sorted(completed)),
                tuple(sorted(rolling_state.done_tasks)),
                tuple(sorted(rolling_state.remaining_tasks)),
                plan_change,
                float(primary_value),
                float(final_secondary),
                milp_calls,
                runtime_s,
                step_validation,
                step_safety_status,
                tuple(sorted(
                    (vehicle_id, vehicle.active_arc_id, vehicle.active_arc_arrival_layer)
                    for vehicle_id, vehicle in rolling_state.vehicles.items()
                    if vehicle.active_arc_id is not None
                )),
                tuple(sorted(uav_id for uav_id, uav in rolling_state.uavs.items() if uav.is_airborne)),
                tuple(sorted((uav_id, uav.remaining_energy) for uav_id, uav in rolling_state.uavs.items())),
                tuple(sorted((execution_result.energy_consumed or {}).items())),
            )
        )

    final_active_events = tuple(
        event for event in scenario.events if event.layer <= decision_layers[-1]
    )
    final_problem = materialize_dynamic_problem(
        instance,
        rolling_state,
        final_active_events,
    )
    if final_problem.safety_status != "feasible":
        validation_ok = False
        safety_status = final_problem.safety_status
    weighted_misses = sum(
        final_problem.miss_penalty[task_id]
        for task_id in rolling_state.remaining_tasks
    )
    completion_rate = len(rolling_state.done_tasks) / len(instance.task_ids)
    return DynamicTrajectoryResult(
        instance.instance_id,
        scenario.name,
        policy.name,
        tuple(transitions),
        trigger_count,
        replan_count,
        float(weighted_misses),
        float(final_secondary),
        float(completion_rate),
        total_runtime,
        validation_ok,
        safety_status,
        rolling_state,
    )


def default_e5_scenarios():
    return (
        DynamicScenario(
            "weather",
            (DisturbanceEvent("weather", 3, 0.8),),
        ),
        DynamicScenario(
            "road",
            (DisturbanceEvent("road", 3, 1.0, target_id=1),),
        ),
        DynamicScenario(
            "risk",
            (DisturbanceEvent("risk", 3, 0.8, target_id=4),),
        ),
        DynamicScenario(
            "vehicle_delay",
            (DisturbanceEvent("vehicle_delay", 3, 2.0, target_id=1),),
        ),
        DynamicScenario(
            "uav_energy",
            (DisturbanceEvent("uav_energy", 3, 0.6, target_id=1),),
        ),
    )


def default_replan_policies():
    return (
        ReplanPolicy.one_shot(),
        ReplanPolicy.fixed_period(interval=4),
        ReplanPolicy.event_only(),
        ReplanPolicy.combined(interval=4),
    )


def evaluate_dynamic_grid(
    instances,
    scenarios,
    policies,
    methods,
    search_config,
    checkpoint_seed,
    decision_layers=(1, 3, 5, 7),
    terminal_layer=None,
):
    summaries = []
    transition_rows = []
    for instance in sorted(instances, key=lambda item: item.instance_id):
        for scenario in sorted(scenarios, key=lambda item: item.name):
            for policy in sorted(policies, key=lambda item: item.name):
                for method_name in sorted(methods):
                    specification = method_spec(methods[method_name])
                    applied_config = specification.apply(search_config)
                    if (
                        specification.evaluation_seed is not None
                        and str(specification.evaluation_seed) != str(checkpoint_seed)
                    ):
                        raise ValueError("method evaluation seed mismatch")
                    result = run_dynamic_trajectory(
                        instance,
                        scenario,
                        policy,
                        methods[method_name],
                        search_config,
                        decision_layers,
                        terminal_layer=terminal_layer,
                    )
                    summaries.append(
                        {
                            "checkpoint_seed": str(checkpoint_seed),
                            "instance_id": instance.instance_id,
                            "instance_hash": instance.instance_hash,
                            "scenario": scenario.name,
                            "policy": policy.name,
                            "method": method_name,
                            "method_family": method_family(methods[method_name]),
                            "budget_id": specification.budget_id,
                            "checkpoint_sha256": specification.checkpoint_sha256,
                            "model_kind": specification.model_kind,
                            "shared_reference": specification.shared_reference,
                            "difficulty": getattr(
                                instance,
                                "difficulty",
                                "not_declared",
                            ),
                            "topology_family": getattr(
                                instance,
                                "topology_family",
                                "not_declared",
                            ),
                            "topology_hash": getattr(
                                instance,
                                "topology_hash",
                                "not_declared",
                            ),
                            "acceptance_mode": applied_config.acceptance_mode,
                            "deterministic_guard_count": (
                                applied_config.deterministic_guard_count
                            ),
                            "repair_time_limit_s": applied_config.repair_time_limit_s,
                            "time_weight": applied_config.time_weight,
                            "risk_weight": applied_config.risk_weight,
                            "energy_weight": applied_config.energy_weight,
                            "makespan_weight": applied_config.makespan_weight,
                            "plan_change_weight": applied_config.plan_change_weight,
                            "trigger_count": result.trigger_count,
                            "replan_count": result.replan_count,
                            "plan_changes": sum(
                                step.plan_change for step in result.transitions
                            ),
                            "final_weighted_misses": result.final_weighted_misses,
                            "final_secondary": result.final_secondary,
                            "completion_rate": result.completion_rate,
                            "total_runtime_s": result.total_runtime_s,
                            "milp_calls": sum(
                                step.milp_calls for step in result.transitions
                            ),
                            "validation_ok": result.validation_ok,
                            "safety_status": result.safety_status,
                        }
                    )
                    for step_index, step in enumerate(result.transitions):
                        transition_rows.append(
                            {
                                "checkpoint_seed": str(checkpoint_seed),
                                "instance_id": instance.instance_id,
                                "instance_hash": instance.instance_hash,
                                "scenario": scenario.name,
                                "policy": policy.name,
                                "method": method_name,
                                "budget_id": specification.budget_id,
                                "checkpoint_sha256": (
                                    specification.checkpoint_sha256
                                ),
                                "step_index": step_index,
                                **asdict(step),
                            }
                        )
    return tuple(summaries), tuple(transition_rows)
