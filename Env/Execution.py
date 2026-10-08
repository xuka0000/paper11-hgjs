from dataclasses import dataclass, replace
from Env.Runtime import ActiveFlight


@dataclass(frozen=True)
class ExecutionResult:
    end_layer: int
    completed_tasks: frozenset[int] = frozenset()
    vehicles: dict[int, object] | None = None
    uavs: dict[int, object] | None = None
    active_service_ids: dict[int, str] | None = None
    airborne_services: dict[int, object] | None = None
    energy_consumed: dict[int, float] | None = None
    safety_status: str = "feasible"


@dataclass(frozen=True)
class PlannedLaunchEvent:
    vehicle_id: int
    uav_id: int
    tuple_id: str
    launch_layer: int
    task_ids: tuple[int, ...]
    recovery_layer: int | None = None
    recovery_node: object = None
    completion_layers: tuple[tuple[int, int], ...] = ()
    expected_energy: float = 0.0
    energy_std: float = 0.0
    battery_capacity: float = 0.0
    reserve_ratio: float = 0.0


@dataclass(frozen=True, order=True)
class PlannedTaskEvent:
    task_id: int
    vehicle_id: int
    uav_id: int
    completion_layer: int


@dataclass(frozen=True)
class PlannedRecoveryEvent:
    vehicle_id: int
    uav_id: int
    recovery_node: object
    recovery_layer: int
    ready_layer: int | None = None


@dataclass(frozen=True)
class PlannedVehicleArcEvent:
    arc_id: str
    vehicle_id: int
    head: object
    arrival_layer: int
    duty_time: float
    departure_layer: int | None = None
    tail: object = None


def record_plan(
    state,
    arcs,
    tuples,
    selected_arc_ids,
    selected_tuple_ids,
):
    arcs = tuple(arcs)
    tuples = tuple(tuples)
    selected_arc_ids = frozenset(selected_arc_ids)
    selected_tuple_ids = frozenset(selected_tuple_ids)
    arc_domain = {arc.arc_id for arc in arcs}
    tuple_domain = {item.tuple_id for item in tuples}
    if selected_arc_ids - arc_domain:
        raise ValueError("selected arc lies outside the current plan domain")
    if selected_tuple_ids - tuple_domain:
        raise ValueError("selected tuple lies outside the current plan domain")
    active_vehicle_ids = {
        arc.vehicle_id for arc in arcs if arc.arc_id in selected_arc_ids
    }
    vehicles = {
        vehicle_id: replace(
            vehicle,
            must_return=(
                vehicle.must_return
                or (
                    vehicle_id in active_vehicle_ids
                    and vehicle.node_id != "sink"
                )
            ),
        )
        for vehicle_id, vehicle in state.vehicles.items()
    }
    return replace(
        state,
        vehicles=vehicles,
        previous_arc_values={
            arc.arc_id: int(arc.arc_id in selected_arc_ids) for arc in arcs
        },
        previous_tuple_values={
            item.tuple_id: int(item.tuple_id in selected_tuple_ids)
            for item in tuples
        },
    )


def _event_sort_key(event):
    if isinstance(event, PlannedLaunchEvent):
        return (event.launch_layer, -1, event.vehicle_id, event.uav_id, event.tuple_id)
    if isinstance(event, PlannedTaskEvent):
        return (
            event.completion_layer,
            0,
            event.vehicle_id,
            event.uav_id,
            event.task_id,
        )
    if isinstance(event, PlannedVehicleArcEvent):
        return (
            event.arrival_layer,
            1,
            event.vehicle_id,
            event.arc_id,
        )
    return (
        event.recovery_layer,
        2,
        event.vehicle_id,
        event.uav_id,
        repr(event.recovery_node),
    )


def execute_due_events(
    state,
    events,
    end_layer: int,
    blocked_vehicle_ids=frozenset(),
    blocked_uav_ids=frozenset(),
):
    if end_layer < state.current_layer:
        raise ValueError("execution end layer precedes current layer")
    blocked_vehicle_ids = set(blocked_vehicle_ids)
    blocked_uav_ids = set(blocked_uav_ids)
    ordered_events = tuple(sorted(events, key=_event_sort_key))
    launched_events = tuple(
        event for event in ordered_events
        if isinstance(event, PlannedLaunchEvent)
        and event.launch_layer <= end_layer
        and event.vehicle_id not in blocked_vehicle_ids
        and event.uav_id not in blocked_uav_ids
    )
    completed_events = tuple(
        event
        for event in ordered_events
        if isinstance(event, PlannedTaskEvent)
        if event.task_id in state.remaining_tasks
        and event.completion_layer <= end_layer
        and event.vehicle_id not in blocked_vehicle_ids
        and event.uav_id not in blocked_uav_ids
    )
    completed_tasks = frozenset(event.task_id for event in completed_events)
    recovered_events = tuple(
        event
        for event in ordered_events
        if isinstance(event, PlannedRecoveryEvent)
        and event.recovery_layer <= end_layer
        and event.vehicle_id not in blocked_vehicle_ids
        and event.uav_id not in blocked_uav_ids
    )
    completed_vehicle_events = tuple(
        event
        for event in ordered_events
        if isinstance(event, PlannedVehicleArcEvent)
        and event.arrival_layer <= end_layer
        and event.vehicle_id not in blocked_vehicle_ids
    )
    pending = tuple(
        event
        for event in ordered_events
        if (
            isinstance(event, PlannedTaskEvent)
            and event.task_id in state.remaining_tasks
            and event.task_id not in completed_tasks
        )
        or (
            isinstance(event, PlannedRecoveryEvent)
            and event not in recovered_events
        )
        or (
            isinstance(event, PlannedVehicleArcEvent)
            and event not in completed_vehicle_events
        )
        or (
            isinstance(event, PlannedLaunchEvent)
            and event not in launched_events
        )
    )
    vehicle_events = {vehicle_id: [] for vehicle_id in state.vehicles}
    for event in completed_vehicle_events:
        vehicle_events[event.vehicle_id].append(event)
    for event in recovered_events:
        vehicle_events[event.vehicle_id].append(event)
    vehicles = {}
    for vehicle_id, vehicle in state.vehicles.items():
        updated = vehicle
        for event in sorted(vehicle_events[vehicle_id], key=_event_sort_key):
            if isinstance(event, PlannedVehicleArcEvent):
                remaining_duty = updated.remaining_duty_time
                if remaining_duty is not None:
                    remaining_duty = max(0.0, remaining_duty - event.duty_time)
                updated = replace(
                    updated,
                    node_id=event.head,
                    available_layer=max(updated.available_layer, event.arrival_layer),
                    must_return=event.head != "sink",
                    remaining_duty_time=remaining_duty,
                    active_arc=None,
                    active_arc_arrival_layer=None,
                    active_arc_id=None,
                    active_arc_departure_layer=None,
                    active_arc_duty_time=0.0,
                )
            else:
                updated = replace(
                    updated,
                    node_id=event.recovery_node,
                    available_layer=max(updated.available_layer, event.recovery_layer),
                )
        active_events = [
            event for event in pending
            if isinstance(event, PlannedVehicleArcEvent)
            and event.vehicle_id == vehicle_id
            and event.departure_layer is not None
            and event.departure_layer < end_layer < event.arrival_layer
            and vehicle_id not in blocked_vehicle_ids
        ]
        if len(active_events) > 1:
            raise ValueError("overlapping vehicle arcs during execution")
        if active_events:
            event = active_events[0]
            continuing = updated.active_arc_id == event.arc_id
            updated = replace(
                updated,
                active_arc=updated.active_arc if continuing else (event.tail, event.head),
                active_arc_id=event.arc_id,
                active_arc_departure_layer=(
                    updated.active_arc_departure_layer if continuing else event.departure_layer
                ),
                active_arc_arrival_layer=event.arrival_layer,
                active_arc_duty_time=event.duty_time,
                available_layer=event.arrival_layer,
                must_return=True,
            )
        vehicles[vehicle_id] = updated
    uavs = dict(state.uavs)
    active_service_ids = dict(state.active_service_ids)
    airborne_services = dict(state.airborne_services)
    energy_consumed = {uav_id: 0.0 for uav_id in uavs}
    safety_status = "feasible"
    cursor = state.current_layer

    def consume_until(layer):
        nonlocal cursor, safety_status
        elapsed = max(0, layer - cursor)
        for uav_id, flight in airborne_services.items():
            uav = uavs[uav_id]
            if not uav.is_airborne or uav.remaining_energy is None:
                continue
            used = elapsed * flight.energy_rate
            energy_consumed[uav_id] += used
            if used > uav.remaining_energy + 1e-9:
                safety_status = "energy_exhausted"
            uavs[uav_id] = replace(uav, remaining_energy=max(0.0, uav.remaining_energy - used))
        cursor = max(cursor, layer)

    for event in sorted(launched_events + recovered_events, key=_event_sort_key):
        event_layer = event.launch_layer if isinstance(event, PlannedLaunchEvent) else event.recovery_layer
        consume_until(event_layer)
        uav = uavs[event.uav_id]
        if isinstance(event, PlannedLaunchEvent):
            if uav.is_airborne:
                raise ValueError("cannot relaunch an airborne UAV")
            if event.launch_layer < uav.available_layer:
                raise ValueError("UAV launch precedes recovery and turnaround readiness")
            uavs[event.uav_id] = replace(
                uav, is_airborne=True, remaining_tasks=event.task_ids,
                # New sorties use a replaced battery under the manuscript assumption.
                remaining_energy=(event.battery_capacity if event.battery_capacity > 0 else uav.remaining_energy),
            )
            active_service_ids[event.uav_id] = event.tuple_id
            if event.recovery_layer is not None:
                airborne_services[event.uav_id] = ActiveFlight(
                    event.tuple_id, event.vehicle_id, event.uav_id,
                    event.launch_layer, event.recovery_layer, event.recovery_node,
                    event.completion_layers, event.expected_energy, event.energy_std,
                    event.battery_capacity, event.reserve_ratio,
                )
        else:
            uavs[event.uav_id] = replace(
                uav, available_layer=max(uav.available_layer, event.ready_layer or event.recovery_layer),
                is_airborne=False, remaining_tasks=(),
            )
            active_service_ids.pop(event.uav_id, None)
            airborne_services.pop(event.uav_id, None)
    consume_until(end_layer)
    return (
        ExecutionResult(
            end_layer=end_layer,
            completed_tasks=completed_tasks,
            vehicles=vehicles,
            uavs=uavs,
            active_service_ids=active_service_ids,
            airborne_services=airborne_services,
            energy_consumed=energy_consumed,
            safety_status=safety_status,
        ),
        pending,
    )


def advance_state(state, result: ExecutionResult):
    done = state.done_tasks | result.completed_tasks
    remaining = state.remaining_tasks - result.completed_tasks
    uavs = {
        uav_id: replace(
            uav,
            remaining_tasks=tuple(
                task for task in uav.remaining_tasks if task not in done
            ) if uav.is_airborne else (),
        )
        for uav_id, uav in (result.uavs if result.uavs is not None else state.uavs).items()
    }
    airborne_tasks = {
        uav_id: uav.remaining_tasks
        for uav_id, uav in uavs.items()
        if uav.is_airborne
    }
    return replace(
        state,
        epoch=state.epoch + 1,
        current_layer=result.end_layer,
        done_tasks=done,
        remaining_tasks=remaining,
        vehicles=result.vehicles or state.vehicles,
        uavs=uavs,
        airborne_tasks=airborne_tasks,
        locked_arc_ids=(state.locked_arc_ids - frozenset(
            vehicle.active_arc_id for vehicle in state.vehicles.values()
            if vehicle.active_arc_id is not None
        )) | frozenset(
            vehicle.active_arc_id for vehicle in (result.vehicles or state.vehicles).values()
            if vehicle.active_arc_id is not None
        ),
        airborne_services={
            key: value for key, value in (
                result.airborne_services if result.airborne_services is not None else state.airborne_services
            ).items() if key in airborne_tasks
        },
        active_service_ids={
            key: value for key, value in (
                result.active_service_ids
                if result.active_service_ids is not None else state.active_service_ids
            ).items() if key in airborne_tasks
        },
    )
