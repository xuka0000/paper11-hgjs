"""Isolated emergency recovery on existing physical road edges.

No new UAV launches, reassignment, or changed recovery stops. A closure prevents
new entry; an occupied edge retains its recorded arrival and unpaid duty. The
forecast is the registered synthetic baseline, with no future closure inputs.
"""
from dataclasses import dataclass, replace
from heapq import heappop, heappush

from Agent.Tool.Benchmark.benchmark_v3 import _formal_v3_weather
from Agent.Tool.Evaluation.dynamic_evaluation import DynamicProblem, validate_dynamic_plan
from Mask.Road import RoadArc
from Mask.ServiceTuple import SearchArc, SearchState, SearchTuple
from Mask.TimeNetwork import arrival_layer
from Mask.TupleFactory import EnergyMoments, build_air_residual, build_service_resource_keys

HORIZON = 256


def validate_recovery_plan(problem):
    """Expose a depot waiting interval to the existing time-node validator.

    Its sink is an absorbing optimization terminal, whereas the physical depot
    may still host recovery service. A validation-only terminal separates these
    objects. No artificial edge is returned to the planner or execution engine.
    """
    alias = '__physical_depot__'
    arcs = [replace(a, tail=alias if a.tail == 'sink' else a.tail,
                    head=alias if a.head == 'sink' else a.head)
            for a in problem.search_state.arcs]
    for vid in sorted({a.vehicle_id for a in arcs}):
        route = [a for a in arcs if a.vehicle_id == vid]
        last = route[-1]
        if last.head == alias:
            arcs.append(SearchArc(f'validation_terminal:{vid}', vid, alias, 'sink',
                                  last.arrival_layer, last.arrival_layer, 'validation_only'))
    tuples = tuple(replace(t, recovery_node=alias if t.recovery_node == 'sink' else t.recovery_node)
                   for t in problem.search_state.tuples)
    search = SearchState(tuple(arcs), tuples, frozenset(a.arc_id for a in arcs),
                         problem.search_state.selected_tuple_ids)
    return validate_dynamic_plan(replace(problem, search_state=search), search)


@dataclass(frozen=True)
class SegmentClosure:
    edge_id: str
    reveal_layer: int

    def __post_init__(self):
        if not self.edge_id or self.reveal_layer < 1:
            raise ValueError('Closure requires a physical edge and positive reveal layer')


def edge_arrival(instance, edge, departure, weather):
    road = RoadArc(0, 1, edge.geometry.length_3d,
                   edge.safe_speed_kmph / 60., 1., edge.available)
    if not road.available:
        return None
    return arrival_layer(road, departure, weather, HORIZON, 1.,
                         instance.road_beta_w, instance.road_beta_eta)


def earliest_path(instance, start, goal, departure, closed, weather):
    """FIFO earliest arrival with deterministic edge-id tie ordering."""
    outgoing = {}
    for edge in sorted(instance.physical_scenario.road.edges, key=lambda e: e.edge_id):
        if edge.available and edge.edge_id not in closed:
            outgoing.setdefault(edge.tail, []).append(edge)
    queue = [(departure, start)]
    best = {start: departure}
    previous = {}
    while queue:
        layer, node = heappop(queue)
        if layer != best[node]:
            continue
        if node == goal:
            path = []
            while node != start:
                edge = previous[node]
                path.append(edge)
                node = edge.tail
            return tuple(reversed(path))
        for edge in outgoing.get(node, ()):
            end = edge_arrival(instance, edge, layer, weather)
            if end is not None and end < best.get(edge.head, HORIZON + 1):
                best[edge.head] = end
                previous[edge.head] = edge
                heappush(queue, (end, edge.head))
    return None


def build_recovery_problem(instance, state, closures=(), *, allow_detour=False):
    """Return a fully specified feasible plan or an explicit diagnostic failure.

    Original remaining legs are retained when open. Only blocked legs between
    mandatory recovery stops and the depot are reconstructed. Recovery stops keep
    their original order. All selected variables are fixed emergency obligations.
    """
    edges = {e.edge_id: e for e in instance.physical_scenario.road.edges}
    if any(c.edge_id not in edges for c in closures):
        raise ValueError('Unknown physical road edge')
    if any(c.reveal_layer > state.current_layer for c in closures):
        raise ValueError('Unrevealed closure supplied to decision')
    closed = {c.edge_id for c in closures}
    by_pair = {(e.tail, e.head): e for e in edges.values()}
    if len(by_pair) != len(edges):
        raise ValueError('Parallel physical edges require explicit original arc mapping')
    weather = _formal_v3_weather(instance.seed, HORIZON)
    arcs, tuples, capacities, failures, route_records = [], [], {}, [], []
    capacity = min(instance.operation_capacity.values(), default=1)
    service_layers = instance.recovery_service_layers
    ready_layers = instance.turnaround_layers
    airborne = {i for i, u in state.uavs.items() if u.is_airborne}
    if airborne != set(state.airborne_services):
        failures.append('airborne_flight_state_missing')
    for vid, vehicle in sorted(state.vehicles.items()):
        flights = sorted((f for f in state.airborne_services.values() if f.vehicle_id == vid),
                         key=lambda f: (f.recovery_layer, f.uav_id))
        required = vehicle.must_return or vehicle.active_arc is not None or bool(flights)
        if not required:
            continue
        original = sorted((a for a in instance.search_state.arcs if a.vehicle_id == vid),
                          key=lambda a: (a.departure_layer, a.arrival_layer, a.arc_id))
        original_nodes = [original[0].tail] + [a.head for a in original] if original else []
        route = []
        layer = max(state.current_layer, vehicle.available_layer)
        node = vehicle.node_id
        if vehicle.active_arc is not None:
            matching = [a for a in original if a.arc_id == vehicle.active_arc_id]
            if (len(matching) != 1 or vehicle.active_arc_arrival_layer is None
                    or vehicle.active_arc_arrival_layer < state.current_layer
                    or matching[0].head != vehicle.active_arc[1]):
                failures.append(f'vehicle={vid}:invalid_committed_arc')
                continue
            node = vehicle.active_arc[1]
            layer = vehicle.active_arc_arrival_layer
            route.append(SearchArc(vehicle.active_arc_id, vid, 'source', node,
                                   state.current_layer, layer, 'committed', 0.,
                                   vehicle.active_arc_duty_time))
        elif node != 'source':
            route.append(SearchArc(f'anchor:{vid}:{state.current_layer}:{node}', vid,
                                   'source', node, state.current_layer, layer, 'anchor'))

        def wait_until(end):
            nonlocal layer
            if end > layer:
                route.append(SearchArc(f'wait:{vid}:{node}:{layer}:{end}', vid, node,
                                       node, layer, end, 'wait', 0., float(end - layer)))
                layer = end

        # One tuple per airborne UAV. Sequential recovery also respects capacity 1.
        targets = [(f.recovery_node, f) for f in flights] + [('sink', None)]
        failed = False
        for goal, flight in targets:
            reference = None
            if node in original_nodes and goal in original_nodes:
                left, right = original_nodes.index(node), original_nodes.index(goal)
                if right >= left:
                    reference = tuple(by_pair[(a.tail, a.head)] for a in original[left:right])
            path = reference
            blocked = path is None or any(e.edge_id in closed or not e.available for e in path)
            if blocked:
                path = earliest_path(instance, node, goal, layer, closed, weather) if allow_detour else None
            if path is None:
                failures.append(f'vehicle={vid}:unreachable_{"recovery" if flight else "return"}:{goal}')
                failed = True
                break
            start_node, start_layer = node, layer
            for edge in path:
                end = edge_arrival(instance, edge, layer, weather)
                if end is None:
                    failures.append(f'vehicle={vid}:road_horizon_exceeded')
                    failed = True
                    break
                route.append(SearchArc(f'emergency:{vid}:{edge.edge_id}:{layer}', vid,
                                       edge.tail, edge.head, layer, end, 'road',
                                       edge.geometry.length_3d, float(end - layer)))
                node, layer = edge.head, end
            if failed:
                break
            route_records.append(dict(vehicle_id=vid, start=start_node, goal=goal,
                departure=start_layer, arrival=layer, detoured=bool(blocked),
                physical_edge_ids=[e.edge_id for e in path]))
            if flight is None:
                continue
            # No early recovery without a model for changing an airborne flight.
            wait_until(max(layer, flight.recovery_layer,
                           max((t for _, t in flight.completion_layers), default=layer)))
            moments = EnergyMoments(*flight.remaining_moments(state.current_layer, layer))
            try:
                residual = build_air_residual(state.uavs[flight.uav_id], goal, layer,
                    moments, flight.battery_capacity, flight.reserve_ratio, instance.energy_alpha)
            except ValueError as error:
                failures.append(f'uav={flight.uav_id}:energy:{error}')
                failed = True
                break
            resources = build_service_resource_keys(vid, flight.uav_id, '', goal,
                state.current_layer, layer, 0, service_layers, ready_layers)
            capacities.update({key: capacity for key in resources.operation_keys})
            tasks = tuple(t for t, _ in flight.completion_layers if t in state.remaining_tasks)
            tuples.append(SearchTuple(residual.residual_id, vid, flight.uav_id, tasks, '',
                goal, state.current_layer, layer, required_arc_ids=tuple(a.arc_id for a in route),
                interval_keys=resources.interval_keys, operation_keys=resources.operation_keys,
                completion_layers=tuple((t, k) for t, k in flight.completion_layers if t in tasks),
                expected_energy=moments.mean, energy_std=moments.std,
                reliable_energy=residual.q_bound, battery_capacity=flight.battery_capacity,
                reserve_ratio=flight.reserve_ratio, admission_certificate=residual.residual_id,
                is_air_residual=True))
            wait_until(layer + service_layers)
        if not failed:
            excess = sum(a.duty_time for a in route) - (vehicle.remaining_duty_time
                if vehicle.remaining_duty_time is not None else float('inf'))
            if excess > 1e-8:
                failures.append(f'vehicle={vid}:remaining_duty_exceeded')
        arcs.extend(route)
    selected_arcs = frozenset(a.arc_id for a in arcs)
    selected_tuples = frozenset(t.tuple_id for t in tuples)
    search = SearchState(tuple(arcs), tuple(tuples), selected_arcs, selected_tuples)
    residual_state = replace(state, locked_arc_ids=selected_arcs, locked_tuple_ids=selected_tuples)
    problem = DynamicProblem(tuple(sorted(state.remaining_tasks)),
        {t: instance.miss_penalty[t] for t in state.remaining_tasks}, capacities,
        search, residual_state, 'infeasible_emergency_recovery' if failures else 'feasible',
        ';'.join(failures))
    if not failures:
        report = validate_recovery_plan(problem)
        if not report.ok:
            problem = replace(problem, safety_status='infeasible_emergency_recovery',
                              safety_reason=f'validation:{report.first_violation}:{report.object_id}')
    return problem, route_records
