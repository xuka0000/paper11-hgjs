from dataclasses import dataclass

from Mask.Road import RoadArc
from Mask.Weather import WeatherSlot, road_effective_speed


@dataclass(frozen=True)
class TimeNetwork:
    nodes: frozenset[tuple[int, int]]
    arcs: tuple[tuple[tuple[int, int], tuple[int, int], str], ...]
    vehicle_anchors: dict[int, tuple[int, int]]


def arrival_layer(
    arc: RoadArc,
    departure: int,
    weather_by_layer: dict[int, WeatherSlot],
    horizon: int,
    delta_t: float,
    beta_w: float,
    beta_eta: float,
) -> int | None:
    distance = 0.0
    for layer in range(departure, horizon):
        speed = road_effective_speed(
            arc.nominal_speed,
            arc.structure_factor,
            weather_by_layer[layer],
            beta_w,
            beta_eta,
        )
        distance += delta_t * speed
        if distance + 1e-12 >= arc.length:
            return layer + 1
    return None


def build_time_network(
    state,
    road_arcs,
    weather_by_layer,
    horizon: int,
    delta_t: float,
    beta_w: float,
    beta_eta: float,
) -> TimeNetwork:
    nodes = {
        (node, layer)
        for arc in road_arcs
        for node in (arc.tail, arc.head)
        for layer in range(state.current_layer, horizon + 1)
    }
    anchors = {}
    for vehicle_id, vehicle in state.vehicles.items():
        anchors[vehicle_id] = (
            (vehicle.active_arc[1], vehicle.active_arc_arrival_layer)
            if vehicle.active_arc
            else (vehicle.node_id, max(state.current_layer, vehicle.available_layer))
        )
        nodes.add(anchors[vehicle_id])
    arcs_out = []
    for arc in road_arcs:
        if not arc.available:
            continue
        for departure in range(state.current_layer, horizon):
            arrival = arrival_layer(
                arc,
                departure,
                weather_by_layer,
                horizon,
                delta_t,
                beta_w,
                beta_eta,
            )
            if arrival is not None:
                arcs_out.append(((arc.tail, departure), (arc.head, arrival), "road"))
    return TimeNetwork(frozenset(nodes), tuple(arcs_out), anchors)
