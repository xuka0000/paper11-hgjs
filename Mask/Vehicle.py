from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleState:
    vehicle_id: int
    node_id: int | str
    available_layer: int
    active_arc: tuple[int | str, int | str] | None = None
    active_arc_arrival_layer: int | None = None
    must_return: bool = False
    remaining_duty_time: float | None = None
    active_arc_id: str | None = None
    active_arc_departure_layer: int | None = None
    active_arc_duty_time: float = 0.0
