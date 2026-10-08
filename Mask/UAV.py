from dataclasses import dataclass


@dataclass(frozen=True)
class UavState:
    uav_id: int
    owner_vehicle_id: int
    available_layer: int
    is_airborne: bool = False
    remaining_energy: float | None = None
    remaining_tasks: tuple[int, ...] = ()
