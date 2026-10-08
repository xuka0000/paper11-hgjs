from dataclasses import dataclass, field
from typing import Mapping

from Mask.UAV import UavState
from Mask.Vehicle import VehicleState


@dataclass(frozen=True)
class RollingState:
    epoch: int
    current_layer: int
    done_tasks: frozenset[int]
    remaining_tasks: frozenset[int]
    vehicles: Mapping[int, VehicleState]
    uavs: Mapping[int, UavState]
    locked_arc_ids: frozenset[str] = frozenset()
    locked_tuple_ids: frozenset[str] = frozenset()
    airborne_tasks: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    previous_arc_values: Mapping[str, int] = field(default_factory=dict)
    previous_tuple_values: Mapping[str, int] = field(default_factory=dict)
    active_service_ids: Mapping[int, str] = field(default_factory=dict)
    airborne_services: Mapping[int, object] = field(default_factory=dict)
    energy_observations: Mapping[int, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.current_layer < 1:
            raise ValueError("current_layer must be positive")
        if self.done_tasks & self.remaining_tasks:
            raise ValueError("done and remaining tasks overlap")
        airborne = {task for tasks in self.airborne_tasks.values() for task in tasks}
        if airborne & self.done_tasks:
            raise ValueError("airborne task already completed")
        object.__setattr__(self, "vehicles", dict(self.vehicles))
        object.__setattr__(self, "uavs", dict(self.uavs))
        object.__setattr__(self, "airborne_tasks", dict(self.airborne_tasks))
        object.__setattr__(self, "active_service_ids", dict(self.active_service_ids))
        object.__setattr__(self, "airborne_services", dict(self.airborne_services))
        object.__setattr__(self, "energy_observations", dict(self.energy_observations))

    def as_kwargs(self) -> dict[str, object]:
        return dict(self.__dict__)
