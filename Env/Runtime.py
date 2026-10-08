"""Runtime observations and the declared constant-rate simulation model.

Energy units are inherited from the instance. Model estimates are not telemetry.
The launch-time mean and standard deviation define fixed per-layer rates; an
extended recovery time therefore increases the residual energy requirement.
"""
from dataclasses import dataclass, replace
from math import isfinite


@dataclass(frozen=True)
class ActiveFlight:
    tuple_id: str
    vehicle_id: int
    uav_id: int
    launch_layer: int
    recovery_layer: int
    recovery_node: object
    completion_layers: tuple[tuple[int, int], ...]
    expected_energy: float
    energy_std: float
    battery_capacity: float
    reserve_ratio: float

    def remaining_moments(self, current_layer, recovery_layer):
        duration = max(1, self.recovery_layer - self.launch_layer)
        share = max(0, recovery_layer - current_layer) / duration
        return self.expected_energy * share, self.energy_std * share

    @property
    def energy_rate(self):
        return self.expected_energy / max(1, self.recovery_layer - self.launch_layer)


@dataclass(frozen=True)
class EnergyObservation:
    uav_id: int
    layer: int
    remaining_energy: float
    source: str

    def __post_init__(self):
        if self.layer < 1 or not isfinite(self.remaining_energy) or self.remaining_energy < 0:
            raise ValueError("energy observation must be finite, nonnegative and dated")
        if not self.source.strip():
            raise ValueError("energy observation source is required")


def apply_energy_observations(state, observations):
    uavs = dict(state.uavs)
    records = dict(state.energy_observations)
    seen = set()
    for observation in observations:
        if observation.layer != state.current_layer:
            raise ValueError("energy observation is not from the current decision layer")
        if observation.uav_id not in uavs:
            raise ValueError("energy observation identifies an unknown UAV")
        if observation.uav_id in seen:
            raise ValueError("duplicate energy observation at the decision layer")
        seen.add(observation.uav_id)
        uavs[observation.uav_id] = replace(
            uavs[observation.uav_id], remaining_energy=observation.remaining_energy,
        )
        records[observation.uav_id] = observation
    return replace(state, uavs=uavs, energy_observations=records)
