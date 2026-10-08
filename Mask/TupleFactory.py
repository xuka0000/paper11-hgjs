from dataclasses import dataclass
from hashlib import sha256
from math import sqrt


@dataclass(frozen=True)
class EnergyMoments:
    mean: float
    std: float


@dataclass(frozen=True)
class EnergyCertificate:
    certificate_id: str
    mean: float
    std: float
    reliable_bound: float
    full_capacity: float
    reserve_ratio: float
    usable_capacity: float
    alpha: float
    admitted: bool


@dataclass(frozen=True)
class ServiceResourceKeys:
    interval_keys: tuple[str, ...]
    operation_keys: tuple[str, ...]


def build_service_resource_keys(
    vehicle_id: int,
    uav_id: int,
    launch_node,
    recovery_node,
    launch_layer: int,
    recovery_layer: int,
    preparation_layers: int,
    recovery_service_layers: int,
    turnaround_layers: int,
) -> ServiceResourceKeys:
    durations = (
        preparation_layers,
        recovery_service_layers,
        turnaround_layers,
    )
    if any(int(value) < 0 for value in durations):
        raise ValueError("resource durations must be nonnegative")
    if recovery_layer < launch_layer:
        raise ValueError("recovery layer precedes launch layer")
    launch_keys = (
        f"op:{vehicle_id}:{launch_node}:{layer}"
        for layer in range(launch_layer, launch_layer + preparation_layers)
    )
    recovery_keys = (
        f"op:{vehicle_id}:{recovery_node}:{layer}"
        for layer in range(
            recovery_layer,
            recovery_layer + recovery_service_layers,
        )
    )
    operation_keys = tuple(dict.fromkeys((*launch_keys, *recovery_keys)))
    occupied_until = recovery_layer + recovery_service_layers + turnaround_layers
    interval_keys = tuple(
        f"uav:{uav_id}:{layer}"
        for layer in range(launch_layer, occupied_until)
    )
    return ServiceResourceKeys(interval_keys, operation_keys)


def cantelli_bound(moments: EnergyMoments, alpha: float) -> float:
    if not 0.5 < alpha < 1.0:
        raise ValueError("alpha must lie in (0.5, 1)")
    return moments.mean + sqrt(alpha / (1.0 - alpha)) * moments.std


def admit_new_tuple(q: float, full_capacity: float, reserve_ratio: float) -> bool:
    return q <= full_capacity * (1.0 - reserve_ratio) + 1e-9


def build_energy_certificate(
    moments: EnergyMoments,
    alpha: float,
    full_capacity: float,
    reserve_ratio: float,
) -> EnergyCertificate:
    if moments.mean < 0 or moments.std < 0:
        raise ValueError("energy moments must be nonnegative")
    if full_capacity <= 0:
        raise ValueError("full capacity must be positive")
    if not 0 <= reserve_ratio < 1:
        raise ValueError("reserve ratio must lie in [0, 1)")
    reliable = cantelli_bound(moments, alpha)
    usable = full_capacity * (1.0 - reserve_ratio)
    payload = (
        float(moments.mean),
        float(moments.std),
        float(reliable),
        float(full_capacity),
        float(reserve_ratio),
        float(alpha),
    )
    certificate_id = "energy:" + sha256(
        repr(payload).encode("utf-8")
    ).hexdigest()[:20]
    return EnergyCertificate(
        certificate_id,
        float(moments.mean),
        float(moments.std),
        float(reliable),
        float(full_capacity),
        float(reserve_ratio),
        float(usable),
        float(alpha),
        admit_new_tuple(reliable, full_capacity, reserve_ratio),
    )


@dataclass(frozen=True)
class StructuralTuple:
    tuple_id: str
    vehicle_id: int
    uav_id: int
    task_ids: tuple[int, ...]
    launch_node: tuple[int, int]
    recovery_node: tuple[int, int]

    def is_eligible(self, state) -> bool:
        committed = {task for tasks in state.airborne_tasks.values() for task in tasks}
        return set(self.task_ids) <= set(state.remaining_tasks) and not (
            set(self.task_ids) & committed
        )


@dataclass(frozen=True)
class AirResidualTuple:
    residual_id: str
    uav_id: int
    remaining_tasks: tuple[int, ...]
    launch_node: None
    recovery_stop: int
    recovery_layer: int
    q_bound: float


def build_air_residual(
    uav,
    recovery_stop: int,
    recovery_layer: int,
    moments: EnergyMoments,
    full_capacity: float,
    reserve_ratio: float,
    alpha: float,
) -> AirResidualTuple:
    q = cantelli_bound(moments, alpha)
    usable = float(uav.remaining_energy) - reserve_ratio * full_capacity
    if q > usable + 1e-9:
        raise ValueError("unsafe airborne recovery")
    return AirResidualTuple(
        f"air:{uav.uav_id}:{recovery_stop}:{recovery_layer}",
        uav.uav_id,
        uav.remaining_tasks,
        None,
        recovery_stop,
        recovery_layer,
        q,
    )
