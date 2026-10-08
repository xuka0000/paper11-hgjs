from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping

from Mask.Geometry import Point3D, Polyline3D
from Mask.PowerNetwork import PowerLine, PowerNetwork, Tower


@dataclass(frozen=True)
class RoadNode:
    node_id: str
    location: Point3D

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must not be empty")


@dataclass(frozen=True)
class RoadEdge:
    edge_id: str
    tail: str
    head: str
    geometry: Polyline3D
    road_class: str
    safe_speed_kmph: float
    weather_zone: str
    available: bool

    def __post_init__(self) -> None:
        if not self.edge_id or not self.tail or not self.head:
            raise ValueError("road edge identifiers must not be empty")
        if self.tail == self.head:
            raise ValueError("road edge must connect distinct nodes")
        if self.safe_speed_kmph <= 0.0:
            raise ValueError("safe_speed_kmph must be positive")


@dataclass(frozen=True)
class RoadNetwork:
    nodes: tuple[RoadNode, ...]
    edges: tuple[RoadEdge, ...]
    depot_node_id: str

    def __post_init__(self) -> None:
        nodes = tuple(sorted(self.nodes, key=lambda item: item.node_id))
        edges = tuple(sorted(self.edges, key=lambda item: item.edge_id))
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        node_by_id = _unique_by_id(nodes, "node_id", "road node")
        _unique_by_id(edges, "edge_id", "road edge")
        if self.depot_node_id not in node_by_id:
            raise ValueError("road depot references unknown node")
        for edge in edges:
            if edge.tail not in node_by_id or edge.head not in node_by_id:
                raise ValueError("road edge references unknown node")
            if edge.geometry.points[0] != node_by_id[edge.tail].location:
                raise ValueError("road edge start does not match tail node")
            if edge.geometry.points[-1] != node_by_id[edge.head].location:
                raise ValueError("road edge end does not match head node")


@dataclass(frozen=True)
class CandidateStop:
    stop_id: str
    location: Point3D
    road_node_id: str | None = None
    road_edge_id: str | None = None
    edge_fraction: float | None = None

    def __post_init__(self) -> None:
        if not self.stop_id:
            raise ValueError("stop_id must not be empty")
        on_node = self.road_node_id is not None
        on_edge = self.road_edge_id is not None
        if on_node == on_edge:
            raise ValueError("stop must reference exactly one road node or edge")
        if on_node and self.edge_fraction is not None:
            raise ValueError("node stop must not define edge_fraction")
        if on_edge and (
            self.edge_fraction is None or not 0.0 <= self.edge_fraction <= 1.0
        ):
            raise ValueError("edge stop requires edge_fraction in [0, 1]")


@dataclass(frozen=True)
class VehicleSpec:
    vehicle_id: int
    depot_stop_id: str
    max_duty_minutes: float
    operation_capacity: int = 1
    operation_minutes: float = 5.0

    def __post_init__(self) -> None:
        if self.vehicle_id <= 0:
            raise ValueError("vehicle_id must be positive")
        if self.max_duty_minutes <= 0.0:
            raise ValueError("max_duty_minutes must be positive")
        if self.operation_capacity <= 0:
            raise ValueError("operation_capacity must be positive")
        if self.operation_minutes <= 0.0:
            raise ValueError("operation_minutes must be positive")


@dataclass(frozen=True)
class UavSpec:
    uav_id: int
    owner_vehicle_id: int
    cruise_speed_kmph: float
    battery_capacity: float
    reserve_ratio: float
    sensor_types: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_types", tuple(self.sensor_types))
        if self.uav_id <= 0 or self.owner_vehicle_id <= 0:
            raise ValueError("UAV identifiers must be positive")
        if self.cruise_speed_kmph <= 0.0 or self.battery_capacity <= 0.0:
            raise ValueError("UAV speed and capacity must be positive")
        if not 0.0 <= self.reserve_ratio < 1.0:
            raise ValueError("reserve_ratio must be in [0, 1)")
        if not self.sensor_types:
            raise ValueError("sensor_types must not be empty")


@dataclass(frozen=True)
class PhysicalServiceCandidate:
    candidate_id: str
    vehicle_id: int
    uav_id: int
    launch_stop_id: str
    recovery_stop_id: str
    task_ids: tuple[str, ...]
    duration_minutes: float
    completion_offsets_minutes: tuple[tuple[str, float], ...]
    expected_energy: float
    energy_std: float
    reliable_energy: float
    admitted: bool

    def __post_init__(self) -> None:
        task_ids = tuple(self.task_ids)
        completions = tuple(self.completion_offsets_minutes)
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "completion_offsets_minutes", completions)
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("candidate tasks must be nonempty unique values")
        if self.duration_minutes <= 0.0:
            raise ValueError("duration_minutes must be positive")
        completion_tasks = tuple(task_id for task_id, _ in completions)
        if set(completion_tasks) != set(task_ids) or len(completion_tasks) != len(
            set(completion_tasks)
        ):
            raise ValueError("completion offsets must cover candidate tasks once")
        if any(
            offset <= 0.0 or offset > self.duration_minutes
            for _, offset in completions
        ):
            raise ValueError("completion offset is outside candidate duration")
        if min(self.expected_energy, self.energy_std, self.reliable_energy) < 0.0:
            raise ValueError("candidate energy fields must be nonnegative")


@dataclass(frozen=True)
class ScenarioSpec:
    split: str
    size: str
    geometry_family: str
    seed: int
    task_count: int
    road_node_count: int
    stop_count: int
    vehicle_count: int
    uav_count: int
    horizon_minutes: float

    def __post_init__(self) -> None:
        counts = (
            self.task_count,
            self.road_node_count,
            self.stop_count,
            self.vehicle_count,
            self.uav_count,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("scenario counts must be positive")
        if self.horizon_minutes <= 0.0:
            raise ValueError("horizon_minutes must be positive")

    @property
    def road_geometry_seed(self) -> int:
        return _derived_seed("road", self.seed)

    @property
    def power_geometry_seed(self) -> int:
        return _derived_seed("power", self.seed)

    @property
    def candidate_seed(self) -> int:
        return _derived_seed("candidate", self.seed)


@dataclass(frozen=True)
class IrregularDualNetworkScenario:
    spec: ScenarioSpec
    road: RoadNetwork
    power: PowerNetwork
    stops: tuple[CandidateStop, ...]
    vehicles: tuple[VehicleSpec, ...]
    uavs: tuple[UavSpec, ...]
    candidates: tuple[PhysicalServiceCandidate, ...]
    miss_penalty: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        stops = tuple(sorted(self.stops, key=lambda item: item.stop_id))
        vehicles = tuple(sorted(self.vehicles, key=lambda item: item.vehicle_id))
        uavs = tuple(sorted(self.uavs, key=lambda item: item.uav_id))
        candidates = tuple(
            sorted(self.candidates, key=lambda item: item.candidate_id)
        )
        miss_penalty = tuple(sorted(self.miss_penalty))
        object.__setattr__(self, "stops", stops)
        object.__setattr__(self, "vehicles", vehicles)
        object.__setattr__(self, "uavs", uavs)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "miss_penalty", miss_penalty)
        stop_by_id = _unique_by_id(stops, "stop_id", "candidate stop")
        vehicle_by_id = _unique_by_id(vehicles, "vehicle_id", "vehicle")
        uav_by_id = _unique_by_id(uavs, "uav_id", "UAV")
        _unique_by_id(candidates, "candidate_id", "service candidate")
        road_node_by_id = {item.node_id: item for item in self.road.nodes}
        road_edge_ids = {item.edge_id for item in self.road.edges}
        tower_ids = {item.tower_id for item in self.power.towers}
        for stop in stops:
            if stop.road_node_id is not None:
                if stop.road_node_id not in road_node_by_id:
                    raise ValueError("stop references unknown road node")
                if stop.location != road_node_by_id[stop.road_node_id].location:
                    raise ValueError("node stop location does not match road node")
            elif stop.road_edge_id not in road_edge_ids:
                raise ValueError("stop references unknown road edge")
        for vehicle in vehicles:
            if vehicle.depot_stop_id not in stop_by_id:
                raise ValueError("vehicle depot references unknown stop")
        for uav in uavs:
            if uav.owner_vehicle_id not in vehicle_by_id:
                raise ValueError("UAV references unknown owner vehicle")
        for candidate in candidates:
            if (
                candidate.launch_stop_id not in stop_by_id
                or candidate.recovery_stop_id not in stop_by_id
            ):
                raise ValueError("candidate references unknown stop")
            if candidate.vehicle_id not in vehicle_by_id:
                raise ValueError("candidate references unknown vehicle")
            if candidate.uav_id not in uav_by_id:
                raise ValueError("candidate references unknown UAV")
            if uav_by_id[candidate.uav_id].owner_vehicle_id != candidate.vehicle_id:
                raise ValueError("candidate UAV is owned by another vehicle")
            if not set(candidate.task_ids) <= tower_ids:
                raise ValueError("candidate references unknown tower task")
        penalty_ids = [task_id for task_id, _ in miss_penalty]
        if len(penalty_ids) != len(set(penalty_ids)):
            raise ValueError("duplicate miss penalty task")
        if set(penalty_ids) != tower_ids:
            raise ValueError("miss penalties must cover every tower task")
        if any(value < 0.0 for _, value in miss_penalty):
            raise ValueError("miss penalties must be nonnegative")
        expected_counts = (
            self.spec.task_count,
            self.spec.road_node_count,
            self.spec.stop_count,
            self.spec.vehicle_count,
            self.spec.uav_count,
        )
        actual_counts = (
            len(self.power.towers),
            len(self.road.nodes),
            len(stops),
            len(vehicles),
            len(uavs),
        )
        if actual_counts != expected_counts:
            raise ValueError(
                f"scenario counts differ from spec: actual={actual_counts} expected={expected_counts}"
            )

    @property
    def candidate_pool_hash(self) -> str:
        return _sha256_payload(self.candidates)

    @property
    def scenario_hash(self) -> str:
        return _sha256_payload(self)

    def canonical_json(self) -> str:
        return canonical_json(self)


def _unique_by_id(items, attribute: str, label: str) -> dict:
    output = {}
    for item in items:
        item_id = getattr(item, attribute)
        if item_id in output:
            raise ValueError(f"duplicate {label} id: {item_id}")
        output[item_id] = item
    return output


def _derived_seed(role: str, seed: int) -> int:
    digest = sha256(f"{role}:{seed}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _normalize(value):
    if is_dataclass(value):
        return {field.name: _normalize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("canonical payload contains non-finite float")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value) -> str:
    return json.dumps(
        _normalize(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_payload(value) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def scenario_from_json(payload: str) -> IrregularDualNetworkScenario:
    data = json.loads(payload)
    road_data = data["road"]
    road = RoadNetwork(
        tuple(
            RoadNode(item["node_id"], _point_from_dict(item["location"]))
            for item in road_data["nodes"]
        ),
        tuple(
            RoadEdge(
                item["edge_id"],
                item["tail"],
                item["head"],
                _polyline_from_dict(item["geometry"]),
                item["road_class"],
                item["safe_speed_kmph"],
                item["weather_zone"],
                item["available"],
            )
            for item in road_data["edges"]
        ),
        road_data["depot_node_id"],
    )
    power_data = data["power"]
    power = PowerNetwork(
        tuple(
            Tower(
                item["tower_id"],
                _point_from_dict(item["location"]),
                item["risk"],
                item["asset_value"],
                item["topology_importance"],
                item["service_minutes"],
            )
            for item in power_data["towers"]
        ),
        tuple(
            PowerLine(
                item["line_id"],
                tuple(item["tower_ids"]),
                _polyline_from_dict(item["geometry"]),
            )
            for item in power_data["lines"]
        ),
    )
    return IrregularDualNetworkScenario(
        ScenarioSpec(**data["spec"]),
        road,
        power,
        tuple(
            CandidateStop(
                item["stop_id"],
                _point_from_dict(item["location"]),
                item["road_node_id"],
                item["road_edge_id"],
                item["edge_fraction"],
            )
            for item in data["stops"]
        ),
        tuple(VehicleSpec(**item) for item in data["vehicles"]),
        tuple(
            UavSpec(
                item["uav_id"],
                item["owner_vehicle_id"],
                item["cruise_speed_kmph"],
                item["battery_capacity"],
                item["reserve_ratio"],
                tuple(item["sensor_types"]),
            )
            for item in data["uavs"]
        ),
        tuple(
            PhysicalServiceCandidate(
                item["candidate_id"],
                item["vehicle_id"],
                item["uav_id"],
                item["launch_stop_id"],
                item["recovery_stop_id"],
                tuple(item["task_ids"]),
                item["duration_minutes"],
                tuple(
                    (task_id, offset)
                    for task_id, offset in item["completion_offsets_minutes"]
                ),
                item["expected_energy"],
                item["energy_std"],
                item["reliable_energy"],
                item["admitted"],
            )
            for item in data["candidates"]
        ),
        tuple((task_id, value) for task_id, value in data["miss_penalty"]),
    )


def _point_from_dict(data) -> Point3D:
    return Point3D(data["x"], data["y"], data["z"])


def _polyline_from_dict(data) -> Polyline3D:
    return Polyline3D(
        tuple(_point_from_dict(point) for point in data["points"])
    )


__all__ = [
    "CandidateStop",
    "IrregularDualNetworkScenario",
    "PhysicalServiceCandidate",
    "RoadEdge",
    "RoadNetwork",
    "RoadNode",
    "ScenarioSpec",
    "UavSpec",
    "VehicleSpec",
    "canonical_json",
    "scenario_from_json",
]
