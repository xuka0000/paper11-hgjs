from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from random import Random

from Env.IrregularDualNetworkFactory import generate_topology_scenario
from Env.State import RollingState
from Mask.Road import RoadArc
from Mask.ServiceTuple import (
    SearchArc,
    SearchState,
    SearchTuple,
    ServiceAdmissionChecks,
)
from Mask.Scenario import IrregularDualNetworkScenario, ScenarioSpec
from Mask.TimeNetwork import arrival_layer
from Mask.TupleFactory import (
    EnergyMoments,
    build_energy_certificate,
    build_service_resource_keys,
)
from Mask.UAV import UavState
from Mask.Vehicle import VehicleState
from Mask.Weather import WeatherSlot


TOPOLOGY_FAMILIES = ("corridor", "branch", "ring", "multi_corridor")


@dataclass(frozen=True)
class DifficultyProfile:
    name: str
    task_count: int
    vehicle_count: int
    uavs_per_vehicle: int
    min_corridors: int
    max_corridors: int
    resource_pressure: str

    def __post_init__(self):
        if self.vehicle_count <= 0:
            raise ValueError("vehicle_count must be positive")
        if self.task_count < 2:
            raise ValueError("task_count must be at least two")
        if self.uavs_per_vehicle <= 0:
            raise ValueError("uavs_per_vehicle must be positive")
        if not 1 <= self.min_corridors <= self.max_corridors:
            raise ValueError("invalid corridor range")
        if self.resource_pressure not in {"low", "medium", "high"}:
            raise ValueError("unknown resource pressure")


DIFFICULTY_PROFILES = {
    "easy": DifficultyProfile("easy", 8, 2, 2, 1, 2, "low"),
    "medium": DifficultyProfile("medium", 12, 3, 2, 2, 3, "medium"),
    "hard": DifficultyProfile("hard", 16, 4, 2, 3, 4, "high"),
}


def _canonical_topology_hash(nodes, edges):
    incoming = {node: [] for node in nodes}
    outgoing = {node: [] for node in nodes}
    for tail, head in edges:
        outgoing[tail].append(head)
        incoming[head].append(tail)
    labels = {
        node: f"{len(incoming[node])}:{len(outgoing[node])}" for node in nodes
    }
    for _ in range(3):
        updated = {}
        for node in nodes:
            payload = (
                labels[node],
                tuple(sorted(labels[item] for item in incoming[node])),
                tuple(sorted(labels[item] for item in outgoing[node])),
            )
            updated[node] = sha256(repr(payload).encode("utf-8")).hexdigest()[:16]
        labels = updated
    signature = (
        len(nodes),
        len(edges),
        tuple(sorted(labels.values())),
        tuple(sorted((labels[tail], labels[head]) for tail, head in edges)),
    )
    return sha256(repr(signature).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TopologyTemplate:
    family: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    vehicle_corridors: tuple[tuple[str, ...], ...]

    @property
    def topology_hash(self):
        return _canonical_topology_hash(self.nodes, self.edges)

    def renamed(self, mapping):
        return TopologyTemplate(
            self.family,
            tuple(mapping[node] for node in self.nodes),
            tuple((mapping[tail], mapping[head]) for tail, head in self.edges),
            tuple(
                tuple(mapping[node] for node in corridor)
                for corridor in self.vehicle_corridors
            ),
        )


def _path_edges(path):
    return tuple(zip(path[:-1], path[1:]))


def generate_topology(family, seed, vehicle_count, corridor_count):
    if family not in TOPOLOGY_FAMILIES:
        raise ValueError(f"unknown topology family: {family!r}")
    if vehicle_count <= 0:
        raise ValueError("vehicle_count must be positive")
    if corridor_count <= 0:
        raise ValueError("corridor_count must be positive")
    variant = 1 + int(seed) % 20
    source, sink = "source", "sink"
    corridors = []
    edges = set()
    if family == "corridor":
        path = (source,) + tuple(f"c{i}" for i in range(variant + 2)) + (sink,)
        corridors.append(path)
        edges.update(_path_edges(path))
    elif family == "branch":
        trunk = (source,) + tuple(f"b{i}" for i in range(variant))
        branch_root = trunk[-1]
        edges.update(_path_edges(trunk))
        for index in range(max(2, corridor_count)):
            path = trunk + (f"leaf{index}", sink)
            corridors.append(path)
            edges.add((branch_root, f"leaf{index}"))
            edges.add((f"leaf{index}", sink))
    elif family == "ring":
        ring = tuple(f"r{i}" for i in range(variant + 3))
        edges.add((source, ring[0]))
        edges.update(_path_edges(ring))
        edges.add((ring[-1], ring[0]))
        exits = sorted({len(ring) // 2, len(ring) - 1})
        for index in exits:
            edges.add((ring[index], sink))
            corridors.append((source,) + ring[: index + 1] + (sink,))
    else:
        for corridor_index in range(max(2, corridor_count)):
            middle = tuple(
                f"m{corridor_index}_{step}" for step in range(variant + 1)
            )
            path = (source,) + middle + (sink,)
            corridors.append(path)
            edges.update(_path_edges(path))
        if len(corridors) >= 2:
            edges.add((corridors[0][1], corridors[1][-2]))
    if not corridors:
        corridors.append((source, sink))
    nodes = tuple(sorted({node for edge in edges for node in edge}))
    assigned = tuple(corridors[index % len(corridors)] for index in range(vehicle_count))
    return TopologyTemplate(
        family,
        nodes,
        tuple(sorted(edges)),
        assigned,
    )


@dataclass(frozen=True)
class FormalV3Spec:
    split: str
    difficulty: str
    topology_family: str
    seed: int
    topology_slot: int
    energy_alpha: float = 0.9
    road_beta_w: float = 0.02
    road_beta_eta: float = 0.4
    preparation_layers: int = 1
    recovery_service_layers: int = 1
    turnaround_layers: int = 1
    max_duty_time: float = 128.0

    def __post_init__(self):
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("unknown split")
        if self.difficulty not in DIFFICULTY_PROFILES:
            raise ValueError("unknown difficulty")
        if self.topology_family not in TOPOLOGY_FAMILIES:
            raise ValueError("unknown topology family")
        if self.topology_slot <= 0:
            raise ValueError("topology_slot must be positive")
        if not 0.5 < self.energy_alpha < 1.0:
            raise ValueError("energy_alpha must lie in (0.5, 1)")
        if self.road_beta_w < 0 or self.road_beta_eta < 0:
            raise ValueError("road-weather coefficients must be nonnegative")
        if min(
            self.preparation_layers,
            self.recovery_service_layers,
            self.turnaround_layers,
        ) < 0:
            raise ValueError("service resource durations must be nonnegative")
        if self.max_duty_time <= 0:
            raise ValueError("max_duty_time must be positive")


@dataclass(frozen=True)
class FormalV3Instance:
    instance_id: str
    split: str
    seed: int
    difficulty: str
    topology_family: str
    topology_hash: str
    task_ids: tuple[int, ...]
    miss_penalty: dict[int, float]
    operation_capacity: dict[str, int]
    search_state: SearchState
    rolling_state: RollingState
    physical_scenario: IrregularDualNetworkScenario
    energy_alpha: float = 0.9
    road_beta_w: float = 0.02
    road_beta_eta: float = 0.4
    preparation_layers: int = 1
    recovery_service_layers: int = 1
    turnaround_layers: int = 1
    max_duty_time: float = 128.0

    @property
    def scenario_hash(self):
        return self.physical_scenario.scenario_hash

    @property
    def candidate_pool_hash(self):
        return self.physical_scenario.candidate_pool_hash

    @property
    def instance_hash(self):
        tuple_payloads = []
        for item in self.search_state.tuples:
            item_payload = asdict(item)
            item_payload.pop("is_air_residual", None)
            tuple_payloads.append(item_payload)
        payload = {
            "instance_id": self.instance_id,
            "split": self.split,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "topology_family": self.topology_family,
            "topology_hash": self.topology_hash,
            "scenario_hash": self.scenario_hash,
            "candidate_pool_hash": self.candidate_pool_hash,
            "task_ids": self.task_ids,
            "miss_penalty": self.miss_penalty,
            "operation_capacity": self.operation_capacity,
            "energy_alpha": self.energy_alpha,
            "road_beta_w": self.road_beta_w,
            "road_beta_eta": self.road_beta_eta,
            "preparation_layers": self.preparation_layers,
            "recovery_service_layers": self.recovery_service_layers,
            "turnaround_layers": self.turnaround_layers,
            "max_duty_time": self.max_duty_time,
            "arcs": [asdict(arc) for arc in self.search_state.arcs],
            "tuples": tuple_payloads,
            "selected_arc_ids": sorted(self.search_state.selected_arc_ids),
            "selected_tuple_ids": sorted(self.search_state.selected_tuple_ids),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class V3ValidationReport:
    ok: bool
    reason: str | None = None


def build_vehicle_route_arcs(
    vehicle_id,
    corridor,
    weather_by_layer,
    horizon,
    delta_t,
    beta_w,
    beta_eta,
    physical_edge_by_pair=None,
):
    departure = min(weather_by_layer)
    route = []
    for index, (tail, head) in enumerate(zip(corridor[:-1], corridor[1:])):
        physical_edge = (
            None
            if physical_edge_by_pair is None
            else physical_edge_by_pair[(tail, head)]
        )
        if physical_edge is None:
            road = RoadArc(
                index,
                index + 1,
                9.0 + 0.2 * ((vehicle_id + index) % 3),
                10.0,
                1.0,
                True,
            )
            arc_id = f"v{vehicle_id}:e{index}:{tail}:{head}"
            secondary_cost = 0.0
        else:
            road = RoadArc(
                index,
                index + 1,
                physical_edge.geometry.length_3d,
                physical_edge.safe_speed_kmph / 60.0,
                1.0,
                physical_edge.available,
            )
            arc_id = f"v{vehicle_id}:{physical_edge.edge_id}:{tail}:{head}"
            secondary_cost = physical_edge.geometry.length_3d
        arrival = arrival_layer(
            road,
            departure,
            weather_by_layer,
            horizon,
            delta_t,
            beta_w,
            beta_eta,
        )
        if arrival is None:
            raise RuntimeError(
                f"vehicle route exceeds horizon: vehicle={vehicle_id} edge={index}"
            )
        route.append(
            SearchArc(
                arc_id,
                vehicle_id,
                tail,
                head,
                departure,
                arrival,
                "road",
                secondary_cost,
                duty_time=(arrival - departure) * delta_t,
            )
        )
        departure = arrival
    return tuple(route)


def _tuple_from_physical_candidate(
    candidate,
    scenario,
    corridor,
    route_arcs,
    spec,
):
    try:
        launch_index = corridor.index(candidate.launch_stop_id)
        recovery_index = corridor.index(candidate.recovery_stop_id)
    except ValueError:
        return None
    if recovery_index <= launch_index:
        return None
    launch_layer = (
        1 if launch_index == 0 else route_arcs[launch_index - 1].arrival_layer
    )
    recovery_layer = route_arcs[recovery_index - 1].arrival_layer
    completion_layers = tuple(
        (
            int(task_id),
            launch_layer + max(1, int(ceil(offset - 1e-12))),
        )
        for task_id, offset in candidate.completion_offsets_minutes
    )
    if any(layer > recovery_layer for _, layer in completion_layers):
        return None
    required = tuple(
        arc.arc_id for arc in route_arcs[launch_index:recovery_index]
    )
    resources = build_service_resource_keys(
        candidate.vehicle_id,
        candidate.uav_id,
        candidate.launch_stop_id,
        candidate.recovery_stop_id,
        launch_layer,
        recovery_layer,
        spec.preparation_layers,
        spec.recovery_service_layers,
        spec.turnaround_layers,
    )
    uav_by_id = {item.uav_id: item for item in scenario.uavs}
    certificate = build_energy_certificate(
        EnergyMoments(candidate.expected_energy, candidate.energy_std),
        spec.energy_alpha,
        uav_by_id[candidate.uav_id].battery_capacity,
        uav_by_id[candidate.uav_id].reserve_ratio,
    )
    tower_by_id = {item.tower_id: item for item in scenario.power.towers}
    risk_exposure = round(
        sum(
            tower_by_id[str(task_id)].risk * (1.0 + 0.05 * layer)
            for task_id, layer in completion_layers
        ),
        6,
    )
    checks = ServiceAdmissionChecks(
        task_sequence=bool(candidate.task_ids),
        carrier_assignment=True,
        sensing=True,
        path_3d=True,
        range_limit=candidate.admitted,
        terrain_clearance=True,
        airspace=True,
        timing=recovery_layer >= launch_layer,
    )
    return SearchTuple.create(
        candidate.vehicle_id,
        candidate.uav_id,
        tuple(int(task_id) for task_id in candidate.task_ids),
        candidate.launch_stop_id,
        candidate.recovery_stop_id,
        launch_layer,
        recovery_layer,
        required_arc_ids=required,
        interval_keys=resources.interval_keys,
        operation_keys=resources.operation_keys,
        completion_layers=completion_layers,
        secondary_cost=candidate.duration_minutes,
        admitted=candidate.admitted and certificate.admitted,
        risk_exposure=risk_exposure,
        expected_energy=candidate.expected_energy,
        energy_std=candidate.energy_std,
        reliable_energy=candidate.reliable_energy,
        battery_capacity=certificate.full_capacity,
        reserve_ratio=certificate.reserve_ratio,
        admission_certificate=certificate.certificate_id,
        admission_checks=checks,
        physical_candidate_id=candidate.candidate_id,
    )


def _formal_v3_weather(seed, horizon):
    random = Random(seed ^ 0x5A17)
    return {
        layer: WeatherSlot(
            layer,
            1,
            round(0.1 + 0.02 * (layer % 4), 6),
            round(1.5 + 1.5 * random.random(), 6),
            round(360.0 * random.random(), 6),
            round(15.0 + 10.0 * random.random(), 6),
            round(0.04 + 0.08 * random.random(), 6),
        )
        for layer in range(1, horizon + 1)
    }


def _tuple_for_tasks(
    vehicle_id,
    uav_id,
    task_ids,
    corridor,
    route_arcs,
    random,
    energy_alpha,
    preparation_layers,
    recovery_service_layers,
    turnaround_layers,
):
    internal_count = len(corridor) - 2
    launch_offset = (sum(task_ids) + vehicle_id + uav_id) % max(1, internal_count)
    launch_index = 1 + min(launch_offset, max(0, internal_count - 1))
    recovery_index = min(
        len(corridor) - 2,
        launch_index + 1 + (sum(task_ids) % 2),
    )
    if recovery_index <= launch_index:
        launch_index = max(1, recovery_index - 1)
    launch_layer = route_arcs[launch_index - 1].arrival_layer
    recovery_layer = route_arcs[recovery_index - 1].arrival_layer
    required = tuple(
        arc.arc_id for arc in route_arcs[launch_index:recovery_index]
    )
    completion_layers = tuple(
        (task_id, min(recovery_layer, launch_layer + index + 1))
        for index, task_id in enumerate(task_ids)
    )
    duration = max(1, recovery_layer - launch_layer)
    energy_mean = round(
        4.0
        + 1.25 * len(task_ids)
        + 0.35 * duration
        + 0.1 * (uav_id % 3)
        + 0.25 * random.random(),
        6,
    )
    energy_std = round(
        0.06 * energy_mean + 0.02 * ((sum(task_ids) + uav_id) % 3),
        6,
    )
    battery_capacity = float(32 + 2 * (uav_id % 3))
    reserve_ratio = 0.2
    certificate = build_energy_certificate(
        EnergyMoments(energy_mean, energy_std),
        alpha=energy_alpha,
        full_capacity=battery_capacity,
        reserve_ratio=reserve_ratio,
    )
    risk_exposure = round(
        sum(
            (1.0 + 0.08 * task_id) * (1.0 + 0.05 * layer)
            for task_id, layer in completion_layers
        ),
        6,
    )
    admission_checks = ServiceAdmissionChecks(
        task_sequence=bool(task_ids),
        carrier_assignment=vehicle_id > 0 and uav_id > 0,
        sensing=bool(task_ids),
        path_3d=True,
        range_limit=True,
        terrain_clearance=True,
        airspace=True,
        timing=recovery_layer >= launch_layer,
    )
    resources = build_service_resource_keys(
        vehicle_id,
        uav_id,
        corridor[launch_index],
        corridor[recovery_index],
        launch_layer,
        recovery_layer,
        preparation_layers,
        recovery_service_layers,
        turnaround_layers,
    )
    return SearchTuple.create(
        vehicle_id,
        uav_id,
        task_ids,
        corridor[launch_index],
        corridor[recovery_index],
        launch_layer,
        recovery_layer,
        required_arc_ids=required,
        interval_keys=resources.interval_keys,
        operation_keys=resources.operation_keys,
        completion_layers=completion_layers,
        secondary_cost=0.0,
        admitted=certificate.admitted and admission_checks.all_passed,
        risk_exposure=risk_exposure,
        expected_energy=certificate.mean,
        energy_std=certificate.std,
        reliable_energy=certificate.reliable_bound,
        battery_capacity=certificate.full_capacity,
        reserve_ratio=certificate.reserve_ratio,
        admission_certificate=certificate.certificate_id,
        admission_checks=admission_checks,
    )


def generate_profiled_v3_instance(
    spec,
    profile,
    *,
    difficulty_label,
    instance_namespace,
    service_choices_per_carrier_task=1,
    service_choice_reference_count=None,
    require_all_service_choices_admitted=False,
    uav_battery_capacity_scale=1.0,
    service_choice_speed_multipliers=None,
    service_choice_energy_multipliers=None,
):
    if not difficulty_label:
        raise ValueError("difficulty label must not be empty")
    if not instance_namespace:
        raise ValueError("instance namespace must not be empty")
    if not 1 <= service_choices_per_carrier_task <= profile.uavs_per_vehicle:
        raise ValueError(
            "service choices must lie between one and the UAVs per vehicle"
        )
    random = Random(spec.seed)
    corridor_count = random.randint(profile.min_corridors, profile.max_corridors)
    topology_seed = spec.topology_slot
    topology = generate_topology(
        spec.topology_family,
        topology_seed,
        profile.vehicle_count,
        corridor_count,
    )
    capacity_value = 2 if profile.resource_pressure == "low" else 1
    uav_count = profile.vehicle_count * profile.uavs_per_vehicle
    physical_scenario = generate_topology_scenario(
        ScenarioSpec(
            spec.split,
            difficulty_label,
            spec.topology_family,
            spec.seed,
            profile.task_count,
            len(topology.nodes),
            len(topology.nodes),
            profile.vehicle_count,
            uav_count,
            128.0,
        ),
        topology.nodes,
        topology.edges,
        topology.vehicle_corridors,
        energy_alpha=spec.energy_alpha,
        operation_capacity=capacity_value,
        service_choices_per_carrier_task=service_choices_per_carrier_task,
        service_choice_reference_count=service_choice_reference_count,
        require_all_service_choices_admitted=(
            require_all_service_choices_admitted
        ),
        uav_battery_capacity_scale=uav_battery_capacity_scale,
        service_choice_speed_multipliers=(
            service_choice_speed_multipliers
        ),
        service_choice_energy_multipliers=(
            service_choice_energy_multipliers
        ),
    )
    arcs = []
    route_by_vehicle = {}
    corridor_by_vehicle = {}
    horizon = 128
    delta_t = 1.0
    weather_by_layer = _formal_v3_weather(spec.seed, horizon)
    physical_edge_by_pair = {
        (edge.tail, edge.head): edge for edge in physical_scenario.road.edges
    }
    for vehicle_id, corridor in enumerate(topology.vehicle_corridors, start=1):
        route = build_vehicle_route_arcs(
            vehicle_id,
            corridor,
            weather_by_layer,
            horizon,
            delta_t,
            spec.road_beta_w,
            spec.road_beta_eta,
            physical_edge_by_pair,
        )
        arcs.extend(route)
        route_by_vehicle[vehicle_id] = route
        corridor_by_vehicle[vehicle_id] = corridor

    task_ids = tuple(int(tower.tower_id) for tower in physical_scenario.power.towers)
    tuples = [
        item
        for candidate in physical_scenario.candidates
        for item in (
            _tuple_from_physical_candidate(
                candidate,
                physical_scenario,
                corridor_by_vehicle[candidate.vehicle_id],
                route_by_vehicle[candidate.vehicle_id],
                spec,
            ),
        )
        if item is not None and item.admitted
    ]
    tuples = tuple(sorted({item.tuple_id: item for item in tuples}.values(), key=lambda item: item.tuple_id))
    if not tuples:
        raise RuntimeError("formal-v3 generator produced no tuples")
    seed_tuple = min(
        (item for item in tuples if len(item.task_ids) == 1),
        key=lambda item: (item.task_ids, item.secondary_cost, item.tuple_id),
    )
    selected_arcs = frozenset(
        arc.arc_id for arc in route_by_vehicle[seed_tuple.vehicle_id]
    )
    search_state = SearchState(
        tuple(sorted(arcs)),
        tuples,
        selected_arcs,
        frozenset({seed_tuple.tuple_id}),
    )
    operation_keys = sorted(
        {key for item in tuples for key in item.operation_keys}
    )
    operation_capacity = {key: capacity_value for key in operation_keys}
    vehicles = {
        vehicle.vehicle_id: VehicleState(
            vehicle.vehicle_id,
            vehicle.depot_stop_id,
            1,
            remaining_duty_time=spec.max_duty_time,
        )
        for vehicle in physical_scenario.vehicles
    }
    uavs = {
        uav.uav_id: UavState(
            uav.uav_id,
            uav.owner_vehicle_id,
            1,
            remaining_energy=uav.battery_capacity,
        )
        for uav in physical_scenario.uavs
    }
    rolling_state = RollingState(
        epoch=0,
        current_layer=1,
        done_tasks=frozenset(),
        remaining_tasks=frozenset(task_ids),
        vehicles=vehicles,
        uavs=uavs,
    )
    miss_penalty = {
        int(task_id): value for task_id, value in physical_scenario.miss_penalty
    }
    if instance_namespace == "v3" and service_choices_per_carrier_task == 1:
        instance_id = (
            f"{spec.split}:v3:{difficulty_label}:{spec.topology_family}:"
            f"{spec.seed}:{spec.topology_slot}"
        )
    else:
        instance_id = (
            f"{spec.split}:{instance_namespace}:{difficulty_label}:"
            f"m{service_choices_per_carrier_task}:{spec.topology_family}:"
            f"{spec.seed}:{spec.topology_slot}"
        )
    return FormalV3Instance(
        instance_id,
        spec.split,
        spec.seed,
        difficulty_label,
        spec.topology_family,
        topology.topology_hash,
        task_ids,
        miss_penalty,
        operation_capacity,
        search_state,
        rolling_state,
        physical_scenario,
        spec.energy_alpha,
        spec.road_beta_w,
        spec.road_beta_eta,
        spec.preparation_layers,
        spec.recovery_service_layers,
        spec.turnaround_layers,
        spec.max_duty_time,
    )


def generate_formal_v3_instance(spec):
    return generate_profiled_v3_instance(
        spec,
        DIFFICULTY_PROFILES[spec.difficulty],
        difficulty_label=spec.difficulty,
        instance_namespace="v3",
        service_choices_per_carrier_task=1,
        service_choice_reference_count=1,
        require_all_service_choices_admitted=False,
        uav_battery_capacity_scale=1.0,
    )


def build_formal_v3_specs(base_seed=20260720):
    split_plan = {
        "train": (12, tuple(range(1, 13)), 0),
        "validation": (4, (13, 14, 15, 16), 100_000),
        "test": (4, (17, 18, 19, 40), 200_000),
    }
    specs = []
    for split, (per_difficulty, topology_slots, seed_offset) in split_plan.items():
        for difficulty_index, difficulty in enumerate(DIFFICULTY_PROFILES):
            for local_index in range(per_difficulty):
                specs.append(
                    FormalV3Spec(
                        split,
                        difficulty,
                        TOPOLOGY_FAMILIES[local_index % len(TOPOLOGY_FAMILIES)],
                        base_seed
                        + seed_offset
                        + 1000 * difficulty_index
                        + local_index,
                        topology_slots[local_index % len(topology_slots)],
                    )
                )
    return tuple(specs)


def build_chdgrl_development_specs(base_seed=20260722):
    return tuple(
        spec
        for spec in build_formal_v3_specs(base_seed)
        if spec.split in {"train", "validation"}
    )


def build_chdgrl_test_specs(base_seed=20280722):
    return tuple(
        spec
        for spec in build_formal_v3_specs(base_seed)
        if spec.split == "test"
    )


def build_formal_v3_pilot_specs(base_seed=20260720):
    plan = (
        ("train", "easy", "corridor", 0, 1),
        ("train", "medium", "branch", 1, 2),
        ("train", "hard", "ring", 2, 3),
        ("validation", "easy", "branch", 100_000, 13),
        ("validation", "medium", "ring", 100_001, 14),
        ("validation", "hard", "multi_corridor", 100_002, 15),
        ("test", "easy", "corridor", 200_000, 17),
        ("test", "medium", "branch", 200_001, 18),
        ("test", "hard", "ring", 200_002, 19),
    )
    return tuple(
        FormalV3Spec(split, difficulty, family, base_seed + offset, slot)
        for split, difficulty, family, offset, slot in plan
    )


def validate_search_state(instance):
    state = instance.search_state
    tuple_by_id = {item.tuple_id: item for item in state.tuples}
    arc_by_id = {arc.arc_id: arc for arc in state.arcs}
    if state.selected_tuple_ids - set(tuple_by_id):
        return V3ValidationReport(False, "unknown selected tuple")
    if state.selected_arc_ids - set(arc_by_id):
        return V3ValidationReport(False, "unknown selected arc")
    selected = [tuple_by_id[item] for item in state.selected_tuple_ids]
    task_counts = Counter(task for item in selected for task in item.task_ids)
    if any(count > 1 for count in task_counts.values()):
        return V3ValidationReport(False, "duplicate selected task")
    interval_counts = Counter(key for item in selected for key in item.interval_keys)
    if any(count > 1 for count in interval_counts.values()):
        return V3ValidationReport(False, "uav interval conflict")
    operation_counts = Counter(key for item in selected for key in item.operation_keys)
    if any(
        operation_counts[key] > instance.operation_capacity.get(key, 0)
        for key in operation_counts
    ):
        return V3ValidationReport(False, "operation capacity conflict")
    for item in selected:
        if not set(item.required_arc_ids) <= state.selected_arc_ids:
            return V3ValidationReport(False, "required arc missing")
        if item.vehicle_id not in instance.rolling_state.vehicles:
            return V3ValidationReport(False, "unknown vehicle")
        if item.uav_id not in instance.rolling_state.uavs:
            return V3ValidationReport(False, "unknown uav")
    return V3ValidationReport(True)


@dataclass(frozen=True)
class FormalV3ManifestRow:
    instance_id: str
    split: str
    difficulty: str
    topology_family: str
    topology_hash: str
    seed: int
    topology_slot: int
    task_count: int
    vehicle_count: int
    uav_count: int
    arc_count: int
    tuple_count: int
    interval_conflict_count: int
    operation_conflict_count: int
    instance_hash: str
    scenario_hash: str = ""
    candidate_pool_hash: str = ""


def build_manifest_row(instance, spec):
    interval_counts = Counter(
        key for item in instance.search_state.tuples for key in item.interval_keys
    )
    operation_counts = Counter(
        key for item in instance.search_state.tuples for key in item.operation_keys
    )
    return FormalV3ManifestRow(
        instance.instance_id,
        instance.split,
        instance.difficulty,
        instance.topology_family,
        instance.topology_hash,
        instance.seed,
        spec.topology_slot,
        len(instance.task_ids),
        len(instance.rolling_state.vehicles),
        len(instance.rolling_state.uavs),
        len(instance.search_state.arcs),
        len(instance.search_state.tuples),
        sum(count > 1 for count in interval_counts.values()),
        sum(count > 1 for count in operation_counts.values()),
        instance.instance_hash,
        instance.scenario_hash,
        instance.candidate_pool_hash,
    )


def write_manifest(path, rows):
    path = Path(path)
    if "formal_v2" in str(path).lower():
        raise ValueError("formal_v2 paths are read-only")
    ordered = tuple(sorted(rows, key=lambda row: row.instance_id))
    if len({row.instance_hash for row in ordered}) != len(ordered):
        raise ValueError("duplicate instance hash")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(asdict(row), sort_keys=True) + "\n" for row in ordered
    )
    path.write_text(text, encoding="utf-8")


def read_manifest(path):
    path = Path(path)
    if not path.exists():
        return ()
    return tuple(
        FormalV3ManifestRow(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
