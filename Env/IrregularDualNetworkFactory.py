from hashlib import sha256
from math import cos, hypot, pi, sin, sqrt
from random import Random

from Mask.Geometry import Point3D, Polyline3D
from Mask.PowerNetwork import PowerLine, PowerNetwork, Tower
from Mask.Scenario import (
    CandidateStop,
    IrregularDualNetworkScenario,
    PhysicalServiceCandidate,
    RoadEdge,
    RoadNetwork,
    RoadNode,
    ScenarioSpec,
    UavSpec,
    VehicleSpec,
    canonical_json,
)
from Mask.TupleFactory import EnergyMoments, build_energy_certificate


SIZE_PROFILES = {
    "S": {
        "task_count": 8,
        "road_node_count": 10,
        "stop_count": 6,
        "vehicle_count": 1,
        "uav_count": 2,
    },
    "M": {
        "task_count": 12,
        "road_node_count": 16,
        "stop_count": 9,
        "vehicle_count": 2,
        "uav_count": 4,
    },
    "L": {
        "task_count": 16,
        "road_node_count": 24,
        "stop_count": 12,
        "vehicle_count": 3,
        "uav_count": 6,
    },
}
GEOMETRY_FAMILIES = ("crossing", "offset", "mismatched_branch")
FORMAL_SEEDS = tuple(range(20260801, 20260806))
PILOT_SEEDS = (20260719, 20260720)


def build_formal_specs() -> tuple[ScenarioSpec, ...]:
    return _build_specs("formal", FORMAL_SEEDS)


def build_pilot_specs() -> tuple[ScenarioSpec, ...]:
    return _build_specs("pilot", PILOT_SEEDS)


def _build_specs(split: str, seeds: tuple[int, ...]) -> tuple[ScenarioSpec, ...]:
    specs = []
    for size, profile in SIZE_PROFILES.items():
        for family in GEOMETRY_FAMILIES:
            for seed in seeds:
                specs.append(
                    ScenarioSpec(
                        split,
                        size,
                        family,
                        seed,
                        profile["task_count"],
                        profile["road_node_count"],
                        profile["stop_count"],
                        profile["vehicle_count"],
                        profile["uav_count"],
                        180.0,
                    )
                )
    return tuple(specs)


def generate_scenario(spec: ScenarioSpec) -> IrregularDualNetworkScenario:
    _validate_profile(spec)
    road_random = Random(spec.road_geometry_seed)
    power_random = Random(spec.power_geometry_seed)
    candidate_random = Random(spec.candidate_seed)
    road = _build_road_network(spec, road_random)
    power = _build_power_network(spec, power_random)
    stops = _build_stops(road, spec.stop_count)
    vehicles = tuple(
        VehicleSpec(vehicle_id, stops[0].stop_id, spec.horizon_minutes)
        for vehicle_id in range(1, spec.vehicle_count + 1)
    )
    uavs = tuple(
        UavSpec(
            uav_id,
            1 + (uav_id - 1) % spec.vehicle_count,
            60.0 + 2.0 * (uav_id % 3),
            180.0,
            0.2,
            ("rgb", "thermal"),
        )
        for uav_id in range(1, spec.uav_count + 1)
    )
    candidates = _build_candidates(
        spec,
        power,
        stops,
        uavs,
        candidate_random,
    )
    miss_penalty = tuple(
        (
            tower.tower_id,
            round(
                5.0
                + 2.0 * tower.risk
                + tower.asset_value
                + tower.topology_importance,
                9,
            ),
        )
        for tower in power.towers
    )
    return IrregularDualNetworkScenario(
        spec,
        road,
        power,
        stops,
        vehicles,
        uavs,
        candidates,
        miss_penalty,
    )


def generate_topology_scenario(
    spec: ScenarioSpec,
    topology_nodes: tuple[str, ...],
    topology_edges: tuple[tuple[str, str], ...],
    vehicle_corridors: tuple[tuple[str, ...], ...],
    *,
    energy_alpha: float,
    operation_capacity: int,
    service_choices_per_carrier_task: int = 1,
    service_choice_reference_count: int | None = None,
    require_all_service_choices_admitted: bool = False,
    uav_battery_capacity_scale: float = 1.0,
    service_choice_speed_multipliers: tuple[float, ...] | None = None,
    service_choice_energy_multipliers: tuple[float, ...] | None = None,
) -> IrregularDualNetworkScenario:
    nodes = tuple(topology_nodes)
    edges = tuple(topology_edges)
    corridors = tuple(tuple(path) for path in vehicle_corridors)
    _validate_topology_contract(spec, nodes, edges, corridors)
    road_random = Random(spec.road_geometry_seed)
    power_random = Random(spec.power_geometry_seed)
    candidate_random = Random(spec.candidate_seed)
    road = _build_topology_road_network(nodes, edges, corridors, road_random)
    stops = tuple(
        CandidateStop(
            node.node_id,
            node.location,
            road_node_id=node.node_id,
        )
        for node in road.nodes
    )
    vehicles = tuple(
        VehicleSpec(
            vehicle_id,
            road.depot_node_id,
            spec.horizon_minutes,
            operation_capacity,
        )
        for vehicle_id in range(1, spec.vehicle_count + 1)
    )
    uavs = tuple(
        UavSpec(
            uav_id,
            1 + (uav_id - 1) % spec.vehicle_count,
            60.0 + 2.0 * (uav_id % 3),
            uav_battery_capacity_scale
            * float(32 + 2 * (uav_id % 3)),
            0.2,
            ("rgb", "thermal"),
        )
        for uav_id in range(1, spec.uav_count + 1)
    )
    if uav_battery_capacity_scale <= 0.0:
        raise ValueError("UAV battery capacity scale must be positive")
    uavs_by_vehicle = {
        vehicle_id: tuple(
            uav for uav in uavs if uav.owner_vehicle_id == vehicle_id
        )
        for vehicle_id in range(1, spec.vehicle_count + 1)
    }
    if service_choices_per_carrier_task <= 0:
        raise ValueError("service choices per carrier-task pair must be positive")
    reference_count = (
        service_choices_per_carrier_task
        if service_choice_reference_count is None
        else service_choice_reference_count
    )
    if reference_count < service_choices_per_carrier_task:
        raise ValueError("service choice reference count is too small")
    if any(
        len(vehicle_uavs) < reference_count
        for vehicle_uavs in uavs_by_vehicle.values()
    ):
        raise ValueError(
            "service choices per carrier-task pair exceed available UAVs"
        )
    profiled_service_choices = (
        service_choice_speed_multipliers is not None
        or service_choice_energy_multipliers is not None
    )
    speed_multipliers = (
        (1.0,) * reference_count
        if service_choice_speed_multipliers is None
        else tuple(float(value) for value in service_choice_speed_multipliers)
    )
    energy_multipliers = (
        (1.0,) * reference_count
        if service_choice_energy_multipliers is None
        else tuple(float(value) for value in service_choice_energy_multipliers)
    )
    if len(speed_multipliers) < reference_count:
        raise ValueError("service choice speed profiles are incomplete")
    if len(energy_multipliers) < reference_count:
        raise ValueError("service choice energy profiles are incomplete")
    if any(value <= 0.0 for value in speed_multipliers + energy_multipliers):
        raise ValueError("service choice profile multipliers must be positive")
    power = _build_topology_power_network(
        spec,
        road,
        corridors,
        power_random,
    )
    candidates = _build_topology_candidates(
        spec,
        road,
        power,
        stops,
        uavs,
        corridors,
        candidate_random,
        energy_alpha,
        service_choices_per_carrier_task,
        reference_count,
        require_all_service_choices_admitted,
        speed_multipliers if profiled_service_choices else None,
        energy_multipliers if profiled_service_choices else None,
    )
    miss_penalty = tuple(
        (
            tower.tower_id,
            round(
                5.0
                + 2.0 * tower.risk
                + tower.asset_value
                + tower.topology_importance,
                9,
            ),
        )
        for tower in power.towers
    )
    return IrregularDualNetworkScenario(
        spec,
        road,
        power,
        stops,
        vehicles,
        uavs,
        candidates,
        miss_penalty,
    )


def _validate_topology_contract(spec, nodes, edges, corridors) -> None:
    node_ids = set(nodes)
    if len(node_ids) != len(nodes):
        raise ValueError("topology contains duplicate nodes")
    if {"source", "sink"} - node_ids:
        raise ValueError("topology must contain source and sink")
    if len(edges) != len(set(edges)):
        raise ValueError("topology contains duplicate edges")
    if any(tail not in node_ids or head not in node_ids for tail, head in edges):
        raise ValueError("topology edge references unknown node")
    if len(corridors) != spec.vehicle_count:
        raise ValueError("vehicle corridor count differs from scenario spec")
    edge_set = set(edges)
    for corridor in corridors:
        if corridor[0] != "source" or corridor[-1] != "sink":
            raise ValueError("vehicle corridor must connect source to sink")
        if any(edge not in edge_set for edge in zip(corridor[:-1], corridor[1:])):
            raise ValueError("vehicle corridor references unknown edge")
    expected = (spec.road_node_count, spec.stop_count)
    if expected != (len(nodes), len(nodes)):
        raise ValueError("topology node and stop counts differ from scenario spec")


def _build_topology_road_network(nodes, edges, corridors, random) -> RoadNetwork:
    max_steps = max(len(path) for path in corridors) - 1
    positions = {"source": _point(0.0, 0.0, 1.5)}
    lane_count = len(corridors)
    for corridor_index, corridor in enumerate(corridors):
        lane_y = 5.0 * (corridor_index - (lane_count - 1) / 2.0)
        for step, node_id in enumerate(corridor):
            if node_id == "source":
                continue
            if node_id == "sink":
                proposed = _point(1.5 * max_steps, 0.0, 1.5)
            else:
                proposed = _point(
                    1.5 * step,
                    lane_y + 0.4 * sin(0.7 * step + corridor_index),
                    1.5 + 0.2 * sin(0.5 * step),
                )
            positions.setdefault(node_id, proposed)
    for index, node_id in enumerate(sorted(set(nodes) - set(positions))):
        positions[node_id] = _point(
            1.5 * (index + 1),
            6.0 + 2.0 * index,
            1.5,
        )
    road_nodes = tuple(RoadNode(node_id, positions[node_id]) for node_id in nodes)
    node_by_id = {node.node_id: node for node in road_nodes}
    road_edges = tuple(
        _road_edge(index, node_by_id[tail], node_by_id[head], random)
        for index, (tail, head) in enumerate(sorted(edges))
    )
    return RoadNetwork(road_nodes, road_edges, "source")


def _build_topology_power_network(
    spec: ScenarioSpec,
    road: RoadNetwork,
    corridors: tuple[tuple[str, ...], ...],
    random: Random,
) -> PowerNetwork:
    node_by_id = {node.node_id: node for node in road.nodes}
    towers = []
    for index in range(spec.task_count):
        corridor = corridors[index % len(corridors)]
        internal = corridor[1:-1] or corridor[:1]
        anchor = node_by_id[internal[index % len(internal)]].location
        side = -1.0 if index % 2 else 1.0
        location = _point(
            anchor.x + random.uniform(-0.35, 0.35),
            anchor.y + side * random.uniform(0.8, 1.4),
            anchor.z + random.uniform(0.08, 0.18),
        )
        towers.append(
            Tower(
                str(index + 1),
                location,
                round(0.8 + 1.2 * random.random(), 9),
                round(1.0 + 2.0 * random.random(), 9),
                round(0.5 + 1.5 * random.random(), 9),
                4.0 + float(index % 3),
            )
        )
    if spec.geometry_family == "multi_corridor" and len(towers) >= 4:
        indexes = (
            tuple(range(0, len(towers), 2)),
            tuple(range(1, len(towers), 2)),
        )
    elif spec.geometry_family == "branch" and len(towers) >= 5:
        root = len(towers) // 2
        indexes = (
            tuple(range(root + 1)),
            (root, *tuple(range(root + 1, len(towers), 2))),
            (root, *tuple(range(root + 2, len(towers), 2))),
        )
    else:
        indexes = (tuple(range(len(towers))),)
    lines = tuple(
        PowerLine(
            f"l{line_index:02d}",
            tuple(towers[index].tower_id for index in line),
            Polyline3D(tuple(towers[index].location for index in line)),
        )
        for line_index, line in enumerate(indexes)
        if len(line) >= 2
    )
    return PowerNetwork(tuple(towers), lines)


def _build_topology_candidates(
    spec,
    road,
    power,
    stops,
    uavs,
    corridors,
    random,
    energy_alpha,
    service_choices_per_carrier_task=1,
    service_choice_reference_count=None,
    require_all_service_choices_admitted=False,
    service_choice_speed_multipliers=None,
    service_choice_energy_multipliers=None,
):
    stop_by_id = {stop.stop_id: stop for stop in stops}
    edge_by_pair = {(edge.tail, edge.head): edge for edge in road.edges}
    uavs_by_vehicle = {
        vehicle_id: tuple(
            uav for uav in uavs if uav.owner_vehicle_id == vehicle_id
        )
        for vehicle_id in range(1, spec.vehicle_count + 1)
    }
    candidates = []
    for tower_index, tower in enumerate(power.towers):
        for vehicle_id, corridor in enumerate(corridors, start=1):
            vehicle_uavs = uavs_by_vehicle[vehicle_id]
            reference_count = (
                service_choices_per_carrier_task
                if service_choice_reference_count is None
                else service_choice_reference_count
            )
            selected_uavs = tuple(
                vehicle_uavs[(tower_index + offset) % len(vehicle_uavs)]
                for offset in range(reference_count)
            )
            base_energy_jitter = random.uniform(0.0, 0.2)
            pair_candidates = []
            if service_choice_speed_multipliers is not None:
                energy_jitters = []
                for choice_index, uav in enumerate(selected_uavs):
                    if choice_index == 0:
                        energy_jitters.append(base_energy_jitter)
                    else:
                        jitter_payload = canonical_json((
                            "service-choice-jitter-v1",
                            tower.tower_id,
                            vehicle_id,
                            uav.uav_id,
                        ))
                        jitter_integer = int(
                            sha256(jitter_payload.encode("utf-8")).hexdigest()[:15],
                            16,
                        )
                        energy_jitters.append(
                            0.2 * jitter_integer / float(16**15 - 1)
                        )
                launch_indexes = sorted(
                    range(0, len(corridor) - 1),
                    key=lambda index: _distance_3d(
                        stop_by_id[corridor[index]].location,
                        tower.location,
                    ),
                )
                selected_geometry = None
                for launch_index in launch_indexes:
                    launch = stop_by_id[corridor[launch_index]]
                    travel_minutes = 0.0
                    for recovery_index in range(
                        launch_index + 1, len(corridor)
                    ):
                        edge = edge_by_pair[
                            (
                                corridor[recovery_index - 1],
                                corridor[recovery_index],
                            )
                        ]
                        travel_minutes += (
                            60.0
                            * edge.geometry.length_3d
                            / edge.safe_speed_kmph
                        )
                        recovery = stop_by_id[corridor[recovery_index]]
                        outbound = _distance_3d(
                            launch.location, tower.location
                        )
                        inbound = _distance_3d(
                            tower.location, recovery.location
                        )
                        rows = []
                        for choice_index, uav in enumerate(selected_uavs):
                            duration = (
                                60.0 * (outbound + inbound)
                                / (
                                    uav.cruise_speed_kmph
                                    * service_choice_speed_multipliers[choice_index]
                                )
                                + tower.service_minutes
                            )
                            base_energy = (
                                0.42 * (outbound + inbound)
                                + 0.16 * tower.service_minutes
                            )
                            expected_energy = round(
                                base_energy
                                * service_choice_energy_multipliers[choice_index]
                                + energy_jitters[choice_index],
                                9,
                            )
                            energy_std = round(0.08 * expected_energy, 9)
                            certificate = build_energy_certificate(
                                EnergyMoments(expected_energy, energy_std),
                                energy_alpha,
                                uav.battery_capacity,
                                uav.reserve_ratio,
                            )
                            rows.append((
                                uav,
                                duration,
                                expected_energy,
                                energy_std,
                                certificate,
                            ))
                        if travel_minutes + 1e-9 < max(
                            row[1] for row in rows
                        ):
                            continue
                        if (
                            require_all_service_choices_admitted
                            and not all(row[4].admitted for row in rows)
                        ):
                            continue
                        selected_geometry = (
                            launch, recovery, outbound, inbound, rows
                        )
                        break
                    if selected_geometry is not None:
                        break
                if selected_geometry is None:
                    if require_all_service_choices_admitted:
                        continue
                else:
                    launch, recovery, outbound, inbound, rows = selected_geometry
                    for (
                        uav,
                        duration,
                        expected_energy,
                        energy_std,
                        certificate,
                    ) in rows[:service_choices_per_carrier_task]:
                        payload = (
                            vehicle_id,
                            uav.uav_id,
                            launch.stop_id,
                            recovery.stop_id,
                            tower.tower_id,
                            round(duration, 9),
                            expected_energy,
                            round(certificate.reliable_bound, 9),
                        )
                        candidate_id = "candidate:" + sha256(
                            canonical_json(payload).encode("utf-8")
                        ).hexdigest()[:20]
                        pair_candidates.append(
                            PhysicalServiceCandidate(
                                candidate_id,
                                vehicle_id,
                                uav.uav_id,
                                launch.stop_id,
                                recovery.stop_id,
                                (tower.tower_id,),
                                round(duration, 9),
                                ((tower.tower_id, round(duration, 9)),),
                                expected_energy,
                                energy_std,
                                round(certificate.reliable_bound, 9),
                                certificate.admitted,
                            )
                        )
                candidates.extend(pair_candidates)
                continue
            for choice_index, uav in enumerate(selected_uavs):
                speed_multiplier = (
                    1.0
                    if service_choice_speed_multipliers is None
                    else service_choice_speed_multipliers[choice_index]
                )
                energy_multiplier = (
                    1.0
                    if service_choice_energy_multipliers is None
                    else service_choice_energy_multipliers[choice_index]
                )
                if choice_index == 0:
                    energy_jitter = base_energy_jitter
                else:
                    jitter_payload = canonical_json((
                        "service-choice-jitter-v1",
                        tower.tower_id,
                        vehicle_id,
                        uav.uav_id,
                    ))
                    jitter_integer = int(
                        sha256(jitter_payload.encode("utf-8")).hexdigest()[:15],
                        16,
                    )
                    energy_jitter = 0.2 * jitter_integer / float(16**15 - 1)
                launch_indexes = sorted(
                    range(0, len(corridor) - 1),
                    key=lambda index: _distance_3d(
                        stop_by_id[corridor[index]].location,
                        tower.location,
                    ),
                )
                selected = None
                for launch_index in launch_indexes:
                    launch = stop_by_id[corridor[launch_index]]
                    travel_minutes = 0.0
                    for recovery_index in range(
                        launch_index + 1, len(corridor)
                    ):
                        edge = edge_by_pair[
                            (
                                corridor[recovery_index - 1],
                                corridor[recovery_index],
                            )
                        ]
                        travel_minutes += (
                            60.0
                            * edge.geometry.length_3d
                            / edge.safe_speed_kmph
                        )
                        recovery = stop_by_id[corridor[recovery_index]]
                        outbound = _distance_3d(
                            launch.location, tower.location
                        )
                        inbound = _distance_3d(
                            tower.location, recovery.location
                        )
                        duration = (
                            60.0
                            * (outbound + inbound)
                            / (uav.cruise_speed_kmph * speed_multiplier)
                            + tower.service_minutes
                        )
                        if travel_minutes + 1e-9 >= duration:
                            base_energy = (
                                0.42 * (outbound + inbound)
                                + 0.16 * tower.service_minutes
                            )
                            expected_energy = round(
                                base_energy * energy_multiplier
                                + energy_jitter,
                                9,
                            )
                            energy_std = round(0.08 * expected_energy, 9)
                            certificate = build_energy_certificate(
                                EnergyMoments(expected_energy, energy_std),
                                energy_alpha,
                                uav.battery_capacity,
                                uav.reserve_ratio,
                            )
                            if (
                                require_all_service_choices_admitted
                                and not certificate.admitted
                            ):
                                continue
                            selected = (
                                launch,
                                recovery,
                                duration,
                                outbound,
                                inbound,
                                expected_energy,
                                energy_std,
                                certificate,
                            )
                            break
                    if selected is not None:
                        break
                if selected is None:
                    if require_all_service_choices_admitted:
                        pair_candidates = []
                        break
                    continue
                (
                    launch,
                    recovery,
                    duration,
                    outbound,
                    inbound,
                    expected_energy,
                    energy_std,
                    certificate,
                ) = selected
                payload = (
                    vehicle_id,
                    uav.uav_id,
                    launch.stop_id,
                    recovery.stop_id,
                    tower.tower_id,
                    round(duration, 9),
                    expected_energy,
                    round(certificate.reliable_bound, 9),
                )
                candidate_id = "candidate:" + sha256(
                    canonical_json(payload).encode("utf-8")
                ).hexdigest()[:20]
                pair_candidates.append(
                    PhysicalServiceCandidate(
                        candidate_id,
                        vehicle_id,
                        uav.uav_id,
                        launch.stop_id,
                        recovery.stop_id,
                        (tower.tower_id,),
                        round(duration, 9),
                        ((tower.tower_id, round(duration, 9)),),
                        expected_energy,
                        energy_std,
                        round(certificate.reliable_bound, 9),
                        certificate.admitted,
                    )
                )
            candidates.extend(
                pair_candidates[:service_choices_per_carrier_task]
            )
    admitted_tasks = {
        task_id
        for candidate in candidates
        if candidate.admitted
        for task_id in candidate.task_ids
    }
    expected_tasks = {tower.tower_id for tower in power.towers}
    if admitted_tasks != expected_tasks:
        raise RuntimeError("topology scenario leaves a task without an admitted candidate")
    return tuple(candidates)


def _validate_profile(spec: ScenarioSpec) -> None:
    if spec.size not in SIZE_PROFILES:
        raise ValueError(f"unknown model-comparison size: {spec.size}")
    if spec.geometry_family not in GEOMETRY_FAMILIES:
        raise ValueError(
            f"unknown model-comparison geometry family: {spec.geometry_family}"
        )
    expected = SIZE_PROFILES[spec.size]
    actual = {
        "task_count": spec.task_count,
        "road_node_count": spec.road_node_count,
        "stop_count": spec.stop_count,
        "vehicle_count": spec.vehicle_count,
        "uav_count": spec.uav_count,
    }
    if actual != expected:
        raise ValueError(
            f"scenario spec differs from frozen size profile: actual={actual} expected={expected}"
        )


def _build_road_network(spec: ScenarioSpec, random: Random) -> RoadNetwork:
    node_count = spec.road_node_count
    main_count = max(4, round(0.7 * node_count))
    nodes = []
    for index in range(main_count):
        x = 11.0 * index + random.uniform(-0.8, 0.8)
        y = 8.0 * sin(0.85 * index) + random.uniform(-2.0, 2.0)
        z = 1.5 + 1.2 * sin(0.45 * index) + random.uniform(-0.3, 0.3)
        nodes.append(RoadNode(f"r{index:03d}", _point(x, y, z)))
    for branch_index in range(node_count - main_count):
        parent_index = 1 + branch_index % (main_count - 2)
        parent = nodes[parent_index].location
        side = -1.0 if branch_index % 2 else 1.0
        angle = side * (0.8 + 0.12 * (branch_index % 3))
        length = 11.0 + 2.5 * (branch_index % 4)
        location = _point(
            parent.x + length * cos(angle),
            parent.y + length * sin(angle),
            parent.z + random.uniform(-1.0, 1.0),
        )
        nodes.append(RoadNode(f"r{len(nodes):03d}", location))
    edges = []
    for index in range(main_count - 1):
        edges.append(
            _road_edge(
                len(edges),
                nodes[index],
                nodes[index + 1],
                random,
            )
        )
    for branch_index, node in enumerate(nodes[main_count:]):
        parent_index = 1 + branch_index % (main_count - 2)
        edges.append(
            _road_edge(
                len(edges),
                nodes[parent_index],
                node,
                random,
            )
        )
    return RoadNetwork(tuple(nodes), tuple(edges), nodes[0].node_id)


def _road_edge(
    edge_index: int,
    tail: RoadNode,
    head: RoadNode,
    random: Random,
) -> RoadEdge:
    dx = head.location.x - tail.location.x
    dy = head.location.y - tail.location.y
    scale = hypot(dx, dy)
    offset = random.choice((-1.0, 1.0)) * random.uniform(0.08, 0.18) * scale
    middle = _point(
        (tail.location.x + head.location.x) / 2.0 - offset * dy / scale,
        (tail.location.y + head.location.y) / 2.0 + offset * dx / scale,
        (tail.location.z + head.location.z) / 2.0 + random.uniform(-0.25, 0.25),
    )
    return RoadEdge(
        f"e{edge_index:03d}",
        tail.node_id,
        head.node_id,
        Polyline3D((tail.location, middle, head.location)),
        "local" if edge_index % 3 else "collector",
        30.0 if edge_index % 3 else 40.0,
        f"w{edge_index % 3}",
        True,
    )


def _build_power_network(spec: ScenarioSpec, random: Random) -> PowerNetwork:
    count = spec.task_count
    span = max(75.0, 7.5 * spec.road_node_count)
    coordinates, line_indexes = _power_geometry(
        spec.geometry_family,
        count,
        span,
        random,
    )
    towers = tuple(
        Tower(
            f"p{index:03d}",
            coordinate,
            round(0.8 + 1.2 * random.random(), 9),
            round(1.0 + 2.0 * random.random(), 9),
            round(0.5 + 1.5 * random.random(), 9),
            4.0 + float(index % 3),
        )
        for index, coordinate in enumerate(coordinates)
    )
    lines = []
    for line_index, indexes in enumerate(line_indexes):
        line_towers = tuple(towers[index] for index in indexes)
        lines.append(
            PowerLine(
                f"l{line_index:02d}",
                tuple(tower.tower_id for tower in line_towers),
                Polyline3D(tuple(tower.location for tower in line_towers)),
            )
        )
    return PowerNetwork(towers, tuple(lines))


def _power_geometry(
    family: str,
    count: int,
    span: float,
    random: Random,
) -> tuple[tuple[Point3D, ...], tuple[tuple[int, ...], ...]]:
    if family == "crossing":
        coordinates = tuple(
            _point(
                span * (index + 1) / (count + 1) + random.uniform(-1.5, 1.5),
                -0.42 * span
                + 0.84 * span * index / (count - 1)
                + random.uniform(-2.0, 2.0),
                8.0 + random.uniform(-2.0, 3.0),
            )
            for index in range(count)
        )
        return coordinates, (tuple(range(count)),)
    if family == "offset":
        coordinates = tuple(
            _point(
                span * (index + 1) / (count + 1) + random.uniform(-2.0, 2.0),
                0.32 * span
                + 0.17 * span * sin(1.05 * index + 0.4)
                + random.uniform(-1.5, 1.5),
                7.0 + 2.0 * sin(0.6 * index) + random.uniform(-1.0, 1.0),
            )
            for index in range(count)
        )
        return coordinates, (tuple(range(count)),)
    trunk_count = max(3, count // 2)
    coordinates = [
        _point(
            0.55 * span + 0.06 * span * sin(0.9 * index) + random.uniform(-1.0, 1.0),
            -0.35 * span + 0.7 * span * index / (trunk_count - 1),
            8.0 + random.uniform(-1.5, 1.5),
        )
        for index in range(trunk_count)
    ]
    root_index = trunk_count // 2
    branch_indexes = [[], []]
    remaining = count - trunk_count
    for local_index in range(remaining):
        branch = local_index % 2
        step = 1 + local_index // 2
        angle = 0.15 * pi if branch == 0 else 0.85 * pi
        root = coordinates[root_index]
        distance = 13.0 * step
        coordinates.append(
            _point(
                root.x + distance * cos(angle) + random.uniform(-1.0, 1.0),
                root.y + distance * sin(angle) + random.uniform(-1.0, 1.0),
                root.z + random.uniform(-1.0, 1.0),
            )
        )
        branch_indexes[branch].append(len(coordinates) - 1)
    lines = [tuple(range(trunk_count))]
    lines.extend(
        (root_index, *indexes)
        for indexes in branch_indexes
        if indexes
    )
    return tuple(coordinates), tuple(lines)


def _build_stops(
    road: RoadNetwork,
    stop_count: int,
) -> tuple[CandidateStop, ...]:
    if stop_count > len(road.nodes):
        raise ValueError("stop_count exceeds road node count")
    if stop_count == 1:
        indexes = (0,)
    else:
        indexes = tuple(
            round(index * (len(road.nodes) - 1) / (stop_count - 1))
            for index in range(stop_count)
        )
    if len(indexes) != len(set(indexes)):
        raise RuntimeError("deterministic stop sampling produced duplicates")
    return tuple(
        CandidateStop(
            f"s{stop_index:03d}",
            road.nodes[node_index].location,
            road_node_id=road.nodes[node_index].node_id,
        )
        for stop_index, node_index in enumerate(indexes)
    )


def _build_candidates(
    spec: ScenarioSpec,
    power: PowerNetwork,
    stops: tuple[CandidateStop, ...],
    uavs: tuple[UavSpec, ...],
    random: Random,
) -> tuple[PhysicalServiceCandidate, ...]:
    uav_by_id = {item.uav_id: item for item in uavs}
    candidates = []
    for tower_index, tower in enumerate(power.towers):
        nearest = sorted(
            stops,
            key=lambda stop: _distance_3d(stop.location, tower.location),
        )[:2]
        for option, launch in enumerate(nearest):
            recovery = nearest[0] if option else launch
            uav_id = 1 + (tower_index + option) % len(uavs)
            uav = uav_by_id[uav_id]
            outbound = _distance_3d(launch.location, tower.location)
            inbound = _distance_3d(tower.location, recovery.location)
            flight_minutes = 60.0 * (outbound + inbound) / uav.cruise_speed_kmph
            duration = flight_minutes + tower.service_minutes
            expected_energy = (
                0.42 * (outbound + inbound)
                + 0.16 * tower.service_minutes
                + random.uniform(0.0, 0.2)
            )
            energy_std = 0.08 * expected_energy + 0.02 * option
            reliable_energy = expected_energy + 2.0 * energy_std
            admitted = (
                reliable_energy
                <= uav.battery_capacity * (1.0 - uav.reserve_ratio)
                and duration <= spec.horizon_minutes
            )
            payload = (
                uav.owner_vehicle_id,
                uav_id,
                launch.stop_id,
                recovery.stop_id,
                tower.tower_id,
                round(duration, 9),
                round(expected_energy, 9),
                round(reliable_energy, 9),
            )
            candidate_id = "candidate:" + sha256(
                canonical_json(payload).encode("utf-8")
            ).hexdigest()[:20]
            candidates.append(
                PhysicalServiceCandidate(
                    candidate_id,
                    uav.owner_vehicle_id,
                    uav_id,
                    launch.stop_id,
                    recovery.stop_id,
                    (tower.tower_id,),
                    round(duration, 9),
                    ((tower.tower_id, round(duration, 9)),),
                    round(expected_energy, 9),
                    round(energy_std, 9),
                    round(reliable_energy, 9),
                    admitted,
                )
            )
    return tuple(candidates)


def _distance_3d(left: Point3D, right: Point3D) -> float:
    return sqrt(
        (right.x - left.x) ** 2
        + (right.y - left.y) ** 2
        + (right.z - left.z) ** 2
    )


def _point(x: float, y: float, z: float) -> Point3D:
    return Point3D(round(x, 9), round(y, 9), round(z, 9))


__all__ = [
    "FORMAL_SEEDS",
    "GEOMETRY_FAMILIES",
    "PILOT_SEEDS",
    "SIZE_PROFILES",
    "build_formal_specs",
    "build_pilot_specs",
    "generate_scenario",
    "generate_topology_scenario",
]
