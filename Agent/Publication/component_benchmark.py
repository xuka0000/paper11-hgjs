"""Declared development corridors with multiple carriers and components.

Geometry is in km and time in minute layers. These are synthetic stress
conditions, not calibrated inspection deadlines or sensing requirements.
"""
from dataclasses import replace
from random import Random
from types import SimpleNamespace as NS

from Mask.Geometry import Point3D, Polyline3D
from Mask.Scenario import CandidateStop, UavSpec, VehicleSpec, RoadNode, RoadEdge, RoadNetwork
from Mask.Vehicle import VehicleState
from Mask.UAV import UavState
from Env.State import RollingState
from Agent.Publication.task_groups import TaskGroup, TaskGroupModel, ObservationRequirement
from Agent.Publication.group_scenarios import group_base
from Agent.Publication.problem import candidate_pool, make_search_problem
from Agent.Run.publication_committed_service_domain import initial_context, build_committed_domain, Plan

VERSION = 'component-corridor-development-v1'
CONDITIONS = ('slack', 'resource_competition', 'route_synchronization')
SMOKE_CASES = (
    dict(seed=38101, components=6, condition='slack'),
    dict(seed=38101, components=6, condition='resource_competition'),
    dict(seed=38102, components=12, condition='slack'),
    dict(seed=38102, components=12, condition='route_synchronization'),
)


def scenario(seed, components, condition, *, label_seed=None):
    if components not in (6, 12) or condition not in CONDITIONS:
        raise ValueError('Frozen development sizes/conditions required')
    rng = Random(seed)
    stops_per_lane = components // 2
    horizon = 8 * stops_per_lane + 4
    locations = {'source': Point3D(0., 0., 0.), 'sink': Point3D(1.3 * (stops_per_lane + 1), 0., 0.)}
    for i in range(components):
        locations[f'site{i}'] = Point3D(1.3 * (i // 2 + 1), (-1 if i % 2 else 1) * 1.0, 0.)
    links = []
    for lane in (0, 1):
        chain = ['source'] + [f'site{i}' for i in range(lane, components, 2)] + ['sink']
        links.extend(zip(chain, chain[1:]))
    # Identical observable road geometry in paired conditions; synchronization
    # changes completion windows, not the set of geometrically valid roads.
    labels = list(range(components))
    Random(seed + 917 if label_seed is None else label_seed).shuffle(labels)
    towers, groups, assets = [], [], []
    for i in range(components):
        centre = locations[f'site{i}']
        radius = rng.uniform(.07, .13)
        priority = float(rng.randint(4, 12))
        ids = (2 * i + 1, 2 * i + 2)
        for task, direction in zip(ids, (-1, 1)):
            towers.append(NS(tower_id=str(task), location=Point3D(centre.x, centre.y + direction * radius, .03), service_minutes=1., risk=1.))
        group = TaskGroup(f'component{labels[i]}', tuple(ObservationRequirement(name, (task,)) for name, task in zip(('view_a', 'view_b'), ids)),
                          priority, horizon, 3 if condition == 'route_synchronization' else None,
                          source='synthetic two-view requirement; no field calibration')
        groups.append(group)
        assets.append(dict(physical_site=f'site{i}', component=group.group_id, tasks=ids, priority=priority, radius_km=radius))
    capacity = 2 if condition == 'slack' else 1
    physical = NS(road=RoadNetwork(tuple(RoadNode(k, v) for k, v in locations.items()),
        tuple(RoadEdge(a+'_'+b, a, b, Polyline3D((locations[a], locations[b])), 'road', 60., 'synthetic', True) for a, b in links), 'source'),
        stops=tuple(CandidateStop(k, v, road_node_id=k) for k, v in locations.items()),
        power=NS(towers=tuple(towers)),
        uavs=tuple(UavSpec(u, (u-1)//2+1, 60., 1.8, .2, ('rgb',)) for u in range(1, 5)),
        vehicles=tuple(VehicleSpec(v, 'source', float(horizon), capacity) for v in (1, 2)))
    tasks = frozenset(range(1, 2 * components + 1))
    state = RollingState(0, 1, frozenset(), tasks,
        {v: VehicleState(v, 'source', 1, remaining_duty_time=float(horizon)) for v in (1, 2)},
        {u: UavState(u, (u-1)//2+1, 1, remaining_energy=1.8) for u in range(1, 5)})
    instance = NS(split='train', seed=seed, instance_id=f'{VERSION}:{seed}:{components}:{condition}',
        physical_scenario=physical, rolling_state=state, miss_penalty={t: 1. for t in tasks},
        energy_alpha=.9, road_beta_w=0., road_beta_eta=0., preparation_layers=1, recovery_service_layers=1, turnaround_layers=1)
    model = TaskGroupModel(tuple(groups), 1)
    declaration = dict(version=VERSION, seed=seed, components=components, condition=condition, assets=assets,
        vehicles=2, uavs_per_vehicle=2, horizon_layers=horizon, operation_capacity=capacity,
        maximum_observation_span=3 if condition == 'route_synchronization' else None,
        geometry='Two directed source-sink corridors, with several service locations per route',
        assumptions='Synthetic geometry, quality-admitted RGB views, time windows and validity; no field requirement claim',
        label_seed=seed+917 if label_seed is None else label_seed, final_test=False)
    return instance, model, declaration


def construct_case(spec, weights, deadline):
    instance, model, declaration = scenario(**spec)
    base = group_base(instance)
    domain = build_committed_domain(base, initial_context(base))
    copies, catalogue = candidate_pool(domain, deadline)
    problem = make_search_problem(domain, Plan(), copies, weights, deadline, catalogue=catalogue)
    model.validate_domain(domain)
    return problem, model, dict(declaration, catalogue=catalogue)
