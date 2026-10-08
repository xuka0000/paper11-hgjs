"""Frozen development benchmark: quota 2/4, orthogonal capacity and duty.

No sensing-expiry assumption; no test data. T0 is the checked return duration
of a deterministic two-UAV quota tour at capacity two, not a field deadline.
"""
from collections import Counter
from dataclasses import replace
from math import ceil
from random import Random
from time import perf_counter
from types import SimpleNamespace as NS

from Mask.Geometry import Point3D, Polyline3D
from Mask.Scenario import RoadEdge
from Agent.Publication.component_benchmark import scenario as original_scenario
from Agent.Publication.task_groups import TaskGroupModel, TaskGroup, ObservationRequirement
from Agent.Publication.group_scenarios import group_base
from Agent.Publication.problem import candidate_pool, make_search_problem
from Agent.Publication.component_objective import evaluate_plan
from Agent.Run.publication_committed_service_domain import (
    Plan, build_committed_domain, initial_context, handling)
from Agent.Run.optimize_publication_hamming_routes import optimize_committed_routes

VERSION = 'quota-foundation-v1'
WEIGHTS = dict(time_weight=.2, risk_weight=.1, energy_weight=.05,
               makespan_weight=.1, plan_change_weight=.1)


def base_specs():
    return [dict(base_id=f'b{i:02d}', seed=49201+i, components=n, topology=topology)
            for i, (n, topology, _) in enumerate(
                (n, t, r) for n in (6, 12)
                for t in ('parallel_corridors', 'crosslinked_corridors') for r in range(4))]


def instance_model(spec, capacity, horizon):
    instance, _, declaration = original_scenario(spec['seed'], spec['components'], 'slack')
    physical = instance.physical_scenario
    locations = {s.stop_id: s.location for s in physical.stops}
    rng = Random(spec['seed'] + 100003)
    labels = list(range(spec['components']))
    Random(spec['seed'] + 200003).shuffle(labels)
    towers, groups, assets = [], [], []
    for i in range(spec['components']):
        centre = locations[f'site{i}']; radius = rng.uniform(.07, .13)
        ids = tuple(range(4*i+1, 4*i+5)); priority = float(rng.randint(4, 12))
        for task, (dx, dy) in zip(ids, ((1, 0), (-1, 0), (0, 1), (0, -1))):
            towers.append(NS(tower_id=str(task), location=Point3D(
                centre.x+dx*radius, centre.y+dy*radius, .03), service_minutes=1., risk=1.))
        groups.append(TaskGroup(f'component{labels[i]}',
            (ObservationRequirement('admitted_rgb_points', ids, minimum=2),),
            priority, 1+horizon, source='synthetic quota 2 of 4; engineering motivation Zhao 2025'))
        assets.append(dict(site=f'site{i}', group=groups[-1].group_id, tasks=ids,
                           priority=priority, radius_km=radius))
    edges = list(physical.road.edges)
    if spec['topology'] == 'crosslinked_corridors':
        for i in range(0, spec['components'], 2):
            for a, b in ((f'site{i}', f'site{i+1}'), (f'site{i+1}', f'site{i}')):
                edges.append(RoadEdge(a+'_'+b, a, b, Polyline3D((locations[a], locations[b])),
                                      'road', 60., 'synthetic', True))
    physical = NS(**vars(physical))
    physical.road = replace(physical.road, edges=tuple(edges))
    physical.power = NS(towers=tuple(towers))
    physical.vehicles = tuple(replace(v, operation_capacity=capacity,
                                      max_duty_minutes=float(horizon)) for v in physical.vehicles)
    instance.physical_scenario = physical
    tasks = frozenset(range(1, 4*spec['components']+1))
    instance.rolling_state = replace(instance.rolling_state, remaining_tasks=tasks,
        vehicles={v: replace(s, remaining_duty_time=float(horizon))
                  for v, s in instance.rolling_state.vehicles.items()})
    instance.miss_penalty = {t: 1. for t in tasks}
    instance.instance_id = f'{VERSION}:{spec["base_id"]}:c{capacity}:h{horizon}'
    return instance, TaskGroupModel(tuple(groups), 1), assets


def domain_for(spec, capacity, horizon):
    instance, model, assets = instance_model(spec, capacity, horizon)
    base = group_base(instance)
    domain = build_committed_domain(base, initial_context(base))
    model.validate_domain(domain)
    return domain, model, assets


def quota_tour(domain, model, deadline):
    """Deterministic spatial order and first two admitted views, no learned score.

    Carriers use their original lanes; crosslinks remain available to solvers.
    Complete components are inserted together. Failed insertions are skipped.
    """
    plan = Plan(); now = domain.context.state.current_layer
    capacity = {v.vehicle_id: v.operation_capacity for v in domain.instance.physical_scenario.vehicles}
    templates = {(t.task_id, t.uav_id): t for t in domain.templates
                 if t.launch_node == t.recovery_node == f'physical:site{(t.task_id-1)//4}'}
    for i in range(len(model.groups)):
        deadline.check(); vehicle = i % 2 + 1; trial = plan; accepted = True
        for offset in (0, 1):
            task = 4*i+1+offset; uav = 2*(vehicle-1)+1+offset
            template = templates.get((task, uav)); found = False
            if template is None: accepted = False; break
            busy = {(c.uav_id, k) for c in trial.copies for k in range(
                c.launch_layer-1, c.recovery_layer+2)}
            operations = Counter(k for c in trial.copies for k in handling(domain.instance, c, now))
            for launch in template.launches:
                deadline.check()
                recoveries = domain.recovery_options(template, launch)
                if not recoveries: continue
                copy = domain.materialize(template, launch, recoveries[0])
                if any((uav, k) in busy for k in range(launch-1, copy.recovery_layer+2)): continue
                if any(operations[k] >= capacity[vehicle] for k in handling(domain.instance, copy, now)): continue
                try: candidate, _ = optimize_committed_routes(domain, trial.copies+(copy,), WEIGHTS)
                except ValueError as exc:
                    if str(exc) not in ('No route for fixed continuation services',
                                        'No common return-time cap', 'Conflicting mandatory vehicle actions'): raise
                    continue
                trial = candidate; found = True; break
            if not found: accepted = False; break
        if accepted: plan = trial
    checked = evaluate_plan(domain, plan, WEIGHTS, model)
    return plan, checked


def prepare_base(spec, deadline):
    domain, model, _ = domain_for(spec, 2, 96)
    reference, checked = quota_tour(domain, model, deadline)
    if len(checked['completed_groups']) != spec['components']:
        raise ValueError('T0 reference tour did not complete every quota')
    t0 = max(a.arrival_layer for a in reference.arcs if a.head == 'sink') - 1
    result = []
    for capacity in (1, 2):
        for scale in (.75, 1.25):
            horizon = ceil(t0*scale)
            d, m, assets = domain_for(spec, capacity, horizon)
            # Full finite domain, without the former three-launch sampling.
            copies, catalogue = candidate_pool(d, deadline, complete_limit=10**9)
            p = make_search_problem(d, Plan(), copies, WEIGHTS, deadline, catalogue=catalogue)
            warm_started = perf_counter()
            warm, warm_checked = quota_tour(d, m, deadline)
            warm_seconds = perf_counter()-warm_started
            if not warm_checked['completed_groups']:
                raise ValueError('No nonempty quota witness for this declared case')
            if not set(c.tuple_id for c in warm.copies) <= set(p.full_copies):
                raise ValueError('Warm tour outside full common catalogue')
            case_id = f'{spec["base_id"]}_c{capacity}_s{int(scale*100)}'
            declaration = dict(spec, case_id=case_id, capacity=capacity, duty_scale=scale,
                horizon=horizon, T0=t0, T0_certified_groups=spec['components'],
                warm_construction_seconds=warm_seconds,
                quota='2 of 4 distinct admitted RGB points', assets=assets,
                catalogue=catalogue, max_span=None, split='development', final_test=False)
            result.append((case_id, p, m, warm, warm_checked, declaration))
    return result
