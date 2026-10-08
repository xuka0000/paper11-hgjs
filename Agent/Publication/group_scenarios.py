"""Declared synthetic inspection viewpoints, never a field-validated sensor model.

Distance units follow the physical constructor (km), time is minutes/layers.
Each observation is a separate RGB viewpoint around one component centre.
The geometry, shared fleet and requirements are generated before optimization.
"""
from math import cos, sin, pi
from random import Random
from types import SimpleNamespace as NS

from Mask.Geometry import Point3D, Polyline3D
from Mask.Scenario import CandidateStop, UavSpec, VehicleSpec, RoadNode, RoadEdge, RoadNetwork
from Mask.Vehicle import VehicleState
from Mask.UAV import UavState
from Env.State import RollingState
from Agent.Publication.task_groups import ObservationRequirement, TaskGroup, TaskGroupModel


def synthetic_inspection(seed, *, split='train', branches=3, order=None, capacity=None):
    if split not in ('train','validation','test') or branches < 2:
        raise ValueError('Explicit split and at least two competing road branches required')
    rng=Random(seed)
    capacity=rng.choice((1,2)) if capacity is None else capacity
    locations={'source':Point3D(0.,0.,0.),'sink':Point3D(5.,0.,0.)}
    centres={f'asset{i}':Point3D(2.5,(i-(branches-1)/2)*3.,0.) for i in range(branches)}
    locations.update(centres)
    links=tuple((a,b) for name in centres for a,b in (('source',name),(name,'sink')))
    nodes=tuple(RoadNode(k,v) for k,v in locations.items())
    edges=tuple(RoadEdge(a+'_'+b,a,b,Polyline3D((locations[a],locations[b])),
                        'road',60.,'synthetic',True) for a,b in links)
    towers=[];groups=[];assets=[]
    for name,centre in centres.items():
        count=rng.choice((2,3)) if order is None else order
        requirements=[];priority=float(rng.randint(4,12))
        radius=rng.uniform(.07,.13)
        for view in range(count):
            task=len(towers)+1;angle=2*pi*view/count
            point=Point3D(centre.x+radius*cos(angle),centre.y+radius*sin(angle),.03)
            towers.append(NS(tower_id=str(task),location=point,service_minutes=1.,risk=1.))
            requirements.append(ObservationRequirement(f'rgb_view_{view}',(task,)))
        groups.append(TaskGroup(name,tuple(requirements),priority,20,12,
            source='synthetic viewpoint requirement; no claimed field standard'))
        assets.append(dict(component=name,centre=(centre.x,centre.y,centre.z),
                           radius_km=radius,priority=priority,required_views=count,
                           observation_tasks=[r.alternatives[0] for r in requirements]))
    physical=NS(road=RoadNetwork(nodes,edges,'source'),
        stops=tuple(CandidateStop(k,v,road_node_id=k) for k,v in locations.items()),
        power=NS(towers=tuple(towers)),
        uavs=tuple(UavSpec(u,1,60.,1.8,.2,('rgb',)) for u in (1,2)),
        vehicles=(VehicleSpec(1,'source',20.,capacity),))
    tasks=frozenset(range(1,len(towers)+1))
    state=RollingState(0,1,frozenset(),tasks,
        {1:VehicleState(1,'source',1,remaining_duty_time=20.)},
        {u:UavState(u,1,1,remaining_energy=1.8) for u in (1,2)})
    instance=NS(split=split,seed=seed,instance_id=f'synthetic:group-inspection:v1:{split}:{seed}',
        physical_scenario=physical,rolling_state=state,miss_penalty={t:1. for t in tasks},
        energy_alpha=.9,road_beta_w=0.,road_beta_eta=0.,preparation_layers=1,
        recovery_service_layers=1,turnaround_layers=1)
    declaration=dict(seed=seed,split=split,instance_id=instance.instance_id,assets=assets,
        handling_capacity=capacity,uavs=2,vehicles=1,duty_minutes=20,
        source='constructed synthetic physical geometry; development only',
        quality_model='each admitted RGB viewpoint assumed valid; no accuracy/occlusion claims')
    return instance,TaskGroupModel(tuple(groups),1),declaration


def group_base(instance):
    """Same immutable physical inputs for all partitions; no split-specific logic."""
    state=instance.rolling_state
    # Retain the physical task universe: an initially completed observation
    # may need reacquisition if its partial evidence expires during execution.
    return NS(instance=instance,problem=NS(task_ids=tuple(sorted(state.remaining_tasks|state.done_tasks)),
                                         rolling_state=instance.rolling_state))


def advance_group(base,context,plan,model,end,*,validity_layers=None,lock_until=None):
    from dataclasses import replace
    from Agent.Run.publication_committed_service_domain import advance
    next_context,execution=advance(base,context,plan,end,lock_until=lock_until)
    updated=model.executed_update(plan,end,validity_layers)
    next_context=replace(next_context,state=updated.reopen_expired(next_context.state))
    return next_context,updated,execution
