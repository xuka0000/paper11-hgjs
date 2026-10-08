"""Route-independent finite single-task service domain for development.

All task/UAV/ordered stop pairs and all feasible integer launch/recovery times
are represented. Times are factored by endpoint instead of materializing their
Cartesian product. No old candidate, assigned corridor or selected route is an
input to this constructor. The fixed geometric/energy surrogate is explicit.
"""
from bisect import bisect_left,bisect_right
from collections import defaultdict
from dataclasses import dataclass,replace
from hashlib import sha256
from math import ceil,floor,sqrt

from Agent.Run.publication_route_choice_model import build_route_choice_domain
from Agent.Tool.Evaluation.dynamic_evaluation import DynamicProblem
from Mask.ServiceTuple import SearchState,SearchTuple
from Mask.TupleFactory import EnergyMoments,build_energy_certificate


@dataclass(frozen=True)
class FlightTemplate:
    template_id:str
    task_id:int
    vehicle_id:int
    uav_id:int
    launch_node:str
    recovery_node:str
    flight_minutes:float
    inspection_completion_minutes:float
    base_mean:float
    base_std:float
    min_duration:int
    max_duration:int
    launches:tuple
    recoveries:tuple


@dataclass
class ServiceDomain:
    instance:object
    problem:object
    road_origins:dict
    templates:tuple
    earliest:dict
    waiting:dict
    statistics:dict

    def recovery_options(self,template,launch):
        if launch not in template.launches:return ()
        lower=max(launch+template.min_duration,self.earliest[(template.vehicle_id,template.recovery_node)].get((template.launch_node,launch),10**9))
        upper=launch+template.max_duration
        return template.recoveries[bisect_left(template.recoveries,lower):bisect_right(template.recoveries,upper)]

    def materialize(self,template,launch,recovery):
        if recovery not in self.recovery_options(template,launch):raise ValueError('Not a feasible service copy in this domain')
        instance=self.instance
        spec=next(u for u in instance.physical_scenario.uavs if u.uav_id==template.uav_id)
        ratio=(recovery-launch)/template.flight_minutes
        cert=build_energy_certificate(EnergyMoments(template.base_mean*ratio,template.base_std*ratio),
            instance.energy_alpha,spec.battery_capacity,spec.reserve_ratio)
        if not cert.admitted:raise ValueError('Copy violates its energy certificate')
        p=instance.preparation_layers;r=instance.recovery_service_layers;t=instance.turnaround_layers
        required=tuple(self.waiting[(template.vehicle_id,node,k)] for node,lo,hi in
            ((template.launch_node,launch-p,launch),(template.recovery_node,recovery,recovery+r)) for k in range(lo,hi))
        operations=tuple(f'op:{template.vehicle_id}:{node}:{k}' for node,lo,hi in
            ((template.launch_node,launch-p,launch),(template.recovery_node,recovery,recovery+r)) for k in range(lo,hi))
        completion=launch+ceil(template.inspection_completion_minutes-1e-12)
        tower=next(t for t in instance.physical_scenario.power.towers if int(t.tower_id)==template.task_id)
        return SearchTuple(f'service:{template.template_id}:{launch}:{recovery}',template.vehicle_id,template.uav_id,
            (template.task_id,),template.launch_node,template.recovery_node,launch,recovery,
            required_arc_ids=required,interval_keys=tuple(f'uav:{template.uav_id}:{k}' for k in range(launch-p,recovery+r+t)),
            operation_keys=operations,completion_layers=((template.task_id,completion),),
            secondary_cost=template.flight_minutes,risk_exposure=tower.risk*(1+.05*completion),
            expected_energy=cert.mean,energy_std=cert.std,reliable_energy=cert.reliable_bound,
            battery_capacity=spec.battery_capacity,reserve_ratio=spec.reserve_ratio,
            admission_certificate=cert.certificate_id,physical_candidate_id=template.template_id)

    def materialized_problem(self,copies):
        capacities={v.vehicle_id:v.operation_capacity for v in self.instance.physical_scenario.vehicles}
        return replace(self.problem,search_state=SearchState(self.problem.search_state.arcs,tuple(copies)),
            operation_capacity={key:capacities[c.vehicle_id] for c in copies for key in c.operation_keys})


def geometric_flight(instance,tower,uav,launch,recovery):
    distance=lambda a,b:sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)
    outbound=distance(launch.location,tower.location);inbound=distance(tower.location,recovery.location)
    duration=60*(outbound+inbound)/uav.cruise_speed_kmph+tower.service_minutes
    completion=60*outbound/uav.cruise_speed_kmph+tower.service_minutes
    # Same geometric energy recipe as the synthetic parent generator. Its
    # nuisance is now keyed by task/UAV, independent of corridor traversal or
    # enumeration order. These are NEW model inputs, not measured calibration
    # or numerically identical historical flight certificates.
    key=f'{instance.seed}|{tower.tower_id}|{uav.uav_id}|service-energy-v1'
    noise=.2*int.from_bytes(sha256(key.encode()).digest()[:8],'big')/(2**64)
    mean=.42*(outbound+inbound)+.16*tower.service_minutes+noise
    return duration,completion,mean,.08*mean


def build_service_domain(instance):
    if instance.split not in ('train','validation'):raise ValueError('Development data only')
    if instance.preparation_layers<1 or instance.recovery_service_layers<1:
        raise ValueError('This implementation requires positive preparation and recovery durations')
    physical=instance.physical_scenario
    if any(s.road_node_id!=s.stop_id for s in physical.stops):raise ValueError('Only explicit road-node stops are supported')
    empty=DynamicProblem(tuple(sorted(instance.rolling_state.remaining_tasks)),dict(instance.miss_penalty),{},
        SearchState((),()),instance.rolling_state,'feasible','')
    route=build_route_choice_domain(instance,empty)
    arcs=route.problem.search_state.arcs
    waiting={(a.vehicle_id,a.tail,a.departure_layer):a.arc_id for a in arcs if a.arc_type=='wait'}
    by_vehicle=defaultdict(list)
    for a in arcs:
        if a.arc_type in ('road','wait'):by_vehicle[a.vehicle_id].append(a)
    # Reverse dynamic programming provides earliest arrivals to each stop for
    # every time-node. Waiting and every physical road alternative are included.
    earliest={}
    for vid,vehicle_arcs in by_vehicle.items():
        descending=sorted(vehicle_arcs,key=lambda a:a.departure_layer,reverse=True)
        nodes={(a.tail,a.departure_layer) for a in vehicle_arcs}|{(a.head,a.arrival_layer) for a in vehicle_arcs}
        for stop in physical.stops:
            target='physical:'+stop.stop_id
            labels={node:node[1] for node in nodes if node[0]==target}
            for a in descending:
                tail=(a.tail,a.departure_layer);head=(a.head,a.arrival_layer)
                if head in labels:labels[tail]=min(labels.get(tail,10**9),labels[head])
            earliest[(vid,target)]=labels
    templates=[];raw=0;pair_count=0
    ready=instance.rolling_state;durations={}
    stop_times={}
    for vid in by_vehicle:
        for stop in physical.stops:
            node='physical:'+stop.stop_id
            at={k for v,n,k in waiting if v==vid and n==node}
            p=instance.preparation_layers;r=instance.recovery_service_layers
            launch=tuple(k for k in range(ready.current_layer+p,257) if all(j in at for j in range(k-p,k)))
            recovery=tuple(k for k in range(ready.current_layer,257) if all(j in at for j in range(k,k+r)))
            stop_times[(vid,node)]=(launch,recovery)
    for tower in sorted(physical.power.towers,key=lambda t:int(t.tower_id)):
        task=int(tower.tower_id)
        if task not in ready.remaining_tasks:continue
        for uav in sorted(physical.uavs,key=lambda u:u.uav_id):
            vid=uav.owner_vehicle_id
            for launch in sorted(physical.stops,key=lambda s:s.stop_id):
                a='physical:'+launch.stop_id
                launches=tuple(k for k in stop_times[(vid,a)][0] if k-instance.preparation_layers>=ready.uavs[uav.uav_id].available_layer)
                for recovery in sorted(physical.stops,key=lambda s:s.stop_id):
                    raw+=1;b='physical:'+recovery.stop_id
                    recoveries=stop_times[(vid,b)][1]
                    if not launches or not recoveries:continue
                    duration,completion,mean,std=geometric_flight(instance,tower,uav,launch,recovery)
                    minimum=ceil(duration-1e-12)
                    base_bound=mean+sqrt(instance.energy_alpha/(1-instance.energy_alpha))*std
                    maximum=floor(uav.battery_capacity*(1-uav.reserve_ratio)*duration/base_bound+1e-10)
                    if maximum<minimum:continue
                    labels=earliest[(vid,b)];valid=[];count=0
                    for k in launches:
                        lo=max(k+minimum,labels.get((a,k),10**9));hi=k+maximum
                        n=max(0,bisect_right(recoveries,hi)-bisect_left(recoveries,lo))
                        if n:valid.append(k);count+=n
                    if not valid:continue
                    identity=f'{task}|{uav.uav_id}|{launch.stop_id}|{recovery.stop_id}'
                    tid='physical-service:'+sha256(identity.encode()).hexdigest()[:20]
                    templates.append(FlightTemplate(tid,task,vid,uav.uav_id,a,b,duration,completion,mean,std,
                        minimum,maximum,tuple(valid),recoveries))
                    pair_count+=count
    return ServiceDomain(instance,route.problem,route.road_origins,tuple(templates),earliest,waiting,
        dict(raw_task_uav_stop_pairs=raw,retained_templates=len(templates),implicit_feasible_time_copies=pair_count,
            road_time_arcs=len(arcs),single_task_sorties=True,old_candidates_read=False,assigned_routes_read=False,
            assumptions='Single-task straight-line 3D flights, constant cruise, synthetic geometric energy with task/UAV keyed nuisance, conservative scaled moments and full battery at each new launch. No obstacle/airspace or measured-energy claim.'))
