"""Commitment-aware reconstruction using the frozen single-task flight recipe.

This module preserves occupied roads, begun handling, future locks and airborne
recovery. It never resets a deployed vehicle/UAV to its initial state. Closures
prevent new entry; an occupied edge keeps its recorded endpoint and arrival.
New task arrivals, changed forecasts and altered airborne recovery stops are
outside this implementation. Energy is the declared constant-rate surrogate.
"""
from collections import defaultdict, Counter
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from hashlib import sha256
from math import floor, ceil, sqrt
from types import SimpleNamespace

from Agent.Run.publication_service_domain import ServiceDomain, FlightTemplate, geometric_flight
from Agent.Run.publication_route_choice_model import _reachable, time_node
from Agent.Run.publication_road_recovery import edge_arrival
from Agent.Tool.Benchmark.benchmark_v3 import _formal_v3_weather
from Agent.Tool.Evaluation.dynamic_evaluation import DynamicProblem
from Env.Runtime import ActiveFlight
from Mask.ServiceTuple import SearchArc, SearchState


@dataclass(frozen=True)
class Plan:
    arcs: tuple = ()
    copies: tuple = ()


@dataclass(frozen=True)
class Continuation:
    state: object
    previous: Plan = Plan()


@dataclass
class CommittedDomain:
    base: object
    services: object
    context: Continuation
    specials: tuple
    groups: dict
    required_arc_ids: frozenset
    required_vehicle_ids: frozenset
    fixed_busy: dict
    fixed_operations: dict
    previous_copies: tuple
    closures: tuple

    @property
    def instance(self): return self.services.instance
    @property
    def problem(self): return self.services.problem
    @property
    def templates(self): return self.services.templates
    @property
    def waiting(self): return self.services.waiting
    @property
    def road_origins(self): return self.services.road_origins
    @property
    def statistics(self): return self.services.statistics
    def materialize(self, template, launch, recovery):
        return self.services.materialize(template, launch, recovery)
    def recovery_options(self, template, launch):
        return self.services.recovery_options(template, launch)


def initial_context(base):
    return Continuation(base.problem.rolling_state)


def ready_layer(instance, copy):
    return copy.recovery_layer + instance.recovery_service_layers + instance.turnaround_layers


def handling(instance, copy, now):
    p=instance.preparation_layers; r=instance.recovery_service_layers
    return tuple((copy.vehicle_id,node,k) for node,lo,hi in
                 ((copy.launch_node,copy.launch_layer-p,copy.launch_layer),
                  (copy.recovery_node,copy.recovery_layer,copy.recovery_layer+r))
                 for k in range(max(now,lo),hi))


def copy_terms(instance, copy, state, weights):
    """Remaining objective terms; launch overhead is sunk once launch occurs."""
    now=state.current_layer
    covered=tuple(task for task in copy.task_ids if task in state.remaining_tasks)
    duration=copy.recovery_layer-copy.launch_layer
    remaining=max(0,copy.recovery_layer-max(now,copy.launch_layer))
    energy=copy.expected_energy*remaining/duration
    towers={int(t.tower_id):t for t in instance.physical_scenario.power.towers}
    risk=sum(towers[t].risk*(1+.05*k) for t,k in copy.completion_layers if t in covered)
    cost=(copy.secondary_cost if copy.launch_layer>=now and not copy.is_air_residual else 0.)
    return covered,cost+weights['energy_weight']*energy+weights['risk_weight']*risk


def physical_road(base, vehicle_id, arc_id):
    prefix=f'route_road:{vehicle_id}:'
    if not arc_id.startswith(prefix):raise ValueError('Unknown physical road identity')
    edge_id,departure,arrival=arc_id[len(prefix):].rsplit(':',2)
    edges={e.edge_id:e for e in base.instance.physical_scenario.road.edges}
    if edge_id not in edges:raise ValueError('Unknown physical edge')
    edge=edges[edge_id];departure=int(departure);arrival=int(arrival)
    if edge_arrival(base.instance,edge,departure,_formal_v3_weather(base.instance.seed,256))!=arrival:
        raise ValueError('Recorded road arrival contradicts frozen forecast')
    return SearchArc(arc_id,vehicle_id,'physical:'+edge.tail,'physical:'+edge.head,departure,arrival,
                     'road',edge.geometry.length_3d,float(arrival-departure))


def services_on_graph(instance, problem, origins, preserved_tasks):
    """All physical task/UAV/stop pairs and times on the actual rolling graph.

    Rebuilding is necessary for idle vehicles whose unused duty permits a later
    departure than the previous static graph represented. No old route or old
    candidate is consulted. The geometric flight recipe and IDs remain stable.
    """
    state=problem.rolling_state;physical=instance.physical_scenario
    waiting={(a.vehicle_id,a.tail,a.departure_layer):a.arc_id for a in problem.search_state.arcs if a.arc_type=='wait'}
    by_vehicle=defaultdict(list)
    for a in problem.search_state.arcs:
        if a.arc_type in ('road','wait'):by_vehicle[a.vehicle_id].append(a)
    earliest={};times={};p=instance.preparation_layers;r=instance.recovery_service_layers
    for vid,arcs in by_vehicle.items():
        descending=sorted(arcs,key=lambda a:a.departure_layer,reverse=True)
        nodes={(a.tail,a.departure_layer) for a in arcs}|{(a.head,a.arrival_layer) for a in arcs}
        for stop in physical.stops:
            node='physical:'+stop.stop_id
            labels={key:key[1] for key in nodes if key[0]==node}
            for a in descending:
                tail=(a.tail,a.departure_layer);head=(a.head,a.arrival_layer)
                if head in labels:labels[tail]=min(labels.get(tail,10**9),labels[head])
            earliest[(vid,node)]=labels
            at={k for v,n,k in waiting if v==vid and n==node}
            launch=tuple(k for k in range(state.current_layer+p,257) if all(j in at for j in range(k-p,k)))
            recovery=tuple(k for k in range(state.current_layer,257) if all(j in at for j in range(k,k+r)))
            times[(vid,node)]=(launch,recovery)
    templates=[];raw=0;count=0
    for tower in sorted(physical.power.towers,key=lambda t:int(t.tower_id)):
        task=int(tower.tower_id)
        if task not in state.remaining_tasks or task in preserved_tasks:continue
        for uav in sorted(physical.uavs,key=lambda u:u.uav_id):
            vid=uav.owner_vehicle_id;actual=state.uavs[uav.uav_id]
            available=state.current_layer if actual.is_airborne else max(state.current_layer,actual.available_layer)
            for launch in sorted(physical.stops,key=lambda s:s.stop_id):
                a='physical:'+launch.stop_id
                ls=tuple(k for k in times.get((vid,a),((),()))[0] if k-p>=available)
                for recovery in sorted(physical.stops,key=lambda s:s.stop_id):
                    raw+=1;b='physical:'+recovery.stop_id;rs=times.get((vid,b),((),()))[1]
                    if not ls or not rs:continue
                    duration,completion,mean,std=geometric_flight(instance,tower,uav,launch,recovery)
                    minimum=ceil(duration-1e-12);bound=mean+sqrt(instance.energy_alpha/(1-instance.energy_alpha))*std
                    maximum=floor(uav.battery_capacity*(1-uav.reserve_ratio)*duration/bound+1e-10)
                    if minimum>maximum:continue
                    valid=[]
                    for k in ls:
                        lo=max(k+minimum,earliest[(vid,b)].get((a,k),10**9));hi=k+maximum
                        n=max(0,bisect_right(rs,hi)-bisect_left(rs,lo))
                        if n:valid.append(k);count+=n
                    if not valid:continue
                    identity=f'{task}|{uav.uav_id}|{launch.stop_id}|{recovery.stop_id}'
                    tid='physical-service:'+sha256(identity.encode()).hexdigest()[:20]
                    templates.append(FlightTemplate(tid,task,vid,uav.uav_id,a,b,duration,completion,mean,std,
                                                    minimum,maximum,tuple(valid),rs))
    return ServiceDomain(instance,problem,origins,tuple(templates),earliest,waiting,
                         dict(raw_task_uav_stop_pairs=raw,retained_templates=len(templates),implicit_feasible_time_copies=count,
                              road_time_arcs=len(problem.search_state.arcs),old_candidates_read=False,assigned_routes_read=False))


def declared_services_on_graph(instance, problem, origins, preserved_tasks):
    """Build rolling templates only from declared physical service modes."""
    state = problem.rolling_state
    physical = instance.physical_scenario
    waiting = {
        (arc.vehicle_id, arc.tail, arc.departure_layer): arc.arc_id
        for arc in problem.search_state.arcs
        if arc.arc_type == "wait"
    }
    by_vehicle = defaultdict(list)
    for arc in problem.search_state.arcs:
        if arc.arc_type in ("road", "wait"):
            by_vehicle[arc.vehicle_id].append(arc)
    earliest = {}
    times = {}
    p = instance.preparation_layers
    r = instance.recovery_service_layers
    for vehicle_id, arcs in by_vehicle.items():
        descending = sorted(
            arcs, key=lambda arc: arc.departure_layer, reverse=True
        )
        nodes = {
            (arc.tail, arc.departure_layer) for arc in arcs
        } | {
            (arc.head, arc.arrival_layer) for arc in arcs
        }
        for stop in physical.stops:
            node = "physical:" + stop.stop_id
            labels = {key: key[1] for key in nodes if key[0] == node}
            for arc in descending:
                tail = arc.tail, arc.departure_layer
                head = arc.head, arc.arrival_layer
                if head in labels:
                    labels[tail] = min(labels.get(tail, 10**9), labels[head])
            earliest[(vehicle_id, node)] = labels
            at = {
                layer
                for owner, name, layer in waiting
                if owner == vehicle_id and name == node
            }
            launches = tuple(
                layer
                for layer in range(state.current_layer + p, 257)
                if all(value in at for value in range(layer - p, layer))
            )
            recoveries = tuple(
                layer
                for layer in range(state.current_layer, 257)
                if all(value in at for value in range(layer, layer + r))
            )
            times[(vehicle_id, node)] = launches, recoveries

    uav_by_id = {uav.uav_id: uav for uav in physical.uavs}
    templates = []
    feasible_copy_count = 0
    declared_count = 0
    for candidate in sorted(
        physical.candidates, key=lambda item: item.candidate_id
    ):
        if not candidate.admitted:
            continue
        if len(candidate.task_ids) != 1:
            raise ValueError("declared matrix modes must be single-task services")
        task = int(candidate.task_ids[0])
        if task not in state.remaining_tasks or task in preserved_tasks:
            continue
        declared_count += 1
        uav = uav_by_id[candidate.uav_id]
        if uav.owner_vehicle_id != candidate.vehicle_id:
            raise ValueError("declared service mode has the wrong carrier")
        actual = state.uavs[uav.uav_id]
        available = (
            state.current_layer
            if actual.is_airborne
            else max(state.current_layer, actual.available_layer)
        )
        launch_node = "physical:" + candidate.launch_stop_id
        recovery_node = "physical:" + candidate.recovery_stop_id
        launches = tuple(
            layer
            for layer in times.get(
                (candidate.vehicle_id, launch_node), ((), ())
            )[0]
            if layer - p >= available
        )
        recoveries = times.get(
            (candidate.vehicle_id, recovery_node), ((), ())
        )[1]
        if not launches or not recoveries:
            continue
        duration = float(candidate.duration_minutes)
        completion = float(candidate.completion_offsets_minutes[0][1])
        mean = float(candidate.expected_energy)
        std = float(candidate.energy_std)
        minimum = ceil(duration - 1e-12)
        bound = mean + sqrt(
            instance.energy_alpha / (1.0 - instance.energy_alpha)
        ) * std
        maximum = floor(
            uav.battery_capacity
            * (1.0 - uav.reserve_ratio)
            * duration
            / bound
            + 1e-10
        )
        if minimum > maximum:
            continue
        valid = []
        for launch in launches:
            lower = max(
                launch + minimum,
                earliest[(candidate.vehicle_id, recovery_node)].get(
                    (launch_node, launch), 10**9
                ),
            )
            upper = launch + maximum
            count = max(
                0,
                bisect_right(recoveries, upper)
                - bisect_left(recoveries, lower),
            )
            if count:
                valid.append(launch)
                feasible_copy_count += count
        if not valid:
            continue
        templates.append(FlightTemplate(
            candidate.candidate_id,
            task,
            candidate.vehicle_id,
            candidate.uav_id,
            launch_node,
            recovery_node,
            duration,
            completion,
            mean,
            std,
            minimum,
            maximum,
            tuple(valid),
            tuple(recoveries),
        ))
    return ServiceDomain(
        instance,
        problem,
        origins,
        tuple(templates),
        earliest,
        waiting,
        dict(
            declared_physical_candidates=declared_count,
            retained_templates=len(templates),
            implicit_feasible_time_copies=feasible_copy_count,
            road_time_arcs=len(problem.search_state.arcs),
            declared_candidates_read=True,
            assigned_routes_read=False,
        ),
    )


def build_committed_domain(
    base, context, closures=(), *, declared_service_candidates=False
):
    state=context.state; now=state.current_layer; instance=base.instance
    if instance.split not in ('train','validation'): raise ValueError('Development only')
    if not state.remaining_tasks <= set(base.problem.task_ids): raise ValueError('New tasks need a new physical catalogue')
    if state.done_tasks & state.remaining_tasks: raise ValueError('Task partition overlaps')
    edges={e.edge_id:e for e in instance.physical_scenario.road.edges}
    if any(c.edge_id not in edges or c.reveal_layer>now for c in closures):
        raise ValueError('Unknown or unrevealed closure')
    closed={c.edge_id for c in closures}
    previous_arcs={a.arc_id:a for a in context.previous.arcs}
    previous_copies={c.tuple_id:c for c in context.previous.copies}
    if state.locked_arc_ids-previous_arcs.keys() or state.locked_tuple_ids-previous_copies.keys():
        raise ValueError('Missing locked action correspondence')
    air={uid for uid,u in state.uavs.items() if u.is_airborne}
    if air!=set(state.airborne_services) or air!=set(state.active_service_ids):
        raise ValueError('Incomplete airborne flight identity')
    required=set(); mandatory=set(); all_arcs=[];origins={}
    weather=_formal_v3_weather(instance.seed,256)
    nodes=sorted({n for edge in edges.values() for n in (edge.tail,edge.head)})
    for vid,v in state.vehicles.items():
        flights=[f for f in state.airborne_services.values() if f.vehicle_id==vid]
        must=v.must_return or v.active_arc is not None or bool(flights)
        if must: mandatory.add(vid)
        start=max(now,v.available_layer); node='physical:'+str(v.node_id)
        end=min(256,start+floor(v.remaining_duty_time+1e-8)) if v.remaining_duty_time is not None else 256
        if v.active_arc is not None:
            old=previous_arcs.get(v.active_arc_id)
            source=physical_road(base,vid,v.active_arc_id)
            if (old is None or source is None or source.arc_type!='road' or
                source.arrival_layer!=v.active_arc_arrival_layer or
                (source.tail.removeprefix('physical:'),source.head.removeprefix('physical:'))!=v.active_arc or
                not source.departure_layer<now<source.arrival_layer or
                abs(v.active_arc_duty_time-(source.arrival_layer-now))>1e-7):
                raise ValueError('Invalid occupied road correspondence')
            start=source.arrival_layer; node=source.head
            end=min(256,now+floor(v.remaining_duty_time+1e-8)) if v.remaining_duty_time is not None else 256
            anchor=replace(source,tail='source',departure_layer=now,arc_type='committed',
                           secondary_cost=0.,duty_time=v.active_arc_duty_time)
            required.add(anchor.arc_id)
        else:
            anchor=SearchArc(f'rolling_anchor:{vid}:{now}:{start}:{node}',vid,'source',node,now,start,'anchor')
        current=[anchor]
        for layer in range(start,end+1):
            current.append(SearchArc(f'route_terminal:{vid}:{layer}',vid,'physical:sink','sink',layer,layer,'terminal'))
            if layer==end:continue
            for name in nodes:
                current.append(SearchArc(f'route_wait:{vid}:{name}:{layer}',vid,'physical:'+name,'physical:'+name,
                                         layer,layer+1,'wait',0.,1.))
            for edge in edges.values():
                if edge.edge_id in closed:continue
                arrival=edge_arrival(instance,edge,layer,weather)
                if arrival is None or arrival>end:continue
                aid=f'route_road:{vid}:{edge.edge_id}:{layer}:{arrival}'
                current.append(SearchArc(aid,vid,'physical:'+edge.tail,'physical:'+edge.head,layer,arrival,
                                         'road',edge.geometry.length_3d,float(arrival-layer)))
                origins[aid]=edge.edge_id
        forward=defaultdict(list); backward=defaultdict(list)
        for arc in current:
            a,b=time_node(arc,True),time_node(arc,False)
            forward[a].append(b);backward[b].append(a)
        reachable=_reachable(('__source__',-1),forward)
        returnable=_reachable(('__sink__',-1),backward)
        retained=[a for a in current if time_node(a,True) in reachable and time_node(a,False) in returnable]
        if must and not retained: raise ValueError(f'Obligated vehicle has no return path: {vid}')
        all_arcs.extend(retained)
    arcs={a.arc_id:a for a in all_arcs}
    waiting={(a.vehicle_id,a.tail,a.departure_layer):a.arc_id for a in all_arcs if a.arc_type=='wait'}
    # Future locks preserve selected decisions, not merely a route preference.
    required.update(k for k in state.locked_arc_ids if previous_arcs[k].arrival_layer>now)
    if required-arcs.keys(): raise ValueError('A locked road or occupied edge is unavailable')
    specials=[]; groups={}; fixed_busy=Counter(); fixed_ops=Counter()
    preserved_tasks=set(); previous_future=[]
    p=instance.preparation_layers; r=instance.recovery_service_layers
    for copy in context.previous.copies:
        start=copy.launch_layer-p; ready=ready_layer(instance,copy)
        if start>=now: previous_future.append(copy)
        locked=copy.tuple_id in state.locked_tuple_ids
        if copy.uav_id in air and copy.launch_layer<=now<copy.recovery_layer:
            continue
        if copy.recovery_layer<=now<ready:
            fixed_busy.update((copy.uav_id,k) for k in range(now,ready))
            fixed_ops.update(handling(instance,copy,now))
            continue
        if ready<=now: continue
        if locked or start<now<copy.launch_layer:
            if not set(copy.task_ids)<=state.remaining_tasks: raise ValueError('Locked task already done')
            specials.append(copy); groups['locked:'+copy.tuple_id]=(copy.tuple_id,)
            preserved_tasks.update(copy.task_ids)
    for uid in sorted(air):
        flight=state.airborne_services[uid]; uav=state.uavs[uid]
        originals=[c for c in context.previous.copies if c.uav_id==uid and c.launch_layer<=now<c.recovery_layer]
        if (len(originals)!=1 or state.active_service_ids[uid]!=flight.tuple_id or
            uav.owner_vehicle_id!=flight.vehicle_id or uav.remaining_energy is None):
            raise ValueError('Missing active service or observed energy')
        old=originals[0]; ids=[]
        if flight.launch_layer!=old.launch_layer or flight.recovery_layer!=old.recovery_layer:
            raise ValueError('Flight timing does not match selected history')
        if flight.recovery_node!=old.recovery_node: raise ValueError('Airborne recovery stop changed')
        fixed_recovery=old.tuple_id in state.locked_tuple_ids
        for end in range(max(now+1,flight.recovery_layer),257):
            if fixed_recovery and end!=flight.recovery_layer: continue
            if not all((flight.vehicle_id,flight.recovery_node,k) in waiting for k in range(end,end+r)):continue
            mean,std=flight.remaining_moments(now,end)
            bound=mean+sqrt(instance.energy_alpha/(1-instance.energy_alpha))*std
            if bound>uav.remaining_energy-flight.reserve_ratio*flight.battery_capacity+1e-8:continue
            factor=(end-old.launch_layer)/(old.recovery_layer-old.launch_layer)
            key=f'residual:{flight.tuple_id}:{end}'
            copy=replace(old,tuple_id=key,recovery_layer=end,expected_energy=old.expected_energy*factor,
                         energy_std=old.energy_std*factor,reliable_energy=(old.expected_energy+
                         sqrt(instance.energy_alpha/(1-instance.energy_alpha))*old.energy_std)*factor,
                         is_air_residual=True,admission_certificate=f'conditional:{flight.tuple_id}:{now}:{end}',
                         required_arc_ids=tuple(waiting[(flight.vehicle_id,flight.recovery_node,k)] for k in range(end,end+r)),
                         interval_keys=tuple(f'uav:{uid}:{k}' for k in range(now,end+r+instance.turnaround_layers)),
                         operation_keys=tuple(f'op:{flight.vehicle_id}:{flight.recovery_node}:{k}' for k in range(end,end+r)))
            specials.append(copy);ids.append(key)
        if not ids: raise ValueError(f'No safe reachable recovery for airborne UAV: {uid}')
        groups['airborne:'+str(uid)]=tuple(ids);preserved_tasks.update(old.task_ids)
    for key in fixed_ops:
        if key not in waiting: raise ValueError('Begun recovery cannot be completed')
        required.add(waiting[key])
    for copy in specials:
        if any(key not in waiting for key in handling(instance,copy,now)):
            raise ValueError('Locked preparation or recovery cannot be completed')
    instance_view=SimpleNamespace(**vars(instance));instance_view.rolling_state=state
    problem=DynamicProblem(tuple(sorted(state.remaining_tasks)),{t:instance.miss_penalty[t] for t in state.remaining_tasks},
                           {},SearchState(tuple(all_arcs),()),state,'feasible','')
    service_builder = (
        declared_services_on_graph
        if declared_service_candidates
        else services_on_graph
    )
    services=service_builder(instance_view,problem,{k:v for k,v in origins.items() if k in arcs},preserved_tasks)
    return CommittedDomain(base,services,context,tuple(specials),groups,frozenset(required),frozenset(mandatory),
                           dict(fixed_busy),dict(fixed_ops),tuple(previous_future),tuple(closures))


def objective(domain, plan, weights):
    state=domain.context.state; now=state.current_layer
    covered=set();cost=0.
    for copy in plan.copies:
        tasks,value=copy_terms(domain.instance,copy,state,weights);covered.update(tasks);cost+=value
    primary=sum(domain.problem.miss_penalty[t] for t in domain.problem.task_ids if t not in covered)
    cost+=sum(a.secondary_cost+weights['time_weight']*a.duty_time for a in plan.arcs)
    ends=[a.arrival_layer for a in plan.arcs if a.head=='sink']
    completions=[k for c in plan.copies for t,k in c.completion_layers if t in covered]
    cost+=weights['makespan_weight']*max(ends+completions,default=0)
    selected_arcs={a.arc_id for a in plan.arcs};selected_copies={c.tuple_id for c in plan.copies}
    # Only not-yet-started actions have a cancellable correspondence.
    old_arcs={a.arc_id for a in domain.context.previous.arcs if a.arc_type in ('road','wait') and a.departure_layer>=now}
    old_copies={c.tuple_id for c in domain.previous_copies}
    cost+=weights['plan_change_weight']*(len(old_arcs-selected_arcs)+len(old_copies-selected_copies))
    return float(primary),float(cost)


def retain_previous_plan(domain):
    """Construct a continuation certificate when the original future stays feasible."""
    now=domain.context.state.current_layer
    arcs={a.arc_id:a for a in domain.problem.search_state.arcs}
    selected=[];vehicles=set(domain.required_vehicle_ids)
    for old in domain.context.previous.arcs:
        if old.arc_type=='anchor' or old.arrival_layer<now:continue
        if old.arc_id not in arcs:raise ValueError('Previous future path is unavailable')
        selected.append(arcs[old.arc_id]);vehicles.add(old.vehicle_id)
    selected.extend(a for a in arcs.values() if a.arc_type=='anchor' and a.vehicle_id in vehicles)
    specials={c.tuple_id:c for c in domain.specials};copies=[]
    for old in domain.context.previous.copies:
        if old.recovery_layer<=now:continue
        if old.uav_id in domain.context.state.airborne_services and old.launch_layer<=now:
            options=[c for c in domain.specials if c.uav_id==old.uav_id and c.is_air_residual and c.recovery_layer==old.recovery_layer]
            if len(options)!=1:raise ValueError('Original recovery is unavailable')
            copies.append(options[0])
        else:copies.append(specials.get(old.tuple_id,old))
    return Plan(tuple(selected),tuple(copies))


def advance(base, context, plan, end, *, lock_until=None):
    """Exact deterministic prefix execution, with paid duty and energy once.

    Events at end are applied before the next decision. A unit handling interval
    beginning at end has not begun. Plans carry complete flight history so that
    residual energy uses the original rate across any number of replans.
    """
    state=context.state;now=state.current_layer;instance=base.instance
    if end<=now or end>256:raise ValueError('Invalid execution interval')
    done=set(state.done_tasks);vehicles={};uavs=dict(state.uavs);active={};flights={}
    for vid,v in state.vehicles.items():
        chosen=sorted((a for a in plan.arcs if a.vehicle_id==vid),key=lambda a:(a.departure_layer,a.arrival_layer,a.arc_id))
        used=0.;node=v.node_id;active_arc=None;arrival=None;aid=None;depart=None;unpaid=0.
        for a in chosen:
            if a.arc_type in ('road','wait','committed'):
                duration=a.arrival_layer-a.departure_layer
                elapsed=max(0,min(end,a.arrival_layer)-max(now,a.departure_layer))
                used+=a.duty_time*elapsed/duration
            if a.arrival_layer<=end and a.head!='sink':node=a.head.removeprefix('physical:')
            if a.head=='sink' and a.arrival_layer<=end:node='sink'
            if a.arc_type in ('road','committed') and a.departure_layer<end<a.arrival_layer:
                original=physical_road(base,vid,a.arc_id)
                active_arc=(original.tail.removeprefix('physical:'),original.head.removeprefix('physical:'))
                aid=a.arc_id;arrival=a.arrival_layer;depart=original.departure_layer;unpaid=a.arrival_layer-end
        remaining=None if v.remaining_duty_time is None else v.remaining_duty_time-used
        if remaining is not None and remaining < -1e-7:raise ValueError('Executed duty exhausted')
        deployed=v.must_return or any(a.arc_type in ('road','wait','committed') and a.departure_layer<end for a in chosen)
        vehicles[vid]=replace(v,node_id=node,available_layer=arrival or end,active_arc=active_arc,
                             active_arc_arrival_layer=arrival,active_arc_id=aid,active_arc_departure_layer=depart,
                             active_arc_duty_time=unpaid,remaining_duty_time=remaining,must_return=deployed and node!='sink')
    energy_used=Counter()
    for uid,u in state.uavs.items():
        energy=u.remaining_energy;availability=max(end,u.available_layer);remaining_tasks=();is_air=False
        copies=sorted((c for c in plan.copies if c.uav_id==uid),key=lambda c:c.launch_layer)
        for c in copies:
            if c.launch_layer>end:continue
            if c.launch_layer>=now and not c.is_air_residual:
                energy=c.battery_capacity
            elapsed=max(0,min(end,c.recovery_layer)-max(now,c.launch_layer))
            consumption=c.expected_energy*elapsed/(c.recovery_layer-c.launch_layer)
            if energy is None:raise ValueError('Missing execution energy')
            energy-=consumption;energy_used[uid]+=consumption
            if energy < -1e-8:raise ValueError('Executed energy exhausted')
            for task,k in c.completion_layers:
                if k<=end:done.add(task)
            if c.launch_layer<=end<c.recovery_layer:
                is_air=True;remaining_tasks=tuple(t for t in c.task_ids if t not in done)
                original=state.active_service_ids.get(uid) if c.is_air_residual else c.tuple_id
                if not original:raise ValueError('Residual lost original flight identity')
                flights[uid]=ActiveFlight(original,c.vehicle_id,uid,c.launch_layer,c.recovery_layer,c.recovery_node,
                                          c.completion_layers,c.expected_energy,c.energy_std,c.battery_capacity,c.reserve_ratio)
                active[uid]=original
            elif c.recovery_layer<=end:
                is_air=False;remaining_tasks=();availability=max(availability,ready_layer(instance,c))
        uavs[uid]=replace(u,available_layer=availability,is_airborne=is_air,remaining_energy=energy,remaining_tasks=remaining_tasks)
    # Carry unfinished recovery/turnaround records as well as future services.
    retained=tuple(c for c in plan.copies if ready_layer(instance,c)>end)
    old_carried=tuple(c for c in context.previous.copies if c.recovery_layer<=now<ready_layer(instance,c) and ready_layer(instance,c)>end)
    retained=retained+tuple(c for c in old_carried if c.tuple_id not in {x.tuple_id for x in retained})
    arcs=tuple(a for a in plan.arcs if a.arrival_layer>end)
    boundary=end if lock_until is None else lock_until
    if boundary<end:raise ValueError('Lock boundary precedes state')
    locks_a={a.arc_id for a in arcs if a.departure_layer<boundary and a.arc_type in ('road','wait','committed')}
    locks_a.update(v.active_arc_id for v in vehicles.values() if v.active_arc_id)
    locks_c={c.tuple_id for c in retained if c.launch_layer-instance.preparation_layers<boundary and c.launch_layer>end}
    # Previously immutable future services remain immutable until completed.
    locks_a.update(state.locked_arc_ids & {a.arc_id for a in arcs})
    locks_c.update(state.locked_tuple_ids & {c.tuple_id for c in retained})
    for c in retained:
        if c.is_air_residual and any(old.tuple_id in state.locked_tuple_ids and
                                    old.uav_id==c.uav_id and old.launch_layer==c.launch_layer
                                    for old in context.previous.copies):
            locks_c.add(c.tuple_id)
    new=replace(state,epoch=state.epoch+1,current_layer=end,done_tasks=frozenset(done),
                remaining_tasks=frozenset(set(base.problem.task_ids)-done),vehicles=vehicles,uavs=uavs,
                locked_arc_ids=frozenset(locks_a),locked_tuple_ids=frozenset(locks_c),
                active_service_ids=active,airborne_services=flights,airborne_tasks={uid:uavs[uid].remaining_tasks for uid in flights},
                previous_arc_values={a.arc_id:1 for a in arcs},previous_tuple_values={c.tuple_id:1 for c in retained},energy_observations={})
    return Continuation(new,Plan(arcs,retained)),dict(energy_consumed=dict(energy_used),completed_tasks=sorted(done-state.done_tasks))
