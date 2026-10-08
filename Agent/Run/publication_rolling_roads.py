"""Isolated rolling inspection with persistent physical roads and timed service copies.

The route constructor retains open legs and tries forward reconnections around
closures. Scheduling optimizers choose services on the constructed routes. This
is a restricted routing heuristic, not joint optimization over every road path.
"""
from dataclasses import dataclass, replace, asdict
from math import ceil
from types import SimpleNamespace
from copy import deepcopy

from Agent.Run.publication_road_recovery import earliest_path, edge_arrival, SegmentClosure
from Agent.Tool.Benchmark.benchmark_v3 import _formal_v3_weather
from Agent.Tool.Evaluation.dynamic_evaluation import DynamicProblem, validate_dynamic_plan, _plan_entries
from Agent.Tool.Evaluation.method_runner import run_method
from Env.Execution import PlannedVehicleArcEvent, PlannedRecoveryEvent, PlannedLaunchEvent, execute_due_events, advance_state, record_plan
from Mask.ServiceTuple import SearchArc, SearchTuple, SearchState
from Mask.TupleFactory import EnergyMoments, build_energy_certificate, build_air_residual

TERMINAL=128
HORIZON=256


def model_node(node):return 'physical:'+node
def physical_node(node):return node.removeprefix('physical:')


@dataclass
class RoadMemory:
    routes: dict
    edge_ids: dict
    flight_origins: dict

    @classmethod
    def initial(cls,instance):
        pairs={(e.tail,e.head):e.edge_id for e in instance.physical_scenario.road.edges}
        return cls({}, {a.arc_id:pairs[(a.tail,a.head)] for a in instance.search_state.arcs},
                   {t.tuple_id:t.physical_candidate_id for t in instance.search_state.tuples})


def service_keys(instance,vid,uid,node,launch,recovery,airborne=False):
    p=0 if airborne else instance.preparation_layers
    r=instance.recovery_service_layers
    ready=instance.turnaround_layers
    operations=tuple([f'op:{vid}:{node[0]}:{k}' for k in range(launch-p,launch)]+
                     [f'op:{vid}:{node[1]}:{k}' for k in range(recovery,recovery+r)])
    intervals=tuple(f'uav:{uid}:{k}' for k in range(launch-p,recovery+r+ready))
    return intervals,operations


def _physical_route(instance,state,vid,memory,closed,allow_detour):
    vehicle=state.vehicles[vid]
    edges={e.edge_id:e for e in instance.physical_scenario.road.edges}
    pairs={(e.tail,e.head):e for e in edges.values()}
    weather=_formal_v3_weather(instance.seed,HORIZON)
    templates=instance.physical_scenario.candidates
    service_nodes={n for c in templates if c.vehicle_id==vid for n in (c.launch_stop_id,c.recovery_stop_id)}
    flights=[f for f in state.airborne_services.values() if f.vehicle_id==vid]
    mandatory={f.recovery_node for f in flights}
    service_nodes.update(mandatory)
    old=tuple(memory.routes.get(vid,()))
    deployed=vehicle.must_return or vehicle.active_arc is not None or bool(flights)
    if not deployed:
        if vehicle.node_id=='sink':return (),None
        original=sorted((a for a in instance.search_state.arcs if a.vehicle_id==vid),key=lambda a:a.departure_layer)
        old=tuple(original)
        fresh=True
    else:
        fresh=False
        if not old:
            return (),f'vehicle={vid}:persistent_route_missing'
    if vehicle.active_arc is not None:
        matches=[a for a in old if a.arc_id==vehicle.active_arc_id]
        if (len(matches)!=1 or matches[0].arrival_layer!=vehicle.active_arc_arrival_layer
            or matches[0].head!=vehicle.active_arc[1]
            or abs(matches[0].duty_time-vehicle.active_arc_duty_time)>1e-8):
            return (),f'vehicle={vid}:invalid_commitment'
    blocked=[i for i,a in enumerate(old) if a.arc_type=='road'
             and a.arc_id!=vehicle.active_arc_id
             and (memory.edge_ids[a.arc_id] in closed or not edges[memory.edge_ids[a.arc_id]].available)]
    if not fresh and not blocked:
        return old,None
    if blocked and not allow_detour:
        return (),f'vehicle={vid}:closed_future_segment'
    # Keep the complete timed prefix before the first blocked segment.
    first=0 if fresh else blocked[0]
    route=list(old[:first])
    node=vehicle.node_id if fresh else old[first].tail
    layer=max(state.current_layer,vehicle.available_layer) if fresh else old[first].departure_layer
    suffix=[a for a in old[first:] if a.arc_type=='road']
    serial=0

    def wait(end,tag):
        nonlocal layer,serial
        if end>layer:
            serial+=1
            route.append(SearchArc(f'{tag}:{vid}:{state.current_layer}:{serial}:{node}:{layer}',vid,node,node,layer,end,tag,0.,float(end-layer)))
            layer=end

    def stop():
        if node not in service_nodes:return
        earliest=max([layer]+[f.recovery_layer for f in flights if f.recovery_node==node])
        wait(earliest,'await_recovery')
        wait(layer+instance.recovery_service_layers+instance.preparation_layers,'service_wait')

    if fresh:stop()
    index=0
    while index<len(suffix):
        leg=suffix[index]
        if leg.tail!=node:return (),f'vehicle={vid}:discontinuous_reference'
        edge=edges[memory.edge_ids[leg.arc_id]]
        path=(edge,);next_index=index+1
        if edge.edge_id in closed or not edge.available:
            path=None
            for end_index in range(index,len(suffix)):
                skipped={a.head for a in suffix[index:end_index]}
                if skipped & mandatory:break
                candidate=earliest_path(instance,node,suffix[end_index].head,layer,closed,weather)
                if candidate is not None:
                    path=candidate;next_index=end_index+1;break
            if path is None:return (),f'vehicle={vid}:unreachable_reconnection'
        for e in path:
            end=edge_arrival(instance,e,layer,weather)
            if end is None:return (),f'vehicle={vid}:road_horizon_exceeded'
            serial+=1
            # Identity survives all subsequent replans, including travel on a new branch.
            arc_id=f'physical_road:{vid}:{state.current_layer}:{serial}:{e.edge_id}'
            memory.edge_ids[arc_id]=e.edge_id
            route.append(SearchArc(arc_id,vid,e.tail,e.head,layer,end,'road',e.geometry.length_3d,float(end-layer)))
            node,layer=e.head,end
            stop()
        index=next_index
    if node!='sink':return (),f'vehicle={vid}:return_endpoint_missing'
    if layer>HORIZON:return (),f'vehicle={vid}:service_horizon_exceeded'
    return tuple(route),None


def _visits(route,state,vehicle):
    """Return recovery and launch times at each occurrence of a physical stop."""
    visits=[]
    if not route:return visits
    node=vehicle.node_id
    arrival=state.current_layer
    rec=arrival
    ready=arrival
    for arc in route:
        if arc.arc_type=='road':
            if arc.departure_layer>=state.current_layer:
                visits.append((node,rec,ready))
            node=arc.head;arrival=arc.arrival_layer;rec=arrival;ready=arrival
        elif arc.arc_type=='await_recovery':
            rec=arc.arrival_layer;ready=rec
        else:
            ready=arc.arrival_layer
    visits.append((node,rec,ready))
    return visits


def build_rolling_problem(instance,state,memory,closures=(),*,allow_detour=True,allow_new=True):
    edge_ids={e.edge_id for e in instance.physical_scenario.road.edges}
    if any(c.edge_id not in edge_ids for c in closures):raise ValueError('Unknown physical closure')
    if any(c.reveal_layer>state.current_layer for c in closures):raise ValueError('Future closure information')
    closed={c.edge_id for c in closures}
    arcs=[];tuples=[];capacity={};errors=[];physical={};visits={};committed=set()
    capacity_value=min(instance.operation_capacity.values(),default=1)
    required={vid for vid,v in state.vehicles.items() if v.must_return or v.active_arc is not None}
    required.update(f.vehicle_id for f in state.airborne_services.values())
    for vid,vehicle in sorted(state.vehicles.items()):
        route,error=_physical_route(instance,state,vid,memory,closed,allow_detour)
        if error:
            if vid in required:errors.append(error)
            continue
        if not route:continue
        # An idle vehicle need not depart when its complete route exceeds its duty.
        duty=sum(a.duty_time for a in route)
        if vehicle.remaining_duty_time is not None and duty>vehicle.remaining_duty_time+1e-8:
            if vid in required:errors.append(f'vehicle={vid}:remaining_duty_exceeded')
            continue
        physical[vid]=route
        visits[vid]=_visits(route,state,vehicle)
        first=route[0]
        if vehicle.active_arc is not None:
            if first.arc_id!=vehicle.active_arc_id:errors.append(f'vehicle={vid}:commitment_not_first');continue
            arcs.append(replace(first,tail='source',head=model_node(first.head),departure_layer=state.current_layer,arc_type='committed',secondary_cost=0.))
            committed.add(first.arc_id)
            rest=route[1:]
        else:
            arcs.append(SearchArc(f'anchor:{vid}:{state.current_layer}',vid,'source',model_node(first.tail),state.current_layer,first.departure_layer,'anchor'))
            rest=route
        arcs.extend(replace(a,tail=model_node(a.tail),head=model_node(a.head)) for a in rest)
        last=route[-1]
        arcs.append(SearchArc(f'terminal:{vid}:{last.arrival_layer}',vid,model_node('sink'),'sink',last.arrival_layer,last.arrival_layer,'terminal'))
    aircraft={u.uav_id:u for u in instance.physical_scenario.uavs}
    admitted_origins={t.physical_candidate_id for t in instance.search_state.tuples if t.admitted}
    risk={int(t.tower_id):t.risk for t in instance.physical_scenario.power.towers}
    for candidate in instance.physical_scenario.candidates:
        vid=candidate.vehicle_id;uid=candidate.uav_id;uav=state.uavs[uid]
        tasks=tuple(int(t) for t in candidate.task_ids)
        if (not allow_new or vid not in physical or uav.is_airborne or
            not set(tasks)<=state.remaining_tasks or candidate.candidate_id not in admitted_origins):continue
        for li,(launch_node,_,launch) in enumerate(visits[vid]):
            if launch_node!=candidate.launch_stop_id or launch-instance.preparation_layers<max(state.current_layer,uav.available_layer):continue
            for ri,(recovery_node,recovery,_) in enumerate(visits[vid]):
                if ri<=li or recovery_node!=candidate.recovery_stop_id:continue
                if recovery-launch+1e-9<candidate.duration_minutes:continue
                # Constant average-power surrogate, including additional flight waiting.
                share=(recovery-launch)/candidate.duration_minutes
                spec=aircraft[uid]
                certificate=build_energy_certificate(EnergyMoments(candidate.expected_energy*share,candidate.energy_std*share),
                    instance.energy_alpha,spec.battery_capacity,spec.reserve_ratio)
                if not certificate.admitted:continue
                intervals,ops=service_keys(instance,vid,uid,(model_node(launch_node),model_node(recovery_node)),launch,recovery)
                required_arcs=tuple(a.arc_id for a in physical[vid] if a.departure_layer>=launch and a.arrival_layer<=recovery)
                completion=tuple((int(t),launch+max(1,ceil(offset-1e-12))) for t,offset in candidate.completion_offsets_minutes)
                item=SearchTuple(f'copy:{candidate.candidate_id}:{launch}:{recovery}',vid,uid,tasks,
                    model_node(launch_node),model_node(recovery_node),launch,recovery,
                    required_arc_ids=required_arcs,interval_keys=intervals,operation_keys=ops,completion_layers=completion,
                    secondary_cost=candidate.duration_minutes,risk_exposure=sum(risk[t]*(1+.05*k) for t,k in completion),
                    expected_energy=certificate.mean,energy_std=certificate.std,reliable_energy=certificate.reliable_bound,
                    battery_capacity=spec.battery_capacity,reserve_ratio=spec.reserve_ratio,
                    admission_certificate=certificate.certificate_id,physical_candidate_id=candidate.candidate_id)
                tuples.append(item);capacity.update({k:capacity_value for k in ops})
                memory.flight_origins[item.tuple_id]=candidate.candidate_id
    residual_ids=set()
    if {u for u,v in state.uavs.items() if v.is_airborne}!=set(state.airborne_services):errors.append('airborne_state_missing')
    for uid,flight in state.airborne_services.items():
        vid=flight.vehicle_id
        if state.active_service_ids.get(uid)!=flight.tuple_id:errors.append(f'uav={uid}:flight_identity_mismatch');continue
        choices=[rec for node,rec,ready in visits.get(vid,()) if node==flight.recovery_node and rec>=flight.recovery_layer]
        if not choices:errors.append(f'uav={uid}:recovery_unreachable');continue
        recovery=min(choices)
        moments=EnergyMoments(*flight.remaining_moments(state.current_layer,recovery))
        try:
            residual=build_air_residual(state.uavs[uid],flight.recovery_node,recovery,moments,flight.battery_capacity,flight.reserve_ratio,instance.energy_alpha)
        except ValueError:
            errors.append(f'uav={uid}:unsafe_recovery_energy');continue
        intervals,ops=service_keys(instance,vid,uid,('',model_node(flight.recovery_node)),state.current_layer,recovery,True)
        tasks=tuple(t for t,k in flight.completion_layers if t in state.remaining_tasks)
        item=SearchTuple(f'residual:{flight.tuple_id}:{recovery}',vid,uid,tasks,'',model_node(flight.recovery_node),state.current_layer,recovery,
            required_arc_ids=tuple(a.arc_id for a in physical[vid] if a.arrival_layer<=recovery),interval_keys=intervals,
            operation_keys=ops,completion_layers=tuple((t,k) for t,k in flight.completion_layers if t in tasks),
            expected_energy=moments.mean,energy_std=moments.std,reliable_energy=residual.q_bound,
            battery_capacity=flight.battery_capacity,reserve_ratio=flight.reserve_ratio,is_air_residual=True,
            admission_certificate=residual.residual_id,physical_candidate_id=memory.flight_origins.get(flight.tuple_id,''))
        if not item.physical_candidate_id:errors.append(f'uav={uid}:physical_origin_missing')
        tuples.append(item);residual_ids.add(item.tuple_id);capacity.update({k:capacity_value for k in ops})
    selected_arcs=frozenset(a.arc_id for a in arcs if a.vehicle_id in required)
    search=SearchState(tuple(arcs),tuple(tuples),selected_arcs,frozenset(residual_ids))
    locked=replace(state,locked_arc_ids=frozenset(committed),locked_tuple_ids=frozenset(residual_ids))
    problem=DynamicProblem(tuple(sorted(state.remaining_tasks)),{t:instance.miss_penalty[t] for t in state.remaining_tasks},capacity,search,locked,
        'infeasible_rolling_roads' if errors else 'feasible',';'.join(errors))
    if not errors:
        check=validate_dynamic_plan(problem,search)
        if not check.ok:problem=replace(problem,safety_status='infeasible_rolling_roads',safety_reason='incumbent:'+str(check.first_violation))
    return problem,physical


def physical_entries(problem,result,routes):
    by_id={a.arc_id:a for route in routes.values() for a in route}
    output=[]
    for event in _plan_entries(problem,result):
        if isinstance(event,PlannedVehicleArcEvent):
            if event.arc_id not in by_id:continue
            a=by_id[event.arc_id]
            output.append(PlannedVehicleArcEvent(a.arc_id,a.vehicle_id,a.head,a.arrival_layer,a.duty_time,a.departure_layer,a.tail))
        elif isinstance(event,(PlannedRecoveryEvent,PlannedLaunchEvent)):
            output.append(replace(event,recovery_node=physical_node(event.recovery_node)))
        else:output.append(event)
    return tuple(output)


def run_rolling(instance,method,config,closures=(),*,allow_detour=True,continue_after_closure=True,
                layers=tuple(range(1,122,8)),initial_state=None,initial_memory=None):
    if not layers or tuple(sorted(set(layers)))!=tuple(layers):raise ValueError('Invalid decision grid')
    if any(c.reveal_layer>=layers[0] and c.reveal_layer not in layers for c in closures):raise ValueError('Closure outside decision grid')
    state=initial_state or instance.rolling_state
    memory=deepcopy(initial_memory) if initial_memory is not None else RoadMemory.initial(instance)
    records=[];pending=();failure=''
    for index,layer in enumerate(layers):
        if state.current_layer!=layer:raise ValueError('Decision grid and execution disagree')
        active=tuple(c for c in closures if c.reveal_layer<=layer)
        memory_before=deepcopy(memory)
        problem,routes=build_rolling_problem(instance,state,memory,active,allow_detour=allow_detour,
            allow_new=continue_after_closure or not active)
        record=dict(layer=layer,state=state,closures=active,problem=problem,routes=routes,
                    edge_ids=dict(memory.edge_ids),flight_origins=dict(memory.flight_origins),memory_before=memory_before)
        if problem.safety_status!='feasible':
            failure=problem.safety_reason;record['failure']=failure;records.append(record);break
        if problem.search_state.tuples or problem.search_state.selected_arc_ids:
            execution=run_method(method,state=problem.search_state,task_ids=problem.task_ids,miss_penalty=problem.miss_penalty,
                validation_state=problem.rolling_state,operation_capacity=problem.operation_capacity,config=config)
            result=execution.result
            selected=replace(problem.search_state,selected_arc_ids=frozenset(result.selected_arc_ids),selected_tuple_ids=frozenset(result.selected_tuple_ids))
            check=validate_dynamic_plan(problem,selected)
            record.update(result=result,config=execution.config)
            if not check.ok:
                failure='returned_plan:'+str(check.first_violation);record['failure']=failure;records.append(record);break
        else:selected=problem.search_state
        entries=physical_entries(problem,selected,routes)
        planned=record_plan(problem.rolling_state,problem.search_state.arcs,problem.search_state.tuples,selected.selected_arc_ids,selected.selected_tuple_ids)
        end=layers[index+1] if index+1<len(layers) else TERMINAL
        executed,pending=execute_due_events(planned,entries,end)
        state=advance_state(planned,executed)
        selected_vids={a.vehicle_id for a in problem.search_state.arcs if a.arc_id in selected.selected_arc_ids}
        pending_ids={e.arc_id for e in pending if isinstance(e,PlannedVehicleArcEvent)}
        for vid in selected_vids:memory.routes[vid]=tuple(a for a in routes[vid] if a.arc_id in pending_ids)
        record.update(selected=selected,entries=entries,planned_state=planned,execution=executed,pending=pending,state_after=state)
        records.append(record)
        if executed.safety_status!='feasible':failure=executed.safety_status;break
    return SimpleNamespace(state=state,pending=pending,records=records,failure=failure,feasible=not failure)
