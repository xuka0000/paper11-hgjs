"""Physical road-time alternatives with the SAME corrected service copies.

This isolated domain extension changes carrier route choice only. It supports
uncommitted static/development states; rolling commitments must be implemented
explicitly before this constructor can be used in a dynamic experiment.
"""
from collections import defaultdict, deque
from dataclasses import dataclass,replace
from math import floor

from Agent.Run.publication_rolling_roads import model_node,physical_node,HORIZON
from Agent.Run.publication_road_recovery import edge_arrival
from Agent.Tool.Benchmark.benchmark_v3 import _formal_v3_weather
from Mask.ServiceTuple import SearchArc,SearchState


@dataclass(frozen=True)
class RouteChoiceDomain:
    problem: object
    fixed_problem: object
    arc_expansion: dict
    road_origins: dict
    boundary: str = 'Same finite service copies and swap assumptions; full physical road/time choices. Uncommitted states only. Not a full rolling-model validation.'


def require_uncommitted(instance,problem):
    if instance.split not in ('train','validation'):
        raise ValueError('Development instances only')
    state=problem.rolling_state
    if (state.locked_arc_ids or state.locked_tuple_ids or state.airborne_services or
        state.airborne_tasks or state.active_service_ids or state.previous_arc_values or state.previous_tuple_values):
        raise ValueError('Commitments and plan correspondence require a separate rolling implementation')
    if any(v.active_arc is not None or v.must_return for v in state.vehicles.values()):
        raise ValueError('In-transit and return-obligated states are not supported by this static experiment')
    if any(u.is_airborne for u in state.uavs.values()):
        raise ValueError('Airborne states cannot be reset to a static route-choice problem')
    if problem.search_state.selected_arc_ids or problem.search_state.selected_tuple_ids:
        raise ValueError('Construct the domain from the shared empty incumbent')


def _reachable(start,adjacency):
    found={start};queue=deque([start])
    while queue:
        key=queue.popleft()
        for other in adjacency.get(key,()):
            if other not in found:found.add(other);queue.append(other)
    return found


def time_node(arc,tail):
    if tail and arc.tail=='source':return ('__source__',-1)
    if not tail and arc.head=='sink':return ('__sink__',-1)
    return (arc.tail,arc.departure_layer) if tail else (arc.head,arc.arrival_layer)


def build_route_choice_domain(instance,fixed_problem,*,omit_operation_waits=False):
    require_uncommitted(instance,fixed_problem)
    rolling=fixed_problem.rolling_state
    physical=instance.physical_scenario
    weather=_formal_v3_weather(instance.seed,HORIZON)
    nodes=sorted({n for edge in physical.road.edges for n in (edge.tail,edge.head)})
    arcs=[];origins={}
    for vid,vehicle in sorted(rolling.vehicles.items()):
        start=max(rolling.current_layer,vehicle.available_layer)
        end=HORIZON
        if vehicle.remaining_duty_time is not None:
            end=min(end,start+floor(vehicle.remaining_duty_time+1e-9))
        current=[SearchArc(f'route_anchor:{vid}:{start}',vid,'source',model_node(vehicle.node_id),
            rolling.current_layer,start,'anchor')]
        for layer in range(start,end+1):
            current.append(SearchArc(f'route_terminal:{vid}:{layer}',vid,model_node('sink'),'sink',layer,layer,'terminal'))
            if layer==end:continue
            for node in nodes:
                current.append(SearchArc(f'route_wait:{vid}:{node}:{layer}',vid,model_node(node),model_node(node),
                    layer,layer+1,'wait',0.,1.))
            for edge in physical.road.edges:
                arrival=edge_arrival(instance,edge,layer,weather)
                if arrival is None or arrival>end:continue
                aid=f'route_road:{vid}:{edge.edge_id}:{layer}:{arrival}'
                current.append(SearchArc(aid,vid,model_node(edge.tail),model_node(edge.head),layer,arrival,
                    'road',edge.geometry.length_3d,float(arrival-layer)))
                origins[aid]=edge.edge_id
        forward=defaultdict(list);backward=defaultdict(list)
        for arc in current:
            a,b=time_node(arc,True),time_node(arc,False)
            forward[a].append(b);backward[b].append(a)
        reachable=_reachable(('__source__',-1),forward)
        recoverable=_reachable(('__sink__',-1),backward)
        # Exact reachability pruning retains every source-to-return path.
        arcs.extend(a for a in current if time_node(a,True) in reachable and time_node(a,False) in recoverable)
    arc_by_id={a.arc_id:a for a in arcs}
    origins={key:value for key,value in origins.items() if key in arc_by_id}
    by_geometry=defaultdict(list)
    for arc in arcs:
        by_geometry[(arc.vehicle_id,arc.tail,arc.head,arc.departure_layer,arc.arrival_layer)].append(arc)
    expansion={}
    for old in fixed_problem.search_state.arcs:
        if old.tail==old.head:
            mapping=tuple(f'route_wait:{old.vehicle_id}:{physical_node(old.tail)}:{k}' for k in range(old.departure_layer,old.arrival_layer))
        else:
            matches=by_geometry[(old.vehicle_id,old.tail,old.head,old.departure_layer,old.arrival_layer)]
            matches=[a for a in matches if abs(a.secondary_cost-old.secondary_cost)<1e-8 and abs(a.duty_time-old.duty_time)<1e-8]
            if len(matches)!=1:raise ValueError('Old arc has no unique physical embedding: '+old.arc_id)
            mapping=(matches[0].arc_id,)
        if set(mapping)-arc_by_id.keys():raise ValueError('Old waiting interval not contained in new domain')
        if abs(sum(arc_by_id[key].secondary_cost for key in mapping)-old.secondary_cost)>1e-8:
            raise ValueError('Arc embedding changed operating cost')
        if abs(sum(arc_by_id[key].duty_time for key in mapping)-old.duty_time)>1e-8:
            raise ValueError('Arc embedding changed duty')
        expansion[old.arc_id]=mapping
    copies=[]
    for item in fixed_problem.search_state.tuples:
        if item.is_air_residual:raise ValueError('Residual services require dynamic construction')
        waits=tuple(f'route_wait:{item.vehicle_id}:{physical_node(node)}:{k}'
            for node,start,end in ((item.launch_node,item.launch_layer-instance.preparation_layers,item.launch_layer),
                                   (item.recovery_node,item.recovery_layer,item.recovery_layer+instance.recovery_service_layers))
            for k in range(start,end))
        if set(waits)-arc_by_id.keys():raise ValueError('Service handling cannot be embedded without dropping a fixed copy')
        copies.append(replace(item,required_arc_ids=() if omit_operation_waits else waits))
    search=SearchState(tuple(arcs),tuple(copies))
    return RouteChoiceDomain(replace(fixed_problem,search_state=search),fixed_problem,expansion,origins)


def embed_schedule(domain,snapshot):
    return dict(snapshot,selected_arc_ids=sorted({new for old in snapshot['selected_arc_ids'] for new in domain.arc_expansion[old]}))


def structural_summary(domain):
    grouped=defaultdict(list)
    for arc in domain.problem.search_state.arcs:grouped[arc.vehicle_id].append(arc)
    rows=[]
    for vehicle,arcs in sorted(grouped.items()):
        outgoing=defaultdict(list)
        for arc in arcs:outgoing[time_node(arc,True)].append(arc)
        # Different road heads, rather than wait/depart alternatives, certify
        # the presence of spatial route decisions in the retained network.
        spatial=sum(len({a.head for a in values if a.arc_type=='road'})>1 for values in outgoing.values())
        rows.append(dict(vehicle_id=vehicle,arcs=len(arcs),spatial_branch_time_nodes=spatial,
            decision_time_nodes=sum(len(values)>1 for values in outgoing.values())))
    return rows
