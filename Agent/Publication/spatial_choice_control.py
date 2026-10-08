"""Static physical-edge choice control; absolute traversal times remain free."""
from collections import defaultdict
from dataclasses import replace
from heapq import heappop, heappush


def road_key(arc):
    if arc.arc_type != 'road':
        raise ValueError('Physical edge requested for a non-road arc')
    return arc.arc_id.rsplit(':',2)[0]


def shortest_physical_path(arcs, vid):
    edges={}
    for a in arcs:
        if a.vehicle_id==vid and a.arc_type=='road':
            edges.setdefault(road_key(a),(a.tail,a.head,a.secondary_cost))
    starts=sorted({a.head for a in arcs if a.vehicle_id==vid and a.arc_type=='anchor'})
    ends={a.tail for a in arcs if a.vehicle_id==vid and a.arc_type=='terminal'}
    if len(starts)!=1 or not ends:
        raise ValueError('Expected one initial carrier location and at least one terminal')
    adjacency=defaultdict(list)
    for key,(u,v,cost) in edges.items():
        if cost<0:raise ValueError('Nonnegative physical road length required')
        adjacency[u].append((v,cost,key))
    pending=[(0.,(),starts[0])];seen=set()
    while pending:
        cost,path,node=heappop(pending)
        if node in seen:continue
        seen.add(node)
        if node in ends:return frozenset(path)
        for target,weight,key in sorted(adjacency[node]):
            if target not in seen:heappush(pending,(cost+weight,path+(key,),target))
    raise ValueError('No initial-to-terminal physical path')


def fixed_physical_edges(problem):
    domain=problem.domain
    if domain.context.state.epoch or domain.specials or domain.groups:
        raise ValueError('Initial-state spatial control only; not rolling execution')
    selected=problem.state.selected_arc_ids
    edge_sets={}
    for vid in domain.context.state.vehicles:
        chosen={road_key(a) for a in problem.state.arcs
                if a.vehicle_id==vid and a.arc_type=='road' and a.arc_id in selected}
        edge_sets[vid]=frozenset(chosen) if chosen else shortest_physical_path(problem.state.arcs,vid)
    arcs=tuple(a for a in problem.state.arcs
               if a.arc_type!='road' or road_key(a) in edge_sets[a.vehicle_id])
    assert selected <= {a.arc_id for a in arcs}
    dynamic=replace(domain.problem,search_state=replace(domain.problem.search_state,arcs=arcs))
    constrained=replace(domain,services=replace(domain.services,problem=dynamic))
    result=replace(problem,domain=constrained,state=replace(problem.state,arcs=arcs))
    assert result.full_copies==problem.full_copies
    return result,dict(physical_edges={str(k):sorted(v) for k,v in edge_sets.items()},
        full_arc_count=len(problem.state.arcs),retained_arc_count=len(arcs),
        candidate_copies_unchanged=len(problem.full_copies),
        service_selection_free=True,time_assignment_free=True,
        edge_traversal_order_fixed=False,physical_constraints_removed=False,
        unused_carrier_rule='Deterministic shortest physical road-length path to terminal',
        interpretation='Restricts physical edge availability, not time-specific traversal or service assignments.')
