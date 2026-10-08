"""Disclosed catalogue/road adaptation of the native first TSHA stage.

Retains the archived critical-road construction, including near clustering
and connection rules. The physical feasible set is NEVER replaced by the
native zero-handling/ground-waiting model. Native undirected geometry only
guides a subset of the existing directed time-expanded road arcs.
"""
from dataclasses import replace
from math import dist
from Agent.Publication.xia2025_tsha import InspectionInstance,construct_vehicle_route,RouteDecoder
from Agent.Publication.spatial_choice_control import road_key,shortest_physical_path


def native_guided_roads(problem,deadline):
    domain=problem.domain;scenario=domain.instance.physical_scenario
    if domain.context.state.epoch or domain.specials or domain.groups:
        raise ValueError('Static initial-state spatial adapter only')
    nodes={n.node_id:(n.location.x,n.location.y) for n in scenario.road.nodes}
    edges=tuple(sorted({tuple(sorted((e.tail,e.head))) for e in scenario.road.edges}))
    tasks={int(t.tower_id):(t.location.x,t.location.y) for t in scenario.power.towers}
    selected=problem.state.selected_arc_ids;kept={};records=[];positions={}
    for vid,vehicle in sorted(domain.context.state.vehicles.items()):
        deadline.check()
        copies=[c for c in problem.full_copies.values() if c.vehicle_id==vid]
        covered=sorted({t for c in copies for t in c.task_ids})
        incumbent_edges={road_key(a) for a in problem.state.arcs if a.vehicle_id==vid
                         and a.arc_type=='road' and a.arc_id in selected}
        connector=shortest_physical_path(problem.state.arcs,vid)
        allowed=set(incumbent_edges)|set(connector)
        if covered:
            # A geometry-only coverage envelope derived from admitted services;
            # this is NOT an inferred endurance or a new feasibility limit.
            radius=max(min(dist(tasks[t],nodes[c.launch_node.removeprefix('physical:')]),
                           dist(tasks[t],nodes[c.recovery_node.removeprefix('physical:')]))
                       for c in copies for t in c.task_ids)+1e-8
            native=InspectionInstance(nodes,edges,{str(t):tasks[t] for t in covered},vehicle.node_id,
                drone_count=1,endurance_seconds=2*radius+1,service_seconds=1,
                drone_speed_mps=1,vehicle_speed_mps=1)
            route,construction=construct_vehicle_route(native)
            route_edges={frozenset((u,v)) for u,v in zip(route,route[1:])}
            for arc in problem.state.arcs:
                if arc.vehicle_id==vid and arc.arc_type=='road':
                    ends=frozenset((arc.tail.removeprefix('physical:'),arc.head.removeprefix('physical:')))
                    if ends in route_edges:allowed.add(road_key(arc))
            guide=RouteDecoder(native,route)
            for task,pos in zip(guide.tasks,guide.positions):
                positions[int(task)]=min(positions.get(int(task),float('inf')),pos/max(guide.prefix[-1],1e-9))
            records.append(dict(vehicle_id=vid,native_route=route,construction=construction,
                geometry_coverage_radius=radius,source_constructor_called=True,
                covered_tasks=covered,initial_route_edges_retained=sorted(incumbent_edges),
                required_terminal_connector=sorted(connector)))
        kept[vid]=allowed
    deadline.check()
    arcs=tuple(a for a in problem.state.arcs if a.arc_type!='road' or road_key(a) in kept[a.vehicle_id])
    assert selected<={a.arc_id for a in arcs}
    dp=replace(domain.problem,search_state=replace(domain.problem.search_state,arcs=arcs))
    restricted=replace(domain,services=replace(domain.services,problem=dp))
    adapted=replace(problem,domain=restricted,state=replace(problem.state,arcs=arcs))
    return adapted,positions,dict(records=records,original_arcs=len(problem.state.arcs),retained_arcs=len(arcs),
        adaptation='Native geometric road selection plus shared incumbent edges and required terminal connection; feasible traversal times and all physical checks remain original. Road order and multiplicity are not fixed by the native walk.',
        native_physical_model_used=False,source_exact_reproduction=False)
