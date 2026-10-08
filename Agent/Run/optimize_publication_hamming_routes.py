"""Exact conditional route cost with additions, cancellations and return changes."""
from bisect import bisect_left
from Agent.Run.publication_committed_service_domain import Plan, handling
from Agent.Run.publication_plan_distance import objective, plan_reference
from Agent.Run.check_publication_hamming_schedule import check_plan


def optimize_committed_routes(domain, copies, weights, *, deadline=None):
    def tick():
        if deadline is not None: deadline.check()
    tick()
    arcs=domain.problem.search_state.arcs;by_id={a.arc_id:a for a in arcs}
    required=set(domain.required_arc_ids)
    for c in copies:
        tick()
        for key in handling(domain.instance,c,domain.context.state.current_layer):
            if key not in domain.waiting:raise ValueError('Fixed service handling unavailable')
            required.add(domain.waiting[key])
    active=set(domain.required_vehicle_ids)|{c.vehicle_id for c in copies}|{by_id[k].vehicle_id for k in required}
    reference=plan_reference(domain)
    curves={};paths={}
    for vid in sorted(active):
        tick()
        mandatory={}
        for aid in required:
            a=by_id[aid]
            if a.vehicle_id!=vid:continue
            if a.departure_layer in mandatory and mandatory[a.departure_layer]!=aid:
                raise ValueError('Conflicting mandatory vehicle actions')
            mandatory[a.departure_layer]=aid
        times=sorted(mandatory);last=max((by_id[k].arrival_layer for k in mandatory.values()),default=0)
        distance={('source',-1):0.};previous={};returns={}
        ordered=sorted((a for a in arcs if a.vehicle_id==vid),key=lambda a:(a.departure_layer,a.arrival_layer,a.arc_id))
        for a in ordered:
            tick()
            tail=('source',-1) if a.tail=='source' else (a.tail,a.departure_layer)
            if tail not in distance:continue
            index=bisect_left(times,a.departure_layer)
            if a.arc_type=='terminal':
                if a.arrival_layer<last:continue
                head=('sink',a.arrival_layer)
            else:
                if index<len(times) and times[index]<a.arrival_layer and mandatory[times[index]]!=a.arc_id:continue
                head=(a.head,a.arrival_layer)
            cost=distance[tail]+a.secondary_cost+weights['time_weight']*a.duty_time
            cost+=weights['plan_change_weight']*reference.arc_coefficient(a)
            if head not in distance or cost<distance[head]-1e-10:
                distance[head]=cost;previous[head]=(tail,a.arc_id)
                if a.arc_type=='terminal':returns[a.arrival_layer]=cost
        if not returns:raise ValueError('No route for fixed continuation services')
        curves[vid]=returns;paths[vid]={}
        for end in returns:
            tick()
            node=('sink',end);selected=[]
            while node!=('source',-1):
                tick()
                node,aid=previous[node];selected.append(aid)
            if set(mandatory.values())-set(selected):raise ValueError('Route omitted a mandatory action')
            paths[vid][end]=tuple(reversed(selected))
    if not active:
        plan=Plan((),tuple(copies));values=objective(domain,plan,weights);check_plan(domain,plan,weights,values)
        tick()
        return plan,dict(primary=values[0],secondary=values[1],conditional_optimal=True)
    completion=max((k for c in copies for t,k in c.completion_layers if t in domain.context.state.remaining_tasks),default=0)
    best=None
    for cap in sorted({k for curve in curves.values() for k in curve if k>=completion}):
        tick()
        ends={}
        for vid,curve in curves.items():
            allowed=[k for k in curve if k<=cap]
            if not allowed:break
            ends[vid]=min(allowed,key=lambda k:(curve[k],k))
        if len(ends)!=len(active):continue
        maximum=max(completion,*ends.values())
        value=sum(curves[vid][k] for vid,k in ends.items())+weights['makespan_weight']*maximum
        item=(value,maximum,tuple(sorted(ends.items())))
        if best is None or item<best:best=item
    if best is None:raise ValueError('No common return-time cap')
    plan=Plan(tuple(by_id[k] for vid,end in best[2] for k in paths[vid][end]),tuple(copies))
    values=objective(domain,plan,weights);check_plan(domain,plan,weights,values)
    tick()
    return plan,dict(primary=values[0],secondary=values[1],conditional_optimal=True,
                     boundary='Service, recovery-time and task choices fixed. Exact route optimum, not joint secondary optimum.')
