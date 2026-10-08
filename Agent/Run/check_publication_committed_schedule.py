"""Independent selected-plan checks using interval and physical state arithmetic."""
from collections import Counter
from math import sqrt, isfinite


def check_plan(domain, plan, weights, reported=None):
    state=domain.context.state;instance=domain.instance;now=state.current_layer
    allowed={a.arc_id:a for a in domain.problem.search_state.arcs}
    ids=[a.arc_id for a in plan.arcs];cids=[c.tuple_id for c in plan.copies]
    if len(ids)!=len(set(ids)) or len(cids)!=len(set(cids)):raise ValueError('Duplicate selected decision')
    if any(allowed.get(a.arc_id)!=a for a in plan.arcs):raise ValueError('Road outside continuation domain')
    if domain.required_arc_ids-set(ids):raise ValueError('Changed locked or ongoing road/handling')
    for name,group in domain.groups.items():
        if len(set(group)&set(cids))!=1:raise ValueError('Missing or repeated service obligation: '+name)
    active={a.vehicle_id for a in plan.arcs}
    if domain.required_vehicle_ids-active:raise ValueError('Deployed vehicle return removed')
    for vid in active:
        arcs=[a for a in plan.arcs if a.vehicle_id==vid]
        by_tail={};sink_count=0
        for a in arcs:
            node=('source',-1) if a.tail=='source' else (a.tail,a.departure_layer)
            if node in by_tail:raise ValueError('Branching selected vehicle path')
            by_tail[node]=a
            sink_count+=int(a.head=='sink')
        node=('source',-1);visited=set()
        while node in by_tail:
            a=by_tail[node]
            if a.arc_id in visited:raise ValueError('Selected cycle')
            visited.add(a.arc_id)
            node=('sink',-1) if a.head=='sink' else (a.head,a.arrival_layer)
        if node!=('sink',-1) or sink_count!=1 or len(visited)!=len(arcs):raise ValueError('Discontinuous vehicle path')
        limit=state.vehicles[vid].remaining_duty_time
        if limit is not None and sum(a.duty_time for a in arcs)>limit+1e-7:raise ValueError('Remaining duty exceeded')
    waits={(a.vehicle_id,a.tail,k) for a in plan.arcs if a.tail==a.head
           for k in range(a.departure_layer,a.arrival_layer)}
    busy=Counter(domain.fixed_busy);ops=Counter(domain.fixed_operations);tasks=Counter()
    specials={c.tuple_id:c for c in domain.specials}
    previous={c.tuple_id:c for c in domain.context.previous.copies}
    p=instance.preparation_layers;r=instance.recovery_service_layers;turn=instance.turnaround_layers
    cost=0.;energy_total=0.;risk_total=0.;completion=[]
    towers={int(t.tower_id):t for t in instance.physical_scenario.power.towers}
    for c in plan.copies:
        if c.uav_id not in state.uavs or c.vehicle_id!=state.uavs[c.uav_id].owner_vehicle_id:raise ValueError('UAV owner mismatch')
        if c.tuple_id in specials:
            if c!=specials[c.tuple_id]:raise ValueError('Changed special-copy fields')
        else:
            # This frozen independent checker recomputes geometry, keyed energy,
            # completion and full-service IDs, without using the new constructor.
            from Agent.Run.audit_publication_service_domain import (
                verify_copy,
                verify_declared_copy,
            )
            if domain.statistics.get("declared_candidates_read", False):
                verify_declared_copy(instance, c)
            else:
                verify_copy(instance,c)
            if c.launch_layer-p<now:raise ValueError('Uncommitted preparation starts in past')
            if set(c.task_ids)&state.done_tasks:raise ValueError('Completed task relaunched')
        if c.is_air_residual:
            f=state.airborne_services.get(c.uav_id)
            if f is None or c.launch_layer!=f.launch_layer or c.recovery_node!=f.recovery_node:
                raise ValueError('Airborne history or recovery node changed')
            if c.recovery_layer<f.recovery_layer:raise ValueError('Recovery advanced without a flight model')
            duration=f.recovery_layer-f.launch_layer
            mean=f.expected_energy*(c.recovery_layer-now)/duration
            std=f.energy_std*(c.recovery_layer-now)/duration
            bound=mean+sqrt(instance.energy_alpha/(1-instance.energy_alpha))*std
            if bound>state.uavs[c.uav_id].remaining_energy-f.reserve_ratio*f.battery_capacity+1e-7:
                raise ValueError('Residual energy or reserve violated')
            if abs(c.expected_energy/(c.recovery_layer-c.launch_layer)-f.expected_energy/duration)>1e-9:
                raise ValueError('Airborne energy rate reset')
        elif c.launch_layer-p<now:
            if previous.get(c.tuple_id)!=c:raise ValueError('Begun preparation changed')
        elif not state.uavs[c.uav_id].is_airborne and c.launch_layer-p<state.uavs[c.uav_id].available_layer:
            raise ValueError('Launch preparation before UAV ready')
        for t,k in c.completion_layers:
            if not c.launch_layer<=k<=c.recovery_layer:raise ValueError('Invalid inspection completion')
            if t in state.remaining_tasks:
                if k<=now and c.is_air_residual:raise ValueError('Executed task incorrectly remains pending')
                tasks[t]+=1;completion.append(k);risk_total+=towers[t].risk*(1+.05*k)
        busy.update((c.uav_id,k) for k in range(max(now,c.launch_layer-p),c.recovery_layer+r+turn))
        for node,lo,hi in ((c.launch_node,c.launch_layer-p,c.launch_layer),(c.recovery_node,c.recovery_layer,c.recovery_layer+r)):
            for k in range(max(now,lo),hi):ops[(c.vehicle_id,node,k)]+=1
        if c.launch_layer>=now and not c.is_air_residual:cost+=c.secondary_cost
        energy_total+=c.expected_energy*max(0,c.recovery_layer-max(now,c.launch_layer))/(c.recovery_layer-c.launch_layer)
    if any(v>1 for v in tasks.values()):raise ValueError('Repeated task assignment')
    if any(v>1 for v in busy.values()):raise ValueError('UAV preparation/recovery/turnaround overlap')
    capacities={v.vehicle_id:v.operation_capacity for v in instance.physical_scenario.vehicles}
    if any(v>capacities[key[0]] for key,v in ops.items()):raise ValueError('Carrier operation overlap')
    if set(ops)-waits:raise ValueError('Carrier absent during preparation or recovery')
    primary=sum(domain.problem.miss_penalty[t] for t in domain.problem.task_ids if tasks[t]==0)
    cost+=sum(a.secondary_cost for a in plan.arcs)+weights['time_weight']*sum(a.duty_time for a in plan.arcs)
    cost+=weights['risk_weight']*risk_total+weights['energy_weight']*energy_total
    returns=[a.arrival_layer for a in plan.arcs if a.head=='sink']
    cost+=weights['makespan_weight']*max(returns+completion,default=0)
    cancellable_arcs={a.arc_id for a in domain.context.previous.arcs if a.arc_type in ('road','wait') and a.departure_layer>=now}
    cancellable_copies={c.tuple_id for c in domain.context.previous.copies if c.launch_layer-p>=now}
    cost+=weights['plan_change_weight']*(len(cancellable_arcs-set(ids))+len(cancellable_copies-set(cids)))
    if reported is not None:
        if any(not isfinite(v) for v in reported) or abs(primary-reported[0])>1e-7 or abs(cost-reported[1])>1e-7:
            raise ValueError('Reported objective arithmetic mismatch')
    return dict(primary=float(primary),secondary=float(cost),covered_tasks=len(tasks),selected_arcs=len(ids),
                selected_copies=len(cids),airborne_recoveries=sum(c.is_air_residual for c in plan.copies))
