"""Isolated preparation and closure corrections for publication diagnostics.

The frozen production generator and learned-policy experiments are unchanged.
"""
from dataclasses import replace
from Agent.Proposed.closure import close_region
from Mask.TupleFactory import ServiceResourceKeys


def preparation_resource_keys(vehicle_id,uav_id,launch_node,recovery_node,
                              launch,recovery,preparation,recovery_service,turnaround,
                              *,current_layer=0,available_layer=0,air_residual=False):
    if any(int(x)!=x or x<0 for x in (launch,recovery,preparation,recovery_service,turnaround,current_layer,available_layer)):
        raise ValueError('nonnegative integer layers required')
    start=launch if air_residual else launch-preparation
    if recovery<launch or start<current_layer or (not air_residual and start<available_layer):
        raise ValueError('preparation or flight precedes availability')
    launch_ops=() if air_residual else tuple(f'op:{vehicle_id}:{launch_node}:{k}' for k in range(start,launch))
    recovery_ops=tuple(f'op:{vehicle_id}:{recovery_node}:{k}' for k in range(recovery,recovery+recovery_service))
    intervals=tuple(f'uav:{uav_id}:{k}' for k in range(start,recovery+recovery_service+turnaround))
    return ServiceResourceKeys(intervals,launch_ops+recovery_ops)


def close_with_resource_propagation(state,core,budget):
    """Expand selected conflicts transitively through admitted tuple resources.

    This is an independently evaluated alternative, not the frozen controller's
    closure. Existing limits and path-anchor rules remain enforced.
    """
    nodes=set(core.node_ids)
    while True:
        closure=close_region(state,replace(core,node_ids=tuple(sorted(nodes))),budget)
        if not closure.accepted:return closure
        expanded=set(nodes)
        for item in state.tuples:
            if item.tuple_id in closure.tuple_ids:
                expanded.add(item.tuple_id)
                expanded.update(f'interval:{k}' for k in item.interval_keys)
                expanded.update(f'operation:{k}' for k in item.operation_keys)
        if expanded==nodes:return closure
        nodes=expanded


def fixed_boundary_violations(state,closure,rolling,selected_tuples,selected_arcs):
    """Independent post-solve checks omitted by the legacy schedule checker."""
    ty=set(selected_tuples);ax=set(selected_arcs);errors=[]
    tuples={t.tuple_id:t for t in state.tuples};arcs={a.arc_id:a for a in state.arcs}
    if ty-set(tuples) or ax-set(arcs):return ('unknown_decision',)
    for name,domain,selected,old,released,locked in (
        ('tuple',tuples,ty,state.selected_tuple_ids,set(closure.tuple_ids),rolling.locked_tuple_ids),
        ('arc',arcs,ax,state.selected_arc_ids,set(closure.arc_ids),rolling.locked_arc_ids)):
        if set(locked)-set(domain):errors.append('missing_locked_'+name)
        for key in domain:
            if (key not in released or key in locked) and ((key in selected)!=(key in old)):
                errors.append('fixed_'+name+':'+key)
    for vid,vehicle in rolling.vehicles.items():
        chosen=[arcs[a] for a in ax if arcs[a].vehicle_id==vid]
        if vehicle.must_return and not any(a.head=='sink' for a in chosen):errors.append('required_return:'+str(vid))
        if vehicle.remaining_duty_time is not None and sum(a.duty_time for a in chosen)>vehicle.remaining_duty_time+1e-8:
            errors.append('remaining_duty:'+str(vid))
    return tuple(errors)


def physical_operation_violations(item,arcs,preparation,recovery_service,turnaround,*,current_layer=0,available_layer=0):
    try:
        expected=preparation_resource_keys(item.vehicle_id,item.uav_id,item.launch_node,item.recovery_node,item.launch_layer,item.recovery_layer,
                                          preparation,recovery_service,turnaround,current_layer=current_layer,available_layer=available_layer,air_residual=item.is_air_residual)
    except ValueError:
        return ('preparation_or_flight_timing',)
    errors=[]
    if set(expected.interval_keys)!=set(item.interval_keys):errors.append('uav_preparation_occupancy')
    if set(expected.operation_keys)!=set(item.operation_keys):errors.append('carrier_operation_occupancy')
    windows=[] if item.is_air_residual else [(item.launch_node,item.launch_layer-preparation,item.launch_layer)]
    windows.append((item.recovery_node,item.recovery_layer,item.recovery_layer+recovery_service))
    for node,start,end in windows:
        for k in range(start,end):
            if not any(a.vehicle_id==item.vehicle_id and a.tail==a.head==node and a.departure_layer<=k<a.arrival_layer for a in arcs):
                errors.append(f'carrier_absent:{node}:{k}')
    return tuple(errors)
