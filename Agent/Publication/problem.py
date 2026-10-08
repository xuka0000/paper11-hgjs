"""Translate one physical rolling state into the existing search interfaces.

Search features contain remaining work and resources. Complete immutable flight
records are retained separately for execution, energy checks and plan identity.
The bounded heuristic catalogue is explicit and reported; it is not described
as enumeration of the complete implicit service domain.
"""
from collections import defaultdict
from dataclasses import dataclass,replace
from types import SimpleNamespace as NS

from Agent.Run.publication_committed_service_domain import Plan,handling
from Agent.Run.publication_plan_distance import objective,plan_reference,service_key
from Agent.Run.check_publication_hamming_schedule import check_plan
from Mask.ServiceTuple import SearchState


def candidate_pool(domain,deadline,*,max_templates_per_task=12,time_samples=3,complete_limit=4096):
    if max_templates_per_task<1 or time_samples<1:raise ValueError('Positive catalogue limits required')
    deadline.check();chosen={c.tuple_id:c for c in domain.specials}
    count=domain.services.statistics['implicit_feasible_time_copies']
    complete=count<=complete_limit
    templates=list(domain.templates)
    if not complete:
        by_task=defaultdict(list)
        for t in templates:by_task[t.task_id].append(t)
        templates=[]
        for task,options in sorted(by_task.items()):
            # Balanced allocation over carriers precedes low-flight-time order.
            by_vehicle=defaultdict(list)
            for t in options:by_vehicle[t.vehicle_id].append(t)
            ordered={v:iter(sorted(ts,key=lambda t:(t.flight_minutes,t.min_duration,t.template_id)))
                     for v,ts in by_vehicle.items()}
            retained=[]
            while len(retained)<max_templates_per_task and ordered:
                for v in list(sorted(ordered)):
                    try:retained.append(next(ordered[v]))
                    except StopIteration:del ordered[v]
                    if len(retained)>=max_templates_per_task:break
            templates.extend(retained)
    for t in templates:
        deadline.check()
        if complete:launches=t.launches
        else:
            indices={round(i*(len(t.launches)-1)/max(1,time_samples-1)) for i in range(time_samples)}
            launches=tuple(t.launches[i] for i in sorted(indices))
        for launch in launches:
            recoveries=domain.recovery_options(t,launch)
            if not complete:recoveries=recoveries[:1]
            for recovery in recoveries:
                copy=domain.materialize(t,launch,recovery);chosen[copy.tuple_id]=copy
    deadline.check()
    return tuple(chosen.values()),dict(complete=complete,implicit_fresh_copies=count,
        retained_physical_templates=len(templates),explicit_pool_size=len(chosen),
        max_templates_per_task=max_templates_per_task,time_samples=time_samples,complete_limit=complete_limit)


@dataclass
class SearchProblem:
    domain: object
    state: object
    full_copies: dict
    validation_state: object
    catalogue: dict
    deadline: object

    def plan(self,state):
        arcs=tuple(a for a in state.arcs if a.arc_id in state.selected_arc_ids)
        try:copies=tuple(self.full_copies[k] for k in sorted(state.selected_tuple_ids))
        except KeyError as exc:raise ValueError('Selected copy absent from physical catalogue') from exc
        return Plan(arcs,copies)

    def check(self,state,weights):
        plan=self.plan(state)
        return check_plan(self.domain,plan,weights,objective(self.domain,plan,weights))


def make_search_problem(domain,incumbent,candidates,weights,deadline,*,catalogue=None):
    deadline.check();check_plan(domain,incumbent,weights)
    copies={c.tuple_id:c for c in candidates}
    for c in incumbent.copies:
        if c.tuple_id in copies and copies[c.tuple_id]!=c:raise ValueError('Conflicting physical service record')
        copies[c.tuple_id]=c
    for c in domain.specials:
        if c.tuple_id in copies and copies[c.tuple_id]!=c:raise ValueError('Changed service obligation')
        copies[c.tuple_id]=c
    state=domain.context.state;instance=domain.instance;now=state.current_layer
    towers={int(t.tower_id):t for t in instance.physical_scenario.power.towers}
    views=[]
    for c in copies.values():
        deadline.check()
        pending=tuple((t,k) for t,k in c.completion_layers if t in state.remaining_tasks)
        operations=handling(instance,c,now)
        # A fresh sampled copy is admitted by the physical constructor. Required
        # intervals still use the current graph; no obsolete corridor arcs.
        required=tuple(domain.waiting[key] for key in operations)
        view=replace(c,task_ids=tuple(t for t in c.task_ids if t in state.remaining_tasks),
            completion_layers=pending,required_arc_ids=required,
            interval_keys=tuple(f'uav:{c.uav_id}:{k}' for k in range(max(now,c.launch_layer-instance.preparation_layers),
                c.recovery_layer+instance.recovery_service_layers+instance.turnaround_layers)),
            operation_keys=tuple(f'op:{v}:{node}:{k}' for v,node,k in operations),
            secondary_cost=c.secondary_cost if c.launch_layer>=now and not c.is_air_residual else 0.,
            expected_energy=c.expected_energy*max(0,c.recovery_layer-max(now,c.launch_layer))/(c.recovery_layer-c.launch_layer),
            risk_exposure=sum(towers[t].risk*(1+.05*k) for t,k in pending))
        views.append(view)
    search=SearchState(domain.problem.search_state.arcs,tuple(views),
        frozenset(a.arc_id for a in incumbent.arcs),frozenset(c.tuple_id for c in incumbent.copies))
    reference=plan_reference(domain);previous_arcs={};previous_copies={}
    if reference.present:
        previous_arcs={k:1 for k in reference.old_arcs}
        previous_arcs.update({a.arc_id:int(a.arc_id in reference.old_arcs)
                             for a in search.arcs if reference.arc_adjustable(a)})
        matched=set()
        for c in copies.values():
            if reference.copy_adjustable(c):
                key=service_key(c);previous_copies[c.tuple_id]=int(key in reference.old_services);matched.add(key)
        for key in reference.old_services-matched:
            previous_copies['dropped_physical_service:'+repr(key)]=1
    locked_copies={c.tuple_id for c in copies.values() if service_key(c) in reference.fixed_services}
    locked_copies.update(next(iter(ids)) for name,ids in domain.groups.items() if len(ids)==1)
    validation=NS(**vars(state))
    validation.previous_arc_values=previous_arcs;validation.previous_tuple_values=previous_copies
    validation.locked_arc_ids=domain.required_arc_ids
    validation.locked_tuple_ids=frozenset(locked_copies)
    problem=SearchProblem(domain,search,copies,validation,dict(catalogue or {}),deadline)
    validation.publication_problem=problem
    # Compare the actual historical search arithmetic with the physical model.
    from Agent.Proposed.search import _objective,SearchConfig
    config=SearchConfig(1,1,1,**weights)
    old=_objective(search,domain.problem.task_ids,domain.problem.miss_penalty,config,previous_arcs,previous_copies)
    physical=objective(domain,incumbent,weights)
    if any(abs(a-b)>1e-7 for a,b in zip(old[:2],physical)):
        raise ValueError('Search view and physical objective differ')
    deadline.check()
    return problem
