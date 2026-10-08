"""Physical MILP repair for the existing constraint-closed search action."""
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace as NS

from Agent.Proposed.repair import RepairResult
from Agent.Proposed.validator import ValidationReport
from Agent.Run.publication_plan_distance import objective
from Agent.Run.check_publication_hamming_schedule import check_plan
from .budget import BudgetExpired,Deadline
from .milp import solve_committed
from .path_pruning import prune_forced_paths


def primary_stage_seconds(total_seconds, primary_time_fraction):
    if not 0.<primary_time_fraction<=1.:
        raise ValueError('Primary time fraction must lie in (0, 1]')
    return float(total_seconds)*float(primary_time_fraction)


def repair_region(state,closure,task_ids,miss_penalty,validation_state,operation_capacity,time_limit_s,
                  epsilon_u=1e-8,epsilon_f=1e-8,*,makespan_weight=0.,time_weight=0.,risk_weight=0.,
                  energy_weight=0.,plan_change_weight=0.,unsafe_tuple_ids=frozenset(),
                  solver_backend='SCIP',node_limit=None,validation_reserve_s=.002,
                  primary_time_fraction=.5,task_groups=None):
    problem=validation_state.publication_problem;domain=problem.domain;tick=perf_counter()
    weights=dict(makespan_weight=makespan_weight,time_weight=time_weight,risk_weight=risk_weight,
                 energy_weight=energy_weight,plan_change_weight=plan_change_weight)
    def account(plan):
        value=objective(domain,plan,weights)
        return value if task_groups is None else (task_groups.evaluate(plan)['primary'],value[1])
    current=problem.plan(state);before=account(current)
    def rejected(reason,raw=None):
        hint=(raw or {}).get('incumbent_hint',{})
        return RepairResult(False,reason,tuple(sorted(state.selected_tuple_ids)),tuple(sorted(state.selected_arc_ids)),
            before[0],before[1],ValidationReport(False,reason,0.,None),
            raw['stages'][0]['status'] if raw and raw['stages'] else 'not_solved',
            raw['stages'][-1]['status'] if raw and len(raw['stages'])>1 else 'not_solved',
            runtime_s=perf_counter()-tick,
            solver_nodes=max((stage.get('solver_nodes',0) for stage in raw['stages']),default=0)
            if raw else 0,
            incumbent_hint_accepted=hint.get('accepted'),
            incumbent_hint_violation_count=int(hint.get('violation_count',0)))
    if not closure.accepted:return rejected(closure.reason)
    if str(solver_backend).upper()!='SCIP':raise ValueError('Corrected physical repair currently supports SCIP only')
    primary_stage_seconds(1.,primary_time_fraction)
    if unsafe_tuple_ids:raise ValueError('Unsafe obligations require explicit infeasibility, not silent unlocking')
    if set(task_ids)!=set(domain.problem.task_ids) or dict(miss_penalty)!=domain.problem.miss_penalty:
        raise ValueError('Repair task/objective domain mismatch')
    raw=None
    try:
        seconds=problem.deadline.allowance(time_limit_s)
        if seconds<=0:raise BudgetExpired()
        # Nested allocation ends no later than the overall deadline.
        deadline=Deadline(seconds)
        released_arcs=set(closure.arc_ids);released_copies=set(closure.tuple_ids)
        arc_ids=released_arcs|set(state.selected_arc_ids)
        arcs=tuple(a for a in state.arcs if a.arc_id in arc_ids)
        required=frozenset(set(domain.required_arc_ids)|(set(state.selected_arc_ids)-released_arcs))
        if required-arc_ids:return rejected('immutable_arc_unavailable')
        arcs=prune_forced_paths(arcs,required,deadline)
        waiting={(a.vehicle_id,a.tail,a.departure_layer):a.arc_id for a in arcs if a.arc_type=='wait'}
        allowed_ids=released_copies|set(state.selected_tuple_ids)
        copies=[]
        from Agent.Run.publication_committed_service_domain import handling
        for key in sorted(allowed_ids):
            deadline.check()
            c=problem.full_copies[key]
            if all(k in waiting for k in handling(domain.instance,c,domain.context.state.current_layer)):
                copies.append(c)
        present={c.tuple_id for c in copies}
        groups={key:tuple(cid for cid in ids if cid in present) for key,ids in domain.groups.items()}
        if any(not ids for ids in groups.values()):return rejected('service_obligation_outside_released_route')
        for key in set(state.selected_tuple_ids)-released_copies:groups['fixed_search:'+key]=(key,)
        if any(key not in present for ids in groups.values() for key in ids):return rejected('immutable_copy_unavailable')
        local=NS(instance=domain.instance,context=domain.context,
            problem=replace(domain.problem,search_state=replace(domain.problem.search_state,arcs=arcs)),
            specials=tuple(copies),groups=groups,templates=(),waiting=waiting,
            required_arc_ids=required,required_vehicle_ids=domain.required_vehicle_ids,
            fixed_busy=domain.fixed_busy,fixed_operations=domain.fixed_operations,
            previous_copies=domain.previous_copies)
        # Search-fixed variables are constraints, not physical commitments.
        # The plan-distance reference must still use the original lock boundary.
        local.objective_reference_domain=domain
        raw=solve_committed(local,NS(objective_weights=weights),
                            primary_seconds=primary_stage_seconds(seconds,primary_time_fraction),
                            secondary_seconds=seconds,deadline=deadline,node_limit=node_limit,incumbent=current,
                            validation_reserve_s=validation_reserve_s,task_groups=task_groups)
        # Assembly can consume the allowance before the first solve. The MILP
        # then returns its validated incumbent with no stage records. Preserve
        # the current plan without inventing solver status or accepting a hint
        # as an optimized repair.
        if not raw['stages']:
            return rejected('repair_not_started_before_deadline',raw)
        if raw['selected'] is None:return rejected('repair_'+raw['stages'][-1]['status'],raw)
        chosen=raw['selected']['plan']
        after=account(chosen)
        # Validate against the complete physical domain. The local pooled
        # specials are not allowed to bypass fresh-service admission checks.
        checked=check_plan(domain,chosen,weights,objective(domain,chosen,weights))
        if any(abs(a-b)>1e-7 for a,b in zip(after,(raw['selected']['primary'],raw['selected']['secondary']))):
            raise ValueError('Local repair changed the global objective')
        deadline.check();problem.deadline.check()
        if after[0]>before[0]+epsilon_u or (abs(after[0]-before[0])<=epsilon_u and after[1]>before[1]+epsilon_f):
            return rejected('worsened',raw)
        new_a={a.arc_id for a in chosen.arcs};new_c={c.tuple_id for c in chosen.copies}
        if (new_a^set(state.selected_arc_ids))-released_arcs or (new_c^set(state.selected_tuple_ids))-released_copies:
            raise ValueError('Repair changed decisions outside the released closure')
        stages=raw['stages']
        return RepairResult(True,'repaired',tuple(sorted(new_c)),tuple(sorted(new_a)),after[0],after[1],
            ValidationReport(True,None,0.,None),stages[0]['status'],stages[-1]['status'] if len(stages)>1 else 'not_solved',
            primary_bound=stages[0]['bound'] if stages[0]['bound'] is not None else float('-inf'),
            secondary_bound=stages[-1]['bound'] if len(stages)>1 and stages[-1]['bound'] is not None else float('-inf'),
            runtime_s=perf_counter()-tick,primary_proven_optimal=stages[0]['status']=='optimal',
            solver_nodes=max((stage.get('solver_nodes',0) for stage in stages),default=0),
            incumbent_hint_accepted=raw['incumbent_hint'].get('accepted'),
            incumbent_hint_violation_count=int(
                raw['incumbent_hint'].get('violation_count',0)))
    except BudgetExpired:return rejected('complete_budget_exhausted',raw)
