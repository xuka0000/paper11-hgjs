"""Physical teacher repairs and learned vehicle-group LNS with one action clock."""
from dataclasses import replace
from types import SimpleNamespace as NS
import torch
from Agent.Publication.budget import Deadline,BudgetExpired
from Agent.Publication.physical_search_baselines import _pool_domain
from Agent.Publication.vehicle_lns_milp import solve_committed
from Agent.Publication.zhou2023_vehicle_random import accept_lexicographic,vehicle_signature
from Agent.Publication.zhou2023_learning import build_formulation_graph,sample_groups


def _check(problem,state,weights,task_groups):
    if task_groups is None:return problem.check(state,weights)
    from Agent.Publication.component_objective import evaluate_plan
    return evaluate_plan(problem.domain,problem.plan(state),weights,task_groups)


def repair_once(problem,weights,state,groups,seconds=4.,gap=.1,*,task_groups=None):
    deadline=Deadline(seconds)
    before=_check(problem,state,weights,task_groups)
    result={'objective':[before['primary'],before['secondary']], 'physically_valid':True,
            'within_budget':True,'candidate_returned':False,'selected_groups':list(groups),'solver_stages':[]}
    next_state=state
    try:
        domain=_pool_domain(problem.domain,problem.full_copies.values())
        raw=solve_committed(domain,NS(objective_weights=weights),
            incumbent=problem.plan(state),released_vehicle_ids=groups,deadline=deadline,
            primary_seconds=seconds*2/3,secondary_seconds=seconds,
            relative_mip_gap=gap,validation_reserve_s=.3,task_groups=task_groups)
        result['solver_stages']=raw['stages']
        candidate=raw['candidate']
        if candidate is not None:
            plan=candidate['plan'];incumbent=problem.plan(state)
            for vehicle in set(problem.domain.context.state.vehicles)-set(groups):
                if vehicle_signature(plan,vehicle)!=vehicle_signature(incumbent,vehicle):
                    raise ValueError('Learned repair changed a nonselected vehicle')
            next_state=replace(state,selected_arc_ids=frozenset(a.arc_id for a in plan.arcs),
                selected_tuple_ids=frozenset(c.tuple_id for c in plan.copies))
            checked=_check(problem,next_state,weights,task_groups)
            value=[float(checked['primary']),float(checked['secondary'])]
            if any(abs(a-b)>1e-7 for a,b in zip(value,[candidate['primary'],candidate['secondary']])):
                raise ValueError('Teacher repair objective mismatch')
            deadline.check()
            result.update(objective=value,candidate_returned=True)
    except BudgetExpired:
        next_state=state
        result.update(within_budget=False,candidate_returned=False)
    result['elapsed_seconds']=deadline.elapsed()
    result['selected_arc_ids']=sorted(next_state.selected_arc_ids)
    result['selected_tuple_ids']=sorted(next_state.selected_tuple_ids)
    return next_state,result


def run_learned(problem,weights,model,seed,seconds=15.,*,task_groups=None):
    deadline=Deadline(seconds);current=best=problem.state
    check=_check(problem,current,weights,task_groups)
    current_value=best_value=(float(check['primary']),float(check['secondary']))
    generator=torch.Generator().manual_seed(seed)
    trace=[];graph=None;graph_seconds=0.
    try:
        graph=build_formulation_graph(problem,weights,task_groups=task_groups,deadline=deadline);graph_seconds=deadline.elapsed()
        model.eval()
        for iteration in range(30):
            if deadline.remaining()-.3<4.:break
            tensors=graph.tensors(current,best)
            with torch.no_grad():scores=model(*tensors)
            groups=sample_groups(scores,graph.group_ids,.4,generator)
            deadline.check()
            if deadline.remaining()-.3<4.:break
            proposal,repair=repair_once(problem,weights,current,groups,seconds=4.,task_groups=task_groups)
            candidate=tuple(repair['objective'])
            accepted=(repair['candidate_returned'] and repair['within_budget']
                      and deadline.remaining()>.3 and accept_lexicographic(candidate,current_value))
            before=current_value
            if accepted:
                current,current_value=proposal,candidate
                if current_value<best_value:best,best_value=current,current_value
            trace.append({'iteration':iteration,'before_objective':before,
                'logits':scores.detach().reshape(-1).tolist(),'accepted':accepted,
                'current_objective':current_value,'best_objective':best_value,**repair})
    except BudgetExpired:pass
    final=_check(problem,best,weights,task_groups)
    value=[float(final['primary']),float(final['secondary'])]
    if any(abs(a-b)>1e-7 for a,b in zip(value,best_value)):raise ValueError('Learned final objective mismatch')
    elapsed=deadline.elapsed()
    return {'method_id':'zhou2023_learning_lns','objective':value,'trace':trace,
        'metric':'weighted_incomplete_components' if task_groups is not None else 'weighted_missed_observations',
        'selected_arc_ids':sorted(best.selected_arc_ids),'selected_tuple_ids':sorted(best.selected_tuple_ids),
        'physical_check_passed':True,'total_action_seconds':elapsed,'within_common_budget':elapsed<=seconds,
        'graph_construction_seconds':graph_seconds,'trained_checkpoint_required_for_admission':True}
