"""Zhou paper LNS adapter with an explicit source stopping recipe.

The author GCN, softmax fixed-size group sampling and existing physical MILP
interface are retained. Validation uses the paper's ten iterations; timed
formal comparisons have no undocumented thirty-iteration ceiling. This
adapter is separate from the healthy, frozen development execution.
"""
from dataclasses import replace
import torch
from Agent.Publication.budget import BudgetExpired
from Agent.Publication.zhou2023_learning import build_formulation_graph,sample_groups
from Agent.Publication.zhou2023_learning_search import repair_once
from Agent.Publication.zhou2023_vehicle_random import accept_lexicographic
from Agent.Publication.component_objective import evaluate_plan


def run_source_lns(problem,weights,model,seed,deadline,*,task_groups,validation=False):
    current=best=problem.state
    check=evaluate_plan(problem.domain,problem.plan(current),weights,task_groups)
    current_value=best_value=(float(check['primary']),float(check['secondary']))
    generator=torch.Generator().manual_seed(seed);trace=[];iteration=0
    try:
        graph=build_formulation_graph(problem,weights,task_groups=task_groups,deadline=deadline)
        model.eval()
        while not validation or iteration<10:
            if deadline.remaining()<4.3:break
            with torch.no_grad():scores=model(*graph.tensors(current,best))
            selected=sample_groups(scores,graph.group_ids,.4,generator);deadline.check()
            if deadline.remaining()<4.3:break
            candidate,repair=repair_once(problem,weights,current,selected,seconds=4.,gap=.1,task_groups=task_groups)
            value=tuple(repair['objective'])
            accepted=(repair['candidate_returned'] and repair['within_budget']
                      and deadline.remaining()>.3 and accept_lexicographic(value,current_value))
            if accepted:
                current,current_value=candidate,value
                if current_value<best_value:best,best_value=current,current_value
            trace.append({'iteration':iteration,'selected_groups':selected,'accepted':accepted,
                          'current_objective':current_value,'best_objective':best_value,**repair})
            iteration+=1
    except BudgetExpired:pass
    result=evaluate_plan(problem.domain,problem.plan(best),weights,task_groups)
    if any(abs(float(result[k])-v)>1e-6 for k,v in zip(('primary','secondary'),best_value)):
        raise ValueError('Source-adapted LNS independent objective mismatch')
    return {'method_id':'zhou2023_author_gcn_problem_adapted_lns','objective':best_value,'trace':trace,
            'selected_arc_ids':sorted(best.selected_arc_ids),'selected_tuple_ids':sorted(best.selected_tuple_ids),
            'source_recipe':'Paper V-A validation ni=10; V-B test equal total runtime; SGD/GCN and training unchanged',
            'acceptance_recipe':'Paper V-A1percent, extended lexicographically to component objective; source eval.py unconditional acceptance differs and is not silently claimed identical',
            'adaptation':'Existing two-carrier MILP interface and4second repair cap; max(1,floor(.4*group_count)) avoids empty selection in a two-carrier problem',
            'total_action_seconds':deadline.elapsed(),'within_common_budget':deadline.elapsed()<=deadline.seconds}
