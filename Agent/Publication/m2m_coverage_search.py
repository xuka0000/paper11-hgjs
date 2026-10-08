"""Input-only quota/resource closure, common to every own representation control.

Bounds come from the training audit. The unchanged route decoder validates full
fixed-service requests; neither teacher trajectories nor outcome labels enter
inference. Beam8, expansion8,32 submitted candidates and the outer60s clock
remain fixed. A failed closure is diagnostic, never a feasibility certificate.
"""
from dataclasses import dataclass
from Agent.Publication.joint_service_actions import ActionProposal
from Agent.Publication.joint_service_search import iter_proposals, _diverse_best


@dataclass(frozen=True)
class CoverageLimits:
    additions: int
    releases: int
    target_groups: int
    beam: int = 8
    expansion: int = 8
    terminals: int = 32

    def __post_init__(self):
        if any(type(v) is not int or v<0 for v in (self.additions,self.releases,self.target_groups)):
            raise ValueError('Invalid training-derived caps')
        if self.additions<1 or self.target_groups<1 or (self.beam,self.expansion,self.terminals)!=(8,8,32):
            raise ValueError('Changed shared computational budget')


def _extend(action,kind,key):
    return ActionProposal(tuple(sorted(action.add+(key,))) if kind=='add' else action.add,
                          tuple(sorted(action.release+(key,))) if kind=='release' else action.release)


def close_action(context, policy, deadline, limits, seed):
    """Close duplicate, overloaded-resource and incomplete-target obligations.

    Every choice scores all eligible alternatives for the current obligation.
    Necessary release lower bounds come from the common input-only descriptor.
    No unsuccessful closure is sent to the physical route solver.
    """
    action=seed
    for _ in range(limits.additions+limits.releases):
        deadline.check();d=context.describe(action,limits)
        if d is None:return None,'filtered_partial'
        selected=(context.selected-set(action.release))|set(action.add)
        counts={t:sum(t in context.tasks[k] for k in selected) for t in {t for k in selected for t in context.tasks[k]}}
        duplicate={k for t,n in counts.items() if n>1 for k in context.incumbent_task_services[t]
                   if k not in action.release and k not in context.immutable}
        if duplicate:
            kind='release';choices=sorted(duplicate)
        elif d['overload']>1e-9:
            overloaded={rid for rid,r in context.view.resources.items()
                if r.fixed_load+sum(context.rows[k].get(rid,0.) for k in selected)>r.capacity+1e-9}
            kind='release';choices=sorted(k for k in context.selected-context.immutable-set(action.release)
                if any(context.rows[k].get(rid,0.)>0 for rid in overloaded))
        else:
            deficits=[]
            for gid in d['target_groups']:
                q=context.view.quotas[gid]
                done={t for k in selected for t in context.valid_tasks[k].get(gid,())}
                if len(done)<q.minimum:deficits.append((gid,done))
            if not deficits:return action,'closed'
            kind='add';choices=[k for k in context.ids if k not in selected
                and any(context.valid_tasks[k].get(gid,frozenset())-done for gid,done in deficits)]
        if (kind=='add' and len(action.add)>=limits.additions) or (kind=='release' and len(action.release)>=limits.releases):
            return None,'closure_cap_exhausted'
        actions=[];descriptors=[]
        for number,key in enumerate(choices):
            if number%64==0:deadline.check()
            extension=_extend(action,kind,key);descriptor=context.describe(extension,limits)
            if descriptor is not None:actions.append(extension);descriptors.append(descriptor)
        if not actions:return None,'closure_no_eligible_extension'
        scores=policy.score(context,action,actions,descriptors);deadline.check()
        if len(scores)!=len(actions):raise ValueError('Closure policy returned wrong score count')
        from math import isfinite
        if any(not isfinite(float(s)) for s in scores):raise ValueError('Nonfinite closure policy score')
        action=max(zip(scores,actions),key=lambda p:(float(p[0]),p[1].add,p[1].release))[1]
    d=context.describe(action,limits)
    if d is not None and d['terminal']:
        selected=(context.selected-set(action.release))|set(action.add)
        if all(len({t for k in selected for t in context.valid_tasks[k].get(g,())})>=context.view.quotas[g].minimum
               for g in d['target_groups']):return action,'closed'
    return None,'closure_depth_exhausted'


def coverage_proposals(context,policy,deadline,limits):
    """At most8 quota/resource closures, then residual sequential candidates."""
    seeds=[];descriptors=[];empty=ActionProposal()
    for number,key in enumerate(context.ids):
        if number%64==0:deadline.check()
        if key in context.selected:continue
        action=ActionProposal((key,),());d=context.describe(action,limits)
        if d is not None:seeds.append(action);descriptors.append(d)
    scores=policy.score(context,empty,seeds,descriptors);deadline.check()
    if len(scores)!=len(seeds):raise ValueError('Seed policy returned wrong score count')
    from math import isfinite
    if any(not isfinite(float(s)) for s in scores):raise ValueError('Nonfinite seed policy score')
    ranked=_diverse_best([(float(s),a,d) for s,a,d in zip(scores,seeds,descriptors)],limits.beam,context)
    seen=set();count=0
    for _,seed,_ in ranked:
        action,status=close_action(context,policy,deadline,limits,seed)
        if action is None or action in seen:continue
        descriptor=context.describe(action,limits)
        if descriptor is None or not descriptor['terminal']:raise RuntimeError('Invalid closure terminal')
        score=policy.score(context,empty,[action],[descriptor])[0];deadline.check()
        seen.add(action);count+=1
        yield action,dict(descriptor,policy_score=float(score),depth=len(action.add)+len(action.release),
                          catalogue_size=len(context.ids),generator='quota_resource_closure',closure=status)
    # Keep the original search as a common fallback; its emitted shallow
    # terminals cannot consume slots reserved for the closure candidates above.
    for action,descriptor in iter_proposals(context,policy,deadline,limits):
        if count>=limits.terminals:return
        if action in seen:continue
        seen.add(action);count+=1
        yield action,dict(descriptor,generator='sequential')
