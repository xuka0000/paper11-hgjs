"""Quota-bound pinned Zhou GCN/LNS comparator; no outcome-driven adaptation."""
from dataclasses import replace
from hashlib import sha256
import pickle

import torch

from Agent.Publication.component_objective import evaluate_plan
from Agent.Publication.foundation_benchmark import WEIGHTS
from Agent.Publication.zhou2023_learning import MODEL_SHA, MODEL_SOURCE, new_policy
from Agent.Publication.zhou2023_learning_search import run_learned
from Agent.Run.publication_committed_service_domain import Plan


METHOD = 'public_learned_lns'


def verify_pinned_model():
    actual=sha256(MODEL_SOURCE.read_bytes()).hexdigest()
    if actual!=MODEL_SHA:raise ValueError('Pinned public GCN source changed')
    return dict(path=str(MODEL_SOURCE),sha256=actual,architecture='author GCN class unchanged')


def load_case(batch,manifest,initial):
    if manifest['split'] not in ('train','validation') or manifest.get('final_test'):
        raise ValueError('Public development runner refuses final/test/extrapolation case')
    path=batch/'cases'/(manifest['case_id']+'.pkl')
    data=path.read_bytes()
    if sha256(data).hexdigest()!=manifest['pickle_sha256']:
        raise ValueError('Changed frozen case '+manifest['case_id'])
    problem,groups,warm,declaration=pickle.loads(data)
    if (declaration['capacity'],declaration['duty_scale'])!=(manifest['capacity'],manifest['duty_scale']):
        raise ValueError('Case condition mismatch')
    selected=warm if initial=='warm' else Plan() if initial=='cold' else None
    if selected is None:raise ValueError('Unknown initializer')
    problem.state=replace(problem.state,
        selected_arc_ids=frozenset(a.arc_id for a in selected.arcs),
        selected_tuple_ids=frozenset(c.tuple_id for c in selected.copies))
    checked=evaluate_plan(problem.domain,selected,WEIGHTS,groups)
    return problem,groups,checked


def restore_policy(checkpoint,expected_sha256):
    raw=checkpoint.read_bytes()
    if sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Selected trained checkpoint changed')
    # TrainingStore verifies its enclosed model hash before loading.
    from Agent.Publication.training_store import TrainingStore
    saved=TrainingStore(checkpoint.parent).load_model(checkpoint.name)
    if not saved.get('full_training_completed'):
        raise ValueError('Incomplete training checkpoint cannot enter comparison')
    verify_pinned_model()
    model=new_policy();model.load_state_dict(saved['model']);model.eval()
    return model


def evaluate_comparison(problem,groups,model,seed,seconds=60.):
    """Return the initial checked plan if action computation returns late."""
    from time import perf_counter
    start=perf_counter()
    initial=evaluate_plan(problem.domain,problem.plan(problem.state),WEIGHTS,groups)
    raw=run_learned(problem,WEIGHTS,model,seed,seconds=seconds,task_groups=groups)
    elapsed=perf_counter()-start
    ids=dict(selected_arc_ids=raw['selected_arc_ids'],selected_tuple_ids=raw['selected_tuple_ids'])
    selected=problem.plan(replace(problem.state,
        selected_arc_ids=frozenset(ids['selected_arc_ids']),
        selected_tuple_ids=frozenset(ids['selected_tuple_ids'])))
    check=evaluate_plan(problem.domain,selected,WEIGHTS,groups)
    if any(abs(a-b)>1e-6 for a,b in zip(raw['objective'],(check['primary'],check['secondary']))):
        raise ValueError('Learned result differs from independent physical/quota score')
    on_time=elapsed<=seconds and raw['within_common_budget']
    effective=check if on_time else initial
    effective_ids=ids if on_time else dict(selected_arc_ids=sorted(problem.state.selected_arc_ids),
                                          selected_tuple_ids=sorted(problem.state.selected_tuple_ids))
    return dict(status='returned' if on_time else 'late_return_initial',
        objective=[effective['primary'],effective['secondary']],
        initial=[initial['primary'],initial['secondary']],
        raw_objective=[check['primary'],check['secondary']],raw_plan=ids,
        selected_arc_ids=effective_ids['selected_arc_ids'],
        selected_tuple_ids=effective_ids['selected_tuple_ids'],
        completed_groups=effective['completed_groups'],physical_passed=True,
        within_budget=on_time,total_seconds=elapsed,details=raw)


def verify_recorded_comparison(problem,groups,record):
    """Physical replay for a persisted row before completing an interrupted commit."""
    def replay(ids):
        state=replace(problem.state,selected_arc_ids=frozenset(ids['selected_arc_ids']),
                      selected_tuple_ids=frozenset(ids['selected_tuple_ids']))
        if len(state.selected_arc_ids)!=len(ids['selected_arc_ids']) or len(state.selected_tuple_ids)!=len(ids['selected_tuple_ids']):
            raise ValueError('Repeated result decision ID')
        return evaluate_plan(problem.domain,problem.plan(state),WEIGHTS,groups)
    effective=replay(record)
    raw=replay(record['raw_plan'])
    for label,check in (('objective',effective),('raw_objective',raw)):
        if any(abs(a-b)>1e-6 for a,b in zip(record[label],(check['primary'],check['secondary']))):
            raise ValueError('Persisted learned result score mismatch')
    if set(record['completed_groups'])!=set(effective['completed_groups']):
        raise ValueError('Persisted learned completion mismatch')
    if record['status']=='late_return_initial':
        if (set(record['selected_arc_ids'])!=set(problem.state.selected_arc_ids) or
                set(record['selected_tuple_ids'])!=set(problem.state.selected_tuple_ids)):
            raise ValueError('Late result did not preserve initial plan')
    elif record['status']!='returned':
        raise ValueError('Unknown learned result status')
    return True
