"""Isolated expert adaptation on the preparation-before-takeoff domain."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dataclasses import replace
from hashlib import sha256
from collections import defaultdict
from statistics import mean
import json
import math
import torch
from Agent.Run.publication_timing_comparison import build_problem, signature
from Agent.Run.validate_publication_timing_comparison import check_schedule
from Agent.Run.publication_late_dynamic import pack
from Agent.Run.evaluate_publication_rolling_roads import decode
from Agent.Tool.Benchmark.formal_v3_loader import FormalV3InstanceLoader
from Agent.Tool.Training.chdgrl_config import load_chdgrl_experiment_config
from Agent.Tool.Training.chdgrl_expert_labels import (
    _label_expert_actions_and_repairs, _bundle_to_dict, _bundle_from_dict,
    compare_expert_action_labels, _action_id)
from Agent.Tool.Training.chdgrl_pretraining import ExpertRankingExample, _best_label_index
from Agent.Proposed.dgrl_actions import JointAction
from Agent.Proposed.hgnn import tensorize_graph
from Agent.Proposed.hypergraph import build_constraint_hypergraph
from Agent.Proposed.search import _dgrl_context
from Agent.Tool.Evaluation.chdgrl_methods import load_frozen_policy

NAME = 'publication_corrected_training_20260910'
PHASE = ROOT/'Data/Paper'/NAME
DATA = ROOT/'Data/Result/Diagnostic'/NAME
MODELS = ROOT/'Data/Checkpoint'/NAME
MANIFEST = ROOT/'Data/Instance/Benchmark/chdgrl_physical_v2/development_manifest.jsonl'
CONFIG = ROOT/'Data/Config/chdgrl_lns/scale_normalized_v3_reward_bounds.json'
SEEDS = tuple(range(20260730,20260735))
LIMITS = {'original':(4,2),'extended':(32,8)}


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def digest(path): return sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def physical_instance(instance):
    if instance.split not in ('train','validation'):
        raise ValueError('Development instances only')
    problem,_=build_problem(instance,'prelaunch_occupancy')
    return replace(instance,search_state=problem.search_state,rolling_state=problem.rolling_state,
                   operation_capacity=problem.operation_capacity),problem


def snapshot(problem,primary,secondary,tuples,arcs):
    return dict(primary=float(primary),secondary=float(secondary),selected_tuple_ids=sorted(tuples),selected_arc_ids=sorted(arcs))


def label_instance(instance,config,*,max_states=4,max_candidates=48):
    """Save all repair witnesses, including rejected alternatives, before training."""
    transformed,problem=physical_instance(instance)
    state=problem.search_state
    records=[]
    seen=set()
    for index in range(max_states):
        state_key=(state.selected_tuple_ids,state.selected_arc_ids)
        if state_key in seen: break
        seen.add(state_key)
        bundle,repairs=_label_expert_actions_and_repairs(transformed,config,max_candidates=max_candidates,
            seed=20260910,time_indices=(0,1,2),search_state=state,state_index=index)
        current_problem=replace(problem,search_state=state)
        initial=snapshot(problem,bundle.initial_primary,bundle.initial_secondary,state.selected_tuple_ids,state.selected_arc_ids)
        errors,physical,_=check_schedule(current_problem,initial,config.search.objective_weights)
        if errors or physical: raise ValueError('Invalid expert input schedule')
        outcomes={}
        for label in bundle.labels:
            if label.action_id not in repairs: continue
            result=repairs[label.action_id]
            finite=math.isfinite(result.primary_value) and math.isfinite(result.secondary_value)
            witness=dict(selected_tuple_ids=sorted(result.selected_tuple_ids),selected_arc_ids=sorted(result.selected_arc_ids),
                primary=float(result.primary_value) if finite else None,secondary=float(result.secondary_value) if finite else None,
                accepted=result.accepted,primary_status=result.primary_status,secondary_status=result.secondary_status,
                validation_ok=result.validation.ok)
            if label.outcome_feasible:
                if not finite: raise ValueError('Nonfinite feasible expert label')
                model_errors,physical_errors,_=check_schedule(current_problem,witness,config.search.objective_weights)
                if model_errors or physical_errors: raise ValueError('Expert repair violates independently checked constraints')
            outcomes[label.action_id]=witness
        improving=[label for label in bundle.labels if label.strict_improvement]
        best=None
        for label in improving:
            if best is None or compare_expert_action_labels(label,best)>0 or (
                compare_expert_action_labels(label,best)==0 and label.action_id<best.action_id): best=label
        next_action=best.action_id if best else None
        records.append(dict(bundle=_bundle_to_dict(bundle),outcomes=outcomes,next_action=next_action))
        if best is None: break
        repaired=repairs[best.action_id]
        state=replace(state,selected_tuple_ids=frozenset(repaired.selected_tuple_ids),selected_arc_ids=frozenset(repaired.selected_arc_ids))
    return dict(instance_id=instance.instance_id,split=instance.split,original_instance_hash=instance.instance_hash,
        corrected_instance_hash=transformed.instance_hash,problem_hash=signature(problem),problem=pack(problem),records=records)


def make_examples(payload,config,regime='original'):
    problem=decode(payload['problem'])
    if payload['split'] not in ('train','validation'): raise ValueError('Test labels forbidden')
    if signature(problem)!=payload['problem_hash']: raise ValueError('Changed corrected domain')
    maximum,patience=LIMITS[regime]
    root=payload['records'][0]['bundle']
    result=[]
    for record in payload['records']:
        bundle=_bundle_from_dict(record['bundle'])
        if bundle.instance_hash!=payload['corrected_instance_hash'] or bundle.split!=payload['split']:
            raise ValueError('Expert label/domain mismatch')
        state=replace(problem.search_state,selected_tuple_ids=frozenset(bundle.selected_tuple_ids),selected_arc_ids=frozenset(bundle.selected_arc_ids))
        graph=tensorize_graph(build_constraint_hypergraph(state,problem.task_ids,task_weights=problem.miss_penalty))
        actions=tuple(JointAction(label.neighborhood_type,label.seed_id,label.release_profile,label.time_budget) for label in bundle.labels)
        if any(_action_id(a)!=label.action_id for a,label in zip(actions,bundle.labels)): raise ValueError('Action identity mismatch')
        indices={label.action_id:i for i,label in enumerate(bundle.labels)}
        pairs=tuple((indices[a],indices[b]) for a,b in bundle.ordered_pairs)
        # Offline expert states have no observed policy wall clock. The remaining
        # fraction is prescribed, as in the original expert-training protocol.
        context=_dgrl_context(initial_primary=root['initial_primary'],initial_secondary=root['initial_secondary'],
            best_primary=bundle.initial_primary,best_secondary=bundle.initial_secondary,iteration=bundle.state_index,
            max_iterations=maximum,no_improve=0,no_improve_limit=patience,
            remaining_budget_s=max(.2,1-bundle.state_index/4),total_budget_s=1.,last_runtime_s=0.,last_failure=False)
        result.append(ExpertRankingExample(bundle.instance_id,bundle.split,bundle.state_hash,graph,context,
            actions,bundle.labels,pairs,_best_label_index(bundle.labels)))
    return tuple(result)


def original_model(seed,config):
    freeze=read(ROOT/'Data/Checkpoint/chdgrl_final_guard8_v1/method_freeze_report.json')
    return load_frozen_policy(ROOT/f'Data/Checkpoint/chdgrl_final_guard8_v1/seed_{seed}/frozen.pt',
        config,digest(MANIFEST),expected_source_sha256=freeze['source_sha256'])[0]


def ranking_metrics(model,examples):
    """Average states within each instance before comparing validation quality."""
    grouped=defaultdict(list)
    model.eval()
    with torch.no_grad():
        for example in examples:
            q1,q2=model.q_values(example.graph,example.context,example.actions)
            q=torch.minimum(q1,q2).flatten()
            selected=int(torch.argmax(q))
            label=example.labels[selected]
            oracle=example.labels[example.oracle_index]
            pairs=example.ordered_pair_indices
            correct=sum(float(q[a]>q[b])+0.5*float(q[a]==q[b]) for a,b in pairs)
            # Feasibility ranks ahead of objective gains; separate hit rate
            # prevents infeasible zero-gain actions from hiding behind regret.
            grouped[example.instance_id].append(dict(feasible=float(label.outcome_feasible),
                primary_regret=max(0.,oracle.primary_gain-label.primary_gain),
                pair_accuracy=correct/len(pairs) if pairs else 1.))
    return {key:mean(mean(row[key] for row in values) for values in grouped.values())
            for key in ('feasible','primary_regret','pair_accuracy')}


def require_training_split(examples):
    if not examples or any(example.split!='train' for example in examples):
        raise ValueError('Only training examples may enter gradient updates')


def selection_score(metrics):
    return (metrics['feasible'],-metrics['primary_regret'],metrics['pair_accuracy'])
