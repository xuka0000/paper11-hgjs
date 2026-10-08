"""Isolated train/validation action-model campaign; sealed final evaluation.

The finite queue implements the October 2 parallel-execution addendum. Public
baseline processes and their records are never modified by this module.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import psutil

from Agent.Run.foundation_campaign import dump, read, identity, alive

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'm2m-action-training-v1'
SEEDS = (17, 29, 43, 71, 101)
MODELS = ('action_hypergraph', 'action_factor', 'action_pairwise', 'action_mlp',
          'no_incumbent', 'legacy_pool')


def raw_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def checked_manifests(batch):
    result = []
    for path in sorted((batch/'cases').glob('*.json')):
        m = read(path)
        if m.get('split') not in ('train', 'validation') or m.get('final_test'):
            raise ValueError('Development input contains a final test case')
        if raw_hash(batch/'cases'/(m['case_id']+'.pkl')) != m['pickle_sha256']:
            raise ValueError('Frozen case digest mismatch: '+m['case_id'])
        result.append(m)
    if len(result) != 160 or sum(x['split']=='train' for x in result) != 128:
        raise ValueError('Expected128 train and32 validation conditions')
    expected=read(batch/'expected_cases.json')
    if len({m['case_id'] for m in result})!=160 or set(expected)!={m['case_id'] for m in result}:
        raise ValueError('Frozen case identity set differs from original contract')
    keys=('base_id','geometry_seed','components','topology','split','capacity','duty_scale')
    for m in result:
        if any(m[k]!=expected[m['case_id']][k] for k in keys):
            raise ValueError('Frozen split/stratum metadata mismatch: '+m['case_id'])
    bases={m['base_id']:(m['split'],m['geometry_seed']) for m in result}
    if len(bases)!=40 or len({x[1] for x in bases.values()})!=40:
        raise ValueError('Base or geometry seed reuse across development split')
    return result


def training_jobs(seeds=SEEDS):
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or s not in SEEDS for s in seeds):
        raise ValueError('Training seeds must be a nonempty distinct subset of the frozen seeds')
    return [dict(id=f'{method}_s{seed}', method=method, seed=seed)
            for seed in seeds for method in MODELS]


def execution_revision(output,binding):
    """Explicit execution-only amendment; scientific binding/budgets stay intact."""
    path=output/'execution_revision.json'
    if not path.exists():return None
    r=read(path);rel='Agent/Run/m2m_own_pipeline.py'
    if (r.get('kind')!='single_seed_cached_inputs_v1' or r.get('seeds')!=[17] or
        r.get('user_authorized_seed_reduction') is not True or
        r.get('scientific_binding_sha256')!=canonical_hash(binding) or
        r.get('previous_source_sha256')!=binding['source_sha256'].get(rel) or
        r.get('executed_source_sha256')!=raw_hash(ROOT/rel) or
        r.get('original_config_sha256')!=binding['config_sha256']):
        raise ValueError('Execution-only amendment identity mismatch')
    evidence=output/'execution_revision_preflight.json'
    if (not evidence.exists() or raw_hash(evidence)!=r.get('preflight_sha256')):
        raise ValueError('Execution amendment requires exact preflight evidence')
    proof=read(evidence)
    if (proof.get('passed') is not True or
        proof.get('executed_source_sha256')!=r['executed_source_sha256'] or
        proof.get('scientific_binding_sha256')!=canonical_hash(binding)):
        raise ValueError('Execution amendment preflight identity mismatch')
    return r


def utility(record, label):
    """A fixed, unitless target; no censored label becomes a negative example."""
    if label['status'] == 'censored':
        return None
    if label['status'] == 'infeasible':
        return -2.0
    if label['status'] not in ('checked','observed_terminal_continuation'):
        raise ValueError('Unrecognized or inconsistent physical label')
    if label['status']=='checked' and label['physical_feasibility'] is not True:
        raise ValueError('Checked label lacks physical feasibility')
    if label['status']=='observed_terminal_continuation' and not label.get('checked_completions',0):
        raise ValueError('Continuation lacks a physically checked full-action witness')
    priorities = [float(g['priority']) for g in record['view']['quotas']]
    total = sum(priorities)
    if total <= 0:
        raise ValueError('Nonpositive total quota weight')
    # Bounded secondary tie term is smaller than the smallest primary quantum.
    epsilon = .1 * min(priorities) / total
    initial_cost = float(record['incumbent']['objective'][1])
    tie = math.tanh((initial_cost-float(label['cost']))/max(1., abs(initial_cost)))
    return float(label['gain'])/total + epsilon*tie


def examples(record):
    from Agent.Publication.joint_service_actions import ActionProposal
    actions, targets, seen = [], [], {}
    for row in list(record['proposals']) + list(record.get('continuations', [])):
        value = utility(record, row['label'])
        if value is None:
            continue
        key = (tuple(sorted(row['add'])), tuple(sorted(row['release'])))
        # A continuation is valued by its best checked completion, never by
        # incorrectly applying a completed-action label to an infeasible prefix.
        seen[key] = max(value, seen.get(key, -math.inf))
    for (add, release), value in sorted(seen.items()):
        actions.append(ActionProposal(add, release)); targets.append(value)
    return actions, targets


def loss_for(scores, targets):
    import torch
    from torch.nn import functional as F
    values = torch.tensor(targets, dtype=scores.dtype, device=scores.device)
    regression = F.smooth_l1_loss(scores, values)
    # Deterministic pair selection, identical across representations and seeds.
    i, j = torch.where((values[:, None]-values[None, :]) > 1e-6)
    if len(i):
        idx = torch.arange(min(256, len(i)), device=scores.device)
        ranking = F.softplus(-(scores[i[idx]]-scores[j[idx]])).mean()
        return regression + .1*ranking
    return regression


def load_records(output):
    from Agent.Publication.m2m_action_dataset import iter_state_records
    records = [record for folder in sorted((output/'labels').iterdir()) if folder.is_dir()
               for record in iter_state_records(folder)]
    splits = {name: [r for r in records if r['split']==name] for name in ('train','validation')}
    if len(records) != sum(map(len, splits.values())) or not all(splits.values()):
        raise ValueError('Unexpected split or empty train/validation labels')
    if len(splits['train']) > 512 or len(splits['validation']) > 128:
        raise ValueError('State budget exceeded')
    if len({r['state_hash'] for r in records}) != len(records):
        raise ValueError('Duplicate state hashes in committed dataset')
    return splits


def model_spec(method):
    if method in ('no_incumbent', 'legacy_pool'):
        return 'action_hypergraph', method
    if method not in MODELS:
        raise ValueError('Unknown trainable method')
    return method, None


def prepare_material(output,model):
    """Cache immutable input tensors only, never trainable model embeddings."""
    from Agent.Publication.m2m_action_dataset import iter_state_records,context_from_record
    material={'train':[],'validation':[]};counts={k:0 for k in material};seen=set()
    for folder in sorted((output/'labels').iterdir()):
        if not folder.is_dir():continue
        for record in iter_state_records(folder):
            split=record['split'];key=record['state_hash']
            if split not in material or key in seen:raise ValueError('Invalid or duplicate training state')
            seen.add(key);counts[split]+=1
            actions,labels=examples(record)
            if actions:
                material[split].append((model.encode(context_from_record(record)),actions,labels))
    if counts['train']>512 or counts['validation']>128 or not all(material.values()):
        raise ValueError('Unexpected state count or empty uncensored split')
    return material


def train_model(output, job, config):
    import torch
    from Agent.Publication.m2m_action_ranker import ActionRanker
    from Agent.Publication.training_store import TrainingStore
    torch.set_num_threads(1)
    torch.manual_seed(job['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(job['seed'])
    random.seed(job['seed'])
    device = config['device']
    variant, ablation = model_spec(job['method'])
    model = ActionRanker(variant=variant, hidden=64, layers=3, ablation=ablation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    store = TrainingStore(output/'models'/job['id'])
    binding = read(output/'binding.json')
    contract_hash = canonical_hash(dict(job=job, binding=binding, version=VERSION))
    best, bad, best_epoch, best_state, history = math.inf, 0, None, None, []
    started = time.time();resume_epoch=0;resume_position=0;resume_losses=[]
    with store.lock():
        if store.exists('summary.json'):
            result = store.get('summary.json')
            if result['contract_hash'] != contract_hash:
                raise ValueError('Completed model has a different binding')
            if raw_hash(store.path('selected.checkpoint')) != result['checkpoint_sha256']:
                raise ValueError('Completed model digest changed')
            return result
        if store.exists('attempt.json'):
            attempt=store.get('attempt.json')
            if not (output/'active_resume_receipt.json').exists() or not store.exists('latest.checkpoint'):
                raise RuntimeError('Interrupted training requires diagnosis, checkpoint and explicit verified resume')
            if attempt['contract_hash']!=contract_hash or alive(attempt['identity']):
                raise ValueError('Training resume binding or old process mismatch')
            restored=store.load_model('latest.checkpoint')
            if restored['contract_hash']!=contract_hash:
                raise ValueError('Training resume checkpoint binding mismatch')
            started=attempt['at'];resume_epoch=restored['next_epoch'];resume_position=restored['next_position']
            resume_losses=restored['train_losses'];history=restored['history']
            best=restored['best'];bad=restored['bad'];best_epoch=restored['best_epoch'];best_state=restored['best_model']
            model.load_state_dict(restored['model']);optimizer.load_state_dict(restored['optimizer'])
            torch.set_rng_state(restored['torch_rng'])
            if device.startswith('cuda'):torch.cuda.set_rng_state_all(restored['cuda_rng'])
        else:
            store.put('attempt.json', dict(at=started,job=job,contract_hash=contract_hash,identity=identity()))
        def save_current(next_epoch,next_position,losses):
            latest=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                optimizer=optimizer.state_dict(),next_epoch=next_epoch,next_position=next_position,
                train_losses=losses,history=history,best_model=best_state,best=best,bad=bad,
                best_epoch=best_epoch,contract_hash=contract_hash,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
            import io
            stream=io.BytesIO();torch.save(latest,stream);raw=stream.getvalue()
            store._write('latest.checkpoint',hashlib.sha256(raw).hexdigest().encode()+b'\n'+raw,mutable=True)
        if not store.exists('latest.checkpoint'):save_current(0,0,[])
        # This deterministic, parameter-free preparation preserves restored RNG,
        # model/optimizer, state order and original attempt/deadline.
        preparation_started=time.time()
        dump(store.path('input_preparation.json'),dict(at=time.time(),status='running',
            identity=identity(),resume_epoch=resume_epoch+1,resume_position=resume_position,
            original_started=started,contract_hash=contract_hash,
            checkpoint_sha256=raw_hash(store.path('latest.checkpoint')),
            model_and_optimizer_restored=store.exists('attempt.json') and resume_epoch>0))
        material=prepare_material(output,model)
        dump(store.path('input_preparation.json'),dict(at=time.time(),
            status='completed',identity=identity(),resume_epoch=resume_epoch+1,resume_position=resume_position,
            original_started=started,contract_hash=contract_hash,
            seconds=time.time()-preparation_started,train_states=len(material['train']),
            validation_states=len(material['validation']),cached='Immutable CPU input tensors only'))
        for epoch in range(resume_epoch,100):
            if bad>=15:break
            epoch_start = time.time()
            order = list(range(len(material['train'])))
            random.Random(job['seed']+epoch*1009).shuffle(order)
            model.train(); train_losses = list(resume_losses) if epoch==resume_epoch else []
            for number, idx in enumerate(order):
                if epoch==resume_epoch and number<resume_position:continue
                if time.time()-started>config['model_worker_hard_seconds']:
                    raise RuntimeError('Original cumulative model deadline exhausted')
                encoded, actions, labels = material['train'][idx]
                optimizer.zero_grad(set_to_none=True)
                scores = model(encoded, actions)
                loss = loss_for(scores, labels)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
                if (number+1)%32==0:save_current(epoch,number+1,train_losses)
                if number % 8 == 0:
                    dump(store.path('progress.json'),dict(at=time.time(),epoch=epoch+1,epochs_max=100,
                        state_completed=number+1,state_total=len(order),job=job,phase='fit',
                        elapsed_seconds=time.time()-started))
            model.eval(); validation_losses = []
            with torch.no_grad():
                for encoded, actions, labels in material['validation']:
                    scores = model(encoded, actions)
                    validation_losses.append(float(loss_for(scores,labels).cpu()))
            value = sum(validation_losses)/len(validation_losses)
            metric = dict(epoch=epoch+1,train_loss=sum(train_losses)/len(train_losses),
                validation_loss=value,seconds=time.time()-epoch_start,
                train_states=len(material['train']),validation_states=len(material['validation']))
            history.append(metric)
            # A crash after writing an epoch record but before its checkpoint
            # may replay that uncommitted epoch tail from the preceding state.
            # Preserve the previous metric as an interrupted attempt receipt.
            name=f'epochs/{epoch+1:03}.json'
            if store.exists(name):
                if not (output/'active_resume_receipt.json').exists():raise ValueError('Existing epoch record')
                store.put(f'interrupted_metrics/{epoch+1:03}_{int(time.time())}.json',store.get(name))
            store.put(name,metric,mutable=True)
            if value < best-1e-8:
                best,bad,best_epoch=value,0,epoch+1
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                bad += 1
            # One current resumable state; replaces the preceding epoch, not a
            # permanent collection of full optimizer/environment snapshots.
            save_current(epoch+1,0,[])
            if bad >= 15:
                break
        if not store.exists('selected.checkpoint'):
            store.save_model('selected.checkpoint',dict(model=best_state,method=job['method'],
                variant=variant,ablation=ablation,seed=job['seed'],hidden=64,layers=3,
                selected_epoch=best_epoch,full_training_completed=True,contract_hash=contract_hash,
                selection='minimum validation supervised loss; no test access',source_version=VERSION))
        else:
            saved=store.load_model('selected.checkpoint')
            if saved['contract_hash']!=contract_hash or saved['selected_epoch']!=best_epoch:
                raise ValueError('Recovered selected checkpoint identity mismatch')
            if any(not torch.equal(saved['model'][k],v) for k,v in best_state.items()):
                raise ValueError('Recovered selected checkpoint model mismatch')
        result=dict(job=job,contract_hash=contract_hash,epochs_completed=len(history),
            selected_epoch=best_epoch,validation_loss=best,history=history,
            checkpoint_sha256=raw_hash(store.path('selected.checkpoint')),
            seconds=time.time()-started,train_states=len(material['train']),
            validation_states=len(material['validation']),formal_test_accessed=False)
        store.put('summary.json',result)
        return result


def evaluate_model(batch, output, task, config):
    import torch
    from Agent.Publication.budget import Deadline
    from Agent.Publication.component_objective import evaluate_plan
    from Agent.Publication.foundation_benchmark import WEIGHTS
    from Agent.Publication.joint_service_search import run_joint_search, SearchLimits
    from Agent.Publication.m2m_action_ranker import ActionRanker, NeuralActionPolicy
    from Agent.Publication.m2m_public_learning import load_case
    from Agent.Publication.training_store import TrainingStore
    torch.set_num_threads(1)
    m=read(batch/'cases'/(task['case_id']+'.json'))
    if m['split']!='validation':
        raise ValueError('Development evaluation accepts validation only')
    problem,groups,_=load_case(batch,m,task['initial'])
    job=task['model'];store=TrainingStore(output/'models'/job['id'])
    summary=store.get('summary.json')
    if raw_hash(store.path('selected.checkpoint'))!=summary['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    saved=store.load_model('selected.checkpoint')
    if not saved['full_training_completed']:
        raise ValueError('Incomplete model cannot enter evaluation')
    model=ActionRanker(saved['variant'],saved['hidden'],saved['layers'],saved['ablation'])
    model.load_state_dict(saved['model']);model.eval()
    policy=NeuralActionPolicy(model,device='cpu')
    initial=problem.plan(problem.state)
    before=evaluate_plan(problem.domain,initial,WEIGHTS,groups)
    started=time.perf_counter();deadline=Deadline(60.);problem.deadline=deadline
    result=run_joint_search(problem,groups,initial,WEIGHTS,deadline,policy=policy)
    selected=result.pop('plan')
    checked=evaluate_plan(problem.domain,selected,WEIGHTS,groups)
    elapsed=time.perf_counter()-started
    if any(abs(a-b)>1e-6 for a,b in zip(result['objective'],[checked['primary'],checked['secondary']])):
        raise ValueError('Neural result differs from independent physical replay')
    raw_ids=dict(selected_arc_ids=sorted(a.arc_id for a in selected.arcs),
                 selected_tuple_ids=sorted(c.tuple_id for c in selected.copies))
    raw_obj=[checked['primary'],checked['secondary']]
    if elapsed>60.:
        selected=initial;checked=before
    row=dict(task=task,at=time.time(),status='returned' if elapsed<=60 else 'late_return_initial',
        physical_passed=True,within_budget=elapsed<=60,total_seconds=elapsed,
        initial=[before['primary'],before['secondary']],objective=[checked['primary'],checked['secondary']],
        raw_objective=raw_obj,raw_plan=raw_ids,completed_groups=checked['completed_groups'],
        selected_arc_ids=sorted(a.arc_id for a in selected.arcs),
        selected_tuple_ids=sorted(c.tuple_id for c in selected.copies),details=result,
        checkpoint_sha256=summary['checkpoint_sha256'],case_sha256=m['pickle_sha256'])
    from Agent.Run.m2m_result_storage import write_result
    write_result(output/'validation',task['id'],row)
    return row


def kill_owned(value):
    if not alive(value):
        return
    p=psutil.Process(value['pid'])
    for child in p.children(recursive=True):
        try: child.kill()
        except psutil.NoSuchProcess: pass
    try:p.kill()
    except psutil.NoSuchProcess:pass


def summarize_round(output,tasks,name):
    """Compact paired first-round results; one seed cannot measure seed stability."""
    from Agent.Run.m2m_result_storage import read_result
    rows=[];by_method={}
    for task in tasks:
        row,_=read_result(output/'validation',task['id'])
        if row['task']!=task or not row['physical_passed']:
            raise ValueError('First-round summary requires physically checked matching results')
        method=task['model']['method'];values=by_method.setdefault(method,[])
        item=dict(method=method,case_id=task['case_id'],initial=task['initial'],
            before=row['initial'],after=row['objective'],seconds=row['total_seconds'],status=row['status'])
        rows.append(item);values.append(item)
    methods={}
    for method,items in by_method.items():
        if len(items)!=64 or len({(x['case_id'],x['initial']) for x in items})!=64:
            raise ValueError('Incomplete method/case/initial matrix')
        methods[method]=dict(rows=len(items),mean_primary=sum(x['after'][0] for x in items)/len(items),
            mean_primary_gain=sum(x['after'][0]-x['before'][0] for x in items)/len(items),
            late_returns=sum(x['status']=='late_return_initial' for x in items))
    dump(output/(name+'.json'),dict(at=time.time(),status='computed_pending_scientific_review',
        seed=17,methods=methods,paired_rows=rows,
        boundary='One training seed,32validation conditions with cold/warm starts; no final test or seed-stability claim'))


def observer(output):
    launch=read(output/'launch.json')
    while True:
        now=time.time();progress=read(output/'progress.json')
        dump(output/'observer.json',dict(at=now,identity=identity(),parent_alive=alive(launch['identity'])))
        if (output/'terminal.json').exists():
            return
        if not alive(launch['identity']) or now-progress.get('at',0)>90:
            for item in progress.get('active',[]):
                kill_owned(item['identity'])
            dump(output/'observer_failure.json',dict(at=now,reason='controller_missing_or_stale',progress=progress))
            return
        time.sleep(5)


def verify_completed_task(batch,output,task):
    """Reuse only independently replayed/hash-checked, committed task outputs."""
    from Agent.Publication.training_store import TrainingStore
    if task['mode']=='labels':
        from Agent.Publication.m2m_action_dataset import iter_state_records
        folder=output/'labels'/task['case_id']
        records=list(iter_state_records(folder))
        index=read(folder/'manifest.json')
        if len(index['states'])!=4 or any(x['status']=='failed' for x in index['states']):
            raise ValueError('Incomplete label state ledger')
        if any(r['case_id']!=task['case_id'] for r in records):raise ValueError('Label case mismatch')
    elif task['mode']=='train':
        store=TrainingStore(output/'models'/task['id']);summary=store.get('summary.json')
        if raw_hash(store.path('selected.checkpoint'))!=summary['checkpoint_sha256']:
            raise ValueError('Completed model changed')
        if summary['job']!={k:task[k] for k in ('id','method','seed')}:
            raise ValueError('Completed model identity changed')
    else:
        from Agent.Run.m2m_result_storage import read_result
        from Agent.Publication.m2m_public_learning import load_case,verify_recorded_comparison
        row,_=read_result(output/'validation',task['id'])
        if row['task']!=task:raise ValueError('Validation task mismatch')
        m=read(batch/'cases'/(task['case_id']+'.json'))
        problem,groups,_=load_case(batch,m,task['initial'])
        verify_recorded_comparison(problem,groups,row)


def admit_resume(output,binding,receipt_path):
    """An explicit, evidence-bound recovery; never an automatic retry."""
    receipt=read(receipt_path)
    prior=read(output/'launch.json')
    progress=read(output/'progress.json') if (output/'progress.json').exists() else {'active':[]}
    if (receipt.get('previous_launch_sha256')!=raw_hash(output/'launch.json') or
        receipt.get('binding_sha256')!=canonical_hash(binding) or not receipt.get('diagnosis') or
        not receipt.get('bounded_recovery_authorized') or receipt.get('attempt_number') not in (2,3) or
        receipt.get('attempt_number')!=prior.get('attempt_number',1)+1):
        raise ValueError('Invalid or unbounded recovery receipt')
    if alive(prior['identity']) or any(alive(x['identity']) for x in progress.get('active',[])):
        raise RuntimeError('Prior controller or worker still alive')
    if (output/'observer.json').exists() and alive(read(output/'observer.json')['identity']):
        raise RuntimeError('Prior observer still alive')
    return receipt


def commit_resume_history(output,receipt):
    """Copy small evidence records; never remove the prior launch authority.

    Reentry after interruption is idempotent. The new launch is an atomic write
    performed only after all bindings and completed artifacts have passed.
    """
    previous=output/'attempt_history'/str(receipt['attempt_number']-1)
    previous.mkdir(parents=True,exist_ok=True)
    for name in ('launch.json','progress.json','observer.json','terminal.json','failure.json','observer_failure.json'):
        path=output/name
        if path.exists():
            target=previous/name
            if target.exists() and raw_hash(target)!=raw_hash(path):
                raise ValueError('Recovery history collision: '+name)
            if not target.exists():
                temporary=target.with_suffix(target.suffix+'.pending')
                temporary.write_bytes(path.read_bytes());temporary.replace(target)
    dump(output/'active_resume_receipt.json',receipt)


def dataset_audit(output):
    data=load_records(output);counts={};opportunities=[]
    for split,records in data.items():
        by_source={s:{'checked':0,'infeasible':0,'censored':0,'unavailable':0} for s in ('quota_guided','resource_guided','random')}
        for record in records:
            for slot in record['unavailable_proposal_slots']:by_source[slot['source']]['unavailable']+=1
            for row in record['proposals']:
                label=row['label'];by_source[row['source']][label['status']]+=1
                if (label['status']=='checked' and label['gain']>0 and len(row['add'])>=2 and
                    set(record['incumbent']['selected_tuple_ids'])-set(row['release'])):
                    opportunities.append(dict(state_id=record['state_id'],add=row['add'],release=row['release'],
                                              gain=label['gain'],split=split))
        counts[split]=dict(states=len(records),by_source=by_source)
    summary=dict(at=time.time(),counts=counts,nominal_states={'train':512,'validation':128},
        max_label_calls=15360,joint_preserved_service_positive_count=len(opportunities),
        joint_preserved_service_examples=opportunities[:32],
        balance='8 nominal slots per source; distinct availability and censoring reported separately',
        scientific_acceptance=False,formal_test_accessed=False)
    dump(output/'dataset_summary.json',summary)
    if not opportunities:
        raise ValueError('G1: no checked positive joint action preserving incumbent service; diagnose before training')
    return summary


def pipeline(batch,output,config,resume_receipt=None):
    from Agent.Publication.training_store import TrainingStore
    import shutil
    output.mkdir(parents=True,exist_ok=True)
    with TrainingStore(output).lock():
        binding=read(output/'binding.json')
        revision=execution_revision(output,binding)
        if canonical_hash(config)!=binding['config_sha256']:
            raise ValueError('Deployment config changed')
        if raw_hash(batch/'expected_cases.json')!=binding.get('expected_cases_sha256'):
            raise ValueError('Original-contract expected case inventory changed')
        for name,digest in binding['case_manifest_sha256'].items():
            if raw_hash(batch/'cases'/(name+'.json'))!=digest:
                raise ValueError('Frozen case manifest changed: '+name)
        manifests=checked_manifests(batch)
        if {m['case_id'] for m in manifests}!=set(binding['case_manifest_sha256']):
            raise ValueError('Manifest hash inventory coverage mismatch')
        for rel,digest in binding['source_sha256'].items():
            if revision and rel=='Agent/Run/m2m_own_pipeline.py':
                digest=revision['executed_source_sha256']
            if raw_hash(ROOT/rel)!=digest:
                raise ValueError('Frozen source changed: '+rel)
        for m in manifests:
            if binding['case_sha256'].get(m['case_id'])!=m['pickle_sha256']:
                raise ValueError('Frozen input inventory changed')
        evidence=output/'preflight.json'
        if (not binding.get('code_preflight_passed') or not evidence.exists() or
            raw_hash(evidence)!=binding.get('preflight_sha256') or not read(evidence).get('passed')):
            raise ValueError('Code preflight evidence required')
        attempt=1;receipt=None
        if (output/'launch.json').exists():
            if resume_receipt is None:raise RuntimeError('Existing controller attempt: explicit verified recovery receipt required')
            receipt=admit_resume(output,binding,resume_receipt);attempt=receipt['attempt_number']
        active={};finished=read(output/'completed.json') if attempt>1 and (output/'completed.json').exists() else []
        for task in finished:verify_completed_task(batch,output,task)
        if receipt:commit_resume_history(output,receipt)
        seeds=tuple(revision['seeds']) if revision else SEEDS
        jobs=training_jobs(seeds)
        launch=dict(at=time.time(),identity=identity(),version=VERSION,attempt_number=attempt,
            finite_queue=f'160 case-label tasks;{len(jobs)} fitted models;{len(jobs)*64} validation runs; then independent gate audit',
            execution_revision_sha256=raw_hash(output/'execution_revision.json') if revision else None,
            no_automatic_retry=True,formal_test_accessed=False)
        dump(output/'launch.json',launch)
        if receipt:
            # After new launch authority is durably established, archived old
            # terminal flags can be retired. The launch remains recoverable.
            for name in ('terminal.json','failure.json','observer_failure.json'):
                path=output/name
                if path.exists():path.unlink()
        phase='labels';status='running';error=None;watch=None
        labels=[dict(id='labels_'+m['case_id'],mode='labels',case_id=m['case_id']) for m in manifests]
        models=[dict(j,mode='train') for j in jobs]
        evaluations=[dict(id=f"{j['id']}_{m['case_id']}_{init}",mode='evaluate',model=j,
                         case_id=m['case_id'],initial=init)
                     for j in jobs for m in manifests if m['split']=='validation'
                     for init in ('cold','warm')]
        queues={'labels':labels,'train':models,'evaluate':evaluations}
        caps={'labels':config['label_workers'],'train':config['training_workers'],
              'evaluate':config['evaluation_workers']}
        limits={'labels':config['case_worker_hard_seconds'],'train':config['model_worker_hard_seconds'],
                'evaluate':240}
        dump(output/'task_matrix.json',queues)
        def snapshot():
            counts={key:sum(t['mode']==key for t in finished) for key in queues}
            value=dict(at=time.time(),started=launch['at'],status=status,phase=phase,
                completed=counts,planned={k:len(v) for k,v in queues.items()},
                active=[dict(task=v['task'],identity=v['identity'],started=v['started']) for v in active.values()],
                failure=error,free_bytes=shutil.disk_usage(output).free,formal_test_accessed=False,
                scientific_acceptance=False,next_check_at=time.time()+5,phone_push_verified=False)
            dump(output/'progress.json',value);return value
        def submit(task):
            path=output/'tasks'/(task['id']+'.json');dump(path,task)
            cmd=[sys.executable,'-m','Agent.Run.m2m_own_pipeline','worker','--batch',str(batch),
                 '--output',str(output),'--config',str(output/'config.json'),'--task',str(path)]
            intent=output/'intents'/(task['id']+'.json')
            first=time.time()
            if intent.exists():
                if attempt==1:raise RuntimeError('Prior task attempt; no implicit retry: '+task['id'])
                old=read(intent);first=old.get('first_at',old['at'])
                dump(output/'attempt_history'/str(attempt-1)/('intent_'+task['id']+'.json'),old)
                if time.time()-first>limits[task['mode']]:raise RuntimeError('Original task deadline exhausted: '+task['id'])
            dump(intent,dict(at=time.time(),first_at=first,command=cmd,task=task,attempt=attempt))
            log=output/'logs'/(task['id']+f'_attempt{attempt}.log');log.parent.mkdir(exist_ok=True)
            with log.open('x') as stream:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                             PYTHONHASHSEED=str(task.get('seed',1701))))
            active[task['id']]=dict(process=proc,task=task,identity=identity(proc.pid),started=first)
        try:
            dump(output/'config.json',config);snapshot()
            with (output/f'observer_attempt{attempt}.log').open('x') as log:
                watch=subprocess.Popen([sys.executable,'-m','Agent.Run.m2m_own_pipeline','observe',
                    '--output',str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            stages=[('labels','labels'),('train','train'),('evaluate','evaluate')]
            if revision:
                stages=[('labels','labels'),('train_primary','train'),('evaluate_primary','evaluate'),
                        ('train_ablation','train'),('evaluate_ablation','evaluate')]
            for stage,phase in stages:
                completed_ids={t['id'] for t in finished}
                chosen=queues[phase]
                if stage.endswith(('_primary','_ablation')):
                    ablation=stage.endswith('_ablation')
                    chosen=[t for t in chosen if ((t.get('method') or t['model']['method']) in MODELS[-2:])==ablation]
                pending=[t for t in chosen if t['id'] not in completed_ids];snapshot()
                while pending or active:
                    if (output/'STOP_REQUESTED').exists():
                        raise RuntimeError('Explicit stop requested')
                    if watch.poll() is not None:
                        raise RuntimeError('Independent observer exited')
                    observed=read(output/'observer.json') if (output/'observer.json').exists() else {}
                    if time.time()-launch['at']>30 and time.time()-observed.get('at',0)>30:
                        raise RuntimeError('Independent observer heartbeat stale')
                    if shutil.disk_usage(output).free<2*1024**3:
                        raise RuntimeError('Remote storage below2GiB reserve')
                    for key,item in list(active.items()):
                        code=item['process'].poll()
                        if code is None:
                            if time.time()-item['started']>limits[phase]:
                                raise RuntimeError('Worker hard deadline exceeded: '+key)
                            continue
                        if code:
                            raise RuntimeError(f'Worker {key} exited {code}; inspect preserved failure')
                        receipt=read(output/'receipts'/(key+'.json'))
                        if receipt['task']!=item['task'] or not receipt['completed']:
                            raise ValueError('Worker completion receipt mismatch')
                        finished.append(item['task']);del active[key]
                        dump(output/'completed.json',finished)
                    while pending and len(active)<caps[phase]:
                        submit(pending.pop(0))
                    snapshot();time.sleep(5)
                if phase=='labels':
                    dataset_audit(output)
                if stage=='evaluate_primary':summarize_round(output,chosen,'primary_round_summary')
                if stage=='evaluate_ablation':summarize_round(output,queues['evaluate'],'single_seed_round_summary')
            status='computed_pending_independent_gate_audit'
        except BaseException as exc:
            status='stopped_requires_diagnosis'
            error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            dump(output/'failure.json',dict(at=time.time(),phase=phase,**error))
        finally:
            for item in active.values():kill_owned(item['identity']);item['process'].wait()
            active.clear();snapshot()
            dump(output/'terminal.json',dict(at=time.time(),status=status,error=error,
                next='Review G1/G2, public admission and freeze before192references/13440formal evaluations'))
            if watch:
                try:watch.wait(timeout=12)
                except subprocess.TimeoutExpired:watch.kill();watch.wait()
        if error:
            raise RuntimeError(error['message'])


def worker(batch,output,config,task):
    started=time.time()
    try:
        if task['mode']=='labels':
            from Agent.Publication.m2m_action_dataset import generate_case
            manifest=read(batch/'cases'/(task['case_id']+'.json'))
            result=generate_case(batch,manifest,output/'labels'/task['case_id'],seed=1701,
                proposal_seconds=config['proposal_seconds'],state_seconds=config['state_seconds'],
                resume_interrupted=(output/'active_resume_receipt.json').exists())
        elif task['mode']=='train':
            result=train_model(output,{k:task[k] for k in ('id','method','seed')},config)
        elif task['mode']=='evaluate':
            result=evaluate_model(batch,output,task,config)
        else:
            raise ValueError('Unknown task mode')
        dump(output/'receipts'/(task['id']+'.json'),dict(at=time.time(),task=task,
            completed=True,seconds=time.time()-started,result_hash=canonical_hash(result)))
    except BaseException as exc:
        dump(output/'failures'/(task['id']+'.json'),dict(at=time.time(),task=task,
            type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','observe','worker'])
    p.add_argument('--batch',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path);p.add_argument('--task',type=Path)
    p.add_argument('--resume-receipt',type=Path)
    a=p.parse_args();a.output=a.output.resolve()
    if a.mode=='observe':observer(a.output)
    elif a.mode=='worker':worker(a.batch.resolve(),a.output,read(a.config),read(a.task))
    else:pipeline(a.batch.resolve(),a.output,read(a.config),a.resume_receipt)
