"""Isolated Wang/HMAGAT-inspired round-two development; not an official baseline.

The training engine is a reviewed port of own_pipeline.train_model. It preserves
Adam3e-4,100epochs,15patience,seed17 and the original12h cumulative model limit.
No CLI path evaluates final-test data or starts cloud instances.
"""
import argparse
import hashlib
import math
from pathlib import Path
import random
import time
from Agent.Run.m2m_own_pipeline import (
    ROOT, raw_hash, canonical_hash, prepare_material, loss_for)
from Agent.Run.foundation_campaign import read, dump, identity, alive
from Agent.Publication.m2m_adaptive_hypergraph import AdaptiveConstraintRanker

VERSION = 'm2m-reference-round2-gpu-v2'
SPECS = {
    'reference_dynamic': {'mode':'dynamic','history':True},
    'reference_static': {'mode':'static','history':True},
    'reference_pairwise': {'mode':'pairwise','history':True},
    'reference_fixed_sum': {'mode':'fixed_sum','history':True},
    'reference_mlp': {'mode':'mlp','history':True},
    'reference_no_history': {'mode':'dynamic','history':False},
}


def make_model(method):
    return AdaptiveConstraintRanker(**SPECS[method], hidden=64, layers=3,
                                    heads=4, threshold=0.)


def check_admission(directory):
    """Require the actual independently replayed public-source receipt."""
    directory = Path(directory)
    receipt = read(directory/'admission.json')
    if (receipt.get('status') != 'pass' or receipt.get('validation_replays') != 320
            or receipt.get('development_replays') != 1600
            or receipt.get('physical_replay_passed') is not True
            or receipt.get('formal_test_accessed') is not False):
        raise ValueError('Public baseline admission is incomplete')
    for field, name in [('source_audit_sha256','source_audit.json'),
                        ('contract_sha256','contract.json'),
                        ('selected_checkpoint_sha256','selected.checkpoint')]:
        if raw_hash(directory/name) != receipt[field]:
            raise ValueError('Public-source admission asset changed: '+name)
    return receipt


def check_binding(output, method, admission):
    binding = read(output/'binding.json')
    if (binding['version'] != VERSION or binding['development_round'] != 2
            or binding['seeds'] != [17] or binding['specs'] != SPECS
            or binding['planned_models'] != 6 or method not in SPECS
            or binding['config'] != {'device':'cuda','model_worker_hard_seconds':43200}
            or binding['formal_test_accessed'] is not False):
        raise ValueError('Changed development recipe, scope or budget')
    for name, expected in binding['source_sha256'].items():
        if raw_hash(ROOT/name) != expected:
            raise ValueError('Candidate source changed: '+name)
    labels = Path(binding['label_source'])
    for name, expected in binding['label_sha256'].items():
        path = (labels/name).resolve()
        if not path.is_relative_to(labels.resolve()) or raw_hash(path) != expected:
            raise ValueError('Preserved development label changed: '+name)
    check_admission(admission)
    # New model families cannot be substituted into the launched old contract.
    # This receipt is produced only after a fresh, scoped queue reconciliation.
    handoff = read(output.parent/'queue_handoff.json')
    if (handoff.get('version') != VERSION or handoff.get('old_queue_reconciled') is not True
            or handoff.get('formal_test_accessed') is not False
            or handoff.get('binding_sha256') != raw_hash(output/'binding.json')
            or handoff.get('old_contract_sha256') != binding['old_contract_sha256']
            or not handoff.get('observation_evidence')):
        raise ValueError('Candidate needs a versioned pre-test queue handoff')
    return binding


def check_resume(output, job, binding):
    from Agent.Publication.training_store import TrainingStore
    store = TrainingStore(output/'models'/job['id'])
    if store.exists('summary.json') or not store.exists('attempt.json'):
        return
    attempt = store.get('attempt.json')
    if alive(attempt['identity']):
        raise ValueError('The exact candidate training process is already alive')
    receipt = read(output/'active_resume_receipt.json')
    expected = canonical_hash(dict(job=job, binding=binding, version=VERSION))
    if (receipt.get('contract_hash') != expected or receipt.get('job_id') != job['id']
            or receipt.get('original_started') != attempt['at']
            or not receipt.get('diagnosis') or receipt.get('attempt') not in (1,2)
            or receipt.get('checkpoint_sha256') != raw_hash(store.path('latest.checkpoint'))):
        raise ValueError('Resume needs diagnosis and the unchanged cumulative deadline')


def load_selected(output, method, *, version=VERSION, model_factory=make_model):
    from Agent.Publication.training_store import TrainingStore
    store = TrainingStore(Path(output)/'models'/(method+'_s17'))
    summary = store.get('summary.json')
    if raw_hash(store.path('selected.checkpoint')) != summary['checkpoint_sha256']:
        raise ValueError('Selected candidate checkpoint changed')
    saved = store.load_model('selected.checkpoint')
    if (saved['source_version'] != version or saved['method'] != method
            or saved['reference_spec'] != SPECS[method]
            or saved['full_training_completed'] is not True
            or saved['contract_hash'] != summary['contract_hash']):
        raise ValueError('Wrong model family or incomplete training')
    model = model_factory(method)
    model.load_state_dict(saved['model']); model.eval()
    return model, summary


def train_model(output, job, config, *, version=VERSION, model_factory=make_model, material_loader=prepare_material):
    import torch
    from Agent.Publication.training_store import TrainingStore
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.20)
    torch.manual_seed(job['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(job['seed'])
    random.seed(job['seed'])
    device = config['device']
    variant, ablation = job['method'], None
    model = model_factory(job['method']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    store = TrainingStore(output/'models'/job['id'])
    binding = read(output/'binding.json')
    contract_hash = canonical_hash(dict(job=job, binding=binding, version=version))
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
        from Agent.Run.m2m_gpu_dispatch import cache_material
        material=cache_material(material_loader(Path(binding['label_source']),model),device)
        dump(store.path('input_preparation.json'),dict(at=time.time(),
            status='completed',identity=identity(),resume_epoch=resume_epoch+1,resume_position=resume_position,
            original_started=started,contract_hash=contract_hash,
            seconds=time.time()-preparation_started,train_states=len(material['train']),
            validation_states=len(material['validation']),cached='Immutable feature tensors on CUDA; physical integer indices remain on CPU',
            cuda_peak_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None))
        for epoch in range(resume_epoch,100):
            if bad>=15:break
            epoch_start = time.time()
            order = list(range(len(material['train'])))
            random.Random(job['seed']+epoch*1009).shuffle(order)
            model.train(); train_losses = list(resume_losses) if epoch==resume_epoch else []
            for number, idx in enumerate(order):
                if epoch==resume_epoch and number<resume_position:continue
                if (output/'STOP_REQUESTED').exists():
                    raise RuntimeError('User pause preserved; no automatic resume')
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
                    if time.time()-started>config['model_worker_hard_seconds']:
                        raise RuntimeError('Original cumulative model deadline exhausted')
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
                reference_spec=SPECS[job['method']],selected_epoch=best_epoch,full_training_completed=True,contract_hash=contract_hash,
                selection='minimum validation supervised loss; no test access',source_version=version))
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
    from Agent.Publication.m2m_action_ranker import NeuralActionPolicy
    from Agent.Publication.m2m_public_learning import load_case
    from Agent.Publication.training_store import TrainingStore
    torch.set_num_threads(1)
    m=read(batch/'cases'/(task['case_id']+'.json'))
    if m['split']!='validation':
        raise ValueError('Development evaluation accepts validation only')
    problem,groups,_=load_case(batch,m,task['initial'])
    job=task['model'];model,summary=load_selected(output,job['method'])
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

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check','train','evaluate'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--admission',type=Path,required=True)
    parser.add_argument('--method',choices=list(SPECS),default='reference_dynamic')
    parser.add_argument('--task-id')
    args=parser.parse_args(); output=args.output.resolve()
    binding=check_binding(output,args.method,args.admission.resolve())
    job={'id':args.method+'_s17','method':args.method,'seed':17}
    check_resume(output,job,binding)
    if args.mode=='check':
        print({'status':'ready','job':job,'training_started':False,'formal_test_accessed':False})
    elif args.mode=='train':
        print(train_model(output,job,binding['config']))
    else:
        task=next(t for t in binding['validation_tasks'] if t['id']==args.task_id)
        if task['model'] != job:
            raise ValueError('Validation task belongs to a different candidate')
        print(evaluate_model(Path(binding['batch']),output,task,binding['config']))


if __name__=='__main__':
    main()
