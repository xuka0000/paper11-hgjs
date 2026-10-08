"""Frozen seed17 formal execution, with explicit scientific and reference admission.

Initialization enumerates tasks without opening test cases. Workers never train
or select models. All optimization and checked output return share one clock;
the subsequent independent audit cannot replace an algorithm's output.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace as NS
import psutil

from Agent.Run.m2m_result_storage import read_result, write_result

VERSION = 'm2m-formal-seed17-v1'
GATES = ('G0','G1','G2','baseline_admission','frozen_versions')
FORMAL_CLASSES = ('formal_test','formal_extrapolation','ablation')
ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,p)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def identity():
    p=psutil.Process();return {'pid':p.pid,'created':p.create_time()}


def alive(value):
    try:return abs(psutil.Process(value['pid']).create_time()-value['created'])<.1
    except (psutil.Error,TypeError,KeyError):return False


def stop_owned(value):
    """Terminate only a verified experiment process and its current descendants."""
    if not alive(value):return
    parent=psutil.Process(value['pid']);children=parent.children(recursive=True)
    for process in reversed(children):
        try:process.kill()
        except psutil.Error:pass
    try:parent.kill()
    except psutil.Error:pass


def begin_attempt(out,contract,resume_receipt=None):
    """No automatic recovery: preserve failure and require a bound diagnosis receipt."""
    if not (out/'launch.json').exists():return 1
    prior=read(out/'launch.json')
    if alive(prior['identity']):raise RuntimeError('Existing controller is live')
    previous=read(out/'progress.json')
    if any(alive(x['identity']) for x in previous.get('active',[])):raise RuntimeError('Existing worker is live')
    if (out/'observer.json').exists() and alive(read(out/'observer.json')['identity']):raise RuntimeError('Existing observer is live')
    if not resume_receipt:raise RuntimeError('Diagnosed bounded resume receipt required')
    r=read(resume_receipt);attempt=prior.get('attempt',1)+1
    if (r.get('previous_launch_sha256')!=sha(out/'launch.json') or r.get('contract_sha256')!=sha(out/'contract.json')
        or not r.get('diagnosis') or not r.get('bounded_recovery_authorized') or r.get('attempt')!=attempt
        or attempt>contract.get('max_resume_attempts',2)):
        raise ValueError('Invalid recovery receipt')
    history=out/'attempt_history'/str(prior.get('attempt',1));history.mkdir(parents=True,exist_ok=True)
    valid={t['task_id'] for t in contract['tasks']}
    for key in r.get('affected_task_ids',[]):
        if key not in valid:raise ValueError('Recovery task outside contract')
        path=out/'results'/(key+'.json.gz')
        if path.exists():
            row,metadata=read_result(out,key)
            if row.get('status')!='error':raise ValueError('Recovery cannot replace a healthy scientific result')
            dump(history/(key+'_failure.json'),{'row':row,'original_metadata':metadata});path.unlink()
        for folder in ['intents','ready']:
            p=out/folder/(key+'.json')
            if p.exists():dump(history/(key+'_'+folder+'.json'),read(p));p.unlink()
    for n in ['launch.json','progress.json','terminal.json','failure.json','observer_failure.json','observer.json']:
        p=out/n
        if p.exists():dump(history/n,read(p));p.unlink()
    dump(history/'resume_receipt.json',r)
    return attempt


def select_tasks(contract):
    rows=[r for r in contract['tasks'] if r['run_class'] in FORMAL_CLASSES and r['method_seed'] in (None,17)]
    refs=[r for r in contract['tasks'] if r['run_class']=='reference']
    counts=Counter(r['run_class'] for r in rows)
    if counts!={'formal_test':2560,'formal_extrapolation':640,'ablation':256} or len(refs)!=192:
        raise ValueError('Unexpected fixed first-round matrix')
    if len({r['task_id'] for r in rows+refs})!=3648:raise ValueError('Duplicate task identity')
    return rows,refs


def init(contract_path,out):
    original=read(contract_path);rows,refs=select_tasks(original)
    if out.exists():raise ValueError('Formal batch exists; reconcile its identity instead of initializing again')
    specs=[b for b in original['bases'] if b['split'] in ('test','extrapolation')]
    preparations=[{'task_id':'formal_prepare_'+b['base_id'],'run_class':'preparation','spec':b,
                   'seconds':600 if b['split']=='test' else 1800} for b in specs]
    document={'version':VERSION,'at':time.time(),'original_contract_sha256':sha(contract_path),
              'tasks':preparations+refs+rows,'planned':{'preparation':48,'reference':192,**dict(Counter(r['run_class'] for r in rows))},
              'active_method_seed':17,'deferred_formal':9984,'deferred_models':24,'deferred_validation':1536,
              'workers':4,'worker_startup_seconds':180,'return_grace_seconds':15,'max_resume_attempts':2,
              'scientific_gates':list(GATES),'source_root':str(ROOT),'formal_test_accessed':False,
              'clock_policy':'Frozen input/model loading and common initial certification precede clock; graph/encoding/search/repair and checked output return are inside. Later independent replay is outside and cannot modify output.',
              'statistical_seed':1701,'resamples':10000,'primary_contrasts':original['formal_analysis']['primary_contrasts'],
              'authorization':'User chose single-seed formal first round and then instructed Implement the plan; original authorization.json remains unchanged.',
              'algorithm_budget_changed':False,'overall_deadline':original['budgets']['absolute_deadline'],
              'independent_observation':'Remote experiment observer plus current session; no automatic scientific approval or cloud power operations'}
    out.mkdir(parents=True);dump(out/'contract.json',document);dump(out/'original_contract.json',original)
    dump(out/'progress.json',{'at':time.time(),'status':'waiting_scientific_gates','planned':document['planned'],
                             'completed':{},'active':[],'failed':0,'formal_test_accessed':False})
    return document


def check_freeze(out):
    freeze=read(out/'freeze.json')
    if freeze.get('contract_sha256')!=sha(out/'contract.json'):raise ValueError('Freeze belongs to another contract')
    gates=read(out/'gates.json')
    for name in GATES:
        row=gates.get(name,{})
        if row.get('status')!='pass' or not row.get('evidence'):
            raise ValueError('Scientific admission missing: '+name)
        for proof in row['evidence']:
            path=Path(proof['path'])
            if not path.is_absolute():path=ROOT/path
            if sha(path)!=proof['sha256']:raise ValueError('Changed gate evidence: '+name)
    if freeze.get('gates_sha256')!=sha(out/'gates.json'):raise ValueError('Changed scientific gate ledger')
    for row in freeze['artifacts']:
        path=Path(row['path']);path=path if path.is_absolute() else ROOT/path
        if sha(path)!=row['sha256']:raise ValueError('Frozen artifact changed: '+row['path'])
    required={'action_mlp','action_pairwise','action_factor','action_hypergraph','no_incumbent','legacy_pool','public_learned_lns'}
    if set(freeze.get('models',{}))!=required:raise ValueError('Missing frozen model mapping')
    return freeze


def load_case(out,key,initial):
    from Agent.Run.publication_committed_service_domain import Plan
    from Agent.Publication.component_objective import evaluate_plan
    from Agent.Publication.foundation_benchmark import WEIGHTS
    m=read(out/'cases'/(key+'.json'));p=out/'cases'/(key+'.pkl')
    if m['split'] not in ('test','extrapolation') or m['final_test'] is not True or sha(p)!=m['pickle_sha256']:
        raise ValueError('Unsealed or changed formal case')
    problem,groups,warm,declaration=pickle.loads(p.read_bytes())
    if declaration['split']!=m['split'] or declaration['final_test'] is not True:raise ValueError('Serialized formal split mismatch')
    plan=warm if initial=='warm' else Plan() if initial=='cold' else None
    if plan is None:raise ValueError('Unknown initial condition')
    problem.state=replace(problem.state,selected_arc_ids=frozenset(a.arc_id for a in plan.arcs),
                          selected_tuple_ids=frozenset(c.tuple_id for c in plan.copies))
    before=evaluate_plan(problem.domain,plan,WEIGHTS,groups)
    return problem,groups,plan,before,m


def prepare(out,task):
    from Agent.Publication.m2m_formal_benchmark import prepare_base
    from Agent.Publication.budget import Deadline
    dump(out/'ready'/(task['task_id']+'.json'),{'at':time.time(),'identity':identity(),'phase':'scientific_clock'})
    spec=task['spec'];generated=prepare_base({'base_id':spec['base_id'],'seed':spec['geometry_seed'],
                           'components':spec['components'],'topology':spec['topology']},Deadline(task['seconds']-5))
    cases=[]
    for key,p,g,w,checked,declaration in generated:
        declaration.update(split=spec['split'],final_test=True,geometry_seed=spec['geometry_seed'])
        data=pickle.dumps((p,g,w,declaration),protocol=5);target=out/'cases'/(key+'.pkl')
        if target.exists():raise ValueError('Formal case already materialized without committed preparation')
        target.parent.mkdir(exist_ok=True);tmp=target.with_suffix('.pkl.pending');tmp.write_bytes(data);tmp.replace(target)
        m={**declaration,'pickle_sha256':sha(target),'physical_passed':True,
           'warm_objective':[checked['primary'],checked['secondary']],'total_priority':sum(x.priority for x in g.groups)}
        dump(target.with_suffix('.json'),m);cases.append(key)
    return {'task':task,'status':'prepared','cases':cases,'at':time.time()}


def restore_model(out,freeze,key):
    import torch
    from Agent.Publication.training_store import TrainingStore
    from Agent.Publication.m2m_action_ranker import ActionRanker,NeuralActionPolicy
    row=freeze['models'][key];path=Path(row['path']);path=path if path.is_absolute() else ROOT/path
    if sha(path)!=row['sha256']:raise ValueError('Changed model')
    saved=TrainingStore(path.parent).load_model(path.name)
    if not saved.get('full_training_completed'):raise ValueError('Incomplete trained model')
    if key=='public_learned_lns':
        from Agent.Publication.zhou2023_learning import new_policy
        model=new_policy();model.load_state_dict(saved['model']);model.eval();return model
    model=ActionRanker(saved['variant'],saved['hidden'],saved['layers'],saved['ablation'])
    model.load_state_dict(saved['model']);model.eval();return NeuralActionPolicy(model,device='cpu')


def plan_ids(plan):
    return {'selected_arc_ids':sorted(a.arc_id for a in plan.arcs),
            'selected_tuple_ids':sorted(c.tuple_id for c in plan.copies)}


def compare(out,task,freeze):
    import torch
    torch.set_num_threads(1)
    from Agent.Publication.budget import BudgetExpired,Deadline
    from Agent.Publication.component_objective import evaluate_plan
    from Agent.Publication.foundation_benchmark import WEIGHTS,quota_tour
    from Agent.Publication.physical_search_baselines import _pool_domain
    from Agent.Publication.vehicle_lns_milp import solve_committed
    from Agent.Publication.joint_service_search import run_joint_search,EpsilonGreedy,SearchLimits
    from Agent.Publication.inspection_tsha_adapter import run_inspection_tsha_adapter
    key=f"{task['base_id']}_c{task['capacity']}_s{int(task['duty_scale']*100)}"
    p,g,initial,before,m=load_case(out,key,task.get('initial','warm'))
    method=task.get('method','SCIP_default');ablation=task.get('ablation');seed=task.get('method_seed') or 1701
    neural={'action_mlp','action_pairwise','action_factor','action_hypergraph'}
    modelkey=ablation if ablation in ('no_incumbent','legacy_pool') else method
    policy=restore_model(out,freeze,modelkey) if method in neural or method=='public_learned_lns' else None
    dump(out/'ready'/(task['task_id']+'.json'),{'at':time.time(),'identity':identity(),'phase':'scientific_clock'})
    deadline=Deadline(task['seconds']);p.deadline=deadline;selected=initial;details={};status='returned'
    try:
        if task['run_class']=='reference' or method=='SCIP_default':
            details=solve_committed(_pool_domain(p.domain,p.full_copies.values()),NS(objective_weights=WEIGHTS),
                incumbent=initial,deadline=deadline,primary_seconds=deadline.allowance(task['seconds'],reserve=1.),
                secondary_seconds=deadline.allowance(task['seconds'],reserve=1.),validation_reserve_s=1.,task_groups=g)
            solution=details.pop('selected');details.pop('candidate',None)
            if solution is None:raise ValueError('Reference/SCIP lost checked incumbent')
            selected=solution['plan']
        elif method=='quota_greedy':
            proposed,values=quota_tour(p.domain,g,deadline)
            if (values['primary'],values['secondary'])<(before['primary'],before['secondary']):selected=proposed
        elif method=='inspection_tsha':
            details=run_inspection_tsha_adapter(p,WEIGHTS,seed=seed,total_seconds=deadline.remaining(),task_groups=g)
            selected=p.plan(replace(p.state,selected_arc_ids=frozenset(details['selected_arc_ids']),
                                    selected_tuple_ids=frozenset(details['selected_tuple_ids'])))
        elif method=='public_learned_lns':
            from Agent.Publication.m2m_source_lns import run_source_lns
            details=run_source_lns(p,WEIGHTS,policy,seed,deadline,task_groups=g)
            selected=p.plan(replace(p.state,selected_arc_ids=frozenset(details['selected_arc_ids']),
                                    selected_tuple_ids=frozenset(details['selected_tuple_ids'])))
        elif method in neural or method in ('joint_heuristic','joint_epsilon_greedy'):
            if method=='joint_epsilon_greedy':policy=EpsilonGreedy(seed)
            limits=SearchLimits(releases=0) if ablation=='no_release' else SearchLimits(additions=1) if ablation=='single_add' else SearchLimits()
            details=run_joint_search(p,g,initial,WEIGHTS,deadline,policy=policy,limits=limits)
            selected=details.pop('plan')
        else:raise ValueError('Unregistered formal method')
        checked=evaluate_plan(p.domain,selected,WEIGHTS,g)
        raw_ids=plan_ids(selected);raw_obj=[checked['primary'],checked['secondary']]
    except BudgetExpired:
        status='budget_exhausted_return_initial';selected=initial;checked=before;raw_ids=plan_ids(initial);raw_obj=[before['primary'],before['secondary']]
    elapsed=deadline.elapsed()
    if elapsed>task['seconds']:
        selected=initial;checked=before;status='late_return_initial'
    return {'task':task,'at':time.time(),'status':status,'case_sha256':m['pickle_sha256'],
        'within_budget':elapsed<=task['seconds'],'total_seconds':elapsed,'objective':[checked['primary'],checked['secondary']],
        'initial':[before['primary'],before['secondary']],'raw_objective':raw_obj,'raw_plan':raw_ids,
        'total_priority':m['total_priority'],'completed_groups':checked['completed_groups'],
        'physical_passed':True,'details':details,**plan_ids(selected)}


def audit_row(out,task,row):
    if row['task']!=task:raise ValueError('Output task identity mismatch')
    if task['run_class']=='preparation':
        for key in row['cases']:load_case(out,key,'warm')
        return
    from Agent.Publication.m2m_public_learning import verify_recorded_comparison
    key=f"{task['base_id']}_c{task['capacity']}_s{int(task['duty_scale']*100)}"
    p,g,_,_,m=load_case(out,key,task.get('initial','warm'))
    if row['case_sha256']!=m['pickle_sha256']:raise ValueError('Output data hash mismatch')
    record=dict(row)
    if record['status']=='budget_exhausted_return_initial':record['status']='late_return_initial'
    verify_recorded_comparison(p,g,record)


def worker(out,task):
    freeze=check_freeze(out)
    if task['run_class'] in FORMAL_CLASSES and not (out/'reference_gate.json').is_file():
        raise ValueError('Formal result access forbidden before reference gate')
    result=prepare(out,task) if task['run_class']=='preparation' else compare(out,task,freeze)
    write_result(out,task['task_id'],result)


def dependencies_met(out,task,done):
    if task['run_class']=='preparation':return True
    if 'formal_prepare_'+task['base_id'] not in done:return False
    return task['run_class']=='reference' or (out/'reference_gate.json').is_file()


def reference_gate(out,contract,done):
    refs=[t for t in contract['tasks'] if t['run_class']=='reference']
    if not all(t['task_id'] in done for t in refs):return False
    if not (out/'reference_gate.json').exists():
        for t in refs:
            row,_=read_result(out,t['task_id']);audit_row(out,t,row)
        dump(out/'reference_gate.json',{'at':time.time(),'status':'pass','checked':192,
             'meaning':'All reference outputs have independent feasible physical witnesses; optimality/bounds remain per-solver statuses and are not inferred from completion.'})
    return True


def observe(out):
    launch=read(out/'launch.json')
    while not (out/'terminal.json').exists():
        p=read(out/'progress.json');now=time.time()
        dump(out/'observer.json',{'at':now,'identity':identity(),'parent_alive':alive(launch['identity'])})
        if not alive(launch['identity']) or now-p['at']>90:
            for row in p.get('active',[]):
                stop_owned(row['identity'])
            dump(out/'observer_failure.json',{'at':now,'reason':'controller_missing_or_stale'});return
        time.sleep(5)


def run(out,resume_receipt=None):
    check_freeze(out);contract=read(out/'contract.json')
    attempt=begin_attempt(out,contract,resume_receipt)
    done=set();completed={};active={};failed=0;error=None;observer=None
    for task in contract['tasks']:
        try:row,record=read_result(out,task['task_id'])
        except FileNotFoundError:continue
        if row.get('status')=='error':raise RuntimeError('Failed row requires diagnosed affected-row recovery')
        audit_row(out,task,row);done.add(task['task_id']);completed[task['task_id']]=record
    dump(out/'launch.json',{'at':time.time(),'identity':identity(),'attempt':attempt,'contract_sha256':sha(out/'contract.json')})
    def progress(status):
        by=Counter(t['run_class'] for t in contract['tasks'] if t['task_id'] in done)
        dump(out/'progress.json',{'at':time.time(),'status':status,'planned':contract['planned'],'completed':dict(by),
            'failed':failed,'active':[v['record'] for v in active.values()],
            'formal_test_accessed':bool(done or active),
            'formal_score_accessed':any(v in by for v in FORMAL_CLASSES),
            'failure':error})
        dump(out/'completed.json',completed)
    progress('running')
    try:
        observer=subprocess.Popen([sys.executable,'-B','-m',__name__ if __name__!='__main__' else 'Agent.Run.m2m_formal_campaign',
                    'observe','--output',str(out)],cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        while len(done)<len(contract['tasks']):
            if (out/'STOP_REQUESTED').exists():raise RuntimeError('User/diagnosed stop request; retain progress')
            if contract['overall_deadline'] is not None and time.time()>=contract['overall_deadline']:raise RuntimeError('Original deadline reached')
            reference_gate(out,contract,done)
            for key,item in list(active.items()):
                task=item['task'];p=item['process'];ready=out/'ready'/(key+'.json')
                limit=task['seconds']+contract['return_grace_seconds']
                started=read(ready)['at'] if ready.exists() else item['started']
                if p.poll() is None and time.time()-started>(limit if ready.exists() else contract['worker_startup_seconds']):
                    stop_owned(item['record']['identity']);p.wait();raise RuntimeError('Hard worker timeout: '+key)
                if p.poll() is None:continue
                item['stream'].close()
                if p.returncode:failed+=1;raise RuntimeError('Worker failed: '+key)
                row,record=read_result(out,key);audit_row(out,task,row);completed[key]=record;done.add(key);del active[key]
            for task in contract['tasks']:
                key=task['task_id']
                if len(active)>=contract['workers']:break
                if key in done or key in active or not dependencies_met(out,task,done):continue
                intent=out/'intents'/(key+'.json')
                if intent.exists():raise RuntimeError('Uncommitted task intent requires explicit diagnosis: '+key)
                dump(intent,{'at':time.time(),'task':task,'attempt':attempt})
                log=out/'logs'/(key+'.log');log.parent.mkdir(exist_ok=True);stream=log.open('a',encoding='utf-8')
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                p=subprocess.Popen([sys.executable,'-B','-m','Agent.Run.m2m_formal_campaign','worker','--output',str(out),'--task',key],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active[key]={'task':task,'process':p,'stream':stream,'started':time.time(),
                             'record':{'task_id':key,'identity':{'pid':p.pid,'created':psutil.Process(p.pid).create_time()}}}
            progress('running');time.sleep(5)
        progress('computed_pending_full_audit');dump(out/'terminal.json',{'at':time.time(),'status':'computed_pending_full_audit'})
    except Exception as exc:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        dump(out/'failure.json',error);progress('failed');dump(out/'terminal.json',{'at':time.time(),'status':'failed','failure':error});raise
    finally:
        for item in active.values():
            if item['process'].poll() is None:stop_owned(item['record']['identity']);item['process'].wait()
            item['stream'].close()
        if observer:
            try:observer.wait(timeout=15)
            except subprocess.TimeoutExpired:observer.terminate();observer.wait()


def audit(out):
    contract=read(out/'contract.json');counts=Counter();missing=[]
    for task in contract['tasks']:
        try:row,_=read_result(out,task['task_id'])
        except FileNotFoundError:missing.append(task['task_id']);continue
        audit_row(out,task,row);counts[task['run_class']]+=1
    result={'at':time.time(),'checked':dict(counts),'missing_count':len(missing),'missing':missing,
            'status':'pass' if not missing else 'partial','scientific_claim_accepted':False}
    dump(out/'audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['init','run','worker','observe','audit','status'])
    p.add_argument('--contract',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--task');p.add_argument('--resume-receipt',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='init':print(json.dumps({'planned':init(a.contract,out)['planned']}))
    elif a.mode=='run':run(out,a.resume_receipt)
    elif a.mode=='observe':observe(out)
    elif a.mode=='status':print(json.dumps(read(out/'progress.json'),ensure_ascii=False))
    elif a.mode=='audit':print(json.dumps(audit(out),ensure_ascii=False))
    else:
        task=next(t for t in read(out/'contract.json')['tasks'] if t['task_id']==a.task)
        try:worker(out,task)
        except Exception:
            write_result(out,a.task,{'task':task,'at':time.time(),'status':'error','traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
