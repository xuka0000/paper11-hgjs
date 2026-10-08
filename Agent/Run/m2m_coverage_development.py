"""One bounded, training-witness action-coverage development round (seed17)."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
from Agent.Run.m2m_reference_development_gpu import SPECS,make_model,train_model,load_selected,check_admission
from Agent.Run.m2m_own_pipeline import examples,raw_hash,canonical_hash
from Agent.Run.m2m_formal_campaign import read,dump
from Agent.Publication.m2m_action_dataset import iter_state_records,context_from_record

VERSION='m2m-action-coverage-v3'
CORE=('reference_dynamic','reference_static','reference_pairwise','reference_mlp')
REST=('reference_fixed_sum','reference_no_history')


def check_binding(out,admission):
    b=read(out/'binding.json')
    if (b['version']!=VERSION or b['development_round']!=2 or b['seeds']!=[17]
            or b['specs']!=SPECS or b['planned_models']!=6 or len(b['validation_tasks'])!=384
            or b['config']!={'device':'cuda','model_worker_hard_seconds':43200}
            or b['formal_test_accessed'] is not False):raise ValueError('Changed bounded recipe')
    from Agent.Run.m2m_own_pipeline import ROOT
    for name,expected in b['source_sha256'].items():
        if raw_hash(ROOT/name)!=expected:raise ValueError('Changed source: '+name)
    for field in ('coverage_audit','supplement_labels'):
        if raw_hash(Path(b[field]))!=b[field+'_sha256']:raise ValueError('Changed training audit/labels')
    audit=read(b['coverage_audit']);supp=read(b['supplement_labels'])
    if (audit['bounds']!=b['search_limits'] or audit['independent_validation_used_for_cap_selection'] is not False
            or supp['validation_supplements']!=0 or supp['new_replay_calls']>128
            or supp['old_nominal_reserved_calls']+supp['new_replay_calls']>15360):raise ValueError('Training/label budget mismatch')
    for item in audit['original_label_manifests']:
        from pathlib import PurePosixPath
        relative=item.get('label_relpath',str(PurePosixPath('labels')/PurePosixPath(item['path'].replace('\\','/')).parent.name/'manifest.json'))
        if raw_hash(Path(b['label_source'])/relative)!=item['sha256']:raise ValueError('Original labels changed')
    check_admission(admission)
    return b


def prepare_augmented(source,model,supplement):
    extra=read(supplement)['records'];seen=set();material={'train':[],'validation':[]};counts={k:0 for k in material}
    for folder in sorted((source/'labels').iterdir()):
        if not folder.is_dir():continue
        for r in iter_state_records(folder):
            split=r['split'];key=r['state_hash']
            if key in seen or split not in material:raise ValueError('Invalid state split/identity')
            seen.add(key);counts[split]+=1
            if key in extra:
                addition=extra[key]
                if split!='train' or addition['state_id']!=r['state_id'] or addition['case_sha256']!=r['case_sha256']:
                    raise ValueError('Supplement changes state identity or leaks validation')
                # Labels enter targets only. Context is reconstructed from the
                # unchanged input view/incumbent, never the successful plan.
                r={**r,'proposals':r['proposals']+addition['proposals'],
                     'continuations':r.get('continuations',[])+addition['continuations']}
            actions,labels=examples(r)
            if actions:material[split].append((model.encode(context_from_record(r)),actions,labels))
    if not set(extra)<=seen or counts!={'train':417,'validation':104} or not all(material.values()):
        raise ValueError('Changed original label population')
    return material


def train(out,method,admission):
    b=check_binding(out,admission)
    if method in REST and read(out/'benefit_gate_core.json').get('status')!='pass':
        raise ValueError('Remaining ablations need scheduling-benefit evidence')
    job=dict(id=method+'_s17',method=method,seed=17)
    return train_model(out,job,b['config'],version=VERSION,
        material_loader=lambda source,model:prepare_augmented(source,model,Path(b['supplement_labels'])))


def evaluate(out,task,admission):
    import torch
    from Agent.Publication.budget import Deadline
    from Agent.Publication.component_objective import evaluate_plan
    from Agent.Publication.foundation_benchmark import WEIGHTS
    from Agent.Publication.joint_service_search import run_joint_search
    from Agent.Publication.m2m_coverage_search import CoverageLimits,coverage_proposals
    from Agent.Publication.m2m_action_ranker import NeuralActionPolicy
    from Agent.Publication.m2m_public_learning import load_case
    from Agent.Run.m2m_result_storage import write_result
    torch.set_num_threads(1);b=check_binding(out,admission);batch=Path(b['batch'])
    m=read(batch/'cases'/(task['case_id']+'.json'))
    if m['split']!='validation' or m.get('final_test'):raise ValueError('Development test leakage')
    p,g,_=load_case(batch,m,task['initial']);method=task['model']['method']
    model,summary=load_selected(out,method,version=VERSION)
    initial=p.plan(p.state);before=evaluate_plan(p.domain,initial,WEIGHTS,g)
    start=time.perf_counter();clock=Deadline(60.);p.deadline=clock
    result=run_joint_search(p,g,initial,WEIGHTS,clock,policy=NeuralActionPolicy(model,device='cpu'),
        limits=CoverageLimits(**b['search_limits']),proposal_generator=coverage_proposals)
    selected=result.pop('plan');checked=evaluate_plan(p.domain,selected,WEIGHTS,g)
    if any(abs(a-c)>1e-6 for a,c in zip(result['objective'],(checked['primary'],checked['secondary']))):
        raise ValueError('Own search result differs from independent physical check')
    elapsed=time.perf_counter()-start;raw_plan=dict(selected_arc_ids=sorted(a.arc_id for a in selected.arcs),
        selected_tuple_ids=sorted(c.tuple_id for c in selected.copies));raw_obj=[checked['primary'],checked['secondary']]
    if elapsed>60.:selected=initial;checked=before
    total=sum(q.priority for q in g.groups)
    row=dict(task=task,at=time.time(),status='returned' if elapsed<=60 else 'late_return_initial',
        physical_passed=True,within_budget=elapsed<=60,total_seconds=elapsed,total_priority=total,
        initial=[before['primary'],before['secondary']],objective=[checked['primary'],checked['secondary']],
        raw_objective=raw_obj,raw_plan=raw_plan,completed_groups=checked['completed_groups'],
        selected_arc_ids=sorted(a.arc_id for a in selected.arcs),selected_tuple_ids=sorted(c.tuple_id for c in selected.copies),
        details=result,checkpoint_sha256=summary['checkpoint_sha256'],case_sha256=m['pickle_sha256'],
        source_version=VERSION,search_limits=b['search_limits'],formal_test_accessed=False)
    write_result(out/'validation',task['id'],row);return row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','evaluate','check'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--method',choices=list(SPECS),default=CORE[0]);p.add_argument('--task-id');a=p.parse_args()
    if a.mode=='train':print(train(a.output,a.method,a.admission))
    elif a.mode=='check':print({'status':'ready','version':check_binding(a.output,a.admission)['version']})
    else:
        b=check_binding(a.output,a.admission);t=next(t for t in b['validation_tasks'] if t['id']==a.task_id)
        print({k:v for k,v in evaluate(a.output,t,a.admission).items() if k!='details'})
