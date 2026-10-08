"""Finite numerical GPU dispatch over existing pinned project SSH transport.

No provider call, resource start, retry, Codex worker, model selection or test
access. Original public training is outside every owned process tree here.
"""
import argparse
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import psutil
from Agent.Run.m2m_hg_reference_formal import read,dump,sha,identity,alive,stop_owned

ROOT=Path(__file__).resolve().parents[2]

GPU_PYTHON='/root/miniconda3/envs/experiment_env_gpu/bin/python'
CPU_SITE='/root/miniconda3/envs/paper11/lib/python3.11/site-packages'
FEATURES=('service_features','factor_features','edge_features')


def cache_material(material,device):
    """Copy immutable input features; retain CPU IDs used by Python action lists."""
    count=sum(getattr(e,name).numel()*getattr(e,name).element_size()
              for rows in material.values() for e,a,y in rows for name in FEATURES)
    if count>4*1024**3:raise ValueError('Immutable CUDA feature cache exceeds reserved4GiB per model')
    return {split:[(replace(e,**{name:getattr(e,name).to(device) for name in FEATURES}),a,y)
                   for e,a,y in rows] for split,rows in material.items()}


def bootstrap(module,args):
    code="import site,runpy,sys;site.addsitedir("+repr(CPU_SITE)+");sys.argv="+repr([module,*map(str,args)])+";runpy.run_module("+repr(module)+",run_name='__main__')"
    return [GPU_PYTHON,'-B','-c',code]


def ssh_command(transport,arguments):
    if transport['host_alias']!='paper11-sat1':raise ValueError('Only already owned CUDA sat1 is bound')
    if sha(transport['ssh_config'])!=transport['ssh_config_sha256']:raise ValueError('Existing pinned SSH configuration changed')
    command='cd '+shlex.quote(transport['source_root'])+' && '+' '.join(shlex.quote(str(x)) for x in arguments)
    return ['ssh','-T','-F',transport['ssh_config'],'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
            '-o','ConnectTimeout=5','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',transport['host_alias'],command]


def existing_child_timeout(progress,now):
    for value in progress['active']:
        if now-value['started']>value['seconds']:raise RuntimeError('Unchanged per-task wall budget exhausted: '+value['id'])


def group_observe(output):
    launch=read(output/'gpu_group_launch.json')
    while not (output/'gpu_group_terminal.json').exists():
        state=read(output/'gpu_group_progress.json');now=time.time()
        parent=alive(launch['identity'])
        dump(output/'gpu_group_observer.json',dict(at=now,identity=identity(),parent_alive=parent,interval_seconds=5))
        if not parent or now-state['at']>30 or now-state['lease_at']>30:
            for row in state['active']:stop_owned(row['identity'])
            dump(output/'gpu_group_observer_failure.json',dict(at=now,reason='controller_or_master_lease_lost',
                 public_baseline_changed=False,automatic_retry=False));return
        time.sleep(5)


def gpu_group(output):
    from Agent.Run.m2m_reference_development_gpu import SPECS,check_binding,check_admission,VERSION
    if (output/'gpu_group_launch.json').exists():raise ValueError('Existing GPU task identity; no duplicate group')
    check_admission(output/'admission')
    check_binding(output/'development','reference_dynamic',output/'admission')
    me=identity();started=time.time();lease=[started,False];active={};finished=[];failure=None;observer=None
    dump(output/'gpu_group_launch.json',dict(at=started,identity=me,version=VERSION,models=list(SPECS),
        parallel_models=3,hard_seconds_per_model=43200,seed=17,scientific_budget_changed=False))
    def listen():
        for line in sys.stdin:
            try:
                value=json.loads(line)
                if value.get('group_binding')!=sha(output/'development/binding.json'):break
                lease[0]=time.time()
            except Exception:break
        lease[1]=True
    threading.Thread(target=listen,daemon=True).start()
    def snapshot(status='running'):
        dump(output/'gpu_group_progress.json',dict(at=time.time(),identity=me,status=status,lease_at=lease[0],
            active=[dict(id=k,identity=v['identity'],started=v['started'],seconds=43200) for k,v in active.items()],
            completed=finished,planned=6,parallel_models=3,formal_test_accessed=False))
    pending=list(SPECS)
    try:
        snapshot()
        observer=subprocess.Popen(bootstrap('Agent.Run.m2m_gpu_dispatch',['gpu_observe','--output',output]),cwd=ROOT)
        while pending or active:
            now=time.time()
            if lease[1] or now-lease[0]>30:raise RuntimeError('Master numerical lease lost; no retry')
            if observer.poll() is not None:raise RuntimeError('GPU numerical observer exited')
            if (output/'STOP_REQUESTED').exists():raise RuntimeError('User pause preserved')
            if shutil.disk_usage(output).free<5*1024**3:raise RuntimeError('Remote 5GiB floor reached')
            existing_child_timeout(read(output/'gpu_group_progress.json'),now)
            for method,value in list(active.items()):
                child=value['process']
                if child.poll() is not None:
                    if child.returncode:raise RuntimeError('GPU training failed: '+method)
                    from Agent.Run.m2m_reference_development_gpu import load_selected
                    load_selected(output/'development',method)
                    finished.append(method);del active[method]
            while pending and len(active)<3:
                method=pending.pop(0);log=output/'logs'/('gpu_'+method+'.log');log.parent.mkdir(exist_ok=True)
                with log.open('x') as stream:
                    child=subprocess.Popen(bootstrap('Agent.Run.m2m_reference_development_gpu',[
                        'train','--output',output/'development','--admission',output/'admission','--method',method]),
                        cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1'))
                active[method]=dict(process=child,identity=dict(pid=child.pid,created=psutil.Process(child.pid).create_time()),started=time.time())
            snapshot();time.sleep(5)
    except BaseException as error:
        failure=dict(at=time.time(),type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        dump(output/'gpu_group_failure.json',failure)
    finally:
        for value in active.values():
            if value['process'].poll() is None:stop_owned(value['identity']);value['process'].wait()
        active.clear();snapshot('failed' if failure else 'completed')
        dump(output/'gpu_group_terminal.json',dict(at=time.time(),identity=me,status='failed' if failure else 'completed',
            failure=failure,completed=finished,public_baseline_changed=False,formal_test_accessed=False))
        if observer:
            try:observer.wait(timeout=10)
            except subprocess.TimeoutExpired:observer.terminate();observer.wait()
    if failure:raise RuntimeError(failure['message'])


def receive_admission(output):
    """Receive only the four already admitted, independently checked assets."""
    incoming=output/'admission'
    if incoming.exists():raise ValueError('Prior admission delivery; inspect instead of repeating')
    incoming.mkdir()
    raw=sys.stdin.buffer.read(32*1024**2+1)
    if len(raw)>32*1024**2:raise ValueError('Unexpected public checkpoint transfer size')
    expected={'admission.json','source_audit.json','contract.json','selected.checkpoint'};seen=set()
    with tarfile.open(fileobj=io.BytesIO(raw)) as stream:
        for member in stream:
            if member.name not in expected or member.name in seen or not member.isfile():raise ValueError('Unsafe admission member')
            seen.add(member.name)
            with stream.extractfile(member) as source,(incoming/member.name).open('xb') as dest:shutil.copyfileobj(source,dest)
    if seen!=expected:raise ValueError('Incomplete admission transfer')
    from Agent.Run.m2m_reference_development_gpu import check_admission
    check_admission(incoming)
    print(json.dumps(dict(status='verified',files=4,formal_test_accessed=False)))


def assets_manifest(output):
    terminal=read(output/'gpu_group_terminal.json')
    if terminal['status']!='completed' or len(terminal['completed'])!=6:raise ValueError('GPU group is incomplete')
    rows=[]
    for path in sorted((output/'development/models').rglob('*')):
        if path.is_file() and path.name not in ['latest.checkpoint','process.lock']:
            rows.append(dict(path=path.relative_to(output/'development').as_posix(),sha256=sha(path),bytes=path.stat().st_size))
    total=sum(r['bytes'] for r in rows)
    if total>256*1024**2:raise ValueError('GPU asset peak exceeds reservation')
    value=dict(at=time.time(),files=rows,bytes=total,gpu_terminal=terminal)
    dump(output/'gpu_assets_manifest.json',value)
    return value


def send_models(output):
    manifest=assets_manifest(output)
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
        raw=json.dumps(manifest).encode();entry=tarfile.TarInfo('manifest.json');entry.size=len(raw);archive.addfile(entry,io.BytesIO(raw))
        for row in manifest['files']:archive.add(output/'development'/row['path'],arcname=row['path'],recursive=False)


def dispatch(output):
    from Agent.Run.m2m_reference_development_gpu import check_admission,SPECS,load_selected
    submission=read(output/'submission.json');transport=submission['gpu_transport']
    admission=Path(submission['admission']);check_admission(admission)
    if (output/'gpu_dispatch_launch.json').exists():raise ValueError('Existing GPU dispatch; no duplicate submission')
    dump(output/'gpu_dispatch_launch.json',dict(at=time.time(),identity=identity(),transport=transport,planned=6,
         formal_test_accessed=False,cloud_operations=False))
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        for name in ['admission.json','source_audit.json','contract.json','selected.checkpoint']:
            archive.add(admission/name,arcname=name,recursive=False)
    command=ssh_command(transport,bootstrap('Agent.Run.m2m_gpu_dispatch',['receive_admission','--output',transport['remote_root']]))
    r=subprocess.run(command,input=buffer.getvalue(),capture_output=True,timeout=30)
    if r.returncode:raise RuntimeError('Verified admission transfer failed: '+r.stderr.decode()[-1000:])
    dump(output/'gpu_admission_delivery.json',json.loads(r.stdout))
    log=output/'logs/gpu_dispatch.log';child=None
    try:
        with log.open('x') as stream:
            child=subprocess.Popen(ssh_command(transport,bootstrap('Agent.Run.m2m_gpu_dispatch',[
                'gpu_group','--output',transport['remote_root']])),stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT)
        binding_sha=sha(output/'development/binding.json');started=time.time()
        while child.poll() is None:
            if time.time()-started>90000:raise RuntimeError('Two-wave original cumulative training ceiling reached')
            if (output/'STOP_REQUESTED').exists():raise RuntimeError('User pause preserved')
            child.stdin.write((json.dumps(dict(group_binding=binding_sha))+'\n').encode());child.stdin.flush()
            dump(output/'gpu_dispatch_progress.json',dict(at=time.time(),identity=identity(),ssh_identity=dict(pid=child.pid,created=psutil.Process(child.pid).create_time()),
                state='waiting_gpu_group',lease_interval_seconds=5,remote_root=transport['remote_root'],formal_test_accessed=False))
            time.sleep(5)
        if child.returncode:raise RuntimeError('Finite GPU group failed; inspect exact logs; no retry')
    finally:
        if child:
            if child.stdin:child.stdin.close()
            if child.poll() is None:child.terminate();child.wait(timeout=15)
    pending=output/'development/.gpu_transfer_pending'
    if pending.exists():raise ValueError('Prior transfer exists; no implicit retry')
    pending.mkdir();total=0;manifest=None
    child=subprocess.Popen(ssh_command(transport,bootstrap('Agent.Run.m2m_gpu_dispatch',[
        'send_models','--output',transport['remote_root']])),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=child.stdout,mode='r|') as archive:
            for member in archive:
                path=pending/member.name
                if not member.isfile() or not path.resolve().is_relative_to(pending.resolve()):raise ValueError('Unsafe GPU result member')
                total+=member.size
                if total>256*1024**2:raise ValueError('Result transfer exceeds bound')
                path.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as source,path.open('xb') as dest:shutil.copyfileobj(source,dest)
        if child.wait(timeout=30):raise RuntimeError('GPU checkpoint preservation transport failed')
        manifest=read(pending/'manifest.json')
        actual={p.relative_to(pending).as_posix() for p in pending.rglob('*') if p.is_file() and p.name!='manifest.json'}
        if actual!={r['path'] for r in manifest['files']}:raise ValueError('Unexpected preserved model coverage')
        for row in manifest['files']:
            p=pending/row['path']
            if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('GPU result digest changed')
        target=output/'development/models'
        if target.exists():raise ValueError('A different candidate model set already exists')
        (pending/'models').rename(target)
        dump(output/'gpu_independent_preservation.json',manifest)
        (pending/'manifest.json').unlink();pending.rmdir()
        for method in SPECS:load_selected(output/'development',method)
        dump(output/'gpu_dispatch_terminal.json',dict(at=time.time(),status='completed_and_preserved',models=6,
            bytes=manifest['bytes'],independent_destination=str(target),public_baseline_changed=False,formal_test_accessed=False))
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=15)


def validate_cpu(output):
    """Four existing-budget validation slots, with CPU inference for every row."""
    from Agent.Run.m2m_result_storage import read_result
    binding=read(output/'development/binding.json');admission=read(output/'submission.json')['admission']
    pending=[];active={};completed=0
    for row in binding['validation_tasks']:
        try:result,_=read_result(output/'development/validation',row['id'])
        except FileNotFoundError:pending.append(row);continue
        if result['task']!=row:raise ValueError('Validation task identity changed')
        completed+=1
    try:
        while pending or active:
            if (output/'STOP_REQUESTED').exists():raise RuntimeError('User pause preserved')
            for key,value in list(active.items()):
                if time.time()-value['started']>255:raise RuntimeError('Original255second validation process budget reached')
                child=value['process']
                if child.poll() is not None:
                    if child.returncode:raise RuntimeError('Validation worker failed: '+key)
                    result,_=read_result(output/'development/validation',key)
                    if result['task']!=value['task']:raise ValueError('Wrong committed validation task')
                    completed+=1;del active[key]
            while pending and len(active)<4:
                task=pending.pop(0)
                with (output/'logs'/('validation_'+task['id']+'.log')).open('x') as stream:
                    child=subprocess.Popen([sys.executable,'-B','-m','Agent.Run.m2m_reference_development_gpu','evaluate',
                        '--output',str(output/'development'),'--admission',admission,'--method',task['model']['method'],'--task-id',task['id']],
                        cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1'))
                active[task['id']]=dict(process=child,task=task,started=time.time(),identity=dict(pid=child.pid,created=psutil.Process(child.pid).create_time()))
            dump(output/'cpu_validation_progress.json',dict(at=time.time(),identity=identity(),completed=completed,planned=384,
                active=[dict(id=k,identity=v['identity']) for k,v in active.items()],parallel_workers=4,device='cpu',formal_test_accessed=False))
            time.sleep(1)
    finally:
        for value in active.values():
            if value['process'].poll() is None:stop_owned(value['identity']);value['process'].wait()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['dispatch','gpu_group','gpu_observe','receive_admission','send_models','validate_cpu'])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    globals()[{'gpu_observe':'group_observe'}.get(a.mode,a.mode)](a.output.resolve())
