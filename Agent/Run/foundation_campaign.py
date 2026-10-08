"""Frozen numerical queue; baseline foundation, without training or test access.

One coordinator, one numerical worker, one independent process observer. Unknown
errors stop the queue. Deadline failures retain the common initial solution and
allow only the next preregistered independent comparison task. No retries.
"""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import time
import traceback

import psutil


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    # Windows readers can briefly hold a file without delete sharing. Retrying
    # this idempotent metadata rename neither repeats a solver nor resets time.
    for attempt in range(21):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            if attempt == 20: raise
            time.sleep(.05)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def alive(identity):
    try: return abs(psutil.Process(identity['pid']).create_time()-identity['created']) < .01
    except psutil.Error: return False


def identity(pid=None):
    p = psutil.Process(pid or os.getpid())
    return dict(pid=p.pid, created=p.create_time())


def alert(out, kind, **details):
    event = dict(at=time.time(), kind=kind, **details)
    # Coordinator and observer each have their own file: no concurrent writes.
    name = 'observer_alerts.jsonl' if details.get('observer') else 'alerts.jsonl'
    with (out/name).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False)+'\n')
    print(json.dumps(event, ensure_ascii=False), flush=True)


def init(root, out):
    from Agent.Publication.foundation_benchmark import base_specs, WEIGHTS, VERSION
    out.mkdir(parents=True, exist_ok=False)
    archive = out/'source_executed'; sources = {}
    for directory in ('Agent', 'Mask', 'Env'):
        for path in sorted((root/directory).rglob('*.py')):
            rel = path.relative_to(root); target = archive/rel
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
            sources[rel.as_posix()] = digest(target)
    cases = [f'{s["base_id"]}_c{c}_s{k}' for s in base_specs() for c in (1, 2) for k in (75, 125)]
    methods = ('SCIP_default', 'vehicle_random', 'inspection_tsha', 'quota_greedy')
    tasks = [dict(id='prepare_'+s['base_id'], phase='prepare', spec=s, seconds=180.) for s in base_specs()]
    tasks += [dict(id='reference_'+c, phase='reference', case=c, method='SCIP_default',
                   initial='warm', seed=1701, seconds=300.) for c in cases]
    for i, case in enumerate(cases):
        for initial in ('cold', 'warm'):
            for seed in (1701, 2701):
                for method in methods[i % 4:] + methods[:i % 4]:
                    tasks.append(dict(id=f'{case}_{initial}_{seed}_{method}', phase='comparison',
                                      case=case, method=method, initial=initial, seed=seed, seconds=15.))
    contract = dict(version=VERSION, created_at=time.time(), bases=base_specs(), independent_bases=16,
        cases=cases, planned_cases=64, methods=methods, planned_references=64, planned_comparisons=1024,
        tasks=tasks, weights=WEIGHTS, source_sha256=sources,
        absolute_duration_seconds=16*3600, queue_task_allowances_seconds=sum(t['seconds'] for t in tasks),
        deadline_policy='Single attempt. Unknown errors stop. Comparison timeout retains initial; proceed to next fixed task.',
        source_policy='Execute archived sources; refuse source or completed artifact hash changes.',
        scoring='Lexicographic weighted incomplete components then checked physical cost. Never convert timeout to infeasibility.',
        catalogue='Full finite single-task domain; no three-time sampling. Does not claim arbitrary multi-task sorties.',
        initialization='Cold empty; warm deterministic quota tour shared by every method, constructed before reference solves.',
        timing='Common input construction/loading excluded and reported; action timer covers algorithm, construction of decisions and validation. Warm startup cost reported separately; comparisons conditional on common input.',
        reference_gate='All 64 independent physical replays and nonempty MILP row witnesses required. Optimality is reported separately and not required.',
        adaptation='Vehicle-random Zhou2023 and TSHA Xia2025 are disclosed common-domain adaptations, not admitted full source reproductions. Quota greedy is a deterministic construction control.',
        inference='16 independent base geometries, four paired conditions; two seeds are repeats, not independent samples. Development only; no superiority test or outcome-dependent retuning.',
        observer='Independent local process checks coordinator identity and heartbeat; alerts.jsonl/observer_alerts.jsonl and logs. No verified external push or out-of-session scientific diagnosis.',
        continuation='No model tuning, training, test evaluation, retries or resource creation. Stop at all tasks complete, model gate failure, unknown exception, observer loss or absolute deadline.',
        training_runs=0, final_test_accessed=0,
        environment=dict(python=sys.version, platform=platform.platform(), executable=sys.executable))
    dump(out/'contract.json', contract)
    print(json.dumps(dict(output=str(out), tasks=len(tasks), cases=64, references=64, comparisons=1024)))


def plan_ids(plan):
    return dict(selected_arc_ids=sorted(a.arc_id for a in plan.arcs),
                selected_tuple_ids=sorted(c.tuple_id for c in plan.copies))


def worker(out, task):
    # Heavy imports are outside the algorithm clock and common to all methods.
    os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    from types import SimpleNamespace as NS
    from Agent.Publication.budget import Deadline, BudgetExpired
    from Agent.Publication.foundation_benchmark import prepare_base, quota_tour, WEIGHTS
    from Agent.Publication.component_objective import evaluate_plan
    from Agent.Publication.physical_search_baselines import _pool_domain
    from Agent.Publication.milp import solve_committed
    from Agent.Publication.zhou2023_vehicle_random import run_vehicle_random, VehicleRandomSettings
    from Agent.Publication.inspection_tsha_adapter import run_inspection_tsha_adapter
    from Agent.Run.publication_committed_service_domain import Plan
    result_path = out/'results'/(task['id']+'.json')
    begin = time.perf_counter()
    if task['phase'] == 'prepare':
        dump(out/'ready.json', dict(task=task['id'], at=time.time(), identity=identity()))
        prepared = prepare_base(task['spec'], Deadline(task['seconds']-5))
        cases = []
        for key, problem, model, warm, checked, declaration in prepared:
            target = out/'cases'/(key+'.pkl'); target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(pickle.dumps((problem, model, warm, declaration), protocol=5))
            summary = dict(declaration, pickle_sha256=digest(target),
                initial_objective=[checked['primary'], checked['secondary']],
                completed_groups=checked['completed_groups'], warm_plan=plan_ids(warm),
                empty_objective=[sum(g.priority for g in model.groups), 0.],
                physical_passed=True)
            dump(out/'cases'/(key+'.json'), summary); cases.append(summary)
        result = dict(task=task, status='prepared', cases=cases,
                      preparation_seconds=time.perf_counter()-begin, at=time.time())
        dump(result_path, result); return
    case_path = out/'cases'/(task['case']+'.pkl'); manifest = read(out/'cases'/(task['case']+'.json'))
    if digest(case_path) != manifest['pickle_sha256']: raise ValueError('Changed frozen case')
    problem, model, warm, declaration = pickle.loads(case_path.read_bytes())
    initial = warm if task['initial'] == 'warm' else Plan()
    problem.state = replace(problem.state, selected_arc_ids=frozenset(a.arc_id for a in initial.arcs),
                            selected_tuple_ids=frozenset(c.tuple_id for c in initial.copies))
    before = evaluate_plan(problem.domain, initial, WEIGHTS, model)
    load_seconds = time.perf_counter()-begin
    dump(out/'ready.json', dict(task=task['id'], at=time.time(), identity=identity()))
    deadline = Deadline(task['seconds']); problem.deadline = deadline
    selected = initial; details = {}; status = 'returned'
    try:
        if task['method'] == 'SCIP_default':
            allowance = deadline.allowance(task['seconds'], reserve=1.)
            details = solve_committed(_pool_domain(problem.domain, problem.full_copies.values()),
                NS(objective_weights=WEIGHTS), deadline=deadline, primary_seconds=allowance,
                secondary_seconds=allowance, validation_reserve_s=1., incumbent=initial, task_groups=model)
            solution = details.pop('selected')
            if solution is None: raise ValueError('Solver lost verified initial solution')
            selected = solution['plan']
            details['selected_validated_seconds'] = solution['validated_at']-deadline.started
        elif task['method'] == 'vehicle_random':
            settings = VehicleRandomSettings(seed=task['seed'], total_seconds=deadline.allowance(task['seconds'], reserve=1.))
            details = run_vehicle_random(problem, WEIGHTS, settings, task_groups=model)
        elif task['method'] == 'inspection_tsha':
            details = run_inspection_tsha_adapter(problem, WEIGHTS, seed=task['seed'],
                total_seconds=deadline.allowance(task['seconds'], reserve=1.), task_groups=model)
        elif task['method'] == 'quota_greedy':
            proposed, values = quota_tour(problem.domain, model, deadline)
            if (values['primary'], values['secondary']) < (before['primary'], before['secondary']): selected = proposed
            details = dict(control='deterministic quota tour, no solver or learned score')
        else: raise ValueError('Unregistered method')
        if task['method'] in ('vehicle_random', 'inspection_tsha'):
            selected = problem.plan(replace(problem.state,
                selected_arc_ids=frozenset(details['selected_arc_ids']),
                selected_tuple_ids=frozenset(details['selected_tuple_ids'])))
        checked = evaluate_plan(problem.domain, selected, WEIGHTS, model)
        if 'objective' in details and any(abs(a-b)>1e-6 for a,b in zip(
                details['objective'], (checked['primary'], checked['secondary']))):
            raise ValueError('Independent component/physical score mismatch')
    except BudgetExpired:
        status = 'budget_exhausted_return_initial'; selected = initial
        checked = evaluate_plan(problem.domain, selected, WEIGHTS, model)
    seconds = deadline.elapsed()
    raw_objective = [checked['primary'], checked['secondary']]
    raw_ids = plan_ids(selected)
    if seconds > task['seconds']:
        status = 'late_return_initial'; selected = initial; checked = before
    result = dict(task=task, status=status, at=time.time(), source='frozen_archive',
        case_sha256=manifest['pickle_sha256'], load_seconds=load_seconds, total_seconds=seconds,
        within_budget=seconds <= task['seconds'], initial=[before['primary'], before['secondary']],
        objective=[checked['primary'], checked['secondary']], raw_objective=raw_objective,
        completed_groups=checked['completed_groups'], physical_passed=True,
        raw_plan=raw_ids, details=details, **plan_ids(selected))
    dump(result_path, result)


def reference_gate(out, contract):
    from dataclasses import replace
    from Agent.Publication.component_objective import evaluate_plan
    from Agent.Publication.foundation_benchmark import WEIGHTS
    rows = []
    for key in contract['cases']:
        record = read(out/'results'/('reference_'+key+'.json'))
        manifest = read(out/'cases'/(key+'.json'))
        path = out/'cases'/(key+'.pkl')
        if digest(path) != manifest['pickle_sha256']: raise ValueError('Case changed at reference gate')
        p, m, _, _ = pickle.loads(path.read_bytes())
        plan = p.plan(replace(p.state, selected_arc_ids=frozenset(record['selected_arc_ids']),
                             selected_tuple_ids=frozenset(record['selected_tuple_ids'])))
        replay = evaluate_plan(p.domain, plan, WEIGHTS, m)
        if any(abs(a-b)>1e-6 for a,b in zip(record['objective'], (replay['primary'], replay['secondary']))):
            raise ValueError('Reference gate replay mismatch')
        hint = record['details'].get('incumbent_hint', {})
        if not hint.get('accepted') or hint.get('violation_count') != 0:
            raise ValueError('No independently physical-checked MILP row witness')
        if not replay['completed_groups']: raise ValueError('Lost nonempty reference incumbent')
        rows.append(dict(case=key, physical_replay=True, row_witness=True,
                         objective=record['objective'], stages=record['details']['stages']))
    dump(out/'reference_gate.json', dict(at=time.time(), status='pass', cases=rows,
         meaning='Physical and model-row consistency on 64 cases; not a universal proof or field validation'))


def coordinator(out, resume_receipt=None):
    contract = read(out/'contract.json')
    receipt = read(resume_receipt) if resume_receipt else None
    archive = out/(receipt['archive'] if receipt else 'source_executed')
    expected_sources = receipt['source_sha256'] if receipt else contract['source_sha256']
    for rel, expected in expected_sources.items():
        if digest(archive/rel) != expected: raise ValueError('Archived source changed')
    if (out/'launch.json').exists() and not receipt:
        raise ValueError('Single attempt campaign; explicit diagnosed resume required')
    completed = read(out/'completed_tasks.json') if receipt else []
    if receipt:
        if digest(out/'contract.json') != receipt['contract_sha256']: raise ValueError('Changed contract')
        for item in completed:
            if digest(out/'results'/(item['id']+'.json')) != item['sha256']:
                raise ValueError('Changed completed result')
        if len(completed) != receipt['completed_tasks']: raise ValueError('Changed resume position')
    start = receipt['original_started'] if receipt else time.time()
    limit = start+contract['absolute_duration_seconds']
    launch = dict(identity=identity(), at=start, absolute_deadline=limit,
                  resumed_at=time.time() if receipt else None)
    dump(out/'launch.json', launch)
    with (out/'observer.log').open('w', encoding='utf-8') as stream:
        observer = subprocess.Popen([sys.executable, '-m', 'Agent.Run.foundation_campaign', 'observe', '--output', str(out)],
            cwd=archive, stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    child = None; current = None; status = 'running'
    try:
        for registered in contract['tasks']:
            if registered['id'] in {r['id'] for r in completed}: continue
            task = dict(registered)
            if receipt and task['id'] in receipt['remaining_task_seconds']:
                task['registered_seconds'] = task['seconds']
                task['seconds'] = receipt['remaining_task_seconds'][task['id']]
                task['previous_interrupted_seconds'] = receipt['charged_interrupted_seconds']
            current = task
            if time.time()+task['seconds']+90 >= limit: raise TimeoutError('Absolute campaign limit')
            if observer.poll() is not None: raise RuntimeError('Independent observer stopped')
            if task['phase'] == 'comparison' and not (out/'reference_gate.json').exists():
                reference_gate(out, contract)
            dump(out/'active_task.json',task)
            with (out/'worker.log').open('a', encoding='utf-8') as stream:
                child = subprocess.Popen([sys.executable, '-m', 'Agent.Run.foundation_campaign', 'worker',
                    '--output', str(out), '--task', task['id']], cwd=archive, stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            child_identity = identity(child.pid); launched = time.time(); ready_at = None
            while child.poll() is None:
                now = time.time()
                ready_path = out/'ready.json'
                if ready_path.exists():
                    ready = read(ready_path)
                    if ready['task'] == task['id']: ready_at = ready['at']
                progress = dict(status='running', observed_at=now, completed=len(completed),
                    planned=len(contract['tasks']), current=task, worker=child_identity,
                    coordinator=launch['identity'], observer=identity(observer.pid),
                    started=start, absolute_deadline=limit, ready_at=ready_at,
                    completed_preparations=sum(r['phase']=='prepare' for r in completed),
                    completed_references=sum(r['phase']=='reference' for r in completed),
                    completed_comparisons=sum(r['phase']=='comparison' for r in completed))
                dump(out/'progress.json', progress)
                if observer.poll() is not None: raise RuntimeError('Independent observer stopped')
                if now >= limit: raise TimeoutError('Absolute campaign limit')
                if ready_at is None and now-launched > 90: raise TimeoutError('Worker input/startup timeout')
                if ready_at and now-ready_at > task['seconds']+15:
                    child.kill(); child.wait();
                    raise TimeoutError('Worker exceeded hard return limit; preserve attempt, diagnose before continuation')
                time.sleep(1)
            result_path = out/'results'/(task['id']+'.json')
            if child.returncode or not result_path.exists(): raise RuntimeError('Worker failed: '+task['id'])
            result = read(result_path)
            if result.get('status') == 'error': raise RuntimeError('Worker error: '+task['id'])
            completed.append(dict(id=task['id'], phase=task['phase'], sha256=digest(result_path),
                                  status=result['status']))
            dump(out/'completed_tasks.json', completed)
            alert(out, 'task_completed', task=task['id'], completed=len(completed), status=result['status'],
                  objective=result.get('objective'))
        status = 'completed'; dump(out/'completion.json', dict(at=time.time(), completed=completed))
    except Exception as exc:
        status = 'stopped_requires_diagnosis'
        failure = dict(at=time.time(), type=type(exc).__name__, message=str(exc),
                       traceback=traceback.format_exc(), task=current, completed=len(completed))
        dump(out/'failure.json', failure); alert(out, 'failure_stop', **failure)
    finally:
        if child is not None and child.poll() is None: child.kill(); child.wait()
        dump(out/'progress.json', dict(status=status, observed_at=time.time(), completed=len(completed),
            planned=len(contract['tasks']), started=start, absolute_deadline=limit, current=current,
            completed_preparations=sum(r['phase']=='prepare' for r in completed),
            completed_references=sum(r['phase']=='reference' for r in completed),
            completed_comparisons=sum(r['phase']=='comparison' for r in completed)))
        dump(out/'coordinator_terminal.json', dict(status=status, at=time.time()))


def observe(out):
    launch = read(out/'launch.json'); last_alert = None
    while True:
        now = time.time(); p = out/'progress.json'
        current = read(p) if p.exists() else {}
        dump(out/'observer_heartbeat.json', dict(at=now, identity=identity(),
            coordinator_alive=alive(launch['identity']), progress_at=current.get('observed_at')))
        if (out/'coordinator_terminal.json').exists():
            alert(out, 'coordinator_terminal_observed', observer=True, terminal=read(out/'coordinator_terminal.json')); return
        issue = None
        if not alive(launch['identity']): issue = 'coordinator_missing'
        elif now-current.get('observed_at', launch['at']) > 45: issue = 'stale_progress'
        elif now > launch['absolute_deadline']: issue = 'absolute_deadline'
        if issue and issue != last_alert:
            alert(out, issue, observer=True, progress=current); last_alert = issue
        if issue in ('coordinator_missing', 'absolute_deadline'):
            # Stop only the exact owned worker identity; never an arbitrary PID.
            owned = current.get('worker')
            if owned and alive(owned): psutil.Process(owned['pid']).kill()
            dump(out/'observer_failure.json', dict(at=now, issue=issue)); return
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('init', 'run', 'worker', 'observe'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--task')
    parser.add_argument('--resume-receipt', type=Path)
    args = parser.parse_args(); out = args.output.resolve()
    if args.mode == 'init': init(args.root.resolve(), out)
    elif args.mode == 'run': coordinator(out,args.resume_receipt)
    elif args.mode == 'observe': observe(out)
    else:
        task = next(t for t in read(out/'contract.json')['tasks'] if t['id'] == args.task)
        # Current task records the reviewed remaining budget after an interrupted
        # attempt. A worker must not silently use the original larger allowance.
        current_path=out/'active_task.json'
        if current_path.exists():
            current=read(current_path)
            if current['id']==args.task: task=current
        try: worker(out, task)
        except Exception as exc:
            dump(out/'results'/(task['id']+'.json'), dict(task=task, status='error', at=time.time(),
                error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__': main()
