"""Frozen-controller branches from identical observed rolling execution states."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dataclasses import asdict,is_dataclass
from hashlib import sha256
from datetime import datetime,timezone
import argparse,csv,gzip,json,os
import torch

from Agent.Run.publication_rolling_roads import run_rolling,SegmentClosure
from Agent.Run.publication_late_dynamic import pack
from Agent.Run.evaluate_publication_late_dynamic import finite_json,digest,TRAINING_SHA
from Agent.Tool.Benchmark.formal_v3_loader import FormalV3InstanceLoader
from Agent.Tool.Training.chdgrl_config import load_chdgrl_experiment_config
from Agent.Tool.Evaluation.chdgrl_methods import build_chdgrl_methods,evaluation_source_hashes
from Agent.Baseline.exact_milp import ExactMilpBaseline
from Agent.Tool.Evaluation.method_runner import MethodSpec
from Env.Execution import execute_due_events,advance_state,PlannedLaunchEvent

NAME='publication_rolling_roads_20260908'
BASE=ROOT/'Data/Result/Diagnostic'/NAME
PHASE=ROOT/'Data/Paper'/NAME
DEV=ROOT/'Data/Instance/Benchmark/chdgrl_physical_v2/development_manifest.jsonl'
TEST=ROOT/'Data/Instance/Benchmark/chdgrl_final_guard8_v1/confirmatory_manifest.jsonl'
CONFIG=ROOT/'Data/Config/chdgrl_lns/scale_normalized_v3_reward_bounds.json'
LAYERS=tuple(range(1,122,8))


def decode(value):
    if isinstance(value,list):return [decode(x) for x in value]
    if not isinstance(value,dict):return value
    if 'map' in value:return {decode(k):decode(v) for k,v in value['map']}
    if 'set' in value:return frozenset(decode(x) for x in value['set'])
    if 'tuple' in value:return tuple(decode(x) for x in value['tuple'])
    if 'type' in value:
        import importlib
        module,name=value['type'].rsplit('.',1)
        allowed={'Env.State','Env.Execution','Env.Runtime','Mask.UAV','Mask.Vehicle','Mask.ServiceTuple',
                 'Agent.Run.publication_rolling_roads','Agent.Run.publication_road_recovery','Agent.Tool.Evaluation.dynamic_evaluation'}
        if module not in allowed:raise ValueError('Unexpected trace type: '+value['type'])
        return getattr(importlib.import_module(module),name)(**{k:decode(v) for k,v in value['fields'].items()})
    raise ValueError('Unregistered trace representation')


def write(path,value):path.write_text(json.dumps(finite_json(value),indent=2,allow_nan=False)+'\n')


def encoded_record(record):
    result=dict(record)
    if 'result' in result:
        value=result['result']
        result['result']={name:getattr(value,name,None) for name in ('selected_arc_ids','selected_tuple_ids','best_primary','best_secondary','validation_ok','status','runtime_s','primary_status','secondary_status','primary_bound','secondary_bound')}
        result['config']=asdict(result['config'])
    return pack(result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['validation','test'],default='validation');parser.add_argument('--limit-instances',type=int)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import psutil
    process=psutil.Process();process.cpu_affinity([process.cpu_affinity()[0]])
    output=BASE/(args.split+('_smoke' if args.limit_instances else ''))
    output.mkdir(parents=True,exist_ok=True);(output/'traces').mkdir(exist_ok=True);PHASE.mkdir(parents=True,exist_ok=True)
    dev=FormalV3InstanceLoader(DEV);loader=dev if args.split=='validation' else FormalV3InstanceLoader(TEST)
    config=load_chdgrl_experiment_config(CONFIG)
    if args.split=='validation':
        chosen={}
        for iid in loader.instance_ids:
            row=loader.row_for(iid)
            if row.split=='validation':chosen.setdefault((row.difficulty,row.topology_family),iid)
        ids=sorted(chosen.values());seeds=[config.training_seeds[0]]
    else:
        validation=json.loads((BASE/'validation/independent_validation.json').read_text())
        assert validation['status']=='PASS' and validation['rows']==204
        old=json.loads((ROOT/'Data/Result/Diagnostic/publication_late_dynamic_20260908/test/contract.json').read_text())
        ids=old['instance_ids'];seeds=list(config.training_seeds)
    if args.limit_instances:ids=ids[:args.limit_instances]
    sources=evaluation_source_hashes(ROOT)
    for name in ('publication_rolling_roads.py','evaluate_publication_rolling_roads.py','publication_road_recovery.py','publication_late_dynamic.py','evaluate_publication_late_dynamic.py'):
        sources['Agent/Run/'+name]=digest(ROOT/'Agent/Run'/name)
    weights={str(seed):digest(ROOT/f'Data/Checkpoint/chdgrl_final_guard8_v1/seed_{seed}/frozen.pt') for seed in seeds}
    contract=dict(version='rolling_roads_fixed_service_allowance_v1',split=args.split,instance_ids=ids,seeds=seeds,
        source_hashes=sources,checkpoint_hashes=weights,manifest=(DEV if args.split=='validation' else TEST).relative_to(ROOT).as_posix(),
        manifest_sha256=digest(DEV if args.split=='validation' else TEST),config_sha256=digest(CONFIG),training_source_sha256=TRAINING_SHA,
        rows=len(ids)*len(seeds)*17,decision_layers=list(LAYERS),terminal128=True,continuation256=True,
        scenarios=['middle17','final17','middle33','final33'],arms=['continued_fixed','continued_detour','emergency_detour','milp_detour'],
        common_prefix='One calm frozen-controller trajectory per physical instance and training seed. Branches start from identical state and persistent road memory immediately before the event.',
        events='Permanent physical edge closures. Middle is floor(n/2) using zero-based indexing, final is n-1 in vehicle1 original tour. Targets affect every vehicle; occupied segments may finish.',
        candidates='Only originally admitted physical template IDs. New absolute-time copies on the constructed route; unchanged physical endpoints and task sequence. Airborne UAVs become eligible for new copies after observed recovery and readiness.',
        resources='Fixed preparation+recovery allowance at stops used by physical templates. Preparation precedes launch. Ground waiting consumes duty. New-copy moments scale by scheduled flight duration divided by physical candidate duration, a constant average-power surrogate.',
        routing='Retain timed open prefix and persist new physical edges. Replace a blocked future leg by the first forward reachable reconnection, without skipping an airborne recovery stop. One constructed route per vehicle; no global road-path optimum claim.',
        failure='Stop a failed branch at its recorded state; do not execute an infeasible plan or fabricate an emergency maneuver.',
        timing='Legacy dynamic limits, not a common external-deadline benchmark. MILP gets0.8s per lexicographic solve.',
        independence='Physical instance. Seeds and closure settings are repeated measurements. Previously examined diagnostic instances, not a fresh holdout.')
    if (output/'contract.json').exists():assert json.loads((output/'contract.json').read_text())==contract,'Contract differs'
    else:write(output/'contract.json',contract)
    for relative,h in sources.items():
        archive=output/'sources'/relative;archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():archive.write_bytes((ROOT/relative).read_bytes())
        assert digest(archive)==h
    summary_path=output/'summary.csv';existing=list(csv.DictReader(summary_path.open())) if summary_path.exists() else []
    done={r['key']:r for r in existing}
    def save(key,result,instance,seed,scenario,arm,prefix_path=''):
        records=[encoded_record(r) for r in result.records]
        doc=dict(key=key,instance_hash=instance.instance_hash,records=records,
            final_state=pack(result.state),pending128=pack(result.pending),failure=result.failure,prefix_trace=prefix_path)
        continued=None
        if result.feasible:
            extension,pending=execute_due_events(result.state,result.pending,256)
            continued=advance_state(result.state,extension)
            assert extension.safety_status=='feasible' and not pending
            assert not continued.airborne_services and not any(v.must_return or v.active_arc is not None for v in continued.vehicles.values())
            doc['state256']=pack(continued)
        token=sha256(key.encode()).hexdigest()[:24];path=output/'traces'/(token+'.json.gz')
        path.write_bytes(gzip.compress(json.dumps(finite_json(doc),separators=(',',':'),allow_nan=False).encode(),mtime=0))
        start=result.records[0]['state'];new_launches=[]
        for r in result.records:
            if 'execution' not in r:continue
            new_launches.extend(e for e in r['entries'] if isinstance(e,PlannedLaunchEvent) and e.launch_layer<=r['execution'].end_layer)
        row=dict(key=key,instance_id=instance.instance_id,instance_hash=instance.instance_hash,difficulty=instance.difficulty,
            topology=instance.topology_family,seed=seed,scenario=scenario,arm=arm,feasible=result.feasible,failure=result.failure,
            last_layer=result.state.current_layer,completed_at_start=len(start.done_tasks),completed_at_end=len(result.state.done_tasks),
            completed_after128=len(continued.done_tasks-result.state.done_tasks) if continued else '',
            launched_after_branch=len(new_launches),replans=sum('result' in r for r in result.records),
            return_open128=sum(v.must_return or v.active_arc is not None for v in result.state.vehicles.values()) if result.feasible else '',
            airborne128=len(result.state.airborne_services) if result.feasible else '',
            checkpoint_sha256=weights[str(seed)],trace_path=path.relative_to(ROOT).as_posix(),trace_sha256=digest(path))
        fresh=not summary_path.exists()
        with summary_path.open('a',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row))
            if fresh:writer.writeheader()
            writer.writerow(row)
        done[key]=row
        write(output/'progress.json',dict(status='running',completed=len(done),expected=contract['rows'],active_pid=os.getpid(),last=key))
        return row
    for iid in ids:
        instance=loader(iid)
        original=sorted((a for a in instance.search_state.arcs if a.vehicle_id==1),key=lambda a:a.departure_layer)
        edgepairs={(e.tail,e.head):e.edge_id for e in instance.physical_scenario.road.edges}
        for seed in seeds:
            def learned():
                methods,_=build_chdgrl_methods(instance,ROOT/f'Data/Checkpoint/chdgrl_final_guard8_v1/seed_{seed}/frozen.pt',config,
                    dev.manifest_sha256,'rolling_roads_legacy',include_exact=False,expected_source_sha256=TRAINING_SHA)
                return methods['chdgrl_lns']
            key='|'.join([iid,str(seed),'calm','prefix'])
            if key not in done:
                result=run_rolling(instance,learned(),config.search.to_search_config())
                assert result.feasible,'Calm prefix failed: '+result.failure
                save(key,result,instance,seed,'calm','prefix')
            prefix_row=done[key];prefix_path=ROOT/prefix_row['trace_path']
            assert digest(prefix_path)==prefix_row['trace_sha256']
            document=json.loads(gzip.decompress(prefix_path.read_bytes()))
            prefix={r['layer']:r for r in map(decode,document['records'])}
            for layer in (17,33):
                for target in ('middle','final'):
                    source_arc=original[len(original)//2] if target=='middle' else original[-1]
                    closure=(SegmentClosure(edgepairs[(source_arc.tail,source_arc.head)],layer),)
                    scenario=target+str(layer)
                    for arm in contract['arms']:
                        key='|'.join([iid,str(seed),scenario,arm])
                        if key in done:continue
                        method=MethodSpec(ExactMilpBaseline(max_tasks=64,time_limit_s=.8)) if arm=='milp_detour' else learned()
                        result=run_rolling(instance,method,config.search.to_search_config(),closure,
                            allow_detour=arm!='continued_fixed',continue_after_closure=arm!='emergency_detour',
                            layers=tuple(k for k in LAYERS if k>=layer),initial_state=prefix[layer]['state'],initial_memory=prefix[layer]['memory_before'])
                        save(key,result,instance,seed,scenario,arm,prefix_row['trace_path'])
                print(json.dumps(dict(completed=len(done),expected=contract['rows'],instance=iid,seed=seed,layer=layer)),flush=True)
    assert len(done)==contract['rows']
    write(output/'progress.json',dict(status='complete',completed=len(done),expected=contract['rows'],active_pid=None,updated_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
