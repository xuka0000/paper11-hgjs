"""Registered long-operation mechanism panel with common complete-MILP prefixes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dataclasses import asdict
from datetime import datetime,timezone
from hashlib import sha256
import argparse,csv,gzip,json,os
import torch,psutil
from Agent.Run.publication_long_operations import generate,scenarios,run,LAYERS,SETTINGS
from Agent.Run.evaluate_publication_rolling_roads import decode,encoded_record,write,digest,CONFIG,DEV,TRAINING_SHA
from Agent.Run.publication_late_dynamic import pack
from Agent.Run.evaluate_publication_late_dynamic import finite_json
from Agent.Tool.Benchmark.formal_v3_loader import FormalV3InstanceLoader
from Agent.Tool.Training.chdgrl_config import load_chdgrl_experiment_config
from Agent.Tool.Evaluation.chdgrl_methods import build_chdgrl_methods,evaluation_source_hashes
from Agent.Baseline.exact_milp import ExactMilpBaseline
from Env.Execution import PlannedLaunchEvent

NAME='publication_long_operations_20260909'
BASE=ROOT/'Data/Result/Diagnostic'/NAME
PHASE=ROOT/'Data/Paper'/NAME


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['validation','test'],default='validation');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    process=psutil.Process();process.cpu_affinity([process.cpu_affinity()[0]])
    output=BASE/args.split;output.mkdir(parents=True,exist_ok=True);(output/'traces').mkdir(exist_ok=True);PHASE.mkdir(parents=True,exist_ok=True)
    config=load_chdgrl_experiment_config(CONFIG);dev=FormalV3InstanceLoader(DEV)
    seeds=[config.training_seeds[0]] if args.split=='validation' else list(config.training_seeds)
    if args.split=='test':
        validation=json.loads((BASE/'validation/independent_validation.json').read_text())
        assert validation['status']=='PASS' and validation['rows']==252
    instances=[generate(args.split,i) for i in range(len(SETTINGS))]
    sources=evaluation_source_hashes(ROOT)
    for relative in ['Agent/Run/'+p for p in ('publication_long_operations.py','evaluate_publication_long_operations.py','publication_rolling_roads.py','publication_road_recovery.py','publication_late_dynamic.py','evaluate_publication_rolling_roads.py','evaluate_publication_late_dynamic.py','validate_publication_rolling_roads.py')]:sources[relative]=digest(ROOT/relative)
    weights={str(seed):digest(ROOT/f'Data/Checkpoint/chdgrl_final_guard8_v1/seed_{seed}/frozen.pt') for seed in seeds}
    n=len(instances)*(1+5*(3*len(seeds)+1))
    contract=dict(version='long_operations_registered_geometry_clocks_v1',split=args.split,rows=n,seeds=seeds,settings=SETTINGS,
        instances=[dict(index=i,instance_id=x.instance_id,instance_sha256=x.instance_hash,scenario_sha256=x.scenario_hash,candidate_pool_sha256=x.candidate_pool_hash,events={k:[asdict(c) for c in v] for k,v in scenarios(x).items()}) for i,x in enumerate(instances)],
        source_hashes=sources,checkpoint_hashes=weights,training_source_sha256=TRAINING_SHA,training_manifest_sha256=dev.manifest_sha256,config_sha256=digest(CONFIG),
        decision_layers=LAYERS,terminal=256,planning_horizon=256,road_weather_coefficients=[0.,0.],
        scenario='Eight inspection stages, one UAV per vehicle, one or two vehicles on a shared directed staged road. Each stage has an existing bypass. Synthetic geometry, tasks, service times and energy parameters are declared inputs. No claim of measured fleet calibration or broad topology coverage.',
        event_rule='Close direct legs of stages3,5,7 separately or sequentially. Compute nominal launch from original physical road lengths, integer travel integration and fixed service waits. Reveal at the first decision layer strictly after nominal launch, before approach arrival. Freeze before any optimization. A cut control closes both the direct leg and bypass entrance of stage5.',
        prefix='One complete-MILP calm trajectory per physical instance, independent of learned outcomes. All arms share its exact state and road memory before their first event. This isolates continuation and does not evaluate learned prefix quality.',
        arms=['continued_fixed','continued_detour','emergency_detour','milp_detour'],
        resources='Use the prior isolated rolling road builder unchanged, including constant average-power duration scaling, fixed preparation/recovery stop allowance, replaced batteries, residual energy and persistent physical edge IDs.',
        comparison='Legacy dynamic search limits. Complete MILP0.8s per lexicographic solve. No common-deadline speed or superiority claim. Repeated seed and event settings aggregate within physical instances.',
        failure='Retain explicit failures, do not execute invalid plans and do not impute terminal completion values.',
        test_rule='Twelve newly generated cases from the same declared structured family. Case settings and both seed lists fixed by generator before test runs. Not a representative sample of real road disruption.')
    contract=json.loads(json.dumps(contract))
    if (output/'contract.json').exists():assert json.loads((output/'contract.json').read_text())==contract,'Immutable contract differs'
    else:write(output/'contract.json',contract)
    for relative,h in sources.items():
        dest=output/'sources'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.write_bytes((ROOT/relative).read_bytes())
        assert digest(dest)==h
    (output/'instances').mkdir(exist_ok=True)
    for index,x in enumerate(instances):
        path=output/'instances'/f'{index:02d}.json';data=x.physical_scenario.canonical_json()
        if path.exists():assert path.read_text()==data
        else:path.write_text(data)
    summary=output/'summary.csv';done={r['key']:r for r in csv.DictReader(summary.open())} if summary.exists() else {}
    def save(key,x,seed,scenario,arm,result,prefix=''):
        assert not result.feasible or (result.state.current_layer==256 and not result.pending and not result.state.airborne_services and not any(v.must_return or v.active_arc is not None for v in result.state.vehicles.values()))
        doc=dict(key=key,instance_hash=x.instance_hash,records=[encoded_record(r) for r in result.records],final_state=pack(result.state),pending=pack(result.pending),failure=result.failure,prefix_trace=prefix)
        path=output/'traces'/(sha256(key.encode()).hexdigest()[:24]+'.json.gz')
        path.write_bytes(gzip.compress(json.dumps(finite_json(doc),separators=(',',':'),allow_nan=False).encode(),mtime=0))
        start=result.records[0]['state'];launches=sum(isinstance(e,PlannedLaunchEvent) and e.launch_layer<=r['execution'].end_layer for r in result.records if 'execution' in r for e in r['entries'])
        row=dict(key=key,instance_id=x.instance_id,instance_hash=x.instance_hash,seed=seed,scenario=scenario,arm=arm,feasible=result.feasible,failure=result.failure,start_layer=start.current_layer,last_layer=result.state.current_layer,completed_at_start=len(start.done_tasks),completed_at_stop=len(result.state.done_tasks),launches_after_branch=launches,trace_path=path.relative_to(ROOT).as_posix(),trace_sha256=digest(path))
        fresh=not summary.exists()
        with summary.open('a',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row))
            if fresh:writer.writeheader()
            writer.writerow(row)
        done[key]=row;write(output/'progress.json',dict(status='running',completed=len(done),expected=n,active_pid=os.getpid(),last=key))
        return row
    for x in instances:
        exact=lambda:ExactMilpBaseline(max_tasks=64,time_limit_s=.8)
        key=x.instance_id+'|milp|calm|prefix'
        if key not in done:
            result=run(x,exact(),config.search.to_search_config());assert result.feasible,'Calm prefix failed: '+result.failure
            save(key,x,'milp','calm','prefix',result)
        prefixrow=done[key];path=ROOT/prefixrow['trace_path'];assert digest(path)==prefixrow['trace_sha256']
        prefix={r['layer']:r for r in map(decode,json.loads(gzip.decompress(path.read_bytes()))['records'])}
        for name,events in scenarios(x).items():
            layer=events[0].reveal_layer;initial=prefix[layer]
            arms=[(seed,arm) for seed in seeds for arm in ('continued_fixed','continued_detour','emergency_detour')]+[('milp','milp_detour')]
            for seed,arm in arms:
                key='|'.join((x.instance_id,str(seed),name,arm))
                if key in done:continue
                if arm=='milp_detour':method=exact()
                else:
                    methods,_=build_chdgrl_methods(x,ROOT/f'Data/Checkpoint/chdgrl_final_guard8_v1/seed_{seed}/frozen.pt',config,dev.manifest_sha256,'long_operations_legacy',include_exact=False,expected_source_sha256=TRAINING_SHA)
                    method=methods['chdgrl_lns']
                result=run(x,method,config.search.to_search_config(),events,allow_detour=arm!='continued_fixed',allow_new=arm!='emergency_detour',initial_state=initial['state'],initial_memory=initial['memory_before'],layers=tuple(k for k in LAYERS if k>=layer))
                save(key,x,seed,name,arm,result,prefixrow['trace_path'])
            print(json.dumps(dict(instance=x.instance_id,scenario=name,completed=len(done),expected=n)),flush=True)
    assert len(done)==n
    write(output/'progress.json',dict(status='complete',completed=len(done),expected=n,active_pid=None,updated_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
