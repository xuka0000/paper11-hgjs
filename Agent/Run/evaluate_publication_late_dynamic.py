"""Resumable late-disturbance diagnostic with complete execution traces."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dataclasses import asdict, replace
from hashlib import sha256
from datetime import datetime, timezone
import argparse
import csv
import gzip
import json
import os
import math
import torch

from Agent.Run.publication_late_dynamic import LAYERS, TERMINAL, scenarios, run_traced
from Agent.Tool.Benchmark.formal_v3_loader import FormalV3InstanceLoader
from Agent.Tool.Training.chdgrl_config import load_chdgrl_experiment_config
from Agent.Tool.Evaluation.chdgrl_methods import build_chdgrl_methods, evaluation_source_hashes
from Agent.Tool.Evaluation.method_runner import MethodSpec
from Agent.Tool.Evaluation.dynamic_evaluation import ReplanPolicy
from Agent.Baseline.exact_milp import ExactMilpBaseline

PHASE = ROOT / 'Data/Paper/publication_late_dynamic_20260908'
BASE = ROOT / 'Data/Result/Diagnostic/publication_late_dynamic_20260908'
DEV = ROOT / 'Data/Instance/Benchmark/chdgrl_physical_v2/development_manifest.jsonl'
TEST = ROOT / 'Data/Instance/Benchmark/chdgrl_final_guard8_v1/confirmatory_manifest.jsonl'
CONFIG = ROOT / 'Data/Config/chdgrl_lns/scale_normalized_v3_reward_bounds.json'
TRAINING_SHA = '1463ae43163e28869196262e9ef4aede404d84430785feceeea47856eb4010d9'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--limit-instances', type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import psutil
    process = psutil.Process()
    allowed = process.cpu_affinity()
    process.cpu_affinity([allowed[0]])
    output = BASE / args.split
    if args.limit_instances:
        output = BASE / (args.split + '_smoke')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'traces').mkdir(exist_ok=True)
    dev = FormalV3InstanceLoader(DEV)
    loader = dev if args.split == 'validation' else FormalV3InstanceLoader(TEST)
    config = load_chdgrl_experiment_config(CONFIG)
    if args.split == 'validation':
        # First registered validation case in every difficulty/topology stratum.
        selected = {}
        for iid in loader.instance_ids:
            row = loader.row_for(iid)
            if row.split == 'validation':
                selected.setdefault((row.difficulty, row.topology_family), iid)
        ids = sorted(selected.values())
        seeds = [config.training_seeds[0]]
    else:
        preflight = json.loads((PHASE / 'validation_preflight.json').read_text())
        assert preflight['status'] == 'PASS', 'Validation preflight is required'
        old = json.loads((ROOT / 'Data/Result/Diagnostic/publication_runtime_terminal_20260907/dynamic_stability_summary.csv.contract.json').read_text())
        ids = old['contract']['instance_ids']
        seeds = list(config.training_seeds)
    if args.limit_instances:
        ids = ids[:args.limit_instances]
    sources = evaluation_source_hashes(ROOT)
    for name in ['publication_late_dynamic.py', 'evaluate_publication_late_dynamic.py']:
        sources['Agent/Run/' + name] = digest(ROOT / 'Agent/Run' / name)
    weights = {str(seed): digest(ROOT / f'Data/Checkpoint/chdgrl_final_guard8_v1/seed_{seed}/frozen.pt') for seed in seeds}
    contract = dict(version='late_dynamic_instrumented_v1', split=args.split, instance_ids=ids,
                    seeds=seeds, decision_layers=LAYERS, terminal_layer=TERMINAL,
                    scenarios=[asdict(x) for x in scenarios()], policy=asdict(ReplanPolicy.combined(interval=8)),
                    source_hashes=sources, checkpoint_hashes=weights,
                    development_sha256=digest(DEV), manifest_sha256=digest(DEV if args.split == 'validation' else TEST),
                    config_sha256=digest(CONFIG), training_source_sha256=TRAINING_SHA,
                    methods=['chdgrl_lns', 'no_stability', 'complete_milp'],
                    stochastic_pairing='Same frozen model and controller seed; only plan_change_weight differs',
                    complete_milp='One trajectory per instance/scenario, 0.8 s per lexicographic solve, no future information',
                    timing='Legacy dynamic search limits. Runtime is descriptive, not a common external-deadline comparison.',
                    independence='Physical instance. Seeds and scenarios are repeated measurements.',
                    test_status='Previously examined diagnostic instances; no new holdout or test-based parameter selection.',
                    environment_boundary='Events update residual forecasts; road and energy shocks also affect execution. No policy comparison.',
                    compound_boundary='Fixed co-occurring and successive shocks, not a fitted stochastic correlation model.')
    contract_path = output / 'contract.json'
    canonical = json.loads(json.dumps(contract))
    if contract_path.exists():
        assert json.loads(contract_path.read_text()) == canonical, 'Resume contract differs'
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + '\n')
    archive = output / 'sources'
    for relative, expected in sources.items():
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes((ROOT / relative).read_bytes())
        assert digest(target) == expected
    summary_path = output / 'summary.csv'
    existing = list(csv.DictReader(summary_path.open())) if summary_path.exists() else []
    done = {(r['instance_id'], r['scenario'], r['method'], r['seed']) for r in existing}
    expected = len(ids) * len(scenarios()) * (len(seeds) * 2 + 1)
    for iid in ids:
        instance = loader(iid)
        for seed in seeds:
            methods, _ = build_chdgrl_methods(instance,
                ROOT / f'Data/Checkpoint/chdgrl_final_guard8_v1/seed_{seed}/frozen.pt',
                config, dev.manifest_sha256, 'late_dynamic_legacy_limits', include_exact=False,
                expected_source_sha256=TRAINING_SHA)
            full = methods['chdgrl_lns']
            arms = {'chdgrl_lns': full, 'no_stability': replace(full, config_overrides=(('plan_change_weight', 0.),))}
            if seed == seeds[0]:
                arms['complete_milp'] = MethodSpec(ExactMilpBaseline(max_tasks=64, time_limit_s=.8))
            for scenario in scenarios():
                for name, method in arms.items():
                    seed_label = str(seed) if name != 'complete_milp' else 'reference'
                    key = (iid, scenario.name, name, seed_label)
                    if key in done:
                        continue
                    result, records = run_traced(instance, scenario, ReplanPolicy.combined(interval=8),
                                                  method, config.search.to_search_config())
                    token = sha256('|'.join(key).encode()).hexdigest()[:24]
                    trace_path = output / 'traces' / (token + '.json.gz')
                    document = dict(key=key, instance_hash=instance.instance_hash, records=records,
                                    transitions=[asdict(x) for x in result.transitions])
                    trace_path.write_bytes(gzip.compress(json.dumps(finite_json(document), separators=(',', ':'), allow_nan=False).encode(), mtime=0))
                    late = [x for x in result.transitions if x.layer >= 17]
                    active_late = sum(bool(x['state_before_execution']['fields']['airborne_services']['map'] or
                        any(v['fields']['must_return'] for _, v in x['state_before_execution']['fields']['vehicles']['map']))
                        for x in records if x['layer'] >= 17)
                    row = dict(instance_id=iid, instance_hash=instance.instance_hash, difficulty=instance.difficulty,
                        topology=instance.topology_family, scenario=scenario.name, method=name, seed=seed_label,
                        checkpoint_sha256=weights.get(seed_label, ''), replan_count=result.replan_count,
                        late_replans=sum(x.replanned for x in late), late_active_steps=active_late,
                        replans_after_completion=sum(x.replanned and bool(records[k-1]['state_after']['fields']['done_tasks']['set'])
                            for k, x in enumerate(result.transitions) if k),
                        initial_plan_events=result.transitions[0].plan_change,
                        legacy_plan_changes=sum(x.plan_change for x in result.transitions),
                        future_plan_changes=sum(x['future_changes'] for x in records),
                        final_weighted_misses=result.final_weighted_misses,
                        completion_rate=result.completion_rate, runtime_s=result.total_runtime_s,
                        validation_ok=result.validation_ok, safety_status=result.safety_status,
                        remaining_returns=sum(v.must_return or v.active_arc is not None for v in result.final_state.vehicles.values()),
                        airborne_at_end=sum(u.is_airborne for u in result.final_state.uavs.values()),
                        trace_path=trace_path.relative_to(ROOT).as_posix(), trace_sha256=digest(trace_path))
                    new = not summary_path.exists()
                    with summary_path.open('a', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        if new:
                            writer.writeheader()
                        writer.writerow(row)
                    done.add(key)
                    (output / 'progress.json').write_text(json.dumps(dict(completed=len(done), expected=expected,
                        pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(), last=key), indent=2))
                print(json.dumps(dict(completed=len(done), expected=expected, instance=iid, seed=seed, scenario=scenario.name)), flush=True)
    assert len(done) == expected
    print('COMPLETE', expected, flush=True)


if __name__ == '__main__':
    main()
