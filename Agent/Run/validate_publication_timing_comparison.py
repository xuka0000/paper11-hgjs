"""Recompute objectives, flow and physical occupancy without the solver checker."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from collections import Counter, defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
import argparse
import csv
import json
import math
from statistics import mean, median

from Agent.Run.evaluate_publication_rolling_roads import decode
from Agent.Tool.Training.chdgrl_config import load_chdgrl_experiment_config

OUT = ROOT / "Data/Result/Diagnostic/publication_timing_comparison_20260910"
PHASE = ROOT / "Data/Paper/publication_timing_comparison_20260910"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def same_number(left, right):
    if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > 1e-7:
        raise ValueError(f"Independent arithmetic mismatch: {left}, {right}")


def check_schedule(problem, snapshot, weights, preparation=1, recovery=1, turnaround=1):
    state, rolling = problem.search_state, problem.rolling_state
    tuples = {item.tuple_id: item for item in state.tuples}
    arcs = {arc.arc_id: arc for arc in state.arcs}
    ty, ax = snapshot["selected_tuple_ids"], snapshot["selected_arc_ids"]
    if len(ty) != len(set(ty)) or len(ax) != len(set(ax)) or set(ty) - tuples.keys() or set(ax) - arcs.keys():
        raise ValueError("Invalid decision IDs")
    chosen, path = [tuples[key] for key in ty], [arcs[key] for key in ax]
    model_errors, physical_errors = [], []
    task_counts = Counter(task for item in chosen for task in item.task_ids)
    if any(value > 1 for value in task_counts.values()) or set(task_counts) - set(problem.task_ids):
        model_errors.append("task_coverage")
    if any(not item.admitted for item in chosen):
        model_errors.append("admission")
    for name, locked, initial, selected in (("tuple", rolling.locked_tuple_ids, state.selected_tuple_ids, ty),
                                           ("arc", rolling.locked_arc_ids, state.selected_arc_ids, ax)):
        if any((key in initial) != (key in selected) for key in locked):
            model_errors.append("changed_lock_" + name)
    degree = Counter()
    active = {arc.vehicle_id for arc in path} | {item.vehicle_id for item in chosen}
    for arc in path:
        degree[(arc.vehicle_id, "source" if arc.tail == "source" else (arc.tail, arc.departure_layer))] -= 1
        degree[(arc.vehicle_id, "sink" if arc.head == "sink" else (arc.head, arc.arrival_layer))] += 1
    for vid in active:
        if degree[(vid, "source")] != -1 or degree[(vid, "sink")] != 1:
            model_errors.append("vehicle_boundary")
    if any(value for (vid, node), value in degree.items() if node not in ("source", "sink")):
        model_errors.append("vehicle_flow")
    for vid, vehicle in rolling.vehicles.items():
        if vehicle.must_return and vid not in active:
            model_errors.append("required_return")
        if vehicle.remaining_duty_time is not None and sum(arc.duty_time for arc in path if arc.vehicle_id == vid) > vehicle.remaining_duty_time + 1e-8:
            model_errors.append("vehicle_duty")
    arrivals = {(arc.vehicle_id, arc.head, arc.arrival_layer) for arc in path}
    model_intervals, model_ops, true_intervals, true_ops = Counter(), Counter(), Counter(), Counter()
    physical_capacity = min(problem.operation_capacity.values(), default=1)
    for item in chosen:
        if (item.launch_node != "source" and (item.vehicle_id, item.launch_node, item.launch_layer) not in arrivals
                or (item.vehicle_id, item.recovery_node, item.recovery_layer) not in arrivals):
            model_errors.append("endpoint")
        if set(item.required_arc_ids) - set(ax):
            model_errors.append("required_arc")
        if any(not item.launch_layer <= layer <= item.recovery_layer for _, layer in item.completion_layers):
            model_errors.append("completion_time")
        model_intervals.update(item.interval_keys)
        model_ops.update(item.operation_keys)
        start = item.launch_layer - preparation
        ready = item.recovery_layer + recovery + turnaround
        if start < max(rolling.current_layer, rolling.uavs[item.uav_id].available_layer):
            physical_errors.append("preparation_before_ready")
        true_intervals.update((item.uav_id, layer) for layer in range(start, ready))
        for node, lower, upper in ((item.launch_node, start, item.launch_layer),
                                   (item.recovery_node, item.recovery_layer, item.recovery_layer + recovery)):
            for layer in range(lower, upper):
                true_ops[(item.vehicle_id, node, layer)] += 1
                if not any(arc.vehicle_id == item.vehicle_id and arc.tail == arc.head == node
                           and arc.departure_layer <= layer < arc.arrival_layer for arc in path):
                    physical_errors.append("carrier_absent")
        if item.battery_capacity > 0 and item.reliable_energy > item.battery_capacity * (1 - item.reserve_ratio) + 1e-8:
            model_errors.append("energy")
    if any(value > 1 for value in model_intervals.values()):
        model_errors.append("uav_resource")
    if any(model_ops[key] > capacity for key, capacity in problem.operation_capacity.items()):
        model_errors.append("operation_resource")
    if any(value > 1 for value in true_intervals.values()):
        physical_errors.append("physical_uav_overlap")
    if any(value > physical_capacity for value in true_ops.values()):
        physical_errors.append("physical_carrier_capacity")
    primary = sum(problem.miss_penalty[task] for task in problem.task_ids if not task_counts[task])
    duty = sum(arc.duty_time for arc in path)
    completion = [layer for item in chosen for _, layer in item.completion_layers]
    returns = [arc.arrival_layer for arc in path if arc.head == "sink"]
    secondary = sum(item.secondary_cost for item in chosen) + sum(arc.secondary_cost for arc in path)
    secondary += weights["time_weight"] * duty
    secondary += weights["risk_weight"] * sum(item.risk_exposure for item in chosen)
    secondary += weights["energy_weight"] * sum(item.expected_energy for item in chosen)
    secondary += weights["makespan_weight"] * max(completion + returns, default=0)
    secondary += weights["plan_change_weight"] * (sum(abs(int(key in ty) - value) for key, value in rolling.previous_tuple_values.items())
                                                + sum(abs(int(key in ax) - value) for key, value in rolling.previous_arc_values.items()))
    same_number(primary, snapshot["primary"])
    same_number(secondary, snapshot["secondary"])
    return sorted(set(model_errors)), sorted(set(physical_errors)), len(task_counts)


def write_csv(path, data):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def audit(allow_partial=False):
    contract = read(OUT / "contract.json")
    from Agent.Run.publication_timing_comparison import signature
    contract_hash = signature(contract)
    progress = read(OUT / "progress.json")
    if not allow_partial and (progress["status"] != "complete" or progress["completed"] != contract["total_jobs"]):
        raise ValueError("Comparison grid incomplete")
    for path, expected in contract["source_hashes"].items():
        if digest(ROOT / path) != expected or digest(OUT / "sources" / path) != expected:
            raise ValueError("Changed registered implementation: " + path)
    config_path = ROOT / "Data/Config/chdgrl_lns/scale_normalized_v3_reward_bounds.json"
    if digest(config_path) != contract["config_sha256"]:
        raise ValueError("Changed objective configuration")
    weights = load_chdgrl_experiment_config(config_path).search.objective_weights
    domains = {}
    for path in (OUT / "domains").glob("*.json"):
        record = read(path)
        problem = decode(record["problem"])
        if signature(problem) != record["problem_hash"]:
            raise ValueError("Changed recorded domain")
        domains[(record["instance_id"], record["timing"])] = problem
    for entry in contract["input_rows"]:
        if signature(domains[(entry["instance_id"], entry["timing"])]) != entry["problem_hash"]:
            raise ValueError("Domain differs from registration")
    from Agent.Tool.Benchmark.formal_v3_loader import FormalV3InstanceLoader
    manifest = ROOT / 'Data/Instance/Benchmark/chdgrl_physical_v2/development_manifest.jsonl'
    if digest(manifest) != contract['manifest_sha256']:
        raise ValueError('Changed development manifest')
    loader = FormalV3InstanceLoader(manifest)
    instance_ids = sorted({key[0] for key in domains})
    if len(instance_ids) != 12:
        raise ValueError('Unexpected validation panel size')
    for key in instance_ids:
        instance = loader(key)
        if instance.split != 'validation' or (instance.preparation_layers, instance.recovery_service_layers, instance.turnaround_layers) != (1, 1, 1):
            raise ValueError('Independent duration assumptions do not match the registered instances')
        old, new = domains[(key, 'legacy_occupancy')], domains[(key, 'prelaunch_occupancy')]
        if replace(old, search_state=new.search_state, operation_capacity=new.operation_capacity) != new:
            raise ValueError('Unmatched physical problem')
        if replace(old.search_state, tuples=new.search_state.tuples) != new.search_state:
            raise ValueError('Unmatched common search state')
        if len(old.search_state.tuples) != len(new.search_state.tuples):
            raise ValueError('Different candidate counts')
        for a, b in zip(old.search_state.tuples, new.search_state.tuples):
            if replace(a, interval_keys=b.interval_keys, operation_keys=b.operation_keys) != b:
                raise ValueError('Timing arm changed more than resource keys')
        for entry in (row for row in contract['input_rows'] if row['instance_id'] == key):
            if (entry['original_instance_hash'], entry['scenario_hash'], entry['physical_candidate_pool_hash']) != (instance.instance_hash, instance.scenario_hash, instance.candidate_pool_hash):
                raise ValueError('Input metadata differs from development instance')
    data, run_hashes, schedule_hashes = [], {}, set()
    counts = Counter()
    for job in contract["jobs"]:
        path = OUT / "runs" / (signature(job)[:24] + ".json")
        if not path.exists() and allow_partial:
            continue
        payload = read(path)
        if payload["job"] != job or payload["contract_hash"] != contract_hash:
            raise ValueError("Unregistered job")
        record = payload["result"]
        problem = domains[(job["instance_id"], job["timing"])]
        if record["original_domain_hash"] != signature(problem):
            raise ValueError("Run input changed")
        previous_time = -1.
        for snapshot in record["snapshots"]:
            if not math.isfinite(snapshot["elapsed_s"]) or snapshot["elapsed_s"] < previous_time:
                raise ValueError("Invalid incumbent clock")
            previous_time = snapshot["elapsed_s"]
            model_errors, physical_errors, _ = check_schedule(problem, snapshot, weights)
            if bool(model_errors) == snapshot["model_feasible"]:
                raise ValueError("Solver checker disagrees with independent feasibility")
            if bool(physical_errors or model_errors) != bool(snapshot["physical_violations"]):
                raise ValueError("Physical verification mismatch")
            counts["snapshots_checked"] += 1
            counts["late_snapshots"] += int(snapshot["elapsed_s"] > job["budget_s"])
            schedule_hashes.add(signature((signature(problem), snapshot["selected_tuple_ids"], snapshot["selected_arc_ids"])))
        eligible = [value for value in record["snapshots"] if value["elapsed_s"] <= job["budget_s"] and value["model_feasible"]]
        expected = min(eligible, key=lambda value: (value["primary"], value["secondary"], value["elapsed_s"]))
        if expected != record["selected"] or expected["elapsed_s"] > job["budget_s"]:
            raise ValueError("Selected an out-of-deadline or nonbest incumbent")
        model_errors, physical_errors, covered = check_schedule(problem, expected, weights)
        if model_errors:
            raise ValueError("Selected model-infeasible output")
        if job["timing"] == "prelaunch_occupancy" and physical_errors:
            raise ValueError("Corrected method returned a physically invalid incumbent")
        projections = record["projections"]
        data.append(dict(**job, primary=expected["primary"], secondary=expected["secondary"], covered_tasks=covered,
            unmet_percent=100 * expected["primary"] / sum(problem.miss_penalty.values()),
            physical_feasible=not physical_errors, physical_violations=";".join(physical_errors),
            runtime_s=record["runtime_s"], selected_time_s=expected["elapsed_s"],
            deadline_overrun=record["runtime_s"] > job["budget_s"], stop_reason=record["stop_reason"],
            projection_calls=len(projections), rejected_projections=sum(not row["accepted"] for row in projections),
            projection_seconds=sum(row["elapsed_s"] for row in projections),
            mean_closure_tuples=mean([row["tuples"] for row in projections]) if projections else 0,
            primary_status=record["primary_status"], secondary_status=record["secondary_status"],
            run_path=path.relative_to(ROOT).as_posix(), run_sha256=digest(path)))
        run_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    counts["runs_checked"] = len(data)
    counts['matched_instance_pairs_checked'] = len(instance_ids)
    counts["distinct_schedules_within_domains"] = len(schedule_hashes)
    counts["physical_invalid_legacy_outputs"] = sum(not row["physical_feasible"] for row in data if row["timing"] == "legacy_occupancy")
    counts["physical_invalid_corrected_outputs"] = sum(not row["physical_feasible"] for row in data if row["timing"] == "prelaunch_occupancy")
    if allow_partial:
        print(json.dumps(dict(status="PARTIAL_DIAGNOSTIC", **counts)))
        return
    if len(data) != contract["total_jobs"]:
        raise ValueError("Missing runs")
    write_csv(PHASE / "verified_runs.csv", data)
    report = dict(status="PASS", counts=dict(counts), contract_sha256=digest(OUT / "contract.json"),
                  verified_runs_sha256=digest(PHASE / "verified_runs.csv"), raw_run_hashes=run_hashes,
                  validator_sha256=digest(Path(__file__)),
                  boundary="Independent arithmetic and timing audit on registered validation instances. Legacy physical failures remain explicit; no results are relabeled or repaired after execution.")
    (PHASE / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="PASS", **counts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    audit(parser.parse_args().allow_partial)
