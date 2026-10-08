"""Validation-only matched timing and closure comparison on fixed physical tours.

The historical generator, policy weights and raw experiments are immutable.
Only completed incumbents observed by each independently executed deadline are
eligible for reported quality. Late solver returns remain in the raw record.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from hashlib import sha256
import math
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

from Agent.Run.publication_rolling_roads import RoadMemory, build_rolling_problem
from Agent.Run.publication_resource_closure import preparation_resource_keys, close_with_resource_propagation, physical_operation_violations
from Agent.Run.publication_rolling_lock_audit import lock_input_violations, lock_output_violations
from Agent.Proposed import dgrl_actions
from Agent.Proposed.closure import close_region
from Agent.Proposed.search import run_search
from Agent.Tool.Evaluation.dynamic_evaluation import validate_dynamic_plan
from Agent.Tool.Evaluation.chdgrl_methods import ControllerSearchSolver, stable_evaluation_seed
from Agent.Baseline.common import calculate_objective
from Agent.Baseline.exact_milp import ExactMilpBaseline
from Agent.Baseline.manual_alns import ManualAlnsBaseline
from Agent.Baseline.risk_greedy import RiskPriorityGreedyBaseline
from Mask.TupleFactory import build_service_resource_keys

TIMINGS = ("legacy_occupancy", "prelaunch_occupancy")
METHODS = ("learned_original", "learned_expanded", "deterministic_original", "deterministic_expanded", "alns", "greedy", "milp")


def signature(value):
    from Agent.Run.publication_late_dynamic import pack
    import json
    return sha256(json.dumps(pack(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_problem(instance, timing):
    if instance.split != "validation" and instance.split != "train":
        raise ValueError("Development instances only")
    if timing not in TIMINGS:
        raise ValueError("Unknown timing convention")
    problem, routes = build_rolling_problem(instance, instance.rolling_state, RoadMemory.initial(instance))
    if problem.safety_status != "feasible":
        raise ValueError("Invalid common physical construction: " + problem.safety_reason)
    if problem.search_state.selected_tuple_ids or problem.search_state.selected_arc_ids:
        raise ValueError("Static matched comparison requires a common empty incumbent")
    original = problem.search_state
    if timing == "legacy_occupancy":
        tuples = []
        capacities = {}
        capacity = min(instance.operation_capacity.values(), default=1)
        for item in original.tuples:
            keys = build_service_resource_keys(item.vehicle_id, item.uav_id, item.launch_node, item.recovery_node,
                item.launch_layer, item.recovery_layer, instance.preparation_layers, instance.recovery_service_layers, instance.turnaround_layers)
            tuples.append(replace(item, interval_keys=keys.interval_keys, operation_keys=keys.operation_keys))
            capacities.update({key: capacity for key in keys.operation_keys})
        problem = replace(problem, search_state=replace(original, tuples=tuple(tuples)), operation_capacity=capacities)
    if lock_input_violations(problem.search_state, problem.rolling_state):
        raise ValueError("Invalid initial lock obligations")
    return problem, routes


def physically_check(instance, problem, selected_tuples, selected_arcs):
    """Derive physical resources independently of the solver's timing keys."""
    state = problem.search_state
    by_id = {item.tuple_id: item for item in state.tuples}
    arc_map = {arc.arc_id: arc for arc in state.arcs}
    if set(selected_tuples) - by_id.keys() or set(selected_arcs) - arc_map.keys():
        return ("unknown_selection",)
    selected = SimpleNamespace(selected_tuple_ids=selected_tuples, selected_arc_ids=selected_arcs)
    report = validate_dynamic_plan(problem, selected)
    errors = [] if report.ok else ["represented_model:" + str(report.first_violation)]
    errors.extend(lock_output_violations(state, problem.rolling_state, selected_tuples, selected_arcs))
    intervals, operations = Counter(), Counter()
    actual_arcs = [arc_map[key] for key in selected_arcs]
    capacity = min(instance.operation_capacity.values(), default=1)
    for key in selected_tuples:
        item = by_id[key]
        resources = preparation_resource_keys(item.vehicle_id, item.uav_id, item.launch_node, item.recovery_node,
            item.launch_layer, item.recovery_layer, instance.preparation_layers, instance.recovery_service_layers, instance.turnaround_layers,
            current_layer=problem.rolling_state.current_layer, available_layer=problem.rolling_state.uavs[item.uav_id].available_layer)
        physical = replace(item, interval_keys=resources.interval_keys, operation_keys=resources.operation_keys)
        errors.extend(physical_operation_violations(physical, actual_arcs, instance.preparation_layers,
            instance.recovery_service_layers, instance.turnaround_layers, current_layer=problem.rolling_state.current_layer,
            available_layer=problem.rolling_state.uavs[item.uav_id].available_layer))
        intervals.update(resources.interval_keys)
        operations.update(resources.operation_keys)
    errors.extend("physical_uav_overlap:" + key for key, value in intervals.items() if value > 1)
    errors.extend("physical_carrier_capacity:" + key for key, value in operations.items() if value > capacity)
    return tuple(sorted(set(errors)))


@contextmanager
def closure_variant(expanded, observations):
    original = dgrl_actions.close_region
    closer = close_with_resource_propagation if expanded else close_region
    def measured(state, core, budget):
        started = perf_counter()
        result = closer(state, core, budget)
        observations.append(dict(accepted=result.accepted, reason=result.reason, tuples=len(result.tuple_ids),
                                 tasks=len(result.task_ids), arcs=len(result.arc_ids), elapsed_s=perf_counter() - started))
        return result
    with patch.object(dgrl_actions, "close_region", measured):
        yield
    if dgrl_actions.close_region is not original:
        raise RuntimeError("Closure patch escaped isolated execution")


def execute(instance, problem, experiment_config, model, checkpoint_seed, method, budget):
    if method not in METHODS or budget <= 0:
        raise ValueError("Unknown method or nonpositive deadline")
    config = replace(experiment_config.search.to_search_config(), total_time_limit_s=float(budget))
    seed = stable_evaluation_seed(checkpoint_seed, instance.instance_id,
                                 "chdgrl_lns" if method.startswith("learned") else "manual_alns" if method == "alns" else "deterministic_joint")
    snapshots, projections = [], []
    started = perf_counter()

    def observe(selected_tuples, selected_arcs, event):
        schedule = SimpleNamespace(selected_tuple_ids=tuple(sorted(selected_tuples)), selected_arc_ids=tuple(sorted(selected_arcs)))
        checked = validate_dynamic_plan(problem, schedule)
        state = replace(problem.search_state, selected_tuple_ids=frozenset(selected_tuples), selected_arc_ids=frozenset(selected_arcs))
        primary, secondary = calculate_objective(state, problem.task_ids, problem.miss_penalty, config, validation_state=problem.rolling_state)
        elapsed = perf_counter() - started
        snapshots.append(dict(event=event, elapsed_s=elapsed, model_feasible=checked.ok, model_violation=checked.first_violation,
                              primary=primary, secondary=secondary, selected_tuple_ids=schedule.selected_tuple_ids, selected_arc_ids=schedule.selected_arc_ids))

    observe(problem.search_state.selected_tuple_ids, problem.search_state.selected_arc_ids, "initial")
    kwargs = dict(state=problem.search_state, task_ids=problem.task_ids, miss_penalty=problem.miss_penalty,
                  validation_state=problem.rolling_state, operation_capacity=problem.operation_capacity, config=config)
    if method.startswith(("learned", "deterministic")):
        kind = "learned" if method.startswith("learned") else "deterministic"
        solver = ControllerSearchSolver(model, experiment_config, seed, kind,
            len(experiment_config.search.release_profiles) // 2, len(experiment_config.search.time_budgets) // 2)
        with closure_variant(method.endswith("expanded"), projections):
            result = run_search(**kwargs, controller=solver._controller(),
                step_observer=lambda event: observe(event.state_after.selected_tuple_ids, event.state_after.selected_arc_ids, "iteration_" + str(event.iteration)))
    elif method == "alns":
        solver = ManualAlnsBaseline(seed=seed, max_iterations=None, projection_limit=None,
                                   time_limit_s=max(experiment_config.search.time_budgets), total_time_limit_s=float(budget))
        result = solver.solve(**kwargs, incumbent_observer=lambda value: observe(value.selected_tuple_ids, value.selected_arc_ids, "alns_incumbent"))
    elif method == "greedy":
        solver = RiskPriorityGreedyBaseline(time_limit_s=max(experiment_config.search.time_budgets), projection_limit=None, total_time_limit_s=float(budget))
        result = solver.solve(**kwargs, incumbent_observer=lambda value: observe(value.selected_tuple_ids, value.selected_arc_ids, "greedy_incumbent"))
    else:
        # Every complete-MILP return is timed and checked in the same way. The
        # small validation domains are expected to finish; late returns remain
        # visible and cannot displace the common feasible initial incumbent.
        result = ExactMilpBaseline(max_tasks=len(problem.task_ids), time_limit_s=float(budget)).solve(**kwargs)
    observe(result.selected_tuple_ids, result.selected_arc_ids, "final")
    runtime = perf_counter() - started
    eligible = [value for value in snapshots if value["elapsed_s"] <= budget and value["model_feasible"]
                and math.isfinite(value["primary"]) and math.isfinite(value["secondary"])]
    if not eligible:
        raise ValueError("No verified incumbent observed within deadline")
    selected = min(eligible, key=lambda value: (value["primary"], value["secondary"], value["elapsed_s"]))
    for value in snapshots:
        value["physical_violations"] = physically_check(instance, problem, value["selected_tuple_ids"], value["selected_arc_ids"])
    return dict(instance_id=instance.instance_id, method=method, seed=checkpoint_seed, controller_seed=seed,
                budget_s=budget, runtime_s=runtime, stop_reason=result.stop_reason, selected=selected,
                snapshots=snapshots, projections=projections, original_domain_hash=signature(problem),
                primary_status=getattr(result, "primary_status", "not_applicable"), secondary_status=getattr(result, "secondary_status", "not_applicable"),
                deadline_rule="Best model-feasible incumbent checked and timestamped no later than this independently executed deadline. Physical verification is subsequent and does not repair or filter the method output.")
