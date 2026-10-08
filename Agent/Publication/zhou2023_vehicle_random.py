"""Zhou 2023 IV-B Vehicle-random LNS under disclosed paper11 adaptation.

No canonical action family, learned score, or physical heuristic selector is used.
The full common MILP retains all physical rows in each vehicle-group repair.
"""
from dataclasses import dataclass, replace
from math import floor
from random import Random
from types import SimpleNamespace as NS

from Agent.Publication.budget import BudgetExpired, Deadline
from Agent.Publication.physical_search_baselines import _pool_domain
from Agent.Publication.vehicle_lns_milp import solve_committed
from Agent.Publication.component_objective import evaluate_plan


@dataclass(frozen=True)
class VehicleRandomSettings:
    seed: int
    destroy_degree: float = .4
    relative_mip_gap: float = .1
    acceptance_ratio: float = 1.01
    total_seconds: float = 15.
    repair_seconds: float = 4.
    validation_reserve_seconds: float = .3
    minimum_repair_seconds: float = 4.
    primary_fraction: float = 2/3
    max_iterations: int = 30


def select_vehicle_groups(vehicles, degree, rng):
    vehicles = tuple(sorted(vehicles))
    if not vehicles or not 0 < degree <= 1:
        raise ValueError('Nonempty vehicle population and a valid destroy fraction required')
    count = max(1, floor(len(vehicles)*degree))
    return tuple(rng.sample(vehicles, count))


def accept_lexicographic(candidate, current, ratio=1.01):
    if candidate[0] < current[0]-1e-7:
        return True
    if abs(candidate[0]-current[0]) <= 1e-7:
        return candidate[1] < current[1]*ratio+1e-9
    return False


def vehicle_signature(plan, vehicle):
    return (frozenset(a.arc_id for a in plan.arcs if a.vehicle_id == vehicle),
            frozenset(c.tuple_id for c in plan.copies if c.vehicle_id == vehicle))


def run_vehicle_random(problem, objective_weights, settings, *, model_observer=None, task_groups=None):
    deadline = Deadline(settings.total_seconds)
    rng = Random(settings.seed)
    vehicles = tuple(sorted(problem.domain.context.state.vehicles))
    domain = _pool_domain(problem.domain, problem.full_copies.values())
    current_state = best_state = problem.state
    check = evaluate_plan(problem.domain, problem.plan(current_state), objective_weights, task_groups)
    current_value = best_value = initial_value = (float(check['primary']), float(check['secondary']))
    trace, stop = [], 'iteration_limit'
    for iteration in range(settings.max_iterations):
        remaining = deadline.remaining()-settings.validation_reserve_seconds
        if remaining < settings.minimum_repair_seconds:
            stop = 'insufficient_common_budget'
            break
        released = select_vehicle_groups(vehicles, settings.destroy_degree, rng)
        incumbent = problem.plan(current_state)
        allocation = min(settings.repair_seconds, remaining)
        local = Deadline(allocation)
        before = current_value
        raw, candidate, valid, accepted, expired = {}, None, False, False, False
        try:
            raw = solve_committed(domain, NS(objective_weights=objective_weights),
                primary_seconds=allocation*settings.primary_fraction, secondary_seconds=allocation,
                incumbent=incumbent, released_vehicle_ids=released,
                relative_mip_gap=settings.relative_mip_gap,
                deadline=local, validation_reserve_s=settings.validation_reserve_seconds,
                model_observer=model_observer, task_groups=task_groups)
            candidate = raw['candidate']
            if candidate is not None:
                proposal = candidate['plan']
                for vehicle in set(vehicles)-set(released):
                    if vehicle_signature(proposal, vehicle) != vehicle_signature(incumbent, vehicle):
                        raise ValueError('A nonselected vehicle changed')
                proposed_state = replace(current_state,
                    selected_arc_ids=frozenset(a.arc_id for a in proposal.arcs),
                    selected_tuple_ids=frozenset(c.tuple_id for c in proposal.copies))
                checked = evaluate_plan(problem.domain, problem.plan(proposed_state), objective_weights, task_groups)
                candidate_value = float(checked['primary']), float(checked['secondary'])
                if any(abs(a-b)>1e-7 for a,b in zip(candidate_value,(candidate['primary'],candidate['secondary']))):
                    raise ValueError('Full-domain objective differs from repair objective')
                valid = True
                local.check()
                deadline.check()
                if candidate['validated_at'] > local.started+allocation:
                    expired = True
                else:
                    accepted = accept_lexicographic(candidate_value, current_value, settings.acceptance_ratio)
                    if accepted:
                        current_state, current_value = proposed_state, candidate_value
                        if current_value < best_value:
                            best_state, best_value = current_state, current_value
        except BudgetExpired:
            expired = True
        trace.append({'iteration': iteration, 'released_vehicles': released,
            'actual_destroy_fraction': len(released)/len(vehicles), 'before_objective': before,
            'candidate_objective': None if candidate is None else [candidate['primary'],candidate['secondary']],
            'candidate_arc_ids': None if candidate is None else sorted(a.arc_id for a in candidate['plan'].arcs),
            'candidate_tuple_ids': None if candidate is None else sorted(c.tuple_id for c in candidate['plan'].copies),
            'current_arc_ids': sorted(current_state.selected_arc_ids),
            'current_tuple_ids': sorted(current_state.selected_tuple_ids),
            'current_objective': current_value, 'best_objective': best_value,
            'accepted': accepted, 'candidate_physically_valid': valid, 'late_candidate_discarded': expired,
            'repair_allocation_seconds': allocation, 'repair_elapsed_seconds': local.elapsed(),
            'fixed_vehicle_variable_count': raw.get('fixed_vehicle_variable_count'),
            'solver_variables': raw.get('variables'), 'solver_constraints': raw.get('constraints'),
            'solver_stages': raw.get('stages', []), 'assembly_seconds': raw.get('assembly_elapsed_s'),
            'incumbent_hint': raw.get('incumbent_hint'),
            'validated_candidate_time_from_action_start': None if candidate is None else candidate['validated_at']-deadline.started})
    checked = evaluate_plan(problem.domain, problem.plan(best_state), objective_weights, task_groups)
    final = float(checked['primary']), float(checked['secondary'])
    if any(abs(a-b)>1e-7 for a,b in zip(final,best_value)):
        raise ValueError('Returned best solution and objective disagree')
    elapsed = deadline.elapsed()
    return {'method_id': 'zhou2023_vehicle_random',
        'metric': 'weighted_missed_observations' if task_groups is None else 'weighted_incomplete_components',
        'method_label': 'Zhou2023 Vehicle-random LNS, paper11 adaptation',
        'initial_objective': initial_value, 'objective': final, 'trace': trace,
        'selected_arc_ids': sorted(best_state.selected_arc_ids),
        'selected_tuple_ids': sorted(best_state.selected_tuple_ids),
        'physical_check_passed': True, 'total_action_seconds': elapsed,
        'within_common_budget': elapsed <= settings.total_seconds,
        'stop_reason': stop, 'iterations': len(trace),
        'accepted_iterations': sum(row['accepted'] for row in trace)}
