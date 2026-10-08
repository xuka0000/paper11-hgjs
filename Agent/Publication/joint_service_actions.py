"""Bounded joint service edits with fixed-service physical route validation."""
from dataclasses import dataclass
from time import perf_counter

from Agent.Publication.budget import BudgetExpired
from Agent.Publication.component_objective import evaluate_plan
from Agent.Publication.many_to_many_model import ManyToManyModel
from Agent.Run.optimize_publication_hamming_routes import optimize_committed_routes
from Agent.Run.publication_plan_distance import service_key


@dataclass(frozen=True)
class ActionProposal:
    add: tuple[str, ...] = ()
    release: tuple[str, ...] = ()


_PHYSICAL_REJECTIONS = (
    'Fixed service handling unavailable', 'Conflicting mandatory vehicle actions',
    'No route for fixed continuation services',
    'No common return-time cap', 'Duplicate selected decision',
    'Road outside continuation domain', 'Changed locked or ongoing road/handling',
    'Missing or repeated service obligation:', 'Deployed vehicle return removed',
    'Branching selected vehicle path', 'Selected cycle', 'Discontinuous vehicle path',
    'Remaining duty exceeded', 'UAV owner mismatch', 'Changed special-copy fields',
    'Uncommitted preparation starts in past', 'Completed task relaunched',
    'Airborne history or recovery node changed', 'Recovery advanced without a flight model',
    'Residual energy or reserve violated', 'Airborne energy rate reset',
    'Begun preparation changed', 'Launch preparation before UAV ready',
    'Invalid inspection completion', 'Executed task incorrectly remains pending',
    'Repeated task assignment', 'UAV preparation/recovery/turnaround overlap',
    'Carrier operation overlap', 'Carrier absent during preparation or recovery',
)


def evaluate_action(problem, model, incumbent, proposal, weights, deadline, *, _projection=None, _limits=None):
    """Return an on-time improvement or the unchanged incumbent on timeout."""
    start = perf_counter()
    try:
        return _evaluate_action(problem, model, incumbent, proposal, weights, deadline, _projection=_projection, _limits=_limits)
    except BudgetExpired:
        before = evaluate_plan(problem.domain, incumbent, weights, model)
        selected = tuple(sorted(c.tuple_id for c in incumbent.copies))
        # Validation of the public request remains strict even when time has run
        # out before projection; a timeout cannot disguise an invalid action.
        if not isinstance(proposal, ActionProposal):
            raise TypeError('Expected ActionProposal')
        add, release = tuple(proposal.add), tuple(proposal.release)
        add_cap, release_cap, group_cap = _action_caps(_limits)
        if (len(add) > add_cap or len(release) > release_cap or len(set(add)) != len(add)
                or len(set(release)) != len(release)):
            raise ValueError('Action exceeds add/release cap or repeats an ID')
        if any(not isinstance(k, str) or not k for k in add + release):
            raise ValueError('Action IDs must be nonempty strings')
        current = frozenset(selected)
        if not set(release) <= current or set(add) & (current | set(release)):
            raise ValueError('Release must be incumbent-only and additions unselected')
        if not set(add) <= problem.full_copies.keys():
            raise ValueError('Unknown service alternative')
        targets = {group.group_id for group in model.groups
                   if any(task in group.task_ids for key in add
                          for task, _ in problem.full_copies[key].completion_layers)}
        if len(targets) > group_cap:
            raise ValueError('Action exceeds target-group cap')
        proposed = tuple(sorted((current - set(release)) | set(add)))
        return dict(accepted=False, reason='deadline_exhausted',
            objective=(before['primary'], before['secondary']),
            selected_tuple_ids=selected,
            selected_arc_ids=tuple(sorted(a.arc_id for a in incumbent.arcs)),
            proposed_selected_tuple_ids=proposed, projected_primary_gain=None,
            physical_passed=False, within_budget=False,
            total_seconds=perf_counter()-start, plan=incumbent,
            proposal=proposal, evidence=dict(stage='before_candidate_validation'))


def _action_caps(limits):
    values=(4,4,2) if limits is None else (limits.additions,limits.releases,limits.target_groups)
    if any(type(v) is not int or v<0 for v in values) or values[2]<1:
        raise ValueError('Invalid bound action protocol')
    return values


def _evaluate_action(problem, model, incumbent, proposal, weights, deadline, *, _projection=None, _limits=None):
    """Return incumbent on every rejected proposal; propagate unexpected defects.

    The route decoder optimizes roads only. Its returned service set is checked
    against the exact projected (selected - release) | add set before acceptance.
    """
    start = perf_counter()
    if not isinstance(proposal, ActionProposal):
        raise TypeError('Expected ActionProposal')
    add, release = tuple(proposal.add), tuple(proposal.release)
    add_cap, release_cap, group_cap = _action_caps(_limits)
    if (len(add) > add_cap or len(release) > release_cap or len(set(add)) != len(add)
            or len(set(release)) != len(release)):
        raise ValueError('Action exceeds add/release cap or repeats an ID')
    if any(not isinstance(k, str) or not k for k in add + release):
        raise ValueError('Action IDs must be nonempty strings')
    deadline.check()
    before = evaluate_plan(problem.domain, incumbent, weights, model)
    deadline.check()
    incumbent_ids = frozenset(c.tuple_id for c in incumbent.copies)
    if len(incumbent_ids) != len(incumbent.copies):
        raise ValueError('Incumbent repeats a service')
    if not incumbent_ids <= problem.full_copies.keys():
        raise ValueError('Incumbent outside physical catalogue')
    # Private search-call cache: the caller owns one frozen problem/model and
    # only changes the incumbent. Every accepted plan still undergoes the full
    # independent physical evaluator and before/after quota-score comparison.
    view = ManyToManyModel.from_problem(problem, model) if _projection is None else _projection
    deadline.check()
    target_groups = {group.group_id for group in model.groups
                     if any(task in group.task_ids for key in add
                            for task, _ in view.services[key].completion_layers)} if set(add) <= view.services.keys() else set()
    if len(target_groups) > group_cap:
        raise ValueError('Action exceeds target-group cap')
    query = view.query(incumbent_ids, add=add, release=release)
    deadline.check()
    selected_ids = frozenset(query['selected'])
    if selected_ids != (incumbent_ids - set(release)) | set(add):
        raise RuntimeError('Projection changed requested service set')
    if abs(query['before']['primary'] - before['primary']) > 1e-7:
        raise ValueError('Quota projection differs from physical incumbent score')
    initial_arcs = tuple(sorted(a.arc_id for a in incumbent.arcs))
    def rejected(reason, *, physical=False, on_time=True):
        if on_time:
            try:
                deadline.check()
            except BudgetExpired:
                reason, on_time = 'deadline_exhausted', False
        return dict(accepted=False, reason=reason, objective=(before['primary'], before['secondary']),
            selected_tuple_ids=tuple(sorted(incumbent_ids)), selected_arc_ids=initial_arcs,
            proposed_selected_tuple_ids=tuple(sorted(selected_ids)),
            projected_primary_gain=query['primary_gain'], physical_passed=physical,
            within_budget=on_time, total_seconds=perf_counter()-start,
            plan=incumbent, proposal=proposal,
            evidence=dict(projection=query, initial_objective=(before['primary'], before['secondary'])))
    if not add and not release:
        return rejected('no_change')
    if not query['after']['projected_constraints_pass']:
        return rejected('projected_resource_or_duplicate_rejection')
    # Commitments are compared by physical service identity to allow equivalent
    # catalogue IDs while disallowing removal of begun/locked work.
    previous = problem.domain.context.previous
    locked_ids = problem.domain.context.state.locked_tuple_ids
    locked_keys = {service_key(c) for c in previous.copies if c.tuple_id in locked_ids}
    locked_keys |= {service_key(c) for c in incumbent.copies if c.tuple_id in locked_ids}
    selected_keys = {service_key(problem.full_copies[k]) for k in selected_ids}
    if not locked_keys <= selected_keys:
        return rejected('locked_service_removed')
    if any(len(set(group) & selected_ids) != 1 for group in problem.domain.groups.values()):
        return rejected('committed_service_obligation_removed')
    if query['after']['primary'] > before['primary'] + 1e-7:
        return rejected('worse_projected_primary')
    try:
        deadline.check()
        fixed = tuple(problem.full_copies[k] for k in sorted(selected_ids))
        candidate, route = optimize_committed_routes(problem.domain, fixed, weights, deadline=deadline)
        deadline.check()
        actual_ids = frozenset(c.tuple_id for c in candidate.copies)
        if actual_ids != selected_ids or len(candidate.copies) != len(selected_ids):
            raise RuntimeError('Route decoder changed fixed service set')
        checked = evaluate_plan(problem.domain, candidate, weights, model)
        deadline.check()
    except BudgetExpired:
        return rejected('deadline_exhausted', on_time=False)
    except ValueError as exc:
        if str(exc).startswith(_PHYSICAL_REJECTIONS):
            return rejected('physical_rejection', physical=False) | {'physical_error': str(exc)}
        raise
    if abs(checked['primary'] - query['after']['primary']) > 1e-7:
        raise ValueError('Quota projection differs from checked candidate score')
    if (checked['primary'], checked['secondary']) >= (before['primary'], before['secondary']):
        return rejected('not_lexicographically_improving', physical=True)
    # No acceptance after the total projection + decode + validation deadline.
    try:
        deadline.check()
    except BudgetExpired:
        return rejected('deadline_exhausted', on_time=False)
    return dict(accepted=True, reason='improved', objective=(checked['primary'], checked['secondary']),
        selected_tuple_ids=tuple(sorted(selected_ids)),
        selected_arc_ids=tuple(sorted(a.arc_id for a in candidate.arcs)),
        proposed_selected_tuple_ids=tuple(sorted(selected_ids)),
        projected_primary_gain=query['primary_gain'], physical_passed=True,
        within_budget=True, total_seconds=perf_counter()-start, plan=candidate,
        proposal=proposal, evidence=dict(projection=query, route=route,
            checked_objective=(checked['primary'], checked['secondary']),
            initial_objective=(before['primary'], before['secondary'])))
