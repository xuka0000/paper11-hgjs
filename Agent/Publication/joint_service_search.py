"""Sequential bounded service decisions; route repair never chooses services.

The deterministic policy below is a baseline, not a trained hypergraph method.
All catalogue services enter extension scoring. Beam pruning limits search,
not the physical domain used by the evaluator or competing methods.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from random import Random
from time import perf_counter

from Agent.Publication.budget import BudgetExpired
from Agent.Publication.component_objective import evaluate_plan
from Agent.Publication.joint_service_actions import ActionProposal, evaluate_action
from Agent.Publication.many_to_many_model import ManyToManyModel


@dataclass(frozen=True)
class SearchLimits:
    additions: int = 4
    releases: int = 4
    target_groups: int = 2
    beam: int = 8
    expansion: int = 8
    terminals: int = 32

    def __post_init__(self):
        upper = (4, 4, 2, 8, 8, 32)
        values = (self.additions, self.releases, self.target_groups,
                  self.beam, self.expansion, self.terminals)
        if any(type(n) is not int or n < 0 or n > cap for n, cap in zip(values, upper)):
            raise ValueError('Search limits exceed the declared action protocol')
        if not all(values[i] > 0 for i in (2, 3, 4, 5)):
            raise ValueError('Positive group, beam, expansion and terminal limits required')


class ActionContext:
    """Input-only sparse indexes; no solver label or future observation."""

    def __init__(self, view, selected, *, costs=None, immutable=(), deadline=None):
        self.view = view
        self.selected = frozenset(selected)
        self.immutable = frozenset(immutable)
        if not self.immutable <= self.selected or not self.selected <= view.services.keys():
            raise ValueError('Invalid incumbent/immutable services')
        self.ids = tuple(sorted(view.services))
        self.costs = costs or {}
        self.tasks = {k: frozenset(t for t, _ in s.completion_layers) for k, s in view.services.items()}
        self.groups = {}
        self.valid_tasks = {}
        self.rows = defaultdict(dict)
        task_groups = defaultdict(set)
        for gid, g in view.quotas.items():
            for t in g.task_ids:
                task_groups[t].add(gid)
        for i, (k, s) in enumerate(view.services.items()):
            if deadline is not None and i % 128 == 0:
                deadline.check()
            self.groups[k] = frozenset(g for t in self.tasks[k] for g in task_groups[t])
            self.valid_tasks[k] = {g: frozenset(t for t, end in s.completion_layers
                                    if t in view.quotas[g].task_ids and end <= view.quotas[g].deadline)
                                   for g in self.groups[k]}
        for rid, r in view.resources.items():
            for k, a in zip(r.members, r.coefficients):
                self.rows[k][rid] = a
        before = view.evaluate(self.selected)
        if not before['projected_constraints_pass']:
            raise ValueError('Incumbent violates projected constraints')
        self.base_loads = {k: v['load'] for k, v in before['resource_loads'].items()}
        self.before_primary = before['primary']
        self.base_counts = Counter(t for k in self.selected for t in self.tasks[k])
        self.base_groups = {g: Counter(t for k in self.selected for t in self.valid_tasks[k].get(g, ()))
                            for g in view.quotas}
        self.incumbent_task_services = defaultdict(set)
        self.removable_rows = defaultdict(list)
        for k in self.selected:
            for t in self.tasks[k]:
                self.incumbent_task_services[t].add(k)
            if k not in self.immutable:
                for rid, coefficient in self.rows[k].items():
                    self.removable_rows[rid].append((coefficient, k))
        # Positive coefficients only; absent members contribute zero. Sorting
        # once gives the identical top-k repair bound for every partial release.
        for members in self.removable_rows.values():
            members.sort(reverse=True)
        self.base_progress = sum(view.quotas[g].priority * min(1., len(v) / view.quotas[g].minimum)
                                 for g, v in self.base_groups.items())
        if deadline is not None:
            deadline.check()

    def describe(self, action, limits):
        """Exact quota delta and necessary repair bounds for a partial action.

        A temporary overload or duplicate with the incumbent is permitted when
        remaining releases could repair it. This is not a physical certificate.
        """
        add, release = frozenset(action.add), frozenset(action.release)
        if (len(add) != len(action.add) or len(release) != len(action.release)
                or len(add) > limits.additions or len(release) > limits.releases
                or add & self.selected or not release <= self.selected
                or release & self.immutable or not add <= self.view.services.keys()):
            return None
        targets = frozenset(g for k in add for g in self.groups[k])
        if len(targets) > limits.target_groups:
            return None
        new_tasks = Counter(t for k in add for t in self.tasks[k])
        if any(n > 1 for n in new_tasks.values()):
            return None
        remaining = self.selected - release
        duplicates = {k for t in new_tasks for k in self.incumbent_task_services[t]
                      if k not in release}
        slots = limits.releases - len(release)
        if duplicates & self.immutable or len(duplicates) > slots:
            return None
        delta = defaultdict(float)
        for keys, sign in ((add, 1), (release, -1)):
            for k in keys:
                for rid, coefficient in self.rows[k].items():
                    delta[rid] += sign * coefficient
        overload = 0.
        for rid, change in delta.items():
            excess = max(0., self.base_loads[rid] + change - self.view.resources[rid].capacity)
            if excess > 1e-9:
                repair = 0.
                used = 0
                for coefficient, key in self.removable_rows[rid]:
                    if used >= slots:
                        break
                    if key not in release:
                        repair += coefficient
                        used += 1
                if repair < excess - 1e-9:
                    return None
                overload += excess
        touched = frozenset(g for k in add | release for g in self.groups[k])
        primary, progress = self.before_primary, self.base_progress
        lost, newly = [], []
        for gid in touched:
            observations = set(self.base_groups[gid])
            for k in release:
                observations.difference_update(self.valid_tasks[k].get(gid, ()))
            for k in add:
                observations.update(self.valid_tasks[k].get(gid, ()))
            n = len(observations)
            g = self.view.quotas[gid]
            old_n = len(self.base_groups[gid])
            old, new = old_n >= g.minimum, n >= g.minimum
            primary += g.priority * (old - new)
            progress += g.priority * (min(1., n / g.minimum) - min(1., old_n / g.minimum))
            if old and not new:
                lost.append(gid)
            if new and not old:
                newly.append(gid)
        direct_delta = sum(self.costs.get(k, 0.) for k in add) - sum(self.costs.get(k, 0.) for k in release)
        return dict(primary_gain=self.before_primary-primary,
                    quota_progress=progress-self.base_progress, direct_cost_delta=direct_delta,
                    overload=overload, duplicate_releases=len(duplicates),
                    terminal=not duplicates and overload <= 1e-9,
                    target_groups=tuple(sorted(targets)), lost_groups=tuple(sorted(lost)),
                    newly_completed=tuple(sorted(newly)), affected_resources=tuple(sorted(delta)),
                    action_size=len(add)+len(release))


class ConditionalHeuristic:
    """Quota potential permits zero-immediate-gain partial additions."""
    def score(self, context, partial, extensions, descriptors):
        return [10*d['primary_gain'] + d['quota_progress']
                - .02*d['direct_cost_delta'] - .1*d['overload']
                - .1*d['duplicate_releases'] - .001*d['action_size'] for d in descriptors]


class EpsilonGreedy(ConditionalHeuristic):
    def __init__(self, seed, epsilon=.1):
        if not 0 <= epsilon <= 1:
            raise ValueError('Invalid epsilon')
        self.rng, self.epsilon = Random(seed), epsilon

    def score(self, context, partial, extensions, descriptors):
        values = super().score(context, partial, extensions, descriptors)
        if self.rng.random() < self.epsilon:
            return [self.rng.random() for _ in values]
        return values


def action_signature(context, action):
    """A diversity stratum for pruning, never a catalogue filter."""
    return (tuple(sorted({str(t) for k in action.add for t in context.tasks[k]})),
            tuple(sorted({str(t) for k in action.release for t in context.tasks[k]})))


def _diverse_best(items, count, context):
    ranked = sorted(items, key=lambda x: (-x[0], x[1].add, x[1].release))
    picked, rest, seen = [], [], set()
    for item in ranked:
        signature = action_signature(context, item[1])
        if signature not in seen and len(picked) < count:
            picked.append(item)
            seen.add(signature)
        else:
            rest.append(item)
    return sorted(picked + rest[:max(0, count-len(picked))], key=lambda x: -x[0])


def iter_proposals(context, policy, deadline, limits=SearchLimits(), *, excluded=()):
    """Yield bounded terminal actions while expanding partial add/release edits.

    Every available service is scored before each beam prune. No target group,
    service closure or chosen route is supplied by a privileged oracle.
    """
    beam = [ActionProposal()]
    emitted = set(excluded)
    count = 0
    for depth in range(limits.additions + limits.releases):
        successors = {}
        for partial in beam:
            deadline.check()
            extensions, descriptors = [], []
            ops = []
            if len(partial.add) < limits.additions:
                ops.extend(('add', k) for k in context.ids if k not in context.selected and k not in partial.add)
            if len(partial.release) < limits.releases:
                ops.extend(('release', k) for k in sorted(context.selected-context.immutable) if k not in partial.release)
            for i, (kind, key) in enumerate(ops):
                if i % 64 == 0:
                    deadline.check()
                action = ActionProposal(tuple(sorted(partial.add+(key,))) if kind == 'add' else partial.add,
                                        tuple(sorted(partial.release+(key,))) if kind == 'release' else partial.release)
                descriptor = context.describe(action, limits)
                if descriptor is not None:
                    extensions.append(action)
                    descriptors.append(descriptor)
            scores = policy.score(context, partial, extensions, descriptors)
            deadline.check()
            if len(scores) != len(extensions) or any(not isfinite(float(v)) for v in scores):
                raise ValueError('Policy must return one finite score per eligible extension')
            items = [(float(v), a, d) for v, a, d in zip(scores, extensions, descriptors)]
            for item in _diverse_best(items, limits.expansion, context):
                action = item[1]
                if action not in successors or item[0] > successors[action][0]:
                    successors[action] = item
        if not successors:
            return
        best = _diverse_best(successors.values(), limits.beam, context)
        beam = [item[1] for item in best]
        # Up to four per depth leaves capacity for deeper joint exchanges.
        # Direct service cost does not bound road or makespan savings. Only the
        # full route evaluator may reject an equal-primary exchange on cost.
        terminals = [item for item in best if item[2]['terminal']]
        for score, action, descriptor in terminals[:4]:
            if action in emitted:
                continue
            deadline.check()
            emitted.add(action)
            count += 1
            yield action, dict(descriptor, policy_score=score, depth=depth+1,
                               catalogue_size=len(context.ids))
            if count >= limits.terminals:
                return


def run_joint_search(problem, model, incumbent, weights, deadline, *, policy=None,
                     limits=SearchLimits(), max_iterations=None, proposal_generator=iter_proposals):
    """One timed pipeline, retaining only independently checked improvements."""
    start = perf_counter()
    policy = ConditionalHeuristic() if policy is None else policy
    trace, iterations = [], 0
    best = incumbent
    score = evaluate_plan(problem.domain, best, weights, model)
    initial = objective = (score['primary'], score['secondary'])
    reason = 'no_improving_action'
    try:
        deadline.check()
        view = ManyToManyModel.from_problem(problem, model)
        deadline.check()
        costs = {k: c.secondary_cost + weights['risk_weight']*c.risk_exposure
                 + weights['energy_weight']*c.expected_energy for k, c in problem.full_copies.items()}
        while max_iterations is None or iterations < max_iterations:
            deadline.check()
            selected = frozenset(c.tuple_id for c in best.copies)
            context = ActionContext(view, selected, costs=costs,
                immutable=selected & problem.domain.context.state.locked_tuple_ids, deadline=deadline)
            iterations += 1
            improved = False
            for action, descriptor in proposal_generator(context, policy, deadline, limits):
                result = evaluate_action(problem, model, best, action, weights, deadline,
                                         _projection=view, _limits=limits)
                trace.append(dict(iteration=iterations, add=action.add, release=action.release,
                    descriptor=descriptor, accepted=result['accepted'], reason=result['reason'],
                    physical_passed=result['physical_passed'], within_budget=result['within_budget'],
                    objective=result['objective'], total_seconds=result['total_seconds'],
                    proposed_selected_tuple_ids=result['proposed_selected_tuple_ids'],
                    selected_tuple_ids=result['selected_tuple_ids']))
                if result['accepted']:
                    best, objective = result['plan'], result['objective']
                    improved = True
                    break  # All subsequent decisions use the new incumbent.
                if not result['within_budget']:
                    raise BudgetExpired('Action exceeded pipeline deadline')
            if not improved:
                break
        else:
            reason = 'iteration_limit'
    except BudgetExpired:
        reason = 'deadline_exhausted'
    return dict(plan=best, initial_objective=initial, objective=objective, trace=trace,
                iterations=iterations, reason=reason, total_seconds=perf_counter()-start,
                accepted=sum(row['accepted'] for row in trace), policy=type(policy).__name__,
                scientific_claim='Untrained action-search implementation, not hypergraph benefit')
