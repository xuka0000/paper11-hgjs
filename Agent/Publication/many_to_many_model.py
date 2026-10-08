"""Opt-in quota/service/resource view of the existing finite decision domain.

This is an exact projection for the frozen quota-only benchmark, not a new
physical problem or a trained hypergraph policy. Query results check quota,
observation exclusivity and resource rows; vehicle paths and every other
physical constraint must still be checked by the original evaluator.
"""
from collections import Counter
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ServiceAlternative:
    service_id: str
    completion_layers: tuple[tuple[object, int], ...]
    vehicle_id: object = None
    uav_id: object = None


@dataclass(frozen=True)
class CompletionQuota:
    group_id: str
    task_ids: frozenset
    minimum: int
    priority: float
    deadline: int


@dataclass(frozen=True)
class SharedResource:
    resource_id: str
    kind: str
    members: tuple[str, ...]
    coefficients: tuple[float, ...]
    capacity: float
    fixed_load: float = 0.


class ManyToManyModel:
    version = 'quota-service-resource-m2m-v1'

    def __init__(self, services, quotas, resources, *, now=1, route_requirements=None):
        services, quotas, resources = tuple(services), tuple(quotas), tuple(resources)
        self.services = {s.service_id: s for s in services}
        self.quotas = {g.group_id: g for g in quotas}
        self.resources = {r.resource_id: r for r in resources}
        if (len(self.services), len(self.quotas), len(self.resources)) != (len(services), len(quotas), len(resources)):
            raise ValueError('Duplicate object identity')
        if now < 1:
            raise ValueError('Invalid current layer')
        self.now = now
        self.route_requirements = route_requirements or {}
        self.task_to_services = {}
        for s in services:
            tasks = [t for t, _ in s.completion_layers]
            if len(tasks) != len(set(tasks)) or any(k < now for _, k in s.completion_layers):
                raise ValueError('Repeated observation or past completion in future service alternative')
            for task in tasks:
                self.task_to_services.setdefault(task, set()).add(s.service_id)
        for g in quotas:
            if (not isinstance(g.minimum, int) or isinstance(g.minimum, bool)
                    or not 1 <= g.minimum <= len(g.task_ids) or g.deadline < now or
                    not isfinite(g.priority) or g.priority <= 0):
                raise ValueError('Invalid completion quota')
            if not g.task_ids <= self.task_to_services.keys():
                raise ValueError('Quota task absent from service domain')
        for r in resources:
            if (len(r.members) != len(r.coefficients) or len(set(r.members)) != len(r.members)
                    or not set(r.members) <= self.services.keys()
                    or not all(isfinite(a) and a > 0 for a in r.coefficients)
                    or not isfinite(r.capacity) or r.capacity <= 0
                    or not isfinite(r.fixed_load) or r.fixed_load < 0):
                raise ValueError('Invalid shared-resource relation')

    @classmethod
    def from_problem(cls, problem, model):
        """Project declared inputs, never labels, solver outcomes or future truth.

        Version1 deliberately supports the completed foundation quota profile.
        Role/window/expiry and executed-history models are rejected rather than
        silently flattened. The incumbent's planned observations are predictions,
        and never added to the executed observation ledger.
        """
        model.validate_domain(problem.domain)
        if model.records or model.completed_group_ids or any(
                len(g.requirements) != 1 or g.max_span_layers is not None for g in model.groups):
            raise ValueError('v1 adapter requires quota-only profile with no history, roles or span')
        from Agent.Publication.group_hypergraph import resource_factors
        from Agent.Run.publication_committed_service_domain import handling
        services = tuple(ServiceAlternative(c.tuple_id, tuple(c.completion_layers), c.vehicle_id, c.uav_id)
                         for c in problem.full_copies.values())
        quotas = tuple(CompletionQuota(g.group_id, g.task_ids, g.requirements[0].minimum,
                                      g.priority, g.deadline_layer) for g in model.groups)
        resources = tuple(SharedResource(r.factor_id, r.kind, r.members, r.coefficients,
                                        r.capacity, r.fixed_load) for r in resource_factors(problem))
        waits = {c.tuple_id: tuple(problem.domain.waiting.get(event)
                                 for event in handling(problem.domain.instance, c, model.now))
                 for c in problem.full_copies.values()}
        return cls(services, quotas, resources, now=model.now, route_requirements=waits)

    def evaluate(self, selected):
        selected = frozenset(selected)
        if not selected <= self.services.keys():
            raise ValueError('Unknown service alternative')
        events = [event for key in selected for event in self.services[key].completion_layers]
        counts = Counter(task for task, _ in events)
        duplicate_tasks = tuple(sorted((task for task, count in counts.items() if count > 1), key=str))
        groups = {}
        for g in self.quotas.values():
            observed = {task for task, k in events if task in g.task_ids and self.now <= k <= g.deadline}
            groups[g.group_id] = dict(distinct_observations=len(observed), minimum=g.minimum,
                remaining=max(0, g.minimum-len(observed)), complete=len(observed) >= g.minimum)
        loads = {}
        for r in self.resources.values():
            load = r.fixed_load + sum(a for key, a in zip(r.members, r.coefficients) if key in selected)
            loads[r.resource_id] = dict(kind=r.kind, load=load, capacity=r.capacity,
                slack=r.capacity-load, overloaded=load > r.capacity+1e-9)
        return dict(primary=sum(g.priority for g in self.quotas.values() if not groups[g.group_id]['complete']),
            groups=groups, resource_loads=loads, duplicate_tasks=duplicate_tasks,
            projected_constraints_pass=not duplicate_tasks and not any(r['overloaded'] for r in loads.values()),
            full_physical_feasibility_checked=False)

    def query(self, incumbent, *, add=(), release=()):
        """State-conditioned joint proposal, not a solver or execution command."""
        incumbent, add, release = map(frozenset, (incumbent, add, release))
        if not release <= incumbent or add & release:
            raise ValueError('Release must be incumbent-only and disjoint from additions')
        if add & incumbent:
            raise ValueError('Already selected service is not an addition')
        before = self.evaluate(incumbent)
        selected = (incumbent-release) | add
        after = self.evaluate(selected)
        changed_groups = tuple(k for k in self.quotas if before['groups'][k] != after['groups'][k])
        changed_resources = tuple(k for k in self.resources
                                  if before['resource_loads'][k]['load'] != after['resource_loads'][k]['load'])
        return dict(before=before, after=after, primary_gain=before['primary']-after['primary'],
            newly_completed=tuple(k for k in self.quotas if after['groups'][k]['complete'] and not before['groups'][k]['complete']),
            lost_completions=tuple(k for k in self.quotas if before['groups'][k]['complete'] and not after['groups'][k]['complete']),
            changed_groups=changed_groups, changed_resources=changed_resources,
            add=tuple(sorted(add)), release=tuple(sorted(release)), selected=tuple(sorted(selected)),
            required_wait_arc_ids={k:self.route_requirements.get(k, ()) for k in sorted(add)},
            gain_requires_physical_validation=True)

    def inventory(self):
        return dict(version=self.version, observation_nodes=len(self.task_to_services),
            service_decision_nodes=len(self.services), completion_relations=len(self.quotas),
            resource_relations=len(self.resources), observation_service_links=sum(map(len, self.task_to_services.values())),
            completion_observation_links=sum(len(g.task_ids) for g in self.quotas.values()),
            resource_service_links=sum(len(r.members) for r in self.resources.values()),
            feature_source='declared current domain and proposed incumbent; no outcomes or labels')
