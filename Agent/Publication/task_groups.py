"""Typed inspection requirements over the existing physical observation tasks.

Time is the existing absolute integer layer. A task denotes one distinct
admitted observation point, not an entire component. No additional flight or
sensor accuracy is inferred by this overlay. It changes completion accounting;
the road, energy, handling, occupancy and commitment model remains in force.
"""
from dataclasses import dataclass, replace
from itertools import combinations, product
from math import isfinite


@dataclass(frozen=True)
class ObservationRequirement:
    requirement_id: str
    alternatives: tuple[int, ...]
    minimum: int = 1

    def __post_init__(self):
        if (not self.requirement_id or len(set(self.alternatives)) != len(self.alternatives)
                or not 1 <= self.minimum <= len(self.alternatives)):
            raise ValueError('Invalid distinct observation requirement')


@dataclass(frozen=True)
class TaskGroup:
    group_id: str
    requirements: tuple[ObservationRequirement, ...]
    priority: float
    deadline_layer: int
    max_span_layers: int | None = None
    source: str = 'declared synthetic multi-view inspection requirement'

    def __post_init__(self):
        ids = [r.requirement_id for r in self.requirements]
        members = [t for r in self.requirements for t in r.alternatives]
        if (not self.group_id or not ids or len(set(ids)) != len(ids)
                or len(set(members)) != len(members)):
            raise ValueError('Group requires distinct roles and nonoverlapping observation identities')
        if not isfinite(self.priority) or self.priority <= 0 or self.deadline_layer < 1:
            raise ValueError('Positive finite priority and absolute deadline required')
        if self.max_span_layers is not None and self.max_span_layers < 0:
            raise ValueError('Observation span must be nonnegative')

    @property
    def task_ids(self):
        return frozenset(t for r in self.requirements for t in r.alternatives)


@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    task_id: int
    completed_layer: int
    available_layer: int
    valid_until_layer: int | None = None
    quality_ok: bool = True
    source: str = 'executed physical observation'

    def __post_init__(self):
        if not self.observation_id or self.completed_layer < 1 or self.available_layer < self.completed_layer:
            raise ValueError('Observation cannot be available before acquisition')
        if self.valid_until_layer is not None and self.valid_until_layer < self.completed_layer:
            raise ValueError('Observation expires before acquisition')


@dataclass(frozen=True)
class RoleWindow:
    requirement: ObservationRequirement
    role_index: int
    historical_tasks: frozenset[int]
    terms: tuple[tuple[str, float], ...]
    lower: int


@dataclass(frozen=True)
class CompletionWindow:
    group_id: str
    end: int
    roles: tuple[RoleWindow, ...]


@dataclass(frozen=True)
class TaskGroupModel:
    groups: tuple[TaskGroup, ...]
    now: int
    records: tuple[ObservationRecord, ...] = ()
    completed_group_ids: frozenset[str] = frozenset()
    version: str = 'inspection-groups-v1'

    def __post_init__(self):
        ids = [g.group_id for g in self.groups]
        events = [r.observation_id for r in self.records]
        if self.now < 1 or len(set(ids)) != len(ids) or len(set(events)) != len(events):
            raise ValueError('Invalid group/record identity or current layer')
        if set(self.completed_group_ids) - set(ids):
            raise ValueError('Unknown completed group')
        if any(r.available_layer > self.now for r in self.records):
            raise ValueError('Future observation unavailable at decision time')

    @classmethod
    def legacy(cls, task_ids, penalties, now, horizon):
        return cls(tuple(TaskGroup(str(t), (ObservationRequirement(str(t), (t,)),),
                                   float(penalties[t]), horizon) for t in sorted(task_ids)), now)

    def validate_domain(self, domain):
        if self.now != domain.context.state.current_layer:
            raise ValueError('Group and physical clocks differ')
        universe = set(domain.problem.task_ids) | set(domain.context.state.done_tasks)
        if any(g.task_ids - universe for g in self.groups):
            raise ValueError('Observation requirement absent from physical domain')

    def evaluate(self, plan):
        """Independent combinatorial witness check; no optimizer variables used.

        Repeated observations of one point cannot fill two required views. A
        durable completed certificate persists; partial records may expire.
        """
        future = []
        for copy in plan.copies:
            future.extend(ObservationRecord(f'{copy.tuple_id}:{t}', t, k, k,
                                           source='predicted admitted observation; quality assumed')
                          for t, k in copy.completion_layers if k >= self.now)
        events = tuple(r for r in self.records if r.quality_ok) + tuple(future)
        result = {}
        for group in self.groups:
            if group.group_id in self.completed_group_ids:
                result[group.group_id] = dict(complete=True, witness='previously_executed_certificate', time=None)
                continue
            role_choices = []
            for requirement in group.requirements:
                eligible = [r for r in events if r.task_id in requirement.alternatives
                            and r.available_layer <= group.deadline_layer
                            and (r.valid_until_layer is None or r.valid_until_layer >= self.now)]
                choices = [combo for combo in combinations(eligible, requirement.minimum)
                           if len({r.task_id for r in combo}) == requirement.minimum]
                role_choices.append(choices)
            certificate = None
            if all(role_choices):
                for selected in product(*role_choices):
                    records = tuple(r for part in selected for r in part)
                    end = max(self.now, *(r.available_layer for r in records))
                    start = min(r.completed_layer for r in records)
                    if end > group.deadline_layer:
                        continue
                    if any(r.valid_until_layer is not None and end > r.valid_until_layer for r in records):
                        continue
                    if group.max_span_layers is not None and end - start > group.max_span_layers:
                        continue
                    certificate = dict(complete=True, witness=tuple(r.observation_id for r in records), time=end)
                    break
            result[group.group_id] = certificate or dict(complete=False, witness=(), time=None)
        missed = sum(g.priority for g in self.groups if not result[g.group_id]['complete'])
        return dict(primary=float(missed), metric='weighted_incomplete_components', groups=result)

    def completion_windows(self, copy_records):
        """Observable role rows shared by the MILP and the v2 encoder.

        The independent evaluator above deliberately does not use these rows.
        Physical task coverage prevents alternative copies from counting twice.
        """
        event_times = {self.now}
        for copy in copy_records.values():
            event_times.update(k for _, k in copy.completion_layers if k >= self.now)
        for record in self.records:
            event_times.add(record.available_layer)
        result = {}
        for group in self.groups:
            windows = []
            if group.group_id in self.completed_group_ids:
                result[group.group_id] = ()
                continue
            for end in sorted(k for k in event_times if self.now <= k <= group.deadline_layer):
                lower = 1 if group.max_span_layers is None else end - group.max_span_layers
                roles = []
                possible = True
                for role_index, requirement in enumerate(group.requirements):
                    past = {r.task_id for r in self.records if r.quality_ok
                            and r.task_id in requirement.alternatives
                            and lower <= r.completed_layer and r.available_layer <= end
                            and (r.valid_until_layer is None or end <= r.valid_until_layer)}
                    terms = []
                    possible_tasks = set(past)
                    for key, copy in copy_records.items():
                        members = {t for t, k in copy.completion_layers
                                   if t in requirement.alternatives and t not in past
                                   and max(lower, self.now) <= k <= end}
                        if members:
                            terms.append((key, float(len(members))))
                            possible_tasks.update(members)
                    if len(possible_tasks) < requirement.minimum:
                        possible = False
                        break
                    roles.append(RoleWindow(requirement, role_index, frozenset(past),
                                            tuple(terms), lower))
                if not possible:
                    continue
                windows.append(CompletionWindow(group.group_id, end, tuple(roles)))
            result[group.group_id] = tuple(windows)
        return result

    def add_milp(self, boolean, row, chosen_vars, copy_records, infinity):
        """Exact role/window inequalities and one completion witness per group."""
        primary_terms, hint = [], {}
        all_windows = self.completion_windows(copy_records)
        for group in self.groups:
            miss = boolean('group_miss:' + group.group_id)
            primary_terms.append((miss, group.priority))
            if group.group_id in self.completed_group_ids:
                row([(miss, 1.)], 0., 0.)
                hint[group.group_id] = (miss, [])
                continue
            windows = []
            for window in all_windows[group.group_id]:
                end = window.end
                witness = boolean(f'group_window:{group.group_id}:{end}')
                for role in window.roles:
                    terms = [(chosen_vars[key], coefficient) for key, coefficient in role.terms]
                    row(terms + [(witness, -float(role.requirement.minimum))],
                        -float(len(role.historical_tasks)), infinity)
                windows.append((end, witness))
            row([(miss, 1.)] + [(v, 1.) for _, v in windows], 1., 1.)
            hint[group.group_id] = (miss, windows)
        return primary_terms, hint

    def executed_update(self, plan, end, validity_layers=None):
        if end < self.now:
            raise ValueError('Cannot rewind observations')
        known = {r.observation_id: r for r in self.records}
        for copy in plan.copies:
            # Use the stable physical service and absolute observation time, not
            # a residual-copy ID which may change after rolling reconstruction.
            for task, layer in copy.completion_layers:
                if self.now < layer <= end:
                    key = f'{copy.physical_candidate_id}:{task}:{layer}'
                    known[key] = ObservationRecord(key, task, layer, layer,
                                                   None if validity_layers is None else layer + validity_layers)
        # Certify at each actual event time, before partial observations expire.
        complete = set(self.completed_group_ids)
        for at in sorted({r.available_layer for r in known.values() if self.now <= r.available_layer <= end}):
            current = replace(self, now=at, records=tuple(r for r in known.values() if r.available_layer <= at),
                              completed_group_ids=frozenset(complete))
            from Agent.Run.publication_committed_service_domain import Plan
            checked = current.evaluate(Plan())
            complete.update(g for g, row in checked['groups'].items() if row['complete'])
        return replace(self, now=end, records=tuple(known.values()), completed_group_ids=frozenset(complete))

    def reopen_expired(self, state):
        if state.current_layer != self.now:
            raise ValueError('Group ledger and execution clock differ')
        # Freshness is group-specific: an unexpired image may still be too old
        # for a group's maximum acquisition span. Reopen if any unfinished,
        # still feasible group needs a fresh version of the observation.
        needed = set()
        for group in self.groups:
            if group.group_id in self.completed_group_ids or group.deadline_layer < self.now:
                continue
            valid = {r.task_id for r in self.records if r.quality_ok
                     and (r.valid_until_layer is None or r.valid_until_layer >= self.now)
                     and (group.max_span_layers is None or self.now-r.completed_layer <= group.max_span_layers)}
            needed.update(group.task_ids-valid)
        airborne = {t for ts in state.airborne_tasks.values() for t in ts}
        reopen = needed & set(state.done_tasks) - airborne
        return replace(state, done_tasks=frozenset(set(state.done_tasks) - reopen),
                       remaining_tasks=frozenset(set(state.remaining_tasks) | reopen))
