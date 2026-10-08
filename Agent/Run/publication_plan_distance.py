"""Sparse Hamming distance for the adjustable physical plan.

Implements main.tex Eq. rolling-plan-change without enumerating zero-valued
service copies. Missing old selections still count; all new selections have a
zero reference. The first plan has no predecessor and incurs no change cost.
Anchors, occupied roads, completed actions and locked decisions are physical
state, not adjustable decisions. Return-time arcs remain adjustable.

A flight's stable key includes its original physical service and both timed
endpoints. Residual record renaming does not change this key. An unlocked
airborne recovery-time change does. This is an isolated revision; the frozen
cancellation-only experiment and production implementation are unchanged.
"""
from dataclasses import dataclass

from Agent.Run.publication_committed_service_domain import objective as legacy_objective


def service_key(copy):
    return (copy.physical_candidate_id, copy.vehicle_id, copy.uav_id,
            copy.launch_node, copy.recovery_node, tuple(copy.task_ids),
            copy.launch_layer, copy.recovery_layer)


@dataclass(frozen=True)
class PlanReference:
    present: bool
    now: int
    preparation_layers: int
    fixed_arcs: frozenset
    fixed_services: frozenset
    old_arcs: frozenset
    old_services: frozenset

    def arc_adjustable(self, arc):
        return (arc.arc_type in ('road', 'wait', 'terminal')
                and arc.arc_id not in self.fixed_arcs
                and arc.departure_layer >= self.now
                and (arc.arc_type != 'terminal' or arc.arrival_layer > self.now))

    def copy_adjustable(self, copy):
        return (copy.recovery_layer > self.now
                and service_key(copy) not in self.fixed_services
                and not (copy.launch_layer - self.preparation_layers < self.now < copy.launch_layer))

    def arc_coefficient(self, arc):
        return int(self.present and self.arc_adjustable(arc)) * (1 - 2 * int(arc.arc_id in self.old_arcs))

    def copy_coefficient(self, copy):
        return int(self.present and self.copy_adjustable(copy)) * (1 - 2 * int(service_key(copy) in self.old_services))


def plan_reference(domain):
    state = domain.context.state
    previous = domain.context.previous
    fixed_services = frozenset(service_key(c) for c in previous.copies
                              if c.tuple_id in state.locked_tuple_ids)
    empty = PlanReference(bool(state.epoch or previous.arcs or previous.copies),
                          state.current_layer, domain.instance.preparation_layers,
                          frozenset(domain.required_arc_ids) | state.locked_arc_ids,
                          fixed_services, frozenset(), frozenset())
    from dataclasses import replace
    return replace(empty,
                   old_arcs=frozenset(a.arc_id for a in previous.arcs if empty.arc_adjustable(a)),
                   old_services=frozenset(service_key(c) for c in previous.copies if empty.copy_adjustable(c)))


def distance_breakdown(domain, plan):
    ref = plan_reference(domain)
    selected_arcs = {a.arc_id for a in plan.arcs if ref.arc_adjustable(a)}
    selected_services = {service_key(c) for c in plan.copies if ref.copy_adjustable(c)}
    values = dict(added_arcs=len(selected_arcs - ref.old_arcs),
                  cancelled_arcs=len(ref.old_arcs - selected_arcs),
                  added_services=len(selected_services - ref.old_services),
                  cancelled_services=len(ref.old_services - selected_services))
    if not ref.present:
        values = dict.fromkeys(values, 0)
    return dict(**values, total=sum(values.values()), reference_present=ref.present)


def objective(domain, plan, weights):
    primary, operating = legacy_objective(domain, plan, dict(weights, plan_change_weight=0.))
    return primary, operating + weights['plan_change_weight'] * distance_breakdown(domain, plan)['total']
