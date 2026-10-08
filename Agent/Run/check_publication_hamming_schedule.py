"""Independent physical feasibility and union-domain binary-vector accounting."""
from math import isfinite
from Agent.Run.check_publication_committed_schedule import check_plan as check_physics


def independent_distance(domain, plan):
    state = domain.context.state
    previous = domain.context.previous
    if not state.epoch and not previous.arcs and not previous.copies:
        return 0
    # Deliberately do not import the optimization reference/key functions.
    def flight(c):
        return (c.physical_candidate_id, c.vehicle_id, c.uav_id, c.launch_node,
                c.recovery_node, tuple(c.task_ids), c.launch_layer, c.recovery_layer)
    immutable = {flight(c) for c in previous.copies if c.tuple_id in state.locked_tuple_ids}
    fixed = set(state.locked_arc_ids) | set(domain.required_arc_ids)
    def vector(plan):
        result = {}
        for a in plan.arcs:
            if (a.arc_type in ('road', 'wait', 'terminal') and a.arc_id not in fixed
                    and a.departure_layer >= state.current_layer
                    and (a.arc_type != 'terminal' or a.arrival_layer > state.current_layer)):
                result[('arc', a.arc_id)] = 1
        for c in plan.copies:
            if (c.recovery_layer > state.current_layer and flight(c) not in immutable
                    and not c.launch_layer - domain.instance.preparation_layers < state.current_layer < c.launch_layer):
                result[('flight', flight(c))] = 1
        return result
    old, new = vector(previous), vector(plan)
    return sum(abs(old.get(k, 0) - new.get(k, 0)) for k in old.keys() | new.keys())


def check_plan(domain, plan, weights, reported=None):
    result = check_physics(domain, plan, dict(weights, plan_change_weight=0.))
    result['plan_distance'] = independent_distance(domain, plan)
    result['secondary'] += weights['plan_change_weight'] * result['plan_distance']
    if reported is not None and (any(not isfinite(x) for x in reported)
            or abs(reported[0] - result['primary']) > 1e-7
            or abs(reported[1] - result['secondary']) > 1e-7):
        raise ValueError('Hamming objective arithmetic mismatch')
    return result
