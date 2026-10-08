"""Exact road-arc elimination conditional on an ordered mandatory path prefix/suffix.

Every feasible unit vehicle flow is a source--sink path in the temporal DAG.
It visits each mandatory arc in temporal order. Thus each intervening path
segment must be reachable from the preceding head and reach the next tail.
The test is necessary, never a heuristic restriction of the feasible paths.
"""
from collections import defaultdict


def vertices(arc):
    return (('source', -1) if arc.tail == 'source' else (arc.tail, arc.departure_layer),
            ('sink', -1) if arc.head == 'sink' else (arc.head, arc.arrival_layer))


def prune_forced_paths(arcs, required, deadline=None):
    by_vehicle = defaultdict(list)
    for arc in arcs:
        by_vehicle[arc.vehicle_id].append(arc)
    retained = set()
    for vehicle_arcs in by_vehicle.values():
        if deadline is not None: deadline.check()
        mandatory = sorted((a for a in vehicle_arcs if a.arc_id in required),
                           key=lambda a: (a.departure_layer, a.arrival_layer, a.arc_id))
        # No mandatory segment provides no additional temporal restriction.
        if not mandatory:
            retained.update(a.arc_id for a in vehicle_arcs)
            continue
        outgoing, incoming = defaultdict(list), defaultdict(list)
        for arc in vehicle_arcs:
            if arc.arc_id in required: continue
            tail, head = vertices(arc)
            outgoing[tail].append(head); incoming[head].append(tail)

        def reach(start, adjacency):
            reached, pending = {start}, [start]
            while pending:
                if deadline is not None: deadline.check()
                for node in adjacency[pending.pop()]:
                    if node not in reached: reached.add(node); pending.append(node)
            return reached

        ends = [vertices(a) for a in mandatory]
        intervals = list(zip([('source', -1)] + [h for _, h in ends],
                             [t for t, _ in ends] + [('sink', -1)]))
        for start, finish in intervals:
            if start == finish: continue
            forward, backward = reach(start, outgoing), reach(finish, incoming)
            for arc in vehicle_arcs:
                tail, head = vertices(arc)
                if tail in forward and head in backward: retained.add(arc.arc_id)
        retained.update(a.arc_id for a in mandatory)
    return tuple(a for a in arcs if a.arc_id in retained)
