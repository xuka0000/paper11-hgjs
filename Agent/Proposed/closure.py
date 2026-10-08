from dataclasses import dataclass


@dataclass(frozen=True)
class ClosureBudget:
    max_tasks: int
    max_tuples: int
    max_vehicles: int
    max_span: int


@dataclass(frozen=True)
class RegionClosure:
    accepted: bool
    reason: str
    task_ids: tuple[int, ...]
    tuple_ids: tuple[str, ...]
    vehicle_ids: tuple[int, ...]
    uav_ids: tuple[int, ...]
    arc_ids: tuple[str, ...]
    anchors: tuple[tuple[int, tuple[str, int], tuple[str, int]], ...]


def close_region(state, core_region, budget: ClosureBudget) -> RegionClosure:
    tuple_by_id = {item.tuple_id: item for item in state.tuples}
    arc_by_id = {arc.arc_id: arc for arc in state.arcs}
    tasks = {
        int(node_id.split(":", 1)[1])
        for node_id in core_region.node_ids
        if node_id.startswith("task:")
    }
    tuple_ids = {node_id for node_id in core_region.node_ids if node_id in tuple_by_id}
    arc_ids = {
        node_id.split(":", 1)[1]
        for node_id in core_region.node_ids
        if node_id.startswith("arc:") and node_id.split(":", 1)[1] in arc_by_id
    }
    vehicle_ids = {
        int(node_id.split(":", 1)[1])
        for node_id in core_region.node_ids
        if node_id.startswith("vehicle:")
    }
    uav_ids = {
        int(node_id.split(":", 1)[1])
        for node_id in core_region.node_ids
        if node_id.startswith("uav:")
    }
    interval_keys = {
        node_id.split(":", 1)[1]
        for node_id in core_region.node_ids
        if node_id.startswith("interval:")
    }
    operation_keys = {
        node_id.split(":", 1)[1]
        for node_id in core_region.node_ids
        if node_id.startswith("operation:")
    }

    def absorb(item) -> None:
        tuple_ids.add(item.tuple_id)
        tasks.update(item.task_ids)
        vehicle_ids.add(item.vehicle_id)
        uav_ids.add(item.uav_id)
        arc_ids.update(item.required_arc_ids)

    for tuple_id in tuple(tuple_ids):
        absorb(tuple_by_id[tuple_id])

    changed = True
    while changed:
        changed = False
        for item in state.tuples:
            if item.tuple_id not in state.selected_tuple_ids or item.tuple_id in tuple_ids:
                continue
            intersects = (
                bool(set(item.task_ids) & tasks)
                or bool(set(item.interval_keys) & interval_keys)
                or bool(set(item.operation_keys) & operation_keys)
            )
            if intersects:
                absorb(item)
                changed = True

    anchors = []
    for vehicle_id in sorted(vehicle_ids):
        vehicle_tuples = [
            tuple_by_id[tuple_id]
            for tuple_id in tuple_ids
            if tuple_by_id[tuple_id].vehicle_id == vehicle_id
        ]
        vehicle_core_arcs = [
            arc_by_id[arc_id]
            for arc_id in arc_ids
            if arc_id in arc_by_id and arc_by_id[arc_id].vehicle_id == vehicle_id
        ]
        layers = [
            layer
            for item in vehicle_tuples
            for layer in (item.launch_layer, item.recovery_layer)
        ]
        layers.extend(
            layer
            for arc in vehicle_core_arcs
            for layer in (arc.departure_layer, arc.arrival_layer)
        )
        if not layers:
            continue
        lower, upper = min(layers), max(layers)
        selected_path = sorted(
            (
                arc
                for arc in state.arcs
                if arc.vehicle_id == vehicle_id and arc.arc_id in state.selected_arc_ids
            ),
            key=lambda arc: (arc.departure_layer, arc.arrival_layer, arc.arc_id),
        )
        if not selected_path:
            vehicle_arcs = [arc for arc in state.arcs if arc.vehicle_id == vehicle_id]
            first = min(vehicle_arcs, key=lambda arc: (arc.departure_layer, arc.arc_id))
            last = max(vehicle_arcs, key=lambda arc: (arc.arrival_layer, arc.arc_id))
            pre_anchor = (first.tail, first.departure_layer)
            post_anchor = (last.head, last.arrival_layer)
            anchors.append((vehicle_id, pre_anchor, post_anchor))
            arc_ids.update(arc.arc_id for arc in vehicle_arcs)
            continue
        pre_candidates = [arc for arc in selected_path if arc.arrival_layer <= lower]
        post_candidates = [arc for arc in selected_path if arc.departure_layer >= upper]
        if pre_candidates:
            pre_arc = max(pre_candidates, key=lambda arc: (arc.arrival_layer, arc.arc_id))
            pre_anchor = (pre_arc.head, pre_arc.arrival_layer)
        else:
            first = min(selected_path, key=lambda arc: (arc.departure_layer, arc.arc_id))
            pre_anchor = (first.tail, first.departure_layer)
        if post_candidates:
            post_arc = min(post_candidates, key=lambda arc: (arc.departure_layer, arc.arc_id))
            post_anchor = (post_arc.tail, post_arc.departure_layer)
        else:
            last = max(selected_path, key=lambda arc: (arc.arrival_layer, arc.arc_id))
            post_anchor = (last.head, last.arrival_layer)
        anchors.append((vehicle_id, pre_anchor, post_anchor))
        arc_ids.update(
            arc.arc_id
            for arc in state.arcs
            if arc.vehicle_id == vehicle_id
            and arc.departure_layer >= pre_anchor[1]
            and arc.arrival_layer <= post_anchor[1]
        )

    span_layers = [
        layer
        for tuple_id in tuple_ids
        for layer in (
            tuple_by_id[tuple_id].launch_layer,
            tuple_by_id[tuple_id].recovery_layer,
        )
    ]
    time_span = max(span_layers) - min(span_layers) if span_layers else 0
    accepted = (
        len(tasks) <= budget.max_tasks
        and len(tuple_ids) <= budget.max_tuples
        and len(vehicle_ids) <= budget.max_vehicles
        and time_span <= budget.max_span
    )
    return RegionClosure(
        accepted,
        "accepted" if accepted else "closure_budget_exceeded",
        tuple(sorted(tasks)),
        tuple(sorted(tuple_ids)),
        tuple(sorted(vehicle_ids)),
        tuple(sorted(uav_ids)),
        tuple(sorted(arc_ids)),
        tuple(anchors),
    )
