from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class HyperNode:
    node_id: str
    node_type: str
    entity_id: str = ""
    vehicle_id: int | None = None
    time_layer: int | None = None
    priority: float = 0.0
    selected: float = 0.0


@dataclass(frozen=True)
class Incidence:
    node_id: str
    role: str


@dataclass(frozen=True)
class Hyperedge:
    edge_id: str
    edge_type: str
    incidences: tuple[Incidence, ...]


@dataclass(frozen=True)
class ConstraintHypergraph:
    nodes: tuple[HyperNode, ...]
    edges: tuple[Hyperedge, ...]

    def adjacency(self):
        result = {node.node_id: set() for node in self.nodes}
        for edge in self.edges:
            members = {incidence.node_id for incidence in edge.incidences}
            for member in members:
                result[member].update(members)
                result[member].discard(member)
        return {key: tuple(sorted(value)) for key, value in result.items()}


def _time_node_id(vehicle_id: int, node: str, layer: int) -> str:
    return f"time:{vehicle_id}:{node}:{layer}"


def build_constraint_hypergraph(state, task_ids=(), task_weights=None) -> ConstraintHypergraph:
    nodes: dict[str, HyperNode] = {}
    edges: list[Hyperedge] = []
    task_weights = task_weights or {}
    selected_items = {
        item.tuple_id for item in state.tuples if item.tuple_id in state.selected_tuple_ids
    }
    covered_tasks = {
        task
        for item in state.tuples
        if item.tuple_id in selected_items
        for task in item.task_ids
    }

    def add_node(node: HyperNode) -> None:
        previous = nodes.get(node.node_id)
        if previous is None:
            nodes[node.node_id] = node
        elif node.priority > previous.priority or node.selected > previous.selected:
            nodes[node.node_id] = HyperNode(
                previous.node_id,
                previous.node_type,
                previous.entity_id,
                previous.vehicle_id,
                previous.time_layer,
                max(previous.priority, node.priority),
                max(previous.selected, node.selected),
            )

    for task_id in sorted(set(task_ids)):
        add_node(
            HyperNode(
                f"task:{task_id}",
                "task",
                str(task_id),
                priority=float(task_weights.get(task_id, 0.0)),
                selected=float(task_id in covered_tasks),
            )
        )

    vehicle_ids = {arc.vehicle_id for arc in state.arcs} | {
        item.vehicle_id for item in state.tuples
    }
    for vehicle_id in sorted(vehicle_ids):
        add_node(
            HyperNode(
                f"vehicle:{vehicle_id}",
                "vehicle",
                str(vehicle_id),
                vehicle_id,
                selected=float(
                    any(
                        item.vehicle_id == vehicle_id and item.tuple_id in selected_items
                        for item in state.tuples
                    )
                ),
            )
        )

    sorted_arcs = tuple(sorted(state.arcs))
    flow_incidences: dict[str, list[Incidence]] = {}
    for arc in sorted_arcs:
        arrival_id = _time_node_id(arc.vehicle_id, arc.head, arc.arrival_layer)
        departure_id = _time_node_id(arc.vehicle_id, arc.tail, arc.departure_layer)
        # Preserve the original sorted-arc order, including incoming before
        # outgoing if both endpoints coincide. Index once instead of sorting
        # and scanning the complete arc set separately for every time node.
        flow_incidences.setdefault(arrival_id, []).append(
            Incidence(f"arc:{arc.arc_id}", "incoming")
        )
        flow_incidences.setdefault(departure_id, []).append(
            Incidence(f"arc:{arc.arc_id}", "outgoing")
        )
        add_node(
            HyperNode(
                f"arc:{arc.arc_id}",
                "arc",
                arc.arc_id,
                arc.vehicle_id,
                arc.departure_layer,
                selected=float(arc.arc_id in state.selected_arc_ids),
            )
        )
        add_node(
            HyperNode(
                _time_node_id(arc.vehicle_id, arc.tail, arc.departure_layer),
                "time_node",
                arc.tail,
                arc.vehicle_id,
                arc.departure_layer,
            )
        )
        add_node(
            HyperNode(
                _time_node_id(arc.vehicle_id, arc.head, arc.arrival_layer),
                "time_node",
                arc.head,
                arc.vehicle_id,
                arc.arrival_layer,
            )
        )

    admitted_tuples = tuple(sorted((item for item in state.tuples if item.admitted), key=lambda item: item.tuple_id))
    for item in admitted_tuples:
        tuple_node = item.tuple_id
        uav_node = f"uav:{item.uav_id}"
        is_selected = float(item.tuple_id in selected_items)
        add_node(
            HyperNode(
                tuple_node,
                "tuple",
                item.tuple_id,
                item.vehicle_id,
                item.launch_layer,
                priority=sum(float(task_weights.get(task, 0.0)) for task in item.task_ids),
                selected=is_selected,
            )
        )
        add_node(
            HyperNode(
                uav_node,
                "uav",
                str(item.uav_id),
                item.vehicle_id,
                selected=is_selected,
            )
        )
        launch_node = _time_node_id(item.vehicle_id, item.launch_node, item.launch_layer)
        recovery_node = _time_node_id(item.vehicle_id, item.recovery_node, item.recovery_layer)
        add_node(HyperNode(launch_node, "time_node", item.launch_node, item.vehicle_id, item.launch_layer))
        add_node(HyperNode(recovery_node, "time_node", item.recovery_node, item.vehicle_id, item.recovery_layer))
        role_incidences = [
            Incidence(tuple_node, "anchor"),
            Incidence(uav_node, "platform"),
            Incidence(launch_node, "prepare"),
            Incidence(launch_node, "takeoff"),
        ]
        role_incidences.extend(
            Incidence(f"task:{task_id}", f"visit_{position}")
            for position, task_id in enumerate(item.task_ids, start=1)
        )
        role_incidences.append(Incidence(recovery_node, "recover"))
        edges.append(Hyperedge(f"role:{item.tuple_id}", "tuple_role", tuple(role_incidences)))
        edges.append(
            Hyperedge(
                f"launch:{item.tuple_id}",
                "launch",
                (Incidence(tuple_node, "tuple"), Incidence(launch_node, "endpoint")),
            )
        )
        edges.append(
            Hyperedge(
                f"recover:{item.tuple_id}",
                "recover",
                (Incidence(tuple_node, "tuple"), Incidence(recovery_node, "endpoint")),
            )
        )
        for key in item.interval_keys:
            resource_node = f"interval:{key}"
            if resource_node not in nodes:
                add_node(HyperNode(resource_node, "interval_resource", key, item.vehicle_id))
            edges.append(
                Hyperedge(
                    f"interval:{key}:{item.tuple_id}",
                    "uav_interval",
                    (Incidence(resource_node, "resource"), Incidence(tuple_node, "consumer")),
                )
            )
        for key in item.operation_keys:
            resource_node = f"operation:{key}"
            if resource_node not in nodes:
                add_node(HyperNode(resource_node, "operation_resource", key, item.vehicle_id))
            edges.append(
                Hyperedge(
                    f"operation:{key}:{item.tuple_id}",
                    "operation_capacity",
                    (Incidence(resource_node, "resource"), Incidence(tuple_node, "consumer")),
                )
            )
        for arc_id in item.required_arc_ids:
            edges.append(
                Hyperedge(
                    f"wait:{arc_id}:{item.tuple_id}",
                    "wait_dependency",
                    (Incidence(f"arc:{arc_id}", "wait_arc"), Incidence(tuple_node, "tuple")),
                )
            )

    for task_id in sorted(set(task_ids)):
        incidences = [Incidence(f"task:{task_id}", "task")]
        incidences.extend(
            Incidence(item.tuple_id, "cover")
            for item in admitted_tuples
            if task_id in item.task_ids
        )
        edges.append(Hyperedge(f"coverage:{task_id}", "coverage", tuple(incidences)))

    for node in sorted((item for item in nodes.values() if item.node_type == "time_node")):
        incidences = [Incidence(node.node_id, "node")]
        incidences.extend(flow_incidences.get(node.node_id, ()))
        edges.append(Hyperedge(f"flow:{node.node_id}", "vehicle_flow", tuple(incidences)))

    for vehicle_id in sorted(vehicle_ids):
        incidences = [Incidence(f"vehicle:{vehicle_id}", "vehicle")]
        incidences.extend(
            Incidence(item.tuple_id, "service")
            for item in admitted_tuples
            if item.vehicle_id == vehicle_id
        )
        edges.append(Hyperedge(f"use:{vehicle_id}", "vehicle_use", tuple(incidences)))

    return ConstraintHypergraph(
        tuple(sorted(nodes.values())),
        tuple(sorted(edges, key=lambda edge: edge.edge_id)),
    )
