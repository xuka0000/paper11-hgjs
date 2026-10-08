from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    first_violation: str | None
    max_residual: float
    object_id: str | None


def check_done_task_reused(schedule, state, context):
    selected_tasks = [task for item in schedule.selected_tuples for task in item.task_ids]
    reused = set(selected_tasks) & set(state.done_tasks)
    return (1.0, min(reused)) if reused else None


def check_airborne_uav_relaunched(schedule, state, context):
    airborne_uavs = {uav_id for uav_id, uav in state.uavs.items() if uav.is_airborne}
    relaunched = airborne_uavs & {
        item.uav_id
        for item in schedule.selected_tuples
        if not getattr(item, "is_air_residual", False)
    }
    return (1.0, min(relaunched)) if relaunched else None


def check_tuple_admission(schedule, state, context):
    if "admitted_tuple_ids" not in context:
        return None
    admitted = set(context["admitted_tuple_ids"])
    rejected = {
        item.tuple_id
        for item in schedule.selected_tuples
        if item.tuple_id not in admitted
    }
    return (1.0, min(rejected)) if rejected else None


def check_coverage(schedule, state, context):
    if not hasattr(schedule, "missed_task_ids"):
        return None
    counts = Counter(task for item in schedule.selected_tuples for task in item.task_ids)
    counts.update(schedule.missed_task_ids)
    tasks = set(state.remaining_tasks) | set(counts)
    residuals = {task: abs(counts[task] - 1) for task in tasks}
    task = max(residuals, key=residuals.get) if residuals else None
    return (residuals[task], task) if task is not None and residuals[task] else None


def check_vehicle_flow(schedule, state, context):
    if "arc_defs" not in context or not hasattr(schedule, "selected_arc_ids"):
        return None
    source = context.get("source_node", "source")
    sink = context.get("sink_node", "sink")
    degree = defaultdict(lambda: [0, 0])
    if "arc_time_defs" in context:
        for arc_id in schedule.selected_arc_ids:
            vehicle_id, tail, departure, head, arrival = context["arc_time_defs"][
                arc_id
            ]
            tail_key = source if tail == source else (tail, departure)
            head_key = sink if head == sink else (head, arrival)
            degree[(vehicle_id, tail_key)][1] += 1
            degree[(vehicle_id, head_key)][0] += 1
    else:
        for arc_id in schedule.selected_arc_ids:
            vehicle_id, tail, head = context["arc_defs"][arc_id]
            degree[(vehicle_id, tail)][1] += 1
            degree[(vehicle_id, head)][0] += 1
    violations = []
    for (vehicle_id, node), (incoming, outgoing) in degree.items():
        if node == source:
            residual = abs(outgoing - incoming - 1)
        elif node == sink:
            residual = abs(incoming - outgoing - 1)
        else:
            residual = abs(incoming - outgoing)
        if residual:
            violations.append((residual, f"{vehicle_id}:{node}"))
    return max(violations) if violations else None


def check_endpoint_occupancy(schedule, state, context):
    if "arc_defs" not in context or not hasattr(schedule, "selected_arc_ids"):
        return None
    if "arc_time_defs" in context:
        incoming = {
            (vehicle_id, head, arrival)
            for arc_id in schedule.selected_arc_ids
            for vehicle_id, _, _, head, arrival in (
                context["arc_time_defs"][arc_id],
            )
        }
        source = context.get("source_node", "source")
        for item in schedule.selected_tuples:
            if (
                not getattr(item, "is_air_residual", False)
                and item.launch_node != source
                and (
                    item.vehicle_id,
                    item.launch_node,
                    getattr(item, "launch_layer", 0),
                )
                not in incoming
            ):
                return 1.0, f"{item.tuple_id}:launch"
            if (
                item.vehicle_id,
                item.recovery_node,
                getattr(item, "recovery_layer", 0),
            ) not in incoming:
                return 1.0, f"{item.tuple_id}:recovery"
        return None
    incoming = {
        (context["arc_defs"][arc_id][0], context["arc_defs"][arc_id][2])
        for arc_id in schedule.selected_arc_ids
    }
    for item in schedule.selected_tuples:
        for role, node in (("launch", item.launch_node), ("recovery", item.recovery_node)):
            if (item.vehicle_id, node) not in incoming:
                return 1.0, f"{item.tuple_id}:{role}"
    return None


def check_required_wait_arcs(schedule, state, context):
    if not hasattr(schedule, "selected_arc_ids"):
        return None
    selected = set(schedule.selected_arc_ids)
    for item in schedule.selected_tuples:
        missing = set(getattr(item, "required_arc_ids", ())) - selected
        if missing:
            return float(len(missing)), min(missing)
    return None


def check_operation_capacity(schedule, state, context):
    capacities = context.get("operation_capacity")
    if capacities is None:
        return None
    usage = Counter(
        key
        for item in schedule.selected_tuples
        for key in getattr(item, "operation_keys", ())
    )
    violations = [
        (usage[key] - capacity, key)
        for key, capacity in capacities.items()
        if usage[key] > capacity
    ]
    return max(violations) if violations else None


def check_uav_intervals(schedule, state, context):
    usage = Counter(
        key
        for item in schedule.selected_tuples
        for key in getattr(item, "interval_keys", ())
    )
    violations = [(count - 1, key) for key, count in usage.items() if count > 1]
    return max(violations) if violations else None


def check_air_residual_unique(schedule, state, context):
    usage = Counter(
        item.uav_id
        for item in schedule.selected_tuples
        if getattr(item, "is_air_residual", False)
    )
    violations = [(count - 1, uav_id) for uav_id, count in usage.items() if count > 1]
    return max(violations) if violations else None


def check_completion_layers(schedule, state, context):
    if not hasattr(schedule, "completion_layers"):
        return None
    expected = {}
    for item in schedule.selected_tuples:
        item_layers = dict(getattr(item, "completion_layers", ()))
        if item_layers:
            expected.update(item_layers)
        else:
            expected.update(
                {task: item.completion_layer for task in item.task_ids}
            )
    keys = set(expected) | set(schedule.completion_layers)
    violations = [
        (
            abs(float(schedule.completion_layers.get(task, 0)) - float(expected.get(task, 0))),
            task,
        )
        for task in keys
        if schedule.completion_layers.get(task) != expected.get(task)
    ]
    return max(violations) if violations else None


def check_makespan(schedule, state, context):
    if not hasattr(schedule, "makespan"):
        return None
    values = list(getattr(schedule, "completion_layers", {}).values())
    values.extend(getattr(schedule, "vehicle_return_layers", {}).values())
    expected = max(values, default=0.0)
    residual = abs(float(schedule.makespan) - float(expected))
    tolerance = float(context.get("tolerance", 1e-8))
    return (residual, "makespan") if residual > tolerance else None


def check_primary_value(schedule, state, context):
    if "miss_penalty" not in context or not hasattr(schedule, "primary_value"):
        return None
    expected = sum(context["miss_penalty"][task] for task in schedule.missed_task_ids)
    residual = abs(float(schedule.primary_value) - float(expected))
    tolerance = float(context.get("tolerance", 1e-8))
    return (residual, "primary") if residual > tolerance else None


def check_secondary_value(schedule, state, context):
    if not hasattr(schedule, "secondary_value"):
        return None
    expected = sum(float(item.secondary_cost) for item in schedule.selected_tuples)
    expected += sum(
        float(context.get("arc_secondary_cost", {}).get(arc_id, 0.0))
        for arc_id in getattr(schedule, "selected_arc_ids", ())
    )
    selected_tuple_ids = {
        item.tuple_id for item in schedule.selected_tuples
    }
    selected_arc_ids = set(getattr(schedule, "selected_arc_ids", ()))
    plan_change = sum(
        abs(int(tuple_id in selected_tuple_ids) - int(value))
        for tuple_id, value in context.get("previous_tuple_values", {}).items()
    )
    plan_change += sum(
        abs(int(arc_id in selected_arc_ids) - int(value))
        for arc_id, value in context.get("previous_arc_values", {}).items()
    )
    expected += float(context.get("plan_change_weight", 0.0)) * plan_change
    expected += float(context.get("makespan_weight", 0.0)) * float(
        getattr(schedule, "makespan", 0.0)
    )
    expected += float(context.get("risk_weight", 0.0)) * sum(
        float(getattr(item, "risk_exposure", 0.0))
        for item in schedule.selected_tuples
    )
    expected += float(context.get("energy_weight", 0.0)) * sum(
        float(getattr(item, "expected_energy", 0.0))
        for item in schedule.selected_tuples
    )
    expected += float(context.get("time_weight", 0.0)) * sum(
        float(context.get("arc_duty_time", {}).get(arc_id, 0.0))
        for arc_id in getattr(schedule, "selected_arc_ids", ())
    )
    residual = abs(float(schedule.secondary_value) - expected)
    tolerance = float(context.get("tolerance", 1e-8))
    return (residual, "secondary") if residual > tolerance else None


CHECKS = (
    ("done_task_reused", check_done_task_reused),
    ("airborne_uav_relaunched", check_airborne_uav_relaunched),
    ("tuple_admission", check_tuple_admission),
    ("coverage", check_coverage),
    ("vehicle_flow", check_vehicle_flow),
    ("endpoint_occupancy", check_endpoint_occupancy),
    ("required_wait_arcs", check_required_wait_arcs),
    ("operation_capacity", check_operation_capacity),
    ("uav_intervals", check_uav_intervals),
    ("air_residual_unique", check_air_residual_unique),
    ("completion_layers", check_completion_layers),
    ("makespan", check_makespan),
    ("primary_value", check_primary_value),
    ("secondary_value", check_secondary_value),
)


def validate_schedule(schedule, state, context=None) -> ValidationReport:
    context = context or {}
    for code, function in CHECKS:
        violation = function(schedule, state, context)
        if violation is not None:
            residual, object_id = violation
            return ValidationReport(False, code, float(residual), str(object_id))
    return ValidationReport(True, None, 0.0, None)
