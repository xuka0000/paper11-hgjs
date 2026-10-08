"""Instrument the frozen dynamic engine without changing its decisions or execution."""
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path
from types import ModuleType
import json

from Agent.Tool.Evaluation.dynamic_evaluation import DynamicScenario, DisturbanceEvent
from Env.Execution import PlannedVehicleArcEvent

ROOT = Path(__file__).resolve().parents[2]
LAYERS = tuple(range(1, 122, 8))
TERMINAL = 128


def scenarios():
    # Exogenous, fixed targets and dates. No method trajectory selects an event.
    return (
        DynamicScenario('calm', ()),
        DynamicScenario('late_weather', (DisturbanceEvent('weather', 17, .8),)),
        DynamicScenario('late_road', (DisturbanceEvent('road', 17, 1., 1),)),
        DynamicScenario('late_risk', (DisturbanceEvent('risk', 17, .8, 4),)),
        DynamicScenario('late_delay', (DisturbanceEvent('vehicle_delay', 17, 2., 1),)),
        DynamicScenario('late_energy', (DisturbanceEvent('uav_energy', 17, .6, 1),)),
        DynamicScenario('compound', (
            DisturbanceEvent('weather', 17, .4),
            DisturbanceEvent('vehicle_delay', 17, 1., 1),
            DisturbanceEvent('uav_energy', 33, .2, 1),
            DisturbanceEvent('risk', 33, .4, 4),
        )),
    )


def pack(value):
    if is_dataclass(value):
        return {'type': type(value).__module__ + '.' + type(value).__name__,
                'fields': {f.name: pack(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, dict):
        return {'map': [[pack(k), pack(v)] for k, v in value.items()]}
    if isinstance(value, (set, frozenset)):
        return {'set': sorted([pack(v) for v in value], key=repr)}
    if isinstance(value, tuple):
        return {'tuple': [pack(v) for v in value]}
    if isinstance(value, list):
        return [pack(v) for v in value]
    return value


def unpack(value):
    if isinstance(value, list):
        return [unpack(v) for v in value]
    if not isinstance(value, dict):
        return value
    if 'map' in value:
        return {unpack(k): unpack(v) for k, v in value['map']}
    if 'set' in value:
        return frozenset(unpack(v) for v in value['set'])
    if 'tuple' in value:
        return tuple(unpack(v) for v in value['tuple'])
    if 'type' in value:
        import importlib
        module, name = value['type'].rsplit('.', 1)
        if module not in {'Env.State', 'Env.Runtime', 'Env.Execution', 'Mask.UAV', 'Mask.Vehicle'}:
            raise ValueError('Unregistered trace object: ' + value['type'])
        cls = getattr(importlib.import_module(module), name)
        return cls(**{k: unpack(v) for k, v in value['fields'].items()})
    raise ValueError('Unregistered trace representation')


def future_signatures(entries, state):
    """Decision changes among pending actions; exclude executed and committed arcs."""
    result = set()
    for event in entries:
        if isinstance(event, PlannedVehicleArcEvent):
            if event.arc_id.startswith('anchor:'):
                continue
            if event.arc_id == state.vehicles[event.vehicle_id].active_arc_id:
                continue
            values = (event.arc_id, event.vehicle_id, event.head,
                      event.departure_layer, event.arrival_layer)
        else:
            values = tuple((f.name, getattr(event, f.name)) for f in fields(event)
                           if f.name not in {'expected_energy', 'energy_std',
                                             'battery_capacity', 'reserve_ratio'})
        result.add(repr((type(event).__name__, values)))
    return result


def run_traced(instance, scenario, policy, method, config):
    # A private module namespace lets observers wrap calls without monkeypatching
    # the production module or changing solver inputs.
    path = ROOT / 'Agent/Tool/Evaluation/dynamic_evaluation.py'
    engine = ModuleType('Agent.Tool.Evaluation.dynamic_evaluation')
    exec(compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec'), engine.__dict__)
    original_materialize = engine.materialize_dynamic_problem
    original_run = engine.run_method
    original_execute = engine.execute_due_events
    original_advance = engine.advance_state
    records = []
    current = {}
    pending_before = ()

    def materialize(instance, state, active):
        problem = original_materialize(instance, state, active)
        current.update(state_for_plan=pack(state), revealed_events=[vars(x) for x in active],
                       problem_status=problem.safety_status, problem_reason=problem.safety_reason)
        return problem

    def solve(*args, **kwargs):
        execution = original_run(*args, **kwargs)
        result = execution.result
        current.update(selected_arc_ids=sorted(result.selected_arc_ids),
                       selected_tuple_ids=sorted(result.selected_tuple_ids),
                       returned_primary=result.best_primary, returned_secondary=result.best_secondary,
                       config=asdict(execution.config))
        return execution

    def execute(state, entries, end_layer, **kwargs):
        nonlocal pending_before
        # Earlier execution was already removed from pending_before.
        changes = len(future_signatures(pending_before, state) ^ future_signatures(entries, state))
        executed, pending = original_execute(state, entries, end_layer, **kwargs)
        current.update(layer=state.current_layer, end_layer=end_layer,
                       state_before_execution=pack(state), entries=pack(entries),
                       pending_before=pack(pending_before), pending_after=pack(pending),
                       future_changes=changes if records else 0,
                       blocked_vehicle_ids=sorted(kwargs.get('blocked_vehicle_ids', ())),
                       execution=pack(executed))
        pending_before = pending
        return executed, pending

    def advance(state, executed):
        updated = original_advance(state, executed)
        current['state_after'] = pack(updated)
        records.append(dict(current))
        current.clear()
        return updated

    engine.materialize_dynamic_problem = materialize
    engine.run_method = solve
    engine.execute_due_events = execute
    engine.advance_state = advance
    result = engine.run_dynamic_trajectory(instance, scenario, policy, method, config,
                                          decision_layers=LAYERS, terminal_layer=TERMINAL)
    return result, records
