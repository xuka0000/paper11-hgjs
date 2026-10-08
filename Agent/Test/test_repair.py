from dataclasses import replace
from types import SimpleNamespace

from Agent.Proposed.closure import ClosureBudget, close_region
from Agent.Proposed.regions import CoreRegion
from Agent.Proposed.repair import repair_region
from Mask.ServiceTuple import SearchArc, SearchState, SearchTuple


def test_repair_keeps_current_solution_when_no_better_candidate(search_state, base_state):
    core = CoreRegion("region:test", "task:2", ("task:2",), 1, 0, 0, 0)
    closure = close_region(search_state, core, ClosureBudget(4, 4, 2, 8))
    validation_state = replace(
        base_state,
        remaining_tasks=frozenset({2, 3}),
        locked_tuple_ids=frozenset(),
    )
    result = repair_region(
        state=search_state,
        closure=closure,
        task_ids=(2, 3),
        miss_penalty={2: 10.0, 3: 10.0},
        validation_state=validation_state,
        operation_capacity={"op:a:2": 1, "op:b:4": 1},
        time_limit_s=5.0,
    )
    assert result.accepted
    assert result.validation.ok
    assert frozenset(result.selected_tuple_ids) == search_state.selected_tuple_ids
    assert result.runtime_s >= 0.0
    assert result.primary_bound <= result.primary_value + 1e-8
    assert result.primary_proven_optimal is True


def test_repair_rejects_budget_failed_closure(search_state, base_state):
    core = CoreRegion("region:test", "task:2", ("task:2",), 1, 0, 0, 0)
    closure = close_region(search_state, core, ClosureBudget(0, 0, 0, 0))
    result = repair_region(
        state=search_state,
        closure=closure,
        task_ids=(2, 3),
        miss_penalty={2: 10.0, 3: 10.0},
        validation_state=base_state,
        operation_capacity={},
        time_limit_s=5.0,
    )
    assert not result.accepted
    assert result.reason == "closure_budget_exceeded"


def test_repair_uses_makespan_in_lexicographic_tie(base_state):
    arcs = (
        SearchArc("s-a", 1, "source", "a", 1, 2, "road"),
        SearchArc("a-b", 1, "a", "b", 2, 3, "wait"),
        SearchArc("b-t", 1, "b", "sink", 3, 4, "road"),
    )
    tuples = (
        SearchTuple(
            "early",
            1,
            1,
            (1,),
            "a",
            "b",
            2,
            3,
            completion_layers=((1, 3),),
        ),
        SearchTuple(
            "late",
            1,
            1,
            (1,),
            "a",
            "b",
            2,
            3,
            completion_layers=((1, 6),),
        ),
    )
    state = SearchState(
        arcs,
        tuples,
        frozenset(arc.arc_id for arc in arcs),
        frozenset({"late"}),
    )
    closure = SimpleNamespace(
        accepted=True,
        reason="closed",
        tuple_ids=("early", "late"),
        arc_ids=tuple(arc.arc_id for arc in arcs),
    )
    validation_state = replace(
        base_state,
        done_tasks=frozenset(),
        remaining_tasks=frozenset({1}),
        locked_tuple_ids=frozenset(),
    )

    result = repair_region(
        state=state,
        closure=closure,
        task_ids=(1,),
        miss_penalty={1: 10.0},
        validation_state=validation_state,
        operation_capacity={},
        time_limit_s=5.0,
        makespan_weight=1.0,
    )

    assert result.accepted
    assert result.selected_tuple_ids == ("early",)
    assert result.secondary_value == 4.0


def test_repair_passes_explicit_objective_components(base_state):
    arcs = (
        SearchArc("s-a", 1, "source", "a", 1, 2, "road", duty_time=1.0),
        SearchArc("a-b", 1, "a", "b", 2, 3, "wait", duty_time=1.0),
        SearchArc("b-t", 1, "b", "sink", 3, 4, "road", duty_time=1.0),
    )
    tuples = (
        SearchTuple(
            "low-risk",
            1,
            1,
            (1,),
            "a",
            "b",
            2,
            3,
            completion_layers=((1, 3),),
            risk_exposure=1.0,
            expected_energy=8.0,
        ),
        SearchTuple(
            "low-energy",
            1,
            1,
            (1,),
            "a",
            "b",
            2,
            3,
            completion_layers=((1, 3),),
            risk_exposure=4.0,
            expected_energy=1.0,
        ),
    )
    state = SearchState(
        arcs,
        tuples,
        frozenset(arc.arc_id for arc in arcs),
        frozenset({"low-energy"}),
    )
    closure = SimpleNamespace(
        accepted=True,
        reason="closed",
        tuple_ids=("low-risk", "low-energy"),
        arc_ids=tuple(arc.arc_id for arc in arcs),
    )
    validation_state = replace(
        base_state,
        done_tasks=frozenset(),
        remaining_tasks=frozenset({1}),
        locked_tuple_ids=frozenset(),
    )

    result = repair_region(
        state=state,
        closure=closure,
        task_ids=(1,),
        miss_penalty={1: 10.0},
        validation_state=validation_state,
        operation_capacity={},
        time_limit_s=5.0,
        time_weight=0.5,
        risk_weight=2.0,
        energy_weight=0.1,
    )

    assert result.accepted
    assert result.selected_tuple_ids == ("low-risk",)
    assert result.secondary_value == 4.3


def test_repair_transfers_required_vehicle_return(base_state):
    arcs = (
        SearchArc("s-a", 1, "source", "a", 1, 2, "road", duty_time=1.0),
        SearchArc("a-t", 1, "a", "sink", 2, 3, "road", duty_time=1.0),
    )
    state = SearchState(arcs, (), frozenset(), frozenset())
    closure = SimpleNamespace(
        accepted=True,
        reason="closed",
        tuple_ids=(),
        arc_ids=tuple(arc.arc_id for arc in arcs),
    )
    vehicle = replace(base_state.vehicles[1], must_return=True)
    validation_state = replace(
        base_state,
        done_tasks=frozenset(),
        remaining_tasks=frozenset({1}),
        vehicles={1: vehicle},
        locked_tuple_ids=frozenset(),
    )

    result = repair_region(
        state=state,
        closure=closure,
        task_ids=(1,),
        miss_penalty={1: 10.0},
        validation_state=validation_state,
        operation_capacity={},
        time_limit_s=5.0,
    )

    assert result.accepted
    assert result.selected_tuple_ids == ()
    assert result.selected_arc_ids == ("a-t", "s-a")


def test_repair_transfers_remaining_vehicle_duty_limit(base_state):
    arcs = (
        SearchArc("s-a", 1, "source", "a", 1, 2, "road", duty_time=1.0),
        SearchArc("a-b", 1, "a", "b", 2, 3, "wait", duty_time=2.0),
        SearchArc("b-t", 1, "b", "sink", 3, 4, "road", duty_time=1.0),
    )
    item = SearchTuple(
        "serve",
        1,
        1,
        (1,),
        "a",
        "b",
        2,
        3,
        completion_layers=((1, 3),),
    )
    state = SearchState(arcs, (item,), frozenset(), frozenset())
    closure = SimpleNamespace(
        accepted=True,
        reason="closed",
        tuple_ids=(item.tuple_id,),
        arc_ids=tuple(arc.arc_id for arc in arcs),
    )
    vehicle = replace(base_state.vehicles[1], remaining_duty_time=3.0)
    validation_state = replace(
        base_state,
        done_tasks=frozenset(),
        remaining_tasks=frozenset({1}),
        vehicles={1: vehicle},
        locked_tuple_ids=frozenset(),
    )

    result = repair_region(
        state=state,
        closure=closure,
        task_ids=(1,),
        miss_penalty={1: 10.0},
        validation_state=validation_state,
        operation_capacity={},
        time_limit_s=5.0,
    )

    assert result.accepted
    assert result.selected_tuple_ids == ()
    assert result.selected_arc_ids == ()


def _released_two_tuple_case(selected_tuple_id):
    arcs = (
        SearchArc("s-a", 1, "source", "a", 1, 2, "road"),
        SearchArc("a-b", 1, "a", "b", 2, 3, "wait"),
        SearchArc("b-t", 1, "b", "sink", 3, 4, "road"),
    )
    tuples = (
        SearchTuple(
            "locked",
            1,
            1,
            (1,),
            "a",
            "b",
            2,
            3,
            completion_layers=((1, 3),),
            secondary_cost=5.0,
        ),
        SearchTuple(
            "alternative",
            1,
            1,
            (1,),
            "a",
            "b",
            2,
            3,
            completion_layers=((1, 3),),
            secondary_cost=0.0,
        ),
    )
    state = SearchState(
        arcs,
        tuples,
        frozenset(arc.arc_id for arc in arcs),
        frozenset({selected_tuple_id}),
    )
    closure = SimpleNamespace(
        accepted=True,
        reason="closed",
        tuple_ids=("locked", "alternative"),
        arc_ids=tuple(arc.arc_id for arc in arcs),
    )
    return state, closure


def test_repair_does_not_release_safe_locked_tuple(base_state):
    state, closure = _released_two_tuple_case("locked")
    validation_state = replace(
        base_state,
        done_tasks=frozenset(),
        remaining_tasks=frozenset({1}),
        locked_tuple_ids=frozenset({"locked"}),
    )

    result = repair_region(
        state=state,
        closure=closure,
        task_ids=(1,),
        miss_penalty={1: 10.0},
        validation_state=validation_state,
        operation_capacity={},
        time_limit_s=5.0,
    )

    assert result.selected_tuple_ids == ("locked",)


def test_repair_releases_locked_tuple_only_when_declared_unsafe(base_state):
    state, closure = _released_two_tuple_case("locked")
    validation_state = replace(
        base_state,
        done_tasks=frozenset(),
        remaining_tasks=frozenset({1}),
        locked_tuple_ids=frozenset({"locked"}),
    )

    result = repair_region(
        state=state,
        closure=closure,
        task_ids=(1,),
        miss_penalty={1: 10.0},
        validation_state=validation_state,
        operation_capacity={},
        time_limit_s=5.0,
        unsafe_tuple_ids=frozenset({"locked"}),
    )

    assert result.selected_tuple_ids == ("alternative",)


def test_repair_uses_previous_plan_to_break_tie(base_state):
    state, closure = _released_two_tuple_case("alternative")
    tied_tuples = tuple(
        replace(item, secondary_cost=0.0) for item in state.tuples
    )
    state = replace(state, tuples=tied_tuples)
    validation_state = replace(
        base_state,
        done_tasks=frozenset(),
        remaining_tasks=frozenset({1}),
        locked_tuple_ids=frozenset(),
        previous_tuple_values={"locked": 1, "alternative": 0},
    )

    result = repair_region(
        state=state,
        closure=closure,
        task_ids=(1,),
        miss_penalty={1: 10.0},
        validation_state=validation_state,
        operation_capacity={},
        time_limit_s=5.0,
        plan_change_weight=5.0,
    )

    assert result.selected_tuple_ids == ("locked",)


def test_repair_instantiates_only_released_boundary_and_audit_decisions(
    base_state,
    monkeypatch,
):
    from Agent.Proposed import repair

    arcs = (
        SearchArc("s-a", 1, "source", "a", 1, 2, "road"),
        SearchArc("a-b", 1, "a", "b", 2, 3, "wait"),
        SearchArc("b-t", 1, "b", "sink", 3, 4, "road"),
    )
    selected = SearchTuple(
        "selected-boundary",
        1,
        1,
        (1,),
        "a",
        "b",
        2,
        3,
        required_arc_ids=("a-b",),
        completion_layers=((1, 3),),
    )
    released = replace(
        selected,
        tuple_id="released-candidate",
        task_ids=(2,),
        completion_layers=((2, 3),),
    )
    inactive = replace(
        selected,
        tuple_id="inactive-outside",
        task_ids=(3,),
        completion_layers=((3, 3),),
    )
    state = SearchState(
        arcs,
        (selected, released, inactive),
        frozenset(arc.arc_id for arc in arcs),
        frozenset({selected.tuple_id}),
    )
    closure = SimpleNamespace(
        accepted=True,
        reason="closed",
        tuple_ids=(released.tuple_id,),
        arc_ids=tuple(arc.arc_id for arc in arcs),
    )
    validation_state = replace(
        base_state,
        done_tasks=frozenset(),
        remaining_tasks=frozenset({1, 2, 3}),
        locked_tuple_ids=frozenset(),
    )
    captured = {}
    original = repair.solve_lexicographic

    def recording_solver(data, time_limit_s, epsilon_u, **kwargs):
        captured["data"] = data
        return original(data, time_limit_s, epsilon_u, **kwargs)

    monkeypatch.setattr(repair, "solve_lexicographic", recording_solver)
    result = repair_region(
        state=state,
        closure=closure,
        task_ids=(1, 2, 3),
        miss_penalty={1: 10.0, 2: 10.0, 3: 10.0},
        validation_state=validation_state,
        operation_capacity={},
        time_limit_s=5.0,
    )

    assert result.validation.ok
    instantiated = {item.tuple_id for item in captured["data"].tuples}
    assert instantiated == {selected.tuple_id, released.tuple_id}
    assert inactive.tuple_id not in instantiated
