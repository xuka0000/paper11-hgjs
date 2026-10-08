import pytest

from Env.State import RollingState
from Mask.ServiceTuple import SearchArc, SearchState, SearchTuple
from Mask.UAV import UavState
from Mask.Vehicle import VehicleState


@pytest.fixture
def base_state():
    return RollingState(
        epoch=0,
        current_layer=2,
        done_tasks=frozenset(),
        remaining_tasks=frozenset({1, 2}),
        vehicles={1: VehicleState(1, 1, 2)},
        uavs={1: UavState(1, 1, 2)},
        locked_tuple_ids=frozenset({"tuple:locked"}),
    )


@pytest.fixture
def in_transit_state(base_state):
    vehicle = VehicleState(1, 1, 2, active_arc=(1, 2), active_arc_arrival_layer=5)
    return RollingState(**{**base_state.as_kwargs(), "vehicles": {1: vehicle}})


@pytest.fixture
def airborne_uav():
    return UavState(1, 1, 2, is_airborne=True, remaining_energy=60.0, remaining_tasks=(2,))


@pytest.fixture
def airborne_state(base_state, airborne_uav):
    return RollingState(**{**base_state.as_kwargs(), "uavs": {1: airborne_uav}, "airborne_tasks": {1: (2,)}})


@pytest.fixture
def search_state():
    arcs = (
        SearchArc("s-a", 1, "source", "a", 1, 2, "road"),
        SearchArc("a-b", 1, "a", "b", 2, 4, "wait"),
        SearchArc("b-t", 1, "b", "sink", 4, 5, "road"),
    )
    item = SearchTuple.create(
        1,
        1,
        (2, 3),
        "a",
        "b",
        2,
        4,
        required_arc_ids=("a-b",),
        interval_keys=("uav:1:2", "uav:1:3"),
        operation_keys=("op:a:2", "op:b:4"),
        completion_layers=((2, 3), (3, 4)),
        secondary_cost=2.0,
    )
    return SearchState(
        arcs=arcs,
        tuples=(item,),
        selected_arc_ids=frozenset(arc.arc_id for arc in arcs),
        selected_tuple_ids=frozenset({item.tuple_id}),
    )
