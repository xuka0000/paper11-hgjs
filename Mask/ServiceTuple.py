from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class ServiceAdmissionChecks:
    task_sequence: bool = True
    carrier_assignment: bool = True
    sensing: bool = True
    path_3d: bool = True
    range_limit: bool = True
    terrain_clearance: bool = True
    airspace: bool = True
    timing: bool = True

    @property
    def all_passed(self) -> bool:
        return all(
            (
                self.task_sequence,
                self.carrier_assignment,
                self.sensing,
                self.path_3d,
                self.range_limit,
                self.terrain_clearance,
                self.airspace,
                self.timing,
            )
        )


@dataclass(frozen=True, order=True)
class SearchArc:
    arc_id: str
    vehicle_id: int
    tail: str
    head: str
    departure_layer: int
    arrival_layer: int
    arc_type: str
    secondary_cost: float = 0.0
    duty_time: float = 0.0


@dataclass(frozen=True)
class SearchTuple:
    tuple_id: str
    vehicle_id: int
    uav_id: int
    task_ids: tuple[int, ...]
    launch_node: str
    recovery_node: str
    launch_layer: int
    recovery_layer: int
    required_arc_ids: tuple[str, ...] = ()
    interval_keys: tuple[str, ...] = ()
    operation_keys: tuple[str, ...] = ()
    completion_layers: tuple[tuple[int, int], ...] = ()
    secondary_cost: float = 0.0
    admitted: bool = True
    risk_exposure: float = 0.0
    expected_energy: float = 0.0
    energy_std: float = 0.0
    reliable_energy: float = 0.0
    battery_capacity: float = 0.0
    reserve_ratio: float = 0.0
    admission_certificate: str = ""
    is_air_residual: bool = False
    admission_checks: ServiceAdmissionChecks = ServiceAdmissionChecks()
    physical_candidate_id: str = ""

    def __post_init__(self):
        if self.admitted and not self.admission_checks.all_passed:
            object.__setattr__(self, "admitted", False)

    @classmethod
    def create(
        cls,
        vehicle_id,
        uav_id,
        task_ids,
        launch_node,
        recovery_node,
        launch_layer,
        recovery_layer,
        **kwargs,
    ):
        payload = (
            vehicle_id,
            uav_id,
            tuple(task_ids),
            launch_node,
            recovery_node,
            launch_layer,
            recovery_layer,
        )
        stable = sha256(repr(payload).encode("utf-8")).hexdigest()[:20]
        return cls(
            f"tuple:{stable}",
            vehicle_id,
            uav_id,
            tuple(task_ids),
            launch_node,
            recovery_node,
            launch_layer,
            recovery_layer,
            **kwargs,
        )


@dataclass(frozen=True)
class SearchState:
    arcs: tuple[SearchArc, ...]
    tuples: tuple[SearchTuple, ...]
    selected_arc_ids: frozenset[str] = frozenset()
    selected_tuple_ids: frozenset[str] = frozenset()

    def __post_init__(self):
        arc_ids = {arc.arc_id for arc in self.arcs}
        tuple_ids = {item.tuple_id for item in self.tuples}
        if self.selected_arc_ids - arc_ids:
            raise ValueError("unknown selected arc")
        if self.selected_tuple_ids - tuple_ids:
            raise ValueError("unknown selected tuple")
