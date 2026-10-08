from dataclasses import dataclass


@dataclass(frozen=True)
class RoadArc:
    tail: int
    head: int
    length: float
    nominal_speed: float
    structure_factor: float
    available: bool
