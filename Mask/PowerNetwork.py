from dataclasses import dataclass

from Mask.Geometry import Point3D, Polyline3D


@dataclass(frozen=True)
class Tower:
    tower_id: str
    location: Point3D
    risk: float
    asset_value: float
    topology_importance: float
    service_minutes: float

    def __post_init__(self) -> None:
        if not self.tower_id:
            raise ValueError("tower_id must not be empty")
        if min(self.risk, self.asset_value, self.topology_importance) < 0.0:
            raise ValueError("tower attributes must be nonnegative")
        if self.service_minutes <= 0.0:
            raise ValueError("service_minutes must be positive")


@dataclass(frozen=True)
class PowerLine:
    line_id: str
    tower_ids: tuple[str, ...]
    geometry: Polyline3D

    def __post_init__(self) -> None:
        object.__setattr__(self, "tower_ids", tuple(self.tower_ids))
        if not self.line_id:
            raise ValueError("line_id must not be empty")
        if len(self.tower_ids) < 2:
            raise ValueError("power line requires at least two towers")
        if len(self.tower_ids) != len(set(self.tower_ids)):
            raise ValueError("power line tower_ids must be unique")


@dataclass(frozen=True)
class PowerNetwork:
    towers: tuple[Tower, ...]
    lines: tuple[PowerLine, ...]

    def __post_init__(self) -> None:
        towers = tuple(sorted(self.towers, key=lambda item: item.tower_id))
        lines = tuple(sorted(self.lines, key=lambda item: item.line_id))
        object.__setattr__(self, "towers", towers)
        object.__setattr__(self, "lines", lines)
        tower_by_id = _unique_by_id(towers, "tower_id", "tower")
        _unique_by_id(lines, "line_id", "power line")
        for line in lines:
            unknown = set(line.tower_ids) - set(tower_by_id)
            if unknown:
                raise ValueError(
                    f"power line references unknown tower: {sorted(unknown)}"
                )
            if line.geometry.points[0] != tower_by_id[line.tower_ids[0]].location:
                raise ValueError("power line start does not match its first tower")
            if line.geometry.points[-1] != tower_by_id[line.tower_ids[-1]].location:
                raise ValueError("power line end does not match its last tower")


def _unique_by_id(items, attribute: str, label: str) -> dict:
    output = {}
    for item in items:
        item_id = getattr(item, attribute)
        if item_id in output:
            raise ValueError(f"duplicate {label} id: {item_id}")
        output[item_id] = item
    return output


__all__ = ["PowerLine", "PowerNetwork", "Tower"]
