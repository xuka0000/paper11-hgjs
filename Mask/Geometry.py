from dataclasses import dataclass
from math import acos, atan2, hypot, sqrt


@dataclass(frozen=True, order=True)
class Point3D:
    x: float
    y: float
    z: float = 0.0


def _distance_2d(left: Point3D, right: Point3D) -> float:
    return hypot(right.x - left.x, right.y - left.y)


def _distance_3d(left: Point3D, right: Point3D) -> float:
    return sqrt(
        (right.x - left.x) ** 2
        + (right.y - left.y) ** 2
        + (right.z - left.z) ** 2
    )


def _point_segment_distance_2d(
    point: Point3D,
    start: Point3D,
    end: Point3D,
) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    denominator = dx * dx + dy * dy
    projection = (
        (point.x - start.x) * dx + (point.y - start.y) * dy
    ) / denominator
    projection = max(0.0, min(1.0, projection))
    nearest_x = start.x + projection * dx
    nearest_y = start.y + projection * dy
    return hypot(point.x - nearest_x, point.y - nearest_y)


@dataclass(frozen=True)
class Polyline3D:
    points: tuple[Point3D, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))
        if len(self.points) < 2:
            raise ValueError("polyline requires at least two points")
        if any(
            _distance_2d(left, right) == 0.0
            for left, right in zip(self.points, self.points[1:])
        ):
            raise ValueError("polyline contains a zero-length segment")

    @property
    def segments(self) -> tuple[tuple[Point3D, Point3D], ...]:
        return tuple(zip(self.points, self.points[1:]))

    @property
    def length_2d(self) -> float:
        return sum(_distance_2d(left, right) for left, right in self.segments)

    @property
    def length_3d(self) -> float:
        return sum(_distance_3d(left, right) for left, right in self.segments)

    @property
    def chord_length_3d(self) -> float:
        return _distance_3d(self.points[0], self.points[-1])

    @property
    def detour_factor(self) -> float:
        chord = self.chord_length_3d
        return self.length_3d / chord if chord > 0.0 else float("inf")

    @property
    def cumulative_turn_radians(self) -> float:
        total = 0.0
        for first, middle, last in zip(
            self.points,
            self.points[1:],
            self.points[2:],
        ):
            incoming = (middle.x - first.x, middle.y - first.y)
            outgoing = (last.x - middle.x, last.y - middle.y)
            denominator = hypot(*incoming) * hypot(*outgoing)
            cosine = (
                incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
            ) / denominator
            total += acos(max(-1.0, min(1.0, cosine)))
        return total

    @property
    def ascent(self) -> float:
        return sum(max(0.0, right.z - left.z) for left, right in self.segments)

    @property
    def descent(self) -> float:
        return sum(max(0.0, left.z - right.z) for left, right in self.segments)

    def nearest_segment_index(self, point: Point3D) -> int:
        distances = tuple(
            _point_segment_distance_2d(point, start, end)
            for start, end in self.segments
        )
        return min(range(len(distances)), key=distances.__getitem__)

    def distance_to_point_2d(self, point: Point3D) -> float:
        index = self.nearest_segment_index(point)
        start, end = self.segments[index]
        return _point_segment_distance_2d(point, start, end)

    def tangent_angle_at_nearest_segment(self, point: Point3D) -> float:
        start, end = self.segments[self.nearest_segment_index(point)]
        return atan2(end.y - start.y, end.x - start.x)


def _orientation(first: Point3D, second: Point3D, third: Point3D) -> float:
    return (second.x - first.x) * (third.y - first.y) - (
        second.y - first.y
    ) * (third.x - first.x)


def _on_segment(first: Point3D, point: Point3D, second: Point3D) -> bool:
    tolerance = 1e-12
    return (
        min(first.x, second.x) - tolerance
        <= point.x
        <= max(first.x, second.x) + tolerance
        and min(first.y, second.y) - tolerance
        <= point.y
        <= max(first.y, second.y) + tolerance
    )


def segments_intersect_2d(
    first_start: Point3D,
    first_end: Point3D,
    second_start: Point3D,
    second_end: Point3D,
) -> bool:
    tolerance = 1e-12
    orientations = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    if orientations[0] * orientations[1] < 0.0 and orientations[2] * orientations[3] < 0.0:
        return True
    collinear_points = (
        (orientations[0], first_start, second_start, first_end),
        (orientations[1], first_start, second_end, first_end),
        (orientations[2], second_start, first_start, second_end),
        (orientations[3], second_start, first_end, second_end),
    )
    return any(
        abs(orientation) <= tolerance and _on_segment(start, point, end)
        for orientation, start, point, end in collinear_points
    )


__all__ = ["Point3D", "Polyline3D", "segments_intersect_2d"]
