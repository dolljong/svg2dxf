from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

Point: TypeAlias = tuple[float, float]


DashPattern = tuple[float, ...] | None


@dataclass(slots=True)
class LineEntity:
    start: Point
    end: Point
    layer: str = "0"
    dash_pattern: DashPattern = None


@dataclass(slots=True)
class PolylineEntity:
    points: list[Point]
    closed: bool = False
    layer: str = "0"
    dash_pattern: DashPattern = None


@dataclass(slots=True)
class CircleEntity:
    center: Point
    radius: float
    layer: str = "0"
    dash_pattern: DashPattern = None


@dataclass(slots=True)
class ArcEntity:
    center: Point
    radius: float
    start_angle: float  # degrees, CCW from positive X
    end_angle: float    # degrees, CCW from positive X
    layer: str = "0"
    dash_pattern: DashPattern = None


@dataclass(slots=True)
class EllipseEntity:
    center: Point
    major_axis: Point
    ratio: float
    start_param: float = 0.0
    end_param: float = math.tau
    layer: str = "0"
    dash_pattern: DashPattern = None


@dataclass(slots=True)
class TextEntity:
    text: str
    insert: Point
    height: float
    rotation: float = 0.0
    layer: str = "0"


Entity: TypeAlias = LineEntity | PolylineEntity | CircleEntity | ArcEntity | EllipseEntity | TextEntity
