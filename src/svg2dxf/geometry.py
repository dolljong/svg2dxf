from __future__ import annotations

import math
import re
from typing import Iterable

from .models import ArcEntity, CircleEntity, EllipseEntity, Entity, LineEntity, Point, PolylineEntity, TextEntity

_LENGTH_RE = re.compile(
    r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-zA-Z%]*)\s*$"
)


def parse_length(value: str | None, default: float = 0.0) -> float:
    """Parse SVG length into px."""
    if value is None:
        return default
    match = _LENGTH_RE.match(value)
    if not match:
        return default
    amount = float(match.group(1))
    unit = match.group(2).lower()

    factors = {
        "": 1.0,
        "px": 1.0,
        "in": 96.0,
        "cm": 96.0 / 2.54,
        "mm": 96.0 / 25.4,
        "pt": 96.0 / 72.0,
        "pc": 16.0,
    }
    factor = factors.get(unit)
    if factor is None:
        return amount
    return amount * factor


def output_unit_scale(unit: str, pixel_size_mm: float) -> float:
    unit_norm = unit.lower()
    if unit_norm == "mm":
        return pixel_size_mm
    if unit_norm == "inch":
        return pixel_size_mm / 25.4
    if unit_norm == "px":
        return 1.0
    raise ValueError(f"Unsupported output unit: {unit}")


def map_point(point: Point, scale: float, y_flip_ref: float | None) -> Point:
    x, y = point
    if y_flip_ref is None:
        return (x * scale, y * scale)
    return (x * scale, (y_flip_ref - y) * scale)


def map_vector(vector: Point, scale: float, y_flipped: bool) -> Point:
    x, y = vector
    if y_flipped:
        return (x * scale, -y * scale)
    return (x * scale, y * scale)


def map_length(length: float, scale: float) -> float:
    return abs(length) * scale


def _minor_axis_from_major(major: Point, ratio: float) -> Point:
    return (-major[1] * ratio, major[0] * ratio)


def compute_y_flip_ref(entities: Iterable[Entity]) -> float:
    min_y: float | None = None
    max_y: float | None = None

    def update(y: float) -> None:
        nonlocal min_y, max_y
        if min_y is None or y < min_y:
            min_y = y
        if max_y is None or y > max_y:
            max_y = y

    for entity in entities:
        if isinstance(entity, LineEntity):
            update(entity.start[1])
            update(entity.end[1])
            continue
        if isinstance(entity, PolylineEntity):
            for _, y in entity.points:
                update(y)
            continue
        if isinstance(entity, CircleEntity):
            cy = entity.center[1]
            update(cy - entity.radius)
            update(cy + entity.radius)
            continue
        if isinstance(entity, ArcEntity):
            cy = entity.center[1]
            update(cy - entity.radius)
            update(cy + entity.radius)
            continue
        if isinstance(entity, EllipseEntity):
            cy = entity.center[1]
            major = entity.major_axis
            minor = _minor_axis_from_major(major, entity.ratio)
            y_extent = abs(major[1]) + abs(minor[1])
            update(cy - y_extent)
            update(cy + y_extent)
            continue
        if isinstance(entity, TextEntity):
            update(entity.insert[1])
            update(entity.insert[1] + entity.height)

    if min_y is None or max_y is None:
        return 0.0
    return min_y + max_y


def almost_equal_points(a: Point, b: Point, eps: float = 1e-9) -> bool:
    return math.isclose(a[0], b[0], abs_tol=eps) and math.isclose(a[1], b[1], abs_tol=eps)
