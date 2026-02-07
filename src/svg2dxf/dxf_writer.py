from __future__ import annotations

import math
from pathlib import Path

import ezdxf
from ezdxf import units

from .models import ArcEntity, CircleEntity, DashPattern, EllipseEntity, Entity, LineEntity, PolylineEntity, TextEntity

_DXF_UNIT_MAP = {
    "mm": units.MM,
    "inch": units.IN,
    "px": 0,
}


def write_dxf(path: str | Path, entities: list[Entity], *, unit: str = "mm") -> None:
    doc = ezdxf.new("R2018")
    doc.units = _DXF_UNIT_MAP.get(unit.lower(), 0)
    msp = doc.modelspace()

    # Register linetypes from dash patterns
    lt_map = _register_dash_linetypes(doc, entities)

    for entity in entities:
        _ensure_layer(doc, entity.layer)
        attribs = {"layer": entity.layer}

        dp: DashPattern = getattr(entity, "dash_pattern", None)
        if dp and dp in lt_map:
            attribs["linetype"] = lt_map[dp]

        if isinstance(entity, LineEntity):
            msp.add_line((entity.start[0], entity.start[1], 0.0), (entity.end[0], entity.end[1], 0.0), dxfattribs=attribs)
            continue

        if isinstance(entity, PolylineEntity):
            msp.add_lwpolyline(entity.points, close=entity.closed, dxfattribs=attribs)
            continue

        if isinstance(entity, CircleEntity):
            msp.add_circle((entity.center[0], entity.center[1], 0.0), entity.radius, dxfattribs=attribs)
            continue

        if isinstance(entity, ArcEntity):
            msp.add_arc(
                center=(entity.center[0], entity.center[1], 0.0),
                radius=entity.radius,
                start_angle=entity.start_angle,
                end_angle=entity.end_angle,
                dxfattribs=attribs,
            )
            continue

        if isinstance(entity, EllipseEntity):
            ratio = min(1.0, max(1e-6, entity.ratio))
            msp.add_ellipse(
                center=(entity.center[0], entity.center[1], 0.0),
                major_axis=(entity.major_axis[0], entity.major_axis[1], 0.0),
                ratio=ratio,
                start_param=entity.start_param,
                end_param=entity.end_param,
                dxfattribs=attribs,
            )
            continue

        if isinstance(entity, TextEntity):
            msp.add_text(
                text=entity.text,
                dxfattribs={
                    "layer": entity.layer,
                    "insert": (entity.insert[0], entity.insert[1], 0.0),
                    "height": max(entity.height, 1e-6),
                    "rotation": entity.rotation,
                },
            )

    doc.saveas(str(path))


def _register_dash_linetypes(
    doc: ezdxf.document.Drawing, entities: list[Entity],
) -> dict[tuple[float, ...], str]:
    patterns: set[tuple[float, ...]] = set()
    for e in entities:
        dp: DashPattern = getattr(e, "dash_pattern", None)
        if dp:
            patterns.add(dp)

    lt_map: dict[tuple[float, ...], str] = {}
    for pattern in patterns:
        name = _linetype_name(pattern)
        total = sum(pattern)
        # DXF pattern: [total_length, dash, -gap, dash, -gap, ...]
        dxf_pattern: list[float] = [total]
        for i, v in enumerate(pattern):
            dxf_pattern.append(v if i % 2 == 0 else -v)
        try:
            doc.linetypes.add(name, pattern=dxf_pattern)
        except ezdxf.DXFTableEntryError:
            pass  # already registered
        lt_map[pattern] = name

    return lt_map


def _linetype_name(pattern: tuple[float, ...]) -> str:
    parts = []
    for v in pattern:
        s = f"{v:.2f}".replace(".", "p").rstrip("0").rstrip("p")
        parts.append(s)
    return "DASH_" + "_".join(parts)


def _ensure_layer(doc: ezdxf.document.Drawing, layer_name: str) -> None:
    if layer_name in doc.layers:
        return
    doc.layers.new(name=layer_name)
