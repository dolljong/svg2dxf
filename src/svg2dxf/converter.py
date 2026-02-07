from __future__ import annotations

from pathlib import Path

from .dxf_writer import write_dxf
from .geometry import compute_y_flip_ref, map_length, map_point, map_vector, output_unit_scale
from .models import ArcEntity, CircleEntity, DashPattern, EllipseEntity, Entity, LineEntity, PolylineEntity, TextEntity
from .svg_parser import parse_svg


def convert_svg_to_dxf(
    input_svg: str | Path,
    output_dxf: str | Path,
    *,
    unit: str = "mm",
    pixel_size_mm: float = 25.4 / 96.0,
    tolerance: float = 1.0,
    flip_y: bool = True,
) -> list[Entity]:
    raw_entities = parse_svg(input_svg, tolerance=tolerance)
    converted = normalize_entities(
        raw_entities,
        unit=unit,
        pixel_size_mm=pixel_size_mm,
        flip_y=flip_y,
    )
    write_dxf(output_dxf, converted, unit=unit)
    return converted


def normalize_entities(
    entities: list[Entity], *, unit: str, pixel_size_mm: float, flip_y: bool
) -> list[Entity]:
    scale = output_unit_scale(unit, pixel_size_mm)
    y_ref = compute_y_flip_ref(entities) if flip_y else None
    out: list[Entity] = []

    def _scale_dash(dp: DashPattern) -> DashPattern:
        return tuple(v * scale for v in dp) if dp else None

    for entity in entities:
        if isinstance(entity, LineEntity):
            out.append(
                LineEntity(
                    start=map_point(entity.start, scale=scale, y_flip_ref=y_ref),
                    end=map_point(entity.end, scale=scale, y_flip_ref=y_ref),
                    layer=entity.layer,
                    dash_pattern=_scale_dash(entity.dash_pattern),
                )
            )
            continue

        if isinstance(entity, PolylineEntity):
            out.append(
                PolylineEntity(
                    points=[map_point(p, scale=scale, y_flip_ref=y_ref) for p in entity.points],
                    closed=entity.closed,
                    layer=entity.layer,
                    dash_pattern=_scale_dash(entity.dash_pattern),
                )
            )
            continue

        if isinstance(entity, CircleEntity):
            out.append(
                CircleEntity(
                    center=map_point(entity.center, scale=scale, y_flip_ref=y_ref),
                    radius=map_length(entity.radius, scale=scale),
                    layer=entity.layer,
                    dash_pattern=_scale_dash(entity.dash_pattern),
                )
            )
            continue

        if isinstance(entity, ArcEntity):
            if flip_y:
                new_start_angle = -entity.end_angle
                new_end_angle = -entity.start_angle
            else:
                new_start_angle = entity.start_angle
                new_end_angle = entity.end_angle
            out.append(
                ArcEntity(
                    center=map_point(entity.center, scale=scale, y_flip_ref=y_ref),
                    radius=map_length(entity.radius, scale=scale),
                    start_angle=new_start_angle,
                    end_angle=new_end_angle,
                    layer=entity.layer,
                    dash_pattern=_scale_dash(entity.dash_pattern),
                )
            )
            continue

        if isinstance(entity, EllipseEntity):
            if flip_y:
                new_start_param = -entity.end_param
                new_end_param = -entity.start_param
            else:
                new_start_param = entity.start_param
                new_end_param = entity.end_param
            out.append(
                EllipseEntity(
                    center=map_point(entity.center, scale=scale, y_flip_ref=y_ref),
                    major_axis=map_vector(entity.major_axis, scale=scale, y_flipped=flip_y),
                    ratio=entity.ratio,
                    start_param=new_start_param,
                    end_param=new_end_param,
                    layer=entity.layer,
                    dash_pattern=_scale_dash(entity.dash_pattern),
                )
            )
            continue

        if isinstance(entity, TextEntity):
            rotation = -entity.rotation if flip_y else entity.rotation
            out.append(
                TextEntity(
                    text=entity.text,
                    insert=map_point(entity.insert, scale=scale, y_flip_ref=y_ref),
                    height=map_length(entity.height, scale=scale),
                    rotation=rotation,
                    layer=entity.layer,
                )
            )

    return out
