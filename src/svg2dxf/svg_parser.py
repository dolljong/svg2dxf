from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from svgpathtools import Arc as _SvgArc, Line as _SvgLine, parse_path as _parse_path
except ImportError:
    _parse_path = None
    _SvgArc = None
    _SvgLine = None

from .geometry import almost_equal_points, parse_length
from .models import (
    ArcEntity, CircleEntity, DashPattern, EllipseEntity, Entity,
    LineEntity, Point, PolylineEntity, TextEntity,
)

Matrix = tuple[float, float, float, float, float, float]
IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_TRANSFORM_RE = re.compile(r"([a-zA-Z]+)\s*\(([^)]*)\)")
_DXF_LAYER_INVALID = re.compile(r"[<>/\\\":;?*|=]")
_CSS_CLASS_RULE_RE = re.compile(r"\.([a-zA-Z_][\w-]*)\s*\{([^}]*)\}")

CssStyles = dict[str, dict[str, str]]


def parse_svg(svg_path: str | Path, tolerance: float = 1.0) -> list[Entity]:
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    css = _parse_embedded_css(root)
    entities: list[Entity] = []
    _walk(root, IDENTITY, "0", entities, tolerance=max(tolerance, 1e-6), css=css)
    return entities


def _walk(
    node: ET.Element,
    inherited_matrix: Matrix,
    current_layer: str,
    out_entities: list[Entity],
    tolerance: float,
    css: CssStyles | None = None,
) -> None:
    if _is_hidden(node):
        return

    local_transform = parse_transform(node.attrib.get("transform"))
    matrix = multiply_matrices(inherited_matrix, local_transform)
    tag = _local_name(node.tag)

    layer = current_layer
    if tag == "g" and node.attrib.get("id"):
        layer = _sanitize_layer(node.attrib["id"])

    out_entities.extend(_node_to_entities(node, tag, matrix, layer, tolerance, css))

    for child in node:
        _walk(child, matrix, layer, out_entities, tolerance, css)


def _node_to_entities(
    node: ET.Element, tag: str, matrix: Matrix, layer: str, tolerance: float,
    css: CssStyles | None = None,
) -> list[Entity]:
    node_layer = layer
    node_id = node.attrib.get("id")
    if node_layer == "0" and node_id:
        node_layer = _sanitize_layer(node_id)

    dash = _resolve_dash_pattern(node, css)

    if tag == "line":
        x1 = parse_length(node.attrib.get("x1"))
        y1 = parse_length(node.attrib.get("y1"))
        x2 = parse_length(node.attrib.get("x2"))
        y2 = parse_length(node.attrib.get("y2"))
        return [
            LineEntity(
                start=apply_matrix(matrix, (x1, y1)),
                end=apply_matrix(matrix, (x2, y2)),
                layer=node_layer, dash_pattern=dash,
            )
        ]

    if tag == "polyline":
        points = _parse_points(node.attrib.get("points", ""))
        if len(points) < 2:
            return []
        transformed = [apply_matrix(matrix, p) for p in points]
        return [PolylineEntity(points=transformed, closed=False, layer=node_layer, dash_pattern=dash)]

    if tag == "polygon":
        points = _parse_points(node.attrib.get("points", ""))
        if len(points) < 2:
            return []
        transformed = [apply_matrix(matrix, p) for p in points]
        return [PolylineEntity(points=transformed, closed=True, layer=node_layer, dash_pattern=dash)]

    if tag == "rect":
        x = parse_length(node.attrib.get("x"))
        y = parse_length(node.attrib.get("y"))
        w = parse_length(node.attrib.get("width"))
        h = parse_length(node.attrib.get("height"))
        if w <= 0 or h <= 0:
            return []
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        transformed = [apply_matrix(matrix, p) for p in corners]
        return [PolylineEntity(points=transformed, closed=True, layer=node_layer, dash_pattern=dash)]

    if tag == "circle":
        cx = parse_length(node.attrib.get("cx"))
        cy = parse_length(node.attrib.get("cy"))
        r = parse_length(node.attrib.get("r"))
        if r <= 0:
            return []
        center = apply_matrix(matrix, (cx, cy))
        kind, params = _circle_or_ellipse_from_linear_transform(matrix, r)
        if kind == "circle":
            return [CircleEntity(center=center, radius=params[0], layer=node_layer, dash_pattern=dash)]
        major_axis, ratio = (params[0], params[1])  # type: ignore[index]
        return [EllipseEntity(center=center, major_axis=major_axis, ratio=ratio, layer=node_layer, dash_pattern=dash)]

    if tag == "ellipse":
        cx = parse_length(node.attrib.get("cx"))
        cy = parse_length(node.attrib.get("cy"))
        rx = parse_length(node.attrib.get("rx"))
        ry = parse_length(node.attrib.get("ry"))
        if rx <= 0 or ry <= 0:
            return []
        center = apply_matrix(matrix, (cx, cy))
        major_axis, ratio = _ellipse_from_linear_transform(matrix, rx, ry)
        return [EllipseEntity(center=center, major_axis=major_axis, ratio=ratio, layer=node_layer, dash_pattern=dash)]

    if tag == "path":
        d = node.attrib.get("d", "").strip()
        if not d:
            return []
        return _path_to_entities(d, matrix, node_layer, tolerance, dash)

    if tag == "text":
        text_value = _text_content(node)
        if not text_value:
            return []

        style = _parse_style(node.attrib.get("style", ""))
        class_style = _resolve_class_style(node, css)
        x = _first_length_coordinate(node.attrib.get("x"), default=0.0)
        y = _first_length_coordinate(node.attrib.get("y"), default=0.0)

        # Priority: inline style > CSS class > attribute > default
        if "font-size" in style:
            font_size_raw = style["font-size"]
        elif "font-size" in class_style:
            font_size_raw = class_style["font-size"]
        elif "font-size" in node.attrib:
            font_size_raw = node.attrib["font-size"]
        else:
            font_size_raw = "16"
        font_size = max(parse_length(font_size_raw, default=16.0), 1e-6)

        insert = apply_matrix(matrix, (x, y))
        up_vector = apply_linear(matrix, (0.0, font_size))
        height = math.hypot(up_vector[0], up_vector[1])
        if height <= 1e-9:
            height = font_size

        x_axis = apply_linear(matrix, (1.0, 0.0))
        if math.hypot(x_axis[0], x_axis[1]) <= 1e-12:
            rotation = 0.0
        else:
            rotation = math.degrees(math.atan2(x_axis[1], x_axis[0]))

        return [
            TextEntity(
                text=text_value,
                insert=insert,
                height=height,
                rotation=rotation,
                layer=node_layer,
            )
        ]

    return []


def _path_to_entities(
    d: str, matrix: Matrix, layer: str, tolerance: float,
    dash_pattern: DashPattern = None,
) -> list[Entity]:
    if _parse_path is None:
        return []
    try:
        path = _parse_path(d)
    except Exception:
        return []

    out: list[Entity] = []
    for subpath in path.continuous_subpaths():
        segments = list(subpath)
        if not segments:
            continue

        closed = _subpath_closed(subpath)
        has_arcs = _SvgArc is not None and any(isinstance(s, _SvgArc) for s in segments)

        if not has_arcs:
            # Original behavior: all segments → one polyline
            points: list[Point] = []
            for idx, segment in enumerate(segments):
                _sample_segment(segment, idx == 0, matrix, tolerance, points)
            if len(points) < 2:
                continue
            if closed and almost_equal_points(points[0], points[-1]):
                points.pop()
            out.append(PolylineEntity(points=points, closed=closed, layer=layer, dash_pattern=dash_pattern))
        else:
            # Split at arc boundaries
            poly_points: list[Point] = []
            need_start = True

            for idx, segment in enumerate(segments):
                if isinstance(segment, _SvgArc):
                    # Flush accumulated polyline
                    if len(poly_points) >= 2:
                        out.append(PolylineEntity(points=poly_points, closed=False, layer=layer, dash_pattern=dash_pattern))
                    poly_points = []

                    # Convert arc segment
                    arc_entity = _arc_segment_to_entity(segment, matrix, layer, dash_pattern)
                    if arc_entity is not None:
                        out.append(arc_entity)
                        need_start = True
                    else:
                        # Fallback: sample as polyline
                        _sample_segment(segment, True, matrix, tolerance, poly_points)
                        need_start = False
                else:
                    _sample_segment(segment, need_start, matrix, tolerance, poly_points)
                    need_start = False

            # Flush remaining polyline
            if len(poly_points) >= 2:
                out.append(PolylineEntity(points=poly_points, closed=False, layer=layer, dash_pattern=dash_pattern))

    return out


def _sample_segment(
    segment: object, include_start: bool, matrix: Matrix, tolerance: float,
    points: list[Point],
) -> None:
    # Straight line: just use endpoints
    if _SvgLine is not None and isinstance(segment, _SvgLine):
        if include_start:
            pt_s = apply_matrix(matrix, (float(segment.start.real), float(segment.start.imag)))
            if not points or not almost_equal_points(points[-1], pt_s):
                points.append(pt_s)
        pt_e = apply_matrix(matrix, (float(segment.end.real), float(segment.end.imag)))
        if not points or not almost_equal_points(points[-1], pt_e):
            points.append(pt_e)
        return

    # Curve: sample with tolerance
    seg_len = _segment_length(segment)
    steps = max(2, int(math.ceil(seg_len / tolerance)) + 1)
    start_idx = 0 if include_start else 1
    for i in range(start_idx, steps):
        t = i / (steps - 1)
        p = segment.point(t)
        pt = apply_matrix(matrix, (float(p.real), float(p.imag)))
        if not points or not almost_equal_points(points[-1], pt):
            points.append(pt)


def _arc_segment_to_entity(
    arc: object, matrix: Matrix, layer: str,
    dash_pattern: DashPattern = None,
) -> ArcEntity | EllipseEntity | None:
    try:
        rx = float(arc.radius.real)  # type: ignore[union-attr]
        ry = float(arc.radius.imag)  # type: ignore[union-attr]
        if rx <= 1e-12 or ry <= 1e-12:
            return None

        rotation = float(arc.rotation)  # type: ignore[union-attr]
        rot_rad = math.radians(rotation)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)

        # Ellipse axes in SVG space
        ex = (rx * cos_r, rx * sin_r)
        ey = (-ry * sin_r, ry * cos_r)

        # Apply linear part of the matrix
        new_ex = apply_linear(matrix, ex)
        new_ey = apply_linear(matrix, ey)

        # Transform center and key points
        center = apply_matrix(matrix, (float(arc.center.real), float(arc.center.imag)))  # type: ignore[union-attr]
        p_start = apply_matrix(matrix, (float(arc.start.real), float(arc.start.imag)))  # type: ignore[union-attr]
        p_end = apply_matrix(matrix, (float(arc.end.real), float(arc.end.imag)))  # type: ignore[union-attr]
        mid = arc.point(0.5)  # type: ignore[union-attr]
        p_mid = apply_matrix(matrix, (float(mid.real), float(mid.imag)))

        # Compute DXF ellipse parameters
        major_axis, ratio = _major_minor_from_columns(new_ex, new_ey)
        major_len = math.hypot(major_axis[0], major_axis[1])
        if major_len <= 1e-12:
            return None

        is_circular = math.isclose(ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6)

        if is_circular:
            radius = major_len
            a_start = math.atan2(p_start[1] - center[1], p_start[0] - center[0])
            a_end = math.atan2(p_end[1] - center[1], p_end[0] - center[0])
            a_mid = math.atan2(p_mid[1] - center[1], p_mid[0] - center[0])

            if not _is_ccw_between(a_start, a_mid, a_end):
                a_start, a_end = a_end, a_start

            return ArcEntity(
                center=center, radius=radius,
                start_angle=math.degrees(a_start),
                end_angle=math.degrees(a_end),
                layer=layer, dash_pattern=dash_pattern,
            )
        else:
            u_start = _eccentric_anomaly(p_start, center, major_axis, ratio)
            u_end = _eccentric_anomaly(p_end, center, major_axis, ratio)
            u_mid = _eccentric_anomaly(p_mid, center, major_axis, ratio)

            if not _is_ccw_between(u_start, u_mid, u_end):
                u_start, u_end = u_end, u_start

            return EllipseEntity(
                center=center, major_axis=major_axis, ratio=ratio,
                start_param=u_start, end_param=u_end,
                layer=layer, dash_pattern=dash_pattern,
            )
    except Exception:
        return None


def _eccentric_anomaly(
    point: Point, center: Point, major_axis: Point, ratio: float,
) -> float:
    major_len = math.hypot(major_axis[0], major_axis[1])
    if major_len <= 1e-12:
        return 0.0
    major_dir = (major_axis[0] / major_len, major_axis[1] / major_len)
    minor_dir = (-major_axis[1] / major_len, major_axis[0] / major_len)
    qx = point[0] - center[0]
    qy = point[1] - center[1]
    cos_u = (qx * major_dir[0] + qy * major_dir[1]) / major_len
    minor_len = major_len * ratio
    if minor_len <= 1e-12:
        sin_u = 0.0
    else:
        sin_u = (qx * minor_dir[0] + qy * minor_dir[1]) / minor_len
    return math.atan2(sin_u, cos_u)


def _is_ccw_between(start: float, mid: float, end: float) -> bool:
    TWO_PI = 2.0 * math.pi
    sweep_to_mid = (mid - start) % TWO_PI
    sweep_to_end = (end - start) % TWO_PI
    if sweep_to_end < 1e-9:
        sweep_to_end = TWO_PI
    return sweep_to_mid <= sweep_to_end + 1e-9


def _segment_length(segment: object) -> float:
    if hasattr(segment, "length"):
        try:
            return float(segment.length(error=1e-4))
        except TypeError:
            return float(segment.length())
        except Exception:
            return 1.0
    return 1.0


def _subpath_closed(subpath: object) -> bool:
    maybe = getattr(subpath, "isclosed", None)
    if callable(maybe):
        try:
            return bool(maybe())
        except Exception:
            return False
    return bool(maybe)


def _parse_embedded_css(root: ET.Element) -> CssStyles:
    css: CssStyles = {}
    for el in root.iter():
        if _local_name(el.tag) != "style":
            continue
        text = el.text or ""
        for match in _CSS_CLASS_RULE_RE.finditer(text):
            class_name = match.group(1)
            declarations = match.group(2)
            css[class_name] = _parse_style(declarations)
    return css


def _resolve_class_style(node: ET.Element, css: CssStyles | None) -> dict[str, str]:
    if not css:
        return {}
    merged: dict[str, str] = {}
    class_attr = node.attrib.get("class", "")
    for cls in class_attr.split():
        if cls in css:
            merged.update(css[cls])
    return merged


def _resolve_dash_pattern(
    node: ET.Element, css: CssStyles | None,
) -> DashPattern:
    style = _parse_style(node.attrib.get("style", ""))
    class_style = _resolve_class_style(node, css)

    raw = None
    if "stroke-dasharray" in style:
        raw = style["stroke-dasharray"]
    elif "stroke-dasharray" in class_style:
        raw = class_style["stroke-dasharray"]
    elif "stroke-dasharray" in node.attrib:
        raw = node.attrib["stroke-dasharray"]

    if not raw or raw.strip().lower() == "none":
        return None

    values = [float(v) for v in _NUMBER_RE.findall(raw)]
    if len(values) < 2:
        return None
    # SVG spec: odd-length arrays are repeated to make even
    if len(values) % 2 == 1:
        values = values + values
    return tuple(values)


def _parse_points(points_raw: str) -> list[Point]:
    values = [float(v) for v in _NUMBER_RE.findall(points_raw)]
    if len(values) < 4:
        return []
    if len(values) % 2 == 1:
        values = values[:-1]
    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def _parse_style(style_raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not style_raw:
        return out
    for item in style_raw.split(";"):
        if ":" not in item:
            continue
        k, v = item.split(":", 1)
        key = k.strip().lower()
        value = v.strip()
        if key:
            out[key] = value
    return out


def _first_length_coordinate(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    token = raw.replace(",", " ").strip().split()
    if not token:
        return default
    return parse_length(token[0], default=default)


def _text_content(node: ET.Element) -> str:
    raw = "".join(node.itertext())
    normalized = " ".join(raw.split())
    return normalized


def _circle_or_ellipse_from_linear_transform(
    matrix: Matrix, radius: float
) -> tuple[str, tuple[float] | tuple[Point, float]]:
    a, b, c, d, _, _ = matrix
    vx = (a * radius, b * radius)
    vy = (c * radius, d * radius)

    len_x = math.hypot(vx[0], vx[1])
    len_y = math.hypot(vy[0], vy[1])
    dot = vx[0] * vy[0] + vx[1] * vy[1]

    if math.isclose(dot, 0.0, abs_tol=1e-8) and math.isclose(len_x, len_y, rel_tol=1e-8, abs_tol=1e-8):
        return ("circle", ((len_x + len_y) * 0.5,))

    major_axis, ratio = _major_minor_from_columns(vx, vy)
    return ("ellipse", (major_axis, ratio))


def _ellipse_from_linear_transform(matrix: Matrix, rx: float, ry: float) -> tuple[Point, float]:
    a, b, c, d, _, _ = matrix
    vx = (a * rx, b * rx)
    vy = (c * ry, d * ry)
    return _major_minor_from_columns(vx, vy)


def _major_minor_from_columns(col1: Point, col2: Point) -> tuple[Point, float]:
    m00, m10 = col1
    m01, m11 = col2

    sxx = m00 * m00 + m01 * m01
    syy = m10 * m10 + m11 * m11
    sxy = m00 * m10 + m01 * m11

    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = max(0.0, trace * trace * 0.25 - det)
    root = math.sqrt(disc)
    l1 = max(0.0, trace * 0.5 + root)
    l2 = max(0.0, trace * 0.5 - root)

    major_len = math.sqrt(l1)
    minor_len = math.sqrt(l2)

    if major_len <= 1e-12:
        return ((1.0, 0.0), 1.0)

    if abs(sxy) > 1e-12:
        vx = l1 - syy
        vy = sxy
    else:
        vx = 1.0 if sxx >= syy else 0.0
        vy = 0.0 if sxx >= syy else 1.0

    norm = math.hypot(vx, vy)
    if norm <= 1e-12:
        vx, vy = 1.0, 0.0
        norm = 1.0
    ux = vx / norm
    uy = vy / norm

    major_axis = (ux * major_len, uy * major_len)
    ratio = 1.0 if major_len <= 1e-12 else max(0.0, min(1.0, minor_len / major_len))
    return major_axis, ratio


def _is_hidden(node: ET.Element) -> bool:
    display = node.attrib.get("display", "").strip().lower()
    visibility = node.attrib.get("visibility", "").strip().lower()
    if display == "none" or visibility == "hidden":
        return True

    style = node.attrib.get("style", "")
    style_lower = style.lower()
    if "display:none" in style_lower or "visibility:hidden" in style_lower:
        return True
    return False


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _sanitize_layer(name: str) -> str:
    sanitized = _DXF_LAYER_INVALID.sub("_", name).strip()
    if not sanitized:
        return "0"
    return sanitized[:255]


def parse_transform(raw: str | None) -> Matrix:
    if not raw:
        return IDENTITY

    matrix = IDENTITY
    for op, args_raw in _TRANSFORM_RE.findall(raw):
        args = [float(v) for v in _NUMBER_RE.findall(args_raw)]
        op_lower = op.lower()

        if op_lower == "matrix" and len(args) == 6:
            t = (args[0], args[1], args[2], args[3], args[4], args[5])
        elif op_lower == "translate":
            tx = args[0] if args else 0.0
            ty = args[1] if len(args) > 1 else 0.0
            t = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif op_lower == "scale":
            sx = args[0] if args else 1.0
            sy = args[1] if len(args) > 1 else sx
            t = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif op_lower == "rotate":
            angle = math.radians(args[0] if args else 0.0)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rot = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                t = multiply_matrices(
                    multiply_matrices((1.0, 0.0, 0.0, 1.0, cx, cy), rot),
                    (1.0, 0.0, 0.0, 1.0, -cx, -cy),
                )
            else:
                t = rot
        elif op_lower == "skewx":
            angle = math.radians(args[0] if args else 0.0)
            t = (1.0, 0.0, math.tan(angle), 1.0, 0.0, 0.0)
        elif op_lower == "skewy":
            angle = math.radians(args[0] if args else 0.0)
            t = (1.0, math.tan(angle), 0.0, 1.0, 0.0, 0.0)
        else:
            continue

        matrix = multiply_matrices(matrix, t)

    return matrix


def multiply_matrices(m1: Matrix, m2: Matrix) -> Matrix:
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def apply_matrix(matrix: Matrix, point: Point) -> Point:
    a, b, c, d, e, f = matrix
    x, y = point
    return (a * x + c * y + e, b * x + d * y + f)


def apply_linear(matrix: Matrix, vector: Point) -> Point:
    a, b, c, d, _, _ = matrix
    x, y = vector
    return (a * x + c * y, b * x + d * y)
