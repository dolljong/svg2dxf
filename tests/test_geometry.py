from svg2dxf.geometry import map_point, output_unit_scale, parse_length


def test_parse_length_mm_to_px() -> None:
    px = parse_length("25.4mm")
    assert abs(px - 96.0) < 1e-6


def test_output_unit_scale_mm() -> None:
    scale = output_unit_scale("mm", 25.4 / 96.0)
    assert abs(scale - (25.4 / 96.0)) < 1e-12


def test_map_point_with_y_flip() -> None:
    point = map_point((10.0, 5.0), scale=2.0, y_flip_ref=20.0)
    assert point == (20.0, 30.0)
