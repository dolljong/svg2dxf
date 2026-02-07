from svg2dxf.svg_parser import apply_matrix, parse_transform


def test_transform_order() -> None:
    matrix = parse_transform("translate(10,0) scale(2)")
    point = apply_matrix(matrix, (1.0, 0.0))
    assert abs(point[0] - 12.0) < 1e-9
    assert abs(point[1] - 0.0) < 1e-9


def test_rotate_about_point() -> None:
    matrix = parse_transform("rotate(90, 1, 1)")
    point = apply_matrix(matrix, (2.0, 1.0))
    assert abs(point[0] - 1.0) < 1e-9
    assert abs(point[1] - 2.0) < 1e-9
