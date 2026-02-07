from pathlib import Path

from svg2dxf.converter import normalize_entities
from svg2dxf.models import TextEntity
from svg2dxf.svg_parser import parse_svg


def test_parse_svg_text_entity(tmp_path: Path) -> None:
    svg_file = tmp_path / "text.svg"
    svg_file.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg">
  <text x="10" y="20" font-size="12">  hello
  world </text>
</svg>""",
        encoding="utf-8",
    )

    entities = parse_svg(svg_file)
    texts = [e for e in entities if isinstance(e, TextEntity)]
    assert len(texts) == 1
    assert texts[0].text == "hello world"
    assert abs(texts[0].height - 12.0) < 1e-9


def test_text_rotation_flips_with_y_axis() -> None:
    entities = [
        TextEntity(
            text="A",
            insert=(10.0, 10.0),
            height=5.0,
            rotation=30.0,
            layer="0",
        )
    ]
    mapped = normalize_entities(entities, unit="px", pixel_size_mm=25.4 / 96.0, flip_y=True)
    assert isinstance(mapped[0], TextEntity)
    assert mapped[0].rotation == -30.0


def test_text_height_compensation_default() -> None:
    entities = [TextEntity(text="A", insert=(0.0, 0.0), height=12.0, rotation=0.0, layer="0")]
    mapped = normalize_entities(entities, unit="px", pixel_size_mm=25.4 / 96.0, flip_y=False)
    assert isinstance(mapped[0], TextEntity)
    assert abs(mapped[0].height - 12.0) < 1e-9


def test_text_xy_with_mm_units(tmp_path: Path) -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg">
  <text x="25.4mm" y="12.7mm" font-size="10mm">A</text>
</svg>"""
    path = tmp_path / "tmp_text_units.svg"
    path.write_text(svg, encoding="utf-8")
    entities = parse_svg(path)

    texts = [e for e in entities if isinstance(e, TextEntity)]
    assert len(texts) == 1
    assert abs(texts[0].insert[0] - 96.0) < 1e-6
    assert abs(texts[0].insert[1] - 48.0) < 1e-6
