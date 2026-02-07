from pathlib import Path

import ezdxf

from svg2dxf.converter import convert_svg_to_dxf


def test_svg_to_dxf_integration(tmp_path: Path) -> None:
    svg_file = tmp_path / "sample.svg"
    dxf_file = tmp_path / "sample.dxf"

    svg_file.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <line x1="0" y1="0" x2="100" y2="0" />
  <circle cx="50" cy="50" r="10" />
  <text x="10" y="20" font-size="12">HELLO</text>
</svg>""",
        encoding="utf-8",
    )

    entities = convert_svg_to_dxf(svg_file, dxf_file, unit="mm", tolerance=0.5)
    assert len(entities) >= 3
    assert dxf_file.exists()

    doc = ezdxf.readfile(str(dxf_file))
    msp = doc.modelspace()
    types = [e.dxftype() for e in msp]
    assert len(types) >= 3
    assert "TEXT" in types
