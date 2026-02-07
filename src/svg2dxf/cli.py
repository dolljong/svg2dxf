from __future__ import annotations

import argparse
import sys

from .converter import convert_svg_to_dxf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SVG file to DXF.")
    parser.add_argument("input_svg", help="Path to input SVG file")
    parser.add_argument("output_dxf", help="Path to output DXF file")
    parser.add_argument(
        "--unit",
        choices=("mm", "inch", "px"),
        default="mm",
        help="Output unit (default: mm)",
    )
    parser.add_argument(
        "--pixel-size-mm",
        type=float,
        default=25.4 / 96.0,
        help="Millimeters per SVG pixel (default: 25.4/96)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Path approximation tolerance in SVG px",
    )
    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        help="Disable SVG Y-down to DXF Y-up conversion",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        convert_svg_to_dxf(
            args.input_svg,
            args.output_dxf,
            unit=args.unit,
            pixel_size_mm=args.pixel_size_mm,
            tolerance=max(args.tolerance, 1e-6),
            flip_y=not args.no_flip_y,
        )
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1
    return 0
