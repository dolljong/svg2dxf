# svg2dxf

A small Python CLI project for converting SVG files to DXF.

## Features (MVP)
- Supports: `line`, `polyline`, `polygon`, `rect`, `circle`, `ellipse`, `path`, `text`
- Handles nested SVG transforms (`matrix`, `translate`, `scale`, `rotate`, `skewX`, `skewY`)
- Converts SVG coordinate system (Y down) to DXF-friendly Y up by default
- Supports output units: `mm`, `inch`, `px`

## Install

```bash
pip install -e .
```

## Usage

```bash
svg2dxf input.svg output.dxf --unit mm --tolerance 1.0
```

### Options
- `--unit {mm,inch,px}`: output unit (default: `mm`)
- `--pixel-size-mm FLOAT`: mm per SVG px (default: `0.2645833333`, i.e. 96 DPI)
- `--tolerance FLOAT`: path approximation tolerance in SVG px (default: `1.0`)
- `--no-flip-y`: disable Y-axis inversion

## Development

```bash
pip install -e .[dev]
pytest
```
"# svg2dxf" 
