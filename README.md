# svg2dxf

SVG 파일을 DXF 파일로 변환하는 Python CLI 도구입니다.

## Features

### SVG 요소 지원
- `line`, `polyline`, `polygon`, `rect`, `circle`, `ellipse`, `path`, `text`

### 변환 품질
- **Arc 변환**: SVG path의 원호(A 명령)를 DXF ARC/ELLIPSE 엔티티로 변환 (polyline 근사 대신 정확한 곡선)
- **직선 최적화**: path 내 직선 세그먼트는 시작점/끝점만 사용하여 불필요한 중간점 제거
- **선 스타일**: SVG `stroke-dasharray`를 DXF linetype으로 자동 변환 (CSS class, inline style, attribute 모두 지원)
- **CSS 클래스 파싱**: SVG `<style>` 블록의 CSS 클래스 스타일 해석 (font-size, stroke-dasharray 등)

### 좌표 변환
- SVG 중첩 transform 처리 (`matrix`, `translate`, `scale`, `rotate`, `skewX`, `skewY`)
- SVG 좌표계(Y-down) → DXF 좌표계(Y-up) 자동 변환
- 출력 단위: `mm`, `inch`, `px`

### 레이어
- SVG `<g>` 요소의 `id`를 DXF 레이어로 매핑

## Install

```bash
pip install -e .
```

## Usage

```bash
svg2dxf input.svg output.dxf --unit mm --tolerance 1.0
```

### Options
| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--unit {mm,inch,px}` | 출력 단위 | `mm` |
| `--pixel-size-mm FLOAT` | SVG 1px당 mm | `0.2646` (96 DPI) |
| `--tolerance FLOAT` | 곡선 근사 허용 오차 (SVG px) | `1.0` |
| `--no-flip-y` | Y축 반전 비활성화 | - |

## Architecture

```
SVG file
  |
  v
svg_parser.py    -- XML 파싱, transform 행렬 누적, 엔티티 추출
  |
  v
models.py        -- LineEntity, PolylineEntity, CircleEntity, ArcEntity,
  |                  EllipseEntity, TextEntity
  v
converter.py     -- 단위 변환, Y축 반전, dash pattern 스케일링
  |
  v
dxf_writer.py    -- ezdxf로 DXF 파일 생성 (linetype 등록 포함)
```

## Dependencies

- [ezdxf](https://github.com/mozman/ezdxf) >= 1.2.0
- [svgpathtools](https://github.com/mathandy/svgpathtools) >= 1.6.1
- Python >= 3.11

## Development

```bash
pip install -e .[dev]
pytest
```
