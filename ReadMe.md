# Build & Run (Linux)

## Ubuntu dependencies

`sudo apt update && sudo apt install -y build-essential cmake ninja-build pkg-config libglfw3-dev libgl1-mesa-dev libx11-dev libxrandr-dev libxi-dev libxinerama-dev libxcursor-dev xorg-dev mesa-common-dev`

## Controls

- Mouse wheel: zoom around the cursor position.
- Left mouse drag: pan.
- Ctrl+Q: quit.


## Build

From the repo root:

1. `chmod +x build.sh`
2. `./build.sh`

Outputs:

* `build-map_viewer/map_viewer`
* `build-gl_converter/glConverter`

## Run map_viewer

* `./build-map_viewer/map_viewer path/to/map.gm`

## Run glConverter

Show usage:

* `./build-gl_converter/glConverter path/to/map.gl path/to/map.gm`


## GM File Parser

A `.gm` file is plain text and must be read in this exact order:

1. 4 count lines: Node / Edge / Polygon / Text
2. 1 bounds line: `BOUNDS minx miny maxx maxy`
3. `numNodes` lines: Nodes
4. `numEdges` lines: Edges
5. `numPolygons` lines: Polygons (one polygon per line, variable length)
6. `numTexts` lines: Text labels (one label per line, variable length)

* Examples

```bash
data/sample_features.gm
```

## Count lines (first 4 lines)

* `numNodes`: number of nodes
* `numEdges`: number of edges
* `numPolygons`: number of polygons
* `numTexts`: number of text labels

---

## BOUNDS

Format: `BOUNDS minx miny maxx maxy`

* `minx, miny`: world-space minimum corner
* `maxx, maxy`: world-space maximum corner

Used to initialize the camera and stored in the cache manifest.

---

## Nodes (`numNodes` lines)

Each line: `x y`

* `x, y`: node position in world space (float)

Used only during cache build to resolve edge endpoints.

---

## Edges (`numEdges` lines)

Each line: `u v w c`

* `u, v`: node indices (0-based) into the node list
* `w`: line width / class (int)
* `c`: type / color index (int)

During cache build, `(u, v)` is resolved to a segment `(x0,y0)-(x1,y1)`. At runtime, `w` drives line width and `c` drives the color mapping.

---

## Polygons (`numPolygons` lines)

Each line (variable length):
`POLY layer r g b a n p0 p1 ... p(n-1)`

* `layer`: draw order key (int; the viewer draws polygons in ascending layer)
* `r g b a`: color (float)
* `n`: vertex count (`n >= 3`)
* `pk`: node index

Polygons are triangulated at runtime (ear clipping). If triangulation fails, the polygon is skipped.

---

## Text (`numTexts` lines)

Each line (variable length):
`TEXT p angle size r g b a <rest of line is text>`

* `p`: label anchor node index
* `angle`: rotation angle (float; if `abs(angle) > 2π`, treat it as degrees and convert to radians. Otherwise, treat it as radians)
* `size`: label height in world units (float)
* `r g b a`: color (float)
* `text`: the remainder of the line, including spaces

## `color.conf`

Colors are independent from the map data. RGBA values are floats in the range `[0, 1]`.

```text
background r g b a
color index r g b a
```

The short form also works:

```text
index r g b a
```

Example:

```text
background 0.06 0.065 0.075 1.0
color 1 0.88 0.88 0.88 1.0
4 0.18 0.35 0.75 0.35
```

See `config/color.conf` and `data/sample_features.gm` for a complete sample with nodes, edges, polygons, and text.
