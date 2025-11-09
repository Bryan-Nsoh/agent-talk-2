#!/usr/bin/env python3
"""Quick script to render a maze ASCII to PNG."""

import json
from pathlib import Path
from PIL import Image, ImageDraw

def render_png(w: int, h: int, walls: set, goal: tuple, starts: dict, out_png: Path) -> None:
    cs = 40
    img = Image.new("RGB", (w * cs, h * cs), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for y in range(h):
        for x in range(w):
            x0, y0 = x * cs, y * cs
            x1, y1 = x0 + cs, y0 + cs
            if (x, y) in walls:
                d.rectangle((x0, y0, x1, y1), fill=(30, 30, 30))
            else:
                d.rectangle((x0, y0, x1, y1), fill=(245, 245, 245))
            d.rectangle((x0, y0, x1, y1), outline=(220, 220, 220))
    gx, gy = goal
    cx, cy = gx * cs + cs // 2, gy * cs + cs // 2
    d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=(0, 180, 0))
    palette = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14)]
    for idx, (aid, (sx, sy)) in enumerate(sorted(starts.items())):
        cx, cy = sx * cs + cs // 2, sy * cs + cs // 2
        color = palette[idx % len(palette)]
        d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=color)
    img.save(out_png)

if __name__ == "__main__":
    import sys
    ascii_path = Path(sys.argv[1])
    meta_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    with open(ascii_path) as f:
        lines = [line.rstrip() for line in f.readlines()]

    h = len(lines)
    w = len(lines[0]) if lines else 0

    walls = set()
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == '#':
                walls.add((x, y))

    with open(meta_path) as f:
        meta = json.load(f)

    goal = (meta["goal"]["x"], meta["goal"]["y"])
    starts = {aid: (data["x"], data["y"]) for aid, data in meta["starts"].items()}

    render_png(w, h, walls, goal, starts, out_path)
    print(f"Rendered {out_path}")
