#!/usr/bin/env python3
"""
Generate open mazes with intentional chokepoints (no big dead zones) by
composing thin barrier lines (vertical/horizontal) with a small number of
1-cell gates, then searching gate placements that maximize shared gates across
all agents' shortest paths to the goal.

Usage (quick start):
  PYTHONPATH=src python scripts/chokepoint_generator.py \
    --width 24 --height 14 \
    --vbars 11,12 \
    --hbars 5,9 \
    --vgates 3,3 \
    --hgates 2,1 \
    --goal 2,2 \
    --starts a1:20,11 a2:18,11 a3:15,11 a4:9,11 a5:6,11 \
    --iters 300 \
    --out-prefix experiments/presets/batch/zipper_auto

This writes `<prefix>.txt`, `<prefix>_meta.json`, and `<prefix>.png`.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

Cell = Tuple[int, int]


@dataclass
class Spec:
    width: int
    height: int
    vbars: List[int]
    hbars: List[int]
    vgates: List[int]
    hgates: List[int]
    goal: Cell
    starts: Dict[str, Cell]
    iters: int
    seed: int
    out_prefix: Path


def bfs_path(width: int, height: int, walls: set[Cell], start: Cell, goal: Cell) -> Optional[List[Cell]]:
    from collections import deque

    if start == goal:
        return [start]
    q = deque([start])
    seen = {start: None}
    nbrs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    while q:
        x, y = q.popleft()
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in walls or (nx, ny) in seen:
                continue
            seen[(nx, ny)] = (x, y)
            if (nx, ny) == goal:
                # backtrack
                path: List[Cell] = [(nx, ny)]
                cur = (x, y)
                while cur is not None:
                    path.append(cur)
                    cur = seen[cur]
                path.reverse()
                return path
            q.append((nx, ny))
    return None


def build_walls(spec: Spec, gate_y_by_x: Dict[int, List[int]], gate_x_by_y: Dict[int, List[int]]) -> set[Cell]:
    w, h = spec.width, spec.height
    walls: set[Cell] = set()
    # border
    for x in range(w):
        walls.add((x, 0)); walls.add((x, h - 1))
    for y in range(h):
        walls.add((0, y)); walls.add((w - 1, y))
    # vertical bars
    for x in spec.vbars:
        gates = set(gate_y_by_x.get(x, []))
        for y in range(1, h - 1):
            if y in gates:
                continue
            walls.add((x, y))
    # horizontal bars
    for y in spec.hbars:
        gates = set(gate_x_by_y.get(y, []))
        for x in range(1, w - 1):
            if x in gates:
                continue
            walls.add((x, y))
    return walls


def score_layout(spec: Spec, walls: set[Cell]) -> Tuple[float, Dict[str, List[Cell]], Dict[Cell, int]]:
    goal = spec.goal
    paths: Dict[str, List[Cell]] = {}
    # compute shortest paths; fail if any unreachable
    for aid, start in spec.starts.items():
        path = bfs_path(spec.width, spec.height, walls, start, goal)
        if path is None:
            return (-1e9, {}, {})
        paths[aid] = path
    # Identify gate cells: any passable cell adjacent to two opposite walls in a bar
    # Simple proxy: cells that lie on any bar coordinate and are passable.
    gate_cells: set[Cell] = set()
    for x in spec.vbars:
        for y in range(1, spec.height - 1):
            if (x, y) not in walls:
                gate_cells.add((x, y))
    for y in spec.hbars:
        for x in range(1, spec.width - 1):
            if (x, y) not in walls:
                gate_cells.add((x, y))
    usage: Dict[Cell, int] = {}
    for path in paths.values():
        for cell in path:
            if cell in gate_cells:
                usage[cell] = usage.get(cell, 0) + 1

    # open ratio
    total = spec.width * spec.height
    open_cells = total - len(walls)
    open_ratio = open_cells / total
    # Quality terms:
    # - encourage open_ratio within [0.55, 0.8]
    ratio_pen = 0.0
    if open_ratio < 0.55:
        ratio_pen = -10.0 * (0.55 - open_ratio)
    elif open_ratio > 0.85:
        ratio_pen = -10.0 * (open_ratio - 0.85)
    # - shared gates across >=3 agents are valuable
    shared3 = sum(1 for k, v in usage.items() if v >= 3)
    shared2 = sum(1 for k, v in usage.items() if v == 2)
    # - path length long enough to matter but not absurd
    plen = sum(len(p) for p in paths.values()) / max(1, len(paths))
    len_bonus = 0.0
    if 30 <= plen <= 140:
        len_bonus = 0.5
    score = 5.0 * shared3 + 2.0 * shared2 + ratio_pen + len_bonus
    return (score, paths, usage)


def parse_pairs(values: Sequence[str]) -> Dict[str, Cell]:
    result: Dict[str, Cell] = {}
    for item in values:
        name, xy = item.split(":", 1)
        x_s, y_s = xy.split(",", 1)
        result[name] = (int(x_s), int(y_s))
    return result


def parse_int_list(csv: str) -> List[int]:
    return [int(s) for s in csv.split(",") if s.strip()]


def render_png(w: int, h: int, walls: set[Cell], goal: Cell, starts: Dict[str, Cell], out_png: Path) -> None:
    cs = 24
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
    d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(0, 180, 0))
    palette = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14)]
    for idx, (aid, (sx, sy)) in enumerate(sorted(starts.items())):
        cx, cy = sx * cs + cs // 2, sy * cs + cs // 2
        color = palette[idx % len(palette)]
        d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=color)
        d.text((cx - 6, cy - 20), aid, fill=color)
    img.save(out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate chokey open mazes with alternating single-file gates.")
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--height", type=int, default=14)
    ap.add_argument("--vbars", type=str, default="11,12", help="CSV of x columns for thin vertical barriers")
    ap.add_argument("--hbars", type=str, default="5,9", help="CSV of y rows for thin horizontal barriers")
    ap.add_argument("--vgates", type=str, default="3,3", help="CSV gate counts per vbar (same length as vbars)")
    ap.add_argument("--hgates", type=str, default="2,1", help="CSV gate counts per hbar (same length as hbars)")
    ap.add_argument("--goal", type=str, default="2,2")
    ap.add_argument("--starts", nargs="*", default=["a1:20,11","a2:18,11","a3:15,11","a4:9,11","a5:6,11"])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=9301)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()

    vbars = parse_int_list(args.vbars)
    hbars = parse_int_list(args.hbars)
    vgates = parse_int_list(args.vgates)
    hgates = parse_int_list(args.hgates)
    if len(vbars) != len(vgates) or len(hbars) != len(hgates):
        raise SystemExit("Length mismatch: vbars vs vgates or hbars vs hgates")
    gx, gy = (int(s) for s in args.goal.split(",", 1))
    starts = parse_pairs(args.starts)

    spec = Spec(
        width=args.width,
        height=args.height,
        vbars=vbars,
        hbars=hbars,
        vgates=vgates,
        hgates=hgates,
        goal=(gx, gy),
        starts=starts,
        iters=args.iters,
        seed=args.seed,
        out_prefix=args.out_prefix,
    )

    rng = random.Random(spec.seed)
    best_score = -1e9
    best: Optional[Tuple[set[Cell], Dict[int, List[int]], Dict[int, List[int]]]] = None
    for _ in range(spec.iters):
        gate_y_by_x: Dict[int, List[int]] = {}
        gate_x_by_y: Dict[int, List[int]] = {}
        # sample vbar gates
        for x, gcount in zip(spec.vbars, spec.vgates):
            candidates = list(range(2, spec.height - 2))
            rng.shuffle(candidates)
            gate_y_by_x[x] = sorted(candidates[:gcount])
        # sample hbar gates
        for y, gcount in zip(spec.hbars, spec.hgates):
            candidates = list(range(2, spec.width - 2))
            rng.shuffle(candidates)
            gate_x_by_y[y] = sorted(candidates[:gcount])

        walls = build_walls(spec, gate_y_by_x, gate_x_by_y)
        score, _, _ = score_layout(spec, walls)
        if score > best_score:
            best_score = score
            best = (walls, gate_y_by_x, gate_x_by_y)

    if best is None:
        raise SystemExit("Failed to generate a valid layout")

    walls, gate_y_by_x, gate_x_by_y = best
    # Write ASCII
    w, h = spec.width, spec.height
    grid = []
    for y in range(h):
        row = []
        for x in range(w):
            row.append('#' if (x, y) in walls else '.')
        grid.append(''.join(row))
    ascii_path = spec.out_prefix.with_suffix('.txt')
    ascii_path.write_text('\n'.join(grid) + '\n', encoding='utf-8')

    # Write meta JSON
    meta = {
        "maze": spec.out_prefix.name,
        "seed": spec.seed,
        "goal": {"x": spec.goal[0], "y": spec.goal[1], "rationale": "auto-generated chokepoint layout"},
        "starts": {aid: {"x": x, "y": y} for aid, (x, y) in spec.starts.items()},
        "vbars": spec.vbars,
        "hbars": spec.hbars,
        "vgates": gate_y_by_x,
        "hgates": gate_x_by_y,
        "score": best_score,
    }
    meta_path = spec.out_prefix.parent / f"{spec.out_prefix.name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding='utf-8')

    # Render PNG
    png_path = spec.out_prefix.with_suffix('.png')
    render_png(w, h, walls, spec.goal, spec.starts, png_path)

    print(str(ascii_path))
    print(str(meta_path))
    print(str(png_path))


if __name__ == "__main__":
    main()

