#!/usr/bin/env python3
"""Check if all agents can reach goal using BFS."""

import json
from pathlib import Path
from collections import deque

def load_maze(ascii_path):
    with open(ascii_path) as f:
        lines = [line.rstrip() for line in f.readlines()]

    h = len(lines)
    w = max(len(line) for line in lines) if lines else 0

    walls = set()
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == '#':
                walls.add((x, y))

    return w, h, walls

def bfs_path(w, h, walls, start, goal):
    if start == goal:
        return [start]

    q = deque([start])
    seen = {start: None}

    while q:
        x, y = q.popleft()
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in walls or (nx, ny) in seen:
                continue
            seen[(nx, ny)] = (x, y)
            if (nx, ny) == goal:
                # Backtrack
                path = [(nx, ny)]
                cur = (x, y)
                while cur is not None:
                    path.append(cur)
                    cur = seen[cur]
                path.reverse()
                return path
            q.append((nx, ny))

    return None

if __name__ == "__main__":
    import sys
    ascii_path = Path(sys.argv[1])
    meta_path = Path(sys.argv[2])

    w, h, walls = load_maze(ascii_path)

    with open(meta_path) as f:
        meta = json.load(f)

    goal = (meta["goal"]["x"], meta["goal"]["y"])
    starts = {aid: (data["x"], data["y"]) for aid, data in meta["starts"].items()}

    print(f"Maze: {w}x{h}, {len(walls)} walls")
    print(f"Goal: {goal}")
    print(f"Starts: {starts}")
    print()

    all_reachable = True
    for aid, start in starts.items():
        path = bfs_path(w, h, walls, start, goal)
        if path:
            print(f"{aid}: REACHABLE (path length {len(path)})")
        else:
            print(f"{aid}: UNREACHABLE ❌")
            all_reachable = False

    if all_reachable:
        print("\n✓ All agents can reach goal")
    else:
        print("\n✗ Some agents cannot reach goal - MAZE IS INVALID")
        sys.exit(1)
