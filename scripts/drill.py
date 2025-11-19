"""Unified drill CLI for episode/movement logs.

Subcommands (all on a single entry point):
  summary       — overview: earliest goal, per-agent audit, distance stats (goal inferred if missing)
  audit         — per-agent first-goal, final position, coverage
  goal-turns    — exact goal-entry turns per agent
  turn-stats    — per-turn active/hits/distances
  progress      — min/mean/max distance to goal per turn (CSV)
  heatmap       — visitation grid (JSON matrix)
  patch-goal-hits — backfill goal_hits into logs (in-place or --out)
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


# ----------------- loaders -----------------
def load_frames(path: Path) -> Tuple[List[dict], Tuple[int, int]]:
    """
    Load frames and goal from supported formats:
      - Episode JSON: {"frames":[...], "meta":{"goal":{"x":..,"y":..}}}
      - Movement JSON/JSONL: list of snapshots with positions
    """
    text = path.read_text()
    obj = json.loads(text)
    if isinstance(obj, dict) and "frames" in obj:
        frames = obj["frames"]
        meta = obj.get("meta", {})
        goal = (meta.get("goal", {}).get("x"), meta.get("goal", {}).get("y"))
        return frames, goal
    if isinstance(obj, list):
        return obj, (None, None)
    raise ValueError(f"Unsupported format: {path}")


def infer_goal_from_hits(frames: List[dict]) -> Tuple[int, int] | Tuple[None, None]:
    """Infer goal coordinate from first goal_hits/positions when the goal is unknown."""
    for frame in frames:
        hits = frame.get("goal_hits", [])
        if not hits:
            continue
        positions = positions_from_frame(frame)
        for aid in hits:
            if aid in positions:
                pos = positions[aid]
                return pos["x"], pos["y"]
    return (None, None)


def positions_from_frame(frame: dict) -> Dict[str, dict]:
    if "positions" in frame:
        return frame["positions"]
    return {a["agent_id"]: a["pos"] for a in frame.get("agents", [])}


# ----------------- functions -----------------
def goal_turns(frames: List[dict], goal: Tuple[int, int]) -> Dict[str, int]:
    gx, gy = goal
    hits: Dict[str, int] = {}
    for frame in frames:
        t = frame.get("turn", frame.get("t", -1))
        for aid in frame.get("goal_hits", []):
            hits.setdefault(aid, t)
        if gx is None:
            continue
        for aid, pos in positions_from_frame(frame).items():
            if (pos["x"], pos["y"]) == (gx, gy):
                hits.setdefault(aid, t)
    return hits


def audit(frames: List[dict], goal: Tuple[int, int]) -> str:
    goal = goal if goal[0] is not None else (None, None)
    paths: Dict[str, List[Tuple[int, Tuple[int, int]]]] = {}
    for frame in frames:
        t = frame.get("turn", frame.get("t", -1))
        for aid, pos in positions_from_frame(frame).items():
            paths.setdefault(aid, []).append((t, (pos["x"], pos["y"])))
    lines: List[str] = []
    fastest = None
    for aid, steps in sorted(paths.items()):
        first_goal = next((t for t, pos in steps if goal[0] is not None and pos == goal), None)
        last_t, last_pos = steps[-1]
        dist = abs(last_pos[0] - goal[0]) + abs(last_pos[1] - goal[1]) if goal[0] is not None else None
        coverage = len({pos for _, pos in steps})
        lines.append(
            f"- {aid}: first_goal_t={first_goal if first_goal is not None else 'none'}, "
            f"last=({last_pos[0]},{last_pos[1]})@t={last_t}, "
            f"manhattan_to_goal={dist}, visited={coverage}"
        )
        if first_goal is not None:
            fastest = min(fastest, (first_goal, aid)) if fastest else (first_goal, aid)
    lines.append(f"Fastest: {fastest[1]} @t={fastest[0]}" if fastest else "Fastest: none")
    return "\n".join(lines)


def turn_stats(frames: List[dict], goal: Tuple[int, int]) -> str:
    lines: List[str] = []
    cumulative_hits = set()
    gx, gy = goal
    goal_known = gx is not None
    for frame in frames:
        t = frame.get("turn", frame.get("t", -1))
        positions = positions_from_frame(frame)
        dist_map = frame.get("dist_to_goal", {})
        hits = frame.get("goal_hits", [])
        cumulative_hits.update(hits)
        active = len(positions)
        finished = len(frame.get("finished", []))
        if goal_known:
            dists = list(dist_map.values()) or [
                abs(pos["x"] - gx) + abs(pos["y"] - gy) for pos in positions.values()
            ]
            min_d = min(dists) if dists else "-"
            max_d = max(dists) if dists else "-"
        else:
            min_d = max_d = "-"
        lines.append(
            f"t={t:03d} active={active} finished={finished} hits_turn={hits} "
            f"hits_cum={sorted(cumulative_hits)} min_d={min_d} max_d={max_d}"
        )
    return "\n".join(lines)


def progress_curve(frames: List[dict], goal: Tuple[int, int]) -> List[tuple[int, float, float, float]]:
    gx, gy = goal
    stats: List[tuple[int, float, float, float]] = []
    for frame in frames:
        t = frame.get("turn", frame.get("t", -1))
        positions = positions_from_frame(frame)
        dists = []
        for pos in positions.values():
            if gx is None:
                continue
            d = abs(pos["x"] - gx) + abs(pos["y"] - gy)
            dists.append(d)
        if not dists:
            continue
        stats.append((t, min(dists), statistics.mean(dists), max(dists)))
    return stats


def heatmap(frames: List[dict], size: Tuple[int, int]) -> List[List[int]]:
    counts = Counter()
    for frame in frames:
        for pos in positions_from_frame(frame).values():
            counts[(pos["x"], pos["y"])] += 1
    w, h = size
    if w is None or h is None:
        if not counts:
            return []
        w = max(x for (x, _) in counts) + 1
        h = max(y for (_, y) in counts) + 1
    grid = [[0 for _ in range(w)] for _ in range(h)]
    for (x, y), c in counts.items():
        if 0 <= y < h and 0 <= x < w:
            grid[y][x] = c
    return grid


def patched_goal_hits(frames: List[dict], goal: Tuple[int, int]) -> List[dict]:
    """Return new frames with goal_hits filled from positions when missing."""
    gx, gy = goal
    patched: List[dict] = []
    for frame in frames:
        hits = set(frame.get("goal_hits", []))
        if gx is not None:
            for aid, pos in positions_from_frame(frame).items():
                if (pos["x"], pos["y"]) == (gx, gy):
                    hits.add(aid)
        new_frame = dict(frame)
        new_frame["goal_hits"] = sorted(hits)
        patched.append(new_frame)
    return patched


def summary(frames: List[dict], goal: Tuple[int, int]) -> str:
    """
    Unified overview:
      - earliest goal turn (if any)
      - per-agent first-goal / final pos / coverage
      - end-of-run distance stats
      - notes whether goal was provided or inferred
    """
    goal_known = goal[0] is not None
    inferred = None
    if not goal_known:
        inferred = infer_goal_from_hits(frames)
        if inferred != (None, None):
            goal = inferred
            goal_known = True
    lines: List[str] = []
    audit_lines = audit(frames, goal).splitlines()
    goal_note = "inferred" if inferred else "provided"
    if goal[0] is None:
        lines.append(f"Frames: {len(frames)}  Goal: unknown ({goal_note})")
    else:
        lines.append(f"Frames: {len(frames)}  Goal: {goal} ({goal_note})")
    # earliest goal
    hits = goal_turns(frames, goal)
    if hits:
        earliest = min(hits.values())
        lines.append(f"Earliest goal turn: {earliest} ({', '.join([aid for aid, t in hits.items() if t==earliest])})")
    else:
        lines.append("Earliest goal turn: none")
    # final distance stats
    gx, gy = goal
    final_dists = []
    for line in audit_lines:
        if line.startswith("- "):
            parts = line.split("manhattan_to_goal=")
            if len(parts) == 2:
                try:
                    d = float(parts[1].split(",")[0])
                    if d >= 0:
                        final_dists.append(d)
                except Exception:
                    pass
    if final_dists:
        lines.append(
            f"Final distance stats: min={min(final_dists):.2f} mean={statistics.mean(final_dists):.2f} max={max(final_dists):.2f}"
        )
    lines.extend(audit_lines)
    return "\n".join(lines)


# ----------------- CLI -----------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Unified drill CLI for episode/movement logs.")
    subs = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("path", type=Path, help="Episode JSON or movement JSON/JSONL")
        p.add_argument("--goal", type=str, default=None, help="Override goal as x,y")

    p_audit = subs.add_parser("audit", help="Per-agent first-goal, final pos, coverage")
    add_common(p_audit)

    p_turns = subs.add_parser("turn-stats", help="Per-turn hits/active/distances")
    add_common(p_turns)

    p_goal = subs.add_parser("goal-turns", help="List goal-entry turns per agent")
    add_common(p_goal)

    p_progress = subs.add_parser("progress", help="Min/mean/max distance CSV")
    add_common(p_progress)
    p_progress.add_argument("--out", type=Path, default=None, help="CSV output path (stdout if omitted)")
    p_progress.add_argument("--require-goal", action="store_true", help="Fail if goal is missing and cannot be inferred")

    p_heat = subs.add_parser("heatmap", help="Visitation grid as JSON matrix")
    add_common(p_heat)

    p_patch = subs.add_parser("patch-goal-hits", help="Backfill goal_hits in-place or to --out")
    add_common(p_patch)
    p_patch.add_argument("--out", type=Path, default=None, help="Optional output path (overwrites if omitted)")

    p_summary = subs.add_parser("summary", help="Combined overview: earliest goal, audit, distance stats")
    add_common(p_summary)

    p_gif = subs.add_parser("gif", help="Render GIF (rich Among Us sprites)")
    add_common(p_gif)
    p_gif.add_argument("--out", type=Path, required=True, help="Output GIF path")
    p_gif.add_argument("--model-name", type=str, default="unknown")
    p_gif.add_argument("--show-gradient", action="store_true", help="Overlay goal distance gradient")

    args = parser.parse_args()

    frames, goal = load_frames(args.path)
    if args.goal:
        gx, gy = map(int, args.goal.split(","))
        goal = (gx, gy)

    if args.cmd == "audit":
        print(audit(frames, goal))
    elif args.cmd == "goal-turns":
        hits = goal_turns(frames, goal)
        if hits:
            earliest = min(hits.values())
            print(f"earliest_goal_turn={earliest} hits={hits}")
        else:
            print("no goal reached")
    elif args.cmd == "turn-stats":
        print(turn_stats(frames, goal))
    elif args.cmd == "progress":
        if goal[0] is None and args.require_goal:
            raise SystemExit("Goal is unknown and cannot compute distances (use --goal x,y or patch goa_hits first).")
        stats = progress_curve(frames, goal)
        lines = ["turn,min,mean,max"]
        for t, mn, mean, mx in stats:
            lines.append(f"{t},{mn:.2f},{mean:.2f},{mx:.2f}")
        csv = "\n".join(lines)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(csv)
            print(f"wrote {args.out}")
        else:
            print(csv)
    elif args.cmd == "heatmap":
        grid = heatmap(frames, goal if goal[0] is not None else (None, None))
        print(json.dumps(grid))
    elif args.cmd == "patch-goal-hits":
        frames = patched_goal_hits(frames, goal)
        out_path = args.out if args.out else args.path
        payload = {"frames": frames}
        # preserve meta if present
        obj = json.loads(args.path.read_text())
        if isinstance(obj, dict) and "meta" in obj:
            payload["meta"] = obj["meta"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"wrote patched goal_hits to {out_path}")
    elif args.cmd == "summary":
        print(summary(frames, goal))
    elif args.cmd == "gif":
        from llmgrid.vis.gif import GifRenderer, RenderOptions

        opts = RenderOptions(show_gradient=args.show_gradient)
        renderer = GifRenderer(options=opts, model_name=args.model_name)
        renderer.render(args.path, args.out)
        print(f"Saved GIF to {args.out}")


if __name__ == "__main__":
    main()
