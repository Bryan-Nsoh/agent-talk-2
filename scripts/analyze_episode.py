"""Quick analyzer for map-sharing episode logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel


class AgentStep(BaseModel):
    t: int
    x: int
    y: int


def load_episode(path: Path) -> dict:
    return json.loads(path.read_text())


def trajectories(frames: List[dict]) -> Dict[str, List[AgentStep]]:
    paths: Dict[str, List[AgentStep]] = {}
    for frame in frames:
        t = frame.get("t", frame.get("turn", -1))
        for agent in frame.get("agents", []):
            aid = agent["agent_id"]
            pos = agent["pos"]
            paths.setdefault(aid, []).append(AgentStep(t=t, x=pos["x"], y=pos["y"]))
    return paths


def first_goal_hits(paths: Dict[str, List[AgentStep]], goal: Tuple[int, int]) -> Dict[str, int]:
    hits: Dict[str, int] = {}
    gx, gy = goal
    for aid, steps in paths.items():
        for step in steps:
            if (step.x, step.y) == (gx, gy):
                hits[aid] = step.t
                break
    return hits


def summary(episode_path: Path) -> str:
    data = load_episode(episode_path)
    frames = data.get("frames", [])
    meta = data.get("meta", {})
    goal = (meta.get("goal", {}).get("x"), meta.get("goal", {}).get("y"))
    grid = meta.get("grid_size", {})
    paths = trajectories(frames)
    hits = first_goal_hits(paths, goal)
    lines = []
    lines.append(f"Episode: {episode_path.name}")
    lines.append(f"Frames: {len(frames)}, Grid: ({grid.get('width')}x{grid.get('height')}), Goal: {goal}")
    for aid, steps in paths.items():
        dist = abs(steps[-1].x - goal[0]) + abs(steps[-1].y - goal[1]) if goal[0] is not None else None
        hit = hits.get(aid)
        lines.append(
            f"- {aid}: steps={len(steps)}"
            + (f", first_goal_t={hit}" if hit is not None else ", goal_unreached")
            + (f", final_pos=({steps[-1].x},{steps[-1].y}), manhattan_to_goal={dist}" if dist is not None else "")
        )
    if hits:
        fastest = min(hits.items(), key=lambda kv: kv[1])
        lines.append(f"Fastest goal: {fastest[0]} at turn {fastest[1]}")
    else:
        lines.append("No agent reached the goal.")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize an episode log.")
    parser.add_argument("episode_json", type=Path, help="Path to episode JSON log")
    args = parser.parse_args()
    print(summary(args.episode_json))


if __name__ == "__main__":
    main()
