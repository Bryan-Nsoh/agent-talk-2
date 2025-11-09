"""Convert episode_stream.jsonl to episode.json format for partial run rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml

app = typer.Typer(add_completion=False)


@app.command()
def main(
    stream: Path = typer.Argument(..., help="Path to episode_stream.jsonl"),
    config: Path = typer.Argument(..., help="Path to config.yaml"),
    out: Path = typer.Option(..., "--out", "-o", help="Output episode.json path"),
    max_turns: Optional[int] = typer.Option(None, "--max-turns", help="Limit to first N turns"),
):
    """Convert streaming episode format to full episode.json for GIF rendering."""

    # Load config
    with config.open("r") as f:
        cfg = yaml.safe_load(f)

    # Read maze to get walls and goal
    maze_path = Path(cfg["manual_ascii_path"])
    with maze_path.open("r") as f:
        maze_lines = [line.rstrip() for line in f.readlines()]

    # Parse maze for walls
    walls = []
    width = cfg["width"]
    height = cfg["height"]
    for y, line in enumerate(maze_lines[:height]):
        for x, char in enumerate(line[:width]):
            if char == "#":
                walls.append({"x": x, "y": y})

    # Get goal from meta file
    meta_path = Path(cfg["manual_meta_path"])
    with meta_path.open("r") as f:
        meta = json.load(f)
    goal = meta["goal"]

    # Read stream frames
    frames = []
    with stream.open("r") as f:
        for line_num, line in enumerate(f, 1):
            frame_data = json.loads(line)
            turn = frame_data["turn"]

            if max_turns is not None and turn > max_turns:
                break

            # Convert agent dict to list format
            agents = []
            for agent_id, agent_state in frame_data["agents"].items():
                agents.append({
                    "agent_id": agent_id,
                    "pos": {"x": agent_state["x"], "y": agent_state["y"]},
                    "orientation": agent_state.get("orientation"),
                    "action": agent_state.get("action"),
                    "status": agent_state.get("status", "ACTIVE"),
                })

            frames.append({
                "t": turn,
                "agents": agents,
                "hazards": [],  # Could parse from artifacts if needed
            })

    # Build episode log
    episode = {
        "meta": {
            "grid_size": {"width": width, "height": height},
            "goal": goal,
            "walls": walls,
            "view": {"kind": "square", "radius": cfg["visibility"]},
            "gradient_mode": "bfs",
            "title": f"Partial run (turns 0-{frames[-1]['t']})",
            "agent_styles": [],  # Will use defaults
        },
        "frames": frames,
    }

    # Write output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(episode, f, indent=2)

    typer.secho(
        f"Converted {len(frames)} frames from {stream} to {out}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
