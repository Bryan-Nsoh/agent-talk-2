"""Convert episode_stream.jsonl to episode.json format for partial run rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import typer
import yaml

app = typer.Typer(add_completion=False)


def _load_walls(cfg: Dict[str, Any]) -> list[dict]:
    if "walls" in cfg and cfg["walls"]:
        return cfg["walls"]
    ascii_path = Path(cfg.get("manual_ascii_path", ""))
    if not ascii_path.is_file():
        return []
    width = cfg["width"]
    height = cfg["height"]
    walls: list[dict] = []
    with ascii_path.open("r", encoding="utf-8") as fh:
        for y, line in enumerate(fh.readlines()[:height]):
            for x, ch in enumerate(line[:width]):
                if ch == "#":
                    walls.append({"x": x, "y": y})
    return walls


@app.command()
def main(
    stream: Path = typer.Argument(..., help="Path to episode_stream.jsonl"),
    config: Path = typer.Argument(..., help="Path to config.yaml"),
    out: Path = typer.Option(..., "--out", "-o", help="Output episode.json path"),
    max_turns: Optional[int] = typer.Option(None, "--max-turns", help="Limit to first N turns"),
):
    """Convert streaming episode format to full episode.json for GIF rendering."""

    cfg = yaml.safe_load(config.read_text())
    width = cfg["width"]
    height = cfg["height"]
    goal = cfg.get("goal") or {"x": 0, "y": 0}
    walls = _load_walls(cfg)
    visibility = cfg.get("visibility", 2)
    agent_styles = cfg.get("agent_styles") or []

    frames = []
    last_turn = None
    with stream.open("r", encoding="utf-8") as fh:
        for line in fh:
            frame_data = json.loads(line)
            turn = frame_data.get("turn", 0)
            if max_turns is not None and turn > max_turns:
                break
            agents_payload = []
            for agent_id, pos in frame_data.get("positions", {}).items():
                agents_payload.append(
                    {
                        "agent_id": agent_id,
                        "pos": {"x": pos["x"], "y": pos["y"]},
                        "orientation": None,
                        "action": None,
                        "status": "FINISHED" if agent_id in frame_data.get("finished", []) else "ACTIVE",
                    }
                )
            frames.append({"t": turn, "agents": agents_payload})
            last_turn = turn

    if not frames:
        raise typer.BadParameter("No frames found in stream; is the run producing movement logs?")

    episode = {
        "meta": {
            "grid_size": {"width": width, "height": height},
            "goal": goal,
            "walls": walls,
            "view": {"kind": "square", "radius": visibility},
            "gradient_mode": "bfs",
            "title": f"Partial run (turns 0-{last_turn})",
            "agent_styles": agent_styles,
        },
        "frames": frames,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(episode, indent=2))
    typer.secho(f"Converted {len(frames)} frames from {stream} to {out}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
