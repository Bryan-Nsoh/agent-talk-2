"""Minimal preset runner using current single-grid env."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Any
import math
import random

import typer

from llmgrid.env.simulate import run_episode
from llmgrid.schema import Position
import yaml


class _TranscriptSink:
    def append(self, _value: dict) -> None:  # pragma: no cover - simple sink
        return


class _TranscriptSink:
    def append(self, _value: dict) -> None:  # pragma: no cover - simple sink
        return

PRESETS: Dict[str, dict] = {
    "long_corridor": {
        "width": 30,
        "height": 10,
        "manual_ascii_path": "experiments/presets/batch/long_corridor_seed606.txt",
        "manual_meta_path": None,
        "goal": {"x": 28, "y": 1},  # from meta in experiment
        "starts": [
            {"agent_id": "a1", "pos": {"x": 4, "y": 1}},
            {"agent_id": "a2", "pos": {"x": 1, "y": 7}},
            {"agent_id": "a3", "pos": {"x": 11, "y": 6}},
            {"agent_id": "a4", "pos": {"x": 5, "y": 9}},
            {"agent_id": "a5", "pos": {"x": 1, "y": 3}},
        ],
    }
}


def load_manual_ascii(path: Path, width: int, height: int):
    walls = []
    lines = path.read_text().splitlines()
    for y, line in enumerate(lines[:height]):
        for x, ch in enumerate(line[:width]):
            if ch == "#":
                walls.append(Position(x=x, y=y))
    return walls


def _default_start_positions(
    width: int,
    height: int,
    goal: Position,
    agent_count: int,
    *,
    walls: set[tuple[int, int]],
    seed: int,
) -> dict[str, Position]:
    """Pick far-from-goal, well-spaced starts on open cells (ported from main)."""

    def manhattan(p: Position) -> int:
        return abs(goal.x - p.x) + abs(goal.y - p.y)

    min_distance = math.ceil((width + height) / 2)
    candidates: list[Position] = []
    for y in range(height - 1, -1, -1):
        for x in range(width):
            if (x, y) == (goal.x, goal.y) or (x, y) in walls:
                continue
            pos = Position(x=x, y=y)
            if manhattan(pos) >= min_distance:
                candidates.append(pos)

    # Relax if not enough
    all_cells: list[Position] = [
        Position(x=x, y=y)
        for y in range(height - 1, -1, -1)
        for x in range(width)
        if (x, y) != (goal.x, goal.y) and (x, y) not in walls
    ]
    all_cells.sort(key=manhattan, reverse=True)
    for pos in all_cells:
        if pos not in candidates:
            candidates.append(pos)
        if len(candidates) >= agent_count * 3:
            break

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selection: list[Position] = []
    min_pairwise = max(2, min_distance // 4)
    for pos in candidates:
        if all(abs(pos.x - chosen.x) + abs(pos.y - chosen.y) >= min_pairwise for chosen in selection):
            selection.append(pos)
        if len(selection) == agent_count:
            break

    if len(selection) < agent_count:
        for pos in all_cells:
            if pos not in selection:
                selection.append(pos)
            if len(selection) == agent_count:
                break

    return {f"a{i+1}": selection[i] for i in range(agent_count)}


def main(
    model: str = typer.Option(..., "--model", help="Model id, e.g. azure:gpt-5-mini"),
    preset: str = typer.Option("long_corridor", "--preset", help="Preset name (long_corridor)"),
    turns: int = typer.Option(100, "--turns"),
    visibility: int = typer.Option(2, "--visibility"),
    radio_range: int = typer.Option(2, "--radio-range"),
    map_sharing: str = typer.Option("radio_sync", "--map-sharing", help="none|radio_sync|global"),
    seed: int = typer.Option(13, "--seed"),
    episode_json: Path = typer.Option(..., "--episode-json"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Use heuristic instead of LLM"),
    concurrency_start: Optional[int] = typer.Option(
        None,
        "--concurrency-start",
        help="Initial number of concurrent LLM calls (defaults to max(6, agent_count)).",
    ),
    concurrency_max: Optional[int] = typer.Option(
        None,
        "--concurrency-max",
        help="Maximum concurrent LLM calls (defaults to agent_count).",
    ),
) -> None:
    if preset not in PRESETS:
        raise typer.BadParameter(f"Unknown preset {preset}")
    cfg = PRESETS[preset]
    width = cfg["width"]
    height = cfg["height"]
    goal = Position(x=cfg["goal"]["x"], y=cfg["goal"]["y"])
    start_positions = {a["agent_id"]: Position(**a["pos"]) for a in cfg["starts"]}

    walls = []
    ascii_path = cfg.get("manual_ascii_path")
    if ascii_path:
        walls = load_manual_ascii(Path(ascii_path), width, height)
    walls_set = {(p.x, p.y) for p in walls}

    # Validate provided starts; fall back to default sampler if any start is invalid.
    invalid = False
    for pos in start_positions.values():
        if (pos.x, pos.y) in walls_set or not (0 <= pos.x < width and 0 <= pos.y < height):
            invalid = True
            break
    if invalid:
        start_positions = _default_start_positions(
            width,
            height,
            goal,
            agent_count=len(cfg["starts"]),
            walls=walls_set,
            seed=seed,
        )

    transcript_records: Optional[list[dict]] = None
    transcript_sink: Optional[Any] = _TranscriptSink()
    results_dir = episode_json.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = results_dir / "transcript.jsonl"
    transcript_writer = transcript_path.open("w", encoding="utf-8")

    movement_records = [] if episode_json else None
    movement_writer = None
    if episode_json:
        movement_stream_path = results_dir / "episode_stream.jsonl"
        movement_writer = movement_stream_path.open("w", encoding="utf-8")

    # Persist config for downstream tools (stream_to_episode)
    wall_payload = [{"x": p.x, "y": p.y} for p in walls]
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#bcbd22", "#e377c2"]
    styles = [
        {"agent_id": aid, "color_hex": palette[idx % len(palette)]}
        for idx, aid in enumerate(start_positions.keys())
    ]
    config_payload = {
        "width": width,
        "height": height,
        "goal": {"x": goal.x, "y": goal.y},
        "walls": wall_payload,
        "visibility": visibility,
        "radio_range": radio_range,
        "turns": turns,
        "preset": preset,
        "map_sharing": map_sharing,
        "agent_styles": styles,
    }
    config_path = results_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as cfg_file:
        yaml.safe_dump(config_payload, cfg_file, sort_keys=False)

    agent_count = len(start_positions)
    concurrency_start = concurrency_start or max(6, agent_count)
    concurrency_max = concurrency_max or agent_count

    try:
        metrics = run_episode(
            use_llm=not dry_run,
            model_id=model,
            width=width,
            height=height,
            obstacles=walls,
            start_positions=start_positions,
            goal=goal,
            turns=turns,
            visibility=visibility,
            radio_range=radio_range,
            map_sharing=map_sharing,
            seed=seed,
            transcript=transcript_sink,
            transcript_writer=transcript_writer,
            movement=movement_records,
            movement_writer=movement_writer,
            history_limit=10,
            concurrency_start=concurrency_start,
            concurrency_max=concurrency_max,
        )
    finally:
        transcript_writer.close()
        if movement_writer:
            movement_writer.close()

    if episode_json:
        episode_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "grid_size": {"width": width, "height": height},
                "goal": {"x": goal.x, "y": goal.y},
                "walls": [{"x": p.x, "y": p.y} for p in walls],
                "agent_styles": [
                    {"agent_id": aid, "color_hex": col}
                    for aid, col in zip(start_positions.keys(), ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"])
                ],
            },
            "frames": movement_records or [],
        }
        episode_json.write_text(json.dumps(payload, indent=2))

    typer.secho(json.dumps(metrics.__dict__, indent=2), fg=typer.colors.GREEN)


if __name__ == "__main__":
    typer.run(main)
