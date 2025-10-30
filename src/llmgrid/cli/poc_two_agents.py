"""CLI entry point for the two-agent proof of concept."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, TextIO
from collections import deque

import typer

from llmgrid.env.simulate import EpisodeCheckpoint, EpisodeMetrics, run_episode
from llmgrid.env.maze_generator import MazeConfig, MazeGenerator
from llmgrid.schema import Position
from llmgrid.logging.episode_log import (
    AgentState as LogAgentState,
    AgentStyle,
    EpisodeLog,
    EpisodeMeta,
    Frame as LogFrame,
    GridSize,
    Position as LogPosition,
    ViewShape,
)

app = typer.Typer(add_completion=False)


MAZE_PRESETS = {
    "long_corridor": {
        "width": 30,
        "height": 10,
        "style": "maze",
        "density": None,
        "extra": 0.2,
        "seed": 606,
        "description": "Wide horizontal corridor network with loops near the goal.",
    },
    "open_sparse": {
        "width": 20,
        "height": 12,
        "style": "random",
        "density": 0.12,
        "extra": 0.0,
        "seed": 101,
        "description": "Light scatter of obstacles for almost-open navigation.",
    },
    "open_dense": {
        "width": 20,
        "height": 12,
        "style": "random",
        "density": 0.25,
        "extra": 0.0,
        "seed": 202,
        "description": "Heavier scatter with narrow passages and pockets.",
    },
    "maze_tight": {
        "width": 21,
        "height": 13,
        "style": "maze",
        "density": None,
        "extra": 0.05,
        "seed": 303,
        "description": "Classic single-path maze with few shortcuts.",
    },
    "maze_loops": {
        "width": 21,
        "height": 13,
        "style": "maze",
        "density": None,
        "extra": 0.35,
        "seed": 404,
        "description": "Maze with many cross-links and alternate loops.",
    },
    "mixed_medium": {
        "width": 24,
        "height": 14,
        "style": "random",
        "density": 0.18,
        "extra": 0.15,
        "seed": 505,
        "description": "Combination of scatter and short corridors.",
    },
}


@app.command()
def main(
    model: str = typer.Option(
        ...,
        "--model",
        help="Fully qualified model id, e.g. openrouter:openai/gpt-oss-20b:free",
    ),
    width: int = typer.Option(12, "--width", help="Grid width."),
    height: int = typer.Option(12, "--height", help="Grid height."),
    visibility: int = typer.Option(1, "--visibility", help="Visibility radius R."),
    radio_range: int = typer.Option(2, "--radio-range", help="Radio range r."),
    turns: int = typer.Option(120, "--turns", help="Turn budget."),
    seed: int = typer.Option(13, "--seed", help="Random seed."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Use the local heuristic baseline instead of calling the LLM."
    ),
    no_obstacles: bool = typer.Option(
        False, "--no-obstacles", help="Start with an empty grid (no static obstacles)."
    ),
    obstacle_density: Optional[float] = typer.Option(
        None,
        "--obstacle-density",
        help="Optional fraction of free cells to fill with random obstacles (0-1). Overrides --no-obstacles.",
    ),
    obstacle_count: Optional[int] = typer.Option(
        None,
        "--obstacle-count",
        help="Optional absolute number of random obstacles. Overrides --no-obstacles and --obstacle-density.",
    ),
    obstacle_seed: Optional[int] = typer.Option(
        None,
        "--obstacle-seed",
        help="Seed for random obstacle placement (defaults to --seed when generating random obstacles).",
    ),
    bearing_bias_seed: Optional[int] = typer.Option(
        None,
        "--bearing-bias-seed",
        help="Enable Gold Drift by setting a deterministic seed (default: disabled).",
    ),
    bearing_bias_p: float = typer.Option(
        0.0,
        "--bearing-bias-p",
        min=0.0,
        max=0.49,
        help="Baseline probability of rotating the bearing by ±45° when bias is enabled.",
    ),
    bearing_bias_wall_bonus: float = typer.Option(
        0.0,
        "--bearing-bias-wall-bonus",
        min=0.0,
        max=0.49,
        help="Additional probability added when the cell touches a wall.",
    ),
    maze_preset: str = typer.Option(
        "long_corridor",
        "--maze-preset",
        help="Curated maze preset name (long_corridor, open_sparse, open_dense, maze_tight, maze_loops, mixed_medium) or 'none' for custom settings.",
    ),
    maze_style: str = typer.Option(
        "maze",
        "--maze-style",
        help="Obstacle generator style: 'maze' (default) or 'random'.",
    ),
    maze_extra_connection: float = typer.Option(
        0.1,
        "--maze-extra-connection",
        help="For maze-style obstacles, probability of carving additional side passages (0 to 1).",
    ),
    log_prompts: bool = typer.Option(
        False,
        "--log-prompts/--no-log-prompts",
        help="Capture full prompts and structured outputs for every agent turn.",
    ),
    log_movements: bool = typer.Option(
        True,
        "--log-movements/--no-log-movements",
        help="Capture agent locations per turn for downstream visualization.",
    ),
    transcript_jsonl: Optional[Path] = typer.Option(
        None,
        "--transcript-jsonl",
        help="Optional path for the prompt transcript (defaults to results/transcript.jsonl if --emit-config is set).",
    ),
    episode_json: Optional[Path] = typer.Option(
        None,
        "--episode-json",
        help="Optional path for the EpisodeLog JSON (defaults to results/episode.json if --emit-config is set).",
    ),
    emit_config: Optional[Path] = typer.Option(
        None,
        "--emit-config",
        help="Optional path to dump the resolved configuration YAML.",
    ),
    checkpoint_json: Optional[Path] = typer.Option(
        None,
        "--checkpoint-json",
        help="Path to write periodic checkpoints for resuming interrupted runs.",
    ),
    checkpoint_interval: int = typer.Option(
        1,
        "--checkpoint-interval",
        min=1,
        help="Turns between checkpoint writes (default: every turn).",
    ),
    resume_from: Optional[Path] = typer.Option(
        None,
        "--resume-from",
        help="Resume a partially completed run from an existing checkpoint JSON file.",
    ),
    agents: int = typer.Option(
        2,
        "--agents",
        min=1,
        max=8,
        help="Number of controllable agents to spawn (default: 2).",
    ),
) -> None:
    # Validate model prefix
    if not (model.startswith("openrouter:") or model.startswith("azure:")):
        typer.secho(
            "Error: use --model with openrouter: or azure: prefix (e.g. azure:gpt-5-mini).",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    # Validate logging flags require output paths
    if log_prompts and emit_config is None:
        typer.secho(
            "Error: --log-prompts requires --emit-config to specify where transcript.jsonl should be written.",
            fg=typer.colors.RED,
        )
        typer.secho(
            "Example: --emit-config experiments/my-run/config.yaml",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=2)

    if log_movements and emit_config is None and episode_json is None:
        typer.secho(
            "Error: --log-movements requires --emit-config or --episode-json to specify where episode logs should be written.",
            fg=typer.colors.RED,
        )
        typer.secho(
            "Example: --emit-config experiments/my-run/config.yaml",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=2)

    preset_name = maze_preset.lower()
    resume_checkpoint: Optional[EpisodeCheckpoint] = None
    if resume_from is not None:
        try:
            resume_checkpoint = EpisodeCheckpoint.load(resume_from)
        except FileNotFoundError as exc:
            typer.secho(f"Checkpoint not found: {resume_from}", fg=typer.colors.RED)
            raise typer.Exit(code=2) from exc

        if model != resume_checkpoint.model_id:
            typer.secho(
                f"Model mismatch: checkpoint expects '{resume_checkpoint.model_id}', got '{model}'.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)

        expected_dry_run = not resume_checkpoint.use_llm
        if dry_run != expected_dry_run:
            mode_msg = "dry-run" if expected_dry_run else "LLM-backed"
            typer.secho(
                f"Checkpoint was recorded for a {mode_msg} run. Adjust --dry-run accordingly.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)

        width = resume_checkpoint.world.width
        height = resume_checkpoint.world.height
        visibility = resume_checkpoint.visibility
        radio_range = resume_checkpoint.radio_range
        turns = resume_checkpoint.turns_total
        seed = resume_checkpoint.seed
        preset_name = resume_checkpoint.maze_metadata.get("preset", preset_name)
        maze_style = resume_checkpoint.maze_metadata.get("maze_style", maze_style)
        maze_extra_connection = resume_checkpoint.maze_metadata.get("maze_extra_connection", maze_extra_connection)
        no_obstacles = resume_checkpoint.maze_metadata.get("no_obstacles", no_obstacles)
        obstacle_density = resume_checkpoint.maze_metadata.get("obstacle_density", obstacle_density)
        obstacle_count = resume_checkpoint.maze_metadata.get("obstacle_count", obstacle_count)
        obstacle_seed = resume_checkpoint.maze_metadata.get("obstacle_seed", obstacle_seed)
        bearing_bias_seed = resume_checkpoint.maze_metadata.get("bearing_bias_seed", bearing_bias_seed)
        bearing_bias_p = resume_checkpoint.maze_metadata.get("bearing_bias_p", bearing_bias_p)
        bearing_bias_wall_bonus = resume_checkpoint.maze_metadata.get("bearing_bias_wall_bonus", bearing_bias_wall_bonus)
        typer.secho(
            f"Resuming from {resume_from} at turn {resume_checkpoint.turn_next}/{turns}",
            fg=typer.colors.BLUE,
        )

    preset_details = None
    if resume_checkpoint is None and preset_name != "none":
        preset_details = MAZE_PRESETS.get(preset_name)
        if preset_details is None:
            typer.secho(
                f"Unknown maze preset '{maze_preset}'. Available presets: {', '.join(MAZE_PRESETS.keys())}, or 'none'.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)
        width = preset_details["width"]
        height = preset_details["height"]
        maze_style = preset_details["style"]
        maze_extra_connection = preset_details["extra"]
        obstacle_density = preset_details["density"]
        obstacle_seed = preset_details["seed"]
        obstacle_count = None
        no_obstacles = False
        typer.secho(
            f"Using maze preset '{preset_name}' (seed={obstacle_seed}) — {preset_details['description']}",
            fg=typer.colors.BLUE,
        )

    checkpoint_path = checkpoint_json if checkpoint_json is not None else resume_from

    maze_metadata: Dict[str, Optional[float | int | str | bool]] = {
        "preset": preset_name,
        "maze_style": maze_style,
        "maze_extra_connection": maze_extra_connection,
        "no_obstacles": no_obstacles,
        "obstacle_density": obstacle_density,
        "obstacle_count": obstacle_count,
        "obstacle_seed": obstacle_seed,
        "bearing_bias_seed": bearing_bias_seed,
        "bearing_bias_p": bearing_bias_p,
        "bearing_bias_wall_bonus": bearing_bias_wall_bonus,
        "agents": agents,
    }

    if emit_config and resume_checkpoint is None:
        _write_config(
            emit_config,
            {
                "model": model,
                "width": width,
                "height": height,
                "visibility": visibility,
                "radio_range": radio_range,
                "turns": turns,
                "seed": seed,
                "obstacle_density": obstacle_density,
                "obstacle_count": obstacle_count,
                "obstacle_seed": obstacle_seed,
                "maze_preset": preset_name,
                "maze_style": maze_style,
                "maze_extra_connection": maze_extra_connection,
                "dry_run": dry_run,
                "no_obstacles": no_obstacles,
                "bearing_bias_seed": bearing_bias_seed,
                "bearing_bias_p": bearing_bias_p,
                "bearing_bias_wall_bonus": bearing_bias_wall_bonus,
                "agents": agents,
            },
        )

    if resume_checkpoint is not None:
        start_positions = resume_checkpoint.start_positions
        goal = resume_checkpoint.goal
        obstacles = [Position(x=p.x, y=p.y) for p in resume_checkpoint.world.walls]
    else:
        goal = _default_goal(width, height)
        start_positions = _default_start_positions(width, height, goal, agents, seed=seed)
        obstacles = _resolve_obstacles(
            width=width,
            height=height,
            start_positions=start_positions,
            goal=goal,
            no_obstacles=no_obstacles,
            obstacle_count=obstacle_count,
            obstacle_density=obstacle_density,
            obstacle_seed=obstacle_seed if obstacle_seed is not None else seed,
            maze_style=maze_style,
            maze_extra_connection=maze_extra_connection,
            agent_count=agents,
            start_seed=seed,
        )

    capture_transcript = log_prompts or transcript_jsonl is not None
    if capture_transcript:
        if resume_checkpoint and resume_checkpoint.transcript is not None:
            transcript_records = list(resume_checkpoint.transcript)
        else:
            transcript_records = []
    else:
        transcript_records = None

    transcript_path: Optional[Path] = transcript_jsonl
    if resume_checkpoint and resume_checkpoint.transcript_path:
        checkpoint_transcript_path = Path(resume_checkpoint.transcript_path).expanduser()
        if transcript_path is None:
            transcript_path = checkpoint_transcript_path
        else:
            if transcript_path.expanduser().resolve(strict=False) != checkpoint_transcript_path.resolve(strict=False):
                typer.secho(
                    "Transcript path differs from checkpoint. Re-run without --transcript-jsonl to reuse the stored path.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=2)

    if capture_transcript and transcript_path is None and emit_config is not None:
        transcript_path = emit_config.parent / "results" / "transcript.jsonl"
    if capture_transcript and transcript_path is None:
        typer.secho(
            "Transcript capture requested but no output path available. Provide --emit-config or --transcript-jsonl.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    capture_movement = log_movements or episode_json is not None
    if capture_movement:
        if resume_checkpoint and resume_checkpoint.movement is not None:
            movement_records = list(resume_checkpoint.movement)
        else:
            movement_records = []
    else:
        movement_records = None

    episode_path: Optional[Path] = episode_json
    if resume_checkpoint and resume_checkpoint.episode_path:
        checkpoint_episode_path = Path(resume_checkpoint.episode_path).expanduser()
        if episode_path is None:
            episode_path = checkpoint_episode_path
        else:
            if episode_path.expanduser().resolve(strict=False) != checkpoint_episode_path.resolve(strict=False):
                typer.secho(
                    "Episode log path differs from checkpoint. Re-run without --episode-json to reuse the stored path.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=2)

    if capture_movement and episode_path is None and emit_config is not None:
        episode_path = emit_config.parent / "results" / "episode.json"
    if capture_movement and episode_path is None:
        typer.secho(
            "Movement logging requested but no output path available. Provide --emit-config or --episode-json.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    movement_stream_path: Optional[Path] = None
    if resume_checkpoint and resume_checkpoint.movement_stream_path:
        movement_stream_path = Path(resume_checkpoint.movement_stream_path).expanduser()

    agent_order = resume_checkpoint.agent_ids if resume_checkpoint else list(start_positions.keys())

    transcript_handle: Optional[TextIO] = None
    movement_stream_handle: Optional[TextIO] = None
    try:
        if transcript_path is not None:
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if resume_checkpoint is not None else "w"
            transcript_handle = transcript_path.open(mode, encoding="utf-8")

        if capture_movement and episode_path is not None:
            if movement_stream_path is None:
                movement_stream_path = episode_path.with_name(episode_path.stem + "_stream.jsonl")
            movement_stream_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if resume_checkpoint is not None else "w"
            movement_stream_handle = movement_stream_path.open(mode, encoding="utf-8")

        metrics = run_episode(
            use_llm=not dry_run,
            model_id=model,
            width=width,
            height=height,
            obstacles=obstacles,
            start_positions=start_positions,
            goal=goal,
            turns=turns,
            visibility=visibility,
            radio_range=radio_range,
            seed=seed,
            transcript=transcript_records,
            movement=movement_records,
            transcript_writer=transcript_handle,
            movement_writer=movement_stream_handle,
            agent_order=agent_order,
            resume=resume_checkpoint,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            transcript_path=str(transcript_path) if transcript_path is not None else None,
            movement_stream_path=str(movement_stream_path) if movement_stream_path is not None else None,
            episode_path=str(episode_path) if episode_path is not None else None,
            maze_metadata={k: v for k, v in maze_metadata.items() if v is not None},
            bearing_bias_seed=bearing_bias_seed,
            bearing_bias_p=bearing_bias_p,
            bearing_bias_wall_bonus=bearing_bias_wall_bonus,
        )
    finally:
        if transcript_handle is not None:
            transcript_handle.flush()
            transcript_handle.close()
        if movement_stream_handle is not None:
            movement_stream_handle.flush()
            movement_stream_handle.close()

    typer.secho(json.dumps(metrics.__dict__, indent=2), fg=typer.colors.GREEN)

    if transcript_records is not None and transcript_path is not None:
        if transcript_handle is None:
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            with transcript_path.open("w", encoding="utf-8") as handle:
                for record in transcript_records:
                    handle.write(json.dumps(record))
                    handle.write("\n")
            typer.secho(f"Prompt transcript saved to {transcript_path}", fg=typer.colors.BLUE)
        else:
            typer.secho(f"Prompt transcript streaming to {transcript_path}", fg=typer.colors.BLUE)

    if movement_records is not None and episode_path is not None:
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        for entry in movement_records:
            agents = []
            for aid, payload in sorted(entry["agents"].items()):
                agents.append(
                    LogAgentState(
                        agent_id=aid,
                        pos=LogPosition(x=payload["x"], y=payload["y"]),
                        orientation=payload.get("orientation"),
                        action=payload.get("action"),
                        status=payload.get("status", "ACTIVE"),
                    )
                )
            frames.append(LogFrame(t=entry["turn"], agents=agents))

        agent_styles = _default_agent_styles(sorted(start_positions.keys()))
        episode_log = EpisodeLog(
            meta=EpisodeMeta(
                grid_size=GridSize(width=width, height=height),
                goal=LogPosition(x=goal.x, y=goal.y),
                walls=[LogPosition(x=p.x, y=p.y) for p in obstacles],
                view=ViewShape(kind="square", radius=visibility),
                gradient_mode="bfs",
                title=f"{model} R={visibility}",
                agent_styles=agent_styles,
            ),
            frames=frames,
        )

        with episode_path.open("w", encoding="utf-8") as handle:
            handle.write(episode_log.model_dump_json(indent=2))
        typer.secho(f"Episode log saved to {episode_path}", fg=typer.colors.BLUE)
        if movement_stream_path is not None:
            typer.secho(f"Movement stream saved to {movement_stream_path}", fg=typer.colors.BLUE)


def _write_config(path: Path, data: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def _resolve_obstacles(
    *,
    width: int,
    height: int,
    start_positions: Optional[dict[str, Position]],
    goal: Optional[Position],
    no_obstacles: bool,
    obstacle_count: Optional[int],
    obstacle_density: Optional[float],
    obstacle_seed: int,
    maze_style: str,
    maze_extra_connection: float,
    agent_count: int,
    start_seed: Optional[int],
    max_attempts: int = 100,
) -> list[Position]:
    style = maze_style.lower()
    start_positions = start_positions or _default_start_positions(
        width,
        height,
        goal or _default_goal(width, height),
        agent_count,
        seed=start_seed,
    )
    goal = goal or _default_goal(width, height)

    if style == "maze" and not no_obstacles:
        config = MazeConfig(
            width=width,
            height=height,
            seed=obstacle_seed,
            extra_connection_prob=maze_extra_connection,
        )
        generator = MazeGenerator(config)
        required_cells = [(pos.x, pos.y) for pos in start_positions.values()]
        required_cells.append((goal.x, goal.y))
        return generator.generate(required_open_cells=required_cells)

    if obstacle_count is None and obstacle_density is None:
        return [] if no_obstacles else _default_obstacles(width, height)

    total_cells = width * height
    if total_cells <= 0:
        return []

    if obstacle_count is not None and obstacle_count < 0:
        raise typer.BadParameter("obstacle-count must be non-negative.")

    if obstacle_density is not None:
        if not 0 <= obstacle_density <= 1:
            raise typer.BadParameter("obstacle-density must be between 0 and 1.")

    rng = random.Random(obstacle_seed)

    forbidden = {(goal.x, goal.y)}
    forbidden.update((pos.x, pos.y) for pos in start_positions.values())

    available: list[tuple[int, int]] = [
        (x, y) for x in range(width) for y in range(height) if (x, y) not in forbidden
    ]
    if not available:
        return []

    if obstacle_count is not None:
        count = min(obstacle_count, len(available))
    else:
        count = int(round(obstacle_density * len(available)))  # type: ignore[arg-type]
        count = max(0, min(count, len(available)))

    # attempt sampling until connectivity holds
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        sampled = rng.sample(available, count)
        sampled_set = set(sampled)
        if _paths_exist(width, height, sampled_set, start_positions, goal):
            return [Position(x=x, y=y) for x, y in sampled_set]
    # fallback: no obstacles to guarantee progress
    typer.secho(
        "Warning: could not sample reachable obstacle layout after "
        f"{max_attempts} attempts. Falling back to empty grid.",
        fg=typer.colors.YELLOW,
    )
    return []


def _neighbors(x: int, y: int) -> list[tuple[int, int]]:
    return [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]


def _paths_exist(
    width: int,
    height: int,
    obstacles: set[tuple[int, int]],
    start_positions: dict[str, Position],
    goal: Position,
) -> bool:
    gx, gy = goal.x, goal.y
    if (gx, gy) in obstacles:
        return False
    for pos in start_positions.values():
        if not _reachable(width, height, obstacles, (pos.x, pos.y), (gx, gy)):
            return False
    return True


def _reachable(
    width: int,
    height: int,
    obstacles: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> bool:
    if start == goal:
        return True
    visited = set()
    dq = deque([start])
    visited.add(start)
    while dq:
        x, y = dq.popleft()
        for nx, ny in _neighbors(x, y):
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in obstacles or (nx, ny) in visited:
                continue
            if (nx, ny) == goal:
                return True
            visited.add((nx, ny))
            dq.append((nx, ny))
    return False


def _default_obstacles(width: int, height: int) -> list[Position]:
    raw = [
        Position(x=4, y=2),
        Position(x=4, y=3),
        Position(x=4, y=4),
        Position(x=7, y=6),
        Position(x=7, y=7),
        Position(x=7, y=8),
        Position(x=2, y=max(0, height - 2)),
        Position(x=3, y=max(0, height - 2)),
    ]
    return [p for p in raw if p.x < width and p.y < height]


def _default_start_positions(
    width: int,
    height: int,
    goal: Position,
    agent_count: int,
    *,
    seed: Optional[int] = None,
) -> dict[str, Position]:
    min_distance = math.ceil((width + height) / 2)

    def manhattan(p: Position) -> int:
        return abs(goal.x - p.x) + abs(goal.y - p.y)

    candidates: List[Position] = []
    for y in range(height - 1, -1, -1):
        for x in range(0, width):
            if (x, y) == (goal.x, goal.y):
                continue
            pos = Position(x=x, y=y)
            if manhattan(pos) >= min_distance:
                candidates.append(pos)

    if len(candidates) < agent_count:
        # Relax requirement gradually until enough positions are available
        all_cells = [
            Position(x=x, y=y)
            for y in range(height - 1, -1, -1)
            for x in range(width)
            if (x, y) != (goal.x, goal.y)
        ]
        all_cells.sort(key=manhattan, reverse=True)
        for pos in all_cells:
            if pos not in candidates:
                candidates.append(pos)
            if len(candidates) >= agent_count:
                break

    rng = random.Random(seed)
    rng.shuffle(candidates)

    selection: List[Position] = []
    min_pairwise = max(2, min_distance // 4)
    for pos in candidates:
        if all(abs(pos.x - chosen.x) + abs(pos.y - chosen.y) >= min_pairwise for chosen in selection):
            selection.append(pos)
        if len(selection) == agent_count:
            break

    if len(selection) < agent_count:
        # Fill any remaining slots without the spacing constraint.
        for pos in candidates:
            if pos in selection:
                continue
            selection.append(pos)
            if len(selection) == agent_count:
                break

    return {f"a{i + 1}": selection[i] for i in range(agent_count)}


def _default_goal(width: int, height: int) -> Position:
    goal_x = max(0, width - 2)
    goal_y = min(1, max(0, height - 1))
    return Position(x=goal_x, y=goal_y)


DEFAULT_AGENT_COLORS = {
    "a1": "#1f77b4",
    "a2": "#d62728",
    "a3": "#2ca02c",
    "a4": "#9467bd",
    "a5": "#ff7f0e",
    "a6": "#17becf",
}


def _default_agent_styles(agent_ids: List[str]) -> List[AgentStyle]:
    styles: List[AgentStyle] = []
    palette_cycle = list(DEFAULT_AGENT_COLORS.values())
    for idx, aid in enumerate(agent_ids):
        color = DEFAULT_AGENT_COLORS.get(aid)
        if color is None:
            color = palette_cycle[idx % len(palette_cycle)]
        styles.append(AgentStyle(agent_id=aid, color_hex=color))
    return styles


if __name__ == "__main__":
    app()
