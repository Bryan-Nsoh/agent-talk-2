# Two Agents – Bearing Sensor – Radius 1

**Last updated:** 2025-10-30  
**Status:** ⏳ running  
**Outcome:** -  
**Started:** 2025-10-28

## Scope

This study tracks two LLM-driven agents with radius-1 visibility, a noisy bearing sensor, and range-two radio coordination. The shared goal and all agents now operate on curated mazes instead of ad-hoc obstacle fields.

## Baseline Maze

- Default preset: `long_corridor` (seed 606, 30×10 maze with wide corridors).
- CLI command template:
  ```
  PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
    --model openrouter:openai/gpt-5-nano \
    --maze-preset long_corridor \
    --log-prompts --log-movements \
    --emit-config experiments/.../runs/<run_id>/config.yaml
  ```
- The CLI automatically applies the preset’s width, height, style, and seed; remove the preset (`--maze-preset none`) to supply custom parameters.

## Alternate Presets

Use any of the curated options when we want variety:

- `open_sparse` — 20×12 grid, 12 % scatter, light congestion.
- `open_dense` — 20×12 grid, 25 % scatter, tighter routes.
- `maze_tight` — 21×13 classic maze with minimal shortcuts.
- `maze_loops` — 21×13 maze with many alternate paths.
- `mixed_medium` — 24×14 hybrid scatter/corridor layout.

Preview PNGs live under `experiments/maze_previews/batch/`. All presets share the same start/goal placement logic and are generated deterministically from their documented seeds.

## Run Logging

- Run folders are only created once a curated preset is executed.
- Each run should capture `command.txt`, `config.yaml`, a stream transcript, the EpisodeLog JSON, and rendered GIFs.
- Current baseline: `long-corridor-nano_20251029T211939Z` — success in 45 turns, zero collisions/messages, adjacency + 5-turn history enabled.

## Next Steps

- Track additional runs across other presets once needed (all baselines complete for `long_corridor`).
- Compare performance across the other five presets on demand.
- Extend the documentation with metrics once baseline runs are recorded.
