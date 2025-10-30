# Experiments: LLM Grid Agents

**Last updated:** 2025-10-30

## Curated Maze Presets

We now standardise all navigation experiments on six hand-picked mazes generated with the new runtime maze generator. They live under `experiments/presets/batch/` as PNG previews and are reproduced deterministically at run time from their seeds.

- `long_corridor` (seed 606, 30×10, maze style). Wide horizontal corridors with loops near the goal. This is the default baseline for all runs unless we ask for a different preset. First successful Nano run (Oct 30, 2025) completed the goal in 45 turns with zero collisions.
- `open_sparse` (seed 101, 20×12, random scatter at 12 % density). Almost-open field with small clusters.
- `open_dense` (seed 202, 20×12, random scatter at 25 % density). Dense pockets of obstacles with narrow passages.
- `maze_tight` (seed 303, 21×13, maze style, low extra connections). Classic, single-path maze with sharp turns.
- `maze_loops` (seed 404, 21×13, maze style, many extra connections). Maze with plentiful shortcuts and loops.
- `mixed_medium` (seed 505, 24×14, random scatter at 18 % plus extra openings). Hybrid scatter-plus-corridor layout.

Use the `--maze-preset` flag on `llmgrid.cli.poc_two_agents` to pick a maze. The CLI now defaults to `long_corridor`; pass `--maze-preset none` to fall back to manual width/height/density flags.

Preview images can be regenerated or extended with:

```
PYTHONPATH=src uv run python -m llmgrid.cli.generate_maze --help
```

## Active Workstream

- `two-agents-bearing-r1_20251028T120000Z/` — proof-of-concept experiment re-based on the curated mazes. All legacy diagnostic runs were archived; future runs will reference the presets above.

We only create new run folders once a curated preset run is executed. Historical ad-hoc runs have been removed to keep the repository focused on the new workflow.
