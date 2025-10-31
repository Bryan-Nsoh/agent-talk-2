# Experiments: LLM Grid Agents

**Last updated:** 2025-10-30

This document is the complete reference for running experiments, managing long-running jobs, and tracking results.

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

## How to Run Experiments

### Quick Test (verify setup)

```bash
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model openrouter:openai/gpt-5-nano \
  --turns 5
```

### Standard Run Command

**ALL experiments MUST use this pattern:**

```bash
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model <MODEL> \
  --maze-preset <PRESET> \
  --agents <N> \
  --turns <T> \
  --log-prompts \
  --log-movements \
  --emit-config experiments/<experiment-dir>/runs/$(date -u +%Y%m%dT%H%M%SZ)/config.yaml
```

**Required:** Both `--log-prompts` and `--log-movements` REQUIRE `--emit-config`. The CLI will fail fast if you forget.

**Models:**
- OpenRouter: `--model openrouter:openai/gpt-5-nano`
- Azure: `--model azure:gpt-5-mini` (or your deployment name)

**Environment:** Ensure `~/.env` contains:
- `OPENROUTER_API_KEY` (for OpenRouter)
- `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` (for Azure)

**Concurrency & history:** The episode driver is fully async. Use `run_episode(..., concurrency_start/ max)` when scripting. Observations now include a `history` block (up to five prior turns with action/comment/message summaries). Azure `gpt-4.1-mini` has been validated at 5 concurrent agents after the refactor.

### Using tmux for Long Runs

```bash
./scripts/run_experiment_tmux.sh \
  --model azure:gpt-5-mini \
  --maze-preset long_corridor \
  --agents 5 \
  --turns 120 \
  --log-prompts \
  --log-movements
```

Monitor: `tmux attach -t run_<timestamp>` or `tail -f logs/run_<timestamp>.log`

**Performance:** With serialized execution (default), expect:
- 2 agents: ~15-20 seconds per turn
- 5 agents: ~30-40 seconds per turn
- 50-turn run with 5 agents: ~25-35 minutes

## Visualization

Render an `episode.json` trace to annotated GIF:

```bash
PYTHONPATH=src uv run python -m llmgrid.cli.render_gif \
  experiments/.../results/episode.json \
  --out experiments/.../results/rollout.gif \
  --cell-size 40 --fps 6
```

Options: `--gradient` for goal-distance tint, `--no-auras` to hide visibility overlays.

## Active Workstream

- `two-agents-bearing-r1_20251028T120000Z/` — Multi-agent bearing-mode navigation on curated mazes

**Current baseline:** 
- `long_corridor` with 2 agents, visibility=1, completed in 45 turns (OpenRouter gpt-5-nano).
- 5-agent Azure `gpt-4.1-mini` runs with history enabled (`azure_history_comms_20251031T163231Z`) and with radio disabled (`azure_history_no_comms_20251031T163443Z`) both finish 60 turns (timed out; collisions increase when comms are disabled).

## Key Fix: Connection Pool Exhaustion (2025-10-30)

**Problem:** 5-agent runs failed with `APIConnectionError: Connection error` on Azure.

**Root cause:** Default `concurrency_start = len(agent_ids)` meant 5 agents triggered 5 simultaneous `asyncio.run()` calls in separate threads, exhausting Azure connection pool.

**Fix:** Rebuilt `LlmPolicy`/`run_episode` to stay on one event loop (no nested `asyncio.run`), loop-scoped limiter semaphores, and added a per-agent turn history injected into each observation.

**Result:** 5-agent Azure runs now complete with `concurrency_start=5`. History can be surfaced to the LLM; comms-enabled run remained collision-free, whereas a no-radio baseline accumulated 8 collisions.
