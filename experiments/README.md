# Experiments: LLM Grid Agents

**Last updated:** 2025-10-31

This document is the complete reference for running experiments, managing long-running jobs, and tracking results.

## Experiments

| Date | Experiment | Status | Outcome | Result |
|------|------------|--------|---------|--------|
| 2025-10-31 | [loop-recovery](./loop-recovery_20251031T213232Z/) | ?running | - | Measuring history window & loop guidance |
| 2025-10-28 | [two-agents-bearing-r1](./two-agents-bearing-r1_20251028T120000Z/) | ?running | ✖ not useful | Bearing-mode multi-agent navigation |

### Status Legend
- ?running | ✔ complete | ✖ failed | ? abandoned

### Outcome Legend
- ✔ useful | ✖ not useful | ? inconclusive | - not determined

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
  --comm-strategy <none|intent|negotiation|freeform> \
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

**Concurrency & history:** The episode driver is fully async. Use `run_episode(..., concurrency_start/ max)` when scripting; Azure `gpt-4.1-mini` and OpenRouter `gpt-5-nano` both handle `concurrency_start=5` with the refactored loop-scoped limiter. Observations now supply:
- `history`: up to five prior turns including action, status-prefixed comment, peers seen, and any outbound message summary.
- `last_move_outcome`: enum flag for the previous turn (OK, BLOCK_WALL, BLOCK_AGENT, SWAP_CONFLICT, etc.).
- `contended_neighbors`: NESW bitmask showing which adjacent tiles were contested last turn.
- `last_move_outcome` plus `recent_positions` survive checkpoint/resume so prompts stay consistent mid-run.

**History & comment guardrails:**
- Turn 0 begins empty; once populated, the window remains capped at five entries (oldest entries roll off).
- Comments are auto-prefixed with a status (e.g., `BLOCKED_AGENT(...)`) and clamped to 25 words; blank or whitespace-only comments get replaced with the status alone.
- When comms are disabled or unused, `sent_message` stays `null`, but inbound radio traffic still populates `received_messages` with sender, hop distance, and age.
- Traffic-cone artifacts (NO_GO markers) persist for three turns; agents see them as `NO_GO` adjacency entries and will find a gray dot overlay in the renderer.
- Resume checkpoints persist both history and artifact TTLs; replays pick up with the exact same hazard context.

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

**Hazard overlay:** Collision-induced `NO_GO` markers now render as translucent gray dots centred in each affected cell and appear in the legend. They decay automatically (default TTL = 3), so the GIF timeline shows congestion clearing over time.

## Active Workstream

- `two-agents-bearing-r1_20251028T120000Z/` — Multi-agent bearing-mode navigation on curated mazes

**Current baseline:** 
- `long_corridor` with 2 agents, visibility=1, completed in 45 turns (OpenRouter gpt-5-nano).
- Oct 31 Azure sweeps: `azure_history_comms_20251031T165305Z` (history + radio=2, 60-turn timeout, 43 collisions, no comms) vs `azure_history_no_comms_20251031T165744Z` (radio=0, 60-turn timeout, 5 collisions, agents `a1`/`a3` finished). Earlier attempt `azure_history_comms_20251031T165135Z` failed immediately due to a wrapper bug (kept for traceability).

## Key Fix: Connection Pool Exhaustion (2025-10-30)

**Problem:** 5-agent runs failed with `APIConnectionError: Connection error` on Azure.

**Root cause:** Default `concurrency_start = len(agent_ids)` meant 5 agents triggered 5 simultaneous `asyncio.run()` calls in separate threads, exhausting Azure connection pool.

**Fix:** Rebuilt `LlmPolicy`/`run_episode` to stay on one event loop (no nested `asyncio.run`), loop-scoped limiter semaphores, and added a per-agent turn history injected into each observation.

**Result:** 5-agent Azure runs now complete with `concurrency_start=5`. History can be surfaced to the LLM; comms-enabled run remained collision-free, whereas a no-radio baseline accumulated 8 collisions.
- `results/metrics.json` now includes `collision_causes`, `hazard_events`, `comments_clamped`, `comments_autofilled`, `no_go_exposures`, and `contended_exposures` for downstream analysis. Adjust aggregators accordingly.
