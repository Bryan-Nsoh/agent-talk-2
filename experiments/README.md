# Experiments: LLM Grid Agents

**Last updated:** 2025-11-19T22:30:00Z

This document is the complete reference for running experiments, managing long-running jobs, and tracking results.

> WARNING: **Engine change (2025-11-06):** Commits `0a0e38d`, `5291aea`, and `e4ce883` corrected multiple simulation defects (frozen orientations, message ages, LLM-owned `seq`, idle sprites). Any runs recorded before 2025-11-06 must be rerun under the fixed engine; treat existing tables as legacy references only.

## Communication Strategy Results

**Canonical seed 13 result (commit fe3ffda):**
- STRUCTURED: 73% success (11/15 agents finished)
- Freeform: 33% success (5/15 agents finished)
- None: 20% success (3/15 agents finished)

**Cross-seed generalization test (seeds 13-17, 45 total runs):**
- FREEFORM: 62.7% success (47/75 agents) - WINNER
- None: 57.3% success (43/75 agents)
- Structured: 56.0% success (42/75 agents)

Canonical seed 13 showed structured winning (73%), but testing across 5 different spawn seeds reverses the ranking. Freeform communication generalizes better across scenarios, and "none" performs nearly as well as structured (questioning the value of the INTENT/REQUEST protocol). See [cross_seed_baseline_20251112T143355Z](./cross_seed_baseline_20251112T143355Z/) for full analysis with token-counted visualizations.

## Experiments

### Key Communication Validation Studies

| Date | Experiment | Status | Outcome | Result |
|------|------------|--------|---------|--------|
| **2025-11-12** | [**cross_seed_baseline_20251112T143355Z**](./cross_seed_baseline_20251112T143355Z/) | **✅ complete (45/45)** | **✓ useful** | **Cross-seed test: freeform 62.7%, none 57.3%, structured 56.0% (45 runs, 5 seeds, with token counting)** |
| 2025-11-10 | [long_corridor_final_20251110T155342Z](./long_corridor_final_20251110T155342Z/) | complete | useful | Canonical seed 13: structured 73%, freeform 33%, none 20% (9 runs, seed-specific) |
| 2025-11-10 | [long_corridor_comms_test_20251110T020144Z](./long_corridor_comms_test_20251110T020144Z/) | complete | useful | Exploratory: structured 3/5, freeform 2/5, none 0/5 - discovered priority deadlock (commit 419c6aa) |

### Map-Sharing Verification (2025-11-19; complete)

| Date | Experiment | Status | Outcome | Result |
|------|------------|--------|---------|--------|
| 2025-11-19 | [mapshare_long_corridor_20251119T202017Z](./mapshare_long_corridor_20251119T202017Z/) | complete | useful | Seeds 13–17, 3 modes (none/radio_sync/global), 15 total runs. Global: 100% success, 0.0 std dev goal discovery. Radio_sync: 60% success, plateaus at turn 40. None: 60% success, 0 collisions. Radio_sync provides no advantage over baseline. Plots: analysis/mapshare/plots/ |

### Other Experiments

None beyond the projects listed above.

## In-flight Verification Plan (2025-11-19)

Now that the single-grid renderer and streaming telemetry are restored on branch `micro-blocked-tunnel`, the remaining work is to re-run the canonical scenarios under the three map-sharing regimes so future agents have a clear reference.

1. **Baseline (no sharing):** `long_corridor`, 5 agents, `map_sharing=none`, seeds 13–17. Capture one `gpt-5-mini` run per seed, document collisions + transcripts in a new experiment folder (e.g., `long_corridor_no_share_YYYYMMDDTHHMMSSZ/`).
2. **Radio sync (default):** same maze/seeds but `map_sharing=radio_sync` to show partial knowledge merging. Keep each run under `experiments/long_corridor_radio_sync_.../runs/<run_ts>/`.
3. **Global sharing:** `map_sharing=global` to prove the renderer + client behave when all base maps merge every turn.

For each regime:
- Command template (tmux):  
  ```bash
  PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
    --model gpt-5-mini \
    --maze-preset long_corridor \
    --agents 5 \
    --turns 100 \
    --comm-strategy none \
    --map-sharing <none|radio_sync|global> \
    --seed <13..17> \
    --log-prompts \
    --log-movements \
    --emit-config experiments/<experiment>/runs/$(date -u +%Y%m%dT%H%M%SZ)/config.yaml
  ```
- Artifacts: transcript.jsonl (streamed), episode_stream.jsonl, episode.json, GIF (`python -m llmgrid.vis.gif`), metrics.json.
- Documentation: update this README and the per-experiment README immediately after each run (status, outcome, links to GIF).

Once these nine runs are on disk, we can mark the rebuilt pipeline as fully verified and move on to fresh science.

### Status Legend
- running | complete | failed | abandoned

### Outcome Legend
- useful | not useful | inconclusive | - not determined

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

## Validated Communication Baseline

**Frozen configuration (commit 40de92b):**
- `long_corridor` with 5 agents, azure:gpt-5-mini
- Structured communication with priority clarification: 73% success rate
- Average ~17 messages/run (efficient coordination)
- No deadlocks, no announcement waste

**Prompt configuration:**
- STRUCTURED: Priority rule "LOWEST agent_id MOVES immediately (no announcement needed)"
- FREEFORM: "DEFAULT TO MOVE" + priority rule
- Structured > Freeform (2.2x better) > None (3.6x better)

Use this as baseline for future communication experiments.

## Key Fix: Connection Pool Exhaustion (2025-10-30)

**Problem:** 5-agent runs failed with `APIConnectionError: Connection error` on Azure.

**Root cause:** Default `concurrency_start = len(agent_ids)` meant 5 agents triggered 5 simultaneous `asyncio.run()` calls in separate threads, exhausting Azure connection pool.

**Fix:** Rebuilt `LlmPolicy`/`run_episode` to stay on one event loop (no nested `asyncio.run`), loop-scoped limiter semaphores, and added a per-agent turn history injected into each observation.

**Result:** 5-agent Azure runs now complete with `concurrency_start=5`. History can be surfaced to the LLM; comms-enabled run remained collision-free, whereas a no-radio baseline accumulated 8 collisions.
- `results/metrics.json` now includes `collision_causes`, `hazard_events`, `comments_clamped`, `comments_autofilled`, `no_go_exposures`, and `contended_exposures` for downstream analysis. Adjust aggregators accordingly.
