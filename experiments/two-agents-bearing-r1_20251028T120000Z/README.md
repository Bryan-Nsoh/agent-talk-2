# Two Agents – Bearing Sensor – Radius 1

**Last updated:** 2025-10-30
**Status:** ✅ complete
**Outcome:** ❌ not useful (timed out)
**Started:** 2025-10-28

## Question

Can LLM agents with partial observability (radius-1 vision, bearing sensor, range-2 comms) navigate cooperatively to a shared goal on curated maze layouts?

## Setup

- **Maze:** `long_corridor` (seed 606, 30×10 maze with wide horizontal corridors)
- **Agents:** 2-5 agents with radius-1 visibility
- **Sensors:** Bearing-based goal sensor (FAR/CLOSE), local patch observation
- **Communication:** Range-2 radio, max 1 message/turn, 96 chars
- **Models tested:** OpenRouter `gpt-5-nano`, Azure `gpt-5-mini`

## Completed Runs

| Run | Model | Agents | Turns | Status | Result |
|-----|-------|--------|-------|--------|--------|
| `long-corridor-nano_20251029T211939Z` | gpt-5-nano | 2 | 45 | ✅ complete | Goal reached, 0 collisions, 0 messages |
| `azure_parallel_fix_20251030T215123Z` | gpt-4.1-mini (Azure) | 5 | 60 | ✅ complete | Goal not reached (timed out), 0 collisions, 0 comms |

## Current Status

**Latest run:** `azure_parallel_fix_20251030T215123Z`
- 5 agents on `long_corridor`, Azure `gpt-4.1-mini`, 60-turn budget
- Concurrency window sustained at 5 without stalls after async refactor
- Timed out at 60 turns (0 collisions, no comms) — requires strategy tweaks for goal capture
- No active runs at the moment

## Key Findings

**Async refactor (2025-10-30):**
- Problem: Multi-agent runs stalled after turn 0 when concurrency >1.
- Root cause: mixing `asyncio.run()` inside thread workers with shared async locks caused cross-loop deadlocks.
- Fix: Policy/driver fully async (`run_episode` keeps single loop) and rate limiter semaphores are loop-scoped.
- Result: 5-agent Azure run completed 60 turns with `concurrency_start=5` without hanging.

**Baseline Performance:**
- 2 agents: Goal reached in 45 turns (OpenRouter gpt-5-nano).
- 5 agents: Stable execution on Azure `gpt-4.1-mini` at full concurrency, but strategy timed out → need improved policy/planning.

## Alternate Presets Available

When ready to test other environments:
- `open_sparse` — 20×12 grid, 12% scatter, light congestion
- `open_dense` — 20×12 grid, 25% scatter, tighter routes
- `maze_tight` — 21×13 classic maze with minimal shortcuts
- `maze_loops` — 21×13 maze with many alternate paths
- `mixed_medium` — 24×14 hybrid scatter/corridor layout

Preview PNGs: `experiments/presets/batch/`

## Next Steps

- [ ] Analyse 60-turn Azure run to improve goal-seeking behaviour
- [ ] Re-run with enhanced policy (comm plans, artifact use) to beat 60-turn cap
- [ ] Sweep other presets (`open_sparse`, `maze_tight`, etc.) with new async stack
- [ ] Compare performance across different agent counts (2, 5, 10) once policy updated
