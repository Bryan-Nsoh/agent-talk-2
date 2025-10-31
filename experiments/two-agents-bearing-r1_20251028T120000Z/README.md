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
| `azure_history_comms_20251031T163231Z` | gpt-4.1-mini (Azure) | 5 | 60 | ✅ complete | Timed out, history enabled, 0 collisions, 0 comms |
| `azure_history_no_comms_20251031T163443Z` | gpt-4.1-mini (Azure) | 5 | 60 | ✅ complete | Timed out, radio disabled, 8 collisions |

## Current Status

**Latest runs:**
- `azure_history_comms_20251031T163231Z` — 5 agents, concurrency=5, radio range 2. History payload enabled; run timed out at 60 turns with 0 collisions.
- `azure_history_no_comms_20251031T163443Z` — identical setup but radio range 0 (no delivery). Timed out at 60 turns with 8 collisions.
- No active runs at the moment.

## Key Findings

**Async refactor + history context (2025-10-30/31):**
- Problem (old code): multi-agent runs stalled after turn 0 when concurrency >1; agents lacked structured memory of prior turns.
- Fix: Policy/driver now fully async with loop-scoped concurrency control, and observations expose the last five decisions/messages in `history`.
- Result: 5-agent Azure runs complete a full 60-turn budget even at concurrency 5. History is available for prompting; comms-enabled run remained collision-free, while disabling radio increased collisions (8) and still failed to reach the goal.

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
