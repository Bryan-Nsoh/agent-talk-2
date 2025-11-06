# Two Agents – Bearing Sensor – Radius 1

**Last updated:** 2025-11-06T19:25:00Z
**Status:** 🔄 running
**Outcome:** ❌ not useful (timed out)
**Started:** 2025-10-28

> ⚠️ **Legacy data:** These runs predate the 2025-11-06 engine fixes (orientation tracking, message aging, server-assigned radio `seq`, idle sprites). Re-run all baselines under the corrected simulator before trusting the metrics below.

## Question

Can LLM agents with partial observability (radius-1 vision, bearing sensor, range-2 comms) navigate cooperatively to a shared goal on curated maze layouts?

## Setup

- **Maze:** `long_corridor` (seed 606, 30×10 maze with wide horizontal corridors)
- **Agents:** 2-5 agents with radius-1 visibility
- **Sensors:** Bearing-based goal sensor (FAR/CLOSE), local patch observation
- **Communication:** Range-2 radio, max 1 message/turn (25-word comment clamp with status prefix), strategies `none` / `intent` / `negotiation` / `freeform`
- **Models tested:** OpenRouter `gpt-5-nano`, Azure `gpt-5-mini`

## Completed Runs

| Run | Model | Agents | Turns | Status | Result |
|-----|-------|--------|-------|--------|--------|
| `long-corridor-nano_20251029T211939Z` | gpt-5-nano | 2 | 45 | ✅ complete | Goal reached, 0 collisions, 0 messages |
| `azure_parallel_fix_20251030T215123Z` | gpt-4.1-mini (Azure) | 5 | 60 | ✅ complete | Timed out, 0 collisions, 0 comms |
| `azure_history_comms_20251031T165135Z` | gpt-4.1-mini (Azure) | 5 | 0 | ❌ failed | Script bug (run_history_comms argv mismatch) |
| `azure_history_comms_20251031T165305Z` | gpt-4.1-mini (Azure) | 5 | 60 | ✅ complete | Timed out, history enabled, 43 collisions, 0 comms |
| `azure_history_no_comms_20251031T165744Z` | gpt-4.1-mini (Azure) | 5 | 60 | ✅ complete | Timed out, radio disabled, 5 collisions, a1/a3 reached goal |

> ⚠️ **Legacy data:** All runs occurred before the 2025-11-06 simulator fixes (orientation tracking, message ages, server-assigned radio `seq`, idle sprites). Treat the findings below as historical only; both 2-agent and 5-agent baselines must be rerun with the corrected engine.

## Current Status

**Latest runs (Oct 31, 2025, UTC):**
- `azure_history_comms_20251031T165305Z` — 5 agents, history enabled, radio range 2. Timed out at 60 turns, 43 collisions, no comms emitted. Spawn/overlap rendering verified from `episode_stream.jsonl`.
- `azure_history_no_comms_20251031T165744Z` — radio range 0 baseline. Timed out at 60 turns with 5 collisions; agents `a1` and `a3` reached the goal but the squad failed to converge within the budget.
- `azure_history_comms_20251031T165135Z` — run aborted immediately (IndexError in scripting wrapper). Retained as failure artifact.

**Instrumentation refresh (Oct 31, late UTC):**
- Added per-agent `last_move_outcome`, congestion bitmasks, and five-turn `history` windows to every observation (persisted across checkpoints).
- Comments now auto-prefix with status tokens and clamp to 25 words; blank comments fall back to the status for consistent logging.
- Collision cells drop short-lived `NO_GO` artifacts (traffic cones) that surface as `NO_GO` adjacency hints and gray dots in GIFs.
- `results/metrics.json` now captures `collision_causes`, `hazard_events`, comment clamp counts, and exposure tallies; CLI exposes `--comm-strategy` to toggle none/intent/negotiation/freeform.
- Azure smoke test (intent strategy, seed 13) revealed an A4 oscillation between (12,7)–(13,7)–(13,8) for turns 41‑60: the agent keeps chasing the eastward bearing, but the “avoid backtracking” heuristic bounces it between two safe cells instead of letting it push west/south twice to exit the L-shaped cul-de-sac. Loop counters reached 3 without altering behaviour, and no NO_GO tiles were emitted because collisions never happened—highlighting that history signals alone aren’t changing the LLM’s reasoning yet.

## Key Findings (pre-fix engine)

**Async refactor + history context (2025-10-30/31):**
- Problem (old code): multi-agent runs stalled after turn 0 when concurrency >1; agents lacked structured memory of prior turns.
- Fix: Policy/driver now fully async with loop-scoped concurrency control, and observations expose the last five decisions/messages in `history`.
- Result: 5-agent Azure runs now finish the full 60-turn budget. History-enabled comms run (165305Z) accumulated 43 collisions despite radio access; the no-radio baseline (165744Z) stayed under 5 collisions but still timed out.

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

- [ ] Rerun 2-agent and 5-agent baselines with the patched engine (intent vs none), regenerate metrics/transcripts/GIFs.
- [ ] Verify hazard overlays and idle sprites in new visualisations.
- [ ] Analyse post-fix `collision_causes`, `hazard_events`, and comment usage; adjust prompt guidance as needed.
- [ ] Expand to other presets (`open_sparse`, `maze_tight`, etc.) and larger agent counts once post-fix baselines are validated.
- [ ] Augment policy guidance to react when `history.loop` climbs (e.g., force alternate axes or mark cul-de-sacs) so agents like A4 break L-shaped traps instead of oscillating indefinitely.
