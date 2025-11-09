# Communication Baseline

**Last updated:** 2025-11-09T15:30:00Z
**Status:** ⏸ paused
**Outcome:** ? inconclusive
**Started:** 2025-11-01

> ⚠️ **Code version tracking:** Runs span three code states with different prompt architectures and simulator behavior. See "Code State by Run" below for which commit each run used. Current findings are mixed and require fresh baseline with unified code state (post-854ee95).

## Question

How do different communication strategies (none, intent-only, negotiation protocol, freeform chat) affect five-agent navigation performance in the `long_corridor` maze when using `azure:gpt-4.1-mini` with the latest loop-aware prompt?

## Why This Matters

The loop-recovery work showed that prompt tweaks alone leave large gaps: weaker models ignore communication entirely while stronger ones over-communicate yet still time out. We need a clean baseline that isolates communication policy modes so we can quantify their impact on completion time, collisions, and coordination quality.

## Setup

- Model: `azure:gpt-4.1-mini`
- Task: 5-agent cooperative navigation on `long_corridor`
- Dataset: Maze preset `long_corridor` (seed 606) with seeds {13, 17, 23} reserved for future replication (current sweep uses seed 13)
- Variables:
  - Communication strategy: `none`, `intent`, `negotiation`, `freeform`
  - Turn budget: 200 turns
- Held constant: agents=5, visibility=1, radio range=2, history_limit=5, loop_guidance=`explore`, logging enabled (prompts + movement), CLI render at 40px cell size, fps 6.

## Code State by Run

| Date Range | Git Commit | Simulator | Prompt Architecture | Affected Runs |
|------------|------------|-----------|---------------------|---------------|
| **Nov 1** | pre-0a0e38d | Buggy (no orientation tracking, broken message aging) | Unified prompt, no strategy scoping | `comm_*_seed13_20251101*` (all Nov 1 runs) |
| **Nov 6-7** | 0a0e38d (2025-11-06 13:20:21) | Fixed (orientation, message seq, idle sprites) | Unified prompt, no strategy scoping | `gpt5_*_seed13_20251107*` |
| **Nov 8+** | 854ee95 (2025-11-08 20:05:23) | Fixed | **Strategy-scoped prompts** (separate blocks per comm mode) | `choke_*_run_20251109*` |

**Key changes:**
- **0a0e38d** (Nov 6): Simulator fixes - agents now track orientation, messages age properly, radio seq assigned by server
- **854ee95** (Nov 8): Prompt scoping - each comm strategy gets tailored prompt blocks instead of unified header
- **e0c7d2b** (Nov 8): Partial GIF rendering added (doesn't affect agent behavior)

**Implication:** Nov 1 runs are unreliable (buggy simulator). Nov 7 runs are more trustworthy but use old unified prompts. Nov 9 runs use current architecture but different maze (choke_points instead of long_corridor).

## Runs

| Run | Started | Status | Notes |
|-----|---------|--------|-------|
| `comm_none_gpt41_seed13_20251101T191609Z` | 2025-11-01 19:16 UTC | ✔ complete | gpt-4.1-mini, comm=none, success=False, turn=200, collisions=10 (agent 6 / wall 4), messages=0 |
| `comm_intent_gpt41_seed13_20251101T192415Z` | 2025-11-01 19:24 UTC | ✔ complete | gpt-4.1-mini, comm=intent, success=False, turn=200, collisions=30 (agent 16 / wall 14), messages=0 |
| `comm_negotiation_gpt41_seed13_20251101T192945Z` | 2025-11-01 19:29 UTC | ✔ complete | gpt-4.1-mini, comm=negotiation, success=False, turn=200, collisions=8 (agent 4 / wall 4), messages=0 |
| `comm_freeform_gpt41_seed13_20251101T193540Z` | 2025-11-01 19:35 UTC | ✔ complete | gpt-4.1-mini, comm=freeform, success=False, turn=200, collisions=27 (agent 12 / wall 15), messages=0 |
| `comm_none_gpt5_seed13_20251101T194121Z` | 2025-11-01 19:41 UTC | ✔ complete | gpt-5-mini, comm=none, success=True, turn=95, collisions=4 (agent only), messages=0 |
| `comm_intent_gpt5_seed13_20251101T195454Z` | 2025-11-01 19:54 UTC | ✔ complete | gpt-5-mini, comm=intent, success=True, turn=70, collisions=0, messages=3 |
| `comm_negotiation_gpt5_seed13_20251101T195510Z` | 2025-11-01 19:55 UTC | ✔ complete | gpt-5-mini, comm=negotiation, success=False, turn=200, collisions=10 (agent only), messages=332 |
| `comm_freeform_gpt5_seed13_20251101T195524Z` | 2025-11-01 19:55 UTC | ✔ complete | gpt-5-mini, comm=freeform, success=True, turn=132, collisions=0, messages=65 |
| `gpt5_none_seed13_20251107T004758Z` | 2025-11-07 00:47 UTC | ✔ complete | gpt-5-mini (Responses API, reasoning_effort=minimal). success=True, turn=54, collisions=4, messages=0 |
| `gpt5_intent_seed13_20251107T004817Z` | 2025-11-07 00:48 UTC | ✔ complete | gpt-5-mini (Responses). success=False (timeout at 100), turn=100, collisions=6, messages=59 |
| `gpt5_negotiation_seed13_20251107T004830Z` | 2025-11-07 00:48 UTC | ✔ complete | gpt-5-mini (Responses). success=True, turn=98, collisions=6, messages=38 |
| `gpt5_freeform_seed13_20251107T004846Z` | 2025-11-07 00:48 UTC | ✔ complete | gpt-5-mini (Responses). success=True, turn=73, collisions=4, messages=27 |
| `gpt5_none_abmarl8103_20251107T210608Z` | 2025-11-07 21:06 UTC | ✔ complete | gpt-5-mini (Responses, reasoning=minimal), comm=none, maze=abmarl_maze_8103, success=False (timeout 100), collisions=22, messages=0 |
| **choke_run_20251109T014905Z** | 2025-11-09 01:49 UTC | ✔ complete | **Post-854ee95**. gpt-5-mini, comm=none, maze=choke_points_comm_test, history=15, success=False (3/5 agents reached goal), turn=200, collisions=18, messages=0 |
| **choke_intent_run_20251109T023446Z** | 2025-11-09 02:34 UTC | ✔ complete | **Post-854ee95**. gpt-5-mini, comm=intent, maze=choke_points_comm_test, history=15, success=False (3/5 reached), turn=200, collisions=0 (✓), messages=2 |
| **choke_negotiation_run_20251109T023446Z** | 2025-11-09 02:34 UTC | ✔ complete | **Post-854ee95**. gpt-5-mini, comm=negotiation, maze=choke_points_comm_test, history=15, success=False (3/5 reached), turn=200, collisions=14, messages=14 |
| **choke_freeform_run_20251109T023446Z** | 2025-11-09 02:34 UTC | ✔ complete | **Post-854ee95**. gpt-5-mini, comm=freeform, maze=choke_points_comm_test, history=15, success=False (4/5 reached ✓), turn=200, collisions=2, messages=2 |
| `choke_map_intel_run_20251109T150841Z` | 2025-11-09 15:08 UTC | ⏹ killed (turn ~100) | **Post-854ee95 + map_intel**. Experimental spatial-intel strategy; halted early; a4/a5 still trapped despite broadcasts |

## Results Summary

### Long Corridor (Post-Fix, Pre-Scoped Prompts - Nov 7)

Using fixed simulator but unified (pre-854ee95) prompts:

| Strategy | Success | LAAS / turns | Messages | Collisions | Code State |
|----------|---------|--------------|----------|------------|------------|
| none | ✓ | 54 | 0 | 4 | 0a0e38d (fixed sim, old prompts) |
| intent | ✗ (timeout) | 100 | 59 | 6 | 0a0e38d (fixed sim, old prompts) |
| negotiation | ✓ | 98 | 38 | 6 | 0a0e38d (fixed sim, old prompts) |
| freeform | ✓ | 73 | 27 | 4 | 0a0e38d (fixed sim, old prompts) |

**Observation:** On easy maze (long_corridor), none was fastest. Communication added overhead without benefit.

### Choke Points (Post-Scoped Prompts - Nov 9)

Using fixed simulator AND scoped (post-854ee95) prompts, history=15:

| Strategy | Agents Reached | Stuck Agents | Collisions | Messages | Code State |
|----------|---------------|--------------|------------|----------|------------|
| none | 3/5 | a4, a5 trapped | 18 | 0 | 854ee95 (scoped prompts) |
| intent | 3/5 | a4, a5 trapped | **0** ✓ | 2 | 854ee95 (scoped prompts) |
| negotiation | 3/5 | a4, a5 trapped | 14 | 14 | 854ee95 (scoped prompts) |
| freeform | **4/5** ✓ | a5 trapped | 2 | 2 | 854ee95 (scoped prompts) |

**Observations:**
- **Intent eliminated collisions** (18→0) with minimal comms
- **Freeform helped one more agent escape** (a4 reached goal)
- **All strategies failed** (timeout at 200 turns, not all agents reached goal)
- **Root cause:** Agents a4/a5 got stuck oscillating in lower-left chamber (coords ~1-5, 5-8). With history_limit=15 and radius=1 vision, they couldn't remember enough to systematically find exits. Communication helped with collision coordination but not spatial exploration.

## Interpretation (Updated 2025-11-09)

### What We Know

1. **Code state matters critically**: Same strategy/maze/seed produced wildly different results across code versions (Nov 1 buggy runs vs Nov 7 fixed-sim vs Nov 9 scoped-prompts)

2. **Long_corridor findings (Nov 7, pre-scoped prompts)**:
   - None was fastest (54 turns)
   - Communication added overhead without collision or completion benefit
   - **BUT**: These used old unified prompts; unclear if scoped prompts would change outcome

3. **Choke_points findings (Nov 9, scoped prompts)**:
   - Intent eliminated collisions (18→0) - clear positive result
   - Freeform helped 1 more agent escape room trap
   - All failed to complete within 200 turns due to spatial exploration limits, not coordination
   - Communication helps with **collision coordination** but not **spatial search**

### What We Don't Know

1. **Does long_corridor + scoped prompts change the story?** Nov 7 runs used old unified prompts. We haven't tested current (post-854ee95) architecture on the easy maze.

2. **Is there a Goldilocks maze?** We have:
   - long_corridor: too easy, agents succeed without coordination
   - choke_points: requires coordination AND spatial search, latter dominates
   - Need: maze that requires coordination but doesn't trap agents in rooms

3. **Nov 1 variance**: Multiple Nov 1 runs with same config produced contradictory results (intent succeeded in 70 turns vs failed in 200 turns). Was this solely due to buggy simulator or is there real non-determinism?

### Current Status

**Inconclusive.** We have evidence communication helps with collisions (choke_points intent: 0 collisions) but mixed/negative results on completion time. Cannot draw firm conclusions because:
- Different code versions across runs
- No apples-to-apples comparison (long_corridor tested with old prompts, choke_points with new prompts)
- Haven't found maze difficulty that isolates coordination benefit

## Hypothesis Register

1. **HYP-20251107A – Chokepoint Advantage:** If we run the same four communication strategies on a maze with multiple single-tile chokepoints and intersecting traffic (e.g., dense maze with alternating one-way corridors), then negotiation/freeform should outperform the silent baseline on LAAS because agents must yield remotely. _Test plan:_ generate a custom preset with `llmgrid.cli.generate_maze --width 24 --height 14 --style maze --connectivity 0.2 --chokepoints 3`, then rerun GPT-5 mini with the 100-turn cap.
2. **HYP-20251107B – Partial Observability Benefit:** If we add hidden switches / key pickups (simulated via artifacts) that only certain agents can see, then structured communications should reduce redundant exploration. _Test plan:_ extend `maze_generator` to tag “switch” cells and add prompt instructions requiring broadcasts when a switch toggles.

## Next Steps (Updated 2025-11-09)

**To establish clean baseline:**

- [ ] **Rerun long_corridor with current code** (post-854ee95 scoped prompts): none, intent, negotiation, freeform with seed 13, history=5 (matching Nov 7 config)
  - This gives apples-to-apples comparison with choke_points runs
  - Eliminates "old prompts" confound from Nov 7 data

- [ ] **Find Goldilocks maze**: Test existing presets (open_sparse, maze_tight, etc.) or design new one that:
  - Has clear paths (no room traps like choke_points)
  - Requires coordination at bottlenecks (unlike easy long_corridor)
  - Predicted candidates: `zipper_comm_hard_v2`, `weave_comm_medium`, `lock_bays_comm_hard` (added in commits 9d53b7e, 9bc2305, ea436fb)

- [ ] **Document artifact locations**: Add paths to GIFs, transcripts, metrics.json for each run in the table above

**For deeper analysis:**

- [ ] Extract and compare prompts actually used in Nov 1 vs Nov 7 vs Nov 9 runs from transcripts
- [ ] Analyze Nov 1 contradictory intent runs (70 turns success vs 200 turns failure) - check transcripts for what differed
- [ ] Quantify "communication prevented collision" events from transcripts (when agent sees INTENT and yields)

**Not prioritizing:**
- Multi-seed runs (17, 23) until we have clean single-seed baseline
- GPT-4.1-mini further testing (consistently fails, focus on GPT-5-mini)

## Appendix: Git Commit Details

**Key commits affecting run validity:**

```
0a0e38d (2025-11-06 13:20:21) - Track orientation, message metadata, and add idle sprite
  - SIMULATOR FIX: Agents now track orientation properly
  - SIMULATOR FIX: Message aging and server-assigned radio seq
  - RENDERING: Added idle sprites
  - Impact: All runs before this have buggy agent state

854ee95 (2025-11-08 20:05:23) - Scope prompts and comm settings per strategy
  - PROMPT ARCHITECTURE CHANGE: Separated unified prompt into strategy-specific blocks
  - Modified: src/llmgrid/prompts.py (+124 lines), src/llmgrid/agent/llm_agent.py
  - Impact: Runs before this used old unified prompt template

e0c7d2b (2025-11-08 20:10:01) - Add partial GIF rendering from live episode streams
  - NEW TOOL: stream_to_episode.py converter
  - Impact: No effect on agent behavior, just visualization
```

**Maze additions (not yet tested in baselines):**
- `ea436fb` (2025-11-08): lock_bays_comm_hard
- `9bc2305` (2025-11-08): weave_comm_medium
- `9d53b7e` (2025-11-08): zipper_comm_hard_v2

**Other choke maze work:**
- `60237fc` (2025-11-09): Parameterized chokepoint generator script
- Today's runs used manually-designed `choke_points_comm_test` (created 2025-11-09)
