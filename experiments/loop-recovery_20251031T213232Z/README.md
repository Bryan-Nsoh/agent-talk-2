# Loop Recovery

**Last updated:** 2025-11-01  
**Status:** ? running  
**Outcome:** -  
**Started:** 2025-10-31

## Question

How do history window length and explicit loop-breaking guidance affect agents’ ability to escape short cycling patterns and reach the goal?

## Why This Matters

During the 5-agent Azure smoke test, agent **a4** oscillated between three cells for 20 turns because it kept following the eastward bearing while the “avoid backtracking” heuristic bounced it back. Without a mitigation signal that survives beyond five turns, adding more history or hazard markers may not fix loopiness. We need to quantify how memory length and prompt rules influence loop dwell time so we can design a reliable escape mechanism.

## Setup

- Model: `azure:gpt-4.1-mini` (fallback: `openrouter:openai/gpt-5-nano` if Azure capacity blocks runs)
- Task: 5-agent cooperative navigation on `long_corridor`
- Dataset: Deterministic maze preset `long_corridor` (seed 606), seeds {13, 17, 23} per configuration
- Variables:
  - History window (`history_limit`): {5 (baseline), 5, 10}
  - Loop guidance: {passive, active (if `history.loop >= 3`, break axis or mark)}
- Held constant: turns=60, visibility=1, radio range=2, comm strategy `intent`, concurrency_start=max=5, comment clamp 25 words, NO_GO TTL=3

### Command Templates (planned)

```
# Baseline — history_limit=5, passive loop guidance
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model azure:gpt-4.1-mini \
  --maze-preset long_corridor \
  --agents 5 \
  --turns 60 \
  --visibility 1 \
  --radio-range 2 \
  --comm-strategy intent \
  --history-limit 5 \
  --loop-guidance passive \
  --log-prompts --log-movements \
  --emit-config <RUN_DIR>/config.yaml

# Active guidance — history_limit=5, loop-guidance=active
... same as baseline but --loop-guidance active

# Long history — history_limit=10, passive guidance
... same as baseline but --history-limit 10
```

Command flags `--history-limit` and `--loop-guidance` will be added before runs begin.

## Runs

| Run | Started | Status | Notes |
|-----|---------|--------|-------|
| `baseline_passive_seed13_20251031T235443Z` | 2025-10-31 23:54 UTC | ✔ complete | history=5, passive; collisions=16, hazard_events=8, loop turns=14 (max loop 9) |
| `baseline_passive_seed17_20251031T235805Z` | 2025-10-31 23:58 UTC | ✔ complete | history=5, passive; collisions=17, hazard_events=11, loop turns=3 (max loop 5) |
| `baseline_passive_seed23_20251101T000123Z` | 2025-11-01 00:01 UTC | ✔ complete | history=5, passive; collisions=14, hazard_events=11, loop turns=8 (max loop 7), messages_sent=1 |
| `active_seed13_20251101T000430Z` | 2025-11-01 00:04 UTC | ✔ complete | history=5, active; collisions=16, hazard_events=8, loop turns=14 (max loop 9) |
| `active_seed17_20251101T000731Z` | 2025-11-01 00:07 UTC | ✔ complete | history=5, active; collisions=6, hazard_events=3, loop turns=4 (max loop 4) |
| `active_seed23_20251101T001032Z` | 2025-11-01 00:10 UTC | ✔ complete | history=5, active; collisions=3, hazard_events=3, loop turns=3 (max loop 5) |
| `long_history_seed13_20251031T233606Z` | 2025-10-31 23:36 UTC | ✔ complete | history=10, passive; collisions=8, hazard_events=0, loop turns=0 |
| `long_history_seed17_20251031T233949Z` | 2025-10-31 23:39 UTC | ✔ complete | history=10, passive; collisions=12, hazard_events=0, loop turns=0 |
| `long_history_seed23_20251031T234331Z` | 2025-10-31 23:43 UTC | ✔ complete | history=10, passive; collisions=4, hazard_events=0, loop turns=0 |
| `explore_seed13_20251101T122211Z` | 2025-11-01 12:22 UTC | ✔ complete | history=5, explore; collisions=21 (BLOCK_AGENT 12 / BLOCK_WALL 9), hazard_events=6, loop turns=15 (max loop 9) |
| `explore5_seed13_20251101T132639Z` | 2025-11-01 13:26 UTC | ✔ complete | history=5, explore; collisions=3 (BLOCK_AGENT only), hazard_events=2, loop turns=2 (max loop 7), messages_sent=58 |

Legacy runs from 2025-10-31 23:06–23:32 UTC (same configurations without loop instrumentation) remain in `runs/` for comparison but are excluded from the analysis below.

## Results

**Baseline (history=5, passive guidance)**
- Seed 13: collisions=16, hazard_events=8, loop turns=14 (max loop=9); still timed out at 60 turns.
- Seed 17: collisions=17, hazard_events=11, loop turns=3 (max loop=5); timed out.
- Seed 23: collisions=14, hazard_events=11, loop turns=8 (max loop=7), messages_sent=1; timed out.

**Active guidance (history=5, loop-guidance=active)**
- Seed 13: collisions=16, hazard_events=8, loop turns=14 (max loop=9); timeout persists.
- Seed 17: collisions=6, hazard_events=3, loop turns=4 (max loop=4); timeout at 60 turns.
- Seed 23: collisions=3, hazard_events=3, loop turns=3 (max loop=5); timeout at 60 turns.

**Long history (history=10, passive guidance)**
- Seeds 13/17/23: collisions=8/12/4, hazard_events=0, loop turns=0 (no loops detected above threshold). Timeouts still occur across all seeds.

**Explore prompt (history=5, loop-guidance=explore)**
- Seed 13: collisions=21 (12 agent, 9 wall), hazard_events=6, loop turns=15 (max loop=9). Agents explicitly tagged comments with `AVOID_LOOP`/`LOOP_BREAK`, but the squad still timed out and never used COMMUNICATE.
- Seed 13 (GPT-5 mini): collisions=3 (all BLOCK_AGENT), hazard_events=2, loop turns=2 (max loop=7), messages_sent=58. Heavy messaging reduced collisions but only a3 reached the goal; others oscillated near mid-corridors despite constant reroute chatter.

**Takeaways**
- The repaired instrumentation now surfaces meaningful loop segments for the 5-turn history configs: loop counters climb as high as 9 despite the active prompt, confirming agents still oscillate.
- Active guidance lowers collision counts on seeds 17 and 23 relative to the passive baseline but does not eliminate loops or timeouts.
- Extending the history window to 10 turns suppresses loop detections entirely, yet agents still fail to coordinate to the goal within 60 turns; additional policy changes are required.
- Exploratory prompts shift behaviour: GPT-4.1 mini acknowledges loops but still slams into walls, while GPT-5 mini talks constantly to coordinate yet leaves three agents short of the goal. Awareness alone does not guarantee progress.
- Communication usage is model-dependent: GPT-4.1 mini never used COMMUNICATE, whereas GPT-5 mini issued 58 packets (INTENT + CHAT) but still failed to shepherd the whole team. We need structural rules (e.g., enforced yields, validated detours) rather than prompt nudges alone.

**GPT-4.1 mini (explore prompt) observations**
- Loop counters now trigger explicit `AVOID_LOOP` comments, but “escape” moves frequently drive into walls or repeat the same axis, converting loops into collisions rather than new progress.
- Action usage remains heavily skewed toward MOVE_E/MOVE_W; only four STAY turns and zero COMMUNICATE actions occurred, so congestion relief never propagates to teammates.
- Wall impacts (12/18 collisions) show the prompt needs guardrails that require selecting a free alternate tile before claiming a loop break.
- Hazard events stayed low (3) and no MARK artifacts were deployed, implying we should couple loop escape with artifact placement to signal congestion.

**GPT-5 mini (explore prompt) observations**
- Messages: 58 out of 60 turns contained a COMMUNICATE action (INTENT + CHAT). Agents negotiate yields and announce reroutes, often while holding position, which slashes collisions to three.
- Loop segments are short (two one-turn segments) but loop counters still spike (a1 max loop 7) because each detour resets the five-turn history. Agents oscillate near mid-corridors and never shepherd the full squad to the goal.
- Message metadata (`seq`) resets repeatedly (e.g., a2 emits seq 17 then 0). Downstream consumers cannot rely on monotonic ids; we should stabilise counters in the policy.
- Only one MARK was placed (a5, turn 50). Congestion warnings still depend on chat spam, and other agents do not treat the lone NO_GO as actionable guidance.

## Interpretation

- Restored instrumentation plus the exploratory prompt make loop awareness visible, but behaviour diverges by model: GPT-4.1 mini converts loop-breaking into wall bumps, whereas GPT-5 mini over-indexes on reroute chatter and still hesitates to finish the maze. Awareness alone is not rescuing team progress.
- Communication is not the silver bullet. GPT-4.1 mini ignored the channel entirely; GPT-5 mini flooded it (58 messages) yet still stranded four agents. We need environment-level rules—e.g., enforced yielding, validated alternate paths, monotonic message ids—to make broadcast intent actionable rather than cosmetic.
- Longer history windows (10 turns) suppress loop counters but do not improve success rate, implying that just exposing more context without stricter decision rules doesn’t solve congestion.
- Future mitigations should combine prompt instructions with structural constraints: validate loop-escape moves against free-space checks, couple loop alerts to mandatory communication or MARK drops, stabilise message sequencing, and add a guard that rejects loop-breaking moves that target walls.

## Decision

Pending — select default history length and loop rule once data collected.

- [ ] Implement configurable history window and loop-guidance prompt toggle
- [ ] Build loop-analysis script (detect collision-free 2-cell cycles ≥3 turns)
- [x] Run baseline (history=5, passive guidance) on seeds 13, 17, 23
- [x] Run active guidance (history=5, active) on seeds 13, 17, 23
- [x] Run long-history (history=10, passive) on seeds 13, 17, 23
- [x] Run explore prompt (history=5, explore) on seed 13 (GPT-4.1 mini)
- [x] Complete explore prompt run on GPT-5 mini and document findings
- [ ] Compare loop dwell metrics across configurations and update policy defaults
- [ ] Prototype mandatory communication or artifact drop when loop ≥ 2 and rerun seed 13
- [ ] Stabilise COMMUNICATE message sequencing and add free-tile validation for loop-escape moves
