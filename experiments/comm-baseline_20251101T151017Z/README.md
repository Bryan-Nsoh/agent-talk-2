# Communication Baseline

**Last updated:** 2025-11-06T19:25:00Z  
**Status:** ? running  
**Outcome:** -  
**Started:** 2025-11-01

> ⚠️ **Legacy data:** These runs were recorded before the 2025-11-06 simulator fixes (orientation tracking, message aging, server-assigned radio `seq`, idle sprites). Treat all numbers below as provisional; every configuration must be rerun under the patched engine.

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

## Results (pre-fix engine)

- `azure:gpt-4.1-mini` still times out on the 200-turn cap for all comm modes, with collisions ranging from 8 (negotiation) to 30 (intent). Multiple agents now finish, confirming the goal is no longer blocked by earlier arrivals.
- `azure:gpt-5-mini` succeeds under `none` (turn 95, 4 collisions), `intent` (turn 70, 0 collisions), and `freeform` (turn 132, 0 collisions, 65 CHAT messages). The negotiation protocol remains problematic: despite 332 structured messages, two agents linger near the exit and the run times out at 200 turns.
- Loop summaries show GPT-5 mini aggressively placing NO_GO markers and broadcasting reroutes near congestion, while GPT-4.1 mini rarely communicates even with channels enabled.

## Interpretation (awaiting rerun confirmation)

- The environment fix did its job: agents now march onto the goal without artificial blocking. GPT-4.1 mini still lacks the planning depth to finish within 200 turns, whereas GPT-5 mini reliably clears the maze except when overloaded with the negotiation schema.
- Negotiation produces intense chatter but insufficient commitment to a shared plan; the protocol likely needs additional guardrails (e.g., enforced move after a certain number of yields) or more capable reasoning to succeed.

## Decision

- Treat GPT-4.1-mini as a baseline for “reasoning-limited” performance and focus further analysis on GPT-5-mini runs, especially the negotiation failure. Use these results to design prompt/policy tweaks and structural experiments.

## Next Steps

- [ ] Rerun seed 13 for all strategies with the corrected engine and refreshed sprites; regenerate metrics, transcripts, and GIFs.
- [ ] Repeat post-fix runs for seeds 17 and 23 (gpt-5-mini and gpt-4.1-mini) to rebuild the comparison table.
- [ ] Re-evaluate negotiation after reruns; investigate enforced follow-through or throttling if issues persist.
- [x] Run communication strategies (none, intent, negotiation, freeform) on seed 13 (gpt-4.1-mini and gpt-5-mini) — legacy baseline.
- [x] Aggregate LAAS, collisions, and message volume per strategy — legacy baseline.
