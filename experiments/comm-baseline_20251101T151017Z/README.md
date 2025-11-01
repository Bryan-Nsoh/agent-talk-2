# Communication Baseline

**Last updated:** 2025-11-01  
**Status:** ? running  
**Outcome:** -  
**Started:** 2025-11-01

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
| `comm_none_seed13_20251101T151044Z` | 2025-11-01 15:10 UTC | ✔ complete | comm=none; LAAS=200 (timeout); collisions=57 (agent 48 / wall 25); messages=0 |
| `comm_intent_seed13_20251101T151925Z` | 2025-11-01 15:19 UTC | ✔ complete | comm=intent; LAAS=200 (timeout); collisions=58 (agent 52 / wall 24); messages=6 |
| `comm_negotiation_seed13_20251101T152449Z` | 2025-11-01 15:24 UTC | ✔ complete | comm=negotiation; LAAS=200 (timeout); collisions=69 (agent 60 / wall 25); messages=3 |
| `comm_freeform_seed13_20251101T153017Z` | 2025-11-01 15:30 UTC | ✔ complete | comm=freeform; LAAS=200 (timeout); collisions=52 (agent 56 / wall 16); messages=0 |

## Results

- All four strategies timed out at 200 turns with only one agent finishing in any run; LAAS therefore equals the turn cap for each arm.
- Collision volume remained high for every configuration (52–69 total events). Negotiation produced the most collisions (69) despite modest messaging, while `none` and `freeform` both stayed above 50.
- Messaging behaviour differentiated the arms: `none` and `freeform` produced zero outbound messages; `intent` issued 6 INTENT packets; `negotiation` sent only 3 protocol messages. GPT-4.1 mini frequently ignored the richer comm channels even when available.
- Loop analyzer outputs (see `loop_summary.json` in each run) continue to show repeated short oscillations (max loop counters 5–7) without successful escape to the goal.

## Interpretation

- At least for `azure:gpt-4.1-mini`, enabling structured communication does not reduce collisions or accelerate completion on `long_corridor`; the model either ignores the channel (freeform) or uses it sparingly (intent/negotiation) while continuing to loop.
- The negotiation protocol appears hardest for the model: despite access to richer message types, it sent only three packets and actually incurred the highest collision count, suggesting the extra schema adds cognitive load without payoff.
- Freeform messaging collapsed back to silence—reinforcing the need for concrete rules or stronger models if we want the channel utilized.

## Decision

- Keep the communication strategies instrumented but treat GPT-4.1 mini results as a lower bound; the model needs stronger guardrails or a higher-capability variant before communication policy changes can be judged effective.

## Next Steps

- [ ] Run communication strategies (none, intent, negotiation, freeform) on seed 13.
- [ ] Aggregate LAAS, collisions, and message volume per strategy.
- [ ] Expand to additional seeds (17, 23) and models (gpt-5-mini) once baseline established.
