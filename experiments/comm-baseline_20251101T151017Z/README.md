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
| `comm_none_gpt5_seed13_20251101T164742Z` | 2025-11-01 16:47 UTC | ✔ complete | model=gpt-5-mini; comm=none; LAAS=200 (timeout); collisions=26 (agent 35); messages=0 |
| `comm_intent_gpt5_seed13_20251101T170541Z` | 2025-11-01 17:05 UTC | ✔ complete | model=gpt-5-mini; comm=intent; LAAS=200 (timeout); collisions=25 (agent 42); messages=268 |
| `comm_negotiation_gpt5_seed13_20251101T170541Z` | 2025-11-01 17:05 UTC | ✔ complete | model=gpt-5-mini; comm=negotiation; LAAS=200 (timeout); collisions=16 (agent 28); messages=427 |
| `comm_freeform_gpt5_seed13_20251101T170541Z` | 2025-11-01 17:05 UTC | ✔ complete | model=gpt-5-mini; comm=freeform; LAAS=200 (timeout); collisions=19 (agent 30); messages=367 |

## Results

- All four strategies timed out at 200 turns with only one agent finishing in any run; LAAS therefore equals the turn cap for each arm.
- Collision volume remained high for every configuration (52–69 total events). Negotiation produced the most collisions (69) despite modest messaging, while `none` and `freeform` both stayed above 50.
- Messaging behaviour differentiated the arms: `none` and `freeform` produced zero outbound messages; `intent` issued 6 INTENT packets; `negotiation` sent only 3 protocol messages. GPT-4.1 mini frequently ignored the richer comm channels even when available.
- Loop analyzer outputs (see `loop_summary.json` in each run) continue to show repeated short oscillations (max loop counters 5–7) without successful escape to the goal.

- For GPT-5 mini, all strategies still hit the 200-turn cap, but collisions dropped sharply (16–26). Negotiation and freeform generated aggressive communication bursts (427 and 367 messages respectively) while baseline (`none`) stayed silent.
- GPT-5 mini uses MARK artifacts and repeated CHAT broadcasts during loops, yet still stalls short of the goal—highlighting that communication saturation alone does not guarantee throughput on `long_corridor` with visibility 1.

## Interpretation

- `azure:gpt-4.1-mini`: enabling structured communication did not reduce collisions or accelerate completion; the model either ignored the channel (freeform/none) or used it sparingly (intent/negotiation) while continuing to loop.
- `azure:gpt-5-mini`: stronger reasoning slashes collisions and saturates communication channels, yet the squad still times out. Negotiation/freeform variants exchanged hundreds of messages but could not coordinate a full finish—pointing to structural limits (visibility=1, latency) rather than pure reasoning failure.
- Negotiation schema seems to help only when the model already exploits it; GPT-4.1 mini struggled with extra complexity, whereas GPT-5 mini used it heavily without translating chatter into completions.

## Decision

- Keep the communication strategies instrumented but treat GPT-4.1 mini results as a lower bound; future evaluations should focus on higher-capability models (GPT-5 mini) plus structural tweaks (visibility, enforced reroutes) to translate intense coordination into completions.

## Next Steps

- [ ] Run communication strategies (none, intent, negotiation, freeform) on seed 13.
- [ ] Aggregate LAAS, collisions, and message volume per strategy.
- [ ] Expand to additional seeds (17, 23) and models (gpt-5-mini) once baseline established.
