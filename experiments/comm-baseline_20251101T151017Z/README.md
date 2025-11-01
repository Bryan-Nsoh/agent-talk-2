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
| _metrics pending_ | - | ☑ | Reruns completed; summarizing metrics now |

## Results

- Earlier runs (recorded before fixing the goal-blocking bug) have been discarded; fresh results will be posted once the rerun completes.

## Interpretation

- Pending rerun.

## Decision

- Await rerun after bug fix; reassess once new baselines are in.

## Next Steps

- [ ] Run communication strategies (none, intent, negotiation, freeform) on seed 13.
- [ ] Aggregate LAAS, collisions, and message volume per strategy.
- [ ] Expand to additional seeds (17, 23) and models (gpt-5-mini) once baseline established.
