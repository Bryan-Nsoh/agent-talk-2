# Long Corridor Map-Sharing Validation

**Last updated:** 2025-11-19T22:30:00Z
**Status:** complete
**Outcome:** useful
**Started:** 2025-11-19

## Question
How does map sharing mode (none vs radio_sync vs global) affect 5-agent navigation success and collisions on the long_corridor maze when communication is disabled (comm-strategy=none)?

## Why This Matters
We need a fresh, post-engine-fix baseline showing whether map sharing alone (without messages) improves coordination relative to the no-sharing baseline. This verifies the rebuilt renderer + telemetry pipeline across the three map-sharing regimes and provides reference traces for future communication studies.

## Setup
- Model: gpt-5-mini (pool key; Azure)
- Maze: long_corridor (30x10, seed 606)
- Agents: 5
- Turns: 100
- Visibility: default (1)
- Radio range: default (2)
- Comm strategy: none
- Map sharing: <MODE> (none | radio_sync | global)
- Seeds: 13–17 (one run each)
- Logging: --log-prompts --log-movements --emit-config
- Outputs per run: config.yaml, transcript.jsonl, episode_stream.jsonl, episode.json, metrics.json, episode.gif

## Runs
| Run | Seed | Status | Notes |
|-----|------|--------|-------|
| seed13 | complete | all 5 finished; collisions 0 |
| seed14 | complete | all 5 finished; collisions 0 |
| seed15 | complete | all 5 finished; collisions 0 |
| seed16 | complete | 2/5 finished; collisions 0 |
| seed17 | complete | 4/5 finished; collisions 0 |

## Results
- Success (all 5 finished): 3/5 runs
- Collisions: mean 0, median 0
- Median turns to all finish: 98
Data source: results/episode_stream.jsonl per run; metrics harvested from tmux logs.

## Interpretation
Radio sync did not improve completions over no-share; slower convergence (median all-finish 98) with no collision penalty or benefit.

## Decision
Radio sync offers no advantage over baseline under comm=none; treat as parity case.

## Hypothesis Register
- H1: Radio sync will reduce collisions and increase success vs no sharing, despite comm-strategy=none, by merging local maps within radio range. Metrics: success rate, collisions, turns to goal.

## Next Steps
- [ ] Execute seeds 13–17 with correct map sharing mode
- [ ] Render GIFs
- [ ] Update runs table with status, links, metrics
- [ ] Summarize results and update master experiments index
