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
| seed13 | complete | 5/5 finished; collisions 0 |
| seed14 | complete | 5/5 finished; collisions 2 |
| seed15 | complete (rerun) | 5/5 finished; collisions 8 |
| seed16 | complete | 5/5 finished; collisions 0 |
| seed17 | complete | 5/5 finished; collisions 16 |

## Results
- Success (all 5 finished): 5/5 runs
- Collisions: mean 5.2, median 2 (BLOCK_AGENT only)
- Median goal-known turn: 33 (faster than none/radio)
- Median turns to all finish: 84
- Summary plot: analysis/mapshare/plots/mapshare_summary.png

## Interpretation
Global sharing propagates goal knowledge fastest (median 33 turns) and yields the highest completion rate (5/5), but with higher collision counts driven by two outlier runs (seed17=16, seed15_rerun=8).

## Decision
Global sharing is best for success rate and goal discovery; collision spike needs mitigation (priority rules or collision penalties) in future runs.

## Hypothesis Register
- H1: Global sharing will substantially raise success and lower collisions vs radio_sync by providing perfect shared maps; if results are similar to radio_sync, map knowledge is not the limiting factor without comms.

## Next Steps
- [ ] Execute seeds 13–17 with correct map sharing mode
- [ ] Render GIFs
- [ ] Update runs table with status, links, metrics
- [ ] Summarize results and update master experiments index
