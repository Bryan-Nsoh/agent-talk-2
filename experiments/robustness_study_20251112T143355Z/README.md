# robustness_study_20251112T143355Z

**Last updated:** 2025-11-12T20:34:35Z  
**Status:** running  
**Outcome:** -  
**Started:** 2025-11-12

## Question

Does the structured communication advantage we validated on seed 13 persist when we change the agent starting positions (seeds 13–17), or was the earlier win a configuration-specific fluke?

## Why This Matters

Seed 14 immediately exposed a 21× drop in structured message volume (one message instead of the canonical ~21). That either means the latest code silently regressed—throttling radios and breaking the baseline—or seed 14’s spawn geometry simply requires less chatter. We need hard evidence before we extend the communication write-up, so we are replicating all three strategies across new seeds and re-running seed 13 under the exact code commit that produced the anomaly.

## Setup

- Model: `azure:gpt-5-mini` via Unified LLM client (OpenRouter disabled for this study).
- Task: `llmgrid.cli.poc_two_agents` configured for the `long_corridor` maze with 5 agents, 100-turn budget, visibility 1, radio range 2 (structured/freeform) or 0 (none).
- Dataset: Deterministic maze preset (`seed=606`) with varied agent spawn seeds 13–17.
- Variables: `--comm-strategy` ∈ {structured, freeform, none} × seed id × three stochastic replications per seed.
- Held constant: Maze topology, loop_guidance=`explore`, bearing sensors (no noise), history_limit=5, retry/backoff parameters, logging of transcript + movement stream.

## Runs

| Run | Started (UTC) | Status | Notes |
|-----|---------------|--------|-------|
| [seed14_structured_20251112T160750Z](./runs/seed14_structured_20251112T160750Z/) | 2025-11-12 16:07 | failed | False start; aborted at turn 5 before logging, produced 0 messages—discarded from replication counts. |
| [seed14_structured_20251112T175321Z](./runs/seed14_structured_20251112T175321Z/) | 2025-11-12 17:53 | complete | Run 1 (structured) finished 100 turns with only 1 message and timeout failure. |
| [seed14_freeform_20251112T175323Z](./runs/seed14_freeform_20251112T175323Z/) | 2025-11-12 17:53 | complete | Run 1 (freeform) finished 100 turns, 0 messages, timeout failure. |
| [seed14_none_20251112T175325Z](./runs/seed14_none_20251112T175325Z/) | 2025-11-12 17:53 | complete | Run 1 (none) finished 100 turns, 0 messages, timeout failure. |
| [seed14_structured_20251112T193605Z](./runs/seed14_structured_20251112T193605Z/) | 2025-11-12 19:36 | running | Run 2 (structured). PID 23118/23115 still active; streaming transcript only (no metrics yet). |
| [seed14_freeform_20251112T193607Z](./runs/seed14_freeform_20251112T193607Z/) | 2025-11-12 19:36 | running | Run 2 (freeform). PID 23120/23116 active. |
| [seed14_none_20251112T193609Z](./runs/seed14_none_20251112T193609Z/) | 2025-11-12 19:36 | running | Run 2 (none). PID 23119/23117 active. |
| [seed14_structured_20251112T193613Z](./runs/seed14_structured_20251112T193613Z/) | 2025-11-12 19:36 | running | Run 3 (structured). PID 23199/23193 active. |
| [seed14_freeform_20251112T193615Z](./runs/seed14_freeform_20251112T193615Z/) | 2025-11-12 19:36 | running | Run 3 (freeform). PID 23197/23194 active. |
| [seed14_none_20251112T193617Z](./runs/seed14_none_20251112T193617Z/) | 2025-11-12 19:36 | running | Run 3 (none). PID 23198/23195 active. |
| [seed13_structured_rerun_20251112T195905Z](./runs/seed13_structured_rerun_20251112T195905Z/) | 2025-11-12 19:59 | running | Regression check: structured strategy on canonical seed 13 under current commit. PID 26165/26160 active. |

## Results

- Seed 14 Run 1 (structured) timed out at 100 turns with 1 message, 19 collisions, and 0 finished agents—evidence that the communication layer is either under-utilised or suppressed by the current maze state.
- Seed 14 Run 1 (freeform) and Run 1 (none) also timed out with 0 finished agents and 0 messages; freeform logged 17 collisions, no-go beacons triggered 8 times, none hit 28 collisions. The collapse is therefore not unique to structured comms but structured still only emitted a single radio packet.
- The seed14 false start at 16:07 UTC terminated after five turns and provides no usable data; it confirms the CLI and logging path but cannot inform the analysis.
- No other runs have produced `metrics.json` yet; the tmux-free background processes (listed above) continue to append `episode_stream.jsonl` and transcripts, so we must wait for completion before interpreting seeds 14 run 2/3 or the seed 13 rerun.

Full artifacts for completed runs live under each run directory (`config.yaml`, `results/episode.json`, `results/metrics.json`, transcripts, and movement streams).

## Interpretation

The cross-seed robustness hypothesis is still unproven. The only completed replicated seed (14) shows uniformly poor outcomes with almost zero radio traffic across all strategies, which tells us that the message collapse is not specific to structured comms. However, until the seed 13 rerun finishes we cannot rule out a regression: if the canonical seed also drops to ~1 message we broke the baseline; if it returns to ~21 messages, then seed-dependent geometry is the culprit and we can proceed to seeds 15–17.

## Decision

Pending the seed 13 rerun and completion of the remaining seed 14 replications. No strategic change yet.

## Next Steps

- [ ] Let the six seed 14 Run 2/Run 3 processes finish and capture metrics/GIFs.
- [ ] Monitor the seed 13 structured rerun for message volume and compare against the historic ~21 baseline.
- [ ] Once seed 14 analysis is stable, schedule seeds 15–17 (3×3 matrix each) to reach the planned 45-run robustness grid.
- [ ] Generate summary plots (messages vs. success, collisions) once we have ≥2 seeds worth of data so we can visualise spread.

## Hypothesis Register

1. **H1 – Structured radio usage remains high (≥15 messages/run) across seeds 13–17.**  
   *Status:* pending.  
   *Test plan:* run structured strategy three times per seed and record `messages_sent` plus arrival counts.  
   *Metrics:* `results/metrics.json.messages_sent`, #agents finished.  
   *Linked runs:* seed14 structured run 1 (complete), run 2/3 (running), seed13 rerun (running).  
   *Outcome:* TBD.

2. **H2 – The seed 14 message collapse stems from code regression (not spawn geometry).**  
   *Status:* pending critical check.  
   *Test plan:* rerun seed 13 structured under commit `HEAD` and compare message volume to historical 21 ± 3 window.  
   *Metrics:* `messages_sent`, `transcript` inspection for suppressed radio actions.  
   *Linked runs:* seed13_structured_rerun_20251112T195905Z (running).  
   *Outcome:* TBD.

3. **H3 – Strategy ranking (structured > freeform > none) holds across seeds 13–17.**  
   *Status:* not started.  
   *Test plan:* complete the planned 45 runs, compute mean success rate and messages per strategy, run ANOVA on collision counts.  
   *Metrics:* success flag, collisions, messages, hazard events.  
   *Linked runs:* will include all seeds once executed.  
   *Outcome:* TBD.

