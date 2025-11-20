# Map-Sharing Modes: Long Corridor Validation

**Last updated:** 2025-11-20T20:30:00Z
**Status:** complete (replication phase - 45/45 complete)
**Outcome:** useful
**Started:** 2025-11-19

## Question

How does map sharing mode (none vs radio_sync vs global) affect 5-agent navigation success and coordination on the long_corridor maze when communication is disabled?

## Setup

- Model: gpt-5-mini (model pool key)
- Maze: long_corridor (30×10, seed 606)
- Agents: 5
- Turns: 100
- Visibility: 1 (Manhattan radius)
- Radio range: 2 (for radio_sync mode)
- Comm strategy: none (no messages)
- Map sharing: **none | radio_sync | global** (3 conditions)
- Seeds: 13-17 (5 seeds)
- Replicas: 3 (all complete)
- **Total runs:** 3 replicas × 5 seeds × 3 modes = **45 runs**
- CLI: `run_preset`
- Outputs per run: config.yaml, transcript.jsonl, episode_stream.jsonl, episode.json

## Runs

**Status Legend:** ✓complete | ⏳running | -pending

### None Mode (Baseline)
| Seed | Rep1 | Rep2 | Rep3 |
|------|------|------|------|
| 13 | ✓ [seed13](./none/runs/seed13_20251119T203622Z/) | ✓ [none_seed13_rep2](./none/runs/none_seed13_rep2_20251120T164100Z/) | ✓ [none_seed13_rep3](./none/runs/none_seed13_rep3_20251120T164100Z/) |
| 14 | ✓ [seed14](./none/runs/seed14_20251119T203622Z/) | ✓ [none_seed14_rep2](./none/runs/none_seed14_rep2_20251120T164100Z/) | ✓ [none_seed14_rep3](./none/runs/none_seed14_rep3_20251120T164100Z/) |
| 15 | ✓ [seed15](./none/runs/seed15_20251119T203622Z/) | ✓ [none_seed15_rep2](./none/runs/none_seed15_rep2_20251120T164100Z/) | ✓ [none_seed15_rep3](./none/runs/none_seed15_rep3_20251120T164100Z/) |
| 16 | ✓ [seed16](./none/runs/seed16_20251119T203623Z/) | ✓ [none_seed16_rep2](./none/runs/none_seed16_rep2_20251120T164100Z/) | ✓ [none_seed16_rep3](./none/runs/none_seed16_rep3_20251120T164100Z/) |
| 17 | ✓ [seed17](./none/runs/seed17_20251119T203623Z/) | ✓ [none_seed17_rep2](./none/runs/none_seed17_rep2_20251120T164100Z/) | ✓ [none_seed17_rep3](./none/runs/none_seed17_rep3_20251120T164100Z/) |

### Radio_sync Mode
| Seed | Rep1 | Rep2 | Rep3 |
|------|------|------|------|
| 13 | ✓ [seed13](./radio_sync/runs/seed13_20251119T203623Z/) | ✓ [radio_sync_seed13_rep2](./radio_sync/runs/radio_sync_seed13_rep2_20251120T171411Z/) | ✓ [radio_sync_seed13_rep3](./radio_sync/runs/radio_sync_seed13_rep3_20251120T171411Z/) |
| 14 | ✓ [seed14](./radio_sync/runs/seed14_20251119T203623Z/) | ✓ [radio_sync_seed14_rep2](./radio_sync/runs/radio_sync_seed14_rep2_20251120T171411Z/) | ✓ [radio_sync_seed14_rep3](./radio_sync/runs/radio_sync_seed14_rep3_20251120T171411Z/) |
| 15 | ✓ [seed15](./radio_sync/runs/seed15_20251119T203623Z/) | ✓ [radio_sync_seed15_rep2](./radio_sync/runs/radio_sync_seed15_rep2_20251120T171411Z/) | ✓ [radio_sync_seed15_rep3](./radio_sync/runs/radio_sync_seed15_rep3_20251120T171411Z/) |
| 16 | ✓ [seed16](./radio_sync/runs/seed16_20251119T203623Z/) | ✓ [radio_sync_seed16_rep2](./radio_sync/runs/radio_sync_seed16_rep2_20251120T171411Z/) | ✓ [radio_sync_seed16_rep3](./radio_sync/runs/radio_sync_seed16_rep3_20251120T171411Z/) |
| 17 | ✓ [seed17](./radio_sync/runs/seed17_20251119T203623Z/) | ✓ [radio_sync_seed17_rep2](./radio_sync/runs/radio_sync_seed17_rep2_20251120T171411Z/) | ✓ [radio_sync_seed17_rep3](./radio_sync/runs/radio_sync_seed17_rep3_20251120T171411Z/) |

### Global Mode
| Seed | Rep1 | Rep2 | Rep3 |
|------|------|------|------|
| 13 | ✓ [seed13](./global/runs/seed13_20251119T203623Z/) | ✓ [global_seed13_rep2](./global/runs/global_seed13_rep2_20251120T174650Z/) | ✓ [global_seed13_rep3](./global/runs/global_seed13_rep3_20251120T174650Z/) |
| 14 | ✓ [seed14](./global/runs/seed14_20251119T203623Z/) | ✓ [global_seed14_rep2](./global/runs/global_seed14_rep2_20251120T174650Z/) | ✓ [global_seed14_rep3](./global/runs/global_seed14_rep3_20251120T174650Z/) |
| 15 | ✓ [seed15](./global/runs/seed15_20251119T222148Z/) | ✓ [global_seed15_rep2](./global/runs/global_seed15_rep2_20251120T174650Z/) | ✓ [global_seed15_rep3](./global/runs/global_seed15_rep3_20251120T174650Z/) |
| 16 | ✓ [seed16](./global/runs/seed16_20251119T203623Z/) | ✓ [global_seed16_rep2](./global/runs/global_seed16_rep2_20251120T174650Z/) | ✓ [global_seed16_rep3](./global/runs/global_seed16_rep3_20251120T174650Z/) |
| 17 | ✓ [seed17](./global/runs/seed17_20251119T203623Z/) | ✓ [global_seed17_rep2](./global/runs/global_seed17_rep2_20251120T174650Z/) | ✓ [global_seed17_rep3](./global/runs/global_seed17_rep3_20251120T174650Z/) |

## Results Summary (3 replicas × 5 seeds = 15 runs per mode)

| Mode | Success Rate | Avg Finished Agents (±std) |
|------|--------------|---------------------------|
| **None** | 40.0% (6/15) | 4.20 ± 0.75 |
| **Radio_sync** | 60.0% (9/15) | 4.40 ± 0.88 |
| **Global** | 100.0% (15/15) | 5.00 ± 0.00 |

## Detailed Results

### Success Rate

- **None:** 40.0% (6 of 15 runs with all 5 agents finished)
- **Radio_sync:** 60.0% (9 of 15 runs with all 5 agents finished)
- **Global:** 100.0% (15 of 15 runs with all 5 agents finished)

### Finished Agents

- **None:** 4.20 ± 0.75 agents finished per run
- **Radio_sync:** 4.40 ± 0.88 agents finished per run
- **Global:** 5.00 ± 0.00 agents finished per run

## Plots

**Location:** `analysis/mapshare/plots/` (generated 2025-11-20)

**Generation script:** `analysis/mapshare/generate_plots.py`
```bash
cd analysis/mapshare && python3 generate_plots.py
```

### Essential Plot

**1_success_rate.png** - Bar chart proving map-sharing solves search problem
- Y-axis: Success rate (% of runs where all 5 agents finished)
- X-axis: Map-sharing modes (None, Radio Sync, Global)
- Shows: None 40%, Radio Sync 60%, Global 100%
- Statistical tests: Global vs None p<0.001 (***), Radio_sync vs None p=0.18 (n.s.)
- This single plot directly demonstrates that global map-sharing achieves perfect coordination
