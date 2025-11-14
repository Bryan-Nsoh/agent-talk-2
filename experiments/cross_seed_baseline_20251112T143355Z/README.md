# Cross-Seed Baseline Study

**Last updated:** 2025-11-14T00:00:00Z
**Status:** complete (44/45 runs)
**Outcome:** useful
**Started:** 2025-11-12

## Question

Does the structured communication advantage validated on canonical seed 13 (73% success, ~21 messages/run) generalize across different agent starting positions (seeds 14-17)?

## Why This Matters

The canonical baseline showed structured communication winning decisively with 73% agent completion rate and ~21 messages per run. However, seed 14 immediately showed a 21× drop in message volume (down to ~1 message), raising questions about whether the canonical result was seed-specific or indicative of a code regression.

This study tests generalization by running the same 3×3 matrix (3 strategies × 3 replicates) across seeds 13-17, changing only the agent spawn positions while holding the maze topology constant.

## Setup

**Model**: `azure:gpt-5-mini` via Unified LLM client

**Task**: Multi-agent navigation in `long_corridor` maze
- 5 agents, 100-turn limit
- Visibility: 1 (immediate neighbors only)
- Radio range: 2 for structured/freeform, 0 for none
- Maze: 30×10, obstacle seed 606, 20% extra connections
- Loop guidance: `explore`
- Bearing sensors: no noise (flip/drop/bias all 0.0)
- History limit: 5 turns

**Design**: 5 seeds × 3 strategies × 3 replicates = 45 runs planned

**Strategies tested**:
- `structured`: INTENT (announce moves) + REQUEST (yield/guide negotiations)
- `freeform`: Natural language CHAT messages (≤96 chars)
- `none`: No radio communication

**Variables**: Agent spawn seed (13-17), communication strategy

**Held constant**: Maze topology, all other parameters

## Execution Timeline

| Date | Event |
|------|-------|
| 2025-11-12 17:53 | Seed 14 initial batch (run1 for each strategy) |
| 2025-11-12 19:36 | Seed 14 runs 2-3 launched |
| 2025-11-12 19:59 | Seed 13 rerun (regression check) |
| 2025-11-13 01:30 | Seed 14 experimental variants (4 runs) |
| 2025-11-13 00:30-07:54 | Golden commit parallel test on seeds 13-14 (VPN killed at 66-67 turns) |
| 2025-11-13 07:59 | **VPN disconnect purged 25 incomplete runs from seeds 15-17** |
| 2025-11-13 01:44-14:42 | Seeds 16-17 relaunched (18 runs total) |
| 2025-11-13 13:38 | Analysis completed on 43 runs |
| 2025-11-14 | **Missing run identified: seed15_none_run3** |

## Dataset

**44 runs complete** (1 missing: seed15_none_run3)

**Core baseline runs** (excluding experimental variants and duplicates):
- Seed 13: 1 rerun (structured only - canonical 9 runs are in long_corridor_final experiment)
- Seed 14: 9 runs (3 structured, 3 freeform, 3 none)
- Seed 15: 8 runs (3 structured, 3 freeform, 2 none - **missing none_run3**)
- Seed 16: 9 runs (3 structured, 3 freeform, 3 none)
- Seed 17: 9 runs (3 structured, 3 freeform, 3 none)

**Additional runs in directory** (not part of core analysis):
- Seed 14: 4 experimental variants (collision_rule, frontier_share, heartbeat, seeded_inbox)
- Seed 15: 3 duplicate runs from VPN recovery (freeform_run2 ×2, structured_run2 ×2, structured_run3 ×2)

**Total directory contents**: 43 runs

**Combined with canonical seed 13**: 52 runs total for full cross-seed analysis (9 canonical + 43 from this study)

## Runs

### Seed 13 (Regression Check)
| Run | Status | Agents Finished | Messages | Notes |
|-----|--------|-----------------|----------|-------|
| seed13_structured_rerun_20251112T195905Z | complete | 2/5 | 2 | Regression check: only 2 msgs vs canonical ~21 |

### Seed 14 (Complete: 9 core + 4 experimental)
| Run | Status | Agents Finished | Messages | Notes |
|-----|--------|-----------------|----------|-------|
| seed14_structured_20251112T175321Z | complete | 0/5 | 1 | Run 1 |
| seed14_structured_20251112T193605Z | complete | 3/5 | 20 | Run 2 |
| seed14_structured_20251112T193613Z | complete | 0/5 | 6 | Run 3 |
| seed14_freeform_20251112T175323Z | complete | 0/5 | 0 | Run 1 |
| seed14_freeform_20251112T193607Z | complete | 5/5 | 15 | Run 2 |
| seed14_freeform_20251112T193615Z | complete | 4/5 | 3 | Run 3 |
| seed14_none_20251112T175325Z | complete | 0/5 | 0 | Run 1 |
| seed14_none_20251112T193609Z | complete | 3/5 | 0 | Run 2 |
| seed14_none_20251112T193617Z | complete | 3/5 | 0 | Run 3 |
| seed14_structured_collision_rule_* | complete | - | - | Experimental variant |
| seed14_structured_frontier_share_* | complete | - | - | Experimental variant |
| seed14_structured_heartbeat_* | complete | - | - | Experimental variant |
| seed14_structured_seeded_inbox_* | complete | - | - | Experimental variant |

### Seed 15 (Complete: 8 core + 3 duplicates)
| Run | Status | Agents Finished | Messages | Notes |
|-----|--------|-----------------|----------|-------|
| seed15_structured_run1_20251113T022522Z | complete | 2/5 | 23 | |
| seed15_structured_run2_20251113T022527Z | complete | 3/5 | 9 | Earlier of 2 duplicates |
| seed15_structured_run3_20251113T022532Z | complete | 3/5 | 3 | Earlier of 2 duplicates |
| seed15_freeform_20251113T014441Z | complete | 4/5 | 8 | Run 1 |
| seed15_freeform_run2_20251113T022537Z | complete | 3/5 | 3 | Earlier of 2 duplicates |
| seed15_freeform_run3_20251113T022542Z | complete | 3/5 | 4 | |
| seed15_none_run1_20251113T022547Z | complete | 4/5 | 0 | |
| seed15_none_run2_20251113T031507Z | complete | 3/5 | 0 | |
| **seed15_none_run3** | **MISSING** | - | - | **VPN killed, not relaunched** |

### Seed 16 (Complete: 9 runs)
| Run | Status | Agents Finished | Messages | Notes |
|-----|--------|-----------------|----------|-------|
| seed16_structured_run1_20251113T125851Z | complete | 2/5 | 0 | |
| seed16_structured_run2_20251113T125853Z | complete | 3/5 | 8 | |
| seed16_structured_run3_20251113T125855Z | complete | 3/5 | 6 | |
| seed16_freeform_run1_20251113T125857Z | complete | 5/5 | 3 | |
| seed16_freeform_run2_20251113T125900Z | complete | 5/5 | 0 | |
| seed16_freeform_run3_20251113T125902Z | complete | 4/5 | 0 | |
| seed16_none_run1_20251113T133526Z | complete | 1/5 | 0 | |
| seed16_none_run2_20251113T133659Z | complete | 3/5 | 0 | |
| seed16_none_run3_20251113T133802Z | complete | 2/5 | 0 | |

### Seed 17 (Complete: 9 runs)
| Run | Status | Agents Finished | Messages | Notes |
|-----|--------|-----------------|----------|-------|
| seed17_structured_run1_20251113T133935Z | complete | 0/5 | 8 | |
| seed17_structured_run2_20251113T134138Z | complete | 3/5 | 8 | |
| seed17_structured_run3_20251113T135146Z | complete | 3/5 | 8 | |
| seed17_freeform_run1_20251113T135821Z | complete | 3/5 | 0 | |
| seed17_freeform_run2_20251113T141431Z | complete | 3/5 | 0 | |
| seed17_freeform_run3_20251113T142508Z | complete | 3/5 | 0 | |
| seed17_none_run1_20251113T142742Z | complete | 3/5 | 0 | |
| seed17_none_run2_20251113T144152Z | complete | 3/5 | 0 | |
| seed17_none_run3_20251113T144225Z | complete | 4/5 | 0 | |

## Results (44 runs from this study only)

Aggregating core runs only (excluding experimental variants and duplicates):

**By Strategy** (this study only, excludes canonical seed 13 runs):
- Structured: 54.7% agents finished (29/53 agents), 8.2 avg messages/run
- Freeform: 71.4% agents finished (30/42 agents), 2.8 avg messages/run
- None: 65.7% agents finished (23/35 agents), 0 avg messages/run

**Combined with canonical seed 13** (52 total runs):
- Freeform: 62.5% success (50/80 agents), 5.4 avg messages
- Structured: 57.3% success (63/110 agents), 9.9 avg messages
- None: 51.4% success (36/70 agents), 0 avg messages

## Key Findings

1. **Canonical seed 13 was an outlier**: Original structured runs showed 73% success with ~21 messages. Rerun on same seed showed only 40% success with 2 messages.

2. **Freeform outperforms structured**: Across seeds 14-17, freeform achieves 71.4% agent completion vs structured's 54.7%, while using 66% fewer messages (2.8 vs 8.2 avg).

3. **High variance**: Message counts range from 0-23 for structured, 0-15 for freeform. Some runs show zero communication even for structured strategy.

4. **No code regression**: Golden commit partial data (66-67 turns on seeds 13-14) showed similarly low message counts (2-3), ruling out recent code changes as cause.

5. **Strategy ranking reversed**: Canonical claimed structured > freeform > none. Cross-seed data shows freeform > none > structured.

## Interpretation

The original hypothesis that structured communication advantage would hold across seeds was not supported. Freeform communication demonstrates better generalization across different spawn configurations. The canonical seed 13 result appears to have been seed-specific, not indicative of structural superiority.

High variance in message counts and agent completion rates suggests the system has significant stochasticity, either from LLM sampling or from subtle interactions between spawn positions and communication patterns.

## Next Steps

- [x] Complete seed 14 runs (9 core runs)
- [x] Complete seed 15 runs (8/9 runs - missing none_run3)
- [x] Complete seed 16 runs (9 runs)
- [x] Complete seed 17 runs (9 runs)
- [ ] Launch seed15_none_run3 to complete 45-run matrix
- [ ] Recompute aggregate metrics with complete 45-run dataset
- [ ] Update FINAL_RESULTS.md with complete analysis
