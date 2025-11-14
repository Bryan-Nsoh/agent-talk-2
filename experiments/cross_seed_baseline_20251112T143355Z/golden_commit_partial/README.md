# Golden Commit Partial Results

**Last updated:** 2025-11-13T07:54:00Z
**Status:** ⚠️ incomplete
**Outcome:** ? inconclusive

## Context

Launched golden commit tests (commit fe3ffda) for seeds 13 & 14 to compare against current code. VPN disconnect overnight killed processes at ~66% completion.

## What We Have

Partial data from interrupted runs:

| Seed | Turns | Messages | Status |
|------|-------|----------|--------|
| 13 | 66/100 | 3 | VPN killed at turn 66 |
| 14 | 65/100 | 2 | VPN killed at turn 65 |

Files saved:
- `seed13/transcript.jsonl` - 330 records (66 turns × 5 agents)
- `seed13/episode_stream.jsonl` - 67 turn snapshots
- `seed14/transcript.jsonl` - 326 records (65 turns × 5 agents)
- `seed14/episode_stream.jsonl` - 84 turn snapshots

## Key Finding

Even at 66% completion, golden commit shows similarly low message counts (2-3) compared to current code, NOT the expected ~21 from canonical seed 13 run. This suggests the communication variance is NOT code regression but rather:
- Time-of-day effects (different backend behavior)
- LLM non-determinism
- Seed-specific behavior (different start positions affect communication patterns)

## Resume Options

1. **Reconstruct checkpoint** - Write script to rebuild `EpisodeCheckpoint` from transcript + episode_stream (complex, ~30 min work)
2. **Accept partial data** - Document these 66 turns as valuable negative result (golden commit ALSO shows low comms)
3. **Restart from turn 0** - Wasteful, loses 66-84 turns of LLM calls

## Decision

[To be determined after analyzing batch runner results from seeds 15-17]
