# Long Corridor Validation Run 1

**Last updated:** 2025-11-10T14:00:00Z
**Status:** ✓ complete
**Outcome:** ⚠ inconclusive
**Started:** 2025-11-10

## Question

Does the priority tiebreaker (commit 6ae6129) consistently prevent deadlocks and maintain structured communication advantage over freeform?

## Why This Matters

Baseline run showed structured (3/5) > freeform (2/5) > none (0/5). Need to validate this pattern holds with priority tiebreaker in place to prevent mutual yielding deadlocks observed in baseline turns 47-53.

## Setup

- **Commit:** 6ae6129 (`feat: structured comms baseline + priority tiebreaker`)
- Model: claude-3-5-sonnet-20241022
- Task: long_corridor preset (30×10 grid)
- Variables: communication strategy (structured, freeform, none)
- Held constant: 5 agents, 100 turn limit, radio range 2, visibility 1

## Runs

| Run | Started | Status | Finished | Notes |
|-----|---------|--------|----------|-------|
| [structured](./runs/structured_20251110T135528Z/) | 2025-11-10 13:55 | ✓ complete | 2/5 | Agents spent turns announcing priority instead of moving |
| [freeform](./runs/freeform_20251110T135528Z/) | 2025-11-10 13:55 | ✓ complete | 1/5 | No priority framework, lower coordination |
| [none](./runs/none_20251110T135528Z/) | 2025-11-10 13:55 | ✓ complete | 0/5 | No comms, stuck as expected |

## Results

**Validation Run 1 (commit 6ae6129):**
- Structured: 2/5 agents finished (worse than baseline 3/5)
- Freeform: 1/5 agents finished (worse than baseline 2/5)
- None: 0/5 agents finished (same as baseline)

**Baseline Run (commit 6ae6129, pre-priority-fix):**
- Structured: 3/5 agents finished
- Freeform: 2/5 agents finished
- None: 0/5 agents finished

## Interpretation

Priority tiebreaker WORKED (prevented deadlock) but introduced inefficiency. Analysis of transcript showed agents wasting turns:
- Turn 45, a3: "Claiming contested cell (5,5); lowest ID should claim it" - COMMUNICATED instead of MOVING
- Turn 50, a3: "lowest agent_id, will move next turn" - announced instead of acting
- Higher-ID agents correctly yielded, but lower-ID agents announced instead of claiming immediately

Pattern: structured still wins (2/5 > 1/5 > 0/5) but performance degraded because rule said "MUST claim immediately" but agents interpreted "claim" as "announce intent to claim."

## Decision

⚠ Need refinement. Priority rule works but needs stronger "DEFAULT TO MOVE" directive.

Changes to make:
- Structured: Add "DEFAULT TO MOVE. Only COMMUNICATE when message prevents imminent collision or shares critical info."
- Structured priority: Change to "LOWEST agent_id MOVES immediately (no announcement needed)"
- Freeform: Add same "DEFAULT TO MOVE" guidance + add priority rule for parity

## Next Steps

- [ ] Launch validation run 2 with updated prompts
- [ ] If structured maintains advantage (2-3/5), freeze as validated baseline
- [ ] Document final commit hash and results
