# Collision Analysis: Complete Research Summary

**Analysis Date:** 2025-11-17  
**Data Source:** `experiments/cross_seed_baseline_20251112T143355Z/runs/` (9 runs)  
**Collisions Analyzed:** 222 events across 2,700 total turns

## Documents in This Analysis

1. **COLLISION_FINDINGS_SUMMARY.txt** (13 KB, plaintext)
   - Executive summary with all key findings
   - Suitable for rapid review
   - Contains heatmap visualization

2. **collision_analysis_report.md** (16 KB, markdown)
   - Full technical report with 12 major sections
   - Deep dive into spatial patterns, temporal trends, contended effectiveness
   - Detailed collision sequences and root cause analysis

## Key Findings at a Glance

### The Critical Paradox
- **96% of collisions are avoidable** (agent had free alternatives when collision occurred)
- **Only 4% are true dead-end forced collisions**
- Problem: agents don't know other agents' intents, so independent moves converge

### CONTENDED Neighbor Flag: Complete Failure
- Flag presented **221 times** (99.5% of collisions)
- Agents obeyed **0 times** (0% compliance rate)
- **99.5% of collisions occur while agent sees CONTENDED != 0**

### Spatial Hotspots
- Row Y=1 corridor dominates (5+ hotspot cells)
- Cells **(3,1)** and **(10,1)** are forced convergence points appearing across all strategies
- Collisions concentrate in 10% of grid (31 unique cells out of 300)

### Temporal Pattern
- **44%** of collisions in early game (turns 0-33%)
- **37%** in mid game (turns 33-67%)
- **18%** in late game (turns 67-100%)
- Peak: simultaneous exploration phase before routes are established

### Communication Ineffectiveness
| Strategy | Avg/Run | vs. None | Success |
|----------|---------|----------|---------|
| **None** | 20.0 | baseline | 0% |
| **Structured** | 21.0 | +5% | 0% |
| **Freeform** | 33.0 | +65% worse | 0% |

**Conclusion:** Communication does not help. Freeform actually makes collisions worse.

### Root Causes (by frequency)
1. **Simultaneous Exploration (44%)** - agents converge on same cells while exploring
2. **Corridor Thrashing (30%)** - agents retry same blocked direction for 5-10 turns
3. **Implicit Coordination Failure (20%)** - no intent broadcast, agents guess same cell
4. **Goal Magnetism (6%)** - late game convergence toward goal location

## How to Use These Documents

### For Quick Briefing
Read: `COLLISION_FINDINGS_SUMMARY.txt` (5-10 minutes)
- All key findings and statistics
- Examples of collision sequences
- Heatmap visualization
- Plain English explanations

### For Detailed Analysis
Read: `collision_analysis_report.md` (15-20 minutes)
- 12 sections covering all aspects
- Tables, charts, sample transcripts
- Technical deep dives into contended persistence, choice pressure, hotspots
- Counterfactual analysis (what would fix it)

### For Specific Questions

**Q: Where do collisions happen?**
- See Section 2 in report + heatmap in summary
- Answer: Row Y=1 corridor, cells (3,1) and (10,1)

**Q: Why don't agents avoid contended cells?**
- See Section 4 in report + explanation in summary
- Answer: 0% compliance rate; flag set after collision; no broadcast

**Q: Do agents have free alternatives?**
- See Section 5 in report
- Answer: Yes, 96% of collisions had free directions available

**Q: Does communication help?**
- See Section 7 in report
- Answer: No. None=20, Freeform=33, Structured=21 collisions/run

**Q: When do collisions happen?**
- See Section 3 in report + summary temporal breakdown
- Answer: 81% in first 2/3 of episode (exploration phase)

**Q: What would actually fix collisions?**
- See Section 9 in report + summary Option A-D
- Answer: Broadcast intent, mutual observation, arbitration, or alternate routes

## Quantitative Summary

```
Total Events:          222 collisions
Episodes:              9 runs × 300 turns = 2,700 turns
Collision Rate:        222/2700 = 8.2% of turns
Avoidable:             213/222 = 96%
Forced:                9/222 = 4%

Hottest Cell:          (10,1) with 18 collisions
Unique Cells:          31 cells (10% of 300-cell grid)

CONTENDED Ignored:     221/221 = 99.5%
CONTENDED Obeyed:      0/221 = 0%

BLOCK_AGENT:           217 (97.7%)
SWAP_CONFLICT:         5 (2.3%)

Early Game (0-33%):    99 collisions (44%)
Mid Game (33-67%):     82 collisions (37%)
Late Game (67-100%):   41 collisions (18%)
```

## Implied Actions for System Design

The 96% avoidability rate indicates:

**This is a COORDINATION problem, not a PATHFINDING problem.**

Possible solutions:
1. **Broadcast intent** (agent says "moving E next" before move)
2. **Mutual observation** (see adjacent agents, predict moves)
3. **Priority arbitration** (agent ID determines who moves first)
4. **Route pre-planning** (establish safe routes before exploration)

Current baseline fails because:
- STAY action is insufficient for implicit coordination
- CONTENDED flag ignored (agents don't check/respect it)
- No communication of intent happens
- No way to resolve simultaneous choices

---

**Generated:** 2025-11-17  
**Analysis Tool:** Python collision event extraction + statistical aggregation  
**Confidence:** High (based on complete transcript analysis, not sampling)
