# Observation Field Usage Analysis - Document Index

**Analysis Date:** 2025-11-17  
**Dataset:** 1,281 decision comments from 3 runs (cross_seed_baseline)  
**Runs Analyzed:**
- seed14_freeform_20251112T193615Z (793 entries)
- seed16_freeform_run1_20251113T125857Z (793 entries)
- seed17_structured_run2_20251113T134138Z (488 entries)

---

## Quick Navigation

### 1. Executive Summary (START HERE)
**File:** `observation_field_SUMMARY.txt`  
**Length:** 7 KB, 160 lines  
**What's Inside:**
- Field usage tiers (Tier 1/2/3/4)
- 5 critical findings with specific numbers
- Misuse and error patterns
- Freeform vs structured comparison
- 6 actionable recommendations

**Read this first for:** High-level findings and decision-making context

---

### 2. Detailed Field Analysis (COMPREHENSIVE)
**File:** `observation_field_usage.md`  
**Length:** 15 KB, 500+ lines  
**What's Inside:**
- Section for each of 11 observation fields
- Usage frequency and percentages
- Actual behavior patterns
- Specific turn/agent examples
- Verdict and assessment for each field
- Complete freeform vs structured comparison
- All recommendations

**Read this for:** Understanding why agents use/ignore each field

---

### 3. Specific Examples (EVIDENCE)
**File:** `field_usage_examples.md`  
**Length:** 12 KB, 400+ lines  
**What's Inside:**
- Exact agent quotes for every field
- 5-7 examples per field showing real usage
- Turn numbers and agent IDs
- Error cases with full context
- "What agents say" vs "what agents never say"
- Side-by-side before/after patterns

**Read this for:** Concrete proof of findings; citation support

---

## Key Metrics At a Glance

| Field | Usage | Verdict |
|-------|-------|---------|
| world_map_ascii | 97.0% | Hyperused; cited in nearly every decision |
| nearest_frontier | 92.0% | Primary nav target; drives 9/10 decisions |
| adjacent_state | 83.5% | Critical safety check; wall validation |
| peer_bits | 62.1% | Moderate coordination signaling |
| recent_positions | 37.7% | Trail avoidance; secondary decision |
| goal_sensor_bearing | 21.3% | Weak signal; always deferred to exploration |
| neighbors_in_view | 12.1% | Low use; agents prefer global map |
| contended_neighbors | 8.0% | Low frequency but reliable when present |
| adjacent_frontiers | 1.7% | Never appears in practice |
| history | 2.3% | Virtually ignored; redundant |
| goal_sensor_strength | 0.1% | **DEAD FIELD** -- 1 mention in 1,281 |

---

## Critical Findings Summary

### 1. Goal Sensor Strength is Completely Ignored (CRITICAL)
- **Field Purpose:** Distance indicator (NEAR vs FAR)
- **Actual Usage:** 0.1% (1 mention)
- **Impact:** Agents never re-prioritize goal even when it's close
- **Fix:** Make strength a primary decision gate in prompt

### 2. Agents Ignore 35% of Inbox Messages
- **Frequency:** 23 turns with messages
- **Acknowledged:** 15 (65%)
- **Ignored:** 8 (35%)
- **Fix:** Require explicit message acknowledgment in comments

### 3. Exploration Bias is Permanent
- **Pattern:** "goal bearing noted, but mapping first" (200+ times)
- **Never Observed:** Goal re-prioritization when exploration complete
- **Impact:** Agents explore indefinitely, delays goal-reaching
- **Fix:** Add explicit frontier queue evaluation rule

### 4. Adjacent Frontiers Never Appears
- **Status:** Never non-empty in 1,281 turns
- **Reason:** Environment structure prevents it
- **Fix:** Remove from prompt if environment unchanged

### 5. History Field is Redundant
- **Usage:** 2.3% (29 mentions)
- **Alternative:** Agents use recent_positions instead
- **Fix:** Merge into last_move_outcome tracking

---

## Freeform vs Structured Protocol Differences

**Structured agents (seed17):**
- Cite peers 6.7% more often
- Handle contention 3.4% more explicitly
- Mention frontiers 6.3% less (more balanced)
- Use trails 4.6% less efficiently

**Interpretation:** Structured protocol improves coordination at cost of exploration efficiency.

---

## Data Sources

**Analyzed Files:**
```
/Users/3bn/Documents/My_Repos/agent-talk-2/experiments/
  cross_seed_baseline_20251112T143355Z/runs/
    seed14_freeform_20251112T193615Z/results/transcript.jsonl
    seed16_freeform_run1_20251113T125857Z/results/transcript.jsonl
    seed17_structured_run2_20251113T134138Z/results/transcript.jsonl
```

**Analysis Method:**
- Regex pattern matching on 1,281 decision comments
- Field-by-field extraction and frequency counting
- Turn-level and agent-level sampling
- Freeform vs structured comparison

---

## Recommendations Priority

### Critical (Fix Immediately)
1. **Goal sensor strength**: Make primary decision gate
2. **Inbox handling**: Require acknowledgment in comments
3. **Frontier re-evaluation**: Add explicit goal pivot rule

### Important (Improve Clarity)
4. **Contention visibility**: Make mandatory in reasoning
5. **History tracking**: Merge into last_move_outcome

### Nice-to-Have (Reduce Overhead)
6. **Remove adjacent_frontiers**: If environment unchanged

---

## Document Usage Guide

**For Prompt Designers:**
- Read SUMMARY first (7 KB)
- Review "Recommendations" section in field_usage.md
- Check specific_examples.md for evidence of each recommendation

**For Researchers Studying Agent Reasoning:**
- Start with field_usage_examples.md (see exact agent quotes)
- Cross-reference specific turns and agents
- Use SUMMARY as interpretation guide

**For Implementation Validation:**
- Check field_usage.md "Verdict" section for each field
- Verify fixes against examples in field_usage_examples.md
- Rerun analysis on new runs to measure improvement

---

## Analysis Metadata

| Metric | Value |
|--------|-------|
| Total entries analyzed | 1,281 |
| Freeform entries | 793 |
| Structured entries | 488 |
| Fields analyzed | 11 |
| Fields hyperused (>80%) | 3 |
| Fields ignored (<10%) | 4 |
| Error patterns found | 3 |
| Recommendations | 6 |

---

**Last Updated:** 2025-11-17 22:10 UTC

For questions or to request additional analysis, review the underlying files in `/tmp/` or re-run the analysis script on new transcript data.

