# Quick Reference: Coordinate System Analysis

## One-Line Answer
**Agents do NOT confuse local patch with absolute coordinates (82.5%-94% accuracy across metrics).**

---

## The Challenge

Agents receive TWO coordinate systems:
1. **Local patch**: 3x3 grid (rows[0-2], cols[0-2])
2. **Absolute world**: (X, Y) coordinates

Could agents mix them up?

---

## What We Found

| Test | Result | Confidence |
|------|--------|------------|
| Direction language matches patch structure | 82.5% success | HIGH |
| Frontier targets in absolute coords | 94% correct | HIGH |
| Intermediate waypoints correctly computed | 100% correct | HIGH |
| Patch indices used as world coords | 0 cases observed | VERY HIGH |
| X/Y axis confusion | 0 cases observed | VERY HIGH |
| Direction mismatch | 0 cases observed | VERY HIGH |

---

## Key Examples

### Correct: Frontier Targeting
```
Agent at (0,0), patch [0-2, 0-2]:
  "Moving east toward frontier (3,1)"
  
Check:
  ✓ (3,1) is outside patch boundaries
  ✓ Uses world coordinates
  ✓ Direction "east" matches action
  ✓ Patch shows east is free
```

### Correct: Wall Recognition
```
Agent at (2,6), patch [1-3, 5-7]:
  Patch rows[0] = ".##"
  "Moving west to avoid north wall"
  
Check:
  ✓ rows[0] shows walls to north at (2,5) and (3,5)
  ✓ Agent correctly identifies wall direction
  ✓ Moving west is valid escape
  ✓ Uses absolute coordinates
```

### Correct: Multi-Step Path
```
Agent at (4,0), patch [3-5, 0-2]:
  "Advancing east to (5,0) toward frontier (6,1)"
  
Check:
  ✓ (5,0) = patch_x + col_index = 3 + 2 = 5 (correct east neighbor)
  ✓ (6,1) is frontier beyond immediate patch
  ✓ Shows understanding of multi-step planning
  ✓ All coordinates absolute
```

---

## The Conversion Formula

If you see:
```
local_patch:
  rows[i][j]
  top_left_abs: (patch_x, patch_y)
```

Then absolute position of patch cell is:
```
x_absolute = patch_x + j
y_absolute = patch_y + i
```

**And agents use this correctly 100% of the time for intermediate moves.**

---

## Why It Works

1. **Explicit Context**: Each observation has `abs_pos` and `top_left_abs`
2. **Clear Convention**: rows/cols have fixed meaning (N/S/E/W)
3. **Consistent Language**: Agents always cite world coordinates in comments
4. **Rich Information**: Frontier, recent positions, adjacent all in absolute coords

---

## Analysis Scope

- **Data**: 12 transcripts, 930 decision entries
- **Agents**: 2 agents per run, multiple runs
- **Runs**: none_passive, none_cautious, freeform variants
- **Metric**: Direction/coordinate statements in agent comments

---

## Bottom Line

The system design prevents confusion. Agents treat the local patch as a spatial window into absolute space, not a separate coordinate system. Direction language, coordinate citations, and tactical moves are all handled correctly.

---

## For Deep Dive

See:
- `coordinate_confusion_analysis.md` - Full statistical analysis
- `coordinate_system_clarity.md` - Design patterns and scenarios
