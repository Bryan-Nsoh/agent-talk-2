# Manuscript: Multi-Agent Coordination Study

**Last updated:** 2025-11-20T21:30:00Z

## Overview

Comparative analysis of communication vs map-sharing for multi-agent coordination in grid navigation tasks.

## Compilation

```bash
cd manuscript
pdflatex manuscript.tex
bibtex manuscript  # if using references
pdflatex manuscript.tex
pdflatex manuscript.tex
```

Or use latexmk:
```bash
latexmk -pdf manuscript.tex
```

## Figures

All figures are symlinked from experiment directories:
- `mapshare_agent_success_vs_baseline.png` - Map-sharing results (none/radio_sync/global)
- `communication_agent_success_vs_baseline.png` - Communication results (none/freeform/structured)
- `communication_collision_rate_vs_baseline.png` - Collision costs by strategy

## Data Sources

**Map-sharing experiment:**
- Location: `experiments/mapshare_long_corridor_20251119T202017Z/`
- Data: 45 runs (3 modes × 5 seeds × 3 replicates)
- Results: None 84.0% (63/75), Radio_sync 88.0% (66/75), Global 100.0% (75/75)

**Communication experiment:**
- Location: `experiments/cross_seed_baseline_20251112T143355Z/`
- Data: 45 runs (3 strategies × 5 seeds × 3 replicates)
- Results: None 57.3% (43/75), Freeform 62.7% (47/75), Structured 56.0% (42/75)
