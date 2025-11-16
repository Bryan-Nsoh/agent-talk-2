# Team Orientation

## Where We're At

Completed 45 experiments (5 seeds × 3 strategies × 3 replicates) testing communication strategies in multi-agent grid navigation.

**Result:** Communication doesn't reliably help.
- Freeform: 62.7% success
- None (no communication): 57.3% success
- Structured: 56.0% success

The gap is tiny. This is surprising - we designed communication to help agents coordinate, but it's barely moving the needle.

**The question:** Why is communication proving ineffective?

## What's Available

**Experiment data:**
- `experiments/cross_seed_baseline_20251112T143355Z/runs/` - all 45 run directories
- Each run has: `episode.json`, `transcript.jsonl`, `metrics.json`

**Generate GIFs to visualize runs:**
```bash
PYTHONPATH=src uv run python -m llmgrid.cli.render_gif \
  experiments/cross_seed_baseline_20251112T143355Z/runs/[run-name]/results/episode.json \
  --out output.gif --cell-size 40 --fps 6
```

**Report:**
- `docs/REPORT.md` - full write-up (directionally correct, needs review)
- `docs/REPORT_email.html` - HTML version

**Slides:**
- `slides/slide-maker/workspace/presentations/agent-talk.pptx`

## Pending Tasks

**Engage with the results:**
- Read through communication logs (`transcript.jsonl`)
- Generate GIFs for different runs
- Look at what agents see, what they communicate, whether it changes behavior
- Form interpretations: why isn't communication working?

**Think critically:**
- What have we actually learned here?
- As a human with vision radius=1, would communication help YOU navigate this maze?
- Is the bottleneck coordination or exploration?
- What hypotheses can we form from the logs?

**Review materials:**
- Read through the report - check accuracy, think toward arXiv submission
- Look at slides - think about content gaps (don't touch aesthetics)

## Core Point

This is interpretation work, not coding work. The data exists. Engage with it, form concepts about what's happening, think critically about what we can learn.
