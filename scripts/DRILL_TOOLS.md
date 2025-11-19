# Log Drill Toolkit (overview)

Concise index of the analysis utilities for map-sharing runs.

## Preferred: unified drill CLI

- `PYTHONPATH=src python3 scripts/drill.py <subcommand> <log> [--goal x,y]`
  - `summary`      → one-shot overview (earliest goal, audit, distance stats; auto-infers goal if missing)
  - `audit`        → per-agent first-goal, final pos, coverage  
  - `goal-turns`   → exact goal-entry turns  
  - `turn-stats`   → per-turn active/hits/distances  
  - `progress`     → min/mean/max distance CSV (`--out`)  
  - `heatmap`      → visitation grid (JSON)
  - `patch-goal-hits` → backfill goal_hits (in-place or with `--out`)
  - `gif`          → render rich Among Us GIF (`--out`, `--model-name`, `--show-gradient`)

## Minimal helper retained (now a drill subcommand)

## Notes

- Logs produced by the new pipeline already contain `goal_hits`, `dist_to_goal`, finished positions, so tools work without patching.  
- For legacy logs, run `patch_goal_hits.py` once to add `goal_hits` and keep future analyses consistent.  
- All tools accept either full episode JSON (with `frames`) or movement JSON/JSONL (positions per turn).  
- Keep `PYTHONPATH=src` when running locally.***

### Render GIF (rich Among Us sprites)

- `python -m llmgrid.vis.gif --episode <log>.episode.json --out <out.gif> [--model-name ...]`
  - Draws finished agents; goal pulses on hit; per-agent colors/legend included.
