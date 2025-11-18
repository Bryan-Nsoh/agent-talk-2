# Lag & Communication Deep Dive — Consolidated Snapshot

**Last updated:** 2025-11-17T18:50:00Z

**One-look summary**
- Micro sweep: currently running in tmux with fixed Azure key export; metrics pending. Run/monitor/aggregate steps and file paths are below.
- Cross-seed findings: lag is secondary; safety/coverage and message semantics dominate; a simple safety veto would remove most crashes.

---
## A) Micro Blocked Tunnel Sweep (10×7) — Ready, not yet run

- Preset: `micro_blocked_tunnel_small` (goal (9,6), starts (0,0) & (0,6)).
- Experiment: `experiments/micro_blocked_tunnel_small_20251116T000000Z/` with 12 baselines (`baselines.json`) and rule text (`rulesets.md`).
- Run script: `scripts/run_micro_blocked_tunnel_small.sh`
  - Azure-only (`--model azure:gpt-5-mini`), key preflight, captures config/transcript/episode/checkpoint, concurrency=5.
- Status: rerun in tmux (`micro_sweep`) after fixing Azure key propagation (`AZURE_OPENAI_API_KEY` exported from `AZURE_API_KEY`). Checkpoints advance; waiting on metrics.
- When network is available:
  1) Ensure `AZURE_API_KEY` or `AZURE_OPENAI_API_KEY` is set in `~/.env`.
  2) Launch: `tmux new -s micro_sweep "./scripts/run_micro_blocked_tunnel_small.sh"`
  3) Monitor: `tmux capture-pane -pt micro_sweep | tail -n 40`
  4) Aggregate after metrics exist:
     ```bash
     uv run python - <<'PY'
     import json, pathlib, collections
     base = pathlib.Path("experiments/micro_blocked_tunnel_small_20251116T000000Z/runs")
     agg = collections.defaultdict(lambda:[0,0,0,0])
     for r in sorted(base.glob("*")):
         m = r/"results"/"metrics.json"
         if not m.exists(): continue
         d = json.loads(m.read_text())
         name = r.name.split("_seed")[0]
         agg[name][0]+=1
         agg[name][1]+=int(d.get("success", False))
         agg[name][2]+=d.get("collisions",0)
         agg[name][3]+=d.get("messages_sent",0)
     for name,(n,s,c,m) in sorted(agg.items()):
         print(f"{name:30s} n={n} success_rate={s/n if n else 0:.2f} avg_collisions={c/n if n else 0:.1f} avg_messages={m/n if n else 0:.1f}")
     PY
     ```
  5) Deep-dive the top baselines with the existing message helpful/harmful script.
- If network stays blocked: run a clearly labeled `--dry-run` (Greedy) sweep only to validate logging/resume; do not use it for comm conclusions.
- Key files: preset + meta in `experiments/presets/batch/`; experiment folder; run script; code hooks in `src/llmgrid/cli/poc_two_agents.py` and `src/llmgrid/agent/llm_agent.py` (freeform extra rules).

---
## B) Cross-seed Baseline (seeds 13–17) — Communication/Lag Findings

Scope: 45 existing runs (freeform / structured / none). No new simulations were run for this summary.

Artifacts: `/tmp/lag_deep_stats_v2.json`, `/tmp/lag_path_stats.json`, `/tmp/collision_heat.json`, `/tmp/lag_msg_semantics.json`, `/tmp/lag_msg_locaware.json`, `/tmp/lag_safety_counterfactual.json`; scripts `lag_analysis*.py`, `lag_collision_heat.py`, `lag_msg_semantics.py`, `lag_message_locaware.py`, `lag_safety_counterfactual.py`. Raw data: `experiments/cross_seed_baseline_20251112T143355Z/runs/**/results/`.

### High-level conclusions
- Lag is secondary: only ~2–5% of collisions have both fresh messages and hazard flags; most crashes happen with no inbox traffic and no hazard flags.
- Safety/representation bottleneck: coverage and looping track success; collisions cluster in one choke. A simple safety veto would remove most crashes.
- Message semantics are noisy: structured REQUEST:YIELD is often harmful; structured INTENT:STAY is least harmful; freeform slices sit near 50/50 helpful–harmful; per-run helpfulness swings widely.
- Collision geography is fixed: ~51–53% of collisions sit in the same 10 cells (long_corridor spine).

### Key numbers
- Outcomes (agents finished): freeform 62.7% (47/75); none 57.3% (43/75); structured 56.0% (42/75).
- Collisions per 1k moves: none 48.3; structured 46.0; freeform 48.8.
- Coverage vs success: successes unique-cells/step ≈0.75 vs ≈0.61 for failures; freeform successes ≈0.72; none successes ≈0.81; structured had no successes.
- Message efficacy (delivered): freeform helpful 36.4%, harmful 51.5%; structured helpful 39.1%, harmful 41.1%; per-run net helpful median 0.0 (wide ranges).
- Location-aware (freeform): of 99 messages, 5 harmful, 71 helpful, 23 neutral in referenced cells — harm is mostly off-target collisions.
- Safety veto counterfactual: freeform veto 558 vs 284 collisions; structured 678 vs 343; none 562 vs 284 — clears nearly all crashes while touching <10% of moves.
- Hotspots: top 10 cells contain ~51–53% of collisions across strategies.
- Looping (closed walks ≤5): structured highest; freeform middle; none lowest (e.g., L3 per 1k moves: freeform ~455, structured ~447, none ~406).

### Interpretations
- Hazard signals are underpowered; when present, models obey, but most danger is unflagged.
- Communication value depends on content, not timing; structured can harm (REQUEST:YIELD), freeform is volatile.
- Success tracks coverage and low looping, not message volume; the choke drives most failures.
- Enforcing a safety veto or simple priority in the choke would likely help more than removing the 1-turn delay.

---
## Quick file map
- Micro sweep: preset + meta in `experiments/presets/batch/micro_blocked_tunnel_small*`; experiment folder `experiments/micro_blocked_tunnel_small_20251116T000000Z/`; run script `scripts/run_micro_blocked_tunnel_small.sh`.
- Cross-seed data: `experiments/cross_seed_baseline_20251112T143355Z/runs/**/results/`.
- Code hooks: `src/llmgrid/cli/poc_two_agents.py`; `src/llmgrid/agent/llm_agent.py`.
