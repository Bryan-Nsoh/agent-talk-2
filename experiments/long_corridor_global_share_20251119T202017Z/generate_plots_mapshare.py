#!/usr/bin/env python3
"""
Map-sharing analysis plots (none vs radio_sync vs global) for 20251119 runs.
Keeps the visual style of the comms plot suite but adapts metrics to shared-map behavior.
Required files per run: results/metrics.json, results/episode_stream.jsonl, results/transcript.jsonl (for goal_known timing).
Outputs PNGs under experiments/long_corridor_global_share_20251119T202017Z/plots/.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

# Utility loaders ------------------------------------------------------------

def load_metrics(run):
    return json.loads((run/"results"/"metrics.json").read_text())

def load_stream(run):
    path = run/"results"/"episode_stream.jsonl"
    frames = []
    for line in path.read_text().splitlines():
        frames.append(json.loads(line))
    return frames

def load_goal_known_turns(run):
    path = run/"results"/"transcript.jsonl"
    first = {}
    for line in path.read_text().splitlines():
        obj = json.loads(line)
        a = obj["agent_id"]
        t = obj["turn"]
        obs = obj.get("observation", {})
        if obs.get("goal_known") and a not in first:
            first[a] = t
    return first

# Feature extraction ---------------------------------------------------------

def extract_run(run):
    m = load_metrics(run)
    frames = load_stream(run)
    goal_turns = load_goal_known_turns(run)

    last = frames[-1]
    finished_agents = len(last.get("finished", []))
    dist = last.get("dist_to_goal", {})
    final_avg_dist = sum(dist.values()) / len(dist) if dist else None
    final_min_dist = min(dist.values()) if dist else None

    # coverage approximation: count X in last frame positions? stream lacks grid; skip.
    # Instead use dist_to_goal zeros as proxy for finishes plus average distance.

    turns_to_first_finish = None
    turns_to_all_finish = None
    finished_set = set()
    for fr in frames:
        for f in fr.get("finished", []):
            finished_set.add(f)
        if turns_to_first_finish is None and finished_set:
            turns_to_first_finish = fr["turn"]
        if len(finished_set) == len(dist):  # agent count
            turns_to_all_finish = fr["turn"]
            break

    goal_known_list = list(goal_turns.values())
    goal_known_median = np.median(goal_known_list) if goal_known_list else None

    return {
        "map_sharing": m.get("map_sharing", "unknown"),
        "success": m.get("success", False),
        "turns": m.get("turns", 0),
        "collisions": m.get("collisions", 0),
        "finished_agents": finished_agents,
        "turns_to_first_finish": turns_to_first_finish,
        "turns_to_all_finish": turns_to_all_finish,
        "goal_known_median": goal_known_median,
        "final_avg_dist": final_avg_dist,
        "final_min_dist": final_min_dist,
    }

# Aggregation ---------------------------------------------------------------

def collect():
    rows = []
    for run in sorted(RUNS.glob("seed*/")):
        if not (run/"results"/"metrics.json").exists():
            continue
        rows.append(extract_run(run))
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["map_sharing"]].append(r)
    return by_mode

# Plot helpers --------------------------------------------------------------

def bar_mean(ax, by_mode, field, title, ylabel):
    modes = sorted(by_mode.keys())
    vals = []
    errs = []
    for m in modes:
        data = [r[field] for r in by_mode[m] if r[field] is not None]
        vals.append(np.mean(data) if data else 0)
        errs.append(np.std(data) if len(data) > 1 else 0)
    ax.bar(modes, vals, yerr=errs, capsize=5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

def boxplot(ax, by_mode, field, title, ylabel):
    modes = sorted(by_mode.keys())
    data = [
        [r[field] for r in by_mode[m] if r[field] is not None]
        for m in modes
    ]
    ax.boxplot(data, labels=modes, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

# Main plotting -------------------------------------------------------------

def main():
    by_mode = collect()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    ax = axes.flat

    # 1 success rate (all agents finished)
    modes = sorted(by_mode.keys())
    success_rates = []
    for m in modes:
        runs = by_mode[m]
        success_full = [r for r in runs if r["finished_agents"] == 5 and r["success"]]
        success_rates.append(len(success_full) / len(runs) * 100)
    ax[0].bar(modes, success_rates)
    ax[0].set_title("Success rate (all 5 finished)")
    ax[0].set_ylabel("% runs")

    # 2 finished agents avg
    bar_mean(ax[1], by_mode, "finished_agents", "Finished agents", "agents")

    # 3 collisions
    bar_mean(ax[2], by_mode, "collisions", "Collisions", "count")

    # 4 turns to all finish
    boxplot(ax[3], by_mode, "turns_to_all_finish", "Turns to all finish", "turns")

    # 5 goal known median turn
    boxplot(ax[4], by_mode, "goal_known_median", "Goal known turn (median per run)", "turns")

    # 6 final average distance
    bar_mean(ax[5], by_mode, "final_avg_dist", "Final avg dist to goal", "Manhattan")

    fig.tight_layout()
    fig.savefig(OUT / "mapshare_summary.png")
    print(f"Wrote {OUT/'mapshare_summary.png'}")

if __name__ == "__main__":
    main()
