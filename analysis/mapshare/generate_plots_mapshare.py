#!/usr/bin/env python3
"""
Map-sharing analysis plots (none vs radio_sync vs global) for the 20251119 runs.
This is a fork of the experiment-local plotter, relocated to analysis/mapshare/
so analysis outputs live outside the experiment tree.

Inputs (relative to repo root):
- experiments/long_corridor_{mode}_20251119T202017Z/runs/*/results/{metrics.json,episode_stream.jsonl,transcript.jsonl}

Outputs:
- analysis/mapshare/plots/mapshare_summary.png

Semantics:
- success := finished_agents == total_agents (run_preset's success flag is ignored)
- Uses only runs that have metrics.json present.
- Global runs use the rerun with metrics (seed15_rerun_20251119T222148Z) and ignore obsolete dirs flagged with IGNORED.txt.
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Config
# Repo root: analysis/mapshare -> analysis -> repo
REPO = Path(__file__).resolve().parents[2]
EXP_BASE = REPO / "experiments"
OUT_DIR = Path(__file__).parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEEP_GLOBAL = {
    "seed13_20251119T203623Z",
    "seed14_20251119T203623Z",
    "seed16_20251119T203623Z",
    "seed17_20251119T203623Z",
    "seed15_rerun_20251119T222148Z",
}

STYLES = {
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
}
plt.rcParams.update(STYLES)


# ---------------------------------------------------------------------------
# Loaders
def stream_frames(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def goal_known_turns(transcript: Path):
    first = {}
    for line in transcript.read_text().splitlines():
        obj = json.loads(line)
        obs = obj.get("observation", {})
        if obs.get("goal_known"):
            a = obj["agent_id"]
            t = obj["turn"]
            if a not in first:
                first[a] = t
    return first


def load_run(mode: str, run: Path):
    metrics = json.loads((run / "results" / "metrics.json").read_text())
    frames = stream_frames(run / "results" / "episode_stream.jsonl")
    transcript_turns = goal_known_turns(run / "results" / "transcript.jsonl")

    last = frames[-1]
    agent_count = len(last.get("positions", {}))
    finished_agents = len(last.get("finished", []))
    success = finished_agents == agent_count

    # turns to all finish
    finished_set = set()
    all_finish = None
    for fr in frames:
        for f in fr.get("finished", []):
            finished_set.add(f)
        if len(finished_set) == agent_count:
            all_finish = fr["turn"]
            break
    if all_finish is None:
        all_finish = frames[-1]["turn"]

    # goal-known timing (median per agent per run)
    goal_med = None
    if transcript_turns:
        goal_med = float(np.median(list(transcript_turns.values())))

    # cumulative curve
    cum = []
    finished_set = set()
    for fr in frames:
        for f in fr.get("finished", []):
            finished_set.add(f)
        cum.append((fr["turn"], len(finished_set)))

    return {
        "mode": mode,
        "finished": finished_agents,
        "agent_count": agent_count,
        "success": success,
        "collisions": metrics.get("collisions", 0),
        "all_finish": all_finish,
        "goal_med": goal_med,
        "cum": cum,
    }


# ---------------------------------------------------------------------------
# Collect
def collect_modes():
    mode_roots = {
        "no_share": EXP_BASE / "long_corridor_no_share_20251119T202017Z" / "runs",
        "radio_sync": EXP_BASE / "long_corridor_radio_sync_20251119T202017Z" / "runs",
        "global": EXP_BASE / "long_corridor_global_share_20251119T202017Z" / "runs",
    }
    rows_by_mode = defaultdict(list)
    for mode, root in mode_roots.items():
        for run in sorted(root.glob("seed*/")):
            if mode == "global" and run.name not in KEEP_GLOBAL:
                continue
            if not (run / "results" / "metrics.json").exists():
                continue
            rows_by_mode[mode].append(load_run(mode, run))
    return rows_by_mode


# ---------------------------------------------------------------------------
# Plot
def bar(ax, modes, values, title, ylabel):
    ax.bar(modes, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def box(ax, modes, series, title, ylabel):
    ax.boxplot(series, tick_labels=modes, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def plot_summary(rows_by_mode):
    modes = sorted(rows_by_mode.keys())
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    ax = axes.flat

    # success rate
    success = [
        sum(1 for r in rows_by_mode[m] if r["success"]) / len(rows_by_mode[m]) * 100
        for m in modes
    ]
    bar(ax[0], modes, success, "Success rate (all 5 finished)", "% runs")

    # finished agents avg
    finished_avg = [np.mean([r["finished"] for r in rows_by_mode[m]]) for m in modes]
    bar(ax[1], modes, finished_avg, "Finished agents (avg)", "agents")

    # collisions avg
    collisions = [np.mean([r["collisions"] for r in rows_by_mode[m]]) for m in modes]
    bar(ax[2], modes, collisions, "Collisions (avg)", "count")

    # turns to all finish
    all_finish_series = [[r["all_finish"] for r in rows_by_mode[m]] for m in modes]
    box(ax[3], modes, all_finish_series, "Turns to all finish", "turns")

    # goal known median
    goal_series = [[r["goal_med"] for r in rows_by_mode[m] if r["goal_med"] is not None] for m in modes]
    box(ax[4], modes, goal_series, "Goal-known median turn", "turns")

    # cumulative finish curves (per mode)
    ax[5].set_title("Cumulative finishes over time")
    ax[5].set_xlabel("turn")
    ax[5].set_ylabel("finished agents")
    ax[5].set_ylim(0, 5)
    colors = {"no_share": "C0", "radio_sync": "C1", "global": "C2"}
    for m in modes:
        for r in rows_by_mode[m]:
            ts = [t for t, c in r["cum"]]
            cs = [c for t, c in r["cum"]]
            ax[5].step(ts, cs, where="post", alpha=0.5, color=colors.get(m, None), label=m)
    # dedupe legend
    handles, labels = ax[5].get_legend_handles_labels()
    bylabel = dict(zip(labels, handles))
    ax[5].legend(bylabel.values(), bylabel.keys())

    fig.tight_layout()
    out = OUT_DIR / "mapshare_summary.png"
    fig.savefig(out)
    print("wrote", out)


def main():
    rows_by_mode = collect_modes()
    plot_summary(rows_by_mode)


if __name__ == "__main__":
    main()
