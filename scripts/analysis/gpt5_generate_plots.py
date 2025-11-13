#!/usr/bin/env python3
"""
gpt5_generate_plots.py

Generates cleaner communication-benchmark figures from the validated runs
under experiments/long_corridor_final_20251110T155342Z.

Outputs figures to docs/figures with "gpt5_" prefix, and opens them.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "experiments" / "long_corridor_final_20251110T155342Z" / "runs"
FIG_DIR = ROOT / "docs" / "figures"


@dataclass
class RunData:
    strategy: str
    run_name: str
    config: dict
    transcript: List[dict]
    episode_stream: List[dict]
    metrics: dict


def load_run(run_path: Path) -> Optional[RunData]:
    strategy = run_path.name.split("_")[0]
    config_path = run_path / "config.yaml"
    tpath = run_path / "results" / "transcript.jsonl"
    spath = run_path / "results" / "episode_stream.jsonl"
    mpath = run_path / "results" / "metrics.json"
    if not (config_path.exists() and tpath.exists() and spath.exists()):
        return None
    config = yaml.safe_load(config_path.read_text())
    transcript = [json.loads(l) for l in tpath.read_text().splitlines() if l.strip()]
    episode_stream = [json.loads(l) for l in spath.read_text().splitlines() if l.strip()]
    metrics = {}
    if mpath.exists():
        try:
            metrics = json.loads(mpath.read_text())
        except Exception:
            metrics = {}
    return RunData(strategy=strategy, run_name=run_path.name, config=config,
                   transcript=transcript, episode_stream=episode_stream, metrics=metrics)


def list_runs() -> List[RunData]:
    out: List[RunData] = []
    for p in sorted(RUNS_DIR.iterdir()):
        if not p.is_dir():
            continue
        rd = load_run(p)
        if rd:
            out.append(rd)
    return out


def compute_metrics(rd: RunData) -> dict:
    # Success / LAAS
    turns_max = int(rd.config.get("turns", 100))
    last_frame = rd.episode_stream[-1] if rd.episode_stream else {"turn": 0, "agents": {}}
    # arrival per agent = first frame with status FINISHED
    arrivals: Dict[str, int] = {}
    for frame in rd.episode_stream:
        t = frame.get("turn", 0)
        for aid, payload in frame.get("agents", {}).items():
            if payload.get("status") == "FINISHED" and aid not in arrivals:
                arrivals[aid] = t
    finished_agents = len(arrivals)
    laas = max(arrivals.values()) if arrivals else turns_max
    # LAAS_k (k=2)
    laas2 = turns_max
    if len(arrivals) >= 2:
        laas2 = sorted(arrivals.values())[1]

    # Messages and opportunities
    messages = 0
    opp = 0
    msg_lengths: List[int] = []
    for rec in rd.transcript:
        obs = rec.get("observation", {})
        if obs.get("any_peer_in_range"):
            opp += 1
        dec = rec.get("decision", {})
        act = dec.get("action", {})
        if act.get("kind") == "COMMUNICATE":
            messages += 1
            msg = act.get("message", {})
            if isinstance(msg, dict):
                msg_lengths.append(len(json.dumps(msg, separators=(",", ":"))))
            elif isinstance(msg, str):
                msg_lengths.append(len(msg))

    selectivity = messages / max(1, opp)
    avg_msg_len = sum(msg_lengths) / len(msg_lengths) if msg_lengths else 0.0

    # Collisions per turn (prefer metrics.json)
    collisions = rd.metrics.get("collisions")
    if collisions is None:
        # fallback: infer from stationary move counts
        collisions = 0
        prev = None
        for frame in rd.episode_stream:
            if prev is None:
                prev = frame
                continue
            for aid in frame.get("agents", {}):
                a_cur = frame["agents"][aid]
                a_prev = prev["agents"].get(aid)
                if a_prev and a_prev["action"].startswith("MOVE") and (a_cur["x"], a_cur["y"]) == (a_prev["x"], a_prev["y"]):
                    collisions += 1
            prev = frame
    coll_per_turn = collisions / max(1, len(rd.episode_stream))

    return {
        "strategy": rd.strategy,
        "run": rd.run_name,
        "finished": finished_agents,
        "laas": laas,
        "laas2": laas2,
        "messages": messages,
        "opportunities": opp,
        "selectivity": selectivity,
        "avg_msg_len": avg_msg_len,
        "collisions": collisions,
        "collisions_per_turn": coll_per_turn,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = list_runs()
    if not runs:
        print("No runs found at", RUNS_DIR)
        return
    rows = [compute_metrics(r) for r in runs]
    df = pd.DataFrame(rows)

    # Figure 1: Finished agents by strategy (+ LAAS2 annotation)
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, x="strategy", y="finished", errorbar="sd", estimator="mean", capsize=0.1)
    for i, s in enumerate(sorted(df["strategy"].unique())):
        m = df[df.strategy == s]["laas2"].mean()
        ax.text(i, df[df.strategy == s]["finished"].mean() + 0.1, f"LAAS2≈{m:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("Agents finished (0–5)")
    ax.set_xlabel("Communication strategy")
    plt.tight_layout()
    f1 = FIG_DIR / "gpt5_finished_agents.png"
    plt.savefig(f1, dpi=180)
    plt.close()

    # Figure 2: Messages/opportunity vs LAAS2 (per-run scatter)
    plt.figure(figsize=(6, 4))
    ax = sns.scatterplot(data=df, x="selectivity", y="laas2", hue="strategy", style="strategy", s=70)
    ax.set_xlabel("Messages / Opportunity")
    ax.set_ylabel("LAAS2 (turns, lower is better)")
    plt.tight_layout()
    f2 = FIG_DIR / "gpt5_messages_vs_laas2.png"
    plt.savefig(f2, dpi=180)
    plt.close()

    # Figure 3: Collisions per turn by strategy
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="strategy", y="collisions_per_turn", errorbar="sd", estimator="mean", capsize=0.1)
    plt.ylabel("Collisions per turn")
    plt.xlabel("Communication strategy")
    plt.tight_layout()
    f3 = FIG_DIR / "gpt5_collisions_per_turn.png"
    plt.savefig(f3, dpi=180)
    plt.close()

    # Figure 5: Message cost (avg length) — only where messages>0
    df_cost = df[df["messages"] > 0]
    if not df_cost.empty:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df_cost, x="strategy", y="avg_msg_len", errorbar="sd", estimator="mean", capsize=0.1)
        plt.ylabel("Avg message length (chars)")
        plt.xlabel("Communication strategy")
        plt.tight_layout()
        f5 = FIG_DIR / "gpt5_message_cost.png"
        plt.savefig(f5, dpi=180)
        plt.close()

    # Open figures (macOS `open`, otherwise print paths)
    paths = [str(p) for p in [f1, f2, f3] if p.exists()]
    if (FIG_DIR / "gpt5_message_cost.png").exists():
        paths.append(str(FIG_DIR / "gpt5_message_cost.png"))
    if paths:
        try:
            os.system("open " + " ".join(paths))
        except Exception:
            print("Figures:")
            for p in paths:
                print(p)

    print("Wrote figures to", FIG_DIR)


if __name__ == "__main__":
    main()

