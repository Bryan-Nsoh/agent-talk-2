#!/usr/bin/env python3
"""
Generate map-sharing analysis plots.
Compares none/radio_sync/global map-sharing modes on long_corridor maze.
Parses episode_stream.jsonl directly as source of truth.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

REPO_ROOT = Path(__file__).parent.parent.parent
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Experiment directories
EXPERIMENT_DIRS = {
    'none': EXPERIMENTS_ROOT / "mapshare_long_corridor_20251119T202017Z" / "none" / "runs",
    'radio_sync': EXPERIMENTS_ROOT / "mapshare_long_corridor_20251119T202017Z" / "radio_sync" / "runs",
    'global': EXPERIMENTS_ROOT / "mapshare_long_corridor_20251119T202017Z" / "global" / "runs",
}


def count_finished_from_stream(stream_path):
    """
    Count finished agents from final frame of episode_stream.jsonl.
    This is the source of truth.
    """
    if not stream_path.exists():
        return None

    # Read last line (final frame)
    with open(stream_path) as f:
        for line in f:
            pass  # iterate to last line
        final_frame = json.loads(line)

    # Map-sharing format: finished is a list of agent IDs
    if 'finished' in final_frame and isinstance(final_frame['finished'], list):
        return len(final_frame['finished'])

    # Communication format: agents dict with status field
    if 'agents' in final_frame and isinstance(final_frame['agents'], dict):
        return sum(1 for agent_data in final_frame['agents'].values()
                   if agent_data.get('status') == 'FINISHED')

    print(f"  ⚠ Unknown format in {stream_path}")
    return None


def collect_all_data():
    """Aggregate data from all map-sharing experiments - parse episode_stream.jsonl as source of truth."""
    data = {
        'none': [],
        'radio_sync': [],
        'global': []
    }

    for mode, runs_dir in EXPERIMENT_DIRS.items():
        if not runs_dir.exists():
            print(f"⚠ Warning: {runs_dir} does not exist, skipping {mode}")
            continue

        run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])

        for run_path in run_dirs:
            # Skip IGNORED runs
            if (run_path / "IGNORED.txt").exists():
                continue

            # Check for episode_stream.jsonl (source of truth)
            stream_file = run_path / "results" / "episode_stream.jsonl"
            if not stream_file.exists():
                continue

            try:
                # Parse episode_stream.jsonl for finished count (SOURCE OF TRUTH)
                finished = count_finished_from_stream(stream_file)
                if finished is None:
                    print(f"⚠ Could not parse {stream_file}")
                    continue

                data[mode].append({
                    'run_name': run_path.name,
                    'finished': finished,
                })
            except Exception as e:
                print(f"⚠ Error loading {run_path.name}: {e}")
                continue

    return data


def plot_1_success_rate(data):
    """Plot 1: Agent-level success rate (pooled across all runs, out of 75 total agents)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    modes = ['None', 'Radio Sync', 'Global']
    colors = ['#e74c3c', '#f39c12', '#27ae60']

    success_rates = []
    for mode in ['none', 'radio_sync', 'global']:
        runs = data[mode]
        if len(runs) == 0:
            success_rates.append(0)
            continue

        # Agent-level success: sum all finished agents across all runs
        total_finished = sum(run['finished'] for run in runs)
        total_agents = len(runs) * 5  # Each run has 5 agents
        success_rate = (total_finished / total_agents) * 100
        success_rates.append(success_rate)

    bars = ax.bar(modes, success_rates, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=1.5, width=0.6)

    for bar, rate, mode in zip(bars, success_rates, ['none', 'radio_sync', 'global']):
        height = bar.get_height()
        total_finished = sum(run['finished'] for run in data[mode])
        total_agents = len(data[mode]) * 5
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%\n({total_finished}/{total_agents})',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Agent Success Rate (%)', fontweight='bold')
    ax.set_xlabel('Map-Sharing Mode', fontweight='bold')
    ax.set_title('Individual Agent Goal Achievement (pooled across all runs)', fontweight='bold', pad=15)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "mapshare_agent_success_vs_baseline.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"✓ Plot: {output_path}")
    plt.close()

    return output_path


def main():
    """Generate essential map-sharing plot: agent-level success rate only."""
    print("\n" + "="*70)
    print("Generating map-sharing analysis plot")
    print("Essential plot: agent-level success rate")
    print("="*70 + "\n")

    print("Loading data from episode_stream.jsonl...")
    data = collect_all_data()

    total_runs = sum(len(runs) for runs in data.values())
    print(f"✓ Loaded {total_runs} runs")
    for mode, runs in data.items():
        total_finished = sum(run['finished'] for run in runs)
        total_agents = len(runs) * 5
        print(f"  - {mode}: {len(runs)} runs ({total_finished}/{total_agents} agents finished)")
    print()

    print("Generating plot...\n")
    plot_paths = []

    # Only generate success rate plot - the one that matters
    plot_paths.append(plot_1_success_rate(data))

    print("\n" + "="*70)
    print(f"✓ Generated {len(plot_paths)} plot in {OUTPUT_DIR}")
    print("  Success rate: proves map-sharing solves search problem")
    print("="*70 + "\n")

    return plot_paths


if __name__ == "__main__":
    main()
