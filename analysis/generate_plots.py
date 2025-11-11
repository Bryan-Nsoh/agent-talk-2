#!/usr/bin/env python3
"""
Generate publication-quality plots for agent communication study.
Uses actual data from 9 experiment runs (3 per strategy).
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set publication style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

EXPERIMENT_ROOT = Path(__file__).parent.parent / "experiments" / "long_corridor_final_20251110T155342Z" / "runs"
OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_metrics(run_path):
    """Load metrics.json from a run directory."""
    metrics_file = run_path / "results" / "metrics.json"
    with open(metrics_file) as f:
        return json.load(f)


def load_episode(run_path):
    """Load episode.json from a run directory."""
    episode_file = run_path / "results" / "episode.json"
    with open(episode_file) as f:
        return json.load(f)


def count_finished_agents(episode_data):
    """Count how many agents reached FINISHED status."""
    final_frame = episode_data['frames'][-1]
    return sum(1 for agent in final_frame['agents'] if agent['status'] == 'FINISHED')


def get_arrival_times(episode_data):
    """Extract turn index when each agent finished (or None)."""
    arrivals = {f"a{i}": None for i in range(1, 6)}

    for frame in episode_data['frames']:
        for agent in frame['agents']:
            aid = agent['agent_id']
            if agent['status'] == 'FINISHED' and arrivals[aid] is None:
                arrivals[aid] = frame['t']

    return arrivals


def collect_all_data():
    """Aggregate data from all 9 runs."""
    data = {
        'structured': [],
        'freeform': [],
        'none': []
    }

    run_dirs = {
        'structured': [
            'structured_run1_20251110T155342Z',
            'structured_run2_20251110T155342Z',
            'structured_run3_20251110T155343Z',
        ],
        'freeform': [
            'freeform_run1_20251110T155343Z',
            'freeform_run2_20251110T155344Z',
            'freeform_run3_20251110T155344Z',
        ],
        'none': [
            'none_run1_20251110T155345Z',
            'none_run2_20251110T155345Z',
            'none_run3_20251110T155346Z',
        ]
    }

    for strategy, dirs in run_dirs.items():
        for run_num, run_dir in enumerate(dirs, 1):
            run_path = EXPERIMENT_ROOT / run_dir

            metrics = load_metrics(run_path)
            episode = load_episode(run_path)

            finished = count_finished_agents(episode)
            arrivals = get_arrival_times(episode)

            data[strategy].append({
                'run_num': run_num,
                'finished': finished,
                'messages': metrics['messages_sent'],
                'collisions': metrics['collisions'],
                'turns': metrics['turns'],
                'arrivals': arrivals,
                'collision_causes': metrics.get('collision_causes', {}),
            })

    return data


def plot_success_rate(data):
    """Plot 1: Success rate by strategy."""
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = ['None', 'Freeform', 'Structured']
    colors = ['#e74c3c', '#f39c12', '#27ae60']

    # Calculate success rates
    success_rates = []
    for strategy in ['none', 'freeform', 'structured']:
        total_finished = sum(run['finished'] for run in data[strategy])
        success_rates.append(total_finished / 15 * 100)  # 15 = 3 runs × 5 agents

    bars = ax.bar(strategies, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.0f}%\n({int(rate*15/100)}/15)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_xlabel('Communication Strategy', fontweight='bold')
    ax.set_title('Agent Success Rate by Communication Strategy', fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "1_success_rate.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    return output_path


def plot_messages_vs_success(data):
    """Plot 2: Messages sent vs agents finished (scatter)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {'none': '#e74c3c', 'freeform': '#f39c12', 'structured': '#27ae60'}
    markers = {'none': 'o', 'freeform': 's', 'structured': '^'}

    for strategy in ['none', 'freeform', 'structured']:
        messages = [run['messages'] for run in data[strategy]]
        finished = [run['finished'] for run in data[strategy]]

        ax.scatter(messages, finished,
                  color=colors[strategy],
                  marker=markers[strategy],
                  s=150,
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=1.5,
                  label=strategy.capitalize())

    ax.set_xlabel('Messages Sent', fontweight='bold')
    ax.set_ylabel('Agents Finished (out of 5)', fontweight='bold')
    ax.set_title('Communication Volume vs Task Success', fontweight='bold', pad=15)
    ax.set_ylim(-0.5, 5.5)
    ax.set_xlim(-2, max([run['messages'] for runs in data.values() for run in runs]) + 3)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "2_messages_vs_success.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    return output_path


def plot_collision_rate(data):
    """Plot 3: Average collisions per run by strategy."""
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = ['None', 'Freeform', 'Structured']
    colors = ['#e74c3c', '#f39c12', '#27ae60']

    # Calculate average collisions
    avg_collisions = []
    std_collisions = []
    for strategy in ['none', 'freeform', 'structured']:
        collisions = [run['collisions'] for run in data[strategy]]
        avg_collisions.append(np.mean(collisions))
        std_collisions.append(np.std(collisions))

    bars = ax.bar(strategies, avg_collisions, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, yerr=std_collisions,
                   capsize=5, error_kw={'linewidth': 2})

    # Add value labels
    for bar, avg, std in zip(bars, avg_collisions, std_collisions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                f'{avg:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Average Collisions per Run', fontweight='bold')
    ax.set_xlabel('Communication Strategy', fontweight='bold')
    ax.set_title('Collision Frequency by Strategy', fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "3_collision_rate.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    return output_path


def plot_arrival_timeline(data):
    """Plot 4: Agent arrival timeline across all runs."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    strategy_names = ['None', 'Freeform', 'Structured']
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    agent_colors = {'a1': '#3498db', 'a2': '#e74c3c', 'a3': '#2ecc71',
                    'a4': '#9b59b6', 'a5': '#f39c12'}

    for idx, (strategy, strategy_name, color) in enumerate(zip(
        ['none', 'freeform', 'structured'], strategy_names, colors)):

        ax = axes[idx]

        for run_num, run in enumerate(data[strategy], 1):
            y_pos = run_num

            # Plot timeline bar
            ax.barh(y_pos, 100, left=0, height=0.6,
                   color='lightgray', alpha=0.3, edgecolor='black', linewidth=0.5)

            # Plot agent arrivals
            for agent_id, arrival_time in run['arrivals'].items():
                if arrival_time is not None:
                    ax.scatter(arrival_time, y_pos,
                             color=agent_colors[agent_id],
                             s=100,
                             marker='o',
                             edgecolors='black',
                             linewidth=1,
                             zorder=3,
                             label=agent_id if run_num == 1 else "")
                else:
                    # Mark as didn't finish
                    ax.scatter(100, y_pos,
                             color=agent_colors[agent_id],
                             s=100,
                             marker='x',
                             linewidth=2,
                             zorder=3)

        ax.set_ylabel(f'{strategy_name}\nRun #', fontweight='bold')
        ax.set_yticks([1, 2, 3])
        ax.set_ylim(0.5, 3.5)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        if idx == 0:
            # Add legend only to first subplot
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=agent_colors[f'a{i}'],
                                 markersize=8, markeredgecolor='black',
                                 label=f'Agent {i}')
                      for i in range(1, 6)]
            handles.append(plt.Line2D([0], [0], marker='x', color='gray',
                                     linestyle='', markersize=8,
                                     markeredgewidth=2, label='Did not finish'))
            ax.legend(handles=handles, loc='upper right', ncol=6,
                     frameon=True, fancybox=True, shadow=True, fontsize=8)

    axes[-1].set_xlabel('Turn Index', fontweight='bold')
    axes[-1].set_xlim(0, 105)

    fig.suptitle('Agent Arrival Timeline by Strategy',
                fontweight='bold', fontsize=14, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_path = OUTPUT_DIR / "4_arrival_timeline.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    return output_path


def plot_efficiency_metric(data):
    """Plot 5: Communication efficiency (finished / messages)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = ['Freeform', 'Structured']  # Exclude none (0 messages)
    colors = ['#f39c12', '#27ae60']

    # Calculate efficiency for each run
    efficiency = []
    for strategy in ['freeform', 'structured']:
        eff_values = []
        for run in data[strategy]:
            if run['messages'] > 0:
                eff = run['finished'] / run['messages']
                eff_values.append(eff)
            else:
                eff_values.append(0)
        efficiency.append(np.mean(eff_values))

    bars = ax.bar(strategies, efficiency, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{eff:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Efficiency (Agents Finished / Messages Sent)', fontweight='bold')
    ax.set_xlabel('Communication Strategy', fontweight='bold')
    ax.set_title('Communication Efficiency Metric', fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add note about None strategy
    ax.text(0.5, 0.95, 'Note: "None" excluded (0 messages)',
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=9, style='italic')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "5_efficiency_metric.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    return output_path


def main():
    """Generate all plots."""
    print("\n" + "="*60)
    print("Generating plots from experiment data...")
    print("="*60 + "\n")

    # Collect data
    print("Loading data from 9 runs...")
    data = collect_all_data()
    print(f"✓ Loaded data for {sum(len(runs) for runs in data.values())} runs\n")

    # Generate plots
    print("Generating plots...\n")
    plot_paths = []

    plot_paths.append(plot_success_rate(data))
    plot_paths.append(plot_messages_vs_success(data))
    plot_paths.append(plot_collision_rate(data))
    plot_paths.append(plot_arrival_timeline(data))
    plot_paths.append(plot_efficiency_metric(data))

    print("\n" + "="*60)
    print(f"✓ Generated {len(plot_paths)} plots in {OUTPUT_DIR}")
    print("="*60 + "\n")

    return plot_paths


if __name__ == "__main__":
    main()
