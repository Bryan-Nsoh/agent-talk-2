#!/usr/bin/env python3
"""
Generate refined, high-clarity plots for agent communication study.
Uses tiktoken o200k_base tokenizer for accurate token counting.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
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

EXPERIMENT_ROOT = Path(__file__).parent.parent / "experiments" / "long_corridor_final_20251110T155342Z" / "runs"
OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize tokenizer
TOKENIZER = tiktoken.get_encoding("o200k_base")


def load_metrics(run_path):
    """Load metrics.json from a run directory."""
    with open(run_path / "results" / "metrics.json") as f:
        return json.load(f)


def load_episode(run_path):
    """Load episode.json from a run directory."""
    with open(run_path / "results" / "episode.json") as f:
        return json.load(f)


def load_transcript_with_tokens(run_path):
    """Load transcript and count tokens using tiktoken o200k_base."""
    transcript_file = run_path / "results" / "transcript.jsonl"

    messages = []
    total_tokens = 0

    with open(transcript_file) as f:
        for line in f:
            record = json.loads(line)
            decision = record.get('decision', {})
            action = decision.get('action', {})

            if action.get('kind') == 'COMMUNICATE':
                message = action.get('message', {})

                # Extract text content
                if 'text' in message:
                    text = message['text']
                elif 'next_action' in message:
                    text = f"INTENT:{message['next_action']}"
                elif 'req' in message:
                    text = f"REQUEST:{message['req']}"
                    if 'target' in message:
                        text += f"@({message['target']['x']},{message['target']['y']})"
                else:
                    text = json.dumps(message)

                tokens = len(TOKENIZER.encode(text))
                total_tokens += tokens

                messages.append({
                    'turn': record['turn'],
                    'agent': record['agent_id'],
                    'text': text,
                    'tokens': tokens
                })

    return messages, total_tokens


def count_finished_agents(episode_data):
    """Count how many agents reached FINISHED status."""
    final_frame = episode_data['frames'][-1]
    return sum(1 for agent in final_frame['agents'] if agent['status'] == 'FINISHED')


def get_cumulative_finishes(episode_data):
    """Get cumulative count of finished agents over time."""
    finished_set = set()
    cumulative = []

    for frame in episode_data['frames']:
        for agent in frame['agents']:
            if agent['status'] == 'FINISHED':
                finished_set.add(agent['agent_id'])
        cumulative.append((frame['t'], len(finished_set)))

    return cumulative


def collect_all_data():
    """Aggregate data from all 9 runs with accurate token counts."""
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
            messages, tokens = load_transcript_with_tokens(run_path)

            finished = count_finished_agents(episode)
            cumulative_finishes = get_cumulative_finishes(episode)

            data[strategy].append({
                'run_num': run_num,
                'finished': finished,
                'messages': metrics['messages_sent'],
                'tokens': tokens,
                'collisions': metrics['collisions'],
                'turns': metrics['turns'],
                'cumulative_finishes': cumulative_finishes,
                'message_details': messages,
            })

    return data


def plot_1_success_rate(data):
    """Plot 1: Success rate."""
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = ['None', 'Freeform', 'Structured']
    colors = ['#e74c3c', '#f39c12', '#27ae60']

    success_rates = []
    for strategy in ['none', 'freeform', 'structured']:
        total_finished = sum(run['finished'] for run in data[strategy])
        success_rates.append(total_finished / 15 * 100)

    bars = ax.bar(strategies, success_rates, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=1.5, width=0.6)

    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.0f}%\n({int(rate*15/100)}/15)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_xlabel('Communication Strategy', fontweight='bold')
    ax.set_title('Agent Goal Achievement by Strategy', fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "1_success_rate.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"✓ Plot 1: {output_path}")
    plt.close()

    return output_path


def plot_2_communication_volume(data):
    """Plot 2: Messages AND tokens (accurate) per strategy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    strategies = ['None', 'Freeform', 'Structured']
    colors = ['#e74c3c', '#f39c12', '#27ae60']

    # Left: Messages
    for idx, strategy in enumerate(['none', 'freeform', 'structured']):
        messages = [run['messages'] for run in data[strategy]]
        avg_msg = np.mean(messages)

        x_positions = [idx] * len(messages)
        ax1.scatter(x_positions, messages, color=colors[idx], s=120,
                   alpha=0.6, edgecolors='black', linewidth=1.5, zorder=3)

        ax1.hlines(avg_msg, idx - 0.3, idx + 0.3, color=colors[idx],
                  linewidth=4, alpha=0.9, zorder=2, label=f'{strategies[idx]}: {avg_msg:.1f} avg')

    ax1.set_ylabel('Messages Sent (per run)', fontweight='bold')
    ax1.set_xlabel('Strategy', fontweight='bold')
    ax1.set_title('Message Volume', fontweight='bold')
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(strategies)
    ax1.set_ylim(-2, max([run['messages'] for runs in data.values() for run in runs]) + 5)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_axisbelow(True)

    # Right: Tokens (tiktoken o200k_base)
    for idx, strategy in enumerate(['none', 'freeform', 'structured']):
        tokens = [run['tokens'] for run in data[strategy]]
        avg_tok = np.mean(tokens)

        x_positions = [idx] * len(tokens)
        ax2.scatter(x_positions, tokens, color=colors[idx], s=120,
                   alpha=0.6, edgecolors='black', linewidth=1.5, zorder=3)

        ax2.hlines(avg_tok, idx - 0.3, idx + 0.3, color=colors[idx],
                  linewidth=4, alpha=0.9, zorder=2, label=f'{strategies[idx]}: {avg_tok:.0f} avg')

    ax2.set_ylabel('Tokens Sent (tiktoken o200k_base)', fontweight='bold')
    ax2.set_xlabel('Strategy', fontweight='bold')
    ax2.set_title('Token Volume', fontweight='bold')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(strategies)
    ax2.set_ylim(-10, max([run['tokens'] for runs in data.values() for run in runs]) + 50)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "2_communication_volume.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"✓ Plot 2: {output_path}")
    plt.close()

    return output_path


def plot_3_collision_rate(data):
    """Plot 3: Collision rate with standard deviation."""
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = ['None', 'Freeform', 'Structured']
    colors = ['#e74c3c', '#f39c12', '#27ae60']

    avg_collisions = []
    std_collisions = []
    for strategy in ['none', 'freeform', 'structured']:
        collisions = [run['collisions'] for run in data[strategy]]
        avg_collisions.append(np.mean(collisions))
        std_collisions.append(np.std(collisions))

    bars = ax.bar(strategies, avg_collisions, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=1.5, width=0.6,
                   yerr=std_collisions, capsize=5,
                   error_kw={'linewidth': 2, 'elinewidth': 2})

    for bar, avg, std in zip(bars, avg_collisions, std_collisions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                f'{avg:.1f} ± {std:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Collisions per Run (mean ± std)', fontweight='bold')
    ax.set_xlabel('Communication Strategy', fontweight='bold')
    ax.set_title('Collision Frequency by Strategy', fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    ax.text(0.98, 0.97, 'Error bars: standard deviation\nacross 3 runs',
           transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
           fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "3_collision_rate.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"✓ Plot 3: {output_path}")
    plt.close()

    return output_path


def plot_4_completion_over_time(data):
    """Plot 4: Cumulative task completion over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = ['none', 'freeform', 'structured']
    strategy_labels = ['None', 'Freeform', 'Structured']
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    linestyles = ['--', '-.', '-']

    for idx, (strategy, label, color, ls) in enumerate(zip(strategies, strategy_labels, colors, linestyles)):
        max_turns = 100
        avg_cumulative = np.zeros(max_turns + 1)

        for run in data[strategy]:
            cumulative = np.zeros(max_turns + 1)
            for turn, count in run['cumulative_finishes']:
                if turn <= max_turns:
                    cumulative[turn] = count

            # Forward fill
            for i in range(1, len(cumulative)):
                if cumulative[i] == 0:
                    cumulative[i] = cumulative[i-1]

            avg_cumulative += cumulative

        avg_cumulative /= 3

        turns = np.arange(max_turns + 1)
        ax.plot(turns, avg_cumulative, color=color, linewidth=3,
               linestyle=ls, label=label, alpha=0.9)

        final_val = avg_cumulative[-1]
        ax.scatter([max_turns], [final_val], color=color, s=150,
                  edgecolors='black', linewidth=2, zorder=10)
        ax.text(max_turns + 1, final_val, f'{final_val:.1f}',
               fontweight='bold', fontsize=10, va='center')

    ax.set_xlabel('Turn Index', fontweight='bold')
    ax.set_ylabel('Cumulative Agents Finished (avg)', fontweight='bold')
    ax.set_title('Task Completion Timeline', fontweight='bold', pad=15)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 5.5)
    ax.axhline(5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='All 5 agents')
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "4_completion_timeline.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"✓ Plot 4: {output_path}")
    plt.close()

    return output_path


def plot_5_success_vs_tokens(data):
    """Plot 5: Success vs token cost."""
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {'none': '#e74c3c', 'freeform': '#f39c12', 'structured': '#27ae60'}
    markers = {'none': 'o', 'freeform': 's', 'structured': '^'}

    for strategy in ['none', 'freeform', 'structured']:
        tokens = [run['tokens'] for run in data[strategy]]
        finished = [run['finished'] for run in data[strategy]]

        for i, (t, f) in enumerate(zip(tokens, finished), 1):
            ax.scatter(t, f, color=colors[strategy], marker=markers[strategy],
                      s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
            ax.text(t + 10, f + 0.05, f'R{i}', fontsize=8, ha='left', va='center')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                  markersize=10, markeredgecolor='black', label='None', markeredgewidth=1.5),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#f39c12',
                  markersize=10, markeredgecolor='black', label='Freeform', markeredgewidth=1.5),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#27ae60',
                  markersize=10, markeredgecolor='black', label='Structured', markeredgewidth=1.5),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
             fancybox=True, shadow=True, fontsize=10)

    ax.set_xlabel('Total Tokens (tiktoken o200k_base)', fontweight='bold')
    ax.set_ylabel('Agents Finished (out of 5)', fontweight='bold')
    ax.set_title('Task Success vs Communication Cost', fontweight='bold', pad=15)
    ax.set_ylim(-0.3, 5.3)
    ax.set_xlim(-20, max([run['tokens'] for runs in data.values() for run in runs]) + 60)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    ax.axhline(2.5, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    ax.text(0.02, 0.98, 'Ideal: High success, low tokens (upper left)',
           transform=ax.transAxes, ha='left', va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6),
           fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "5_success_vs_tokens.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"✓ Plot 5: {output_path}")
    plt.close()

    return output_path


def main():
    """Generate all plots with accurate token counting."""
    print("\n" + "="*70)
    print("Generating plots with tiktoken o200k_base tokenizer...")
    print("="*70 + "\n")

    print("Loading data (with accurate token counts)...")
    data = collect_all_data()
    print(f"✓ Loaded {sum(len(runs) for runs in data.values())} runs\n")

    print("Generating plots...\n")
    plot_paths = []

    plot_paths.append(plot_1_success_rate(data))
    plot_paths.append(plot_2_communication_volume(data))
    plot_paths.append(plot_3_collision_rate(data))
    plot_paths.append(plot_4_completion_over_time(data))
    plot_paths.append(plot_5_success_vs_tokens(data))

    print("\n" + "="*70)
    print(f"✓ Generated {len(plot_paths)} plots in {OUTPUT_DIR}")
    print("="*70 + "\n")

    return plot_paths


if __name__ == "__main__":
    main()
