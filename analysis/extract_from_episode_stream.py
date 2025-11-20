#!/usr/bin/env python3
"""
Extract finished agent counts DIRECTLY from episode_stream.jsonl.
This is the source of truth. No metrics.json, no episode.json.
"""

import json
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent


def count_finished_from_stream(stream_path):
    """Count agents with status='FINISHED' from final frame of episode_stream.jsonl."""
    if not stream_path.exists():
        return None

    # Read last line (final frame)
    with open(stream_path) as f:
        for line in f:
            pass  # iterate to last line
        final_frame = json.loads(line)

    # Count FINISHED agents
    agents = final_frame.get('agents', {})
    finished_count = sum(1 for agent_data in agents.values() if agent_data.get('status') == 'FINISHED')

    return finished_count


def extract_mapshare_data():
    """Extract from map-sharing runs using episode_stream.jsonl."""
    base = REPO_ROOT / "experiments/mapshare_long_corridor_20251119T202017Z"

    data = {'none': [], 'radio_sync': [], 'global': []}

    for mode in ['none', 'radio_sync', 'global']:
        runs_dir = base / mode / "runs"
        if not runs_dir.exists():
            print(f"⚠ {runs_dir} does not exist")
            continue

        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if (run_dir / "IGNORED.txt").exists():
                continue

            stream_file = run_dir / "results" / "episode_stream.jsonl"
            finished = count_finished_from_stream(stream_file)

            if finished is not None:
                data[mode].append({
                    'run': run_dir.name,
                    'finished': finished
                })
                print(f"  {mode:12s} {run_dir.name:50s} finished={finished}")

    return data


def extract_comm_data():
    """Extract from communication runs using episode_stream.jsonl."""
    runs_dir = REPO_ROOT / "experiments/cross_seed_baseline_20251112T143355Z/runs"

    data = {'none': [], 'freeform': [], 'structured': []}

    if not runs_dir.exists():
        print(f"⚠ {runs_dir} does not exist")
        return data

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # Skip experimental variants
        if any(x in run_dir.name for x in ["collision_rule", "frontier_share", "heartbeat", "seeded_inbox", "map_radio"]):
            continue

        # Skip rerun/partial runs
        if "rerun" in run_dir.name or "partial" in run_dir.name:
            continue

        stream_file = run_dir / "results" / "episode_stream.jsonl"

        # Need metrics.json only to know which strategy
        metrics_file = run_dir / "results" / "metrics.json"
        if not metrics_file.exists():
            print(f"  ⚠ No metrics.json for {run_dir.name}")
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        strategy = metrics.get('comm_strategy')
        collisions = metrics.get('collisions', 0)

        if strategy not in data:
            print(f"  ⚠ Unknown strategy '{strategy}' in {run_dir.name}")
            continue

        finished = count_finished_from_stream(stream_file)

        if finished is not None:
            data[strategy].append({
                'run': run_dir.name,
                'finished': finished,
                'collisions': collisions
            })
            print(f"  {strategy:12s} {run_dir.name:50s} finished={finished} collisions={collisions}")

    return data


def main():
    print("\n" + "="*80)
    print("HONEST DATA EXTRACTION FROM episode_stream.jsonl")
    print("="*80)

    print("\n--- MAP-SHARING EXPERIMENT ---\n")
    mapshare_data = extract_mapshare_data()

    print("\n--- COMMUNICATION EXPERIMENT ---\n")
    comm_data = extract_comm_data()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nMap-Sharing:")
    for mode in ['none', 'radio_sync', 'global']:
        counts = [r['finished'] for r in mapshare_data[mode]]
        print(f"  {mode:12s}: N={len(counts):2d}  finished={counts}")

    print("\nCommunication:")
    for strategy in ['none', 'freeform', 'structured']:
        finished_counts = [r['finished'] for r in comm_data[strategy]]
        collision_counts = [r['collisions'] for r in comm_data[strategy]]
        total_finished = sum(finished_counts)
        total_agents = len(finished_counts) * 5
        print(f"  {strategy:12s}: N={len(finished_counts):2d}  finished={total_finished}/{total_agents} ({100*total_finished/total_agents:.1f}%)")
        print(f"  {'':12s}            collisions={collision_counts}")

    # Save
    output_file = Path(__file__).parent / "honest_data_from_stream.json"
    honest_data = {
        'mapshare': mapshare_data,
        'communication': comm_data
    }

    with open(output_file, 'w') as f:
        json.dump(honest_data, f, indent=2)

    print(f"\nHonest data saved to: {output_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
