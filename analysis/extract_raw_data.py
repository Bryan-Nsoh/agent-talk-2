#!/usr/bin/env python3
"""
Extract raw finished agent counts directly from episode.json files.
Makes ZERO assumptions about metrics.json structure or existence.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

def count_finished_from_episode(episode_path):
    """
    Count finished agents from episode.json final frame.
    Handles TWO data structures:
    1. Communication exp: final_frame['agents'] with status='FINISHED'
    2. Map-sharing exp: final_frame['finished'] as a list
    """
    if not episode_path.exists():
        return None

    with open(episode_path) as f:
        episode = json.load(f)

    final_frame = episode['frames'][-1]

    # Map-sharing structure: finished is a list of agent IDs
    if 'finished' in final_frame and isinstance(final_frame['finished'], list):
        return len(final_frame['finished'])

    # Communication structure: agents array with status field
    if 'agents' in final_frame and isinstance(final_frame['agents'], list):
        return sum(1 for agent in final_frame['agents'] if agent.get('status') == 'FINISHED')

    # Unknown structure
    print(f"  ⚠ Unknown episode structure in {episode_path}")
    return None


def extract_mapshare_data():
    """Extract finished counts from map-sharing experiment."""
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

            episode_file = run_dir / "results" / "episode.json"
            finished = count_finished_from_episode(episode_file)

            if finished is not None:
                data[mode].append({
                    'run': run_dir.name,
                    'finished': finished
                })
                print(f"  {mode:12s} {run_dir.name:50s} finished={finished}")

    return data


def extract_comm_data():
    """Extract finished counts and collisions from communication experiment."""
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

        episode_file = run_dir / "results" / "episode.json"
        metrics_file = run_dir / "results" / "metrics.json"

        finished = count_finished_from_episode(episode_file)

        # Get strategy and collisions from metrics.json
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
    print("RAW DATA EXTRACTION (direct from episode.json)")
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
        print(f"  {strategy:12s}: N={len(finished_counts):2d}  finished={finished_counts}")
        print(f"  {'':12s}            collisions={collision_counts}")

    # Save raw data
    output_file = Path(__file__).parent / "raw_data.json"
    raw_data = {
        'mapshare': mapshare_data,
        'communication': comm_data
    }

    with open(output_file, 'w') as f:
        json.dump(raw_data, f, indent=2)

    print(f"\nRaw data saved to: {output_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
