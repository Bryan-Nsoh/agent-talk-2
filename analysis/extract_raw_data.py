#!/usr/bin/env python3
"""
Extract raw data directly from transcript.jsonl and episode_stream.jsonl.
Parses raw logs - does NOT rely on pre-computed metrics.json or episode.json.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

def extract_from_stream(stream_path):
    """
    Extract finished count from episode_stream.jsonl.
    Handles BOTH branch structures:
    - Main (Communication): agents dict with status field
    - Map-share: finished list
    """
    if not stream_path.exists():
        return None

    frames = []
    with open(stream_path) as f:
        for line in f:
            frames.append(json.loads(line))

    if not frames:
        return None

    final_frame = frames[-1]

    # Map-share structure: finished is a list
    if 'finished' in final_frame:
        return len(final_frame['finished'])

    # Communication structure: agents dict with status field
    if 'agents' in final_frame:
        agents = final_frame['agents']
        if isinstance(agents, dict):
            return sum(1 for data in agents.values() if data.get('status') == 'FINISHED')

    return None

def count_collisions_from_stream(stream_path):
    """
    Count collisions from episode_stream.jsonl (Communication branch only).
    Map-share branch must use transcript.
    """
    if not stream_path.exists():
        return None

    collision_count = 0
    with open(stream_path) as f:
        for line in f:
            frame = json.loads(line)
            agents = frame.get('agents', {})
            if isinstance(agents, dict):
                for agent_data in agents.values():
                    if agent_data.get('action') == 'BLOCK_AGENT':
                        collision_count += 1

    return collision_count

def count_collisions_from_transcript(transcript_path):
    """
    Count collisions from transcript.jsonl.
    Works for BOTH branches:
    - Main (Communication): checks last_move_outcome
    - Map-share: checks last_result.kind
    """
    if not transcript_path.exists():
        return None

    collision_count = 0
    with open(transcript_path) as f:
        for line in f:
            record = json.loads(line)
            obs = record.get('observation', {})

            # Communication branch: last_move_outcome field
            last_move_outcome = obs.get('last_move_outcome')
            if last_move_outcome in ['BLOCK_AGENT', 'BLOCK_WALL', 'SWAP_CONFLICT']:
                collision_count += 1
                continue

            # Map-share branch: last_result.kind field
            last_result = obs.get('last_result', {})
            if last_result.get('kind') in ['BLOCK_AGENT', 'BLOCK_WALL', 'SWAP_CONFLICT']:
                collision_count += 1

    return collision_count


def extract_mapshare_data():
    """
    Extract data from map-sharing experiment.
    Parses episode_stream.jsonl and transcript.jsonl directly.
    """
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

            results_dir = run_dir / "results"
            stream_file = results_dir / "episode_stream.jsonl"
            transcript_file = results_dir / "transcript.jsonl"

            # Extract finished count from stream
            finished = extract_from_stream(stream_file)
            if finished is None:
                print(f"  ⚠ {mode:12s} {run_dir.name:50s} NO STREAM DATA")
                continue

            # Extract collision count from transcript (map-share uses this)
            collisions = count_collisions_from_transcript(transcript_file)
            if collisions is None:
                print(f"  ⚠ {mode:12s} {run_dir.name:50s} NO TRANSCRIPT DATA")
                collisions = 0

            data[mode].append({
                'run': run_dir.name,
                'finished': finished,
                'collisions': collisions
            })
            print(f"  {mode:12s} {run_dir.name:50s} finished={finished} collisions={collisions}")

    return data


def extract_comm_data():
    """
    Extract data from communication experiment.
    Parses episode_stream.jsonl and transcript.jsonl directly.
    Infers strategy from run name (not from metrics.json).
    """
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

        # Infer strategy from run name
        if "none" in run_dir.name:
            strategy = "none"
        elif "freeform" in run_dir.name:
            strategy = "freeform"
        elif "structured" in run_dir.name:
            strategy = "structured"
        else:
            print(f"  ⚠ Cannot infer strategy from {run_dir.name}")
            continue

        results_dir = run_dir / "results"
        stream_file = results_dir / "episode_stream.jsonl"
        transcript_file = results_dir / "transcript.jsonl"

        # Extract finished count from stream
        finished = extract_from_stream(stream_file)
        if finished is None:
            print(f"  ⚠ {strategy:12s} {run_dir.name:50s} NO STREAM DATA")
            continue

        # Extract collision count - ALWAYS use transcript for reliability
        # (stream action field is not populated correctly on Communication branch)
        collisions = count_collisions_from_transcript(transcript_file)
        if collisions is None:
            print(f"  ⚠ {strategy:12s} {run_dir.name:50s} NO COLLISION DATA")
            collisions = 0

        data[strategy].append({
            'run': run_dir.name,
            'finished': finished,
            'collisions': collisions
        })
        print(f"  {strategy:12s} {run_dir.name:50s} finished={finished} collisions={collisions}")

    return data


def main():
    print("\n" + "="*80)
    print("ROBUST RAW DATA EXTRACTION")
    print("Parses episode_stream.jsonl and transcript.jsonl directly")
    print("Does NOT rely on metrics.json or episode.json")
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
        runs = mapshare_data[mode]
        finished_counts = [r['finished'] for r in runs]
        collision_counts = [r['collisions'] for r in runs]
        print(f"  {mode:12s}: N={len(runs):2d}")
        print(f"    finished:   {finished_counts}")
        print(f"    collisions: {collision_counts}")

    print("\nCommunication:")
    for strategy in ['none', 'freeform', 'structured']:
        runs = comm_data[strategy]
        finished_counts = [r['finished'] for r in runs]
        collision_counts = [r['collisions'] for r in runs]
        print(f"  {strategy:12s}: N={len(runs):2d}")
        print(f"    finished:   {finished_counts}")
        print(f"    collisions: {collision_counts}")

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
