#!/usr/bin/env python3
"""
Generic monitoring script that writes data to JSON for dashboard.

Usage:
    python monitor_script.py

Then open monitor_dashboard.html in your browser.
"""
import json
import time
from pathlib import Path
import glob
import yaml

def analyze_run(run_dir):
    """Analyze a single run and return stats."""
    stream_path = run_dir / "results" / "episode_stream.jsonl"
    if not stream_path.exists():
        return None

    with open(stream_path) as f:
        lines = f.readlines()

    if not lines:
        return None

    # Get config
    try:
        with open(run_dir / "config.yaml") as f:
            config = yaml.safe_load(f)
    except:
        config = {}

    goal_x = config.get('width', 30) - 2
    goal_y = 1

    last_turn = json.loads(lines[-1])
    turn_num = last_turn['turn']
    agents = last_turn['agents']

    # Calculate per-agent stats
    agent_stats = []
    distances = []
    finished_count = 0

    for aid in sorted(agents.keys()):
        adata = agents[aid]
        x, y = adata['x'], adata['y']
        status = adata.get('status', 'ACTIVE')

        if status == 'FINISHED':
            finished_count += 1
            dist = 0
        else:
            dist = abs(x - goal_x) + abs(y - goal_y)
            distances.append(dist)

        agent_stats.append({
            'id': aid,
            'x': x,
            'y': y,
            'dist': dist,
            'status': status
        })

    # Count messages
    total_delivered = sum(json.loads(line).get('delivered', 0) for line in lines)

    avg_dist = sum(distances) / len(distances) if distances else 0
    min_dist = min(distances) if distances else 0
    max_dist = max(distances) if distances else 0

    return {
        'turn': turn_num,
        'finished': finished_count,
        'messages': total_delivered,
        'avg_dist': avg_dist,
        'min_dist': min_dist,
        'max_dist': max_dist,
        'agents': agent_stats
    }

def collect_data():
    """Collect data from all active runs."""
    baseline_dir = Path("experiments/long_corridor_comms_test_20251110T020144Z/runs")
    validation_dir = Path("experiments/long_corridor_validation_20251110T135528Z/runs")

    data = {}

    # Baseline runs
    if baseline_dir.exists():
        data['baseline'] = {}
        for strategy in ["structured", "freeform", "none"]:
            pattern = baseline_dir / f"{strategy}_20251110T020144Z"
            if pattern.exists():
                stats = analyze_run(pattern)
                if stats:
                    data['baseline'][strategy] = stats

    # Validation runs
    if validation_dir.exists():
        data['validation'] = {}
        for strategy in ["structured", "freeform", "none"]:
            matches = glob.glob(str(validation_dir / f"{strategy}_*"))
            if matches:
                stats = analyze_run(Path(matches[0]))
                if stats:
                    data['validation'][strategy] = stats

    return data

def main():
    print("🚀 Agent Run Monitor")
    print("=" * 60)
    print("Writing data to monitor_data.json")
    print("Open monitor_dashboard.html in your browser")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    try:
        while True:
            data = collect_data()

            # Write to JSON
            with open('monitor_data.json', 'w') as f:
                json.dump(data, f, indent=2)

            # Print summary
            print(f"\r[{time.strftime('%H:%M:%S')}] Updated - ", end="")
            if 'validation' in data:
                for strategy, stats in data['validation'].items():
                    if stats:
                        print(f"{strategy}:{stats['finished']}/5 ", end="")
            print("  ", end="", flush=True)

            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\nStopped monitoring.")

if __name__ == "__main__":
    main()
