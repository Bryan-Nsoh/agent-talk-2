#!/usr/bin/env python3
"""
Granular monitoring for validation run 2.
Tracks per-agent microparameters: position, distance, messages, actions.
"""
import json
import time
from pathlib import Path
from datetime import datetime

def manhattan_distance(x1, y1, x2, y2):
    return abs(x2 - x1) + abs(y2 - y1)

def analyze_run(run_dir, strategy):
    """Analyze a single run and return granular stats."""
    stream_path = run_dir / "results" / "episode_stream.jsonl"
    transcript_path = run_dir / "results" / "transcript.jsonl"

    if not stream_path.exists():
        return {
            "strategy": strategy,
            "status": "not_started",
            "turn": 0,
            "agents": {}
        }

    # Read last turn from stream
    with open(stream_path) as f:
        lines = list(f)
        if not lines:
            return {"strategy": strategy, "status": "empty", "turn": 0, "agents": {}}
        last_turn = json.loads(lines[-1])

    # Goal is at (28, 1) for long_corridor
    goal_x, goal_y = 28, 1

    agents = {}
    total_distance = 0
    finished_count = 0

    for agent_id, agent_data in last_turn["agents"].items():
        x = agent_data["x"]
        y = agent_data["y"]
        status = agent_data["status"]
        orientation = agent_data["orientation"]
        action = agent_data.get("action")

        distance = manhattan_distance(x, y, goal_x, goal_y)
        total_distance += distance

        if status == "FINISHED":
            finished_count += 1

        agents[agent_id] = {
            "position": [x, y],
            "orientation": orientation,
            "status": status,
            "distance_to_goal": distance,
            "last_action": action,
            "at_goal": status == "FINISHED"
        }

    avg_distance = total_distance / len(agents) if agents else 0

    # Get message count from transcript
    messages_sent = 0
    last_turn_messages = []

    if transcript_path.exists():
        with open(transcript_path) as f:
            for line in f:
                entry = json.loads(line)
                if "decision" in entry and entry["decision"].get("action", {}).get("kind") == "COMMUNICATE":
                    messages_sent += 1
                    # Track messages from last turn
                    if entry["turn"] == last_turn["turn"]:
                        msg_data = entry["decision"]["action"].get("message", {})
                        last_turn_messages.append({
                            "agent": entry["agent_id"],
                            "kind": msg_data.get("kind", "CHAT"),
                            "content": str(msg_data)[:80]  # Truncate for display
                        })

    return {
        "strategy": strategy,
        "status": "running" if finished_count < 5 and last_turn["turn"] < 100 else "complete",
        "turn": last_turn["turn"],
        "finished": finished_count,
        "total_agents": len(agents),
        "messages_sent": messages_sent,
        "avg_distance": round(avg_distance, 1),
        "agents": agents,
        "last_turn_messages": last_turn_messages
    }

def collect_data():
    """Collect data from all validation runs."""
    base_dir = Path("experiments/long_corridor_validation2_20251110T152208Z/runs")

    if not base_dir.exists():
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "waiting",
            "runs": {}
        }

    runs = {}
    for run_dir in base_dir.iterdir():
        if run_dir.is_dir():
            strategy = run_dir.name.split("_")[0]  # Extract strategy from dir name
            runs[strategy] = analyze_run(run_dir, strategy)

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": "active",
        "experiment_dir": str(base_dir.parent),
        "commit": "324ec42",
        "runs": runs
    }

def main():
    print("Starting validation run 2 monitor...")
    print("Writing to: validation2_data.json")
    print("Press Ctrl+C to stop")
    print()

    while True:
        try:
            data = collect_data()
            with open("validation2_data.json", "w") as f:
                json.dump(data, f, indent=2)

            # Print quick status
            if data["status"] == "active":
                print(f"{data['timestamp']} - ", end="")
                for strategy, run_data in data["runs"].items():
                    if run_data.get("turn"):
                        print(f"{strategy}:T{run_data['turn']}/{run_data['finished']}fin ", end="")
                print()

            time.sleep(10)
        except KeyboardInterrupt:
            print("\nMonitor stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
