#!/usr/bin/env python3
"""Monitor all 9 final validation runs."""
import json
import time
from pathlib import Path
from datetime import datetime

def manhattan_distance(x1, y1, x2, y2):
    return abs(x2 - x1) + abs(y2 - y1)

def analyze_run(run_dir):
    """Analyze a single run."""
    stream_path = run_dir / "results" / "episode_stream.jsonl"
    transcript_path = run_dir / "results" / "transcript.jsonl"

    if not stream_path.exists():
        return None

    with open(stream_path) as f:
        lines = list(f)
        if not lines:
            return None
        last_turn = json.loads(lines[-1])

    goal_x, goal_y = 28, 1
    finished = sum(1 for a in last_turn["agents"].values() if a["status"] == "FINISHED")

    messages_sent = 0
    if transcript_path.exists():
        with open(transcript_path) as f:
            for line in f:
                entry = json.loads(line)
                if "decision" in entry and entry["decision"].get("action", {}).get("kind") == "COMMUNICATE":
                    messages_sent += 1

    return {
        "turn": last_turn["turn"],
        "finished": finished,
        "messages": messages_sent
    }

def collect_data():
    """Collect data from all 9 runs."""
    base_dir = Path("experiments/long_corridor_final_20251110T155342Z/runs")

    if not base_dir.exists():
        return {"status": "waiting", "timestamp": datetime.utcnow().isoformat() + "Z"}

    runs = {"structured": [], "freeform": [], "none": []}

    for run_dir in sorted(base_dir.iterdir()):
        if run_dir.is_dir():
            name = run_dir.name
            if name.startswith("structured"):
                strategy = "structured"
                run_num = int(name.split("_")[1].replace("run", ""))
            elif name.startswith("freeform"):
                strategy = "freeform"
                run_num = int(name.split("_")[1].replace("run", ""))
            elif name.startswith("none"):
                strategy = "none"
                run_num = int(name.split("_")[1].replace("run", ""))
            else:
                continue

            result = analyze_run(run_dir)
            if result:
                result["run_num"] = run_num
                runs[strategy].append(result)

    return {
        "status": "active",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "commit": "76d0799",
        "runs": runs
    }

def main():
    print("Monitoring 9 final validation runs...")
    print("Writing to: final_data.json")
    print()

    while True:
        try:
            data = collect_data()
            with open("final_data.json", "w") as f:
                json.dump(data, f, indent=2)

            if data["status"] == "active":
                print(f"{data['timestamp']}")
                for strategy in ["structured", "freeform", "none"]:
                    runs = data["runs"][strategy]
                    if runs:
                        avg_turn = sum(r["turn"] for r in runs) / len(runs)
                        total_fin = sum(r["finished"] for r in runs)
                        print(f"  {strategy:12s}: {len(runs)} runs, avg T{avg_turn:.0f}, total {total_fin}/15 finished")

            time.sleep(10)
        except KeyboardInterrupt:
            print("\nMonitor stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
