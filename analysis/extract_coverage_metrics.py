#!/usr/bin/env python3
"""
Extract exploration and coverage metrics from run outputs.

Usage:
    python extract_coverage_metrics.py <run_dir> [--output metrics.json]
    python extract_coverage_metrics.py <run_dir> --compare <other_run_dir>

Run directory structure expected:
    run_dir/
    ├── results/
    │   ├── episode_stream.jsonl
    │   ├── transcript.jsonl
    │   ├── config.yaml
    │   └── metrics.json
"""

from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class AgentCoverage:
    """Per-agent coverage metrics."""
    agent_id: str
    cells_visited: int
    cells_known: int
    goal_discovery_turn: Optional[int]
    coverage_rate: float  # cells/turn


@dataclass
class TeamCoverage:
    """Team-wide coverage metrics."""
    total_cells_visited: int  # Union of all agents
    total_cells_known: int    # Union of all agents
    avg_cells_per_agent: float
    coverage_pct: float  # total_cells_visited / grid_cells * 100
    knowledge_pct: float
    avg_goal_discovery: Optional[float]
    num_agents: int
    num_turns: int
    success: bool


@dataclass
class RunMetrics:
    """Complete metrics for one run."""
    run_name: str
    per_agent: Dict[str, AgentCoverage]
    team: TeamCoverage
    frontier_mean: Optional[float]  # Average frontier size over time


def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def visited_per_agent(frames: List[dict]) -> Dict[str, Set[Tuple[int, int]]]:
    """Extract all visited cells per agent from episode_stream frames."""
    visited = defaultdict(set)
    for frame in frames:
        for aid, pos in frame["positions"].items():
            visited[aid].add((pos["x"], pos["y"]))
    return dict(visited)


def known_per_agent(transcript_records: List[dict]) -> Dict[str, int]:
    """Extract known cell count per agent from transcript (final observation)."""
    known = {}
    for record in transcript_records:
        aid = record["agent_id"]
        obs = record.get("observation", {})
        grid = obs.get("grid", {}).get("rows", [])
        # Count non-'X' (unknown) cells
        count = sum(1 for row in grid for cell in row if cell != 'X')
        known[aid] = count
    return known


def goal_discovery_per_agent(transcript_records: List[dict]) -> Dict[str, int]:
    """Find turn when each agent first observes goal."""
    first_goal = {}
    for record in transcript_records:
        aid = record["agent_id"]
        if aid not in first_goal and record["observation"].get("goal_known"):
            first_goal[aid] = record.get("turn", record["observation"].get("turn_index", 0))
    return first_goal


def frontier_progression(transcript_records: List[dict]) -> Dict[str, List[int]]:
    """Track frontier (unknown cells adjacent to known) per turn."""
    frontiers = defaultdict(lambda: [])
    for record in transcript_records:
        aid = record["agent_id"]
        turn = record.get("turn", record["observation"].get("turn_index", 0))
        obs = record.get("observation", {})
        count = len(obs.get("adjacent_frontiers", []))
        
        # Pad list if needed
        while len(frontiers[aid]) <= turn:
            frontiers[aid].append(0)
        frontiers[aid][turn] = count
    return dict(frontiers)


def extract_metrics(run_dir: Path, config_yaml: Optional[dict] = None) -> RunMetrics:
    """
    Extract all coverage metrics from a single run.
    
    Args:
        run_dir: Path to run directory with results/
        config_yaml: Optional parsed config (for grid_size); if None, tries to load
    
    Returns:
        RunMetrics dataclass
    """
    results_dir = run_dir / "results"
    
    # Load data
    frames = load_jsonl(results_dir / "episode_stream.jsonl")
    transcript_records = load_jsonl(results_dir / "transcript.jsonl")
    
    if config_yaml is None:
        try:
            import yaml
            with open(results_dir / "config.yaml") as f:
                config_yaml = yaml.safe_load(f)
        except:
            config_yaml = {}
    
    # Extract basics
    visited = visited_per_agent(frames)
    known = known_per_agent(transcript_records)
    goal_discovery = goal_discovery_per_agent(transcript_records)
    frontiers = frontier_progression(transcript_records)
    
    # Grid info
    grid_width = config_yaml.get("width", 30)
    grid_height = config_yaml.get("height", 10)
    total_cells = grid_width * grid_height
    num_turns = frames[-1]["turn"] + 1
    
    # Compute success
    final_frame = frames[-1]
    finished_agents = len(final_frame.get("finished", []))
    success = finished_agents == len(visited)
    
    # Per-agent metrics
    per_agent = {}
    for aid in visited.keys():
        cells_v = len(visited[aid])
        cells_k = known.get(aid, 0)
        goal_turn = goal_discovery.get(aid)
        coverage_rate = cells_v / num_turns if num_turns > 0 else 0
        
        per_agent[aid] = AgentCoverage(
            agent_id=aid,
            cells_visited=cells_v,
            cells_known=cells_k,
            goal_discovery_turn=goal_turn,
            coverage_rate=coverage_rate,
        )
    
    # Team metrics
    team_visited = set()
    team_known = set()
    for aid, cells in visited.items():
        team_visited.update(cells)
    
    # Reconstruct known cells from grids
    if transcript_records:
        last_record = transcript_records[-1]
        grid = last_record["observation"]["grid"]["rows"]
        for y in range(len(grid)):
            for x in range(len(grid[y])):
                if grid[y][x] != 'X':
                    team_known.add((x, y))
    
    avg_cells = sum(a.cells_visited for a in per_agent.values()) / len(per_agent) if per_agent else 0
    goal_times = [t for t in goal_discovery.values() if t is not None]
    avg_goal_turn = sum(goal_times) / len(goal_times) if goal_times else None
    
    # Frontier analysis
    frontier_means = []
    for aid, frontier_list in frontiers.items():
        if frontier_list:
            mean = sum(frontier_list) / len(frontier_list)
            frontier_means.append(mean)
    frontier_mean = sum(frontier_means) / len(frontier_means) if frontier_means else None
    
    team = TeamCoverage(
        total_cells_visited=len(team_visited),
        total_cells_known=len(team_known),
        avg_cells_per_agent=avg_cells,
        coverage_pct=len(team_visited) / total_cells * 100,
        knowledge_pct=len(team_known) / total_cells * 100,
        avg_goal_discovery=avg_goal_turn,
        num_agents=len(per_agent),
        num_turns=num_turns,
        success=success,
    )
    
    return RunMetrics(
        run_name=run_dir.name,
        per_agent=per_agent,
        team=team,
        frontier_mean=frontier_mean,
    )


def compare_runs(run1: RunMetrics, run2: RunMetrics) -> dict:
    """Compare metrics between two runs."""
    return {
        "run1_name": run1.run_name,
        "run2_name": run2.run_name,
        "coverage_diff_pct": run2.team.coverage_pct - run1.team.coverage_pct,
        "avg_cells_diff": run2.team.avg_cells_per_agent - run1.team.avg_cells_per_agent,
        "goal_discovery_speedup": (
            run1.team.avg_goal_discovery / run2.team.avg_goal_discovery
            if run2.team.avg_goal_discovery and run1.team.avg_goal_discovery else None
        ),
        "success_change": run2.team.success - run1.team.success,
    }


def format_results(metrics: RunMetrics) -> str:
    """Format RunMetrics as readable text."""
    lines = []
    lines.append(f"=== {metrics.run_name} ===")
    lines.append(f"Turns: {metrics.team.num_turns}")
    lines.append(f"Success: {metrics.team.success}")
    lines.append(f"Agents: {metrics.team.num_agents}")
    lines.append("")
    lines.append("TEAM COVERAGE:")
    lines.append(f"  Total cells visited: {metrics.team.total_cells_visited} ({metrics.team.coverage_pct:.1f}%)")
    lines.append(f"  Total cells known: {metrics.team.total_cells_known} ({metrics.team.knowledge_pct:.1f}%)")
    lines.append(f"  Avg cells/agent: {metrics.team.avg_cells_per_agent:.1f}")
    if metrics.team.avg_goal_discovery is not None:
        lines.append(f"  Avg goal discovery: turn {metrics.team.avg_goal_discovery:.1f}")
    if metrics.frontier_mean is not None:
        lines.append(f"  Avg frontier size: {metrics.frontier_mean:.1f}")
    lines.append("")
    lines.append("PER-AGENT:")
    for aid, cov in sorted(metrics.per_agent.items()):
        lines.append(f"  {aid}:")
        lines.append(f"    Visited: {cov.cells_visited} cells ({cov.coverage_rate:.2f} cells/turn)")
        lines.append(f"    Known: {cov.cells_known} cells")
        if cov.goal_discovery_turn is not None:
            lines.append(f"    Goal found: turn {cov.goal_discovery_turn}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract coverage metrics from run directory.")
    parser.add_argument("run_dir", type=Path, help="Run directory with results/")
    parser.add_argument("--output", type=Path, help="Output JSON file (default: stdout)")
    parser.add_argument("--compare", type=Path, help="Compare with another run directory")
    parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    # Extract metrics
    metrics1 = extract_metrics(args.run_dir)
    
    if args.format == "text":
        print(format_results(metrics1))
    else:
        output = asdict(metrics1)
        # Convert dataclasses to dicts
        output["per_agent"] = {
            aid: asdict(cov) for aid, cov in output["per_agent"].items()
        }
        output["team"] = asdict(output["team"])
        print(json.dumps(output, indent=2))
    
    if args.compare:
        metrics2 = extract_metrics(args.compare)
        comparison = compare_runs(metrics1, metrics2)
        print("\n=== COMPARISON ===")
        print(json.dumps(comparison, indent=2))
