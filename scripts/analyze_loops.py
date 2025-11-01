#!/usr/bin/env python3
"""
Detect short-cycle loops in agent trajectories.

Usage:
    PYTHONPATH=src python scripts/analyze_loops.py \
        --transcript /path/to/transcript.jsonl \
        --threshold 3

The script scans the prompt transcript (one JSON object per line) and emits a
summary JSON report describing loop segments per agent. A loop segment is
counted when all of the following hold at a turn:

- The most recent history entry (`history[0]['loop']`) is at least `threshold`.
- The agent's comment status for the prior turn is not a BLOCK / collision
  outcome (we only track voluntary oscillations).
- The five `recent_positions` reported in the observation cover at most three
  unique coordinates (indicating shuttling within a tiny area).

This matches the manual diagnosis where agent a4 oscillated between three cells
without collisions. The script does not require the movement stream, but can be
extended to include it if finer-grained analysis is needed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ALLOWED_STATUSES = {"OK", "YIELDING", "FINISHED"}
BLOCK_PREFIXES = ("BLOCK", "SWAP_CONFLICT", "OOB")


@dataclass
class LoopSegment:
    start_turn: int
    end_turn: int
    max_loop: int
    turns: int = field(init=False)

    def __post_init__(self) -> None:
        self.turns = self.end_turn - self.start_turn + 1

    def as_dict(self) -> Dict[str, int]:
        return {
            "start_turn": self.start_turn,
            "end_turn": self.end_turn,
            "turns": self.turns,
            "max_loop": self.max_loop,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise short loops per agent.")
    parser.add_argument("--transcript", required=True, type=Path, help="Path to transcript.jsonl")
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Loop counter threshold to start tracking a segment (default: 3)",
    )
    parser.add_argument("--output", type=Path, help="Optional path to write JSON summary")
    return parser.parse_args()


def extract_status(comment: Optional[str]) -> str:
    if not comment:
        return ""
    token = comment.split(";", 1)[0].strip()
    return token


def load_transcript(path: Path) -> Dict[str, List[dict]]:
    per_agent: Dict[str, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            agent_id = record.get("agent_id")
            if agent_id is None:
                continue
            per_agent[agent_id].append(record)
    # ensure chronological order
    for records in per_agent.values():
        records.sort(key=lambda rec: rec.get("turn", -1))
    return per_agent


def detect_loops(records: List[dict], threshold: int) -> List[LoopSegment]:
    segments: List[LoopSegment] = []
    current_start: Optional[int] = None
    current_max_loop = 0

    for rec in records:
        turn = rec.get("turn")
        obs = rec.get("observation", {})
        history = obs.get("history") or []
        recent_positions = obs.get("recent_positions") or []
        decision = rec.get("decision") or {}
        comment = decision.get("comment")
        status = extract_status(comment)

        # Determine the latest loop counter (previous turn's loop state).
        loop_value = 0
        if history:
            latest = history[0]
            loop_value = int(latest.get("loop", 0))

        unique_positions = {tuple((pos["x"], pos["y"])) for pos in recent_positions if pos}

        is_collision = status.startswith(BLOCK_PREFIXES)
        qualifies = (
            loop_value >= threshold
            and status in ALLOWED_STATUSES
            and len(unique_positions) <= 3
            and not is_collision
        )

        if qualifies:
            if current_start is None:
                current_start = int(turn)
                current_max_loop = loop_value
            else:
                current_max_loop = max(current_max_loop, loop_value)
        elif current_start is not None:
            # Close the current segment.
            segments.append(LoopSegment(current_start, int(turn) - 1, current_max_loop))
            current_start = None
            current_max_loop = 0

    # Flush trailing segment
    if current_start is not None and records:
        final_turn = records[-1].get("turn", current_start)
        segments.append(LoopSegment(current_start, int(final_turn), current_max_loop))

    return segments


def summarise_loops(per_agent: Dict[str, List[dict]], threshold: int) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for agent_id, records in per_agent.items():
        segments = detect_loops(records, threshold)
        total_turns = sum(seg.turns for seg in segments)
        max_loop = max((seg.max_loop for seg in segments), default=0)
        summary[agent_id] = {
            "segments": [seg.as_dict() for seg in segments],
            "total_loop_turns": total_turns,
            "max_loop": max_loop,
        }
    return summary


def main() -> None:
    args = parse_args()
    per_agent = load_transcript(args.transcript)
    summary = summarise_loops(per_agent, args.threshold)
    payload = {
        "threshold": args.threshold,
        "agents": summary,
    }
    serialized = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
