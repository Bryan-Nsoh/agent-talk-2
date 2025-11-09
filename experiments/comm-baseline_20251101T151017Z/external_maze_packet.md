# Maze Stress-Test Packet for External Expert

We’re running five-agent GPT-5 mini simulations on curated mazes with a 100-turn cap, visibility radius 1, and radio range 2. The current default preset (`long_corridor`, seed 606) is too forgiving: silence beats every communication strategy because the layout rarely forces collisions or hidden-information coordination. We need new layouts—still keeping roughly the same surface area as a 24×14 grid (~336 cells) so 100 turns remains meaningful, while allowing any interior shape (jagged, hollow, etc.)—that maximise chokepoints, intersecting flows, and scenarios where agents must announce unseen info (switches, role constraints, etc.).

This packet contains:
- Full text of the relevant experiment documentation and environment source files.
- Plain-text representations of existing presets (including `long_corridor` and a dense 24×14 seed 777 maze).
- Metrics + transcript snippets from the four latest GPT-5 runs (none/intent/negotiation/freeform) that show why communication currently hurts rather than helps.

Please review these materials and design new mazes (as plain-text grids using `#` walls and `.` walkable tiles) plus a short rationale for each. Highlight any additional mechanics (switches, doors) we would need to implement.

## File: experiments/README.md

````markdown
# Experiments: LLM Grid Agents

**Last updated:** 2025-11-07T00:55:00Z

This document is the complete reference for running experiments, managing long-running jobs, and tracking results.

> ⚠️ **Engine change (2025-11-06):** Commits `0a0e38d`, `5291aea`, and `e4ce883` corrected multiple simulation defects (frozen orientations, message ages, LLM-owned `seq`, idle sprites). Any runs recorded before 2025-11-06 must be rerun under the fixed engine; treat existing tables as legacy references only.

## Experiments

| Date | Experiment | Status | Outcome | Result |
|------|------------|--------|---------|--------|
| 2025-11-01 | [comm-baseline](./comm-baseline_20251101T151017Z/) | ?running | - | GPT-5 communication strategies on curated mazes |
| 2025-10-31 | [loop-recovery](./loop-recovery_20251031T213232Z/) | ?running | - | Measuring history window & loop guidance |
| 2025-10-28 | [two-agents-bearing-r1](./two-agents-bearing-r1_20251028T120000Z/) | ?running | ✖ not useful | Bearing-mode multi-agent navigation |

### Status Legend
- ?running | ✔ complete | ✖ failed | ? abandoned

### Outcome Legend
- ✔ useful | ✖ not useful | ? inconclusive | - not determined

## Curated Maze Presets

We now standardise all navigation experiments on six hand-picked mazes generated with the new runtime maze generator. They live under `experiments/presets/batch/` as PNG previews and are reproduced deterministically at run time from their seeds.

- `long_corridor` (seed 606, 30×10, maze style). Wide horizontal corridors with loops near the goal. This is the default baseline for all runs unless we ask for a different preset. First successful Nano run (Oct 30, 2025) completed the goal in 45 turns with zero collisions.
- `open_sparse` (seed 101, 20×12, random scatter at 12 % density). Almost-open field with small clusters.
- `open_dense` (seed 202, 20×12, random scatter at 25 % density). Dense pockets of obstacles with narrow passages.
- `maze_tight` (seed 303, 21×13, maze style, low extra connections). Classic, single-path maze with sharp turns.
- `maze_loops` (seed 404, 21×13, maze style, many extra connections). Maze with plentiful shortcuts and loops.
- `mixed_medium` (seed 505, 24×14, random scatter at 18 % plus extra openings). Hybrid scatter-plus-corridor layout.

Use the `--maze-preset` flag on `llmgrid.cli.poc_two_agents` to pick a maze. The CLI now defaults to `long_corridor`; pass `--maze-preset none` to fall back to manual width/height/density flags.

Preview images can be regenerated or extended with:

```
PYTHONPATH=src uv run python -m llmgrid.cli.generate_maze --help
```

## How to Run Experiments

### Quick Test (verify setup)

```bash
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model openrouter:openai/gpt-5-nano \
  --turns 5
```

### Standard Run Command

**ALL experiments MUST use this pattern:**

```bash
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model <MODEL> \
  --maze-preset <PRESET> \
  --agents <N> \
  --turns <T> \
  --comm-strategy <none|intent|negotiation|freeform> \
  --log-prompts \
  --log-movements \
  --emit-config experiments/<experiment-dir>/runs/$(date -u +%Y%m%dT%H%M%SZ)/config.yaml
```

**Required:** Both `--log-prompts` and `--log-movements` REQUIRE `--emit-config`. The CLI will fail fast if you forget.

**Models:**
- OpenRouter: `--model openrouter:openai/gpt-5-nano`
- Azure: `--model azure:gpt-5-mini` (or your deployment name)

**Environment:** Ensure `~/.env` contains:
- `OPENROUTER_API_KEY` (for OpenRouter)
- `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` (for Azure)

**Concurrency & history:** The episode driver is fully async. Use `run_episode(..., concurrency_start/ max)` when scripting; Azure `gpt-4.1-mini` and OpenRouter `gpt-5-nano` both handle `concurrency_start=5` with the refactored loop-scoped limiter. Observations now supply:
- `history`: up to five prior turns including action, status-prefixed comment, peers seen, and any outbound message summary.
- `last_move_outcome`: enum flag for the previous turn (OK, BLOCK_WALL, BLOCK_AGENT, SWAP_CONFLICT, etc.).
- `contended_neighbors`: NESW bitmask showing which adjacent tiles were contested last turn.
- `last_move_outcome` plus `recent_positions` survive checkpoint/resume so prompts stay consistent mid-run.

**History & comment guardrails:**
- Turn 0 begins empty; once populated, the window remains capped at five entries (oldest entries roll off).
- Comments are auto-prefixed with a status (e.g., `BLOCKED_AGENT(...)`) and clamped to 25 words; blank or whitespace-only comments get replaced with the status alone.
- When comms are disabled or unused, `sent_message` stays `null`, but inbound radio traffic still populates `received_messages` with sender, hop distance, and age.
- Traffic-cone artifacts (NO_GO markers) persist for three turns; agents see them as `NO_GO` adjacency entries and will find a gray dot overlay in the renderer.
- Resume checkpoints persist both history and artifact TTLs; replays pick up with the exact same hazard context.

### Using tmux for Long Runs

```bash
./scripts/run_experiment_tmux.sh \
  --model azure:gpt-5-mini \
  --maze-preset long_corridor \
  --agents 5 \
  --turns 120 \
  --log-prompts \
  --log-movements
```

Monitor: `tmux attach -t run_<timestamp>` or `tail -f logs/run_<timestamp>.log`

**Performance:** With serialized execution (default), expect:
- 2 agents: ~15-20 seconds per turn
- 5 agents: ~30-40 seconds per turn
- 50-turn run with 5 agents: ~25-35 minutes

## Visualization

Render an `episode.json` trace to annotated GIF:

```bash
PYTHONPATH=src uv run python -m llmgrid.cli.render_gif \
  experiments/.../results/episode.json \
  --out experiments/.../results/rollout.gif \
  --cell-size 40 --fps 6
```

Options: `--gradient` for goal-distance tint, `--no-auras` to hide visibility overlays.

**Hazard overlay:** Collision-induced `NO_GO` markers now render as translucent gray dots centred in each affected cell and appear in the legend. They decay automatically (default TTL = 3), so the GIF timeline shows congestion clearing over time.

## Active Workstream

- `two-agents-bearing-r1_20251028T120000Z/` — Multi-agent bearing-mode navigation on curated mazes

**Current baseline:** 
- `long_corridor` with 2 agents, visibility=1, completed in 45 turns (OpenRouter gpt-5-nano).
- Oct 31 Azure sweeps: `azure_history_comms_20251031T165305Z` (history + radio=2, 60-turn timeout, 43 collisions, no comms) vs `azure_history_no_comms_20251031T165744Z` (radio=0, 60-turn timeout, 5 collisions, agents `a1`/`a3` finished). Earlier attempt `azure_history_comms_20251031T165135Z` failed immediately due to a wrapper bug (kept for traceability).

## Key Fix: Connection Pool Exhaustion (2025-10-30)

**Problem:** 5-agent runs failed with `APIConnectionError: Connection error` on Azure.

**Root cause:** Default `concurrency_start = len(agent_ids)` meant 5 agents triggered 5 simultaneous `asyncio.run()` calls in separate threads, exhausting Azure connection pool.

**Fix:** Rebuilt `LlmPolicy`/`run_episode` to stay on one event loop (no nested `asyncio.run`), loop-scoped limiter semaphores, and added a per-agent turn history injected into each observation.

**Result:** 5-agent Azure runs now complete with `concurrency_start=5`. History can be surfaced to the LLM; comms-enabled run remained collision-free, whereas a no-radio baseline accumulated 8 collisions.
- `results/metrics.json` now includes `collision_causes`, `hazard_events`, `comments_clamped`, `comments_autofilled`, `no_go_exposures`, and `contended_exposures` for downstream analysis. Adjust aggregators accordingly.

````

## File: experiments/comm-baseline_20251101T151017Z/README.md

````markdown
# Communication Baseline

**Last updated:** 2025-11-07T00:55:00Z  
**Status:** ? running  
**Outcome:** -  
**Started:** 2025-11-01

> ⚠️ **Legacy data:** These runs were recorded before the 2025-11-06 simulator fixes (orientation tracking, message aging, server-assigned radio `seq`, idle sprites). Treat all numbers below as provisional; every configuration must be rerun under the patched engine.

## Question

How do different communication strategies (none, intent-only, negotiation protocol, freeform chat) affect five-agent navigation performance in the `long_corridor` maze when using `azure:gpt-4.1-mini` with the latest loop-aware prompt?

## Why This Matters

The loop-recovery work showed that prompt tweaks alone leave large gaps: weaker models ignore communication entirely while stronger ones over-communicate yet still time out. We need a clean baseline that isolates communication policy modes so we can quantify their impact on completion time, collisions, and coordination quality.

## Setup

- Model: `azure:gpt-4.1-mini`
- Task: 5-agent cooperative navigation on `long_corridor`
- Dataset: Maze preset `long_corridor` (seed 606) with seeds {13, 17, 23} reserved for future replication (current sweep uses seed 13)
- Variables:
  - Communication strategy: `none`, `intent`, `negotiation`, `freeform`
  - Turn budget: 200 turns
- Held constant: agents=5, visibility=1, radio range=2, history_limit=5, loop_guidance=`explore`, logging enabled (prompts + movement), CLI render at 40px cell size, fps 6.

## Runs

| Run | Started | Status | Notes |
|-----|---------|--------|-------|
| `comm_none_gpt41_seed13_20251101T191609Z` | 2025-11-01 19:16 UTC | ✔ complete | gpt-4.1-mini, comm=none, success=False, turn=200, collisions=10 (agent 6 / wall 4), messages=0 |
| `comm_intent_gpt41_seed13_20251101T192415Z` | 2025-11-01 19:24 UTC | ✔ complete | gpt-4.1-mini, comm=intent, success=False, turn=200, collisions=30 (agent 16 / wall 14), messages=0 |
| `comm_negotiation_gpt41_seed13_20251101T192945Z` | 2025-11-01 19:29 UTC | ✔ complete | gpt-4.1-mini, comm=negotiation, success=False, turn=200, collisions=8 (agent 4 / wall 4), messages=0 |
| `comm_freeform_gpt41_seed13_20251101T193540Z` | 2025-11-01 19:35 UTC | ✔ complete | gpt-4.1-mini, comm=freeform, success=False, turn=200, collisions=27 (agent 12 / wall 15), messages=0 |
| `comm_none_gpt5_seed13_20251101T194121Z` | 2025-11-01 19:41 UTC | ✔ complete | gpt-5-mini, comm=none, success=True, turn=95, collisions=4 (agent only), messages=0 |
| `comm_intent_gpt5_seed13_20251101T195454Z` | 2025-11-01 19:54 UTC | ✔ complete | gpt-5-mini, comm=intent, success=True, turn=70, collisions=0, messages=3 |
| `comm_negotiation_gpt5_seed13_20251101T195510Z` | 2025-11-01 19:55 UTC | ✔ complete | gpt-5-mini, comm=negotiation, success=False, turn=200, collisions=10 (agent only), messages=332 |
| `comm_freeform_gpt5_seed13_20251101T195524Z` | 2025-11-01 19:55 UTC | ✔ complete | gpt-5-mini, comm=freeform, success=True, turn=132, collisions=0, messages=65 |
| `gpt5_none_seed13_20251107T004758Z` | 2025-11-07 00:47 UTC | ✔ complete | gpt-5-mini (Responses API, reasoning_effort=minimal). success=True, turn=54, collisions=4, messages=0 |
| `gpt5_intent_seed13_20251107T004817Z` | 2025-11-07 00:48 UTC | ✔ complete | gpt-5-mini (Responses). success=False (timeout at 100), turn=100, collisions=6, messages=59 |
| `gpt5_negotiation_seed13_20251107T004830Z` | 2025-11-07 00:48 UTC | ✔ complete | gpt-5-mini (Responses). success=True, turn=98, collisions=6, messages=38 |
| `gpt5_freeform_seed13_20251107T004846Z` | 2025-11-07 00:48 UTC | ✔ complete | gpt-5-mini (Responses). success=True, turn=73, collisions=4, messages=27 |

## Results (post-fix vs legacy)

- `azure:gpt-4.1-mini` still times out on the 200-turn cap for all comm modes, with collisions ranging from 8 (negotiation) to 30 (intent). Multiple agents now finish, confirming the goal is no longer blocked by earlier arrivals.
- `azure:gpt-5-mini` succeeds under `none` (turn 95, 4 collisions), `intent` (turn 70, 0 collisions), and `freeform` (turn 132, 0 collisions, 65 CHAT messages). The negotiation protocol remains problematic: despite 332 structured messages, two agents linger near the exit and the run times out at 200 turns.
- Loop summaries show GPT-5 mini aggressively placing NO_GO markers and broadcasting reroutes near congestion, while GPT-4.1 mini rarely communicates even with channels enabled.

### 2025-11-07 (Responses API, GPT-5 mini, 100-turn budget)

| Strategy | Success | LAAS / turns | Messages | Collisions | Key observation |
|----------|---------|--------------|----------|------------|-----------------|
| none | ✓ | 54 | 0 | 4 | Fastest completion; greedy marching works in long_corridor |
| intent | ✗ (timeout) | 100 | 59 | 6 | Agents over-announce while clustered near goal; no benefit over silence |
| negotiation | ✓ | 98 | 38 | 6 | Finishes but slower than no comm due to negotiation latency |
| freeform | ✓ | 73 | 27 | 4 | Finishes, still slower than silent baseline |

**Current learning (seed 13, long_corridor):** Communication adds latency without measurable benefit because the maze rarely forces coordination. The corridor layout has few choke points and limited collision pressure, so structured messaging just stalls agents while they “sync up.”

## Interpretation (2025-11-07)

- Environment + renderer fixes are validated: all four GPT-5 mini runs completed end-to-end via the Responses API with reasoning_effort=`minimal`, proving the new client wiring works.
- On `long_corridor`, silence remains the optimal policy—communication strategies consume 27–59 messages yet finish slower (or time out) because congestion is low. This reinforces that we need mazes where blind coordination matters (tight bottlenecks, conflicting bearings, shared switches) before comm strategies can shine.
- Negotiation still underperforms: even with structured REQUEST/YIELD messages, agents hesitate at the exit and burn nearly the full 100-turn budget. We likely need stronger enforcement (e.g., “move after two unanswered requests”) or more constrained scenarios where the protocol’s extra latency pays for itself.

## Hypothesis Register

1. **HYP-20251107A – Chokepoint Advantage:** If we run the same four communication strategies on a maze with multiple single-tile chokepoints and intersecting traffic (e.g., dense maze with alternating one-way corridors), then negotiation/freeform should outperform the silent baseline on LAAS because agents must yield remotely. _Test plan:_ generate a custom preset with `llmgrid.cli.generate_maze --width 24 --height 14 --style maze --connectivity 0.2 --chokepoints 3`, then rerun GPT-5 mini with the 100-turn cap.
2. **HYP-20251107B – Partial Observability Benefit:** If we add hidden switches / key pickups (simulated via artifacts) that only certain agents can see, then structured communications should reduce redundant exploration. _Test plan:_ extend `maze_generator` to tag “switch” cells and add prompt instructions requiring broadcasts when a switch toggles.

## Decision

- Treat GPT-4.1-mini as a baseline for “reasoning-limited” performance and focus further analysis on GPT-5-mini runs, especially the negotiation failure. Use these results to design prompt/policy tweaks and structural experiments.

## Next Steps

- [x] Rerun seed 13 for all GPT-5 mini strategies under the Responses API to validate the new client wiring.
- [ ] Port the rerun command to seeds 17 and 23 (GPT-5 mini + GPT-4.1 mini) so the summary table covers multiple layouts.
- [ ] Execute HYP-20251107A by generating a chokepoint-heavy maze (use `PYTHONPATH=src uv run python -m llmgrid.cli.generate_maze --width 24 --height 14 --style maze --connectivity 0.1 --keep-corridors 0` and promote the preset to `experiments/presets`).
- [ ] Instrument the renderer/metrics to tag “communication prevented collision” events so we can quantify when chatter helps.
- [x] Regenerate GIFs (`render_gif`) for the four post-fix runs; files now live alongside each run (`rollout_comm-*.gif`).

````

## File: src/llmgrid/env/grid.py

````python
"""Environment primitives for the partially observable grid world."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from llmgrid.schema import (
    AgentSelf,
    ArtifactNoGo,
    AdjacentCell,
    AdjacentState,
    BlockReason,
    CommLimits,
    Direction,
    GoalSensorBearing,
    GoalSensorReading,
    GridSize,
    LocalPatch,
    MarkLimits,
    MoveOutcome,
    MsgHere,
    MsgIntent,
    MsgMarkInfo,
    MsgSense,
    NeighborSummary,
    Observation,
    Octant,
    PlacedArtifact,
    Position,
    ReceivedMessage,
    RelativeOffset,
    StrengthBucket,
    TurnHistory,
)

TileChar = str  # ".", "#", "G", "A", "*"

TRAFFIC_CONE_TTL = 3


@dataclass
class MoveResult:
    final: Tuple[int, int]
    outcome: MoveOutcome
    target: Optional[Tuple[int, int]]
    opponents: List[str]
    cause_cell: Optional[Tuple[int, int]]


def _direction_delta(direction: Direction) -> Tuple[int, int]:
    return {
        Direction.N: (0, -1),
        Direction.E: (1, 0),
        Direction.S: (0, 1),
        Direction.W: (-1, 0),
    }[direction]


class GridWorld:
    """Grid-based environment with synchronous turns."""

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: Iterable[Position],
        goal: Position,
        *,
        seed: int = 0,
        bearing_flip_p: float = 0.15,
        bearing_drop_p: float = 0.10,
        bearing_bias_seed: Optional[int] = None,
        bearing_bias_p: float = 0.0,
        bearing_bias_wall_bonus: float = 0.0,
        history_limit: int = 5,
    ) -> None:
        self.size = GridSize(width=width, height=height)
        self.goal = goal
        self.walls = {(p.x, p.y) for p in obstacles}
        self.rng = random.Random(seed)
        self.bearing_flip_p = bearing_flip_p
        self.bearing_drop_p = bearing_drop_p
        self.bearing_bias_seed = bearing_bias_seed
        self.bearing_bias_p = bearing_bias_p
        self.bearing_bias_wall_bonus = bearing_bias_wall_bonus
        self.history_limit = max(1, history_limit)

        self.occupancy: Dict[str, Tuple[int, int]] = {}
        self.orientation: Dict[str, Direction] = {}
        self.inboxes: Dict[str, List[ReceivedMessage]] = {}
        self.artifacts: Dict[Tuple[int, int], PlacedArtifact] = {}
        self.finished_agents: Dict[str, bool] = {}
        self.position_history: Dict[str, List[Tuple[int, int]]] = {}
        self.turn_history: Dict[str, Deque[dict]] = {}
        self.last_move_outcome: Dict[str, MoveOutcome] = {}
        self.loop_counters: Dict[str, int] = {}
        self.last_goal_distance: Dict[str, int] = {}
        self.last_intent_target: Dict[str, Optional[Tuple[int, int]]] = {}
        self.contended_neighbors: Dict[str, int] = {}
        self.message_seq: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Agent placement and utility helpers
    # ------------------------------------------------------------------

    def add_agent(self, agent_id: str, pos: Position, orientation: Direction) -> None:
        key = (pos.x, pos.y)
        if key in self.walls:
            raise ValueError("Cannot spawn agent on a wall.")
        if key in self.occupancy.values():
            raise ValueError("Spawn cell already occupied.")
        if not self._in_bounds(*key):
            raise ValueError("Spawn position out of bounds.")
        self.occupancy[agent_id] = key
        self.orientation[agent_id] = orientation
        self.inboxes[agent_id] = []
        self.finished_agents[agent_id] = False
        self.position_history[agent_id] = [key]
        self.turn_history[agent_id] = deque(maxlen=self.history_limit)
        self.last_move_outcome[agent_id] = MoveOutcome.OK
        self.loop_counters[agent_id] = 0
        self.last_goal_distance[agent_id] = abs(self.goal.x - pos.x) + abs(self.goal.y - pos.y)
        self.last_intent_target[agent_id] = None
        self.contended_neighbors[agent_id] = 0
        self.message_seq[agent_id] = 0

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size.width and 0 <= y < self.size.height

    def _passable(self, x: int, y: int) -> bool:
        return (x, y) not in self.walls

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def build_observation(
        self,
        agent_id: str,
        *,
        turn_index: int,
        max_turns: int,
        visibility_radius: int,
        radio_range: int,
    ) -> Observation:
        ax, ay = self.occupancy[agent_id]
        local_patch = self._render_patch(ax, ay, visibility_radius)
        neighbors = self._neighbors_in_view(agent_id, visibility_radius)
        artifacts = self._artifacts_in_view(ax, ay, visibility_radius)
        inbox = list(self.inboxes.get(agent_id, []))
        self.inboxes[agent_id] = []

        obs = Observation(
            protocol_version="1.0.0",
            turn_index=turn_index,
            max_turns=max_turns,
            grid_size=self.size,
            self_state=AgentSelf(
                agent_id=agent_id,
                abs_pos=Position(x=ax, y=ay),
                orientation=self.orientation[agent_id],
            ),
            local_patch=local_patch,
            neighbors_in_view=neighbors,
            artifacts_in_view=artifacts,
            inbox=inbox,
            adjacent=self._adjacent_summary(agent_id, ax, ay),
            recent_positions=[
                Position(x=px, y=py)
                for px, py in self.position_history.get(agent_id, [])[:5]
            ],
            comm_limits=CommLimits(
                range=radio_range,
                max_outbound_per_turn=1,
                max_payload_chars=96,
            ),
            mark_limits=MarkLimits(max_ttl=12, allow_mark_info_broadcast=True),
            goal_sensor=self._bearing_sensor(ax, ay),
            last_move_outcome=self.last_move_outcome.get(agent_id, MoveOutcome.OK),
            contended_neighbors=self.contended_neighbors.get(agent_id, 0),
            history=[
                TurnHistory.model_validate(item)
                for item in list(self.turn_history.get(agent_id, []))
            ],
        )
        return obs

    def _render_patch(self, cx: int, cy: int, radius: int) -> LocalPatch:
        rows: List[str] = []
        active_positions = {
            pos for aid, pos in self.occupancy.items() if not self.is_finished(aid)
        }
        for dy in range(-radius, radius + 1):
            line_chars: List[str] = []
            for dx in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if not self._in_bounds(x, y):
                    line_chars.append("#")
                    continue
                ch: TileChar = "."
                if (x, y) in self.walls:
                    ch = "#"
                elif (x, y) == (self.goal.x, self.goal.y):
                    ch = "G"
                elif (x, y) in active_positions:
                    ch = "A"
                elif (x, y) in self.artifacts:
                    ch = "*"
                line_chars.append(ch)
            rows.append("".join(line_chars))
        top_left = Position(x=max(0, cx - radius), y=max(0, cy - radius))
        return LocalPatch(radius=radius, top_left_abs=top_left, rows=rows)

    def _neighbors_in_view(self, agent_id: str, radius: int) -> List[NeighborSummary]:
        cx, cy = self.occupancy[agent_id]
        neighbors: List[NeighborSummary] = []
        for other_id, (ox, oy) in self.occupancy.items():
            if other_id == agent_id:
                continue
            if self.is_finished(other_id):
                continue
            if abs(ox - cx) <= radius and abs(oy - cy) <= radius:
                neighbors.append(
                    NeighborSummary(
                        agent_id=other_id,
                        abs_pos=Position(x=ox, y=oy),
                        rel=RelativeOffset(dx=ox - cx, dy=oy - cy),
                    )
                )
        return neighbors

    def _artifacts_in_view(
        self, cx: int, cy: int, radius: int
    ) -> List[PlacedArtifact]:
        results: List[PlacedArtifact] = []
        for (ax, ay), artifact in self.artifacts.items():
            if abs(ax - cx) <= radius and abs(ay - cy) <= radius:
                results.append(artifact)
        return results

    def _has_active_no_go(self, x: int, y: int) -> bool:
        artifact = self.artifacts.get((x, y))
        return isinstance(artifact, ArtifactNoGo) and artifact.ttl_remaining > 0

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def _adjacent_summary(self, agent_id: str, ax: int, ay: int) -> List[AdjacentCell]:
        active_positions = {
            pos: other_id
            for other_id, pos in self.occupancy.items()
            if not self.is_finished(other_id)
        }
        mask = self.contended_neighbors.get(agent_id, 0)
        summary: List[AdjacentCell] = []
        for idx, (dir_name, delta) in enumerate({"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}.items()):
            dx, dy = delta
            nx, ny = ax + dx, ay + dy
            if not self._in_bounds(nx, ny):
                state = AdjacentState.OUT_OF_BOUNDS
            elif (nx, ny) in self.walls:
                state = AdjacentState.WALL
            elif (nx, ny) == (self.goal.x, self.goal.y):
                state = AdjacentState.GOAL
            elif (nx, ny) in active_positions and active_positions[(nx, ny)] != agent_id:
                state = AdjacentState.AGENT
            else:
                state = AdjacentState.FREE

            if state == AdjacentState.FREE and mask & (1 << idx):
                state = AdjacentState.CONTENDED

            if state in (AdjacentState.FREE, AdjacentState.CONTENDED) and self._has_active_no_go(nx, ny):
                state = AdjacentState.NO_GO

            summary.append(AdjacentCell(dir=dir_name, state=state))
        return summary

    def _record_position(self, agent_id: str) -> None:
        history = self.position_history.setdefault(agent_id, [])
        current = self.occupancy.get(agent_id)
        if current is None:
            return
        if history and history[0] == current:
            return
        history.insert(0, current)
        if len(history) > self.history_limit:
            del history[self.history_limit :]

    def _bearing_sensor(self, x: int, y: int) -> GoalSensorReading:
        if self.rng.random() < self.bearing_drop_p:
            return GoalSensorBearing(bearing=None, strength=None, available=False)

        dx = self.goal.x - x
        dy = self.goal.y - y
        if dx == 0 and dy == 0:
            bearing = Octant.N
        else:
            angle = math.degrees(math.atan2(-dy, dx)) % 360.0
            bins = [
                Octant.E,
                Octant.NE,
                Octant.N,
                Octant.NW,
                Octant.W,
                Octant.SW,
                Octant.S,
                Octant.SE,
            ]
            idx = int((angle + 22.5) // 45) % 8
            bearing = bins[idx]
        if self.bearing_bias_seed is not None:
            steps = self._bias_steps(
                x,
                y,
                self.bearing_bias_seed,
                self.bearing_bias_p,
                self.bearing_bias_wall_bonus,
            )
            if steps != 0:
                bearing = self._rotate_octant(bearing, steps)
        if self.rng.random() < self.bearing_flip_p:
            order = [
                Octant.N,
                Octant.NE,
                Octant.E,
                Octant.SE,
                Octant.S,
                Octant.SW,
                Octant.W,
                Octant.NW,
            ]
            j = (order.index(bearing) + self.rng.choice([-1, 1])) % 8
            bearing = order[j]

        manhattan = abs(dx) + abs(dy)
        if manhattan <= 4:
            strength = StrengthBucket.NEAR
        elif manhattan <= 10:
            strength = StrengthBucket.MID
        else:
            strength = StrengthBucket.FAR
        return GoalSensorBearing(bearing=bearing, strength=strength, available=True)

    def _neighbor_has_wall(self, x: int, y: int) -> bool:
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if (nx, ny) in self.walls:
                return True
        return False

    def _bias_steps(
        self,
        x: int,
        y: int,
        seed: int,
        base_prob: float,
        wall_bonus: float,
    ) -> int:
        if base_prob <= 0 and wall_bonus <= 0:
            return 0
        h = ((x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)) & 0xFFFFFFFF
        primary = ((h >> 8) & 0xFFFF) / 65535.0
        secondary = (h & 0xFF) / 255.0
        bias_p = base_prob + (wall_bonus if self._neighbor_has_wall(x, y) else 0.0)
        bias_p = max(0.0, min(bias_p, 0.49))
        if primary < bias_p:
            return 1 if secondary < 0.5 else -1
        return 0

    @staticmethod
    def _rotate_octant(bearing: Octant, steps: int) -> Octant:
        order = [
            Octant.N,
            Octant.NE,
            Octant.E,
            Octant.SE,
            Octant.S,
            Octant.SW,
            Octant.W,
            Octant.NW,
        ]
        idx = order.index(bearing)
        return order[(idx + steps) % len(order)]

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def deliver_message(self, recipient_id: str, message: ReceivedMessage) -> None:
        envelope = message.envelope
        sender = getattr(envelope, "sender_id", None)
        if sender is not None and envelope.seq is None:
            envelope.seq = self.next_message_seq(sender)
        if recipient_id not in self.inboxes:
            self.inboxes[recipient_id] = []
        self.inboxes[recipient_id].append(message)

    def is_finished(self, agent_id: str) -> bool:
        return self.finished_agents.get(agent_id, False)

    def next_message_seq(self, sender_id: str) -> int:
        current = self.message_seq.get(sender_id, 0)
        self.message_seq[sender_id] = current + 1
        return current

    def increment_inbox_ages(self) -> None:
        for messages in self.inboxes.values():
            for msg in messages:
                msg.age += 1

    def mark_finished(self, agent_id: str) -> None:
        self.finished_agents[agent_id] = True
        history = self.position_history.get(agent_id)
        if history is not None and (self.goal.x, self.goal.y) not in history[:1]:
            history.insert(0, (self.goal.x, self.goal.y))
            if len(history) > self.history_limit:
                del history[self.history_limit :]
        self.last_move_outcome[agent_id] = MoveOutcome.FINISHED
        self.loop_counters[agent_id] = 0
        self.contended_neighbors[agent_id] = 0

    def record_history(self, agent_id: str, payload: dict) -> None:
        if agent_id not in self.turn_history:
            self.turn_history[agent_id] = deque(maxlen=self.history_limit)
        self.turn_history[agent_id].append(payload)

    # ------------------------------------------------------------------
    # Movement and artifacts
    # ------------------------------------------------------------------

    def resolve_moves(self, intents: Dict[str, Optional[Direction]]) -> Dict[str, MoveResult]:
        start_positions = {aid: self.occupancy[aid] for aid in self.occupancy.keys()}
        targets: Dict[str, Optional[Tuple[int, int]]] = {}
        proposed: Dict[str, Tuple[int, int]] = {}
        results: Dict[str, MoveResult] = {}

        for aid in self.occupancy.keys():
            sx, sy = start_positions[aid]
            if self.is_finished(aid):
                proposed[aid] = (sx, sy)
                targets[aid] = None
                results[aid] = MoveResult(final=(sx, sy), outcome=MoveOutcome.FINISHED, target=None, opponents=[], cause_cell=None)

        for agent_id, direction in intents.items():
            sx, sy = start_positions[agent_id]
            if self.is_finished(agent_id):
                continue
            if direction is None:
                proposed[agent_id] = (sx, sy)
                targets[agent_id] = None
                results[agent_id] = MoveResult(final=(sx, sy), outcome=MoveOutcome.YIELD, target=None, opponents=[], cause_cell=None)
                continue
            self.orientation[agent_id] = direction
            dx, dy = _direction_delta(direction)
            tx, ty = sx + dx, sy + dy
            targets[agent_id] = (tx, ty)
            if not self._in_bounds(tx, ty):
                proposed[agent_id] = (sx, sy)
                results[agent_id] = MoveResult(
                    final=(sx, sy),
                    outcome=MoveOutcome.BLOCK_OOB,
                    target=(tx, ty),
                    opponents=[],
                    cause_cell=(tx, ty),
                )
            elif not self._passable(tx, ty):
                proposed[agent_id] = (sx, sy)
                results[agent_id] = MoveResult(
                    final=(sx, sy),
                    outcome=MoveOutcome.BLOCK_WALL,
                    target=(tx, ty),
                    opponents=[],
                    cause_cell=(tx, ty),
                )
            else:
                proposed[agent_id] = (tx, ty)
                results[agent_id] = MoveResult(
                    final=(tx, ty),
                    outcome=MoveOutcome.OK,
                    target=(tx, ty),
                    opponents=[],
                    cause_cell=None,
                )

        for aid in self.occupancy.keys():
            if aid not in proposed:
                proposed[aid] = start_positions[aid]
                targets.setdefault(aid, None)
                results.setdefault(
                    aid,
                    MoveResult(final=start_positions[aid], outcome=MoveOutcome.OK, target=None, opponents=[], cause_cell=None),
                )

        occupants: Dict[Tuple[int, int], List[str]] = {}
        for aid, cell in proposed.items():
            if self.is_finished(aid):
                continue
            occupants.setdefault(cell, []).append(aid)

        swap_lookup: Dict[str, List[str]] = {}
        for aid, target in targets.items():
            if target is None or self.is_finished(aid):
                continue
            for other, other_target in targets.items():
                if other <= aid or self.is_finished(other):
                    continue
                if other_target is None:
                    continue
                if target == start_positions.get(other) and other_target == start_positions.get(aid):
                    swap_lookup.setdefault(aid, []).append(other)
                    swap_lookup.setdefault(other, []).append(aid)

        contested_cells: List[Tuple[int, int]] = []
        for cell, ids in occupants.items():
            if len(ids) == 1:
                aid = ids[0]
                if aid in swap_lookup:
                    self.occupancy[aid] = start_positions[aid]
                    opponents = swap_lookup[aid]
                    results[aid] = MoveResult(
                        final=start_positions[aid],
                        outcome=MoveOutcome.SWAP_CONFLICT,
                        target=targets.get(aid),
                        opponents=opponents,
                        cause_cell=targets.get(aid),
                    )
                    if targets.get(aid) is not None:
                        contested_cells.append(targets[aid])
                    continue

                self.occupancy[aid] = cell
                result = results[aid]
                if result.outcome == MoveOutcome.OK and cell == (self.goal.x, self.goal.y):
                    results[aid] = MoveResult(
                        final=cell,
                        outcome=MoveOutcome.FINISHED,
                        target=result.target,
                        opponents=result.opponents,
                        cause_cell=result.cause_cell,
                    )
                continue

            swap = False
            if len(ids) == 2:
                a, b = ids
                if targets.get(a) == start_positions.get(b) and targets.get(b) == start_positions.get(a):
                    swap = True

            for aid in ids:
                self.occupancy[aid] = start_positions[aid]
                opponents = [other for other in ids if other != aid]
                outcome = MoveOutcome.SWAP_CONFLICT if swap else MoveOutcome.BLOCK_AGENT
                results[aid] = MoveResult(
                    final=start_positions[aid],
                    outcome=outcome,
                    target=targets.get(aid),
                    opponents=opponents,
                    cause_cell=cell,
                )
            contested_cells.append(cell)

        for aid in intents.keys():
            self._record_position(aid)

        for cell in contested_cells:
            if self._in_bounds(*cell) and cell != (self.goal.x, self.goal.y) and cell not in self.walls:
                self._place_congestion_marker(cell)

        return results

    def place_artifact(self, agent_id: str, artifact: PlacedArtifact) -> None:
        if not isinstance(artifact, ArtifactNoGo):
            raise ValueError("Only NO_GO artifacts are supported.")
        ax, ay = self.occupancy[agent_id]
        self.artifacts[(ax, ay)] = ArtifactNoGo(
            kind="NO_GO",
            reason=artifact.reason,
            ttl_remaining=artifact.ttl_remaining,
        )

    def _place_congestion_marker(self, cell: Tuple[int, int]) -> None:
        existing = self.artifacts.get(cell)
        ttl = TRAFFIC_CONE_TTL
        if isinstance(existing, ArtifactNoGo):
            ttl = max(existing.ttl_remaining, ttl)
        self.artifacts[cell] = ArtifactNoGo(kind="NO_GO", reason=BlockReason.CONGESTION, ttl_remaining=ttl)

    def decay_artifacts(self) -> None:
        expired: List[Tuple[int, int]] = []
        for key, artifact in self.artifacts.items():
            ttl = artifact.ttl_remaining - 1
            if ttl <= 0:
                expired.append(key)
            else:
                if isinstance(artifact, ArtifactNoGo):
                    self.artifacts[key] = ArtifactNoGo(
                        kind="NO_GO", reason=artifact.reason, ttl_remaining=ttl
                    )
        for key in expired:
            self.artifacts.pop(key, None)

    # ------------------------------------------------------------------
    # Helpers for checking progress
    # ------------------------------------------------------------------

    def agent_on_goal(self, agent_id: str) -> bool:
        return self.occupancy[agent_id] == (self.goal.x, self.goal.y)

    def all_agents_on_goal(self, agent_ids: Iterable[str]) -> bool:
        return all(self.agent_on_goal(aid) for aid in agent_ids)

    # ------------------------------------------------------------------
    # Message constructors to keep schema usage centralised
    # ------------------------------------------------------------------

    @staticmethod
    def message_here(sender_id: str, seq: int, pos: Position, orientation: Direction) -> MsgHere:
        return MsgHere(kind="HERE", sender_id=sender_id, seq=seq, pos=pos, orientation=orientation)

    @staticmethod
    def message_intent(sender_id: str, seq: int, intent: str) -> MsgIntent:
        return MsgIntent(kind="INTENT", sender_id=sender_id, seq=seq, next_action=intent)

    @staticmethod
    def message_sense(
        sender_id: str,
        seq: int,
        at: Position,
        bearing: Optional[Octant],
        strength: Optional[StrengthBucket],
    ) -> MsgSense:
        return MsgSense(
            kind="SENSE",
            sender_id=sender_id,
            seq=seq,
            at=at,
            mode="BEARING",
            bearing=bearing,
            strength=strength,
        )

    @staticmethod
    def message_mark_info(sender_id: str, seq: int, artifact: PlacedArtifact) -> MsgMarkInfo:
        return MsgMarkInfo(kind="MARK_INFO", sender_id=sender_id, seq=seq, placed=artifact)

````

## File: src/llmgrid/env/maze_generator.py

````python
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from llmgrid.schema import Position


Direction = Tuple[int, int]

CARDINALS: Sequence[Direction] = [(0, -1), (1, 0), (0, 1), (-1, 0)]


@dataclass
class MazeConfig:
    width: int
    height: int
    seed: int
    extra_connection_prob: float = 0.0

    def __post_init__(self) -> None:
        if self.width < 3 or self.height < 3:
            raise ValueError("Maze dimensions must be at least 3x3.")
        if not (0.0 <= self.extra_connection_prob <= 1.0):
            raise ValueError("extra_connection_prob must be between 0 and 1.")


class MazeGenerator:
    """Generate carved mazes with guaranteed connectivity between supply points."""

    def __init__(self, config: MazeConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    def generate(
        self,
        required_open_cells: Iterable[Tuple[int, int]],
    ) -> List[Position]:
        """
        Generate a maze and return obstacle positions.

        The generator ensures that all `required_open_cells` are free and
        mutually reachable.
        """
        width, height = self.config.width, self.config.height
        grid = [[1 for _ in range(width)] for _ in range(height)]

        start_x, start_y = self._initial_cell()
        grid[start_y][start_x] = 0
        stack: List[Tuple[int, int]] = [(start_x, start_y)]

        while stack:
            cx, cy = stack[-1]
            neighbors = self._uncarved_neighbors(cx, cy, grid)
            if not neighbors:
                stack.pop()
                continue
            nx, ny, wx, wy = self.rng.choice(neighbors)
            grid[wy][wx] = 0  # carve hallway between cells
            grid[ny][nx] = 0  # carve the neighbor cell
            stack.append((nx, ny))

        if self.config.extra_connection_prob > 0:
            self._add_extra_connections(grid)

        for cell in required_open_cells:
            x, y = cell
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = 0

        self._ensure_connection(grid, required_open_cells)

        obstacles = [
            Position(x=x, y=y)
            for y in range(height)
                for x in range(width)
                    if grid[y][x] == 1
        ]
        return obstacles

    def render_ascii(self, obstacles: Sequence[Position]) -> str:
        width, height = self.config.width, self.config.height
        obstacle_set = {(p.x, p.y) for p in obstacles}
        rows: List[str] = []
        for y in range(height):
            line = []
            for x in range(width):
                line.append("#" if (x, y) in obstacle_set else ".")
            rows.append("".join(line))
        return "\n".join(rows)

    def _initial_cell(self) -> Tuple[int, int]:
        # use odd coordinates to maximise uniformity
        x = 1 if self.config.width > 1 else 0
        y = 1 if self.config.height > 1 else 0
        return x, y

    def _uncarved_neighbors(
        self,
        x: int,
        y: int,
        grid: List[List[int]],
    ) -> List[Tuple[int, int, int, int]]:
        neighbors: List[Tuple[int, int, int, int]] = []
        width, height = self.config.width, self.config.height
        for dx, dy in CARDINALS:
            nx = x + dx * 2
            ny = y + dy * 2
            if (
                0 <= nx < width
                and 0 <= ny < height
                and grid[ny][nx] == 1
            ):
                wx = x + dx
                wy = y + dy
                neighbors.append((nx, ny, wx, wy))
        return neighbors

    def _add_extra_connections(self, grid: List[List[int]]) -> None:
        width, height = self.config.width, self.config.height
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if grid[y][x] == 1 and self.rng.random() < self.config.extra_connection_prob:
                    # Only carve if it connects two distinct corridors
                    passages = sum(
                        grid[y + dy][x + dx] == 0
                        for dx, dy in CARDINALS
                    )
                    if passages >= 2:
                        grid[y][x] = 0

    def _ensure_connection(
        self,
        grid: List[List[int]],
        required_open_cells: Iterable[Tuple[int, int]],
    ) -> None:
        cells = list(required_open_cells)
        if not cells:
            return
        base = cells[0]
        reachable = self._flood_fill(grid, base)
        for cell in cells[1:]:
            if cell not in reachable:
                self._carve_path(grid, base, cell)
                reachable = self._flood_fill(grid, base)

    def _flood_fill(
        self,
        grid: List[List[int]],
        start: Tuple[int, int],
    ) -> set[Tuple[int, int]]:
        width, height = self.config.width, self.config.height
        sx, sy = start
        if not (0 <= sx < width and 0 <= sy < height):
            return set()
        if grid[sy][sx] == 1:
            return set()
        visited: set[Tuple[int, int]] = {start}
        queue: List[Tuple[int, int]] = [start]
        while queue:
            x, y = queue.pop()
            for dx, dy in CARDINALS:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < width
                    and 0 <= ny < height
                    and grid[ny][nx] == 0
                    and (nx, ny) not in visited
                ):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return visited

    def _carve_path(
        self,
        grid: List[List[int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> None:
        width, height = self.config.width, self.config.height
        sx, sy = start
        gx, gy = goal
        sx = min(max(sx, 0), width - 1)
        sy = min(max(sy, 0), height - 1)
        gx = min(max(gx, 0), width - 1)
        gy = min(max(gy, 0), height - 1)

        grid[sy][sx] = 0
        grid[gy][gx] = 0

        cx, cy = sx, sy
        while (cx, cy) != (gx, gy):
            if cx < gx:
                cx += 1
            elif cx > gx:
                cx -= 1
            elif cy < gy:
                cy += 1
            else:
                cy -= 1
            grid[cy][cx] = 0


````

## File: src/llmgrid/cli/poc_two_agents.py

````python
"""CLI entry point for the two-agent proof of concept."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, TextIO
from collections import deque

import typer

from llmgrid.env.simulate import EpisodeCheckpoint, EpisodeMetrics, run_episode
from llmgrid.env.maze_generator import MazeConfig, MazeGenerator
from llmgrid.schema import Position
from llmgrid.logging.episode_log import (
    AgentState as LogAgentState,
    AgentStyle,
    EpisodeLog,
    EpisodeMeta,
    Frame as LogFrame,
    GridSize,
    NoGoCell,
    Position as LogPosition,
    ViewShape,
)

app = typer.Typer(add_completion=False)


MAZE_PRESETS = {
    "long_corridor": {
        "width": 30,
        "height": 10,
        "style": "maze",
        "density": None,
        "extra": 0.2,
        "seed": 606,
        "description": "Wide horizontal corridor network with loops near the goal.",
    },
    "open_sparse": {
        "width": 20,
        "height": 12,
        "style": "random",
        "density": 0.12,
        "extra": 0.0,
        "seed": 101,
        "description": "Light scatter of obstacles for almost-open navigation.",
    },
    "open_dense": {
        "width": 20,
        "height": 12,
        "style": "random",
        "density": 0.25,
        "extra": 0.0,
        "seed": 202,
        "description": "Heavier scatter with narrow passages and pockets.",
    },
    "maze_tight": {
        "width": 21,
        "height": 13,
        "style": "maze",
        "density": None,
        "extra": 0.05,
        "seed": 303,
        "description": "Classic single-path maze with few shortcuts.",
    },
    "maze_loops": {
        "width": 21,
        "height": 13,
        "style": "maze",
        "density": None,
        "extra": 0.35,
        "seed": 404,
        "description": "Maze with many cross-links and alternate loops.",
    },
    "mixed_medium": {
        "width": 24,
        "height": 14,
        "style": "random",
        "density": 0.18,
        "extra": 0.15,
        "seed": 505,
        "description": "Combination of scatter and short corridors.",
    },
}


@app.command()
def main(
    model: str = typer.Option(
        ...,
        "--model",
        help="Fully qualified model id, e.g. openrouter:openai/gpt-oss-20b:free",
    ),
    width: int = typer.Option(12, "--width", help="Grid width."),
    height: int = typer.Option(12, "--height", help="Grid height."),
    visibility: int = typer.Option(1, "--visibility", help="Visibility radius R."),
    radio_range: int = typer.Option(2, "--radio-range", help="Radio range r."),
    turns: int = typer.Option(120, "--turns", help="Turn budget."),
    seed: int = typer.Option(13, "--seed", help="Random seed."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Use the local heuristic baseline instead of calling the LLM."
    ),
    no_obstacles: bool = typer.Option(
        False, "--no-obstacles", help="Start with an empty grid (no static obstacles)."
    ),
    obstacle_density: Optional[float] = typer.Option(
        None,
        "--obstacle-density",
        help="Optional fraction of free cells to fill with random obstacles (0-1). Overrides --no-obstacles.",
    ),
    obstacle_count: Optional[int] = typer.Option(
        None,
        "--obstacle-count",
        help="Optional absolute number of random obstacles. Overrides --no-obstacles and --obstacle-density.",
    ),
    obstacle_seed: Optional[int] = typer.Option(
        None,
        "--obstacle-seed",
        help="Seed for random obstacle placement (defaults to --seed when generating random obstacles).",
    ),
    bearing_bias_seed: Optional[int] = typer.Option(
        None,
        "--bearing-bias-seed",
        help="Enable Gold Drift by setting a deterministic seed (default: disabled).",
    ),
    bearing_bias_p: float = typer.Option(
        0.0,
        "--bearing-bias-p",
        min=0.0,
        max=0.49,
        help="Baseline probability of rotating the bearing by ±45° when bias is enabled.",
    ),
    bearing_bias_wall_bonus: float = typer.Option(
        0.0,
        "--bearing-bias-wall-bonus",
        min=0.0,
        max=0.49,
        help="Additional probability added when the cell touches a wall.",
    ),
    maze_preset: str = typer.Option(
        "long_corridor",
        "--maze-preset",
        help="Curated maze preset name (long_corridor, open_sparse, open_dense, maze_tight, maze_loops, mixed_medium) or 'none' for custom settings.",
    ),
    maze_style: str = typer.Option(
        "maze",
        "--maze-style",
        help="Obstacle generator style: 'maze' (default) or 'random'.",
    ),
    maze_extra_connection: float = typer.Option(
        0.1,
        "--maze-extra-connection",
        help="For maze-style obstacles, probability of carving additional side passages (0 to 1).",
    ),
    log_prompts: bool = typer.Option(
        False,
        "--log-prompts/--no-log-prompts",
        help="Capture full prompts and structured outputs for every agent turn.",
    ),
    log_movements: bool = typer.Option(
        True,
        "--log-movements/--no-log-movements",
        help="Capture agent locations per turn for downstream visualization.",
    ),
    transcript_jsonl: Optional[Path] = typer.Option(
        None,
        "--transcript-jsonl",
        help="Optional path for the prompt transcript (defaults to results/transcript.jsonl if --emit-config is set).",
    ),
    episode_json: Optional[Path] = typer.Option(
        None,
        "--episode-json",
        help="Optional path for the EpisodeLog JSON (defaults to results/episode.json if --emit-config is set).",
    ),
    emit_config: Optional[Path] = typer.Option(
        None,
        "--emit-config",
        help="Optional path to dump the resolved configuration YAML.",
    ),
    checkpoint_json: Optional[Path] = typer.Option(
        None,
        "--checkpoint-json",
        help="Path to write periodic checkpoints for resuming interrupted runs.",
    ),
    checkpoint_interval: int = typer.Option(
        1,
        "--checkpoint-interval",
        min=1,
        help="Turns between checkpoint writes (default: every turn).",
    ),
    resume_from: Optional[Path] = typer.Option(
        None,
        "--resume-from",
        help="Resume a partially completed run from an existing checkpoint JSON file.",
    ),
    agents: int = typer.Option(
        2,
        "--agents",
        min=1,
        max=8,
        help="Number of controllable agents to spawn (default: 2).",
    ),
    comm_strategy: str = typer.Option(
        "none",
        "--comm-strategy",
        help="Communication strategy: none, intent, negotiation, freeform, or oracle.",
    ),
    reasoning_effort: str = typer.Option(
        "minimal",
        "--reasoning-effort",
        help="Reasoning effort for GPT-5 models (minimal, low, medium, high). Ignored by non-reasoning models.",
    ),
    reasoning_verbosity: str = typer.Option(
        "high",
        "--reasoning-verbosity",
        help="Reasoning verbosity (low, medium, high). Only applicable to reasoning models.",
    ),
    reasoning_include_encrypted: bool = typer.Option(
        False,
        "--reasoning-include-encrypted",
        help="Include encrypted reasoning trace in responses (useful when forwarding chain of thought).",
    ),
    history_limit: int = typer.Option(
        5,
        "--history-limit",
        min=1,
        max=20,
        help="Number of prior turns to include in observation history (default: 5).",
    ),
    loop_guidance: str = typer.Option(
        "passive",
        "--loop-guidance",
        help="Loop-handling instructions: passive, active (break loops >=3), or explore (aggressive loop escape).",
    ),
) -> None:
    # Validate model prefix
    if not (model.startswith("openrouter:") or model.startswith("azure:")):
        typer.secho(
            "Error: use --model with openrouter: or azure: prefix (e.g. azure:gpt-5-mini).",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    comm_strategy = comm_strategy.lower()
    loop_guidance = loop_guidance.lower()
    allowed_strategies = {"none", "intent", "negotiation", "freeform", "oracle"}
    if comm_strategy not in allowed_strategies:
        typer.secho(
            f"Unknown communication strategy '{comm_strategy}'. Choose from: {', '.join(sorted(allowed_strategies))}.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    allowed_loop_guidance = {"passive", "active", "explore"}
    if loop_guidance not in allowed_loop_guidance:
        typer.secho(
            f"Unknown loop-guidance '{loop_guidance}'. Choose from: {', '.join(sorted(allowed_loop_guidance))}.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    oracle_enabled = comm_strategy == "oracle"

    # Validate logging flags require output paths
    if log_prompts and emit_config is None:
        typer.secho(
            "Error: --log-prompts requires --emit-config to specify where transcript.jsonl should be written.",
            fg=typer.colors.RED,
        )
        typer.secho(
            "Example: --emit-config experiments/my-run/config.yaml",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=2)

    if log_movements and emit_config is None and episode_json is None:
        typer.secho(
            "Error: --log-movements requires --emit-config or --episode-json to specify where episode logs should be written.",
            fg=typer.colors.RED,
        )
        typer.secho(
            "Example: --emit-config experiments/my-run/config.yaml",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=2)

    preset_name = maze_preset.lower()
    resume_checkpoint: Optional[EpisodeCheckpoint] = None
    if resume_from is not None:
        try:
            resume_checkpoint = EpisodeCheckpoint.load(resume_from)
        except FileNotFoundError as exc:
            typer.secho(f"Checkpoint not found: {resume_from}", fg=typer.colors.RED)
            raise typer.Exit(code=2) from exc

        if model != resume_checkpoint.model_id:
            typer.secho(
                f"Model mismatch: checkpoint expects '{resume_checkpoint.model_id}', got '{model}'.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)

        expected_dry_run = not resume_checkpoint.use_llm
        if dry_run != expected_dry_run:
            mode_msg = "dry-run" if expected_dry_run else "LLM-backed"
            typer.secho(
                f"Checkpoint was recorded for a {mode_msg} run. Adjust --dry-run accordingly.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)

        checkpoint_strategy = getattr(resume_checkpoint, "comm_strategy", "none")
        if comm_strategy != checkpoint_strategy:
            typer.secho(
                f"Checkpoint was recorded with --comm-strategy {checkpoint_strategy!r}; got {comm_strategy!r}.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)
        checkpoint_history = getattr(resume_checkpoint, "history_limit", history_limit)
        if history_limit != checkpoint_history:
            typer.secho(
                f"Checkpoint was recorded with --history-limit {checkpoint_history}; got {history_limit}.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)
        history_limit = checkpoint_history
        checkpoint_loop_guidance = getattr(resume_checkpoint, "loop_guidance", loop_guidance)
        if loop_guidance != checkpoint_loop_guidance:
            typer.secho(
                f"Checkpoint was recorded with --loop-guidance {checkpoint_loop_guidance!r}; got {loop_guidance!r}.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)
        loop_guidance = checkpoint_loop_guidance

        width = resume_checkpoint.world.width
        height = resume_checkpoint.world.height
        visibility = resume_checkpoint.visibility
        radio_range = resume_checkpoint.radio_range
        turns = resume_checkpoint.turns_total
        seed = resume_checkpoint.seed
        preset_name = resume_checkpoint.maze_metadata.get("preset", preset_name)
        maze_style = resume_checkpoint.maze_metadata.get("maze_style", maze_style)
        maze_extra_connection = resume_checkpoint.maze_metadata.get("maze_extra_connection", maze_extra_connection)
        no_obstacles = resume_checkpoint.maze_metadata.get("no_obstacles", no_obstacles)
        obstacle_density = resume_checkpoint.maze_metadata.get("obstacle_density", obstacle_density)
        obstacle_count = resume_checkpoint.maze_metadata.get("obstacle_count", obstacle_count)
        obstacle_seed = resume_checkpoint.maze_metadata.get("obstacle_seed", obstacle_seed)
        bearing_bias_seed = resume_checkpoint.maze_metadata.get("bearing_bias_seed", bearing_bias_seed)
        bearing_bias_p = resume_checkpoint.maze_metadata.get("bearing_bias_p", bearing_bias_p)
        bearing_bias_wall_bonus = resume_checkpoint.maze_metadata.get("bearing_bias_wall_bonus", bearing_bias_wall_bonus)
        typer.secho(
            f"Resuming from {resume_from} at turn {resume_checkpoint.turn_next}/{turns}",
            fg=typer.colors.BLUE,
        )

    preset_details = None
    if resume_checkpoint is None and preset_name != "none":
        preset_details = MAZE_PRESETS.get(preset_name)
        if preset_details is None:
            typer.secho(
                f"Unknown maze preset '{maze_preset}'. Available presets: {', '.join(MAZE_PRESETS.keys())}, or 'none'.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)
        width = preset_details["width"]
        height = preset_details["height"]
        maze_style = preset_details["style"]
        maze_extra_connection = preset_details["extra"]
        obstacle_density = preset_details["density"]
        obstacle_seed = preset_details["seed"]
        obstacle_count = None
        no_obstacles = False
        typer.secho(
            f"Using maze preset '{preset_name}' (seed={obstacle_seed}) — {preset_details['description']}",
            fg=typer.colors.BLUE,
        )

    checkpoint_path = checkpoint_json if checkpoint_json is not None else resume_from

    maze_metadata: Dict[str, Optional[float | int | str | bool]] = {
        "preset": preset_name,
        "maze_style": maze_style,
        "maze_extra_connection": maze_extra_connection,
        "no_obstacles": no_obstacles,
        "obstacle_density": obstacle_density,
        "obstacle_count": obstacle_count,
        "obstacle_seed": obstacle_seed,
        "bearing_bias_seed": bearing_bias_seed,
        "bearing_bias_p": bearing_bias_p,
        "bearing_bias_wall_bonus": bearing_bias_wall_bonus,
        "agents": agents,
        "comm_strategy": comm_strategy,
        "history_limit": history_limit,
        "loop_guidance": loop_guidance,
    }

    if emit_config and resume_checkpoint is None:
        _write_config(
            emit_config,
            {
                "model": model,
                "width": width,
                "height": height,
                "visibility": visibility,
                "radio_range": radio_range,
                "turns": turns,
                "seed": seed,
                "obstacle_density": obstacle_density,
                "obstacle_count": obstacle_count,
                "obstacle_seed": obstacle_seed,
                "maze_preset": preset_name,
                "maze_style": maze_style,
                "maze_extra_connection": maze_extra_connection,
                "dry_run": dry_run,
                "no_obstacles": no_obstacles,
                "bearing_bias_seed": bearing_bias_seed,
                "bearing_bias_p": bearing_bias_p,
                "bearing_bias_wall_bonus": bearing_bias_wall_bonus,
                "agents": agents,
                "comm_strategy": comm_strategy,
                "history_limit": history_limit,
                "loop_guidance": loop_guidance,
            },
        )

    if resume_checkpoint is not None:
        start_positions = resume_checkpoint.start_positions
        goal = resume_checkpoint.goal
        obstacles = [Position(x=p.x, y=p.y) for p in resume_checkpoint.world.walls]
    else:
        goal = _default_goal(width, height)
        start_positions = _default_start_positions(width, height, goal, agents, seed=seed)
        obstacles = _resolve_obstacles(
            width=width,
            height=height,
            start_positions=start_positions,
            goal=goal,
            no_obstacles=no_obstacles,
            obstacle_count=obstacle_count,
            obstacle_density=obstacle_density,
            obstacle_seed=obstacle_seed if obstacle_seed is not None else seed,
            maze_style=maze_style,
            maze_extra_connection=maze_extra_connection,
            agent_count=agents,
            start_seed=seed,
        )

    capture_transcript = log_prompts or transcript_jsonl is not None
    if capture_transcript:
        if resume_checkpoint and resume_checkpoint.transcript is not None:
            transcript_records = list(resume_checkpoint.transcript)
        else:
            transcript_records = []
    else:
        transcript_records = None

    transcript_path: Optional[Path] = transcript_jsonl
    if resume_checkpoint and resume_checkpoint.transcript_path:
        checkpoint_transcript_path = Path(resume_checkpoint.transcript_path).expanduser()
        if transcript_path is None:
            transcript_path = checkpoint_transcript_path
        else:
            if transcript_path.expanduser().resolve(strict=False) != checkpoint_transcript_path.resolve(strict=False):
                typer.secho(
                    "Transcript path differs from checkpoint. Re-run without --transcript-jsonl to reuse the stored path.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=2)

    if capture_transcript and transcript_path is None and emit_config is not None:
        transcript_path = emit_config.parent / "results" / "transcript.jsonl"
    if capture_transcript and transcript_path is None:
        typer.secho(
            "Transcript capture requested but no output path available. Provide --emit-config or --transcript-jsonl.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    capture_movement = log_movements or episode_json is not None
    if capture_movement:
        if resume_checkpoint and resume_checkpoint.movement is not None:
            movement_records = list(resume_checkpoint.movement)
        else:
            movement_records = []
    else:
        movement_records = None

    episode_path: Optional[Path] = episode_json
    if resume_checkpoint and resume_checkpoint.episode_path:
        checkpoint_episode_path = Path(resume_checkpoint.episode_path).expanduser()
        if episode_path is None:
            episode_path = checkpoint_episode_path
        else:
            if episode_path.expanduser().resolve(strict=False) != checkpoint_episode_path.resolve(strict=False):
                typer.secho(
                    "Episode log path differs from checkpoint. Re-run without --episode-json to reuse the stored path.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=2)

    if capture_movement and episode_path is None and emit_config is not None:
        episode_path = emit_config.parent / "results" / "episode.json"
    if capture_movement and episode_path is None:
        typer.secho(
            "Movement logging requested but no output path available. Provide --emit-config or --episode-json.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    movement_stream_path: Optional[Path] = None
    if resume_checkpoint and resume_checkpoint.movement_stream_path:
        movement_stream_path = Path(resume_checkpoint.movement_stream_path).expanduser()

    agent_order = resume_checkpoint.agent_ids if resume_checkpoint else list(start_positions.keys())

    transcript_handle: Optional[TextIO] = None
    movement_stream_handle: Optional[TextIO] = None
    try:
        if transcript_path is not None:
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if resume_checkpoint is not None else "w"
            transcript_handle = transcript_path.open(mode, encoding="utf-8")

        if capture_movement and episode_path is not None:
            if movement_stream_path is None:
                movement_stream_path = episode_path.with_name(episode_path.stem + "_stream.jsonl")
            movement_stream_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if resume_checkpoint is not None else "w"
            movement_stream_handle = movement_stream_path.open(mode, encoding="utf-8")

        metrics = run_episode(
            use_llm=not dry_run,
            model_id=model,
            width=width,
            height=height,
            obstacles=obstacles,
            start_positions=start_positions,
            goal=goal,
            turns=turns,
            visibility=visibility,
            radio_range=radio_range,
            seed=seed,
            transcript=transcript_records,
            movement=movement_records,
            transcript_writer=transcript_handle,
            movement_writer=movement_stream_handle,
            agent_order=agent_order,
            resume=resume_checkpoint,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            transcript_path=str(transcript_path) if transcript_path is not None else None,
            movement_stream_path=str(movement_stream_path) if movement_stream_path is not None else None,
            episode_path=str(episode_path) if episode_path is not None else None,
            maze_metadata={k: v for k, v in maze_metadata.items() if v is not None},
            bearing_bias_seed=bearing_bias_seed,
            bearing_bias_p=bearing_bias_p,
            bearing_bias_wall_bonus=bearing_bias_wall_bonus,
            comm_strategy=comm_strategy,
            history_limit=history_limit,
            loop_guidance=loop_guidance,
            oracle_enabled=oracle_enabled,
            reasoning_effort=reasoning_effort,
            reasoning_verbosity=reasoning_verbosity,
            reasoning_include_encrypted=reasoning_include_encrypted,
        )
    finally:
        if transcript_handle is not None:
            transcript_handle.flush()
            transcript_handle.close()
        if movement_stream_handle is not None:
            movement_stream_handle.flush()
            movement_stream_handle.close()

    typer.secho(json.dumps(metrics.__dict__, indent=2), fg=typer.colors.GREEN)

    if emit_config is not None:
        results_dir = emit_config.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = results_dir / "metrics.json"
        metrics_payload = {
            "model": model,
            "comm_strategy": comm_strategy,
            "seed": seed,
            "agents": agents,
            "maze_preset": preset_name,
            "success": metrics.success,
            "turns": metrics.turns,
            "collisions": metrics.collisions,
            "messages_sent": metrics.messages_sent,
            "collision_causes": metrics.collision_causes,
            "hazard_events": metrics.hazard_events,
            "comments_clamped": metrics.comments_clamped,
            "comments_autofilled": metrics.comments_autofilled,
            "no_go_exposures": metrics.no_go_exposures,
            "contended_exposures": metrics.contended_exposures,
            "history_limit": metrics.history_limit,
            "loop_guidance": metrics.loop_guidance,
            "oracle_requests": metrics.oracle_requests,
            "oracle_enabled": oracle_enabled,
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        typer.secho(f"Metrics saved to {metrics_path}", fg=typer.colors.BLUE)

    if transcript_records is not None and transcript_path is not None:
        if transcript_handle is None:
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            with transcript_path.open("w", encoding="utf-8") as handle:
                for record in transcript_records:
                    handle.write(json.dumps(record))
                    handle.write("\n")
            typer.secho(f"Prompt transcript saved to {transcript_path}", fg=typer.colors.BLUE)
        else:
            typer.secho(f"Prompt transcript streaming to {transcript_path}", fg=typer.colors.BLUE)

    if movement_records is not None and episode_path is not None:
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        for entry in movement_records:
            agents = []
            for aid, payload in sorted(entry["agents"].items()):
                agents.append(
                    LogAgentState(
                        agent_id=aid,
                        pos=LogPosition(x=payload["x"], y=payload["y"]),
                        orientation=payload.get("orientation"),
                        action=payload.get("action"),
                        status=payload.get("status", "ACTIVE"),
                    )
                )
            hazards = [
                NoGoCell(pos=LogPosition(x=haz["x"], y=haz["y"]), ttl=haz["ttl"])
                for haz in entry.get("hazards", [])
            ]
            frames.append(LogFrame(t=entry["turn"], agents=agents, hazards=hazards))

        agent_styles = _default_agent_styles(sorted(start_positions.keys()))
        episode_log = EpisodeLog(
            meta=EpisodeMeta(
                grid_size=GridSize(width=width, height=height),
                goal=LogPosition(x=goal.x, y=goal.y),
                walls=[LogPosition(x=p.x, y=p.y) for p in obstacles],
                view=ViewShape(kind="square", radius=visibility),
                gradient_mode="bfs",
                title=f"{model} R={visibility}",
                agent_styles=agent_styles,
            ),
            frames=frames,
        )

        with episode_path.open("w", encoding="utf-8") as handle:
            handle.write(episode_log.model_dump_json(indent=2))
        typer.secho(f"Episode log saved to {episode_path}", fg=typer.colors.BLUE)
        if movement_stream_path is not None:
            typer.secho(f"Movement stream saved to {movement_stream_path}", fg=typer.colors.BLUE)


def _write_config(path: Path, data: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def _resolve_obstacles(
    *,
    width: int,
    height: int,
    start_positions: Optional[dict[str, Position]],
    goal: Optional[Position],
    no_obstacles: bool,
    obstacle_count: Optional[int],
    obstacle_density: Optional[float],
    obstacle_seed: int,
    maze_style: str,
    maze_extra_connection: float,
    agent_count: int,
    start_seed: Optional[int],
    max_attempts: int = 100,
) -> list[Position]:
    style = maze_style.lower()
    start_positions = start_positions or _default_start_positions(
        width,
        height,
        goal or _default_goal(width, height),
        agent_count,
        seed=start_seed,
    )
    goal = goal or _default_goal(width, height)

    if style == "maze" and not no_obstacles:
        config = MazeConfig(
            width=width,
            height=height,
            seed=obstacle_seed,
            extra_connection_prob=maze_extra_connection,
        )
        generator = MazeGenerator(config)
        required_cells = [(pos.x, pos.y) for pos in start_positions.values()]
        required_cells.append((goal.x, goal.y))
        return generator.generate(required_open_cells=required_cells)

    if obstacle_count is None and obstacle_density is None:
        return [] if no_obstacles else _default_obstacles(width, height)

    total_cells = width * height
    if total_cells <= 0:
        return []

    if obstacle_count is not None and obstacle_count < 0:
        raise typer.BadParameter("obstacle-count must be non-negative.")

    if obstacle_density is not None:
        if not 0 <= obstacle_density <= 1:
            raise typer.BadParameter("obstacle-density must be between 0 and 1.")

    rng = random.Random(obstacle_seed)

    forbidden = {(goal.x, goal.y)}
    forbidden.update((pos.x, pos.y) for pos in start_positions.values())

    available: list[tuple[int, int]] = [
        (x, y) for x in range(width) for y in range(height) if (x, y) not in forbidden
    ]
    if not available:
        return []

    if obstacle_count is not None:
        count = min(obstacle_count, len(available))
    else:
        count = int(round(obstacle_density * len(available)))  # type: ignore[arg-type]
        count = max(0, min(count, len(available)))

    # attempt sampling until connectivity holds
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        sampled = rng.sample(available, count)
        sampled_set = set(sampled)
        if _paths_exist(width, height, sampled_set, start_positions, goal):
            return [Position(x=x, y=y) for x, y in sampled_set]
    # fallback: no obstacles to guarantee progress
    typer.secho(
        "Warning: could not sample reachable obstacle layout after "
        f"{max_attempts} attempts. Falling back to empty grid.",
        fg=typer.colors.YELLOW,
    )
    return []


def _neighbors(x: int, y: int) -> list[tuple[int, int]]:
    return [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]


def _paths_exist(
    width: int,
    height: int,
    obstacles: set[tuple[int, int]],
    start_positions: dict[str, Position],
    goal: Position,
) -> bool:
    gx, gy = goal.x, goal.y
    if (gx, gy) in obstacles:
        return False
    for pos in start_positions.values():
        if not _reachable(width, height, obstacles, (pos.x, pos.y), (gx, gy)):
            return False
    return True


def _reachable(
    width: int,
    height: int,
    obstacles: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> bool:
    if start == goal:
        return True
    visited = set()
    dq = deque([start])
    visited.add(start)
    while dq:
        x, y = dq.popleft()
        for nx, ny in _neighbors(x, y):
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in obstacles or (nx, ny) in visited:
                continue
            if (nx, ny) == goal:
                return True
            visited.add((nx, ny))
            dq.append((nx, ny))
    return False


def _default_obstacles(width: int, height: int) -> list[Position]:
    raw = [
        Position(x=4, y=2),
        Position(x=4, y=3),
        Position(x=4, y=4),
        Position(x=7, y=6),
        Position(x=7, y=7),
        Position(x=7, y=8),
        Position(x=2, y=max(0, height - 2)),
        Position(x=3, y=max(0, height - 2)),
    ]
    return [p for p in raw if p.x < width and p.y < height]


def _default_start_positions(
    width: int,
    height: int,
    goal: Position,
    agent_count: int,
    *,
    seed: Optional[int] = None,
) -> dict[str, Position]:
    min_distance = math.ceil((width + height) / 2)

    def manhattan(p: Position) -> int:
        return abs(goal.x - p.x) + abs(goal.y - p.y)

    candidates: List[Position] = []
    for y in range(height - 1, -1, -1):
        for x in range(0, width):
            if (x, y) == (goal.x, goal.y):
                continue
            pos = Position(x=x, y=y)
            if manhattan(pos) >= min_distance:
                candidates.append(pos)

    if len(candidates) < agent_count:
        # Relax requirement gradually until enough positions are available
        all_cells = [
            Position(x=x, y=y)
            for y in range(height - 1, -1, -1)
            for x in range(width)
            if (x, y) != (goal.x, goal.y)
        ]
        all_cells.sort(key=manhattan, reverse=True)
        for pos in all_cells:
            if pos not in candidates:
                candidates.append(pos)
            if len(candidates) >= agent_count:
                break

    rng = random.Random(seed)
    rng.shuffle(candidates)

    selection: List[Position] = []
    min_pairwise = max(2, min_distance // 4)
    for pos in candidates:
        if all(abs(pos.x - chosen.x) + abs(pos.y - chosen.y) >= min_pairwise for chosen in selection):
            selection.append(pos)
        if len(selection) == agent_count:
            break

    if len(selection) < agent_count:
        # Fill any remaining slots without the spacing constraint.
        for pos in candidates:
            if pos in selection:
                continue
            selection.append(pos)
            if len(selection) == agent_count:
                break

    return {f"a{i + 1}": selection[i] for i in range(agent_count)}


def _default_goal(width: int, height: int) -> Position:
    goal_x = max(0, width - 2)
    goal_y = min(1, max(0, height - 1))
    return Position(x=goal_x, y=goal_y)


DEFAULT_AGENT_COLORS = {
    "a1": "#1f77b4",
    "a2": "#d62728",
    "a3": "#2ca02c",
    "a4": "#9467bd",
    "a5": "#ff7f0e",
    "a6": "#17becf",
}


def _default_agent_styles(agent_ids: List[str]) -> List[AgentStyle]:
    styles: List[AgentStyle] = []
    palette_cycle = list(DEFAULT_AGENT_COLORS.values())
    for idx, aid in enumerate(agent_ids):
        color = DEFAULT_AGENT_COLORS.get(aid)
        if color is None:
            color = palette_cycle[idx % len(palette_cycle)]
        styles.append(AgentStyle(agent_id=aid, color_hex=color))
    return styles


if __name__ == "__main__":
    app()

````

## File: src/llmgrid/agent/llm_agent.py

````python
"""LLM-backed policy that produces structured actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llmgrid.llm_clients.unified_llm import UnifiedLLM
from llmgrid.prompts import STATIC_HEADER
from llmgrid.schema import Decision, Observation


@dataclass
class DecisionTrace:
    """Structured decision bundle that includes the raw prompt (trace messages optional)."""

    decision: Decision
    prompt: str
    trace_messages: List[dict]


class LlmPolicy:
    """Async wrapper that turns observations into structured decisions via UnifiedLLM."""

    def __init__(
        self,
        model_id: str,
        *,
        strategy: str,
        loop_guidance: str,
        history_limit: int,
        reasoning_effort: Optional[str] = None,
        reasoning_verbosity: Optional[str] = None,
        reasoning_include_encrypted: bool = False,
    ) -> None:
        self.model_id = model_id
        self.strategy = strategy
        self.loop_guidance = loop_guidance
        self.history_limit = max(1, history_limit)
        self.unified = UnifiedLLM()
        reasoning_payload: Dict[str, Any] = {}
        if reasoning_effort:
            reasoning_payload["effort"] = reasoning_effort
        if reasoning_verbosity:
            reasoning_payload["verbosity"] = reasoning_verbosity
        if reasoning_include_encrypted:
            reasoning_payload["include_encrypted"] = True
        self._reasoning_options = reasoning_payload or None

    def _strategy_block(self) -> str:
        strategy = self.strategy.lower()

        general_rules = [
            "Comments must start with a status token (e.g., OK; BLOCKED_AGENT(a2@11,1)) and remain within 25 words.",
            "If last_move_outcome != OK, do not repeat the same direction; prefer STAY or a safe alternate and coordinate.",
            "Treat CONTENDED or NO_GO neighbors as high risk: only enter if no safer option, and communicate or yield when you do.",
        ]

        if strategy == "none":
            strategy_rules = ["Communication disabled; do not choose COMMUNICATE."]
        elif strategy == "intent":
            strategy_rules = [
                "If a neighbor within 2 cells could collide with your intended move (same target cell or swap) next turn, COMMUNICATE exactly one INTENT now.",
                "If you receive an INTENT for the same target or swap and your agent_id is lexicographically larger, yield on the next turn (STAY or a safe alternate); otherwise MOVE.",
                "When no conflict is likely, MOVE and skip communication.",
            ]
        elif strategy == "negotiation":
            strategy_rules = [
                "Allowed messages: HERE, INTENT, SENSE, REQUEST(YIELD|GUIDE|MEET); send at most one message when you COMMUNICATE.",
                "Use the same conflict trigger as INTENT; prefer INTENT unless another message resolves the risk more clearly.",
                "When no conflict is likely, MOVE and skip communication.",
            ]
        elif strategy == "freeform":
            strategy_rules = [
                "If a neighbor within 2 cells could collide with your intended move next turn, COMMUNICATE one <=96-char sentence with your plan and a simple request.",
                "When no conflict is likely, MOVE and do not communicate.",
            ]
        elif strategy == "oracle":
            strategy_rules = [
                "Peer radio is disabled; do not choose COMMUNICATE.",
                "You may choose ASK_ORACLE when uncertain or when history.loop indicates you are stuck.",
                "After receiving an oracle suggestion, either follow it or briefly justify any override in your comment.",
            ]
        else:
            strategy_rules = ["Communication rules unspecified; default to MOVE and avoid COMMUNICATE."]

        lines = general_rules + strategy_rules
        rules = "\n".join(f"- {line}" for line in lines)
        return f"COMMUNICATION_RULES:\n{rules}\n\n"

    def _loop_block(self) -> str:
        lines = [
            f"Loop monitor: observation history only includes the last {self.history_limit} turns.",
        ]
        if self.loop_guidance.lower() == "active":
            lines.append(
                "If history.loop >= 3 or you have toggled between the same cells repeatedly, change axis or choose a different safe action (STAY, mark, or explore a new direction) before repeating the same move."
            )
            lines.append(
                "Optionally communicate your intent when breaking a loop so nearby agents can coordinate."
            )
        elif self.loop_guidance.lower() == "explore":
            lines.extend(
                [
                    "If history.loop >= 2 or you notice the same two cells in `history`, you MUST break the pattern: pick a perpendicular or backward move even if it points away from the goal.",
                    "Going away from the goal is acceptable when escaping traps—prioritise clearing the congestion first, then re-approach.",
                    "Consider dropping a MARK/NO_GO artifact or broadcasting a message that you are rerouting, so teammates yield or take an alternate path.",
                    "Never repeat the same move twice in a row while loop >= 2; choose a different axis or STAY + communicate.",
                ]
            )
        return "LOOP_RULES:\n" + "\n".join(f"- {line}" for line in lines) + "\n\n"

    def _prompt_for(self, observation: Observation) -> str:
        payload = observation.model_dump(mode="json")
        header = STATIC_HEADER.replace(
            "<OBSERVATION_JSON>\n",
            f"{self._strategy_block()}{self._loop_block()}<OBSERVATION_JSON>\n",
            1,
        )
        return f"{header}{json.dumps(payload, separators=(',', ':'))}\n</OBSERVATION_JSON>"

    async def decide_async(self, observation: Observation) -> Decision:
        prompt = self._prompt_for(observation)
        decision, _, _ = await self.unified.run(
            [{"role": "user", "content": prompt}],
            model=self.model_id,
            output_schema=Decision,
            reasoning_options=self._reasoning_options,
        )
        return decision

    async def decide_with_trace_async(self, observation: Observation) -> DecisionTrace:
        prompt = self._prompt_for(observation)
        decision, _, _ = await self.unified.run(
            [{"role": "user", "content": prompt}],
            model=self.model_id,
            output_schema=Decision,
            reasoning_options=self._reasoning_options,
        )
        return DecisionTrace(decision=decision, prompt=prompt, trace_messages=[])

    def decide(self, observation: Observation) -> Decision:  # pragma: no cover - guard rail
        raise RuntimeError(
            "LlmPolicy.decide() is disabled; use decide_async() within the episode event loop."
        )

    def decide_with_trace(self, observation: Observation) -> DecisionTrace:  # pragma: no cover - guard rail
        raise RuntimeError(
            "LlmPolicy.decide_with_trace() is disabled; use decide_with_trace_async()."
        )

````

## File: src/llmgrid/prompts.py

````python
"""Prompt templates that follow the cache-friendly header rules."""

STATIC_HEADER = """OBJECTIVE:
Keep the entire team safe and moving. “Progress” means avoiding stalemates, exploring new corridors, and guiding everyone to the goal—sometimes by stepping away from it temporarily. The episode ends after 60 turns; a timeout is a failure even if some agents reach the goal.

MISSION BRIEF:
- Grid awareness comes from the JSON: `grid_size`, `local_patch`, and `adjacent` describe nearby tiles; `self_state` gives your orientation.
- Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates.
 - Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates, and ASK_ORACLE (request global guidance when available).
- Collisions (BLOCK_AGENT or SWAP_CONFLICT) reset you, waste a turn, and often leave NO_GO markers. `contended_neighbors` tells you which adjacent directions collided last turn—treat them as hotspots and coordinate before retrying.
- History: `history` holds your last turns with a `loop` counter and notes; `recent_positions` lists the cells you just visited.
- Goal sensor (`goal_sensor`) is a noisy hint. Treat it as guidance, not a command.
- Teammates do not see your thoughts—announce reroutes, hazards, or intents when relevant.
- Messages you send this turn arrive in teammate inboxes at the start of the next turn; broadcast before stepping back into a contested direction.

TOOL ARSENAL (with quick cues):
- MOVE_N/E/S/W — default travel. Example: `adjacent.E = FREE`, loop=0 → MOVE_E, comment `OK; advancing east toward open corridor`.
- STAY — hold position when moving would collide or you need to communicate/mark first. Example: all sides blocked, teammate approaching → STAY + COMMUNICATE “yielding”.
- COMMUNICATE — one structured message to share intent, hazards, or reroutes (range {radio_range}). Use especially when loop ≥ 2 or entering a contested cell.
- MARK / NO_GO — drop on a hotspot after repeated conflicts; teammates should treat it as high risk for a few turns.
- HISTORY / LOOP COUNTER — diagnostic tool: if `history.loop` climbs or `recent_positions` oscillate, immediately select a different axis, even if it increases distance.
- ASK_ORACLE — spend the turn requesting the Oracle’s recommendation. The reply arrives in your inbox at the start of the next turn; follow it or explain why you override it.

DECISION HIERARCHY (apply in order every turn):
1. ESCAPE LOOPS: If `history.loop ≥ 2` or you see back-and-forth patterns in `history` / `recent_positions`, you MUST break the cycle. Choose a perpendicular or backward move, STAY + communicate a reroute, or drop a MARK/NO_GO—even if that increases your goal distance.
2. PREVENT COLLISIONS: Respect WALL / NO_GO / contended cells. Yield or coordinate before entering tight corridors.
3. EXPLORE: Prefer safe tiles you haven’t occupied recently to open new paths and relieve congestion.
4. ADVANCE TOWARD GOAL: Only after you are loop-free and clear of hazards should you follow the goal bearing or Manhattan gradient.

LOOP ESCAPE EXAMPLE:
- Turn t: `history.loop = 3`, last intents [MOVE_E, MOVE_W, MOVE_E]. Action: MOVE_N; comment “AVOID_LOOP; exploring north to clear congestion.” Optionally COMMUNICATE “rerouting north to break loop.”
- Turn t+1: loop resets to 0 → reassess hazards, then resume goal-oriented planning.

COMMENT & COMMUNICATION GUIDELINES:
- Begin comments with a status token (e.g., “OK;”, “BLOCKED_AGENT(…)”) and keep them ≤25 words.
- When you take a detour or STAY to break a loop, explain it so teammates know you’re clearing space. Use COMMUNICATE to broadcast reroutes, hazards, or intent when appropriate.
- If `contended_neighbors` flags a direction, STAY or communicate first—the warning arrived from last turn’s collision.
- Reference any Oracle guidance you follow (e.g., “ORACLE; following move east suggestion”).

OUTPUT CONTRACT:
Return a single structured object that conforms to the `Decision` model. Do not output any other text.

EXECUTION RULES:
1. Read <OBSERVATION_JSON>.
2. Respect walls, bounds, NO_GO markers, and agent collisions visible in the patch.
3. Consult `adjacent` (NESW labels) plus `recent_positions` to avoid immediate backtracking unless it is the only safe option.
4. Use `history`, `goal_sensor`, `neighbors_in_view`, `artifacts_in_view`, and `inbox` to inform your choice.
5. If you COMMUNICATE, send the minimal helpful message.
6. Populate `comment` with one concise paragraph (1–3 sentences) explaining your reasoning for this turn.
7. If you choose ASK_ORACLE, remain in place this turn; the Oracle reply will arrive in your inbox before the next decision.

QUALITY GATE:
Return exactly one `Decision` object that validates.

FINAL INSTRUCTIONS:
Emit only the structured `Decision` object.

=== INPUTS ===
<OBSERVATION_JSON>
"""

````

## File: experiments/presets/batch/long_corridor_seed606.txt

````text
##############################
#...........#.........#.......
###...#####.#.###.###...#####.
#...#...#.......#...#.........
#.##....#.#.#.#.#.#.#.#.###.#.
#.#...#...#...#.....#.#.#...#.
#..#####..#.#########.###.###.
#.........#...#...#.......#...
###.####....#.#.#...##..##..#.
#...........#...#.......#.....

````

## File: experiments/presets/batch/maze_seed777_width24_height14.txt

````text
########################
#...#.....#.............
###.#.###.##.##.#####.##
#.#.#.#.......#.....#...
#.#.#.#######.###.#.###.
#.#.#...#...#...#.#...#.
#.#.#.#.#.#.###.###.#.#.
#...#.#...#.#...#...#...
#.###.###.#.#.###.#.###.
#.#...#...#...#.....#...
#.#####.#.#.#######.#.##
#.....#.#...#.......#...
#####.#.#.###.#########.
#.......#.....#.........
````

## Run: gpt5_none_seed13_20251107T004758Z

### metrics.json

````json
{
  "model": "azure:gpt-5-mini",
  "comm_strategy": "none",
  "seed": 13,
  "agents": 5,
  "maze_preset": "long_corridor",
  "success": true,
  "turns": 54,
  "collisions": 4,
  "messages_sent": 0,
  "collision_causes": {
    "BLOCK_AGENT": 4
  },
  "hazard_events": 2,
  "comments_clamped": 0,
  "comments_autofilled": 0,
  "no_go_exposures": 5,
  "contended_exposures": 0,
  "history_limit": 5,
  "loop_guidance": "explore",
  "oracle_requests": 0,
  "oracle_enabled": false
}
````

### transcript.jsonl (excerpt)

````text
{"turn": 0, "agent_id": "a1", "prompt": "OBJECTIVE:\nKeep the entire team safe and moving. \u201cProgress\u201d means avoiding stalemates, exploring new corridors, and guiding everyone to the goal\u2014sometimes by stepping away from it temporarily. The episode ends after 60 turns; a timeout is a failure even if some agents reach the goal.\n\nMISSION BRIEF:\n- Grid awareness comes from the JSON: `grid_size`, `local_patch`, and `adjacent` describe nearby tiles; `self_state` gives your orientation.\n- Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates.\n - Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates, and ASK_ORACLE (request global guidance when available).\n- Collisions (BLOCK_AGENT or SWAP_CONFLICT) reset you, waste a turn, and often leave NO_GO markers. `contended_neighbors` tells you which adjacent directions collided last turn\u2014treat them as hotspots and coordinate before retrying.\n- History: `history` holds your last turns with a `loop` counter and notes; `recent_positions` lists the cells you just visited.\n- Goal sensor (`goal_sensor`) is a noisy hint. Treat it as guidance, not a command.\n- Teammates do not see your thoughts\u2014announce reroutes, hazards, or intents when relevant.\n- Messages you send this turn arrive in teammate inboxes at the start of the next turn; broadcast before stepping back into a contested direction.\n\nTOOL ARSENAL (with quick cues):\n- MOVE_N/E/S/W \u2014 default travel. Example: `adjacent.E = FREE`, loop=0 \u2192 MOVE_E, comment `OK; advancing east toward open corridor`.\n- STAY \u2014 hold position when moving would collide or you need to communicate/mark first. Example: all sides blocked, teammate approaching \u2192 STAY + COMMUNICATE \u201cyielding\u201d.\n- COMMUNICATE \u2014 one structured message to share intent, hazards, or reroutes (range {radio_range}). Use especially when loop \u2265 2 or entering a contested cell.\n- MARK / NO_GO \u2014 drop on a hotspot after repeated conflicts; teammates should treat it as high risk for a few turns.\n- HISTORY / LOOP COUNTER \u2014 diagnostic tool: if `history.loop` climbs or `recent_positions` oscillate, immediately select a different axis, even if it increases distance.\n- ASK_ORACLE \u2014 spend the turn requesting the Oracle\u2019s recommendation. The reply arrives in your inbox at the start of the next turn; follow it or explain why you override it.\n\nDECISION HIERARCHY (apply in order every turn):\n1. ESCAPE LOOPS: If `history.loop \u2265 2` or you see back-and-forth patterns in `history` / `recent_positions`, you MUST break the cycle. Choose a perpendicular or backward move, STAY + communicate a reroute, or drop a MARK/NO_GO\u2014even if that increases your goal distance.\n2. PREVENT COLLISIONS: Respect WALL / NO_GO / contended cells. Yield or coordinate before entering tight corridors.\n3. EXPLORE: Prefer safe tiles you haven\u2019t occupied recently to open new paths and relieve congestion.\n4. ADVANCE TOWARD GOAL: Only after you are loop-free and clear of hazards should you follow the goal bearing or Manhattan gradient.\n\nLOOP ESCAPE EXAMPLE:\n- Turn t: `history.loop = 3`, last intents [MOVE_E, MOVE_W, MOVE_E]. Action: MOVE_N; comment \u201cAVOID_LOOP; exploring north to clear congestion.\u201d Optionally COMMUNICATE \u201crerouting north to break loop.\u201d\n- Turn t+1: loop resets to 0 \u2192 reassess hazards, then resume goal-oriented planning.\n\nCOMMENT & COMMUNICATION GUIDELINES:\n- Begin comments with a status token (e.g., \u201cOK;\u201d, \u201cBLOCKED_AGENT(\u2026)\u201d) and keep them \u226425 words.\n- When you take a detour or STAY to break a loop, explain it so teammates know you\u2019re clearing space. Use COMMUNICATE to broadcast reroutes, hazards, or intent when appropriate.\n- If `contended_neighbors` flags a direction, STAY or communicate first\u2014the warning arrived from last turn\u2019s collision.\n- Reference any Oracle guidance you follow (e.g., \u201cORACLE; following move east suggestion\u201d).\n\nOUTPUT CONTRACT:\nReturn a single structured object that conforms to the `Decision` model. Do not output any other text.\n\nEXECUTION RULES:\n1. Read <OBSERVATION_JSON>.\n2. Respect walls, bounds, NO_GO markers, and agent collisions visible in the patch.\n3. Consult `adjacent` (NESW labels) plus `recent_positions` to avoid immediate backtracking unless it is the only safe option.\n4. Use `history`, `goal_sensor`, `neighbors_in_view`, `artifacts_in_view`, and `inbox` to inform your choice.\n5. If you COMMUNICATE, send the minimal helpful message.\n6. Populate `comment` with one concise paragraph (1\u20133 sentences) explaining your reasoning for this turn.\n7. If you choose ASK_ORACLE, remain in place this turn; the Oracle reply will arrive in your inbox before the next decision.\n\nQUALITY GATE:\nReturn exactly one `Decision` object that validates.\n\nFINAL INSTRUCTIONS:\nEmit only the structured `Decision` object.\n\n=== INPUTS ===\nCOMMUNICATION_RULES:\n- Comments must start with a status token (e.g., OK; BLOCKED_AGENT(a2@11,1)) and remain within 25 words.\n- If last_move_outcome != OK, do not repeat the same direction; prefer STAY or a safe alternate and coordinate.\n- Treat CONTENDED or NO_GO neighbors as high risk: only enter if no safer option, and communicate or yield when you do.\n- Communication disabled; do not choose COMMUNICATE.\n\nLOOP_RULES:\n- Loop monitor: observation history only includes the last 5 turns.\n- If history.loop >= 2 or you notice the same two cells in `history`, you MUST break the pattern: pick a perpendicular or backward move even if it points away from the goal.\n- Going away from the goal is acceptable when escaping traps\u2014prioritise clearing the congestion first, then re-approach.\n- Consider dropping a MARK/NO_GO artifact or broadcasting a message that you are rerouting, so teammates yield or take an alternate path.\n- Never repeat the same move twice in a row while loop >= 2; choose a different axis or STAY + communicate.\n\n<OBSERVATION_JSON>\n{\"protocol_version\":\"1.0.0\",\"turn_index\":0,\"max_turns\":100,\"grid_size\":{\"width\":30,\"height\":10},\"self_state\":{\"agent_id\":\"a1\",\"abs_pos\":{\"x\":4,\"y\":0},\"orientation\":\"N\"},\"local_patch\":{\"radius\":1,\"top_left_abs\":{\"x\":3,\"y\":0},\"rows\":[\"###\",\".A#\",\"...\"]},\"neighbors_in_view\":[],\"artifacts_in_view\":[],\"inbox\":[],\"adjacent\":[{\"dir\":\"N\",\"state\":\"OUT_OF_BOUNDS\"},{\"dir\":\"E\",\"state\":\"WALL\"},{\"dir\":\"S\",\"state\":\"FREE\"},{\"dir\":\"W\",\"state\":\"FREE\"}],\"recent_positions\":[{\"x\":4,\"y\":0}],\"comm_limits\":{\"range\":2,\"max_outbound_per_turn\":1,\"max_payload_chars\":96},\"mark_limits\":{\"max_ttl\":12,\"allow_mark_info_broadcast\":true},\"goal_sensor\":{\"mode\":\"BEARING\",\"bearing\":\"E\",\"strength\":\"FAR\",\"available\":true},\"last_move_outcome\":\"OK\",\"contended_neighbors\":0,\"history\":[]}\n</OBSERVATION_JSON>", "observation": {"protocol_version": "1.0.0", "turn_index": 0, "max_turns": 100, "grid_size": {"width": 30, "height": 10}, "self_state": {"agent_id": "a1", "abs_pos": {"x": 4, "y": 0}, "orientation": "N"}, "local_patch": {"radius": 1, "top_left_abs": {"x": 3, "y": 0}, "rows": ["###", ".A#", "..."]}, "neighbors_in_view": [], "artifacts_in_view": [], "inbox": [], "adjacent": [{"dir": "N", "state": "OUT_OF_BOUNDS"}, {"dir": "E", "state": "WALL"}, {"dir": "S", "state": "FREE"}, {"dir": "W", "state": "FREE"}], "recent_positions": [{"x": 4, "y": 0}], "comm_limits": {"range": 2, "max_outbound_per_turn": 1, "max_payload_chars": 96}, "mark_limits": {"max_ttl": 12, "allow_mark_info_broadcast": true}, "goal_sensor": {"mode": "BEARING", "bearing": "E", "strength": "FAR", "available": true}, "last_move_outcome": "OK", "contended_neighbors": 0, "history": []}, "decision": {"action": {"kind": "MOVE", "direction": "S"}, "comment": "OK; advancing south to explore since east is blocked."}, "trace_messages": []}
{"turn": 0, "agent_id": "a2", "prompt": "OBJECTIVE:\nKeep the entire team safe and moving. \u201cProgress\u201d means avoiding stalemates, exploring new corridors, and guiding everyone to the goal\u2014sometimes by stepping away from it temporarily. The episode ends after 60 turns; a timeout is a failure even if some agents reach the goal.\n\nMISSION BRIEF:\n- Grid awareness comes from the JSON: `grid_size`, `local_patch`, and `adjacent` describe nearby tiles; `self_state` gives your orientation.\n- Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates.\n - Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates, and ASK_ORACLE (request global guidance when available).\n- Collisions (BLOCK_AGENT or SWAP_CONFLICT) reset you, waste a turn, and often leave NO_GO markers. `contended_neighbors` tells you which adjacent directions collided last turn\u2014treat them as hotspots and coordinate before retrying.\n- History: `history` holds your last turns with a `loop` counter and notes; `recent_positions` lists the cells you just visited.\n- Goal sensor (`goal_sensor`) is a noisy hint. Treat it as guidance, not a command.\n- Teammates do not see your thoughts\u2014announce reroutes, hazards, or intents when relevant.\n- Messages you send this turn arrive in teammate inboxes at the start of the next turn; broadcast before stepping back into a contested direction.\n\nTOOL ARSENAL (with quick cues):\n- MOVE_N/E/S/W \u2014 default travel. Example: `adjacent.E = FREE`, loop=0 \u2192 MOVE_E, comment `OK; advancing east toward open corridor`.\n- STAY \u2014 hold position when moving would collide or you need to communicate/mark first. Example: all sides blocked, teammate approaching \u2192 STAY + COMMUNICATE \u201cyielding\u201d.\n- COMMUNICATE \u2014 one structured message to share intent, hazards, or reroutes (range {radio_range}). Use especially when loop \u2265 2 or entering a contested cell.\n- MARK / NO_GO \u2014 drop on a hotspot after repeated conflicts; teammates should treat it as high risk for a few turns.\n- HISTORY / LOOP COUNTER \u2014 diagnostic tool: if `history.loop` climbs or `recent_positions` oscillate, immediately select a different axis, even if it increases distance.\n- ASK_ORACLE \u2014 spend the turn requesting the Oracle\u2019s recommendation. The reply arrives in your inbox at the start of the next turn; follow it or explain why you override it.\n\nDECISION HIERARCHY (apply in order every turn):\n1. ESCAPE LOOPS: If `history.loop \u2265 2` or you see back-and-forth patterns in `history` / `recent_positions`, you MUST break the cycle. Choose a perpendicular or backward move, STAY + communicate a reroute, or drop a MARK/NO_GO\u2014even if that increases your goal distance.\n2. PREVENT COLLISIONS: Respect WALL / NO_GO / contended cells. Yield or coordinate before entering tight corridors.\n3. EXPLORE: Prefer safe tiles you haven\u2019t occupied recently to open new paths and relieve congestion.\n4. ADVANCE TOWARD GOAL: Only after you are loop-free and clear of hazards should you follow the goal bearing or Manhattan gradient.\n\nLOOP ESCAPE EXAMPLE:\n- Turn t: `history.loop = 3`, last intents [MOVE_E, MOVE_W, MOVE_E]. Action: MOVE_N; comment \u201cAVOID_LOOP; exploring north to clear congestion.\u201d Optionally COMMUNICATE \u201crerouting north to break loop.\u201d\n- Turn t+1: loop resets to 0 \u2192 reassess hazards, then resume goal-oriented planning.\n\nCOMMENT & COMMUNICATION GUIDELINES:\n- Begin comments with a status token (e.g., \u201cOK;\u201d, \u201cBLOCKED_AGENT(\u2026)\u201d) and keep them \u226425 words.\n- When you take a detour or STAY to break a loop, explain it so teammates know you\u2019re clearing space. Use COMMUNICATE to broadcast reroutes, hazards, or intent when appropriate.\n- If `contended_neighbors` flags a direction, STAY or communicate first\u2014the warning arrived from last turn\u2019s collision.\n- Reference any Oracle guidance you follow (e.g., \u201cORACLE; following move east suggestion\u201d).\n\nOUTPUT CONTRACT:\nReturn a single structured object that conforms to the `Decision` model. Do not output any other text.\n\nEXECUTION RULES:\n1. Read <OBSERVATION_JSON>.\n2. Respect walls, bounds, NO_GO markers, and agent collisions visible in the patch.\n3. Consult `adjacent` (NESW labels) plus `recent_positions` to avoid immediate backtracking unless it is the only safe option.\n4. Use `history`, `goal_sensor`, `neighbors_in_view`, `artifacts_in_view`, and `inbox` to inform your choice.\n5. If you COMMUNICATE, send the minimal helpful message.\n6. Populate `comment` with one concise paragraph (1\u20133 sentences) explaining your reasoning for this turn.\n7. If you choose ASK_ORACLE, remain in place this turn; the Oracle reply will arrive in your inbox before the next decision.\n\nQUALITY GATE:\nReturn exactly one `Decision` object that validates.\n\nFINAL INSTRUCTIONS:\nEmit only the structured `Decision` object.\n\n=== INPUTS ===\nCOMMUNICATION_RULES:\n- Comments must start with a status token (e.g., OK; BLOCKED_AGENT(a2@11,1)) and remain within 25 words.\n- If last_move_outcome != OK, do not repeat the same direction; prefer STAY or a safe alternate and coordinate.\n- Treat CONTENDED or NO_GO neighbors as high risk: only enter if no safer option, and communicate or yield when you do.\n- Communication disabled; do not choose COMMUNICATE.\n\nLOOP_RULES:\n- Loop monitor: observation history only includes the last 5 turns.\n- If history.loop >= 2 or you notice the same two cells in `history`, you MUST break the pattern: pick a perpendicular or backward move even if it points away from the goal.\n- Going away from the goal is acceptable when escaping traps\u2014prioritise clearing the congestion first, then re-approach.\n- Consider dropping a MARK/NO_GO artifact or broadcasting a message that you are rerouting, so teammates yield or take an alternate path.\n- Never repeat the same move twice in a row while loop >= 2; choose a different axis or STAY + communicate.\n\n<OBSERVATION_JSON>\n{\"protocol_version\":\"1.0.0\",\"turn_index\":0,\"max_turns\":100,\"grid_size\":{\"width\":30,\"height\":10},\"self_state\":{\"agent_id\":\"a2\",\"abs_pos\":{\"x\":1,\"y\":7},\"orientation\":\"S\"},\"local_patch\":{\"radius\":1,\"top_left_abs\":{\"x\":0,\"y\":6},\"rows\":[\"#..\",\"#A.\",\"###\"]},\"neighbors_in_view\":[],\"artifacts_in_view\":[],\"inbox\":[],\"adjacent\":[{\"dir\":\"N\",\"state\":\"FREE\"},{\"dir\":\"E\",\"state\":\"FREE\"},{\"dir\":\"S\",\"state\":\"WALL\"},{\"dir\":\"W\",\"state\":\"WALL\"}],\"recent_positions\":[{\"x\":1,\"y\":7}],\"comm_limits\":{\"range\":2,\"max_outbound_per_turn\":1,\"max_payload_chars\":96},\"mark_limits\":{\"max_ttl\":12,\"allow_mark_info_broadcast\":true},\"goal_sensor\":{\"mode\":\"BEARING\",\"bearing\":\"E\",\"strength\":\"FAR\",\"available\":true},\"last_move_outcome\":\"OK\",\"contended_neighbors\":0,\"history\":[]}\n</OBSERVATION_JSON>", "observation": {"protocol_version": "1.0.0", "turn_index": 0, "max_turns": 100, "grid_size": {"width": 30, "height": 10}, "self_state": {"agent_id": "a2", "abs_pos": {"x": 1, "y": 7}, "orientation": "S"}, "local_patch": {"radius": 1, "top_left_abs": {"x": 0, "y": 6}, "rows": ["#..", "#A.", "###"]}, "neighbors_in_view": [], "artifacts_in_view": [], "inbox": [], "adjacent": [{"dir": "N", "state": "FREE"}, {"dir": "E", "state": "FREE"}, {"dir": "S", "state": "WALL"}, {"dir": "W", "state": "WALL"}], "recent_positions": [{"x": 1, "y": 7}], "comm_limits": {"range": 2, "max_outbound_per_turn": 1, "max_payload_chars": 96}, "mark_limits": {"max_ttl": 12, "allow_mark_info_broadcast": true}, "goal_sensor": {"mode": "BEARING", "bearing": "E", "strength": "FAR", "available": true}, "last_move_outcome": "OK", "contended_neighbors": 0, "history": []}, "decision": {"action": {"kind": "MOVE", "direction": "E"}, "comment": "OK; advancing east toward goal bearing (FAR)."}, "trace_messages": []}
{"turn": 0, "agent_id": "a3", "prompt": "OBJECTIVE:\nKeep the entire team safe and moving. \u201cProgress\u201d means avoiding stalemates, exploring new corridors, and guiding everyone to the goal\u2014sometimes by stepping away from it temporarily. The episode ends after 60 turns; a timeout is a failure even if some agents reach the goal.\n\nMISSION BRIEF:\n- Grid awareness comes from the JSON: `grid_size`, `local_patch`, and `adjacent` describe nearby tiles; `self_state` gives your orientation.\n- Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates.\n - Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates, and ASK_ORACLE (request global guidance when available).\n- Collisions (BLOCK_AGENT or SWAP_CONFLICT) reset you, waste a turn, and often leave NO_GO markers. `contended_neighbors` tells you which adjacent directions collided last turn\u2014treat them as hotspots and coordinate before retrying.\n- History: `history` holds your last turns with a `loop` counter and notes; `recent_positions` lists the cells you just visited.\n- Goal sensor (`goal_sensor`) is a noisy hint. Treat it as guidance, not a command.\n- Teammates do not see your thoughts\u2014announce reroutes, hazards, or intents when relevant.\n- Messages you send this turn arrive in teammate inboxes at the start of the next turn; broadcast before stepping back into a contested direction.\n\nTOOL ARSENAL (with quick cues):\n- MOVE_N/E/S/W \u2014 default travel. Example: `adjacent.E = FREE`, loop=0 \u2192 MOVE_E, comment `OK; advancing east toward open corridor`.\n- STAY \u2014 hold position when moving would collide or you need to communicate/mark first. Example: all sides blocked, teammate approaching \u2192 STAY + COMMUNICATE \u201cyielding\u201d.\n- COMMUNICATE \u2014 one structured message to share intent, hazards, or reroutes (range {radio_range}). Use especially when loop \u2265 2 or entering a contested cell.\n- MARK / NO_GO \u2014 drop on a hotspot after repeated conflicts; teammates should treat it as high risk for a few turns.\n- HISTORY / LOOP COUNTER \u2014 diagnostic tool: if `history.loop` climbs or `recent_positions` oscillate, immediately select a different axis, even if it increases distance.\n- ASK_ORACLE \u2014 spend the turn requesting the Oracle\u2019s recommendation. The reply arrives in your inbox at the start of the next turn; follow it or explain why you override it.\n\nDECISION HIERARCHY (apply in order every turn):\n1. ESCAPE LOOPS: If `history.loop \u2265 2` or you see back-and-forth patterns in `history` / `recent_positions`, you MUST break the cycle. Choose a perpendicular or backward move, STAY + communicate a reroute, or drop a MARK/NO_GO\u2014even if that increases your goal distance.\n2. PREVENT COLLISIONS: Respect WALL / NO_GO / contended cells. Yield or coordinate before entering tight corridors.\n3. EXPLORE: Prefer safe tiles you haven\u2019t occupied recently to open new paths and relieve congestion.\n4. ADVANCE TOWARD GOAL: Only after you are loop-free and clear of hazards should you follow the goal bearing or Manhattan gradient.\n\nLOOP ESCAPE EXAMPLE:\n- Turn t: `history.loop = 3`, last intents [MOVE_E, MOVE_W, MOVE_E]. Action: MOVE_N; comment \u201cAVOID_LOOP; exploring north to clear congestion.\u201d Optionally COMMUNICATE \u201crerouting north to break loop.\u201d\n- Turn t+1: loop resets to 0 \u2192 reassess hazards, then resume goal-oriented planning.\n\nCOMMENT & COMMUNICATION GUIDELINES:\n- Begin comments with a status token (e.g., \u201cOK;\u201d, \u201cBLOCKED_AGENT(\u2026)\u201d) and keep them \u226425 words.\n- When you take a detour or STAY to break a loop, explain it so teammates know you\u2019re clearing space. Use COMMUNICATE to broadcast reroutes, hazards, or intent when appropriate.\n- If `contended_neighbors` flags a direction, STAY or communicate first\u2014the warning arrived from last turn\u2019s collision.\n- Reference any Oracle guidance you follow (e.g., \u201cORACLE; following move east suggestion\u201d).\n\nOUTPUT CONTRACT:\nReturn a single structured object that conforms to the `Decision` model. Do not output any other text.\n\nEXECUTION RULES:\n1. Read <OBSERVATION_JSON>.\n2. Respect walls, bounds, NO_GO markers, and agent collisions visible in the patch.\n3. Consult `adjacent` (NESW labels) plus `recent_positions` to avoid immediate backtracking unless it is the only safe option.\n4. Use `history`, `goal_sensor`, `neighbors_in_view`, `artifacts_in_view`, and `inbox` to inform your choice.\n5. If you COMMUNICATE, send the minimal helpful message.\n6. Populate `comment` with one concise paragraph (1\u20133 sentences) explaining your reasoning for this turn.\n7. If you choose ASK_ORACLE, remain in place this turn; the Oracle reply will arrive in your inbox before the next decision.\n\nQUALITY GATE:\nReturn exactly one `Decision` object that validates.\n\nFINAL INSTRUCTIONS:\nEmit only the structured `Decision` object.\n\n=== INPUTS ===\nCOMMUNICATION_RULES:\n- Comments must start with a status token (e.g., OK; BLOCKED_AGENT(a2@11,1)) and remain within 25 words.\n- If last_move_outcome != OK, do not repeat the same direction; prefer STAY or a safe alternate and coordinate.\n- Treat CONTENDED or NO_GO neighbors as high risk: only enter if no safer option, and communicate or yield when you do.\n- Communication disabled; do not choose COMMUNICATE.\n\nLOOP_RULES:\n- Loop monitor: observation history only includes the last 5 turns.\n- If history.loop >= 2 or you notice the same two cells in `history`, you MUST break the pattern: pick a perpendicular or backward move even if it points away from the goal.\n- Going away from the goal is acceptable when escaping traps\u2014prioritise clearing the congestion first, then re-approach.\n- Consider dropping a MARK/NO_GO artifact or broadcasting a message that you are rerouting, so teammates yield or take an alternate path.\n- Never repeat the same move twice in a row while loop >= 2; choose a different axis or STAY + communicate.\n\n<OBSERVATION_JSON>\n{\"protocol_version\":\"1.0.0\",\"turn_index\":0,\"max_turns\":100,\"grid_size\":{\"width\":30,\"height\":10},\"self_state\":{\"agent_id\":\"a3\",\"abs_pos\":{\"x\":11,\"y\":6},\"orientation\":\"E\"},\"local_patch\":{\"radius\":1,\"top_left_abs\":{\"x\":10,\"y\":5},\"rows\":[\"#..\",\"#A#\",\"#..\"]},\"neighbors_in_view\":[],\"artifacts_in_view\":[],\"inbox\":[],\"adjacent\":[{\"dir\":\"N\",\"state\":\"FREE\"},{\"dir\":\"E\",\"state\":\"WALL\"},{\"dir\":\"S\",\"state\":\"FREE\"},{\"dir\":\"W\",\"state\":\"WALL\"}],\"recent_positions\":[{\"x\":11,\"y\":6}],\"comm_limits\":{\"range\":2,\"max_outbound_per_turn\":1,\"max_payload_chars\":96},\"mark_limits\":{\"max_ttl\":12,\"allow_mark_info_broadcast\":true},\"goal_sensor\":{\"mode\":\"BEARING\",\"bearing\":\"E\",\"strength\":\"FAR\",\"available\":true},\"last_move_outcome\":\"OK\",\"contended_neighbors\":0,\"history\":[]}\n</OBSERVATION_JSON>", "observation": {"protocol_version": "1.0.0", "turn_index": 0, "max_turns": 100, "grid_size": {"width": 30, "height": 10}, "self_state": {"agent_id": "a3", "abs_pos": {"x": 11, "y": 6}, "orientation": "E"}, "local_patch": {"radius": 1, "top_left_abs": {"x": 10, "y": 5}, "rows": ["#..", "#A#", "#.."]}, "neighbors_in_view": [], "artifacts_in_view": [], "inbox": [], "adjacent": [{"dir": "N", "state": "FREE"}, {"dir": "E", "state": "WALL"}, {"dir": "S", "state": "FREE"}, {"dir": "W", "state": "WALL"}], "recent_positions": [{"x": 11, "y": 6}], "comm_limits": {"range": 2, "max_outbound_per_turn": 1, "max_payload_chars": 96}, "mark_limits": {"max_ttl": 12, "allow_mark_info_broadcast": true}, "goal_sensor": {"mode": "BEARING", "bearing": "E", "strength": "FAR", "available": true}, "last_move_outcome": "OK", "contended_neighbors": 0, "history": []}, "decision": {"action": {"kind": "MOVE", "direction": "N"}, "comment": "OK; advancing north to explore open corridor and avoid wall east."}, "trace_messages": []}
{"turn": 0, "agent_id": "a4", "prompt": "OBJECTIVE:\nKeep the entire team safe and moving. \u201cProgress\u201d means avoiding stalemates, exploring new corridors, and guiding everyone to the goal\u2014sometimes by stepping away from it temporarily. The episode ends after 60 turns; a timeout is a failure even if some agents reach the goal.\n\nMISSION BRIEF:\n- Grid awareness comes from the JSON: `grid_size`, `local_patch`, and `adjacent` describe nearby tiles; `self_state` gives your orientation.\n- Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates.\n - Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates, and ASK_ORACLE (request global guidance when available).\n- Collisions (BLOCK_AGENT or SWAP_CONFLICT) reset you, waste a turn, and often leave NO_GO markers. `contended_neighbors` tells you which adjacent directions collided last turn\u2014treat them as hotspots and coordinate before retrying.\n- History: `history` holds your last turns with a `loop` counter and notes; `recent_positions` lists the cells you just visited.\n- Goal sensor (`goal_sensor`) is a noisy hint. Treat it as guidance, not a command.\n- Teammates do not see your thoughts\u2014announce reroutes, hazards, or intents when relevant.\n- Messages you send this turn arrive in teammate inboxes at the start of the next turn; broadcast before stepping back into a contested direction.\n\nTOOL ARSENAL (with quick cues):\n- MOVE_N/E/S/W \u2014 default travel. Example: `adjacent.E = FREE`, loop=0 \u2192 MOVE_E, comment `OK; advancing east toward open corridor`.\n- STAY \u2014 hold position when moving would collide or you need to communicate/mark first. Example: all sides blocked, teammate approaching \u2192 STAY + COMMUNICATE \u201cyielding\u201d.\n- COMMUNICATE \u2014 one structured message to share intent, hazards, or reroutes (range {radio_range}). Use especially when loop \u2265 2 or entering a contested cell.\n- MARK / NO_GO \u2014 drop on a hotspot after repeated conflicts; teammates should treat it as high risk for a few turns.\n- HISTORY / LOOP COUNTER \u2014 diagnostic tool: if `history.loop` climbs or `recent_positions` oscillate, immediately select a different axis, even if it increases distance.\n- ASK_ORACLE \u2014 spend the turn requesting the Oracle\u2019s recommendation. The reply arrives in your inbox at the start of the next turn; follow it or explain why you override it.\n\nDECISION HIERARCHY (apply in order every turn):\n1. ESCAPE LOOPS: If `history.loop \u2265 2` or you see back-and-forth patterns in `history` / `recent_positions`, you MUST break the cycle. Choose a perpendicular or backward move, STAY + communicate a reroute, or drop a MARK/NO_GO\u2014even if that increases your goal distance.\n2. PREVENT COLLISIONS: Respect WALL / NO_GO / contended cells. Yield or coordinate before entering tight corridors.\n3. EXPLORE: Prefer safe tiles you haven\u2019t occupied recently to open new paths and relieve congestion.\n4. ADVANCE TOWARD GOAL: Only after you are loop-free and clear of hazards should you follow the goal bearing or Manhattan gradient.\n\nLOOP ESCAPE EXAMPLE:\n- Turn t: `history.loop = 3`, last intents [MOVE_E, MOVE_W, MOVE_E]. Action: MOVE_N; comment \u201cAVOID_LOOP; exploring north to clear congestion.\u201d Optionally COMMUNICATE \u201crerouting north to break loop.\u201d\n- Turn t+1: loop resets to 0 \u2192 reassess hazards, then resume goal-oriented planning.\n\nCOMMENT & COMMUNICATION GUIDELINES:\n- Begin comments with a status token (e.g., \u201cOK;\u201d, \u201cBLOCKED_AGENT(\u2026)\u201d) and keep them \u226425 words.\n- When you take a detour or STAY to break a loop, explain it so teammates know you\u2019re clearing space. Use COMMUNICATE to broadcast reroutes, hazards, or intent when appropriate.\n- If `contended_neighbors` flags a direction, STAY or communicate first\u2014the warning arrived from last turn\u2019s collision.\n- Reference any Oracle guidance you follow (e.g., \u201cORACLE; following move east suggestion\u201d).\n\nOUTPUT CONTRACT:\nReturn a single structured object that conforms to the `Decision` model. Do not output any other text.\n\nEXECUTION RULES:\n1. Read <OBSERVATION_JSON>.\n2. Respect walls, bounds, NO_GO markers, and agent collisions visible in the patch.\n3. Consult `adjacent` (NESW labels) plus `recent_positions` to avoid immediate backtracking unless it is the only safe option.\n4. Use `history`, `goal_sensor`, `neighbors_in_view`, `artifacts_in_view`, and `inbox` to inform your choice.\n5. If you COMMUNICATE, send the minimal helpful message.\n6. Populate `comment` with one concise paragraph (1\u20133 sentences) explaining your reasoning for this turn.\n7. If you choose ASK_ORACLE, remain in place this turn; the Oracle reply will arrive in your inbox before the next decision.\n\nQUALITY GATE:\nReturn exactly one `Decision` object that validates.\n\nFINAL INSTRUCTIONS:\nEmit only the structured `Decision` object.\n\n=== INPUTS ===\nCOMMUNICATION_RULES:\n- Comments must start with a status token (e.g., OK; BLOCKED_AGENT(a2@11,1)) and remain within 25 words.\n- If last_move_outcome != OK, do not repeat the same direction; prefer STAY or a safe alternate and coordinate.\n- Treat CONTENDED or NO_GO neighbors as high risk: only enter if no safer option, and communicate or yield when you do.\n- Communication disabled; do not choose COMMUNICATE.\n\nLOOP_RULES:\n- Loop monitor: observation history only includes the last 5 turns.\n- If history.loop >= 2 or you notice the same two cells in `history`, you MUST break the pattern: pick a perpendicular or backward move even if it points away from the goal.\n- Going away from the goal is acceptable when escaping traps\u2014prioritise clearing the congestion first, then re-approach.\n- Consider dropping a MARK/NO_GO artifact or broadcasting a message that you are rerouting, so teammates yield or take an alternate path.\n- Never repeat the same move twice in a row while loop >= 2; choose a different axis or STAY + communicate.\n\n<OBSERVATION_JSON>\n{\"protocol_version\":\"1.0.0\",\"turn_index\":0,\"max_turns\":100,\"grid_size\":{\"width\":30,\"height\":10},\"self_state\":{\"agent_id\":\"a4\",\"abs_pos\":{\"x\":5,\"y\":9},\"orientation\":\"W\"},\"local_patch\":{\"radius\":1,\"top_left_abs\":{\"x\":4,\"y\":8},\"rows\":[\"###\",\".A.\",\"###\"]},\"neighbors_in_view\":[],\"artifacts_in_view\":[],\"inbox\":[],\"adjacent\":[{\"dir\":\"N\",\"state\":\"WALL\"},{\"dir\":\"E\",\"state\":\"FREE\"},{\"dir\":\"S\",\"state\":\"OUT_OF_BOUNDS\"},{\"dir\":\"W\",\"state\":\"FREE\"}],\"recent_positions\":[{\"x\":5,\"y\":9}],\"comm_limits\":{\"range\":2,\"max_outbound_per_turn\":1,\"max_payload_chars\":96},\"mark_limits\":{\"max_ttl\":12,\"allow_mark_info_broadcast\":true},\"goal_sensor\":{\"mode\":\"BEARING\",\"bearing\":\"E\",\"strength\":\"FAR\",\"available\":true},\"last_move_outcome\":\"OK\",\"contended_neighbors\":0,\"history\":[]}\n</OBSERVATION_JSON>", "observation": {"protocol_version": "1.0.0", "turn_index": 0, "max_turns": 100, "grid_size": {"width": 30, "height": 10}, "self_state": {"agent_id": "a4", "abs_pos": {"x": 5, "y": 9}, "orientation": "W"}, "local_patch": {"radius": 1, "top_left_abs": {"x": 4, "y": 8}, "rows": ["###", ".A.", "###"]}, "neighbors_in_view": [], "artifacts_in_view": [], "inbox": [], "adjacent": [{"dir": "N", "state": "WALL"}, {"dir": "E", "state": "FREE"}, {"dir": "S", "state": "OUT_OF_BOUNDS"}, {"dir": "W", "state": "FREE"}], "recent_positions": [{"x": 5, "y": 9}], "comm_limits": {"range": 2, "max_outbound_per_turn": 1, "max_payload_chars": 96}, "mark_limits": {"max_ttl": 12, "allow_mark_info_broadcast": true}, "goal_sensor": {"mode": "BEARING", "bearing": "E", "strength": "FAR", "available": true}, "last_move_outcome": "OK", "contended_neighbors": 0, "history": []}, "decision": {"action": {"kind": "MOVE", "direction": "E"}, "comment": "OK; advancing east toward goal; east tile FREE and no contention."}, "trace_messages": []}
{"turn": 0, "agent_id": "a5", "prompt": "OBJECTIVE:\nKeep the entire team safe and moving. \u201cProgress\u201d means avoiding stalemates, exploring new corridors, and guiding everyone to the goal\u2014sometimes by stepping away from it temporarily. The episode ends after 60 turns; a timeout is a failure even if some agents reach the goal.\n\nMISSION BRIEF:\n- Grid awareness comes from the JSON: `grid_size`, `local_patch`, and `adjacent` describe nearby tiles; `self_state` gives your orientation.\n- Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates.\n - Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates, and ASK_ORACLE (request global guidance when available).\n- Collisions (BLOCK_AGENT or SWAP_CONFLICT) reset you, waste a turn, and often leave NO_GO markers. `contended_neighbors` tells you which adjacent directions collided last turn\u2014treat them as hotspots and coordinate before retrying.\n- History: `history` holds your last turns with a `loop` counter and notes; `recent_positions` lists the cells you just visited.\n- Goal sensor (`goal_sensor`) is a noisy hint. Treat it as guidance, not a command.\n- Teammates do not see your thoughts\u2014announce reroutes, hazards, or intents when relevant.\n- Messages you send this turn arrive in teammate inboxes at the start of the next turn; broadcast before stepping back into a contested direction.\n\nTOOL ARSENAL (with quick cues):\n- MOVE_N/E/S/W \u2014 default travel. Example: `adjacent.E = FREE`, loop=0 \u2192 MOVE_E, comment `OK; advancing east toward open corridor`.\n- STAY \u2014 hold position when moving would collide or you need to communicate/mark first. Example: all sides blocked, teammate approaching \u2192 STAY + COMMUNICATE \u201cyielding\u201d.\n- COMMUNICATE \u2014 one structured message to share intent, hazards, or reroutes (range {radio_range}). Use especially when loop \u2265 2 or entering a contested cell.\n- MARK / NO_GO \u2014 drop on a hotspot after repeated conflicts; teammates should treat it as high risk for a few turns.\n- HISTORY / LOOP COUNTER \u2014 diagnostic tool: if `history.loop` climbs or `recent_positions` oscillate, immediately select a different axis, even if it increases distance.\n- ASK_ORACLE \u2014 spend the turn requesting the Oracle\u2019s recommendation. The reply arrives in your inbox at the start of the next turn; follow it or explain why you override it.\n\nDECISION HIERARCHY (apply in order every turn):\n1. ESCAPE LOOPS: If `history.loop \u2265 2` or you see back-and-forth patterns in `history` / `recent_positions`, you MUST break the cycle. Choose a perpendicular or backward move, STAY + communicate a reroute, or drop a MARK/NO_GO\u2014even if that increases your goal distance.\n2. PREVENT COLLISIONS: Respect WALL / NO_GO / contended cells. Yield or coordinate before entering tight corridors.\n3. EXPLORE: Prefer safe tiles you haven\u2019t occupied recently to open new paths and relieve congestion.\n4. ADVANCE TOWARD GOAL: Only after you are loop-free and clear of hazards should you follow the goal bearing or Manhattan gradient.\n\nLOOP ESCAPE EXAMPLE:\n- Turn t: `history.loop = 3`, last intents [MOVE_E, MOVE_W, MOVE_E]. Action: MOVE_N; comment \u201cAVOID_LOOP; exploring north to clear congestion.\u201d Optionally COMMUNICATE \u201crerouting north to break loop.\u201d\n- Turn t+1: loop resets to 0 \u2192 reassess hazards, then resume goal-oriented planning.\n\nCOMMENT & COMMUNICATION GUIDELINES:\n- Begin comments with a status token (e.g., \u201cOK;\u201d, \u201cBLOCKED_AGENT(\u2026)\u201d) and keep them \u226425 words.\n- When you take a detour or STAY to break a loop, explain it so teammates know you\u2019re clearing space. Use COMMUNICATE to broadcast reroutes, hazards, or intent when appropriate.\n- If `contended_neighbors` flags a direction, STAY or communicate first\u2014the warning arrived from last turn\u2019s collision.\n- Reference any Oracle guidance you follow (e.g., \u201cORACLE; following move east suggestion\u201d).\n\nOUTPUT CONTRACT:\nReturn a single structured object that conforms to the `Decision` model. Do not output any other text.\n\nEXECUTION RULES:\n1. Read <OBSERVATION_JSON>.\n2. Respect walls, bounds, NO_GO markers, and agent collisions visible in the patch.\n3. Consult `adjacent` (NESW labels) plus `recent_positions` to avoid immediate backtracking unless it is the only safe option.\n4. Use `history`, `goal_sensor`, `neighbors_in_view`, `artifacts_in_view`, and `inbox` to inform your choice.\n5. If you COMMUNICATE, send the minimal helpful message.\n6. Populate `comment` with one concise paragraph (1\u20133 sentences) explaining your reasoning for this turn.\n7. If you choose ASK_ORACLE, remain in place this turn; the Oracle reply will arrive in your inbox before the next decision.\n\nQUALITY GATE:\nReturn exactly one `Decision` object that validates.\n\nFINAL INSTRUCTIONS:\nEmit only the structured `Decision` object.\n\n=== INPUTS ===\nCOMMUNICATION_RULES:\n- Comments must start with a status token (e.g., OK; BLOCKED_AGENT(a2@11,1)) and remain within 25 words.\n- If last_move_outcome != OK, do not repeat the same direction; prefer STAY or a safe alternate and coordinate.\n- Treat CONTENDED or NO_GO neighbors as high risk: only enter if no safer option, and communicate or yield when you do.\n- Communication disabled; do not choose COMMUNICATE.\n\nLOOP_RULES:\n- Loop monitor: observation history only includes the last 5 turns.\n- If history.loop >= 2 or you notice the same two cells in `history`, you MUST break the pattern: pick a perpendicular or backward move even if it points away from the goal.\n- Going away from the goal is acceptable when escaping traps\u2014prioritise clearing the congestion first, then re-approach.\n- Consider dropping a MARK/NO_GO artifact or broadcasting a message that you are rerouting, so teammates yield or take an alternate path.\n- Never repeat the same move twice in a row while loop >= 2; choose a different axis or STAY + communicate.\n\n<OBSERVATION_JSON>\n{\"protocol_version\":\"1.0.0\",\"turn_index\":0,\"max_turns\":100,\"grid_size\":{\"width\":30,\"height\":10},\"self_state\":{\"agent_id\":\"a5\",\"abs_pos\":{\"x\":0,\"y\":2},\"orientation\":\"N\"},\"local_patch\":{\"radius\":1,\"top_left_abs\":{\"x\":0,\"y\":1},\"rows\":[\"#..\",\"#A#\",\"##.\"]},\"neighbors_in_view\":[],\"artifacts_in_view\":[],\"inbox\":[],\"adjacent\":[{\"dir\":\"N\",\"state\":\"FREE\"},{\"dir\":\"E\",\"state\":\"WALL\"},{\"dir\":\"S\",\"state\":\"WALL\"},{\"dir\":\"W\",\"state\":\"OUT_OF_BOUNDS\"}],\"recent_positions\":[{\"x\":0,\"y\":2}],\"comm_limits\":{\"range\":2,\"max_outbound_per_turn\":1,\"max_payload_chars\":96},\"mark_limits\":{\"max_ttl\":12,\"allow_mark_info_broadcast\":true},\"goal_sensor\":{\"mode\":\"BEARING\",\"bearing\":\"NE\",\"strength\":\"FAR\",\"available\":true},\"last_move_outcome\":\"OK\",\"contended_neighbors\":0,\"history\":[]}\n</OBSERVATION_JSON>", "observation": {"protocol_version": "1.0.0", "turn_index": 0, "max_turns": 100, "grid_size": {"width": 30, "height": 10}, "self_state": {"agent_id": "a5", "abs_pos": {"x": 0, "y": 2}, "orientation": "N"}, "local_patch": {"radius": 1, "top_left_abs": {"x": 0, "y": 1}, "rows": ["#..", "#A#", "##."]}, "neighbors_in_view": [], "artifacts_in_view": [], "inbox": [], "adjacent": [{"dir": "N", "state": "FREE"}, {"dir": "E", "state": "WALL"}, {"dir": "S", "state": "WALL"}, {"dir": "W", "state": "OUT_OF_BOUNDS"}], "recent_positions": [{"x": 0, "y": 2}], "comm_limits": {"range": 2, "max_outbound_per_turn": 1, "max_payload_chars": 96}, "mark_limits": {"max_ttl": 12, "allow_mark_info_broadcast": true}, "goal_sensor": {"mode": "BEARING", "bearing": "NE", "strength": "FAR", "available": true}, "last_move_outcome": "OK", "contended_neighbors": 0, "history": []}, "decision": {"action": {"kind": "MOVE", "direction": "N"}, "comment": "OK; advancing north toward open corridor and NE goal bearing."}, "trace_messages": []}
{"turn": 1, "agent_id": "a1", "prompt": "OBJECTIVE:\nKeep the entire team safe and moving. \u201cProgress\u201d means avoiding stalemates, exploring new corridors, and guiding everyone to the goal\u2014sometimes by stepping away from it temporarily. The episode ends after 60 turns; a timeout is a failure even if some agents reach the goal.\n\nMISSION BRIEF:\n- Grid awareness comes from the JSON: `grid_size`, `local_patch`, and `adjacent` describe nearby tiles; `self_state` gives your orientation.\n- Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates.\n - Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates, and ASK_ORACLE (request global guidance when available).\n- Collisions (BLOCK_AGENT or SWAP_CONFLICT) reset you, waste a turn, and often leave NO_GO markers. `contended_neighbors` tells you which adjacent directions collided last turn\u2014treat them as hotspots and coordinate before retrying.\n- History: `history` holds your last turns with a `loop` counter and notes; `recent_positions` lists the cells you just visited.\n- Goal sensor (`goal_sensor`) is a noisy hint. Treat it as guidance, not a command.\n- Teammates do not see your thoughts\u2014announce reroutes, hazards, or intents when relevant.\n- Messages you send this turn arrive in teammate inboxes at the start of the next turn; broadcast before stepping back into a contested direction.\n\nTOOL ARSENAL (with quick cues):\n- MOVE_N/E/S/W \u2014 default travel. Example: `adjacent.E = FREE`, loop=0 \u2192 MOVE_E, comment `OK; advancing east toward open corridor`.\n- STAY \u2014 hold position when moving would collide or you need to communicate/mark first. Example: all sides blocked, teammate approaching \u2192 STAY + COMMUNICATE \u201cyielding\u201d.\n- COMMUNICATE \u2014 one structured message to share intent, hazards, or reroutes (range {radio_range}). Use especially when loop \u2265 2 or entering a contested cell.\n- MARK / NO_GO \u2014 drop on a hotspot after repeated conflicts; teammates should treat it as high risk for a few turns.\n- HISTORY / LOOP COUNTER \u2014 diagnostic tool: if `history.loop` climbs or `recent_positions` oscillate, immediately select a different axis, even if it increases distance.\n- ASK_ORACLE \u2014 spend the turn requesting the Oracle\u2019s recommendation. The reply arrives in your inbox at the start of the next turn; follow it or explain why you override it.\n\nDECISION HIERARCHY (apply in order every turn):\n1. ESCAPE LOOPS: If `history.loop \u2265 2` or you see back-and-forth patterns in `history` / `recent_positions`, you MUST break the cycle. Choose a perpendicular or backward move, STAY + communicate a reroute, or drop a MARK/NO_GO\u2014even if that increases your goal distance.\n2. PREVENT COLLISIONS: Respect WALL / NO_GO / contended cells. Yield or coordinate before entering tight corridors.\n3. EXPLORE: Prefer safe tiles you haven\u2019t occupied recently to open new paths and relieve congestion.\n4. ADVANCE TOWARD GOAL: Only after you are loop-free and clear of hazards should you follow the goal bearing or Manhattan gradient.\n\nLOOP ESCAPE EXAMPLE:\n- Turn t: `history.loop = 3`, last intents [MOVE_E, MOVE_W, MOVE_E]. Action: MOVE_N; comment \u201cAVOID_LOOP; exploring north to clear congestion.\u201d Optionally COMMUNICATE \u201crerouting north to break loop.\u201d\n- Turn t+1: loop resets to 0 \u2192 reassess hazards, then resume goal-oriented planning.\n\nCOMMENT & COMMUNICATION GUIDELINES:\n- Begin comments with a status token (e.g., \u201cOK;\u201d, \u201cBLOCKED_AGENT(\u2026)\u201d) and keep them \u226425 words.\n- When you take a detour or STAY to break a loop, explain it so teammates know you\u2019re clearing space. Use COMMUNICATE to broadcast reroutes, hazards, or intent when appropriate.\n- If `contended_neighbors` flags a direction, STAY or communicate first\u2014the warning arrived from last turn\u2019s collision.\n- Reference any Oracle guidance you follow (e.g., \u201cORACLE; following move east suggestion\u201d).\n\nOUTPUT CONTRACT:\nReturn a single structured object that conforms to the `Decision` model. Do not output any other text.\n\nEXECUTION RULES:\n1. Read <OBSERVATION_JSON>.\n2. Respect walls, bounds, NO_GO markers, and agent collisions visible in the patch.\n3. Consult `adjacent` (NESW labels) plus `recent_positions` to avoid immediate backtracking unless it is the only safe option.\n4. Use `history`, `goal_sensor`, `neighbors_in_view`, `artifacts_in_view`, and `inbox` to inform your choice.\n5. If you COMMUNICATE, send the minimal helpful message.\n6. Populate `comment` with one concise paragraph (1\u20133 sentences) explaining your reasoning for this turn.\n7. If you choose ASK_ORACLE, remain in place this turn; the Oracle reply will arrive in your inbox before the next decision.\n\nQUALITY GATE:\nReturn exactly one `Decision` object that validates.\n\nFINAL INSTRUCTIONS:\nEmit only the structured `Decision` object.\n\n=== INPUTS ===\nCOMMUNICATION_RULES:\n- Comments must start with a status token (e.g., OK; BLOCKED_AGENT(a2@11,1)) and remain within 25 words.\n- If last_move_outcome != OK, do not repeat the same direction; prefer STAY or a safe alternate and coordinate.\n- Treat CONTENDED or NO_GO neighbors as high risk: only enter if no safer option, and communicate or yield when you do.\n- Communication disabled; do not choose COMMUNICATE.\n\nLOOP_RULES:\n- Loop monitor: observation history only includes the last 5 turns.\n- If history.loop >= 2 or you notice the same two cells in `history`, you MUST break the pattern: pick a perpendicular or backward move even if it points away from the goal.\n- Going away from the goal is acceptable when escaping traps\u2014prioritise clearing the congestion first, then re-approach.\n- Consider dropping a MARK/NO_GO artifact or broadcasting a message that you are rerouting, so teammates yield or take an alternate path.\n- Never repeat the same move twice in a row while loop >= 2; choose a different axis or STAY + communicate.\n\n<OBSERVATION_JSON>\n{\"protocol_version\":\"1.0.0\",\"turn_index\":1,\"max_turns\":100,\"grid_size\":{\"width\":30,\"height\":10},\"self_state\":{\"agent_id\":\"a1\",\"abs_pos\":{\"x\":4,\"y\":1},\"orientation\":\"S\"},\"local_patch\":{\"radius\":1,\"top_left_abs\":{\"x\":3,\"y\":0},\"rows\":[\"..#\",\".A.\",\"...\"]},\"neighbors_in_view\":[],\"artifacts_in_view\":[],\"inbox\":[],\"adjacent\":[{\"dir\":\"N\",\"state\":\"FREE\"},{\"dir\":\"E\",\"state\":\"FREE\"},{\"dir\":\"S\",\"state\":\"FREE\"},{\"dir\":\"W\",\"state\":\"FREE\"}],\"recent_positions\":[{\"x\":4,\"y\":1},{\"x\":4,\"y\":0}],\"comm_limits\":{\"range\":2,\"max_outbound_per_turn\":1,\"max_payload_chars\":96},\"mark_limits\":{\"max_ttl\":12,\"allow_mark_info_broadcast\":true},\"goal_sensor\":{\"mode\":\"BEARING\",\"bearing\":\"NE\",\"strength\":\"FAR\",\"available\":true},\"last_move_outcome\":\"OK\",\"contended_neighbors\":0,\"history\":[{\"turn_index\":0,\"intent\":\"MOVE_S\",\"outcome\":\"OK\",\"delta\":\"CLOSER\",\"loop\":0,\"peer_bits\":\"N0E0S0W0|intent:-\",\"note\":null}]}\n</OBSERVATION_JSON>", "observation": {"protocol_version": "1.0.0", "turn_index": 1, "max_turns": 100, "grid_size": {"width": 30, "height": 10}, "self_state": {"agent_id": "a1", "abs_pos": {"x": 4, "y": 1}, "orientation": "S"}, "local_patch": {"radius": 1, "top_left_abs": {"x": 3, "y": 0}, "rows": ["..#", ".A.", "..."]}, "neighbors_in_view": [], "artifacts_in_view": [], "inbox": [], "adjacent": [{"dir": "N", "state": "FREE"}, {"dir": "E", "state": "FREE"}, {"dir": "S", "state": "FREE"}, {"dir": "W", "state": "FREE"}], "recent_positions": [{"x": 4, "y": 1}, {"x": 4, "y": 0}], "comm_limits": {"range": 2, "max_outbound_per_turn": 1, "max_payload_chars": 96}, "mark_limits": {"max_ttl": 12, "allow_mark_info_broadcast": true}, "goal_sensor": {"mode": "BEARING", "bearing": "NE", "strength": "FAR", "available": true}, "last_move_outcome": "OK", "contended_neighbors": 0, "history": [{"turn_index": 0, "intent": "MOVE_S", "outcome": "OK", "delta": "CLOSER", "loop": 0, "peer_bits": "N0E0S0W0|intent:-", "note": null}]}, "decision": {"action": {"kind": "MOVE", "direction": "E"}, "comment": "OK; advancing east toward open corridor and goal (NE bearing); avoiding backtrack."}, "trace_messages": []}
{"turn": 1, "agent_id": "a2", "prompt": "OBJECTIVE:\nKeep the entire team safe and moving. \u201cProgress\u201d means avoiding stalemates, exploring new corridors, and guiding everyone to the goal\u2014sometimes by stepping away from it temporarily. The episode ends after 60 turns; a timeout is a failure even if some agents reach the goal.\n\nMISSION BRIEF:\n- Grid awareness comes from the JSON: `grid_size`, `local_patch`, and `adjacent` describe nearby tiles; `self_state` gives your orientation.\n- Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates.\n - Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (one message per turn within radio range), optional artifacts (e.g., NO_GO cones) to warn teammates, and ASK_ORACLE (request global guidance when available).\n- Collisions (BLOCK_AGENT or SWAP_CONFLICT) reset you, waste a turn, and often leave NO_GO markers. `contended_neighbors` tells you which adjacent directions collided last turn\u2014treat them as hotspots and coordinate before retrying.\n- History: `history` holds your last turns with a `loop` counter and notes; `recent_positions` lists the cells you just visited.\n- Goal sensor (`goal_sensor`) is a noisy hint. Treat it as guidance, not a command.\n- Teammates do not see your thoughts\u2014announce reroutes, hazards, or intents when relevant.\n- Messages you send this turn arrive in teammate inboxes at the start of the next turn; broadcast before stepping back into a contested direction.\n\nTOOL ARSENAL (with quick cues):\n- MOVE_N/E/S/W \u2014 default travel. Example: `adjacent.E = FREE`, loop=0 \u2192 MOVE_E, comment `OK; advancing east toward open corridor`.\n- STAY \u2014 hold position when moving would collide or you need to communicate/mark first. Example: all sides blocked, teammate approaching \u2192 STAY + COMMUNICATE \u201cyielding\u201d.\n- COMMUNICATE \u2014 one structured message to share intent, hazards, or reroutes (range {radio_range}). Use especially when loop \u2265 2 or entering a contested cell.\n- MARK / NO_GO \u2014 drop on a hotspot after repeated conflicts; teammates should treat it as high risk for a few turns.\n- HISTORY / LOOP COUNTER \u2014 diagnostic tool: if `history.loop` climbs or `recent_positions` oscillate, immediately select a different axis, even if it increases distance.\n- ASK_ORACLE \u2014 spend the turn requesting the Oracle\u2019s recommendation. The reply arrives in your inbox at the start of the next turn; follow it or explain why you override it.\n\nDECISION HIERARCHY (apply in order every turn):\n1. ESCAPE LOOPS: If `history.loop \u2265 2` or you see back-and-forth patterns in `history` / `recent_positions`, you MUST break the cycle. Choose a perpendicular or backward move, STAY + communicate a reroute, or drop a MARK/NO_GO\u2014even if that increases your goal distance.\n2. PREVENT COLLISIONS: Respect WALL / NO_GO / contended cells. Yield or coordinate before entering tight corridors.\n3. EXPLORE: Prefer safe tiles you haven\u2019t occupied recently to open new paths and relieve congestion.\n4. ADVANCE TOWARD GOAL: Only after you are loop-free and clear of hazards should you follow the goal bearing or Manhattan gradient.\n\nLOOP ESCAPE EXAMPLE:\n- Turn t: `history.loop = 3`, last intents [MOVE_E, MOVE_W, MOVE_E]. Action: MOVE_N; comment \u201cAVOID_LOOP; exploring north to clear congestion.\u201d Optionally COMMUNICATE \u201crerouting north to break loop.\u201d\n- Turn t+1: loop resets to 0 \u2192 reassess hazards, then resume goal-oriented planning.\n\nCOMMENT & COMMUNICATION GUIDELINES:\n- Begin comments with a status token (e.g., \u201cOK;\u201d, \u201cBLOCKED_AGENT(\u2026)\u201d) and keep them \u226425 words.\n- When you take a detour or STAY to break a loop, explain it so teammates know you\u2019re clearing space. Use COMMUNICATE to broadcast reroutes, hazards, or intent when appropriate.\n- If `contended_neighbors` flags a direction, STAY or communicate first\u2014the warning arrived from last turn\u2019s collision.\n- Reference any Oracle guidance you follow (e.g., \u201cORACLE; following move east suggestion\u201d).\n\nOUTPUT CONTRACT:\nReturn a single structured object that conforms to the `Decision` model. Do not output any other text.\n\nEXECUTION RULES:\n1. Read <OBSERVATION_JSON>.\n2. Respect walls, bounds, NO_GO markers, and agent collisions visible in the patch.\n3. Consult `adjacent` (NESW labels) plus `recent_positions` to avoid immediate backtracking unless it is the only safe option.\n4. Use `history`, `goal_sensor`, `neighbors_in_view`, `artifacts_in_view`, and `inbox` to inform your choice.\n5. If you COMMUNICATE, send the minimal helpful message.\n6. Populate `comment` with one concise paragraph (1\u20133 sentences) explaining your reasoning for this turn.\n7. If you choose ASK_ORACLE, remain in place this turn; the Oracle reply will arrive in your inbox before the next decision.\n\nQUALITY GATE:\nReturn exactly one `Decision` object that validates.\n\nFINAL INSTRUCTIONS:\nEmit only the structured `Decision` object.\n\n=== INPUTS ===\nCOMMUNICATION_RULES:\n- Comments must start with a status token (e.g., OK; BLOCKED_AGENT(a2@11,1)) and remain within 25 words.\n- If last_move_outcome != OK, do not repeat the same direction; prefer STAY or a safe alternate and coordinate.\n- Treat CONTENDED or NO_GO neighbors as high risk: only enter if no safer option, and communicate or yield when you do.\n- Communication disabled; do not choose COMMUNICATE.\n\nLOOP_RULES:\n- Loop monitor: observation history only includes the last 5 turns.\n- If history.loop >= 2 or you notice the same two cells in `history`, you MUST break the pattern: pick a perpendicular or backward move even if it points away from the goal.\n- Going away from the goal is acceptable when escaping traps\u2014prioritise clearing the congestion first, then re-approach.\n- Consider dropping a MARK/NO_GO artifact or broadcasting a message that you are rerouting, so teammates yield or take an alternate path.\n- Never repeat the same move twice in a row while loop >= 2; choose a different axis or STAY + communicate.\n\n<OBSERVATION_JSON>\n{\"protocol_version\":\"1.0.0\",\"turn_index\":1,\"max_turns\":100,\"grid_size\":{\"width\":30,\"height\":10},\"self_state\":{\"agent_id\":\"a2\",\"abs_pos\":{\"x\":2,\"y\":7},\"orientation\":\"E\"},\"local_patch\":{\"radius\":1,\"top_left_abs\":{\"x\":1,\"y\":6},\"rows\":[\"..#\",\".A.\",\"##.\"]},\"neighbors_in_view\":[],\"artifacts_in_view\":[],\"inbox\":[],\"adjacent\":[{\"dir\":\"N\",\"state\":\"FREE\"},{\"dir\":\"E\",\"state\":\"FREE\"},{\"dir\":\"S\",\"state\":\"WALL\"},{\"dir\":\"W\",\"state\":\"FREE\"}],\"recent_positions\":[{\"x\":2,\"y\":7},{\"x\":1,\"y\":7}],\"comm_limits\":{\"range\":2,\"max_outbound_per_turn\":1,\"max_payload_chars\":96},\"mark_limits\":{\"max_ttl\":12,\"allow_mark_info_broadcast\":true},\"goal_sensor\":{\"mode\":\"BEARING\",\"bearing\":\"E\",\"strength\":\"FAR\",\"available\":true},\"last_move_outcome\":\"OK\",\"contended_neighbors\":0,\"history\":[{\"turn_index\":0,\"intent\":\"MOVE_E\",\"outcome\":\"OK\",\"delta\":\"CLOSER\",\"loop\":0,\"peer_bits\":\"N0E0S0W0|intent:-\",\"note\":null}]}\n</OBSERVATION_JSON>", "observation": {"protocol_version": "1.0.0", "turn_index": 1, "max_turns": 100, "grid_size": {"width": 30, "height": 10}, "self_state": {"agent_id": "a2", "abs_pos": {"x": 2, "y": 7}, "orientation": "E"}, "local_patch": {"radius": 1, "top_left_abs": {"x": 1, "y": 6}, "rows": ["..#", ".A.", "##."]}, "neighbors_in_view": [], "artifacts_in_view": [], "inbox": [], "adjacent": [{"dir": "N", "state": "FREE"}, {"dir": "E", "state": "FREE"}, {"dir": "S", "state": "WALL"}, {"dir": "W", "state": "FREE"}], "recent_positions": [{"x": 2, "y": 7}, {"x": 1, "y": 7}], "comm_limits": {"range": 2, "max_outbound_per_turn": 1, "max_payload_chars": 96}, "mark_limits": {"max_ttl": 12, "allow_mark_info_broadcast": true}, "goal_sensor": {"mode": "BEARING", "bearing": "E", "strength": "FAR", "available": true}, "last_move_outcome": "OK", "contended_neighbors": 0, "history": [{"turn_index": 0, "intent": "MOVE_E", "outcome": "OK", "delta": "CLOSER", "loop": 0, "peer_bits": "N0E0S0W0|intent:-", "note": null}]}, "decision": {"action": {"kind": "MOVE", "direction": "E"}, "comment": "OK; advancing east toward goal, exploring new corridor."}, "trace_messages": []}
````

## Run: gpt5_intent_seed13_20251107T004817Z

### metrics.json

````json
{
  "model": "azure:gpt-5-mini",
  "comm_strategy": "intent",
  "seed": 13,
  "agents": 5,
  "maze_preset": "long_corridor",
  "success": false,
  "turns": 100,
  "collisions": 6,
  "messages_sent": 59,
  "collision_causes": {
    "BLOCK_AGENT": 6
  },
  "hazard_events": 3,
  "comments_clamped": 0,
  "comments_autofilled": 0,
  "no_go_exposures": 9,
  "contended_exposures": 0,
  "history_limit": 5,
  "loop_guidance": "explore",
  "oracle_requests": 0,
  "oracle_enabled": false
}
````

### transcript.jsonl (excerpt)

````text
````

## Run: gpt5_negotiation_seed13_20251107T004830Z

### metrics.json

````json
{
  "model": "azure:gpt-5-mini",
  "comm_strategy": "negotiation",
  "seed": 13,
  "agents": 5,
  "maze_preset": "long_corridor",
  "success": true,
  "turns": 98,
  "collisions": 6,
  "messages_sent": 38,
  "collision_causes": {
    "BLOCK_AGENT": 6
  },
  "hazard_events": 3,
  "comments_clamped": 0,
  "comments_autofilled": 0,
  "no_go_exposures": 9,
  "contended_exposures": 0,
  "history_limit": 5,
  "loop_guidance": "explore",
  "oracle_requests": 0,
  "oracle_enabled": false
}
````

### transcript.jsonl (excerpt)

````text
````

## Run: gpt5_freeform_seed13_20251107T004846Z

### metrics.json

````json
{
  "model": "azure:gpt-5-mini",
  "comm_strategy": "freeform",
  "seed": 13,
  "agents": 5,
  "maze_preset": "long_corridor",
  "success": true,
  "turns": 73,
  "collisions": 4,
  "messages_sent": 27,
  "collision_causes": {
    "BLOCK_AGENT": 4
  },
  "hazard_events": 2,
  "comments_clamped": 0,
  "comments_autofilled": 0,
  "no_go_exposures": 6,
  "contended_exposures": 0,
  "history_limit": 5,
  "loop_guidance": "explore",
  "oracle_requests": 0,
  "oracle_enabled": false
}
````

### transcript.jsonl (excerpt)

````text
````

