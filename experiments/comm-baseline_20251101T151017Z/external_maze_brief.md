# External Maze Design Brief

## Context
- Project: multi-agent grid navigation experiments (`experiments/` tree). Baseline runs now use GPT-5 mini via the Azure Responses API with 5 agents, visibility radius 1 (5×5 egocentric patch), radio range 2, and a 100-turn cap (down from 200 after `long_corridor` became trivial).
- Current maze presets (PNG/SVG/TXT) live in `experiments/presets/batch/`. `long_corridor` (seed 606) is the default: wide horizontal corridors with mild loops near the goal.
- Latest communication baseline (seed 13, GPT-5 mini, Responses API; see `experiments/comm-baseline_20251101T151017Z/README.md`):
  - comm=none → success in 54 turns, 4 collisions, 0 messages.
  - comm=intent → timeout at 100 turns, 6 collisions, 59 messages.
  - comm=negotiation → success in 98 turns, 6 collisions, 38 messages.
  - comm=freeform → success in 73 turns, 4 collisions, 27 messages.
  - Takeaway: communication currently slows agents because the maze rarely forces coordination.
- Hypotheses to test:
  1. **Chokepoint Advantage (HYP-20251107A):** single-tile chokepoints and intersecting corridors should force alternating right-of-way so negotiation/freeform beats silence.
  2. **Partial Observability Benefit (HYP-20251107B):** hidden switches or role-dependent objectives should reward information sharing.

## Ask
Design new maze layouts that fit within roughly a 24×14 bounding box (interiors can be non-rectangular, jagged, or hollow) so the 100-turn cap still matters, but feel free to carve irregular shapes or voids inside that footprint. We need:
1. Plain-text grids using `#` (wall) and `.` (walkable). Multiple variants welcome (e.g., multi-choke arterial, switch/door puzzles, dense spirals with intersecting flows).
2. A short rationale per maze describing the forced interactions (“Agents from west/east must yield at the bridge; visibility 1 hides approach until last step”).
3. Optional notes on encoding temporary hazards (switches, doors) if your layout requires them.

## Constraints & Environment Details
- Agents: 5 (spawned far apart via `_default_start_positions`). Goal sits near (width-2, min(1,height-1)).
- Actions: MOVE_N/E/S/W, STAY, COMMUNICATE (≤1 message/turn within radio range 2), MARK/NO_GO (hazard cones with TTL=3).
- Collisions: simultaneous entry into a cell triggers BLOCK_AGENT, both agents stay in place, NO_GO marker dropped.
- Visibility: radius 1 (agents see only a 5×5 patch). To stress comms, design corners and junctions where they can’t see the opposing agent until the last moment.
- Runtime: 100-turn limit; mazes should be solvable yet tight enough that silent greedy marching fails.

## Files We’ll Share
- Full text (not just file paths) of:
  - `experiments/README.md` (project index + preset descriptions).
  - `experiments/comm-baseline_20251101T151017Z/README.md` (experiment log, metrics, hypothesis register).
  - `src/llmgrid/env/grid.py`, `src/llmgrid/env/maze_generator.py`, `src/llmgrid/cli/poc_two_agents.py` (arg parsing + run config), `src/llmgrid/agent/llm_agent.py` (observation handling), and `src/llmgrid/prompts.py` (communication schemas).
- Plain-text exports of existing presets (`experiments/presets/batch/*.txt`, including `long_corridor` and the dense seed-777 layout) so you see the exact encoding.
- Run evidence snippets: each run’s `results/metrics.json` plus the first ~20 lines of `results/transcript.jsonl` (pasted inline) to show observations/messages.

## Deliverable Format
- Return each maze as a plain-text grid (fixed width/height, `#` walls, `.` floors; label special cells if needed).
- Include a concise rationale + any required assumptions (e.g., “spawn Agent A near (2,12) to trigger chokepoint A”).
- If the layout depends on new mechanics (switch toggles, teleportation), describe the minimal engine changes required.
