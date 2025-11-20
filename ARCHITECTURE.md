# LLM Grid Agents: Comprehensive Architecture Guide

**Project:** agent-talk-2 (LLM Grid Agents)  
**Repository:** `/Users/3bn/Documents/My_Repos/agent-talk-2`  
**Current Branch:** micro-blocked-tunnel  
**Last Updated:** 2025-11-19

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Architecture](#core-architecture)
3. [Module Breakdown](#module-breakdown)
4. [Data Flow](#data-flow)
5. [Key Abstractions](#key-abstractions)
6. [Communication & Coordination](#communication--coordination)
7. [Maze/Grid Generation](#mazegrid-generation)
8. [LLM Integration Layer](#llm-integration-layer)
9. [Visualization Pipeline](#visualization-pipeline)
10. [Experiment Framework](#experiment-framework)

---

## Project Overview

**LLM Grid Agents** is a research framework for studying populations of LLM-driven agents navigating fixed grid worlds with:
- **Partial observability:** Agents see only within a visibility radius (default 2)
- **Range-limited communication:** Radio-based map sharing within radio_range
- **Map sharing mechanisms:** none | radio_sync | global
- **Synchronous turns:** All agents act simultaneously; conflicts resolved by rules
- **Structured I/O:** Pydantic-based observation/decision schemas

**Current Focus:**
- Comparing communication strategies (freeform vs. structured)
- Validating map-sharing mechanisms (global vs. radio vs. none)
- Cross-seed generalization testing (5+ runs per configuration)
- Integration with Azure OpenAI and OpenRouter providers

**Key Finding (45-run cross-seed baseline, Nov 2025):**
- Freeform natural language: 62.7% success
- No communication: 57.3% success  
- Structured INTENT/REQUEST: 56.0% success
→ Freeform generalizes better across scenarios

---

## Core Architecture

### High-Level Data Flow

```
Configuration (CLI args + models.yaml)
         ↓
Maze Generation (algorithmic or manual presets)
         ↓
GridWorld initialization (obstacles, goal, agents)
         ↓
[SIMULATION LOOP] (for each turn 0..turns-1)
         ├─ Build observations (per agent)
         ├─ [MAP SHARING] (if enabled)
         │  └─ Merge agent maps (radio_sync or global)
         ├─ Gather LLM decisions (async, concurrent)
         ├─ Resolve moves (collision detection, swaps)
         ├─ Log transcript/movements (streamed)
         └─ Update world state
         ↓
Episode completion
         ├─ EpisodeMetrics (success, collisions, turns)
         ├─ transcript.jsonl (all prompts/responses)
         ├─ episode_stream.jsonl (per-turn positions)
         └─ episode.json (complete structured log)
         ↓
Visualization (render_gif → animated GIF)
```

### Directory Structure

```
src/llmgrid/
├── __init__.py
├── schema.py              # Pydantic observation/decision models
├── prompts.py             # LLM system prompt template
├── agent_map.py           # Agent-specific knowledge maps + merging
│
├── agent/
│   ├── llm_agent.py       # LlmPolicy: wraps UnifiedLLM for decisions
│   └── local_baseline.py  # GreedyBaseline: heuristic fallback
│
├── env/
│   ├── grid.py            # GridWorld: state management, movement
│   ├── simulate.py        # run_episode_async: main simulation loop
│   └── maze_generator.py  # MazeGenerator: DFS carving + connectivity
│
├── llm_clients/
│   ├── unified_llm.py     # UnifiedLLM: provider pooling + rate limiting
│   ├── adapter.py         # [Adapter pattern for future providers]
│   └── __init__.py
│
├── logging/
│   └── episode_log.py     # EpisodeLog/Frame schemas for visualization
│
├── cli/
│   ├── poc_two_agents.py  # Main entry point (80+ parameters)
│   ├── run_preset.py      # Simplified preset runner
│   ├── render_gif.py      # Visualization entry point
│   ├── stream_to_episode.py # Convert stream → episode.json
│   ├── render_egocentric.py # [DEPRECATED] Per-agent view
│   └── generate_maze.py   # Maze generation CLI
│
├── vis/
│   ├── gif.py             # GifRenderer: Among Us sprites + PIL
│   └── egocentric.py      # [STUB] Deprecated renderer
│
├── utils/
│   ├── errors.py          # APIError, CircuitBreakerOpenError
│   └── real_time_logger.py # Colored terminal logging
│
└── providers/
    └── openrouter_client.py # [Legacy] OpenRouter support
```

---

## Module Breakdown

### 1. Core Simulation (env/)

#### **grid.py: GridWorld**

The authoritative world state manager.

**Key Classes:**
- `GridWorld`: Manages all agents, walls, goal, and state transitions
- `MoveResult`: Outcome of a single move attempt

**Responsibilities:**
- Track agent positions and occupancy
- Detect collisions (wall blocks, agent blocks, swap conflicts)
- Build observations (visible cells, neighbors, adjacent summary)
- Manage agent-specific maps (persistent knowledge per agent)
- Handle map merging (radio_sync or global)

**Movement Resolution Flow:**
1. Collect all movement intents (None = STAY)
2. Identify wall/OOB blocks
3. Detect swap pairs (mutual position exchanges)
4. Resolve multi-agent collisions (agents trying same cell)
5. Apply moves atomically
6. Mark finished agents
7. Update position history + last_result records

**Example: Collision Detection**
```python
# resolve_moves() handles:
# - Wall/out-of-bounds → BLOCK_WALL outcome
# - Agent occupancy conflict → BLOCK_AGENT outcome
# - Simultaneous swaps → SWAP_CONFLICT outcome
# - Successful move to goal → FINISHED outcome
# - Safe move → OK outcome
```

#### **simulate.py: Episode Driver**

Orchestrates the turn loop and LLM coordination.

**Key Functions:**
- `run_episode_async()`: Main simulation loop (50+ params)
- `_gather_decisions_async()`: Concurrent LLM calls with semaphore
- `_merge_radio()`: Radio-based map sharing (Manhattan distance)
- `_merge_global()`: Global map sharing (all agents sync all base tiles)

**Concurrency Model:**
```python
# Per turn:
# 1. Build observations for all active agents
# 2. [If map_sharing enabled] Merge agent maps
# 3. Spawn N concurrent LLM calls (limited by semaphore)
# 4. Gather all decisions
# 5. Resolve moves synchronously
# 6. Log transcript/movements (streaming JSONL)
```

**Streaming Outputs:**
- `transcript_writer`: JSONL file, one entry per (agent, turn)
- `movement_writer`: JSONL file, one entry per turn (all agent positions)
- Episode metrics collected at end

#### **maze_generator.py: MazeGenerator**

Deterministic maze generation using depth-first carving.

**Algorithm:**
1. Initialize grid as all walls
2. Start from (1,1) and carve using DFS
3. Optionally add extra connections (loop probability)
4. Ensure required cells are open and connected
5. Return obstacles as Position list

**Features:**
- Guaranteed connectivity between supply points
- Tunable loop density via `extra_connection_prob`
- Deterministic seeding for reproducibility

---

### 2. Agent Decision Making (agent/)

#### **llm_agent.py: LlmPolicy**

Wraps the UnifiedLLM client to produce structured decisions.

**Key Methods:**
- `decide_async()`: Returns Decision (MOVE or STAY)
- `decide_with_trace_async()`: Returns DecisionTrace (includes prompt)

**Flow:**
```python
# 1. Build prompt from observation JSON
prompt = build_prompt_header() + observation_json
# 2. Call UnifiedLLM with schema enforcement
decision, _, _ = await unified.run(
    [{"role": "user", "content": prompt}],
    model=model_id,
    output_schema=build_decision_model(strategy),
)
# 3. Coerce to canonical Decision type
return coerce_decision(decision)
```

**Parameters Passed:**
- `model_id`: Key to models.yaml (e.g., "gpt-5-mini")
- `strategy`: Communication strategy (currently unused; always "none")
- `loop_guidance`: Loop detection mode (passive, active, explore)
- `history_limit`: Observation history window
- `radio_range`: For observation building

#### **local_baseline.py: GreedyBaseline**

Heuristic fallback for dry runs (no LLM calls).

**Decision Logic:**
1. Prefer exploring unknown (adjacent_frontiers)
2. Move toward goal if known
3. Pick any free neighbor
4. STAY if blocked

---

### 3. Schema & Observation (schema.py)

**Observation Structure** (sent to LLM):
```python
{
  "protocol_version": "3.0.0",
  "turn_index": int,
  "max_turns": int,
  "grid": {
    "width": int,
    "height": int,
    "rows": [["#","@",".","X",...], ...]  # y-indexed array
  },
  "legend": {
    "#": "WALL",
    "@": "SELF",
    "G": "GOAL",
    "X": "UNKNOWN",
    "1,2,3...": "OTHER AGENTS",
    "!": "last collision target",
    "*": "recent positions (last 3)",
    "~": "visited (older)",
    ".": "free (not visited)"
  },
  "self": {"agent_id": "a1", "pos": {"x": 10, "y": 5}},
  "neighbors_in_view": [
    {"agent_id": "a2", "pos": {"x": 12, "y": 5}}
  ],
  "adjacent": [
    {"dir": "N", "state": "FREE"},
    {"dir": "E", "state": "AGENT"},
    ...
  ],
  "adjacent_frontiers": [{"x": 8, "y": 5}],  # Unknown cells next to known
  "goal_known": true,
  "goal_pos": {"x": 28, "y": 1},
  "last_result": {
    "kind": "OK" | "BLOCK_WALL" | "BLOCK_AGENT" | "SWAP_CONFLICT" | "FINISHED",
    "cell": {"x": 11, "y": 5},
    "opponents": ["a2"]
  },
  "map_sharing": "none" | "radio_sync" | "global"
}
```

**Decision Schema** (returned by LLM):
```python
{
  "action": {
    "kind": "MOVE",
    "direction": "N" | "E" | "S" | "W"
  } | {
    "kind": "STAY"
  },
  "comment": "str <= 25 words"
}
```

---

### 4. Agent Maps & Knowledge Sharing (agent_map.py)

**AgentMap: Per-agent Persistent Knowledge**

Each agent maintains a local `AgentMap` that persists across turns:
- `_base`: Grid of base tiles (wall=#, goal=G, free=., unknown=X)
- `visited`: Set of cells the agent has seen
- `recent`: Deque of last N positions (default 3)
- `last_collision`: Coordinate of last blocked cell

**Rendering (for observation):**
Priority overlay system:
```
@ > digit(other agent) > ! (collision) > * (recent) > ~ (visited) > . (free)
```

**Map Merging:**

Two modes of knowledge sharing:

1. **Radio Sync** (radio_range-based):
   - Within Manhattan distance ≤ radio_range, agents sync base tiles
   - Only base tiles merge; overlays (recent, visited) stay personal
   ```python
   _merge_radio(world, agents, radio_range):
       for each pair (aid, bid) within radio_range:
           merge_base_from(aid → bid)
           merge_base_from(bid → aid)
   ```

2. **Global** (full sharing):
   - All agents accumulate into one master base map
   - All agents' base maps sync to the master
   ```python
   _merge_global(world):
       master = world.agent_maps[agents[0]]
       for each other agent:
           master.merge_base_from(other)
       for each other agent:
           other.merge_base_from(master)
   ```

---

### 5. LLM Integration (llm_clients/unified_llm.py)

**UnifiedLLM: Provider Pooling & Rate Limiting**

The critical bridge between simulation and LLM providers.

**Key Components:**

1. **ModelPool**: Round-robin selection with failure tracking
   - Pools of models for each key (e.g., gpt-5-mini has 2 deployments)
   - Temporary blacklisting on failure (60s default)
   - Circuit breaker on provider (3 failures → 90s cooldown)

2. **RateLimiter**: Three-layer adaptive limiting
   - Concurrency semaphore (per provider)
   - RPM (requests/minute) sliding window
   - TPM (tokens/minute) sliding window with token estimation
   - Adaptive utilization: starts at 85%, reduces on rate limit, recovers on success

3. **TokenTracker**: Accounting for token usage
   - Per-model counters (input/output)
   - Historical call log

**Configuration (models.yaml):**
```yaml
providers:
  azure:
    type: "azure"
    base_url: "https://instance.openai.azure.com"
    api_version: "2025-03-01-preview"
    api_key_env: "AZURE_OPENAI_API_KEY"

model_pools:
  gpt-5-mini:
    - provider: "azure"
      deployment: "gpt-5-mini"
      mode: "sdk"              # How to call provider
      reasoning_effort: "minimal"

limits:
  adaptive:
    enabled: true
    initial_utilization: 0.85
    reduction_factor: 0.8
  models:
    "azure:gpt-5-mini":
      rpm: 360
      tpm: 1426500
```

**API Modes:**
- `mode: "sdk"`: Direct AsyncOpenAI client (Azure, standard OpenAI)
- `mode: "agent"`: Pydantic AI Agent wrapper (Anthropic, OpenRouter)
- `mode: "responses"`: Azure Responses API endpoint (structured output)

**Critical Fix (2025-11-19):**
All AsyncOpenAI clients MUST use `async with` context managers to properly close connection pools. Without this, requests hang indefinitely.

---

### 6. Prompts (prompts.py)

**System Prompt: CORE_HEADER**

70-line instruction set covering:
- Objective (reach goal, help teammates, avoid collisions)
- Grid representation (coordinates, indexing, symbols)
- Symbol legend (walls, goal, unknown, agents, overlays)
- Visibility radius behavior
- Last result interpretation (collision outcomes)
- Decision hierarchy (avoid walls → avoid collisions → explore → move to goal)
- Output contract (JSON object with action and comment)

**Example Prompt Flow:**
```
[CORE_HEADER] + [OBSERVATION_JSON] → LLM → Decision JSON
```

No communication strategies in prompts; reserved for future work.

---

### 7. Logging & Telemetry (logging/episode_log.py)

**EpisodeLog Schema** (for visualization):
```python
{
  "meta": {
    "grid_size": {"width": 30, "height": 10},
    "goal": {"x": 28, "y": 1},
    "walls": [{"x": 0, "y": 0}, ...],
    "view": {"kind": "square", "radius": 2},
    "gradient_mode": "bfs",
    "title": "long_corridor, 5 agents, gpt-5-mini",
    "agent_styles": [
      {"agent_id": "a1", "color_hex": "#1f77b4"},
      ...
    ]
  },
  "frames": [
    {
      "t": 0,
      "agents": [
        {
          "agent_id": "a1",
          "pos": {"x": 9, "y": 9},
          "orientation": "N",
          "action": null,
          "status": "ACTIVE"
        },
        ...
      ]
    },
    ...
  ]
}
```

**Streaming Outputs (written during simulation):**

1. **transcript.jsonl** (one per agent per turn):
   ```json
   {
     "turn": 0,
     "agent_id": "a1",
     "prompt": "[full prompt text]",
     "observation": {...},
     "decision": {...},
     "trace_messages": []
   }
   ```

2. **episode_stream.jsonl** (one per turn):
   ```json
   {
     "turn": 0,
     "agents": {
       "a1": {"x": 9, "y": 9, "orientation": "N", "action": null, "status": "ACTIVE"},
       ...
     }
   }
   ```

**End-of-Run Output:**
- **episode.json**: Complete structured log with metadata
- **metrics.json**: Summary (success, collisions, turns, etc.)

---

## Data Flow

### Turn-by-Turn Execution

```
TURN T
│
├─ [Observation Phase]
│  ├─ For each active agent:
│  │  ├─ Get occupancy, walls, goal
│  │  ├─ Calculate visible cells (Manhattan radius)
│  │  ├─ Find neighbors in view
│  │  ├─ Build adjacent summary (N/E/S/W states)
│  │  ├─ Render grid using AgentMap + overlays
│  │  └─ Create Observation JSON
│  │
│  └─ [Map Sharing] (if enabled, AFTER observations built)
│     ├─ If radio_sync: merge pairs within radio_range
│     └─ If global: sync all base maps
│
├─ [Decision Phase]
│  ├─ Spawn concurrent tasks (max = semaphore size)
│  ├─ Each task:
│  │  ├─ Call LlmPolicy.decide_async(observation)
│  │  └─ Get Decision(action, comment)
│  └─ Gather all decisions
│
├─ [Movement Phase]
│  ├─ Extract intents (Direction | None from decisions)
│  ├─ world.resolve_moves(intents)
│  │  ├─ Check wall/OOB blocks
│  │  ├─ Detect swaps
│  │  ├─ Resolve multi-agent collisions
│  │  ├─ Apply moves
│  │  └─ Mark finished agents
│  └─ Get MoveResults
│
├─ [Logging Phase]
│  ├─ Write transcript.jsonl (if enabled)
│  ├─ Write episode_stream.jsonl (if enabled)
│  ├─ Count collisions
│  └─ Update reasoning_log
│
└─ [Check Completion]
   ├─ If all agents finished OR turn == max_turns: EXIT LOOP
   └─ Else: NEXT TURN

[End of Episode]
├─ Compile EpisodeMetrics
├─ Write episode.json (complete log)
└─ Return metrics
```

---

## Key Abstractions

### GridWorld

**State Machine:**
```python
GridWorld
├── occupancy: Dict[agent_id → (x, y)]
├── finished: Dict[agent_id → bool]
├── walls: Set[(x, y)]
├── goal: (x, y)
├── agent_maps: Dict[agent_id → AgentMap]
├── position_history: Dict[agent_id → List[(x,y)]]
├── last_result: Dict[agent_id → LastResult]
└── [METHODS]
    ├── add_agent()
    ├── build_observation()
    ├── resolve_moves()
    └── merge_base_maps()
```

### AgentMap

**Per-Agent Knowledge:**
```python
AgentMap
├── _base: List[List[str]]        # Wall/goal/free/unknown grid
├── visited: Set[(x,y)]            # All seen cells
├── recent: Deque[(x,y)]           # Last N positions
├── last_collision: Optional[(x,y)] # Most recent blocked cell
└── [METHODS]
    ├── update_visible()
    ├── find_frontiers()
    ├── merge_base_from()          # Radio/global sharing
    └── render_grid()              # For observation
```

### LlmPolicy

**Agent Brain:**
```python
LlmPolicy
├── model_id: str
├── strategy: str
├── history_limit: int
├── unified: UnifiedLLM
└── [METHODS]
    └── decide_async(observation) → Decision
```

---

## Communication & Coordination

**Current Status:** Communication strategies are not yet implemented in the codebase (reserved for future work).

**Map Sharing (Primary Coordination Mechanism):**

Three modes control how agents share knowledge of the world:

1. **none**: No sharing
   - Each agent's AgentMap evolves independently
   - Agents only see what they personally observe

2. **radio_sync**: Distance-based sharing (default)
   - Agents within `radio_range` (Manhattan distance) sync base maps
   - Before observation rebuild each turn
   - Creates "map pockets" of shared knowledge

3. **global**: Full sharing every turn
   - All agents accumulate into master base map
   - Master syncs back to all agents
   - Equivalent to perfect information about walls/free/goal cells

**Observation Fields Supporting Coordination:**
- `neighbors_in_view`: List of visible agents (positions only)
- `adjacent_frontiers`: Unknown cells next to known areas
- `map_sharing`: String indicating current sharing mode

**Example Scenario (2 agents with radio_sync):**
```
Initial state:
- Agent A: knows (0,0) is a wall
- Agent B: no knowledge of (0,0)
- Distance: 2 cells, radio_range: 2

After merge:
- Both agents: (0,0) is a wall
- If moved to distance 3: no merge
```

---

## Maze/Grid Generation

### Presets

**Six curated presets** (deterministic seeds, no generation required):
1. `long_corridor` (30×10, seed 606, maze style)
2. `open_sparse` (20×12, seed 101, random 12%)
3. `open_dense` (20×12, seed 202, random 25%)
4. `maze_tight` (21×13, seed 303, maze style, low loops)
5. `maze_loops` (21×13, seed 404, maze style, many loops)
6. `mixed_medium` (24×14, seed 505, random 18%)

### Manual Preset System

Additional maze presets can be added as static files:
```
experiments/presets/batch/
├── abmarl_maze_8103.txt        # ASCII maze
├── abmarl_maze_8103_meta.json  # Metadata (goal, start positions)
├── choke_points_comm_test.txt
├── choke_points_comm_test_meta.json
└── ... (12+ more custom mazes)
```

### Algorithmic Generation

**MazeGenerator:**
- Depth-first carving with guaranteed connectivity
- Optional extra connections (loop probability 0.0–1.0)
- Deterministic seeding

**Example Usage:**
```python
gen = MazeGenerator(MazeConfig(width=30, height=10, seed=606, extra_connection_prob=0.2))
obstacles = gen.generate(required_open_cells=[(1,1), (28,1)])
```

---

## LLM Integration Layer

### Flow: Observation → Decision

```python
# In LlmPolicy.decide_async()

# 1. Build prompt
prompt = build_prompt_header() + json.dumps(observation, separators=(',', ':'))

# 2. Call unified LLM with Pydantic schema enforcement
wire_decision, _, _ = await unified.run(
    [{"role": "user", "content": prompt}],
    model=model_id,
    output_schema=build_decision_model(strategy),  # Schema-enforced
    max_spatial_retries=3,
)

# 3. Coerce to canonical Decision type
return coerce_decision(wire_decision)
```

### Schema Enforcement

**Pydantic Schema Validation:**
- All decisions validated against `Decision_<strategy>_nocomm` schema
- Invalid JSON returns error; LLM retried (max 3 attempts)
- Prevents hallucinated actions or malformed JSON

**Reasoning Models (gpt-5, o1, o3):**
- Temperature parameter automatically omitted
- Reasoning effort/verbosity passed through
- Critical: code in `unified_llm.py` commit `f4e0863` handles this

### Rate Limiting

**Per-Turn Concurrency:**
- Semaphore-limited concurrent LLM calls
- Default: `concurrency_start = min(agent_count, 6)`
- Adaptive rate limiting: starts at 85% utilization, reduces on limit

**Token Budgeting:**
- TPM (tokens/minute) sliding window
- Estimated tokens/call; actual recorded on completion
- Prevents quota exhaustion

---

## Visualization Pipeline

### Architecture

```
Run produces:
├── transcript.jsonl (streaming: prompts/responses)
├── episode_stream.jsonl (streaming: per-turn positions)
├── config.yaml (maze metadata, styles)
└── episode.json (complete structured log) [written at end]

Converting (if needed):
└─ stream_to_episode.py converts episode_stream + config → episode.json

Rendering:
└─ render_gif.py: episode.json + optional transcript.jsonl → GIF

GifRenderer (gif.py):
├── Loads Among Us sprites (directional: N/E/S/W)
├── Per-frame rendering:
│  ├── Background + grid lines
│  ├── Walls (black)
│  ├── Goal (gold, pulsing when reached)
│  ├── Visibility auras (translucent)
│  ├── Agent sprites (colored, directional)
│  └─ Legend (turn, model, agent list)
└── Outputs animated GIF at specified FPS
```

### GifRenderer (gif.py)

**Features:**
- Among Us crewmate sprites with directional animation
- Per-agent color coding (10 colors by default)
- Visibility auras (3×3 around each agent)
- Goal gradient overlay (optional, BFS-based)
- Grid overlay (optional)
- Legend panel (turn number, model name, agent colors)

**Example Usage:**
```bash
uv run python -m llmgrid.cli.render_gif \
  experiments/run/results/episode.json \
  --out experiments/run/results/episode.gif \
  --cell-size 40 \
  --fps 6 \
  --gradient
```

### stream_to_episode.py

**Purpose:** Convert streaming outputs to complete episode.json (for in-progress visualization).

**Usage:**
```bash
uv run python -m llmgrid.cli.stream_to_episode \
  experiments/run/results/episode_stream.jsonl \
  experiments/run/config.yaml \
  --out /tmp/partial.json \
  --max-turns 20
```

---

## Experiment Framework

### Directory Structure

```
experiments/
├── README.md                    # Master index + results summary
│
├── cross_seed_baseline_20251112T143355Z/
│  ├── README.md                # Experiment story, hypothesis, results
│  ├── baselines.json           # Parameter matrix
│  └─ runs/
│     ├── seed13_structured_20251112T175321Z/
│     │  ├── config.yaml        # Resolved configuration
│     │  ├── run.log            # Complete stdout/stderr
│     │  └── results/
│     │     ├── metrics.json    # Episode summary
│     │     ├── episode.json    # Full visualization log
│     │     ├── transcript.jsonl # Prompts/responses
│     │     └── episode_stream.jsonl # Per-turn positions
│     │
│     └── seed14_structured_...
│
├── long_corridor_global_share_20251119T202017Z/
│  ├── README.md
│  └── runs/
│     ├── seed13_20251119T202030Z/
│     └── seed14_20251119T202045Z/
│
└── presets/
   └── batch/
      ├── long_corridor_preview.png
      ├── choke_points_comm_test.txt
      ├── choke_points_comm_test_meta.json
      └── ... (12+ more)
```

### Naming Conventions

**Experiment folders:** `semantic-name_YYYYMMDDTHHMMSSxZ`
```
temperature-sweep_20251027T143201Z
cross_seed_baseline_20251112T143355Z
```

**Run folders:** `semantic-run-name_YYYYMMDDTHHMMSSxZ`
```
seed13_structured_20251112T175321Z
freeform_kv_light_seed0_20251117T195901Z
```

### CLI Entry Points

**Main Experiment Runner:**
```bash
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model gpt-5-mini \
  --maze-preset long_corridor \
  --agents 5 \
  --turns 100 \
  --comm-strategy none \
  --map-sharing radio_sync \
  --seed 13 \
  --emit-config experiments/my_exp/config.yaml \
  --episode-json experiments/my_exp/results/episode.json \
  --transcript-jsonl experiments/my_exp/results/transcript.jsonl
```

**Simplified Preset Runner:**
```bash
uv run python -m llmgrid.cli.run_preset \
  --preset long_corridor \
  --turns 100 \
  --model gpt-5-mini \
  --episode-json results/episode.json \
  --transcript-jsonl results/transcript.jsonl
```

### Key Parameters (poc_two_agents)

**Grid Configuration:**
- `--width`, `--height`: Grid dimensions
- `--visibility`: Visibility radius (default 2)
- `--radio-range`: Radio range for map sharing
- `--turns`: Turn budget
- `--seed`: Random seed

**Agent Configuration:**
- `--agents`: Number of agents
- `--comm-strategy`: Communication mode (none, structured, freeform)
- `--map-sharing`: Map sharing mode (none, radio_sync, global)
- `--history-limit`: Prior turns in observation

**Maze Configuration:**
- `--maze-preset`: Preset name (long_corridor, etc.)
- `--maze-style`: maze | random | manual
- `--obstacle-density`: For random style

**Output & Logging:**
- `--emit-config`: YAML file with resolved parameters
- `--episode-json`: Episode visualization data
- `--transcript-jsonl`: Prompt/response log
- `--log-movements`: Log agent positions per turn

**Execution:**
- `--dry-run`: Use heuristic baseline instead of LLM
- `--concurrency-start`, `--concurrency-max`: Concurrent LLM calls

### Monitoring Long Runs (tmux)

```bash
# Start in detached tmux session
SESSION="my_exp_$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/${SESSION}.log"
tmux new-session -d -s "$SESSION" \
  "set -euo pipefail
   export PYTHONUNBUFFERED=1
   [ -f ~/.env ] && set -a && source ~/.env && set +a
   echo 'run_start model=gpt-5-mini turns=100' >&2
   exec uv run python -m llmgrid.cli.poc_two_agents ... >> '$LOG' 2>&1"

# Monitor
tail -f "$LOG"

# Or check health
tmux has-session -t "$SESSION" && echo "Running" || echo "Done"
```

---

## Integration Points

### How Everything Connects

1. **Configuration → GridWorld:**
   - CLI params + models.yaml → maze generation → GridWorld init

2. **GridWorld → Observation:**
   - Agent positions + visible cells + AgentMap → Observation JSON

3. **Observation → LLM:**
   - Pydantic schema enforcement → prompt construction → UnifiedLLM.run()

4. **Decision → Movement:**
   - Decision JSON → direction extract → GridWorld.resolve_moves()

5. **Simulation → Logging:**
   - Per-turn decisions + movements → transcript.jsonl + episode_stream.jsonl

6. **Logging → Visualization:**
   - episode.json + transcript.jsonl → GifRenderer → animated GIF

### Code Paths

**Starting a run:**
```
poc_two_agents.main()
 → _load_maze_meta() [if needed]
 → run_episode_async()
   → GridWorld init
   → for turn in range(turns):
     → build_observation() [per agent]
     → [map merge if enabled]
     → _gather_decisions_async()
       → LlmPolicy.decide_async()
         → UnifiedLLM.run()
           → ModelPool.select()
           → RateLimiter.acquire()
           → [AsyncOpenAI or pydantic_ai call]
     → GridWorld.resolve_moves()
     → Log transcript/movements
   → Return EpisodeMetrics
```

**Visualizing results:**
```
render_gif.main()
 → Load episode.json
 → [Auto-detect transcript.jsonl if exists]
 → GifRenderer.render()
   → Load sprites
   → Per-frame rendering
     → Draw grid + walls
     → Draw agents + auras
     → Draw goal + legend
   → PIL.Image.save() → GIF
```

---

## Performance Characteristics

### Expected Latency

**Per-turn (single agent, gpt-5-mini):**
- Observation building: ~10ms
- LLM call: ~500-1000ms (network dependent)
- Movement resolution: ~5ms
- Logging: ~10ms
- **Total: ~1-2 seconds**

**For 5 agents with concurrency_start=5:**
- Parallel LLM calls: ~500-1000ms (same as single)
- Total per turn: ~1-2 seconds
- 100-turn run: ~100-200 seconds (1.5–3 minutes)

**For 50-turn run with 5 agents:**
- Expected: ~50-100 seconds (1–2 minutes)

### Memory Usage

- GridWorld state: O(width × height + agents)
- AgentMaps: O(agents × width × height)
- Transcript buffer: O(turns × agents)
- Episode stream: O(turns × agents)
- **Typical (30×10, 5 agents, 100 turns): ~50MB**

### Concurrency Limits

- Semaphore size limited by provider rate limits
- Default: 4-6 concurrent calls
- Azure tpm limit: 2.7M tokens/minute
- At ~200 tokens/call, supports ~13500 calls/minute → ~225 agents (unrealistic)

---

## Key Learnings & Design Decisions

### Why Pydantic Schemas?

- **Strict Output Validation:** Rejects malformed JSON at provider boundary
- **Type Safety:** Prevents downstream parsing errors
- **Composability:** Easy to extend for future communication strategies

### Why Streaming JSONL?

- **Memory Efficiency:** Large episodes don't require buffering
- **Live Monitoring:** Can analyze in-progress runs
- **Incremental Visualization:** Can render partial episodes while still running

### Why Map Sharing (not communication)?

- **Empirical Finding:** Communication in structured protocols underperformed in 45-run study
- **Simpler Coordination:** Knowledge sharing naturally aligns incentives
- **Reproducibility:** Deterministic behavior for research

### Why Agent Maps?

- **Persistent Knowledge:** Agents learn from exploration history
- **Efficient Rendering:** Separate base (walls/goal) from overlays (recent/visited)
- **Knowledge Merging:** Foundation for radio_sync and global sharing

### Why Async Context Managers?

- **Connection Pool Cleanup:** Prevents resource exhaustion on high-concurrency runs
- **Critical Fix:** Commit f4e0863 added `async with` to all AsyncOpenAI clients
- **Hangs Eliminated:** Previously, 5-agent runs would hang indefinitely

---

## Current Work (as of 2025-11-19)

**In-flight Verification Plan:**

Run canonical scenarios under three map-sharing regimes to establish baseline:
1. No sharing (map_sharing=none)
2. Radio sync (map_sharing=radio_sync, default)
3. Global sharing (map_sharing=global)

Each: 5 seeds (13–17), long_corridor preset, 5 agents, gpt-5-mini

**Branch:** micro-blocked-tunnel  
**Status:** Running (3 experiments, 15 runs)

Once complete, pipeline will be fully verified for fresh science.

---

## Future Work

1. **Communication Strategies:**
   - Implement freeform natural language in prompts
   - Structured INTENT/REQUEST protocol
   - Measure token efficiency vs. map sharing

2. **Advanced Mazes:**
   - Chokepoint grids requiring coordination
   - Lock-and-key structures
   - Dynamically generated mazes during simulation

3. **Agent Modeling:**
   - Opponent intention inference
   - Multi-turn planning
   - Coalition formation

4. **Scaling:**
   - 10+ agent teams
   - Larger grids (100×100)
   - Hierarchical coordination

---

## File Reference Map

| File | Purpose |
|------|---------|
| `schema.py` | Observation/Decision Pydantic models |
| `prompts.py` | LLM system prompt template |
| `agent_map.py` | Per-agent knowledge + merging |
| `agent/llm_agent.py` | LlmPolicy wrapper |
| `agent/local_baseline.py` | Heuristic fallback |
| `env/grid.py` | GridWorld state + movement |
| `env/simulate.py` | Turn loop + LLM coordination |
| `env/maze_generator.py` | Algorithmic maze generation |
| `llm_clients/unified_llm.py` | Provider pooling + rate limiting |
| `logging/episode_log.py` | Visualization schemas |
| `cli/poc_two_agents.py` | Main CLI (80+ params) |
| `vis/gif.py` | GifRenderer with sprites |
| `utils/real_time_logger.py` | Colored logging |

---

## Summary

**LLM Grid Agents** is a tightly integrated research framework balancing:
- **Simulation fidelity:** Detailed collision mechanics, turn synchronization
- **LLM integration:** Schema enforcement, provider pooling, rate limiting
- **Reproducibility:** Deterministic mazes, seeding, checkpoint support
- **Observability:** Streaming telemetry, rich visualization, detailed logging

The architecture cleanly separates concerns (simulation, agents, LLM calls, logging) while maintaining tight coupling through well-defined Pydantic schemas. The map-sharing mechanism provides a foundation for future communication research, and the streaming pipeline enables real-time monitoring of long-running experiments.

