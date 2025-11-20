# LLM Grid Agents: Codebase Exploration Summary

**Exploration Date:** 2025-11-19  
**Repository:** `/Users/3bn/Documents/My_Repos/agent-talk-2` (agent-talk-2)  
**Current Branch:** micro-blocked-tunnel  
**Analysis Depth:** Very Thorough (all core modules examined)

## Quick Navigation

Full architecture guide saved to: `/Users/3bn/Documents/My_Repos/agent-talk-2/ARCHITECTURE.md` (1131 lines)

---

## Project at a Glance

**LLM Grid Agents** is a research framework for studying multi-agent coordination in partially observable grid environments.

Key characteristics:
- Multi-agent LLM-driven navigation (1-8 agents)
- Partial observability with configurable visibility radius
- Synchronous turns with simultaneous decision making
- Three map-sharing modes (none, radio_sync, global)
- Structured I/O via Pydantic schemas
- Rich visualization pipeline (Among Us sprites in GIFs)
- 45-run empirical study showing freeform communication > structured protocols

---

## Core Findings

### Architecture Pattern

The codebase follows a **tightly integrated modular architecture** around Pydantic schemas:

```
Configuration (CLI)
    ↓
Maze Generation (MazeGenerator + presets)
    ↓
GridWorld (state machine)
    ↓
Observation Building (per-agent snapshots)
    ↓
LLM Decisions (AsyncOpenAI/pydantic_ai with schema validation)
    ↓
Movement Resolution (collision detection + synchronization)
    ↓
Streaming Logs (JSONL format)
    ↓
Visualization (PIL-based GIF renderer)
```

### Critical Integration Points

**1. Schema-Enforced I/O (schema.py)**
- Observation: 11 top-level fields (grid, neighbors, adjacent, goal, etc.)
- Decision: action (MOVE | STAY) + comment (<=25 words)
- All validation at provider boundary; prevents hallucination/malformed JSON

**2. Provider Pooling (unified_llm.py)**
- Round-robin model selection across multiple deployments
- Three-layer rate limiting (concurrency semaphore, RPM, TPM)
- Adaptive utilization (starts 85%, reduces on limit, recovers on success)
- Critical fix (f4e0863): All AsyncOpenAI clients use `async with` to prevent hangs

**3. Knowledge Sharing (agent_map.py + simulate.py)**
- Per-agent persistent maps (_base, visited, recent, last_collision)
- Two merging modes:
  - **radio_sync**: Distance-based (Manhattan <= radio_range)
  - **global**: All agents accumulate to master, master syncs back
- Only base tiles (walls/goal/free/unknown) merge; overlays stay personal

**4. Collision Resolution (grid.py)**
- Wall/OOB blocks → BLOCK_WALL
- Agent occupancy conflict → BLOCK_AGENT
- Mutual position swaps → SWAP_CONFLICT
- Reach goal → FINISHED
- Otherwise → OK
- Fully deterministic, no randomness

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pydantic schemas | Strict output validation at provider boundary |
| Streaming JSONL | Memory efficiency + live monitoring for long runs |
| Map sharing over messages | Empirical: freeform > structured; knowledge sharing simpler |
| Agent maps with overlays | Efficient rendering (base separate from visited/recent) |
| Async context managers | Prevents AsyncOpenAI connection pool exhaustion |
| Deterministic mazes + seeding | Reproducibility for research |

---

## Module Breakdown

### Core Simulation (5 files, ~800 LOC)

| Module | Responsibility | Key Classes |
|--------|-----------------|------------|
| `env/grid.py` | World state, collision detection | GridWorld, MoveResult |
| `env/simulate.py` | Turn loop, LLM coordination | run_episode_async(), RateLimiter callbacks |
| `env/maze_generator.py` | DFS carving + connectivity | MazeGenerator, MazeConfig |
| `agent_map.py` | Per-agent persistent maps | AgentMap (with merge_base_from) |
| `schema.py` | Observation/Decision types | Observation, Decision, Direction, etc. |

### Agent Decision Making (2 files, ~150 LOC)

| Module | Purpose | Fallback |
|--------|---------|----------|
| `agent/llm_agent.py` | LLM-based policy | UnifiedLLM wrapper |
| `agent/local_baseline.py` | Heuristic fallback | GreedyBaseline (explore frontiers → goal → free) |

### LLM Integration (1 file, ~800 LOC)

| Component | Role |
|-----------|------|
| ModelPool | Round-robin selection + failure tracking |
| RateLimiter | Concurrency semaphore + RPM/TPM windows |
| TokenTracker | Token accounting |
| UnifiedLLM.run() | Single unified entry point (adapter pattern ready) |

### Visualization (3 files, ~400 LOC)

| Module | Purpose |
|--------|---------|
| `vis/gif.py` | GifRenderer with Among Us sprites |
| `cli/render_gif.py` | CLI wrapper |
| `cli/stream_to_episode.py` | Convert JSONL streams to episode.json |

### CLI & Utilities (6 files, ~500 LOC)

| Module | Params |
|--------|--------|
| `cli/poc_two_agents.py` | 80+ parameters (grid, agents, maze, output, execution) |
| `cli/run_preset.py` | Simplified preset runner |
| `utils/real_time_logger.py` | Colored terminal output |
| `logging/episode_log.py` | Visualization schemas |

---

## Data Structures

### Observation (JSON to LLM)

```python
{
  "protocol_version": "3.0.0",
  "turn_index": int,
  "max_turns": int,
  "grid": {
    "rows": [["#","@",".","X",...], ...]  # y-indexed
  },
  "self": {"agent_id": "a1", "pos": {"x": 10, "y": 5}},
  "neighbors_in_view": [{"agent_id": "a2", "pos": {...}}],
  "adjacent": [{"dir": "N", "state": "FREE"}, ...],
  "adjacent_frontiers": [{"x": 8, "y": 5}],
  "goal_known": bool,
  "goal_pos": {"x": 28, "y": 1},
  "last_result": {
    "kind": "OK|BLOCK_WALL|BLOCK_AGENT|SWAP_CONFLICT|FINISHED",
    "cell": {...},
    "opponents": [...]
  },
  "map_sharing": "none|radio_sync|global"
}
```

### Decision (LLM to Simulation)

```python
{
  "action": {
    "kind": "MOVE|STAY",
    "direction": "N|E|S|W"  # for MOVE
  },
  "comment": "str <= 25 words"
}
```

### GridWorld State

```python
GridWorld
├── occupancy: Dict[agent_id → (x, y)]
├── finished: Dict[agent_id → bool]
├── walls: Set[(x, y)]
├── goal: (x, y)
├── agent_maps: Dict[agent_id → AgentMap]
├── last_result: Dict[agent_id → LastResult]
└── position_history: Dict[agent_id → deque of (x,y)]
```

### AgentMap (Per-Agent Knowledge)

```python
AgentMap
├── _base: List[List[str]]        # X (unknown) | # (wall) | G (goal) | . (free)
├── visited: Set[(x, y)]
├── recent: Deque[(x, y)]         # Last N positions
├── last_collision: Optional[(x, y)]
└── Rendering overlay priority: @ > digit > ! > * > ~ > .
```

---

## Execution Flow

### Single Turn (Detailed)

```
1. OBSERVATION PHASE
   ├─ For each active agent:
   │  ├─ Visible cells (Manhattan radius)
   │  ├─ Neighbors in view (same radius, other agents)
   │  ├─ Adjacent 4-cell summary (N/E/S/W states)
   │  ├─ Frontier detection (unknown adjacent to known)
   │  └─ Grid rendering (AgentMap + overlays)
   └─ Build Observation JSON

2. [OPTIONAL] MAP SHARING
   ├─ If radio_sync: Merge pairs within radio_range
   └─ If global: Sync all base maps

3. DECISION PHASE
   ├─ Spawn concurrent LLM tasks (semaphore-limited)
   ├─ Each task: LlmPolicy.decide_async(observation)
   └─ Gather all Decision objects

4. MOVEMENT PHASE
   ├─ Extract intents (Direction | None)
   ├─ Resolve collisions:
   │  ├─ Wall/OOB blocks
   │  ├─ Swap pairs
   │  └─ Multi-agent conflicts
   └─ Apply moves atomically

5. LOGGING PHASE
   ├─ Write transcript.jsonl (if enabled)
   ├─ Write episode_stream.jsonl (if enabled)
   └─ Update metrics

6. CHECK COMPLETION
   └─ If all finished OR turn >= max_turns: EXIT
```

### End of Episode

```
EpisodeMetrics
├── turns: int
├── success: bool (all agents finished)
├── collisions: int (BLOCK_AGENT outcomes)
├── reasoning_log: [...]
└── collision_causes: Dict[outcome → count]

Output files:
├── episode.json (complete structured log)
├── metrics.json (summary)
├── transcript.jsonl (each line: turn, agent_id, prompt, observation, decision)
└── episode_stream.jsonl (each line: turn, all agent positions)
```

---

## Map Sharing Mechanisms

### Three Modes

**1. none** (baseline)
- Each agent's AgentMap evolves independently
- No knowledge transfer between agents
- Agents only see personal observations

**2. radio_sync** (default)
- Per turn, after observations built
- Agents within Manhattan distance <= radio_range sync base tiles
- Creates "pockets" of shared knowledge
- Example: 2 agents at distance 2 with radio_range=2 → maps merge

**3. global** (perfect information about topology)
- All agents accumulate into master base map
- Master syncs back to all agents
- Equivalent to "perfect information about walls/goal/free, imperfect about positions"

### Implementation

```python
# radio_sync
for each pair (aid, bid) within radio_range:
    world.agent_maps[bid].merge_base_from(world.agent_maps[aid])
    world.agent_maps[aid].merge_base_from(world.agent_maps[bid])

# global
master = world.agent_maps[agents[0]]
for agent in agents[1:]:
    master.merge_base_from(agent)
for agent in agents[1:]:
    agent.merge_base_from(master)
```

---

## Maze System

### Six Presets (Deterministic)

| Preset | Dims | Style | Seed | Extra Connections |
|--------|------|-------|------|-------------------|
| long_corridor | 30×10 | maze | 606 | 0.2 |
| open_sparse | 20×12 | random | 101 | 0.0 (12% density) |
| open_dense | 20×12 | random | 202 | 0.0 (25% density) |
| maze_tight | 21×13 | maze | 303 | 0.05 |
| maze_loops | 21×13 | maze | 404 | 0.35 |
| mixed_medium | 24×14 | random | 505 | 0.0 (18% density) |

### Algorithm (MazeGenerator)

```
1. Initialize grid as all walls
2. Start from (1,1), use DFS to carve passages
3. Optionally add extra connections (loop probability)
4. Ensure required cells (spawn, goal) are open and connected
5. Return obstacle positions
```

Deterministic seeding ensures reproducibility.

---

## LLM Integration Details

### Configuration (models.yaml)

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
      mode: "sdk"  # How to call (sdk|agent|responses)

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

### Rate Limiting (Three Layers)

1. **Concurrency Semaphore**: Max parallel requests per provider
2. **RPM Window**: Requests per minute (sliding 60-second window)
3. **TPM Window**: Tokens per minute (with per-call estimation)

### Adaptive Behavior

```
Start: 85% utilization
Hit rate limit → Reduce to 68% (85% × 0.8)
60s success → Recover to 73% (68% + 5%)
Eventually back to 95% max
```

---

## Current State & Recent Work

### Status (2025-11-19)

**Running:** Map-sharing verification (3 experiments, 5 seeds each)
- `long_corridor_no_share_20251119T202017Z` (map_sharing=none)
- `long_corridor_radio_sync_20251119T202017Z` (map_sharing=radio_sync)
- `long_corridor_global_share_20251119T202017Z` (map_sharing=global)

**Key Finding (45-run cross-seed baseline, Nov 2025):**
- Freeform communication: 62.7% success (47/75 agents)
- No communication: 57.3% success (43/75 agents)
- Structured protocol: 56.0% success (42/75 agents)

**Insight:** Freeform communication generalizes better; structured protocols underperformed (contrary to seed-13 canonical where structured was 73%).

### Recent Fixes

1. **Connection Pool Exhaustion (f4e0863)**: All AsyncOpenAI clients now use `async with` context managers
2. **Temperature for Reasoning Models**: Automatically omitted for gpt-5/o1/o3
3. **Streaming Telemetry**: transcript.jsonl + episode_stream.jsonl written live during simulation
4. **Visualization Pipeline**: Complete from simulation through GIF rendering

---

## Performance Expectations

### Per-Turn Latency

| Component | Latency |
|-----------|---------|
| Observation building | ~10ms |
| LLM call (gpt-5-mini) | ~500-1000ms |
| Movement resolution | ~5ms |
| Logging | ~10ms |
| **Total per turn** | **~1-2 seconds** |

### Scaling

- 5 agents, 100 turns: ~100-200 seconds (1.5-3 minutes)
- 50-turn run: ~50-100 seconds
- Memory: O(grid + agents × turns) ~50MB typical

### Concurrency Limits

- Semaphore size: 4-6 default
- Azure tpm: 2.7M tokens/minute
- At ~200 tokens/call: ~13,500 calls/min possible

---

## Key Abstractions & Design Patterns

### 1. Schema Validation (Strict Output)

```python
# In LlmPolicy.decide_async():
decision, _, _ = await unified.run(
    messages,
    model=model_id,
    output_schema=build_decision_model(strategy),  # Pydantic enforces
    max_spatial_retries=3,  # Retry on schema violation
)
```

Benefits:
- No hallucinated fields
- Type-safe downstream
- Easy to extend for new communication strategies

### 2. Streaming Outputs (Memory Efficient)

```python
# JSONL format (one JSON per line):
{"turn": 0, "agent_id": "a1", "prompt": "...", "observation": {...}, "decision": {...}}
{"turn": 0, "agents": {"a1": {...}, "a2": {...}}}
```

Benefits:
- Can visualize partial episodes while running
- No buffering needed
- Incremental analysis support

### 3. Provider Pooling (Fault Tolerance)

```python
ModelPool
├── Multiple deployments per model key
├── Round-robin selection
├── Failure tracking (60s blacklist)
└── Circuit breaker (3 failures → 90s cooldown)
```

Benefits:
- Automatic failover
- Load balancing
- Resilience to transient failures

### 4. Agent Maps with Overlays (Efficient Rendering)

```python
AgentMap._base          # Walls, goal, free, unknown (persistent)
AgentMap.visited        # All explored cells (personal)
AgentMap.recent         # Last 3 positions (personal)
AgentMap.last_collision # Most recent blocked cell (personal)

Rendering: @ > digit > ! > * > ~ > . (priority)
```

Benefits:
- Efficient merging (only base tiles)
- Clear distinction (base vs. overlays)
- Supports multiple sharing modes

---

## What's Missing (Intentional)

1. **Communication Protocols** (reserved for future work)
   - Freeform natural language (partial scaffolding exists)
   - Structured INTENT/REQUEST (partially investigated)
   - Code exists for parsing but not in prompts yet

2. **Checkpointing** (EpisodeCheckpoint stub only)
   - Was removed during refactoring
   - Can be re-added if needed for very long runs

3. **Reasoning Model Parameters** (partially supported)
   - `reasoning_effort` (minimal|low|medium|high) passed through
   - `reasoning_verbosity` (low|medium|high) passed through
   - Integration exists but not fully tested

---

## File Location Reference

**Core Simulation:** `/Users/3bn/Documents/My_Repos/agent-talk-2/src/llmgrid/env/`
- grid.py (GridWorld)
- simulate.py (run_episode_async)
- maze_generator.py (MazeGenerator)

**Agents:** `/Users/3bn/Documents/My_Repos/agent-talk-2/src/llmgrid/agent/`
- llm_agent.py (LlmPolicy)
- local_baseline.py (GreedyBaseline)

**LLM:** `/Users/3bn/Documents/My_Repos/agent-talk-2/src/llmgrid/llm_clients/`
- unified_llm.py (Provider pooling + rate limiting)

**Schemas:** `/Users/3bn/Documents/My_Repos/agent-talk-2/src/llmgrid/`
- schema.py (Observation, Decision, etc.)
- prompts.py (System prompt)
- agent_map.py (AgentMap + merging)

**Visualization:** `/Users/3bn/Documents/My_Repos/agent-talk-2/src/llmgrid/vis/`
- gif.py (GifRenderer)

**CLI:** `/Users/3bn/Documents/My_Repos/agent-talk-2/src/llmgrid/cli/`
- poc_two_agents.py (Main entry point)
- render_gif.py (GIF rendering)

**Experiments:** `/Users/3bn/Documents/My_Repos/agent-talk-2/experiments/`
- README.md (Master index)
- cross_seed_baseline_20251112T143355Z/ (45-run study)

---

## Starting Points for New Contributors

### To Understand the Core Loop

1. Read `src/llmgrid/env/simulate.py` (~450 LOC) - the main orchestrator
2. Understand `GridWorld.resolve_moves()` in `env/grid.py` - collision logic
3. Check `agent/llm_agent.py` - how decisions are made

### To Add Communication

1. Extend `schema.py` Decision model with message fields
2. Update `prompts.py` system prompt with communication instructions
3. Modify `LlmPolicy.decide_async()` to extract/parse messages
4. Add message handling to `simulate.py` turn loop

### To Visualize Runs

1. Generate episode.json (automatic at end of run)
2. Run `python -m llmgrid.cli.render_gif episode.json --out result.gif`
3. Optional: `--gradient` for goal-distance coloring

### To Run Experiments

```bash
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model gpt-5-mini \
  --maze-preset long_corridor \
  --agents 5 \
  --turns 100 \
  --emit-config config.yaml
```

---

## Summary

**LLM Grid Agents** is a well-structured research framework balancing empirical rigor with implementation clarity. The architecture cleanly separates concerns (simulation, LLM integration, visualization) through Pydantic schemas. The map-sharing mechanism provides a flexible foundation for coordination research, and the streaming pipeline enables real-time monitoring.

The codebase is production-ready for existing use cases (agent navigation, map sharing, LLM integration) and provides clear extension points for future work (communication protocols, advanced reasoning, multi-agent planning).

Key strength: The combination of deterministic simulation + schema-enforced I/O + detailed logging makes this an excellent platform for reproducible multi-agent research.

