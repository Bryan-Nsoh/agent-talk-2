# Agent Talk CLI Reference

Complete documentation of all CLI tools, API surfaces, and usage patterns.

## Quick Reference: Model Specification

### CRITICAL: How to Specify Models

**DO THIS:**
```bash
--model gpt-5-mini
--model gpt-4.1-mini
--model claude-sonnet
```

**NOT THIS:**
```bash
--model azure:gpt-5-mini  # Will hang! Uses wrong mode (agent vs sdk)
```

**Why:** Using `provider:model` syntax bypasses pool configuration lookup and creates a minimal model entry without the correct `mode` field. This causes reasoning models to use the wrong API path.

**Exception:** You can use `provider:model` syntax ONLY if:
1. The model exists in `models.yaml` model_pools
2. You're intentionally overriding provider selection

### Reasoning Models (gpt-5, o1, o3)

These models **reject** the `temperature` parameter. The code now handles this automatically, but be aware:
- gpt-5-mini, gpt-5, gpt-5-pro: temperature omitted
- o1-mini, o1-preview, o3-mini: temperature omitted
- All others: temperature included

### LLM Client Parity (2025-11-19)

- `src/llmgrid/llm_clients/unified_llm.py` is pinned to commit `f4e0863` (the version with async context manager connection pool fix + reasoning-model temperature guard). Do not regress to `a558b8c` or older; those versions leak connection pools and cause indefinite hangs.
- Critical fix: All AsyncOpenAI clients MUST use `async with` context managers to properly close connection pools. Without this, requests hang indefinitely waiting for pool resources.
- When diagnosing inference issues, compare behavior against `f4e0863` before editing the client. Local tweaks (extra logs, speculative fixes, etc.) must live on throwaway branches so the shared branch stays identical to the proven version.
- If you need new instrumentation, add it behind an environment flag or use external tracing so we can revert cleanly after the incident.

## CLI Tools

### run_preset - Production Episode Runner (RECOMMENDED)

**Purpose:** Run episodes with predefined presets (long_corridor, etc.). This is the **actively maintained** CLI that works with the current codebase.

**Location:** `src/llmgrid/cli/run_preset.py`

**Basic Usage:**
```bash
PYTHONPATH=src uv run python -m llmgrid.cli.run_preset \
  --preset long_corridor \
  --turns 100 \
  --model gpt-5-mini \
  --map-sharing radio_sync \
  --seed 13 \
  --episode-json results/episode.json
```

**Required Parameters:**
- `--model MODEL`: Model pool key (e.g., `gpt-5-mini`, `gpt-4.1-mini`) - NOT provider:model syntax
- `--episode-json PATH`: Output path for episode visualization data

**Optional Parameters:**
- `--preset NAME`: Preset name (default: `long_corridor`)
- `--turns N`: Turn budget (default: 100)
- `--visibility N`: Visibility radius (default: 2)
- `--radio-range N`: Radio range for map sharing (default: 2)
- `--map-sharing MODE`: `none`|`radio_sync`|`global` (default: `radio_sync`)
- `--seed N`: Random seed (default: 13)
- `--dry-run`: Use heuristic baseline instead of LLM
- `--concurrency-start N`: Initial concurrent LLM calls (default: max(6, agent_count))
- `--concurrency-max N`: Maximum concurrent LLM calls (default: agent_count)

**Available Presets:**
- `long_corridor`: 30x10 corridor network with 5 agents (seed 606)
  - Width: 30, Height: 10
  - Goal: (28, 1)
  - Agents: a1-a5 with fixed starts
  - Maze file: `experiments/presets/batch/long_corridor_seed606.txt`

**Agent Count:**
Agent count is **determined by the preset**, not a CLI flag. The `long_corridor` preset uses 5 agents.

**Output Files:**
- `episode.json`: Complete episode visualization data (written at completion)
- `episode_stream.jsonl`: Per-turn movement frames (streamed during run, same directory as episode.json)
- `transcript.jsonl`: LLM prompts/responses (streamed during run, same directory as episode.json)
- `metrics.json`: Episode summary (printed to stdout as JSON)

**Example - Production Run:**
```bash
RUN_DIR="experiments/my_exp_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR/results"

PYTHONPATH=src uv run python -m llmgrid.cli.run_preset \
  --preset long_corridor \
  --turns 100 \
  --model gpt-5-mini \
  --map-sharing global \
  --seed 42 \
  --episode-json "$RUN_DIR/results/episode.json" \
  > "$RUN_DIR/results/metrics.json" 2>&1
```

**Example - Quick Dry-Run Test:**
```bash
PYTHONPATH=src uv run python -m llmgrid.cli.run_preset \
  --preset long_corridor \
  --turns 5 \
  --model gpt-5-mini \
  --dry-run \
  --episode-json /tmp/test/episode.json
```

---

### poc_two_agents - Legacy CLI (DEPRECATED)

**Status:** ⚠️ **DO NOT USE** - This CLI is broken and unmaintained.

**Problem:** The CLI passes ~15 kwargs (`resume`, `checkpoint_interval`, `bearing_bias_*`, etc.) that were removed from `run_episode_async()` signature in commit a558b8c (Nov 18, 2025). It will fail with `TypeError: unexpected keyword argument`.

**Migration:** Use `run_preset` instead. If you need custom mazes or advanced configuration, edit the `PRESETS` dict in `run_preset.py`.

**Location:** `src/llmgrid/cli/poc_two_agents.py`

**Required Parameters:**
- `--model MODEL`: Model key from models.yaml (e.g., gpt-5-mini, gpt-4.1-mini)
  - CRITICAL: Use model keys (NOT provider:model syntax) to avoid hanging

**Basic Usage:**
```bash
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model gpt-5-mini \
  --maze-preset micro_blocked_tunnel_small \
  --turns 50 \
  --agents 2 \
  --emit-config experiments/my-run/config.yaml \
  --episode-json experiments/my-run/results/episode.json \
  --transcript-jsonl experiments/my-run/results/transcript.jsonl
```

**Key Options:**
- **Grid:** `--width N`, `--height N`, `--visibility N`, `--radio-range N`, `--turns N`, `--seed N`
- **Agents:** `--agents N` (1-8), `--comm-strategy none|structured|freeform`, `--map-sharing none|radio_sync|global`
- **Maze:** `--maze-preset PRESET` (see below for 20+ presets), `--maze-style maze|random|manual`
- **Checkpointing:** `--checkpoint-json PATH`, `--checkpoint-interval N`, `--resume-from PATH`
- **Output:** `--emit-config PATH`, `--episode-json PATH`, `--transcript-jsonl PATH`, `--log-movements`

**Available Maze Presets:**
- `long_corridor` (30x10): Wide horizontal corridors with loops
- `micro_blocked_tunnel_small` (10x7): Blocked central tunnel forcing detour
- `choke_points_comm_test` (24x14): Three vertical choke columns requiring coordination
- `maze_tight` (21x13): Classic single-path maze
- `maze_loops` (21x13): Maze with many alternate loops
- Plus 15+ more presets (see Maze Generation section below)

**With Checkpointing:**
```bash
RUN_DIR="experiments/long-run-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR/results"

PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model gpt-5-mini \
  --maze-preset long_corridor \
  --turns 200 \
  --checkpoint-json "$RUN_DIR/results/checkpoint.json" \
  --checkpoint-interval 5 \
  --emit-config "$RUN_DIR/config.yaml"
```

**Resume from Checkpoint:**
```bash
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model gpt-5-mini \
  --resume-from experiments/long-run-20251119T120000Z/results/checkpoint.json
```

### Complete Parameter Reference

**80+ parameters organized by category:**

**Grid Configuration (11 parameters):**
- `--width N`, `--height N`: Grid dimensions (default: 12x12)
- `--visibility N`: Visibility radius (default: 1)
- `--radio-range N`: Radio range for map sharing (default: 2)
- `--turns N`: Turn budget (default: 120)
- `--seed N`: Random seed (default: 13)
- `--no-obstacles`: Start with empty grid
- `--obstacle-density F`: Fraction of cells to fill (0.0-1.0)
- `--obstacle-count N`: Absolute number of obstacles
- `--obstacle-seed N`: Seed for obstacle placement

**Agent Configuration (4 parameters):**
- `--agents N`: Number of agents (1-8, default: 2)
- `--comm-strategy`: none|structured|freeform (default: none)
- `--map-sharing`: none|radio_sync|global (default: none)
- `--history-limit N`: Prior turns in history (1-20, default: 5)

**Maze Generation (4 parameters):**
- `--maze-preset NAME`: Select from 17 presets or 'none'
- `--maze-style`: maze|random|manual
- `--maze-extra-connection F`: Loop probability for maze style (default: 0.1)
- Available presets: long_corridor, micro_blocked_tunnel_small, choke_points_comm_test, maze_tight, maze_loops, and 12 more

**Bearing Sensor Noise (5 parameters):**
- `--bearing-bias-seed N`: Enable Gold Drift with seed
- `--bearing-bias-p F`: Baseline rotation probability (default: 0.0)
- `--bearing-bias-wall-bonus F`: Additional probability near walls (default: 0.0)
- `--bearing-flip-p F`: Random flip probability (default: 0.0)
- `--bearing-drop-p F`: Sensor dropout probability (default: 0.0)

**Loop Detection (1 parameter):**
- `--loop-guidance`: passive|active|explore (default: passive)

**Reasoning Model Parameters (3 parameters, for gpt-5/o1/o3 only):**
- `--reasoning-effort`: minimal|low|medium|high (default: minimal)
- `--reasoning-verbosity`: low|medium|high (default: high)
- `--reasoning-include-encrypted`: Include encrypted reasoning trace

**Output and Logging (5 parameters):**
- `--emit-config PATH`: Dump resolved configuration as YAML
- `--transcript-jsonl PATH`: Prompt/response log output
- `--episode-json PATH`: Episode visualization data output
- `--log-movements`: Capture agent locations per turn (default: True)

**Checkpointing (3 parameters):**
- `--checkpoint-json PATH`: Periodic checkpoint file
- `--checkpoint-interval N`: Turns between checkpoints (default: 1)
- `--resume-from PATH`: Resume from existing checkpoint

**Execution (1 parameter):**
- `--dry-run`: Use heuristic baseline instead of LLM

### Checkpoint Resume Workflow

**Checkpoint validation on resume:**
- Model ID must match
- Dry-run mode must match
- Communication strategy must match
- History limit must match
- Loop guidance must match

**What gets restored:**
- Complete world state (walls, positions, goal)
- Turn progress
- Transcript and movement records
- All configuration parameters

**Example:**
```bash
# Initial run with checkpointing
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model gpt-5-mini \
  --maze-preset long_corridor \
  --turns 200 \
  --checkpoint-json experiments/run/checkpoint.json \
  --checkpoint-interval 10 \
  --emit-config experiments/run/config.yaml

# Resume if interrupted (must use same model)
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model gpt-5-mini \
  --resume-from experiments/run/checkpoint.json
```

---

## Model Configuration System

### Overview

The system uses `models.yaml` for centralized configuration of LLM providers, model pools, rate limits, and pricing. Configuration is loaded from:

1. `LLMGRID_MODELS_CONFIG` environment variable
2. `./models.yaml` (repo root)
3. `~/.llmgrid/models.yaml`

### Complete models.yaml Structure

```yaml
# Provider definitions
providers:
  azure:
    type: "azure"
    base_url: "https://your-endpoint.openai.azure.com"
    api_version: "2025-03-01-preview"
    api_key_env: "AZURE_API_KEY"

  openrouter:
    type: "openrouter"
    api_key_env: "OPENROUTER_API_KEY"

# Model pool definitions - CRITICAL SECTION
model_pools:
  gpt-5-mini:
    - provider: "azure"
      deployment: "gpt-5-mini"           # Azure deployment name
      mode: "sdk"                        # CRITICAL: sdk|agent|responses
      reasoning_effort: "minimal"        # For reasoning models
      reasoning_verbosity: "low"
    - provider: "azure"
      deployment: "gpt-5-mini-2"         # Second deployment for load balancing
      mode: "sdk"

  gpt-4.1-mini:
    - provider: "azure"
      deployment: "gpt-4.1-mini"
      mode: "responses"                  # Uses Responses API endpoint

# Rate limiting
limits:
  adaptive:
    enabled: true
    initial_utilization: 0.85            # Start at 85% of quota
    reduction_factor: 0.8                # Drop to 80% on rate limit
    recovery_increment_pct: 0.05         # Recover by 5% per interval

  providers:
    azure:
      concurrency: 12
      rpm: 720
      tpm: 2700000                       # Combined safe budget

  models:
    "azure:gpt-5-mini":                  # provider:deployment format
      rpm: 360
      tpm: 1426500                       # 90% of quota
```

### Model Pool Mechanism

**How it works:**
1. **Pool Key Selection**: Use model pool keys (e.g., `gpt-5-mini`) in CLI
2. **Round-robin Load Balancing**: Multiple deployments rotated automatically
3. **Failure Tracking**: Failed models temporarily blacklisted (60s)
4. **Circuit Breaker**: After 3 provider failures, 90s cooldown

**Why pool keys work but provider:model hangs:**

```python
# Using pool key (CORRECT)
--model gpt-5-mini
# → Looks up model_pools['gpt-5-mini']
# → Gets full config: provider="azure", deployment="gpt-5-mini", mode="sdk"
# → Uses correct SDK path

# Using provider:model syntax (WRONG for most cases)
--model azure:gpt-5-mini
# → Creates fallback with guessed mode
# → May use wrong API path
# → HANGS because reasoning models need special handling
```

### API Modes Explained

**Three modes control how the client communicates:**

#### 1. `mode: "sdk"` (Direct SDK)
- Uses `AsyncOpenAI` client directly
- Best for: Azure OpenAI, standard OpenAI models
- Handles reasoning models (gpt-5, o1, o3) with temperature omission

#### 2. `mode: "agent"` (Pydantic AI Agent)
- Uses `pydantic_ai.Agent` wrapper
- Best for: Non-Azure providers, Anthropic, OpenRouter

#### 3. `mode: "responses"` (Responses API)
- Uses Azure's `/openai/v1/responses` endpoint
- Best for: Azure models with structured output requirements

### Azure OpenAI Configuration

**Required environment variables:**
```bash
# In ~/.env
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com
AZURE_OPENAI_API_KEY=your-key-here
OPENAI_API_VERSION=2025-03-01-preview  # Optional
```

**Important details:**
- `deployment`: Name in Azure portal
- API Version: `2025-03-01-preview` supports reasoning parameters
- Endpoint construction: `{endpoint}/openai/deployments/{deployment}`

### Rate Limiting System

**Three-layer rate limiting:**
1. **Concurrency Semaphore**: Limits parallel requests per provider
2. **RPM Window**: Sliding 60-second window for requests/minute
3. **TPM Budget**: Sliding window for tokens/minute with estimates

**Adaptive behavior:**
```
Initial state: 85% utilization
↓ Rate limit hit
↓ Reduce to 68% (85% × 0.8)
↓ Success for 60s
↓ Recover to 73% (68% + 5%)
↓ Eventually back to 95% max
```

### Reasoning Model Parameters

**Special handling for reasoning models (gpt-5, o1, o3):**

1. **Temperature**: Automatically omitted
2. **Reasoning Effort**: minimal|medium|high (controls depth)
3. **Reasoning Verbosity**: low|medium|high (controls chain-of-thought output)

### Adding New Models

**Steps:**
1. Add to model_pools:
   ```yaml
   model_pools:
     new-model:
       - provider: "azure"
         deployment: "new-model-deployment"
         mode: "sdk"
   ```

2. Add rate limits (optional):
   ```yaml
   limits:
     models:
       "azure:new-model-deployment":
         rpm: 600
         tpm: 1000000
   ```

3. Use in CLI:
   ```bash
   --model new-model  # Uses pool key
   ```

### Troubleshooting

**Model hangs indefinitely:**
- Check: Using pool key `--model gpt-5-mini` (NOT `azure:gpt-5-mini`)
- Check: Model exists in `model_pools`
- Check: `mode` is set correctly for provider

**Rate limit errors:**
- Check: Limits in `models.yaml` below actual quotas
- Check: Adaptive throttling enabled
- Check logs for "reduced utilisation" messages

**Wrong API version:**
- Check: `OPENAI_API_VERSION` environment variable
- Check: `api_version` in providers config

---

## Common Issues and Solutions

### Issue: Model hangs indefinitely

**Symptoms:**
- Process starts, loads config, then no progress
- Log shows "increased utilisation" but nothing else

**Cause:** Using `azure:model-name` syntax instead of just `model-name`

**Solution:**
```bash
# Wrong (hangs)
--model azure:gpt-5-mini

# Right (works)
--model gpt-5-mini
```

### Issue: Temperature parameter error with reasoning models

**Symptoms:**
- API error: "temperature parameter not supported"

**Cause:** gpt-5/o1/o3 models reject temperature

**Solution:** Upgrade to latest code (handles automatically as of 2025-11-19)

---

## Running Long Experiments

### Using tmux for Long-Running Jobs

**Why:** Experiments >2 minutes should use tmux to survive disconnects

**Pattern:**
```bash
SESSION="gpt5mini_exp_$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/${SESSION}.log"
RESULTS_DIR="experiments/my_exp_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p logs "$RESULTS_DIR"

set -a && source ~/.env && set +a

tmux new-session -d -s "$SESSION" \
"set -euo pipefail
export PYTHONUNBUFFERED=1
set -a && source \$HOME/.env && set +a
echo 'run_start model=gpt-5-mini turns=100' >&2
exec uv run python -m llmgrid.cli.run_preset \
  --preset long_corridor \
  --turns 100 \
  --model gpt-5-mini \
  --episode-json '$RESULTS_DIR/episode.json' \
  --transcript-jsonl '$RESULTS_DIR/transcript.jsonl' \
  >> '$LOG' 2>&1
"

echo "Session: $SESSION"
echo "Log: $LOG"
sleep 10 && tail -n 50 "$LOG"
```

**Monitoring:**
```bash
# Check if session is running
tmux has-session -t "$SESSION" && echo "Running" || echo "Finished"

# View log
tail -f logs/SESSION_NAME.log

# Kill session
tmux kill-session -t "$SESSION"
```

---

## Environment Setup

### Required Environment Variables

Store in `~/.env`:
```bash
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_API_KEY=your-key-here
```

**Loading in scripts:**
```bash
set -a && source ~/.env && set +a
```

---

## Experiment Management

### Directory Structure

All experiment work lives under `experiments/` with a three-level hierarchy:

```
experiments/
├── README.md                                    # Master index + project context
├── cross_seed_baseline_20251112T143355Z/
│   ├── README.md                               # Experiment story + run table
│   ├── baselines.json                          # Parameter matrix definition
│   ├── runs/
│   │   └── seed14_structured_20251112T175321Z/
│   │       ├── config.yaml                     # Resolved configuration
│   │       ├── run.log                         # Complete stdout/stderr
│   │       └── results/
│   │           ├── metrics.json                # Episode-level metrics
│   │           ├── episode.json                # Full episode log
│   │           ├── episode_stream.jsonl        # Per-turn movement data
│   │           ├── transcript.jsonl            # LLM prompts/responses
│   │           └── checkpoint.json             # Resume state
│   └── plots/
│       └── *.png                               # Analysis visualizations
```

**Three Levels:**
1. **Master README** - Project context and experiment index
2. **Experiment README** - Complete story of one experiment
3. **Run folders** - Pure artifacts, no narrative

### Naming Conventions

**Experiment folders:** `semantic-name_YYYYMMDDTHHMMSSxZ`

```bash
# Examples:
cross_seed_baseline_20251112T143355Z
micro_blocked_tunnel_small_20251116T000000Z

# Generate timestamp:
date -u +%Y%m%dT%H%M%SZ
```

**Run folders:** `semantic-name_YYYYMMDDTHHMMSSxZ`

```bash
# Examples:
seed14_structured_20251112T175321Z
freeform_kv_light_seed0_20251117T195901Z
```

### Output Files (Per-Run Structure)

**Mandatory:**
- `config.yaml` - Resolved configuration
- `run.log` - Complete stdout/stderr
- `results/metrics.json` - Episode summary

**Standard results:**
- `results/episode.json` - Full episode log with frames
- `results/episode_stream.jsonl` - Per-turn movement data
- `results/transcript.jsonl` - LLM prompts/responses
- `results/checkpoint.json` - Resume state (if checkpointing enabled)

### Batch Runs and Sweeps

**Parameter sweep pattern:** Define baselines in JSON, iterate across seeds.

**Baseline definition** (`baselines.json`):
```json
[
  {"name":"none_passive","comm_strategy":"none","loop_guidance":"passive"},
  {"name":"freeform_kv_light","comm_strategy":"freeform","loop_guidance":"passive"},
  {"name":"structured_intent","comm_strategy":"structured","loop_guidance":"passive"}
]
```

**Sweep script pattern:**
```bash
#!/usr/bin/env bash
MAZE="micro_blocked_tunnel_small"
EXPDIR="experiments/${MAZE}_$(date -u +%Y%m%dT%H%M%SZ)"
SEEDS=(0 1 2 3 4)
MAX_CONCURRENCY=6

run_one() {
  local name="$1" comm="$2" seed="$3"
  run_dir="$EXPDIR/runs/${name}_seed${seed}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$run_dir/results"

  PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
    --model gpt-5-mini \
    --maze-preset "$MAZE" \
    --seed "$seed" \
    --comm-strategy "$comm" \
    --emit-config "$run_dir/config.yaml" \
    > "$run_dir/run.log" 2>&1
}

main() {
  local i=0
  jq -c '.[]' "$EXPDIR/baselines.json" | while read -r cfg; do
    name=$(echo "$cfg" | jq -r '.name')
    comm=$(echo "$cfg" | jq -r '.comm_strategy')
    for seed in "${SEEDS[@]}"; do
      run_one "$name" "$comm" "$seed" &
      ((i++))
      if (( i % MAX_CONCURRENCY == 0 )); then wait; fi
    done
  done
  wait
}

main "$@"
```

### Resumability

**Automatic resume in sweeps:**
```bash
existing=$(find "$EXPDIR/runs" -name "${name}_seed${seed}_*" | sort | tail -n1)
if [ -n "$existing" ] && [ -f "$existing/results/checkpoint.json" ]; then
  turn_next=$(jq -r '.turn_next' "$existing/results/checkpoint.json")
  turns_total=$(jq -r '.turns_total' "$existing/results/checkpoint.json")
  if [ "$turn_next" -ge "$turns_total" ]; then
    echo "SKIP completed: $existing"
    return
  fi
  echo "RESUME at turn $turn_next/$turns_total"
  resume_from="$existing/results/checkpoint.json"
fi
```

### Analysis Workflows

**Load and aggregate metrics:**
```python
from pathlib import Path
import json

def collect_all_data(experiment_root):
    data = defaultdict(list)
    for run_dir in experiment_root.glob("runs/*/"):
        with open(run_dir / "results" / "metrics.json") as f:
            metrics = json.load(f)
        data[metrics['comm_strategy']].append(metrics)
    return data
```

**Common metrics:**
- Success rate: % agents that reached goal
- Message efficiency: messages sent per run
- Collision frequency: average ± std dev
- Token usage: total tokens per run

### Status System

**Two orthogonal indicators:**

**Process state:**
- `running` - In progress
- `complete` - All runs finished
- `failed` - Aborted with errors
- `abandoned` - Discontinued

**Outcome value:**
- `useful` - Produced actionable result
- `not useful` - Failed to answer question
- `inconclusive` - Needs follow-up
- `-` - Not yet determined

### Documentation Standards

**Experiment README template:**
```markdown
# [Experiment Name]

**Last updated:** 2025-11-16T14:45:00Z
**Status:** complete
**Outcome:** useful

## Question
[Single clear question]

## Why This Matters
[Context and decision dependency]

## Setup
- Model: [model]
- Variables: [what changes]
- Held constant: [what stays same]

## Runs
| Run | Started | Status | Notes |
|-----|---------|--------|-------|
| [link](./runs/...) | 2025-11-16 14:30 | complete | Notes |

## Results
[Tables, numbers, references]

## Interpretation
[What we learned]

## Decision
[Action taken]
```

**Update discipline:** Immediately update READMEs when:
1. Creating new experiment/run
2. Run completes/fails
3. Results analyzed
4. Any meaningful change

---

## Visualization Pipeline

### Overview: From Run to GIF

The visualization pipeline converts simulation runs into animated GIFs:

```
[Simulation Run]
     |
     ├─> transcript.jsonl (prompt/response stream; written live next to episode.json)
     ├─> episode_stream.jsonl (frame-by-frame positions; written live next to episode.json)
     ├─> config.yaml (maze metadata + styles; written live next to episode.json)
     └─> episode.json (complete episode + metadata, emitted at run completion)
             |
             v
      [stream_to_episode.py (optional for partials)] --> [python -m llmgrid.vis.gif] --> episode*.gif
```

### Data Formats

#### transcript.jsonl

JSONL file with one entry per agent per turn. Each line contains:

```json
{
  "turn": 0,
  "agent_id": "a1",
  "prompt": "[full prompt text]",
  "observation": {
    "protocol_version": "1.0.0",
    "turn_index": 0,
    "max_turns": 5,
    "grid_size": {"width": 30, "height": 10},
    "self_state": {
      "agent_id": "a1",
      "abs_pos": {"x": 9, "y": 9},
      "orientation": "N"
    },
    "local_patch": {
      "radius": 1,
      "top_left_abs": {"x": 8, "y": 8},
      "rows": ["...", ".A.", "###"]
    },
    "neighbors_in_view": [],
    "artifacts_in_view": [],
    "inbox": [],
    "adjacent": [
      {"dir": "N", "state": "FREE"},
      {"dir": "E", "state": "FREE"}
    ],
    "goal_sensor": {
      "mode": "BEARING",
      "bearing": "E",
      "strength": "FAR"
    }
  },
  "decision": {
    "action": {"kind": "MOVE", "direction": "E"},
    "comment": "OK; advancing east"
  }
}
```

**Use:** Debugging, analyzing decision-making, extracting communication patterns

#### episode_stream.jsonl

JSONL file with one frame per line (fast append during simulation):

```json
{
  "turn": 0,
  "agents": {
    "a1": {
      "x": 9,
      "y": 9,
      "orientation": "N",
      "action": null,
      "status": "ACTIVE"
    },
    "a2": {
      "x": 3,
      "y": 3,
      "orientation": "S",
      "action": null,
      "status": "ACTIVE"
    }
  }
}
```

**Use:** Convert to episode.json for partial/in-progress run visualization (via `stream_to_episode.py`)

#### episode.json

Complete episode file with metadata and all frames:

```json
{
  "meta": {
    "grid_size": {"width": 30, "height": 10},
    "goal": {"x": 28, "y": 1},
    "walls": [
      {"x": 0, "y": 0},
      {"x": 1, "y": 0}
    ],
    "view": {"kind": "square", "radius": 2},
    "gradient_mode": "bfs",
    "title": "Run description",
    "agent_styles": [
      {"agent_id": "a1", "color_hex": "#1f77b4"},
      {"agent_id": "a2", "color_hex": "#d62728"}
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
        }
      ]
    }
  ]
}
```

**Use:** Direct input to GIF renderer

---

### 3. stream_to_episode - Convert Streaming Format to Episode

**Purpose:** Convert episode_stream.jsonl (partial runs, in-progress) to episode.json for visualization

**Location:** `src/llmgrid/cli/stream_to_episode.py`

**Basic Usage:**
```bash
uv run python -m llmgrid.cli.stream_to_episode \
  experiments/my-run/results/episode_stream.jsonl \
  experiments/my-run/config.yaml \
  --out experiments/my-run/results/episode.json
```

**Required Arguments:**
- `stream`: Path to episode_stream.jsonl
- `config`: Path to config.yaml (for maze layout and goal)

**Optional Parameters:**
- `--out PATH`: Output path (required)
- `--max-turns N`: Limit to first N turns (for partial visualization)

**What It Does:**
1. Reads config.yaml to get maze layout, walls, goal
2. Parses episode_stream.jsonl frames
3. Converts agent dict format to list format
4. Generates complete episode.json with metadata

**When to Use:**
- Visualizing in-progress runs (before completion)
- Creating GIFs from partial episodes
- Debugging early-stage behavior without waiting for full run

**Example - Visualize First 20 Turns:**
```bash
uv run python -m llmgrid.cli.stream_to_episode \
  experiments/long-run/results/episode_stream.jsonl \
  experiments/long-run/config.yaml \
  --out /tmp/partial_episode.json \
  --max-turns 20
```

**Output:**
- episode.json with complete metadata and specified frames
- Title auto-generated: "Partial run (turns 0-N)"

---

### 4. render_gif - Generate Animated Visualizations

**Purpose:** Render episode.json into animated GIF with Among Us sprites. Use the legacy entry point (`python -m llmgrid.vis.gif`) for guaranteed compatibility; the CLI wrapper is just a convenience layer around it.

**Basic Usage:**
```bash
PYTHONPATH=src uv run python -m llmgrid.vis.gif \
  --episode experiments/my-run/results/episode.json \
  --out experiments/my-run/results/episode.gif \
  --cell-size 32 \
  --fps 6
```

**Partial Example (render turns 0–20 while a run is still in progress):**
```bash
PYTHONPATH=src uv run python -m llmgrid.cli.stream_to_episode \
  experiments/my-run/results/episode_stream.jsonl \
  experiments/my-run/results/config.yaml \
  --out experiments/my-run/results/episode_partial_turn20.json \
  --max-turns 20

PYTHONPATH=src uv run python -m llmgrid.vis.gif \
  --episode experiments/my-run/results/episode_partial_turn20.json \
  --out experiments/my-run/results/episode_partial_turn20.gif
```

**Current Renderer Features (gif.py):**
- Among Us sprites with directional movement (N/E/S/W)
- Per-agent color coding with palette swap
- Visibility auras (3x3 cells around active agents)
- Goal gradient overlay (BFS distance from goal)
- Wall rendering
- Goal highlighting with pulse on reach
- Legend panel (turn number, model, colors)
- Finished agent graying

**Examples:**

**Basic GIF (auto-detect transcript):**
```bash
uv run python -m llmgrid.cli.render_gif \
  experiments/my-run/results/episode.json \
  --out experiments/my-run/results/episode.gif
```

**High-res with gradient:**
```bash
uv run python -m llmgrid.cli.render_gif \
  experiments/my-run/results/episode.json \
  --out slides/figures/my_run.gif \
  --cell-size 40 \
  --fps 8 \
  --gradient \
  --font-size 32
```

**Minimal (no grid, no gradient):**
```bash
uv run python -m llmgrid.cli.render_gif \
  experiments/my-run/results/episode.json \
  --out docs/figures/clean.gif \
  --no-grid \
  --title "Clean visualization"
```

**Auto-Detection Behavior:**
- If transcript.jsonl exists in same directory as episode.json, automatically loads it
- If metrics.json exists in same directory, extracts model name for legend
- Prints detection messages in blue

**Output:**
- Animated GIF with specified options
- Success message: "Wrote {path} with {N} frames at {fps} fps ({M} messages)"

---

### 5. render_egocentric - Agent-Specific View Rendering

**Purpose:** Render a single agent's egocentric view from a specific turn

**Location:** `src/llmgrid/cli/render_egocentric.py`

**Status:** DEPRECATED. The egocentric rendering module has been removed. This CLI exists but the underlying renderer (/Users/3bn/Documents/My_Repos/agent-talk-2/src/llmgrid/vis/egocentric.py) is now a 2-line stub.

**Historical Usage (no longer functional):**
```bash
uv run python -m llmgrid.cli.render_egocentric \
  experiments/my-run/results/transcript.jsonl \
  --out /tmp/agent_view.png \
  --turn 5 \
  --agent a1 \
  --font-size 18 \
  --cell-size 40
```

**Use unified GIF renderer instead** for all visualization needs.

---

### Visualization Workflow Examples

**Complete workflow from run to GIF:**
```bash
# 1. Run simulation
RUN_DIR="experiments/test_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR/results"

PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model gpt-5-mini \
  --maze-preset long_corridor \
  --turns 50 \
  --emit-config "$RUN_DIR/config.yaml" \
  --episode-json "$RUN_DIR/results/episode.json" \
  --transcript-jsonl "$RUN_DIR/results/transcript.jsonl"

# 2. Generate GIF (auto-detects transcript)
uv run python -m llmgrid.cli.render_gif \
  "$RUN_DIR/results/episode.json" \
  --out "$RUN_DIR/results/episode.gif" \
  --gradient
```

**Visualize in-progress run:**
```bash
# While run is still going, visualize first 20 turns
uv run python -m llmgrid.cli.stream_to_episode \
  "$RUN_DIR/results/episode_stream.jsonl" \
  "$RUN_DIR/config.yaml" \
  --out /tmp/partial.json \
  --max-turns 20

uv run python -m llmgrid.cli.render_gif \
  /tmp/partial.json \
  --out /tmp/partial.gif
```

**Batch GIF generation:**
```bash
# Generate GIFs for all runs in an experiment
for run_dir in experiments/my_exp_*/runs/*/; do
  episode="$run_dir/results/episode.json"
  if [ -f "$episode" ]; then
    uv run python -m llmgrid.cli.render_gif \
      "$episode" \
      --out "$run_dir/results/episode.gif" \
      --gradient \
      --fps 8
  fi
done
```

---

### 6. generate_maze - Custom Maze Generator

**Purpose:** Generate custom maze layouts using algorithmic generation

**Location:** `src/llmgrid/cli/generate_maze.py`

**Basic Usage:**
```bash
# Generate a single maze
uv run llmgrid-generate-maze --width 20 --height 12 --seed 42 --extra-connection 0.2

# Preview multiple variations
uv run llmgrid-generate-maze --width 15 --height 10 --seed 100 --samples 5
```

**Parameters:**
- `--width N`: Maze width in columns (default: 15)
- `--height N`: Maze height in rows (default: 15)
- `--seed N`: Random seed for deterministic generation (default: 0)
- `--extra-connection F`: Probability (0.0-1.0) of carving additional passages (default: 0.1)
  - 0.0 = pure single-path maze
  - Higher values add loops and alternate routes
- `--samples N`: Number of mazes to generate with incrementing seeds (default: 1)

**Algorithm:** Depth-first carving with guaranteed connectivity between required cells. Uses `extra_connection_prob` to add loops after initial maze generation.

**Output Format:** ASCII representation where:
- `#` = wall/obstacle
- `.` = open/passable cell

**ASCII Map Format for Custom Mazes:**

Characters:
- `#` = wall (impassable)
- `.` = open cell (passable)
- `A`, `B`, etc. = optional agent starting positions
- `G` = optional goal position

Requirements:
- Plain text .txt file
- Each line = one row (y-coordinate)
- All lines must have identical length
- Top-left is (0, 0)

**Example - Create and use custom maze:**
```bash
# Generate base maze
uv run llmgrid-generate-maze --width 25 --height 15 --seed 999 > mazes/custom.txt

# Edit manually if desired, then use:
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model gpt-5-mini \
  --maze-preset none \
  --maze-style manual \
  --width 25 \
  --height 15 \
  # Note: Would need to add manual_ascii_path parameter or create preset
```
