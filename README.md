# LLM Grid Agents

This repository explores populations of LLM-driven agents that navigate fixed grid worlds with local observability, range-limited communication, and short-lived artifacts. The codebase is structured for repeatable experiments, strict JSON I/O via Pydantic models, and OpenRouter/Azure-based inference.

## Key Results

**Cross-Seed Baseline Study (45 runs, Nov 2025):**
- 🥇 Freeform communication: **62.7% success**
- 🥈 No communication: **57.3% success**
- 🥉 Structured INTENT/REQUEST: **56.0% success**

**Finding:** Freeform natural language communication generalizes better than structured protocols. Structured messages provide minimal benefit over no communication.

📊 **[See full results →](./experiments/cross_seed_baseline_20251112T143355Z/)**

---

## Quick Start

1. Install [uv](https://docs.astral.sh/uv/)
2. Populate `~/.env` with API keys:
   - OpenRouter: `OPENROUTER_API_KEY`
   - Azure: `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`
3. Sync dependencies: `uv sync`

### Minimal Test

Quick smoke test to verify setup:

```bash
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model openrouter:openai/gpt-5-nano \
  --turns 5
```

### Full Experiment Run

For reproducible experiments with logging:

```bash
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model openrouter:openai/gpt-5-nano \
  --maze-preset long_corridor \
  --agents 2 \
  --turns 120 \
  --log-prompts \
  --log-movements \
  --emit-config experiments/my-run-$(date +%s)/config.yaml
```

**Important:** `--log-prompts` and `--log-movements` require `--emit-config`. The CLI fails fast if you forget.

Azure users: use `--model azure:gpt-5-mini` (or your deployment name).

## Experiments & Results

All experiment documentation and results are in [`experiments/`](./experiments/):

- **[experiments/README.md](./experiments/README.md)** - Master experiment index
- **[cross_seed_baseline_20251112T143355Z/](./experiments/cross_seed_baseline_20251112T143355Z/)** - Final communication strategy study (45 runs)

### Visualising Runs

Render episode JSON to annotated GIF:

```bash
PYTHONPATH=src uv run python -m llmgrid.cli.render_gif \
  experiments/.../results/episode.json \
  --out experiments/.../results/rollout.gif \
  --cell-size 40 --fps 6
```

### Observation Schema

Each LLM observation includes a `history` array summarising up to the last five turns (action label, optional comment, sent message, and received message summaries). Use it to keep prompts grounded without reconstructing full transcripts.

## Full Documentation

- **Experiment workflows, tmux setup, performance notes:** See `experiments/README.md`
- **Curated maze presets:** Six hand-picked mazes with deterministic seeds (default: `long_corridor`)
- **Current work:** Multi-agent bearing-mode navigation under partial observability
