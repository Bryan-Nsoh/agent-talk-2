# LLM Grid Agents

This repository explores populations of LLM-driven agents that navigate fixed grid worlds with local observability, range-limited communication, and short-lived artifacts. The codebase is structured for repeatable experiments, strict JSON I/O via Pydantic models, and OpenRouter-based inference using `openai/gpt-5-nano` by default.

## Quick Start

1. Install [uv](https://docs.astral.sh/uv/).
2. Populate `~/.env` with `OPENROUTER_API_KEY` and reload your shell so keys auto-load.
3. Sync dependencies: `uv sync`.
4. Run the proof-of-concept simulation: `PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents --model openrouter:openai/gpt-5-nano`.
   - Add `--log-prompts` to capture the exact prompts/decisions (requires `--emit-config` so the transcript is written to `results/transcript.jsonl`).
   - Episode logs (`results/episode.json`) are collected by default when `--emit-config` is provided. Set `--no-log-movements` to disable.
   - The CLI now defaults to the curated `long_corridor` maze. Switch presets with `--maze-preset <name>` or supply `--maze-preset none` to control width/height/density manually.

## Visualising Runs

Render an `episode.json` trace into an annotated GIF:

```
PYTHONPATH=src uv run python -m llmgrid.cli.render_gif \
  experiments/.../results/episode.json \
  --out experiments/.../results/rollout.gif \
  --cell-size 40 --fps 6
```

Add `--gradient` if you want the goal-distance tint, or `--no-auras` to hide visibility overlays.

See `experiments/README.md` for experiment tracking conventions.
