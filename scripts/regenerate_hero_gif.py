#!/usr/bin/env python3
"""
Regenerate hero_freeform.gif with communications panel (as in original slides)
"""
import json
from pathlib import Path
from llmgrid.logging.episode_log import EpisodeLog
from llmgrid.vis.gif import GifRenderer, RenderOptions

ROOT = Path(__file__).resolve().parents[1]
SLIDES_ASSETS = ROOT / "slides" / "decks" / "agent-talk" / "assets"

def main() -> None:
    # Use the freeform run from the communication experiments
    run_dir = ROOT / "experiments" / "cross_seed_baseline_20251112T143355Z" / "runs"

    # Find a freeform run
    freeform_runs = list(run_dir.glob("*freeform*"))
    if not freeform_runs:
        print("No freeform runs found")
        return

    run_path = freeform_runs[0] / "results"
    episode_path = run_path / "episode.json"
    transcript_path = run_path / "transcript.jsonl"
    metrics_path = run_path / "metrics.json"

    if not episode_path.exists():
        print(f"Episode not found: {episode_path}")
        return

    # Load episode
    ep = EpisodeLog.model_validate_json(episode_path.read_text(encoding="utf-8"))

    # Get model name from metrics
    model_name = "gpt-5-mini"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            model_name = metrics.get("model", model_name)
        except Exception:
            pass

    # Render with communications panel (original slides style)
    opts = RenderOptions(
        cell_size=40,
        fps=6,
        font_size=22,  # Readable font for comms panel
        show_gradient=True,
        show_gridlines=True,
        show_auras=True,
        show_comms_panel=True,  # THIS IS THE KEY FEATURE
        comms_panel_height=200,  # Bottom panel for messages
        show_legend=False
    )

    out_path = SLIDES_ASSETS / "hero_freeform.gif"
    renderer = GifRenderer(ep, opts, transcript_path=transcript_path, model_name=model_name)
    frames = renderer.render_frames()

    # Take first 100 frames
    frames = frames[:100]

    renderer.save_gif(frames, str(out_path))
    print(f"Wrote {out_path} with {len(frames)} frames (with comms panel)")

if __name__ == "__main__":
    main()
