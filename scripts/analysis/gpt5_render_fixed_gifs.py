#!/usr/bin/env python3
"""
Render exactly 100-turn GIFs (frames 1..100) for one run per baseline
using the project GifRenderer, so we avoid the initial t=0 preview frame.

Outputs to docs/figures with names:
  - gpt5_structured_100t.gif
  - gpt5_freeform_100t.gif
  - gpt5_none_100t.gif
"""
from __future__ import annotations

import json
from pathlib import Path

from llmgrid.logging.episode_log import EpisodeLog
from llmgrid.vis.gif import GifRenderer, RenderOptions

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "figures"


def render_100t(episode_json: Path, out_gif: Path, *, fps: int = 6, cell: int = 40) -> None:
    ep = EpisodeLog.model_validate_json(episode_json.read_text(encoding="utf-8"))
    # Slice frames: skip t=0, take next 100 frames if available
    frames = ep.frames
    if len(frames) > 101:
        subset = frames[1:101]
    else:
        subset = frames[1: min(101, len(frames))]
    ep.frames = subset
    opts = RenderOptions(cell_size=cell, fps=fps, show_gradient=True, show_auras=True, show_gridlines=True)
    renderer = GifRenderer(ep, opts)
    imgs = renderer.render_frames()
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    renderer.save_gif(imgs, str(out_gif))
    print(f"Wrote {out_gif} with {len(imgs)} frames")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = ROOT / "experiments" / "long_corridor_final_20251110T155342Z" / "runs"
    runs = {
        "structured": base / "structured_run1_20251110T155342Z" / "results" / "episode.json",
        "freeform": base / "freeform_run1_20251110T155343Z" / "results" / "episode.json",
        "none": base / "none_run1_20251110T155345Z" / "results" / "episode.json",
    }
    for key, ep in runs.items():
        if not ep.exists():
            print(f"Missing episode.json for {key}: {ep}")
            continue
        out = FIG_DIR / f"gpt5_{key}_100t.gif"
        render_100t(ep, out)


if __name__ == "__main__":
    main()

