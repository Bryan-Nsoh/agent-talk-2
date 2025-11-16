#!/usr/bin/env python3
"""Export specific turn frames from an episode.json using the existing GifRenderer.

Usage:
  PYTHONPATH=src python slides/decks/agent-talk/scripts/export_frames.py \
    --episode experiments/.../results/episode.json \
    --transcript experiments/.../results/transcript.jsonl \
    --metrics experiments/.../results/metrics.json \
    --turns 21 22 23 24 \
    --out-prefix slides/decks/agent-talk/assets/freeform_t

Writes PNGs like freeform_t21.png ...
"""

import argparse
import json
from pathlib import Path

from llmgrid.cli.render_gif import RenderOptions
from llmgrid.vis.gif import GifRenderer
from llmgrid.logging.episode_log import EpisodeLog

def load_episode(path: Path) -> EpisodeLog:
    data = json.loads(path.read_text())
    return EpisodeLog.model_validate(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--turns", nargs="+", type=int, required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--cell-size", type=int, default=32)
    parser.add_argument("--comms-height", type=int, default=200)
    parser.add_argument("--font-size", type=int, default=26)
    parser.add_argument("--gradient", action="store_true")
    args = parser.parse_args()

    ep = load_episode(args.episode)
    model_name = None
    if args.metrics and args.metrics.exists():
        try:
            metrics = json.loads(args.metrics.read_text())
            model_name = metrics.get("model")
        except Exception:
            pass

    options = RenderOptions(
        cell_size=args.cell_size,
        show_gradient=args.gradient,
        show_auras=True,
        show_gridlines=True,
        show_comms_panel=True,
        comms_panel_height=args.comms_height,
        font_size=args.font_size,
        show_legend=False,
    )

    renderer = GifRenderer(ep, options, transcript_path=args.transcript, model_name=model_name)
    frames = renderer.render_frames()
    turn_to_frame = {frame.t: img for frame, img in zip(ep.frames, frames)}

    for t in args.turns:
        if t not in turn_to_frame:
            print(f"Turn {t} not found; skipping")
            continue
        out = Path(f"{args.out_prefix}{t}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        turn_to_frame[t].save(out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
