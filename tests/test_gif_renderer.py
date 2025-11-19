import json
from pathlib import Path

from llmgrid.vis.gif import GifRenderer


def test_gif_renderer_minimal_tmp(tmp_path):
    # minimal 2x2 world, one agent reaches goal at t=1
    episode = {
        "meta": {
            "grid_size": {"width": 2, "height": 2},
            "goal": {"x": 1, "y": 1},
            "walls": [],
        },
        "frames": [
            {
                "t": 0,
                "positions": {"a1": {"x": 0, "y": 0}},
                "goal_hits": [],
                "finished": [],
            },
            {
                "t": 1,
                "positions": {"a1": {"x": 1, "y": 1}},
                "goal_hits": ["a1"],
                "finished": ["a1"],
            },
        ],
    }
    ep = tmp_path / "episode.json"
    ep.write_text(json.dumps(episode))
    out = tmp_path / "out.gif"
    renderer = GifRenderer()
    renderer.render(ep, out)
    assert out.exists() and out.stat().st_size > 0
