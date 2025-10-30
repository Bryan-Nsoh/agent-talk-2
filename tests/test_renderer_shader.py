from llmgrid.logging.episode_log import (
    AgentState,
    AgentStyle,
    EpisodeLog,
    EpisodeMeta,
    Frame,
    GridSize,
    Position,
    ViewShape,
)
from llmgrid.vis.gif import GifRenderer, RenderOptions


def build_tiny_episode() -> EpisodeLog:
    meta = EpisodeMeta(
        grid_size=GridSize(width=8, height=6),
        goal=Position(x=6, y=2),
        walls=[Position(x=3, y=y) for y in range(1, 5)],
        view=ViewShape(kind="square", radius=1),
        gradient_mode="bfs",
        title="viz",
        agent_styles=[AgentStyle(agent_id="a1", color_hex="#1f77b4")],
    )
    frame = Frame(t=0, agents=[AgentState(agent_id="a1", pos=Position(x=1, y=4))])
    return EpisodeLog(meta=meta, frames=[frame])


def average_rgb(image, rect):
    sub = image.crop(rect).convert("RGB")
    pixels = list(sub.getdata())
    r = sum(px[0] for px in pixels) / len(pixels)
    g = sum(px[1] for px in pixels) / len(pixels)
    b = sum(px[2] for px in pixels) / len(pixels)
    return r, g, b


def test_shader_highlights_goal_cell():
    episode = build_tiny_episode()
    options = RenderOptions(cell_size=20, show_gradient=True, show_auras=False, fps=1)
    renderer = GifRenderer(episode, options)
    frame = renderer.render_frames()[0]

    far_rect = renderer._cell_rect(0, episode.meta.grid_size.height - 1)
    goal_rect = renderer._cell_rect(episode.meta.goal.x, episode.meta.goal.y)

    far_rgb = average_rgb(frame, far_rect)
    goal_rgb = average_rgb(frame, goal_rect)

    assert goal_rgb[2] < far_rgb[2]
    assert goal_rgb[1] <= far_rgb[1]
