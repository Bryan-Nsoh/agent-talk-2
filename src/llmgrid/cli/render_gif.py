"""CLI for rendering EpisodeLog JSON files into animated GIFs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError

from llmgrid.logging.episode_log import EpisodeLog
from llmgrid.vis.gif import GifRenderer, RenderOptions

app = typer.Typer(add_completion=False)


@app.command()
def main(
    episode: Path = typer.Argument(..., help="Path to episode.json"),
    out: Path = typer.Option(..., "--out", "-o", help="Output GIF path."),
    fps: int = typer.Option(6, "--fps", help="Frames per second in the GIF."),
    cell_size: int = typer.Option(32, "--cell-size", help="Pixel size per grid cell."),
    gradient: bool = typer.Option(False, "--gradient/--no-gradient", help="Enable goal gradient tint."),
    no_auras: bool = typer.Option(False, "--no-auras", help="Disable visibility auras."),
    no_grid: bool = typer.Option(False, "--no-grid", help="Disable grid lines."),
    title: Optional[str] = typer.Option(None, "--title", help="Override episode title."),
):
    """Render an EpisodeLog JSON into an annotated animated GIF."""

    try:
        data = json.loads(episode.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        typer.secho(f"Episode file not found: {episode}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    try:
        ep = EpisodeLog.model_validate(data)
    except ValidationError as exc:
        typer.secho("Episode JSON failed validation", fg=typer.colors.RED)
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    if title:
        ep.meta.title = title

    options = RenderOptions(
        cell_size=cell_size,
        fps=fps,
        show_gradient=gradient,
        show_auras=not no_auras,
        show_gridlines=not no_grid,
    )

    renderer = GifRenderer(ep, options)
    frames = renderer.render_frames()
    out.parent.mkdir(parents=True, exist_ok=True)
    renderer.save_gif(frames, str(out))
    typer.secho(f"Wrote {out} with {len(frames)} frames at {fps} fps", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
