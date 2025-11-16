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
    transcript: Optional[Path] = typer.Option(None, "--transcript", help="Path to transcript.jsonl (auto-detected if in same dir)."),
    comms_panel: bool = typer.Option(True, "--comms-panel/--no-comms-panel", help="Show communications panel at bottom."),
    comms_height: int = typer.Option(200, "--comms-height", help="Height of communications panel in pixels."),
    font_size: int = typer.Option(28, "--font-size", help="Font size for text in panels."),
):
    """Render an EpisodeLog JSON into an annotated animated GIF with communications timeline."""

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

    # Auto-detect transcript.jsonl in same directory as episode.json
    transcript_path: Optional[Path] = transcript
    if transcript_path is None and comms_panel:
        auto_transcript = episode.parent / "transcript.jsonl"
        if auto_transcript.exists():
            transcript_path = auto_transcript
            typer.secho(f"Auto-detected transcript: {auto_transcript}", fg=typer.colors.BLUE)

    # Auto-detect model from metrics.json in same directory
    model_name: Optional[str] = None
    metrics_path = episode.parent / "metrics.json"
    if metrics_path.exists():
        try:
            metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
            model_name = metrics_data.get("model")
            if model_name:
                typer.secho(f"Detected model: {model_name}", fg=typer.colors.BLUE)
        except Exception:
            pass  # If metrics can't be read, just skip

    options = RenderOptions(
        cell_size=cell_size,
        fps=fps,
        show_gradient=gradient,
        show_auras=not no_auras,
        show_gridlines=not no_grid,
        show_comms_panel=comms_panel,
        comms_panel_height=comms_height,
        font_size=font_size,
        show_legend=not comms_panel,  # Use legacy legend if comms panel disabled
    )

    renderer = GifRenderer(ep, options, transcript_path=transcript_path, model_name=model_name)
    frames = renderer.render_frames()
    out.parent.mkdir(parents=True, exist_ok=True)
    renderer.save_gif(frames, str(out))

    msg_count = sum(len(msgs) for msgs in renderer.messages_by_turn.values())
    typer.secho(f"Wrote {out} with {len(frames)} frames at {fps} fps ({msg_count} messages)", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
