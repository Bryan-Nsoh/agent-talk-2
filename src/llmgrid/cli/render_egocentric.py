"""CLI for rendering agent egocentric views from transcript.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from llmgrid.vis.egocentric import EgocentricRenderOptions, render_agent_view

app = typer.Typer(add_completion=False)


@app.command()
def main(
    transcript: Path = typer.Argument(..., help="Path to transcript.jsonl"),
    out: Path = typer.Option(..., "--out", "-o", help="Output image path (PNG)."),
    turn: int = typer.Option(..., "--turn", "-t", help="Turn number to render."),
    agent: str = typer.Option(..., "--agent", "-a", help="Agent ID (e.g., 'a1', 'a2')."),
    font_size: int = typer.Option(18, "--font-size", help="Font size for text."),
    cell_size: int = typer.Option(40, "--cell-size", help="Cell size for local patch grid."),
):
    """Render an agent's egocentric view from a specific turn in the transcript."""

    if not transcript.exists():
        typer.secho(f"Transcript file not found: {transcript}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Find the observation for the specified agent and turn
    observation = None
    with open(transcript, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('turn') == turn and entry.get('agent_id') == agent:
                observation = entry.get('observation')
                break

    if observation is None:
        typer.secho(f"No observation found for agent={agent} at turn={turn}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    options = EgocentricRenderOptions(
        font_size=font_size,
        cell_size=cell_size,
    )

    map_path, info_path = render_agent_view(observation, out, options)
    typer.secho(f"Rendered egocentric view:", fg=typer.colors.GREEN)
    typer.secho(f"  Map:  {map_path}", fg=typer.colors.BLUE)
    typer.secho(f"  Info: {info_path}", fg=typer.colors.BLUE)
    typer.secho(f"  Agent: {agent}, Turn: {turn}", fg=typer.colors.CYAN)


if __name__ == "__main__":
    app()
