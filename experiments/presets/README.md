## Maze Preview Assets

**Last updated:** 2025-11-06T19:25:00Z

PNG previews for curated maze presets live here. They are generated via the `maze_generator` utility and correspond to the preset names wired into `llmgrid.cli.poc_two_agents`. Among Us sprite bases (NESW + idle) are sourced from [The Spriters Resource](https://www.spriters-resource.com/pc_computer/amongus/sheet/157129/); recoloring happens at render time.

- `batch/open_sparse_seed101.png`
- `batch/open_dense_seed202.png`
- `batch/maze_tight_seed303.png`
- `batch/maze_loops_seed404.png`
- `batch/mixed_medium_seed505.png`
- `batch/long_corridor_seed606.png`

Regenerate or extend the collection:

```
PYTHONPATH=src uv run python -m llmgrid.cli.generate_maze --width 20 --height 12 --seed 101 --extra-connection 0.05 --samples 1
```

or invoke the batch script in `notebooks/` when we add more presets.
