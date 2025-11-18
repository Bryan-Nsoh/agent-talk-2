# Slide Decks

This folder is reserved for slide decks we’ll generate about the LLM Grid Agents project.

## How to build slides here
- Use the vendored slide-maker tools in `slides/slide-maker/` (JavaScript + Python scripts).
- Keep each deck in its own subfolder under `slides/decks/` (e.g., `cross-seed-baseline/`).
- Check in source assets (markdown/HTML/templates, data snippets); generated PPTX/PDF can live in the same subfolder but should be regenerable.
- If a deck depends on experiment data, note the commit hash and experiment path in a short `NOTES.md` inside that deck.

## Suggested layout for a new deck
- `slides/decks/<deck-name>/src/` – source markdown/HTML or prompts
- `slides/decks/<deck-name>/assets/` – images/plots (e.g., from `experiments/cross_seed_baseline_20251112T143355Z/plots/`)
- `slides/decks/<deck-name>/build/` – generated PPTX/PDF/HTML outputs

## Quick start (slide-maker)
- JS: `node slides/slide-maker/scripts/html2pptx.js input.html --out build/deck.pptx`
- Python utils: see `slides/slide-maker/scripts/*.py` for packing/rearranging/thumbnail helpers.

Feel free to create the first deck folder and drop your sources there.
