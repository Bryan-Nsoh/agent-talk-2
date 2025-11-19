"""Rich GIF renderer with Among Us sprites for the single-grid observation logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import functools

WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
GRID_LINE = (220, 220, 220, 255)
AURA_ALPHA = 90
GOAL_GOLD = (255, 215, 0, 255)
DEFAULT_AGENT_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#bcbd22",
    "#e377c2",
    "#7f7f7f",
    "#8c564b",
]


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Sprite loading
# ---------------------------------------------------------------------------


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def _load_sprites(base_dir: Path) -> Dict[str, Image.Image]:
    sprites = {}
    for name in ["crewmate_north", "crewmate_south", "crewmate_east", "crewmate_west", "crewmate_idle"]:
        sprites[name] = _load_image(base_dir / f"{name}.png")
    return sprites


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RenderOptions:
    cell_size: int = 32
    border: int = 8
    fps: int = 6
    font_size: int = 16
    show_grid: bool = True
    show_gradient: bool = False
    aura_alpha: int = 90
    goal_pulse_frames: int = 4


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class GifRenderer:
    def __init__(
        self,
        sprites_dir: Optional[Path] = None,
        options: Optional[RenderOptions] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.opts = options or RenderOptions()
        base = sprites_dir or Path("assets/sprites")
        self.sprites = _load_sprites(base)
        self.font = self._load_font(self.opts.font_size)
        self.model_name = model_name or "unknown"
        # built per-render using meta when available
        self.agent_colors: Dict[str, Tuple[int, int, int]] = {}
        self._sprite_cache: Dict[Tuple[str, Tuple[int, int, int], bool], Image.Image] = {}

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        for path in [
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]:
            try:
                return ImageFont.truetype(path, size)  # type: ignore[arg-type]
            except Exception:
                continue
        return ImageFont.load_default()

    def _build_agent_colors(self, meta: dict) -> Dict[str, Tuple[int, int, int]]:
        palette = [hex_to_rgb(c) for c in DEFAULT_AGENT_COLORS]
        styles = meta.get("agent_styles") or []
        if styles:
            colors: Dict[str, Tuple[int, int, int]] = {}
            for idx, st in enumerate(styles):
                aid = st.get("agent_id", f"a{idx+1}")
                hex_col = st.get("color_hex") or DEFAULT_AGENT_COLORS[idx % len(DEFAULT_AGENT_COLORS)]
                colors[aid] = hex_to_rgb(hex_col)
            return colors
        return {f"a{i+1}": palette[i % len(palette)] for i in range(24)}

    # ------------- public API -------------
    def render(self, episode_json: Path, out_path: Path) -> Path:
        data = json.loads(episode_json.read_text())
        if "frames" not in data or "meta" not in data:
            raise ValueError("Episode JSON must have frames and meta.")

        frames = data["frames"]
        meta = data["meta"]
        width = meta["grid_size"]["width"]
        height = meta["grid_size"]["height"]
        goal = (meta["goal"]["x"], meta["goal"]["y"])
        walls = {(p["x"], p["y"]) for p in meta.get("walls", [])}
        self.agent_colors = self._build_agent_colors(meta)

        gradient = None
        if self.opts.show_gradient:
            gradient = compute_gradient(width, height, walls, goal)

        images: List[Image.Image] = []
        last_positions: Dict[str, Tuple[int, int]] = {}
        for frame in frames:
            headings: Dict[str, str] = {}
            positions = frame.get("positions") or {a["agent_id"]: a["pos"] for a in frame.get("agents", [])}
            for aid, pos in positions.items():
                cur = (pos["x"], pos["y"])
                prev = last_positions.get(aid)
                if prev and prev != cur:
                    dx = cur[0] - prev[0]
                    dy = cur[1] - prev[1]
                    if abs(dx) >= abs(dy):
                        headings[aid] = "E" if dx > 0 else "W"
                    else:
                        headings[aid] = "S" if dy > 0 else "N"
            last_positions = {aid: (p["x"], p["y"]) for aid, p in positions.items()}

            img = self._render_frame(frame, width, height, walls, goal, gradient, headings)
            images.append(img)

        duration = int(1000 / max(1, self.opts.fps))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            out_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=False,
        )
        return out_path

    # ------------- frame rendering -------------
    def _render_frame(
        self,
        frame: dict,
        width: int,
        height: int,
        walls: set[Tuple[int, int]],
        goal: Tuple[int, int],
        gradient: Optional[List[List[float]]] = None,
        headings: Optional[Dict[str, str]] = None,
    ) -> Image.Image:
        cs = self.opts.cell_size
        border = self.opts.border
        grid_w = width * cs + border * 2
        grid_h = height * cs + border * 2
        legend_w = 160
        img = Image.new("RGBA", (grid_w + legend_w, grid_h), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")

        # gradient
        if gradient:
            for y in range(height):
                for x in range(width):
                    g = gradient[y][x]
                    alpha = int(80 * g)
                    if alpha == 0:
                        continue
                    self._cell(draw, x, y, cs, border, fill=(255, 215, 0, alpha))

        # grid lines
        if self.opts.show_grid:
            for x in range(width + 1):
                x0 = border + x * cs
                draw.line([(x0, border), (x0, border + height * cs)], fill=(220, 220, 220, 255))
            for y in range(height + 1):
                y0 = border + y * cs
                draw.line([(border, y0), (border + width * cs, y0)], fill=(220, 220, 220, 255))

        # walls
        for (wx, wy) in walls:
            self._cell(draw, wx, wy, cs, border, fill=(60, 60, 60, 255))

        # goal
        self._cell(draw, goal[0], goal[1], cs, border, fill=(255, 215, 0, 255), outline=BLACK, width=2)

        # positions
        positions = frame.get("positions") or {a["agent_id"]: a["pos"] for a in frame.get("agents", [])}
        finished = set(frame.get("finished", []))
        goal_hits = set(frame.get("goal_hits", []))
        t = frame.get("turn", frame.get("t", -1))

        # aura overlay (square radius 1, skip finished)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay, "RGBA")
        for aid, pos in positions.items():
            if aid in finished:
                continue
            base_col = self.agent_colors.get(aid, hex_to_rgb("#808080"))
            tint = (
                int(base_col[0] * 0.7 + WHITE[0] * 0.3),
                int(base_col[1] * 0.7 + WHITE[1] * 0.3),
                int(base_col[2] * 0.7 + WHITE[2] * 0.3),
            )
            aura_col = tint + (AURA_ALPHA,)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    ax, ay = pos["x"] + dx, pos["y"] + dy
                    if 0 <= ax < width and 0 <= ay < height:
                        self._cell(overlay_draw, ax, ay, cs, border, fill=aura_col)

        # composite aura BEFORE sprites (so sprites appear on top)
        img.alpha_composite(overlay)

        for aid, pos in positions.items():
            x, y = pos["x"], pos["y"]
            # sprite
            heading = (headings or {}).get(aid, None)
            base_col = self.agent_colors.get(aid, hex_to_rgb("#808080"))
            sprite = self._sprite_for_heading(heading, base_col, aid in finished)
            top_left = (border + x * cs, border + y * cs)
            img.alpha_composite(sprite.resize((cs, cs)), dest=top_left)

        # goal pulse on hit
        if goal_hits:
            pulse_alpha = max(30, 180 - (t % self.opts.goal_pulse_frames) * 30)
            self._cell(draw, goal[0], goal[1], cs, border, outline=(255, 50, 50, pulse_alpha), width=4)

        # legend
        lx = grid_w + 10
        ly = border
        draw.text((lx, ly), f"Turn {t}", fill=(0, 0, 0, 255), font=self.font)
        ly += 18
        draw.text((lx, ly), f"Model: {self.model_name}", fill=(0, 0, 0, 255), font=self.font)
        ly += 24
        draw.rectangle([lx, ly, lx + 16, ly + 16], fill=(60, 60, 60, 255))
        draw.text((lx + 22, ly), "wall", fill=(0, 0, 0, 255), font=self.font)
        ly += 20
        draw.rectangle([lx, ly, lx + 16, ly + 16], fill=(255, 215, 0, 255))
        draw.text((lx + 22, ly), "goal", fill=(0, 0, 0, 255), font=self.font)
        ly += 20
        draw.rectangle([lx, ly, lx + 16, ly + 16], fill=(80, 140, 255, 200))
        draw.text((lx + 22, ly), "agent", fill=(0, 0, 0, 255), font=self.font)
        ly += 22
        # per-agent color legend (only agents present this frame)
        for aid in sorted(positions.keys()):
            col = self.agent_colors.get(aid, hex_to_rgb("#808080"))
            draw.rectangle([lx, ly, lx + 16, ly + 16], fill=col + (200,))
            draw.text((lx + 22, ly), aid, fill=(0, 0, 0, 255), font=self.font)
            ly += 18

        return img.convert("RGB")

    def _cell(self, draw: ImageDraw.ImageDraw, x: int, y: int, cs: int, border: int, *, fill=None, outline=None, width: int = 1):
        x0 = border + x * cs
        y0 = border + y * cs
        draw.rectangle([x0, y0, x0 + cs, y0 + cs], fill=fill, outline=outline, width=width)

    def _sprite_for_heading(self, heading: Optional[str], color: Tuple[int, int, int], finished: bool) -> Image.Image:
        name = "crewmate_idle"
        if heading == "N":
            name = "crewmate_north"
        elif heading == "S":
            name = "crewmate_south"
        elif heading == "E":
            name = "crewmate_east"
        elif heading == "W":
            name = "crewmate_west"
        return self._tinted_sprite(name, color, finished)

    def _tinted_sprite(self, sprite_name: str, color: Tuple[int, int, int], finished: bool) -> Image.Image:
        key = (sprite_name, color, finished)
        cached = self._sprite_cache.get(key)
        if cached is not None:
            return cached

        base = self.sprites.get(sprite_name, self.sprites["crewmate_idle"])

        # Pixel-perfect palette swap: replace specific red pixels with agent colors
        # Base sprite has these red shades (body_main, body_shadows, backpack)
        body_main = (220, 30, 30, 255)
        body_shadow_1 = (150, 0, 0, 255)
        body_shadow_2 = (40, 0, 0, 255)  # darker patch present in current assets
        backpack = (180, 20, 20, 255)

        if finished:
            # Finished agents get gray
            target_main = (160, 160, 160, 255)
            target_shadow_light = (110, 110, 110, 255)
            target_shadow_dark = (90, 90, 90, 255)
            target_backpack = (135, 135, 135, 255)
        else:
            # Active agents get their assigned color with shading
            target_main = (*color, 255)
            target_shadow_light = tuple(min(255, int(c * 0.65)) for c in color) + (255,)
            target_shadow_dark = tuple(min(255, int(c * 0.45)) for c in color) + (255,)
            target_backpack = tuple(min(255, int(c * 0.85)) for c in color) + (255,)

        palette_map = {
            body_main: target_main,
            body_shadow_1: target_shadow_light,
            body_shadow_2: target_shadow_dark,
            backpack: target_backpack,
        }

        tinted = base.copy()
        pixels = tinted.load()
        width, height = tinted.size

        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a == 0:
                    continue
                current = (r, g, b, a)
                replacement = palette_map.get(current)
                if replacement:
                    pixels[x, y] = replacement

        self._sprite_cache[key] = tinted
        return tinted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render episode JSON to GIF (Among Us sprites).")
    parser.add_argument("--episode", type=Path, required=True, help="Episode JSON with frames/meta.")
    parser.add_argument("--out", type=Path, required=True, help="Output GIF path.")
    parser.add_argument("--model-name", type=str, default="unknown", help="Model name to annotate.")
    args = parser.parse_args()

    renderer = GifRenderer(model_name=args.model_name)
    renderer.render(args.episode, args.out)
    print(f"Saved GIF to {args.out}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Gradient helpers (lightweight, goal-centric)
# ---------------------------------------------------------------------------


def bfs_distance_map(width: int, height: int, goal: Tuple[int, int], walls: set[Tuple[int, int]]) -> List[List[int]]:
    from collections import deque

    dist = [[-1 for _ in range(width)] for _ in range(height)]
    gx, gy = goal
    if not (0 <= gx < width and 0 <= gy < height):
        return dist
    q = deque()
    dist[gy][gx] = 0
    q.append((gx, gy))
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in walls:
                continue
            if dist[ny][nx] != -1:
                continue
            dist[ny][nx] = dist[y][x] + 1
            q.append((nx, ny))
    return dist


def normalize_distance_map(dist_map: List[List[int]]) -> List[List[float]]:
    flat = [d for row in dist_map for d in row if d >= 0]
    if not flat:
        return [[0.0 for _ in row] for row in dist_map]
    m = max(flat)
    if m == 0:
        return [[0.0 for _ in row] for row in dist_map]
    return [[(d / m) if d >= 0 else 0.0 for d in row] for row in dist_map]


def compute_gradient(width: int, height: int, walls: set[Tuple[int, int]], goal: Tuple[int, int]) -> List[List[float]]:
    return normalize_distance_map(bfs_distance_map(width, height, goal, walls))
