from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from llmgrid.logging.episode_log import AgentState, EpisodeLog

# Colors
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
GRID_LINE = (220, 220, 220, 255)
GOAL_GOLD = (255, 215, 0, 255)
AURA_ALPHA = 90
GRADIENT_ALPHA = 80
GRADIENT_INTENSITY = 0.45
LEGEND_WIDTH = 160

AGENT_ALPHA = 200

DEFAULT_AGENT_COLORS = {
    "a1": "#1f77b4",
    "a2": "#d62728",
    "a3": "#2ca02c",
    "a4": "#9467bd",
    "a5": "#ff7f0e",
    "a6": "#17becf",
}


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


@dataclass
class RenderOptions:
    cell_size: int = 32
    border: int = 12
    show_gradient: bool = False
    show_auras: bool = True
    show_gridlines: bool = True
    fps: int = 6
    font_size: int = 14
    show_legend: bool = True


class GifRenderer:
    def __init__(self, episode: EpisodeLog, options: Optional[RenderOptions] = None):
        self.episode = episode
        self.opts = options or RenderOptions()
        self.grid_w = episode.meta.grid_size.width
        self.grid_h = episode.meta.grid_size.height
        self.goal = (episode.meta.goal.x, episode.meta.goal.y)
        self.walls = {(p.x, p.y) for p in episode.meta.walls}
        self.agent_colors = self._build_agent_colors()
        self.font = self._load_font(self.opts.font_size)
        self.gradient = self._compute_gradient()
        self.inferred_finished_turn: Dict[str, int] = {}
        self._infer_finished_turns()
        self.base_sprites = self._load_base_sprites()
        self.sprite_cache: Dict[Tuple[str, Tuple[int, int, int]], Image.Image] = {}

    def _build_agent_colors(self) -> Dict[str, Tuple[int, int, int]]:
        colors: Dict[str, Tuple[int, int, int]] = {}
        for style in self.episode.meta.agent_styles:
            colors[style.agent_id] = hex_to_rgb(style.color_hex)
        palette = list(DEFAULT_AGENT_COLORS.values())
        # Collect all unique agent IDs
        agent_ids: List[str] = []
        for frame in self.episode.frames:
            for agent in frame.agents:
                if agent.agent_id not in colors and agent.agent_id not in agent_ids:
                    agent_ids.append(agent.agent_id)
        # Assign colors by agent index, not frame index
        for idx, agent_id in enumerate(agent_ids):
            colors[agent_id] = hex_to_rgb(palette[idx % len(palette)])
        return colors

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("arial.ttf", size)  # type: ignore[no-any-return]
        except Exception:
            return ImageFont.load_default()

    def _compute_gradient(self) -> List[List[float]]:
        if not self.opts.show_gradient:
            return [[0.0 for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        if self.episode.meta.gradient_mode == "manhattan":
            raw = manhattan_distance_map(self.grid_w, self.grid_h, self.goal)
        else:
            raw = bfs_distance_map(self.grid_w, self.grid_h, self.goal, self.walls)
        return normalize_distance_map(raw)

    def _infer_finished_turns(self) -> None:
        goal = self.goal
        for frame in self.episode.frames:
            for agent in frame.agents:
                if agent.agent_id in self.inferred_finished_turn:
                    continue
                status = getattr(agent, "status", None)
                if status == "FINISHED":
                    self.inferred_finished_turn[agent.agent_id] = frame.t
                    continue
                if getattr(agent, "action", None) == "FINISHED":
                    self.inferred_finished_turn[agent.agent_id] = frame.t
                    continue
                if (agent.pos.x, agent.pos.y) == goal:
                    self.inferred_finished_turn[agent.agent_id] = frame.t

    def _agent_status(self, agent: AgentState, frame_turn: int) -> str:
        status = getattr(agent, "status", None)
        if status in ("ACTIVE", "FINISHED"):
            return status
        action = getattr(agent, "action", None)
        if action == "FINISHED":
            return "FINISHED"
        finish_turn = self.inferred_finished_turn.get(agent.agent_id)
        if finish_turn is not None and frame_turn >= finish_turn:
            return "FINISHED"
        return "ACTIVE"

    def render_frames(self) -> List[Image.Image]:
        frames: List[Image.Image] = []
        for frame in self.episode.frames:
            canvas = self._create_canvas()
            draw = ImageDraw.Draw(canvas, "RGBA")
            self._draw_gradient(draw)
            self._draw_gridlines(draw)
            self._draw_walls(draw)
            if self.opts.show_auras:
                self._draw_auras(canvas, frame)
            self._draw_goal(draw)
            self._draw_agents(canvas, frame)
            self._draw_hazards(canvas, frame)
            self._draw_hud(draw, frame.t)
            if self.opts.show_legend:
                self._draw_legend(draw, frame)
            frames.append(canvas.convert("RGB"))
        return frames

    def save_gif(self, frames: List[Image.Image], out_path: str) -> None:
        if not frames:
            raise ValueError("No frames to save")
        duration = int(1000 / max(1, self.opts.fps))
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            disposal=2,
        )

    # Drawing helpers -------------------------------------------------

    def _create_canvas(self) -> Image.Image:
        w = self.opts.border * 2 + self.grid_w * self.opts.cell_size
        h = self.opts.border * 2 + self.grid_h * self.opts.cell_size
        if self.opts.show_legend:
            w += LEGEND_WIDTH + self.opts.border
        return Image.new("RGBA", (w, h), WHITE)

    def _cell_rect(self, x: int, y: int) -> Tuple[int, int, int, int]:
        cs = self.opts.cell_size
        bx = self.opts.border + x * cs
        by = self.opts.border + y * cs
        return (bx, by, bx + cs, by + cs)

    def _draw_gradient(self, draw: ImageDraw.ImageDraw) -> None:
        if not self.opts.show_gradient:
            return
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                strength = self.gradient[y][x]
                if strength <= 0:
                    continue
                blend = min(1.0, strength * GRADIENT_INTENSITY)
                alpha = max(0, int(GRADIENT_ALPHA * blend))
                if alpha == 0:
                    continue
                r = int(WHITE[0] * (1 - blend) + GOAL_GOLD[0] * blend)
                g = int(WHITE[1] * (1 - blend) + GOAL_GOLD[1] * blend)
                b = int(WHITE[2] * (1 - blend) + GOAL_GOLD[2] * blend)
                draw.rectangle(self._cell_rect(x, y), fill=(r, g, b, alpha))

    def _draw_gridlines(self, draw: ImageDraw.ImageDraw) -> None:
        if not self.opts.show_gridlines:
            return
        cs = self.opts.cell_size
        left = self.opts.border
        top = self.opts.border
        width_px = self.grid_w * cs
        height_px = self.grid_h * cs
        for x in range(self.grid_w + 1):
            x0 = left + x * cs
            draw.line([(x0, top), (x0, top + height_px)], fill=GRID_LINE, width=1)
        for y in range(self.grid_h + 1):
            y0 = top + y * cs
            draw.line([(left, y0), (left + width_px, y0)], fill=GRID_LINE, width=1)

    def _draw_walls(self, draw: ImageDraw.ImageDraw) -> None:
        for wx, wy in self.walls:
            draw.rectangle(self._cell_rect(wx, wy), fill=BLACK)

    def _draw_goal(self, draw: ImageDraw.ImageDraw) -> None:
        gx, gy = self.goal
        rect = self._cell_rect(gx, gy)
        draw.rectangle(rect, fill=GOAL_GOLD, outline=BLACK, width=3)

    def _draw_auras(self, canvas: Image.Image, frame) -> None:
        agents = frame.agents
        turn = frame.t
        radius = self.episode.meta.view.radius
        shape = self.episode.meta.view.kind
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay, "RGBA")
        for agent in agents:
            status = self._agent_status(agent, turn)
            if status == "FINISHED":
                continue
            base = self.agent_colors.get(agent.agent_id, (80, 80, 80))
            lighten = 0.3
            tinted = (
                int(base[0] * (1 - lighten) + WHITE[0] * lighten),
                int(base[1] * (1 - lighten) + WHITE[1] * lighten),
                int(base[2] * (1 - lighten) + WHITE[2] * lighten),
            )
            aura_color = (*tinted, AURA_ALPHA)
            cells = (
                chebyshev_cells(agent.pos.x, agent.pos.y, radius, self.grid_w, self.grid_h)
                if shape == "square"
                else manhattan_cells(agent.pos.x, agent.pos.y, radius, self.grid_w, self.grid_h)
            )
            for cx, cy in cells:
                overlay_draw.rectangle(self._cell_rect(cx, cy), fill=aura_color)
        canvas.alpha_composite(overlay)

    def _draw_agents(self, canvas: Image.Image, frame) -> None:
        turn = frame.t
        cell_size = self.opts.cell_size
        inset = max(2, cell_size // 5)

        for agent in frame.agents:
            if self._agent_status(agent, turn) == "FINISHED":
                continue
            sprite = self._sprite_for_agent(agent, cell_size)
            if sprite is None:
                continue
            rect = self._cell_rect(agent.pos.x, agent.pos.y)
            dest = (rect[0], rect[1])
            base_is_goal = (agent.pos.x, agent.pos.y) == self.goal
            base_is_wall = (agent.pos.x, agent.pos.y) in self.walls
            to_paste = sprite
            if base_is_goal or base_is_wall:
                inner_size = max(1, cell_size - inset * 2)
                if inner_size > 0:
                    inner = sprite.resize((inner_size, inner_size), Image.LANCZOS)
                    framed = Image.new("RGBA", (cell_size, cell_size), (0, 0, 0, 0))
                    framed.paste(inner, (inset, inset), inner)
                    to_paste = framed
            canvas.paste(to_paste, dest, to_paste)
        draw = ImageDraw.Draw(canvas, "RGBA")
        for agent in frame.agents:
            rect = self._cell_rect(agent.pos.x, agent.pos.y)
            draw.rectangle(rect, outline=BLACK, width=2)

    def _draw_hazards(self, canvas: Image.Image, frame) -> None:
        hazards = getattr(frame, "hazards", None)
        if not hazards:
            return
        draw = ImageDraw.Draw(canvas, "RGBA")
        radius = max(2, self.opts.cell_size // 6)
        fill = (128, 128, 128, 200)
        for hazard in hazards:
            rect = self._cell_rect(hazard.pos.x, hazard.pos.y)
            cx = (rect[0] + rect[2]) // 2
            cy = (rect[1] + rect[3]) // 2
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill, outline=BLACK)

    def _sprite_for_agent(self, agent: AgentState, cell_size: int) -> Optional[Image.Image]:
        orientation = (agent.orientation or "E").upper()
        orient_key = {
            "N": "north",
            "S": "south",
            "E": "east",
            "W": "west",
        }.get(orientation, "east")
        color = self.agent_colors.get(agent.agent_id, (80, 80, 80))
        cache_key = (orient_key, color, cell_size)
        if cache_key in self.sprite_cache:
            sprite = self.sprite_cache[cache_key]
            return sprite
        base = self.base_sprites.get(orient_key)
        if base is None:
            return None
        tinted = self._tint_sprite(base, color)
        if tinted.size != (cell_size, cell_size):
            tinted = tinted.resize((cell_size, cell_size), Image.LANCZOS)
        self.sprite_cache[cache_key] = tinted
        return tinted

    def _tint_sprite(self, sprite: Image.Image, color: Tuple[int, int, int]) -> Image.Image:
        body_main = (220, 30, 30, 255)
        body_shadow = (150, 0, 0, 255)
        backpack = (180, 20, 20, 255)
        target_main = (*color, 255)
        target_shadow = tuple(min(255, int(c * 0.65)) for c in (*color,)) + (255,)
        target_backpack = tuple(min(255, int(c * 0.85)) for c in (*color,)) + (255,)

        tinted = sprite.copy()
        pixels = tinted.load()
        width, height = tinted.size
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a == 0:
                    continue
                current = (r, g, b, a)
                if current == body_main:
                    pixels[x, y] = target_main
                elif current == body_shadow:
                    pixels[x, y] = target_shadow
                elif current == backpack:
                    pixels[x, y] = target_backpack
        return tinted

    def _load_base_sprites(self) -> Dict[str, Image.Image]:
        sprites: Dict[str, Image.Image] = {}
        base_dir = Path(__file__).resolve().parents[3] / "assets" / "sprites"
        for orient in ("north", "south", "east", "west"):
            path = base_dir / f"crewmate_{orient}.png"
            if path.is_file():
                sprites[orient] = Image.open(path).convert("RGBA")
        return sprites

    def _draw_hud(self, draw: ImageDraw.ImageDraw, turn: int) -> None:
        # HUD content rendered in legend; nothing extra here.
        return

    def _draw_legend(self, draw: ImageDraw.ImageDraw, frame) -> None:
        cs = self.opts.cell_size
        grid_right = self.opts.border + self.grid_w * cs
        left = grid_right + self.opts.border
        top = self.opts.border
        height = self.grid_h * cs
        draw.rectangle((left, top, left + LEGEND_WIDTH, top + height), fill=WHITE, outline=BLACK, width=1)

        text_y = top + 8
        title = self.episode.meta.title or "episode"
        draw.text((left + 8, text_y), title, fill=BLACK, font=self.font)
        line_height = self.font.getbbox("Ag")[3]
        text_y += line_height + 6
        draw.text((left + 8, text_y), f"turn {frame.t}", fill=BLACK, font=self.font)
        text_y += line_height + 6
        draw.text((left + 8, text_y), f"goal ({self.goal[0]},{self.goal[1]})", fill=BLACK, font=self.font)
        text_y += line_height + 12

        for agent in frame.agents:
            color = self.agent_colors.get(agent.agent_id, (0, 0, 0))
            status = self._agent_status(agent, frame.t)
            fill_alpha = 180 if status == "FINISHED" else 255
            draw.rectangle((left + 8, text_y, left + 28, text_y + 20), fill=(*color, fill_alpha), outline=BLACK)
            info = f"{agent.agent_id} pos=({agent.pos.x},{agent.pos.y})"
            if agent.orientation:
                info += f" dir={agent.orientation}"
            if agent.action:
                info += f" {agent.action}"
            if status == "FINISHED":
                info += " FINISHED"
            draw.text((left + 36, text_y + 2), info, fill=BLACK, font=self.font)
            text_y += 24

        hazard_rect = (left + 8, text_y, left + 28, text_y + 20)
        draw.ellipse(hazard_rect, fill=(128, 128, 128, 200), outline=BLACK)
        draw.text((left + 36, text_y + 2), "NO_GO cone", fill=BLACK, font=self.font)


# Gradient helpers ----------------------------------------------------

def bfs_distance_map(width: int, height: int, goal: Tuple[int, int], walls: set[Tuple[int, int]]) -> List[List[Optional[int]]]:
    from collections import deque

    distances: List[List[Optional[int]]] = [[None for _ in range(width)] for _ in range(height)]
    gx, gy = goal
    distances[gy][gx] = 0
    queue = deque([(gx, gy)])
    neighbours = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    while queue:
        x, y = queue.popleft()
        for dx, dy in neighbours:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls and distances[ny][nx] is None:
                distances[ny][nx] = distances[y][x] + 1  # type: ignore[index]
                queue.append((nx, ny))
    return distances


def manhattan_distance_map(width: int, height: int, goal: Tuple[int, int]) -> List[List[int]]:
    gx, gy = goal
    return [[abs(x - gx) + abs(y - gy) for x in range(width)] for y in range(height)]


def normalize_distance_map(field: List[List[Optional[int]]]) -> List[List[float]]:
    values = [v for row in field for v in row if v is not None]
    if not values:
        return [[0.0 for _ in row] for row in field]
    max_v = max(values) or 1
    normalized: List[List[float]] = []
    for row in field:
        normalized.append([1.0 - (v / max_v) if v is not None else 0.0 for v in row])
    return normalized


def chebyshev_cells(cx: int, cy: int, radius: int, width: int, height: int) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            cells.append((x, y))
    return cells


def manhattan_cells(cx: int, cy: int, radius: int, width: int, height: int) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if abs(x - cx) + abs(y - cy) <= radius:
                cells.append((x, y))
    return cells
