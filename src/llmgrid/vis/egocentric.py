"""Render agent egocentric views from observation JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Colors
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
GRAY = (200, 200, 200, 255)
FREE_COLOR = (144, 238, 144, 255)  # Light green
WALL_COLOR = (50, 50, 50, 255)  # Dark gray
AGENT_COLOR = (100, 149, 237, 255)  # Cornflower blue
GOAL_COLOR = (255, 215, 0, 255)  # Gold
UNKNOWN_COLOR = (220, 220, 220, 255)  # Light gray
TRAIL_COLOR = (255, 182, 193, 255)  # Light pink


@dataclass
class EgocentricRenderOptions:
    font_size: int = 20
    cell_size: int = 40  # For local patch visualization
    padding: int = 20
    show_goal_arrow: bool = True
    show_orientation_arrow: bool = True
    show_adjacent_states: bool = True
    show_local_patch: bool = True


class EgocentricRenderer:
    """Renders an agent's egocentric view from observation JSON."""

    def __init__(self, options: Optional[EgocentricRenderOptions] = None):
        self.opts = options or EgocentricRenderOptions()
        self.font = self._load_font(self.opts.font_size)
        self.mono_font = self._load_mono_font(self.opts.font_size)

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size)  # type: ignore
            except Exception:
                continue
        return ImageFont.load_default()

    def _load_mono_font(self, size: int) -> ImageFont.ImageFont:
        """Load monospaced font for ASCII map rendering."""
        mono_paths = [
            "/System/Library/Fonts/Courier.dfont",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "C:\\Windows\\Fonts\\cour.ttf",
        ]
        for font_path in mono_paths:
            try:
                return ImageFont.truetype(font_path, size)  # type: ignore
            except Exception:
                continue
        # Fall back to regular font if mono not available
        return self.font

    def render_observation(self, observation: Dict) -> Tuple[Image.Image, Image.Image]:
        """Render egocentric view as TWO images: (map_image, info_image)."""

        # IMAGE 1: ASCII World Map
        map_image = self._render_map(observation)

        # IMAGE 2: Info Panel (compass, ego-view, local patch)
        info_image = self._render_info(observation)

        return (map_image, info_image)

    def _render_map(self, observation: Dict) -> Image.Image:
        """Render just the ASCII world map."""
        padding = self.opts.padding

        ascii_map = observation.get('world_map_ascii', 'No map available')
        map_lines = ascii_map.strip().split('\n')

        # Calculate dimensions
        if map_lines:
            char_width = self.mono_font.getlength('X')  # type: ignore
            char_height = self.opts.font_size + 4
            map_width = int(max(len(line) for line in map_lines) * char_width) + padding * 2
            map_height = len(map_lines) * char_height + padding * 2 + self.opts.font_size + 30
        else:
            map_width = 400
            map_height = 200

        canvas = Image.new('RGBA', (map_width, map_height), WHITE)
        draw = ImageDraw.Draw(canvas, 'RGBA')

        self._draw_ascii_map(draw, observation, padding, padding, map_width - padding * 2)

        return canvas.convert('RGB')

    def _render_info(self, observation: Dict) -> Image.Image:
        """Render info panel HORIZONTALLY: agent-info | compass | ego-view | local-patch."""
        padding = self.opts.padding

        # Section widths
        info_section_width = 180
        compass_width = 200
        ego_width = 200
        patch_width = 160

        total_width = info_section_width + compass_width + ego_width + patch_width + padding * 5
        total_height = 350

        canvas = Image.new('RGBA', (total_width, total_height), WHITE)
        draw = ImageDraw.Draw(canvas, 'RGBA')

        x = padding
        y = padding

        # Section 1: Agent Info
        self._draw_agent_info_compact(draw, observation, x, y, info_section_width)
        x += info_section_width + padding

        # Section 2: Compass
        self._draw_compass_section(draw, observation, x, y, compass_width)
        x += compass_width + padding

        # Section 3: Ego-centric View
        self._draw_ego_section(draw, observation, x, y, ego_width)
        x += ego_width + padding

        # Section 4: Local Patch
        self._draw_patch_section(draw, observation, x, y, patch_width)

        return canvas.convert('RGB')

    def _draw_agent_info_compact(self, draw: ImageDraw.ImageDraw, obs: Dict, x: int, y: int, width: int) -> None:
        """Draw compact agent info (text only)."""
        line_height = self.opts.font_size + 6

        self_state = obs.get('self_state', {})
        pos = self_state.get('abs_pos', {})
        orientation = self_state.get('orientation', 'UNKNOWN')

        draw.text((x, y), f"Agent:", fill=BLACK, font=self.font)
        y += line_height
        draw.text((x, y), f"{self_state.get('agent_id', '?')}", fill=BLACK, font=self.font)
        y += line_height * 1.5

        draw.text((x, y), f"Position:", fill=BLACK, font=self.font)
        y += line_height
        draw.text((x, y), f"({pos.get('x', '?')}, {pos.get('y', '?')})", fill=BLACK, font=self.font)
        y += line_height * 1.5

        draw.text((x, y), f"Facing:", fill=BLACK, font=self.font)
        y += line_height
        draw.text((x, y), f"{orientation}", fill=AGENT_COLOR[:3], font=self.font)
        y += line_height * 1.5

        goal_sensor = obs.get('goal_sensor', {})
        if goal_sensor.get('available'):
            bearing = goal_sensor.get('bearing', '?')
            strength = goal_sensor.get('strength', '?')
            draw.text((x, y), f"Goal:", fill=BLACK, font=self.font)
            y += line_height
            draw.text((x, y), f"{bearing}", fill=GOAL_COLOR[:3], font=self.font)
            y += line_height
            draw.text((x, y), f"({strength})", fill=GOAL_COLOR[:3], font=self.font)

    def _draw_compass_section(self, draw: ImageDraw.ImageDraw, obs: Dict, x: int, y: int, width: int) -> None:
        """Draw compass with title."""
        draw.text((x, y), "Compass:", fill=BLACK, font=self.font)
        y += self.opts.font_size + 10
        center_x = x + width // 2
        center_y = y + 70
        self._draw_compass_rose(draw, obs, center_x, center_y)

    def _draw_ego_section(self, draw: ImageDraw.ImageDraw, obs: Dict, x: int, y: int, width: int) -> None:
        """Draw ego-centric view with title."""
        draw.text((x, y), "Ego View:", fill=BLACK, font=self.font)
        y += self.opts.font_size + 10
        grid_x = x + (width - 150) // 2
        self._draw_egocentric_adjacent(draw, obs, grid_x, y)

    def _draw_patch_section(self, draw: ImageDraw.ImageDraw, obs: Dict, x: int, y: int, width: int) -> None:
        """Draw local patch with title."""
        draw.text((x, y), "Local 3x3:", fill=BLACK, font=self.font)
        y += self.opts.font_size + 10
        patch_x = x + (width - 120) // 2
        self._draw_local_patch(draw, obs, patch_x, y)

    def _draw_ascii_map(self, draw: ImageDraw.ImageDraw, obs: Dict, x: int, y: int, max_width: int) -> None:
        """Render the ASCII world map."""
        ascii_map = obs.get('world_map_ascii', 'No map available')

        # Draw title
        draw.text((x, y), "Agent's World Map", fill=BLACK, font=self.font)
        y += self.opts.font_size + 10

        # Draw ASCII map line by line, fixing y-axis alignment
        char_height = self.opts.font_size + 4

        for line in ascii_map.split('\n'):
            # Fix y= label alignment: ADD 3 spaces to align grid with x-axis numbers
            if line.startswith('y='):
                # Add 3 spaces to push grid content right to align with x(units)
                line = '   ' + line
            # Fix x(tens) alignment: ADD 1 space to align with x(units)
            elif line.startswith('x(tens)'):
                line = ' ' + line

            draw.text((x, y), line, fill=BLACK, font=self.mono_font)
            y += char_height

    def _draw_info_panel(self, draw: ImageDraw.ImageDraw, obs: Dict, x: int, y: int, width: int) -> None:
        """Draw the information panel as clean vertical sections."""
        line_height = self.opts.font_size + 8
        start_y = y

        # ===== SECTION 1: Agent Info =====
        self_state = obs.get('self_state', {})
        pos = self_state.get('abs_pos', {})
        orientation = self_state.get('orientation', 'UNKNOWN')

        draw.text((x, y), f"Agent: {self_state.get('agent_id', 'unknown')}", fill=BLACK, font=self.font)
        y += line_height

        draw.text((x, y), f"Position: ({pos.get('x', '?')}, {pos.get('y', '?')})", fill=BLACK, font=self.font)
        y += line_height

        draw.text((x, y), f"Facing: {orientation}", fill=AGENT_COLOR[:3], font=self.font)
        y += line_height

        goal_sensor = obs.get('goal_sensor', {})
        if goal_sensor.get('available'):
            bearing = goal_sensor.get('bearing', 'UNKNOWN')
            strength = goal_sensor.get('strength', 'UNKNOWN')
            draw.text((x, y), f"Goal: {bearing} ({strength})", fill=GOAL_COLOR[:3], font=self.font)
            y += line_height

        y += 10  # Spacing

        # ===== SECTION 2: Compass Rose =====
        draw.text((x, y), "Compass:", fill=BLACK, font=self.font)
        y += line_height
        compass_center_x = x + width // 2
        self._draw_compass_rose(draw, obs, compass_center_x, y + 60)
        y += 170  # Space for compass + legend

        # ===== SECTION 3: Ego-centric View =====
        if self.opts.show_adjacent_states:
            draw.text((x, y), "View (ego-centric):", fill=BLACK, font=self.font)
            y += line_height
            grid_x = x + (width - 150) // 2  # Center the 3x3 grid
            self._draw_egocentric_adjacent(draw, obs, grid_x, y)
            y += 160  # Space for 3x3 grid

        # ===== SECTION 4: Local Patch =====
        if self.opts.show_local_patch:
            draw.text((x, y), "Local Patch (3x3):", fill=BLACK, font=self.font)
            y += line_height
            patch_x = x + (width - 120) // 2  # Center the 3x3 patch
            self._draw_local_patch(draw, obs, patch_x, y)

    def _draw_compass_rose(self, draw: ImageDraw.ImageDraw, obs: Dict, center_x: int, center_y: int) -> None:
        """Draw a compass rose showing agent orientation and goal bearing."""
        radius = 50

        # Draw compass circle
        draw.ellipse(
            [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
            outline=GRAY,
            width=2
        )

        # Draw cardinal directions
        directions = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}

        for dir_label, (dx, dy) in directions.items():
            text_x = center_x + dx * (radius + 15)
            text_y = center_y + dy * (radius + 15)
            draw.text((text_x - 5, text_y - 10), dir_label, fill=BLACK, font=self.font)

        # Draw agent orientation (blue arrow)
        orientation = obs.get('self_state', {}).get('orientation', 'N')
        if orientation in directions:
            dx, dy = directions[orientation]
            arrow_len = radius - 10
            tip_x = center_x + dx * arrow_len
            tip_y = center_y + dy * arrow_len

            # Thick blue arrow for orientation
            draw.line([center_x, center_y, tip_x, tip_y], fill=AGENT_COLOR[:3], width=6)

            # Arrow head
            perp_x, perp_y = -dy, dx
            head_size = 12
            left_x = tip_x - dx * head_size + perp_x * head_size // 2
            left_y = tip_y - dy * head_size + perp_y * head_size // 2
            right_x = tip_x - dx * head_size - perp_x * head_size // 2
            right_y = tip_y - dy * head_size - perp_y * head_size // 2
            draw.polygon([(tip_x, tip_y), (left_x, left_y), (right_x, right_y)], fill=AGENT_COLOR[:3])

        # Draw goal bearing (gold arrow, thinner)
        goal_sensor = obs.get('goal_sensor', {})
        if goal_sensor.get('available'):
            bearing = goal_sensor.get('bearing', 'N')
            if bearing in directions:
                dx, dy = directions[bearing]
                arrow_len = radius - 5
                tip_x = center_x + dx * arrow_len
                tip_y = center_y + dy * arrow_len

                # Thinner gold arrow for goal
                draw.line([center_x, center_y, tip_x, tip_y], fill=GOAL_COLOR[:3], width=4)

                # Arrow head
                perp_x, perp_y = -dy, dx
                head_size = 10
                left_x = tip_x - dx * head_size + perp_x * head_size // 2
                left_y = tip_y - dy * head_size + perp_y * head_size // 2
                right_x = tip_x - dx * head_size - perp_x * head_size // 2
                right_y = tip_y - dy * head_size - perp_y * head_size // 2
                draw.polygon([(tip_x, tip_y), (left_x, left_y), (right_x, right_y)], fill=GOAL_COLOR[:3])

        # Add legend
        legend_y = center_y + radius + 25
        draw.text((center_x - 60, legend_y), "Blue: Facing", fill=AGENT_COLOR[:3], font=self.font)
        draw.text((center_x - 60, legend_y + 20), "Gold: Goal", fill=GOAL_COLOR[:3], font=self.font)

    def _draw_egocentric_adjacent(self, draw: ImageDraw.ImageDraw, obs: Dict, x: int, y: int) -> None:
        """Draw adjacent cells from agent's perspective (FORWARD/BACK/LEFT/RIGHT)."""
        cell_size = 50

        # Get agent orientation
        orientation = obs.get('self_state', {}).get('orientation', 'N')
        adjacent = obs.get('adjacent', [])

        # Convert absolute directions to ego-centric
        # Rotation mapping: if facing N, N=forward, E=right, S=back, W=left
        rotation_map = {
            'N': {'N': 'FWD', 'E': 'RIGHT', 'S': 'BACK', 'W': 'LEFT'},
            'E': {'E': 'FWD', 'S': 'RIGHT', 'W': 'BACK', 'N': 'LEFT'},
            'S': {'S': 'FWD', 'W': 'RIGHT', 'N': 'BACK', 'E': 'LEFT'},
            'W': {'W': 'FWD', 'N': 'RIGHT', 'E': 'BACK', 'S': 'LEFT'},
        }

        ego_map = rotation_map.get(orientation, {})

        # Build state map
        state_map = {ego_map.get(adj['dir'], adj['dir']): adj['state'] for adj in adjacent}

        # Draw in ego-centric layout
        # Layout:
        #     [FWD]
        # [LEFT][YOU][RIGHT]
        #     [BACK]

        positions = {
            'FWD': (1, 0),
            'RIGHT': (2, 1),
            'BACK': (1, 2),
            'LEFT': (0, 1),
        }

        # Draw agent in center
        center_x = x + cell_size
        center_y = y + cell_size
        draw.rectangle(
            [center_x, center_y, center_x + cell_size, center_y + cell_size],
            fill=AGENT_COLOR,
            outline=BLACK,
            width=3
        )

        # Draw orientation arrow in center cell showing which way is forward
        arrow_tip_x = center_x + cell_size // 2
        arrow_tip_y = center_y + 10
        arrow_base_y = center_y + cell_size - 10
        draw.line([arrow_tip_x, arrow_base_y, arrow_tip_x, arrow_tip_y], fill=WHITE, width=4)
        # Arrow head
        draw.polygon([
            (arrow_tip_x, arrow_tip_y),
            (arrow_tip_x - 8, arrow_tip_y + 12),
            (arrow_tip_x + 8, arrow_tip_y + 12)
        ], fill=WHITE)

        # State colors
        state_colors = {
            'FREE': FREE_COLOR,
            'WALL': WALL_COLOR,
            'OUT_OF_BOUNDS': (50, 50, 50, 255),
            'BLOCK_AGENT': (255, 100, 100, 255),
        }

        # State abbreviations
        state_abbrev = {
            'FREE': 'F',
            'WALL': 'W',
            'OUT_OF_BOUNDS': 'X',
            'BLOCK_AGENT': 'A',
        }

        # Draw ego-centric cells
        for ego_dir, (grid_x, grid_y) in positions.items():
            cell_x = x + grid_x * cell_size
            cell_y = y + grid_y * cell_size

            state = state_map.get(ego_dir, 'UNKNOWN')
            color = state_colors.get(state, GRAY)

            draw.rectangle(
                [cell_x, cell_y, cell_x + cell_size, cell_y + cell_size],
                fill=color,
                outline=BLACK,
                width=3
            )

            # Draw label (direction)
            label_color = WHITE if state != 'FREE' else BLACK
            draw.text((cell_x + 5, cell_y + 5), ego_dir[:3], fill=label_color, font=self.font)

            # Draw state abbreviation
            abbrev = state_abbrev.get(state, '?')
            draw.text((cell_x + 15, cell_y + 28), abbrev, fill=label_color, font=self.font)

    def _draw_direction_arrow(self, draw: ImageDraw.ImageDraw, x: int, y: int, direction: str, color: Tuple[int, int, int], label: str) -> None:
        """Draw an arrow indicating a direction."""
        arrow_size = 30

        # Direction mappings
        dir_map = {
            'N': (0, -1),
            'E': (1, 0),
            'S': (0, 1),
            'W': (-1, 0),
        }

        dx, dy = dir_map.get(direction, (0, 0))

        # Draw arrow
        center_x, center_y = x, y
        tip_x = center_x + dx * arrow_size
        tip_y = center_y + dy * arrow_size

        # Arrow shaft
        draw.line([center_x, center_y, tip_x, tip_y], fill=color, width=4)

        # Arrow head
        if dx != 0 or dy != 0:
            # Calculate perpendicular
            perp_x, perp_y = -dy, dx
            head_size = 10

            left_x = tip_x - dx * head_size + perp_x * head_size // 2
            left_y = tip_y - dy * head_size + perp_y * head_size // 2
            right_x = tip_x - dx * head_size - perp_x * head_size // 2
            right_y = tip_y - dy * head_size - perp_y * head_size // 2

            draw.polygon([(tip_x, tip_y), (left_x, left_y), (right_x, right_y)], fill=color)

    def _draw_local_patch(self, draw: ImageDraw.ImageDraw, obs: Dict, x: int, y: int) -> None:
        """Draw the local 3x3 patch with color coding."""
        local_patch = obs.get('local_patch', {})
        rows = local_patch.get('rows', [])

        if not rows:
            return

        cell_size = self.opts.cell_size

        # Character to color mapping
        char_colors = {
            '.': FREE_COLOR,
            '#': WALL_COLOR,
            'A': AGENT_COLOR,
            'X': UNKNOWN_COLOR,
            '~': TRAIL_COLOR,
            'G': GOAL_COLOR,
        }

        for row_idx, row in enumerate(rows):
            for col_idx, char in enumerate(row):
                cell_x = x + col_idx * cell_size
                cell_y = y + row_idx * cell_size

                color = char_colors.get(char, GRAY)
                draw.rectangle(
                    [cell_x, cell_y, cell_x + cell_size, cell_y + cell_size],
                    fill=color,
                    outline=BLACK,
                    width=2
                )

                # Draw character
                text_color = BLACK if char in ['.', 'X', '~', 'G'] else WHITE
                draw.text((cell_x + 12, cell_y + 8), char, fill=text_color, font=self.font)


def render_agent_view(observation_json: Dict, output_base_path: Path, options: Optional[EgocentricRenderOptions] = None) -> Tuple[Path, Path]:
    """Convenience function to render and save an agent observation as TWO images.

    Returns: (map_path, info_path)
    """
    renderer = EgocentricRenderer(options)
    map_image, info_image = renderer.render_observation(observation_json)

    # Generate output paths
    base_name = output_base_path.stem
    parent = output_base_path.parent
    suffix = output_base_path.suffix

    map_path = parent / f"{base_name}_map{suffix}"
    info_path = parent / f"{base_name}_info{suffix}"

    map_image.save(str(map_path))
    info_image.save(str(info_path))

    return (map_path, info_path)
