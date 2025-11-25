#!/usr/bin/env python3
"""
Extract a specific frame from a GIF and copy to clipboard
"""
import sys
from pathlib import Path
from PIL import Image
import subprocess

def extract_frame(gif_path: Path, frame_num: int, output_path: Path) -> None:
    """Extract frame from GIF and save as PNG"""
    img = Image.open(gif_path)

    # Navigate to the desired frame
    try:
        img.seek(frame_num)
    except EOFError:
        print(f"Frame {frame_num} not found in GIF")
        sys.exit(1)

    # Convert to RGB if needed and save
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img.save(output_path, 'PNG')
    print(f"Extracted frame {frame_num} to {output_path}")

def copy_to_clipboard(image_path: Path) -> None:
    """Copy image to macOS clipboard using osascript"""
    subprocess.run([
        'osascript', '-e',
        f'set the clipboard to (read (POSIX file "{image_path}") as JPEG picture)'
    ], check=True)
    print(f"Copied {image_path} to clipboard")

if __name__ == "__main__":
    gif_path = Path("/Users/3bn/Documents/My_Repos/agent-talk-2/slides/decks/agent-talk/assets/hero_freeform.gif")
    frame_num = 38
    output_path = Path("/tmp/frame38.png")

    extract_frame(gif_path, frame_num, output_path)
    copy_to_clipboard(output_path)
