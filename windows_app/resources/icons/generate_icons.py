"""
Icon Generator for Trading Bot Dashboard

Generates all required icons with proper colors and shapes.
Requires: pip install Pillow
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import os

# Color palette matching the application theme
COLORS = {
    'bg_dark': '#1E1E2E',
    'bg_medium': '#2A2A3E',
    'accent_blue': '#5E81AC',
    'accent_green': '#A3BE8C',
    'accent_red': '#BF616A',
    'accent_yellow': '#EBCB8B',
    'accent_purple': '#B48EAD',
    'accent_orange': '#D08770',
    'text_primary': '#ECEFF4',
    'text_secondary': '#D8DEE9',
}


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_icon(size: int, bg_color: str, fg_color: str, shape: str = 'circle') -> Image.Image:
    """
    Create a base icon with specified shape and colors.

    Args:
        size: Icon size (square)
        bg_color: Background color (hex)
        fg_color: Foreground color (hex)
        shape: 'circle', 'square', 'triangle', 'arrow_up', 'arrow_down', etc.

    Returns:
        PIL Image object
    """
    # Create transparent image
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg_rgb = hex_to_rgb(bg_color) + (255,)
    fg_rgb = hex_to_rgb(fg_color) + (255,)

    margin = size // 8
    inner_size = size - (2 * margin)

    if shape == 'circle':
        draw.ellipse([margin, margin, size - margin, size - margin], fill=fg_rgb)

    elif shape == 'square':
        draw.rectangle([margin, margin, size - margin, size - margin], fill=fg_rgb)

    elif shape == 'triangle':
        points = [
            (size // 2, margin),  # top
            (size - margin, size - margin),  # bottom right
            (margin, size - margin)  # bottom left
        ]
        draw.polygon(points, fill=fg_rgb)

    elif shape == 'arrow_up':
        points = [
            (size // 2, margin),  # top
            (size - margin, size // 2),  # right
            (size * 0.6, size // 2),  # right inner
            (size * 0.6, size - margin),  # bottom right
            (size * 0.4, size - margin),  # bottom left
            (size * 0.4, size // 2),  # left inner
            (margin, size // 2)  # left
        ]
        draw.polygon(points, fill=fg_rgb)

    elif shape == 'arrow_down':
        points = [
            (size * 0.4, margin),  # top left
            (size * 0.6, margin),  # top right
            (size * 0.6, size // 2),  # right inner
            (size - margin, size // 2),  # right
            (size // 2, size - margin),  # bottom
            (margin, size // 2),  # left
            (size * 0.4, size // 2)  # left inner
        ]
        draw.polygon(points, fill=fg_rgb)

    elif shape == 'chart_up':
        # Simple line chart trending up
        points = [
            (margin, size - margin),
            (size * 0.3, size * 0.6),
            (size * 0.5, size * 0.5),
            (size * 0.7, size * 0.3),
            (size - margin, margin)
        ]
        draw.line(points, fill=fg_rgb, width=max(2, size // 12))

    elif shape == 'chart_bars':
        # Bar chart
        bar_width = size // 5
        heights = [0.4, 0.6, 0.5, 0.7]
        for i, height in enumerate(heights):
            x = margin + i * bar_width
            y = size - margin - (inner_size * height)
            draw.rectangle([x, y, x + bar_width - 2, size - margin], fill=fg_rgb)

    elif shape == 'gear':
        # Simple gear icon
        center = size // 2
        outer_radius = inner_size // 2
        inner_radius = outer_radius // 2
        teeth = 8

        import math
        # Draw outer circle with teeth
        for i in range(teeth * 2):
            angle1 = (i * math.pi) / teeth
            angle2 = ((i + 1) * math.pi) / teeth
            radius = outer_radius if i % 2 == 0 else outer_radius * 0.8
            x1 = center + radius * math.cos(angle1)
            y1 = center + radius * math.sin(angle1)
            x2 = center + radius * math.cos(angle2)
            y2 = center + radius * math.sin(angle2)

        # Draw filled circle
        draw.ellipse([margin, margin, size - margin, size - margin], fill=fg_rgb)
        # Draw inner hole
        hole_margin = size // 3
        draw.ellipse([hole_margin, hole_margin, size - hole_margin, size - hole_margin],
                    fill=(0, 0, 0, 0))

    elif shape == 'document':
        # Simple document icon
        corner = size * 0.15
        draw.rectangle([margin, margin, size - margin, size - margin], fill=fg_rgb)
        # Folded corner
        draw.polygon([
            (size - margin, margin),
            (size - margin - corner, margin),
            (size - margin, margin + corner)
        ], fill=bg_rgb)
        # Lines
        for i in range(3):
            y = margin + (i + 1) * inner_size // 4
            draw.line([margin + 4, y, size - margin - 4, y],
                     fill=bg_rgb, width=2)

    elif shape == 'play':
        # Play triangle
        points = [
            (margin + inner_size * 0.2, margin),
            (size - margin, size // 2),
            (margin + inner_size * 0.2, size - margin)
        ]
        draw.polygon(points, fill=fg_rgb)

    elif shape == 'stop':
        # Stop square
        stop_margin = size // 4
        draw.rectangle([stop_margin, stop_margin, size - stop_margin, size - stop_margin],
                      fill=fg_rgb)

    elif shape == 'refresh':
        # Circular arrow
        import math
        center = size // 2
        radius = inner_size // 2
        # Draw arc
        for angle in range(30, 330, 5):
            rad = math.radians(angle)
            x = center + radius * math.cos(rad)
            y = center + radius * math.sin(rad)
            draw.ellipse([x-2, y-2, x+2, y+2], fill=fg_rgb)
        # Arrow head
        arrow_size = size // 6
        draw.polygon([
            (size - margin, margin + arrow_size),
            (size - margin - arrow_size, margin),
            (size - margin, margin)
        ], fill=fg_rgb)

    return img


def generate_all_icons(output_dir: Path):
    """Generate all required icons."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Trading Bot Dashboard Icons...")
    print(f"Output directory: {output_dir}")

    icons = {
        # Main app icon (256x256)
        'splash_logo.png': (200, COLORS['bg_medium'], COLORS['accent_blue'], 'chart_bars'),

        # Tab icons (24x24)
        'dashboard.png': (24, COLORS['bg_medium'], COLORS['accent_blue'], 'chart_bars'),
        'trades.png': (24, COLORS['bg_medium'], COLORS['accent_green'], 'chart_up'),
        'charts.png': (24, COLORS['bg_medium'], COLORS['accent_purple'], 'chart_up'),
        'logs.png': (24, COLORS['bg_medium'], COLORS['accent_yellow'], 'document'),
        'settings.png': (24, COLORS['bg_medium'], COLORS['text_secondary'], 'gear'),

        # Status icons (16x16)
        'bot_running.png': (16, COLORS['bg_medium'], COLORS['accent_green'], 'circle'),
        'bot_stopped.png': (16, COLORS['bg_medium'], COLORS['accent_red'], 'circle'),
        'bot_warning.png': (16, COLORS['bg_medium'], COLORS['accent_yellow'], 'triangle'),

        # Trade direction (16x16)
        'long.png': (16, COLORS['bg_medium'], COLORS['accent_green'], 'arrow_up'),
        'short.png': (16, COLORS['bg_medium'], COLORS['accent_red'], 'arrow_down'),

        # P&L indicators (16x16)
        'profit.png': (16, COLORS['bg_medium'], COLORS['accent_green'], 'arrow_up'),
        'loss.png': (16, COLORS['bg_medium'], COLORS['accent_red'], 'arrow_down'),

        # Action icons (20x20)
        'start.png': (20, COLORS['bg_medium'], COLORS['accent_green'], 'play'),
        'stop.png': (20, COLORS['bg_medium'], COLORS['accent_red'], 'stop'),
        'restart.png': (20, COLORS['bg_medium'], COLORS['accent_blue'], 'refresh'),
        'refresh.png': (20, COLORS['bg_medium'], COLORS['accent_blue'], 'refresh'),
        'save.png': (20, COLORS['bg_medium'], COLORS['accent_blue'], 'document'),
        'reset.png': (20, COLORS['bg_medium'], COLORS['accent_orange'], 'refresh'),
        'export.png': (20, COLORS['bg_medium'], COLORS['accent_green'], 'arrow_down'),

        # Log level icons (14x14)
        'log_info.png': (14, COLORS['bg_medium'], COLORS['accent_blue'], 'circle'),
        'log_warning.png': (14, COLORS['bg_medium'], COLORS['accent_yellow'], 'triangle'),
        'log_error.png': (14, COLORS['bg_medium'], COLORS['accent_red'], 'square'),
        'log_debug.png': (14, COLORS['bg_medium'], COLORS['text_secondary'], 'circle'),
    }

    count = 0
    for filename, (size, bg_color, fg_color, shape) in icons.items():
        icon = create_icon(size, bg_color, fg_color, shape)
        icon_path = output_dir / filename
        icon.save(icon_path)
        print(f"  [OK] Created {filename} ({size}x{size}, {shape})")
        count += 1

    # Create app.ico with multiple sizes
    try:
        sizes = [16, 32, 48, 256]
        ico_images = []
        for size in sizes:
            img = create_icon(size, COLORS['bg_medium'], COLORS['accent_blue'], 'chart_bars')
            ico_images.append(img)

        ico_path = output_dir / 'app.ico'
        ico_images[0].save(ico_path, format='ICO', sizes=[(s, s) for s in sizes],
                          append_images=ico_images[1:])
        print(f"  [OK] Created app.ico (multi-size)")
        count += 1
    except Exception as e:
        print(f"  [FAIL] Failed to create app.ico: {e}")

    print(f"\n[SUCCESS] Generated {count} icons successfully!")
    return count


if __name__ == '__main__':
    # Get script directory
    script_dir = Path(__file__).parent

    print("="  * 60)
    print("Trading Bot Dashboard - Icon Generator")
    print("=" * 60)
    print()

    try:
        count = generate_all_icons(script_dir)
        print()
        print("=" * 60)
        print(f"[SUCCESS] All {count} icons generated!")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"[ERROR] {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
