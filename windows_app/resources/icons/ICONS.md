# Trading Bot Dashboard - Icon Specifications

This document specifies all icons used in the Windows application.

## Application Icons

### Main Application Icon
- **File**: `app.ico`
- **Sizes**: 16x16, 32x32, 48x48, 256x256
- **Description**: Main application icon showing a trading chart or robot symbol
- **Usage**: Window title bar, taskbar, desktop shortcut

### Splash Screen Logo
- **File**: `splash_logo.png`
- **Size**: 200x200
- **Description**: High-resolution logo for splash screen
- **Usage**: Startup splash screen

## Tab Icons (24x24 PNG)

### Dashboard
- **File**: `dashboard.png`
- **Symbol**: 📊 (Chart/Dashboard)
- **Color**: Blue (#5E81AC)
- **Description**: Bar chart or analytics dashboard icon

### Trades History
- **File**: `trades.png`
- **Symbol**: 📈 (Trending Up)
- **Color**: Green (#A3BE8C)
- **Description**: Upward trending chart with data points

### Charts
- **File**: `charts.png`
- **Symbol**: 📉 (Chart Line)
- **Color**: Purple (#B48EAD)
- **Description**: Line chart with multiple data series

### Logs
- **File**: `logs.png`
- **Symbol**: 📝 (Document/Logs)
- **Color**: Yellow (#EBCB8B)
- **Description**: Document or terminal icon

### Settings/Config
- **File**: `settings.png`
- **Symbol**: ⚙️ (Gear)
- **Color**: Gray (#4C566A)
- **Description**: Gear or wrench icon

## Status Icons (16x16 PNG)

### Bot Status
- **File**: `bot_running.png`
- **Symbol**: ✓ (Check/Play)
- **Color**: Green (#A3BE8C)
- **Description**: Green circle or play button

- **File**: `bot_stopped.png`
- **Symbol**: ⏹ (Stop)
- **Color**: Red (#BF616A)
- **Description**: Red circle or stop button

- **File**: `bot_warning.png`
- **Symbol**: ⚠ (Warning)
- **Color**: Yellow (#EBCB8B)
- **Description**: Yellow warning triangle

### Trade Direction
- **File**: `long.png`
- **Symbol**: ⬆ (Up Arrow)
- **Color**: Green (#A3BE8C)
- **Description**: Up arrow for long positions

- **File**: `short.png`
- **Symbol**: ⬇ (Down Arrow)
- **Color**: Red (#BF616A)
- **Description**: Down arrow for short positions

### P&L Indicators
- **File**: `profit.png`
- **Symbol**: + (Plus)
- **Color**: Green (#A3BE8C)
- **Description**: Green plus or up triangle

- **File**: `loss.png`
- **Symbol**: - (Minus)
- **Color**: Red (#BF616A)
- **Description**: Red minus or down triangle

## Action Icons (20x20 PNG)

### Control Actions
- **File**: `start.png`
- **Symbol**: ▶ (Play)
- **Color**: Green (#A3BE8C)
- **Description**: Play/start button

- **File**: `stop.png`
- **Symbol**: ⏹ (Stop)
- **Color**: Red (#BF616A)
- **Description**: Stop button

- **File**: `restart.png`
- **Symbol**: ⟲ (Refresh)
- **Color**: Blue (#5E81AC)
- **Description**: Circular arrow

- **File**: `refresh.png`
- **Symbol**: ↻ (Reload)
- **Color**: Blue (#5E81AC)
- **Description**: Circular arrow (lighter)

### File Actions
- **File**: `save.png`
- **Symbol**: 💾 (Save)
- **Color**: Blue (#5E81AC)
- **Description**: Floppy disk icon

- **File**: `reset.png`
- **Symbol**: ↺ (Reset)
- **Color**: Orange (#D08770)
- **Description**: Counter-clockwise arrow

- **File**: `export.png`
- **Symbol**: ⬇ (Download)
- **Color**: Green (#A3BE8C)
- **Description**: Download arrow

## Log Level Icons (14x14 PNG)

- **File**: `log_info.png`
- **Symbol**: ℹ (Info)
- **Color**: Blue (#5E81AC)

- **File**: `log_warning.png`
- **Symbol**: ⚠ (Warning)
- **Color**: Yellow (#EBCB8B)

- **File**: `log_error.png`
- **Symbol**: ✕ (Error)
- **Color**: Red (#BF616A)

- **File**: `log_debug.png`
- **Symbol**: 🐛 (Bug)
- **Color**: Gray (#4C566A)

## Color Palette

Following the application's dark theme:

```python
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
```

## Usage in Code

```python
from PyQt6.QtGui import QIcon, QPixmap
from pathlib import Path

ICON_DIR = Path(__file__).parent.parent / "resources" / "icons"

def get_icon(name: str) -> QIcon:
    """Get icon by name."""
    icon_path = ICON_DIR / f"{name}.png"
    if icon_path.exists():
        return QIcon(str(icon_path))
    return QIcon()  # Return empty icon if not found

# Usage examples
dashboard_icon = get_icon("dashboard")
bot_running_icon = get_icon("bot_running")
```

## Generation

Icons can be generated using the provided `generate_icons.py` script:

```bash
python windows_app/resources/icons/generate_icons.py
```

This will create all icons with appropriate sizes, colors, and shapes.
