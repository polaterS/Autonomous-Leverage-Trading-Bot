"""
Trading Bot Dashboard - Main Application Entry Point.

Usage:
    python windows_app/app.py

Requirements:
    - PyQt6
    - PostgreSQL database running
    - Redis running (optional but recommended)
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QFont, QPalette, QColor
from windows_app.ui.main_window import MainWindow
from windows_app.ui.styles import COLORS


def create_splash_screen(app):
    """Create a splash screen while loading."""
    # Create a simple colored splash screen
    splash_pix = QPixmap(400, 300)
    splash_pix.fill(QColor(COLORS['bg_dark']))

    splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)

    # Set splash screen message
    splash.setFont(QFont('Arial', 14, QFont.Weight.Bold))
    splash.showMessage(
        "🤖 Loading Trading Bot Dashboard...\n\nPlease wait...",
        Qt.AlignmentFlag.AlignCenter,
        QColor(COLORS['text_primary'])
    )

    splash.show()
    app.processEvents()

    return splash


def setup_application_style(app):
    """Setup application-wide style and palette."""
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Dark palette (backup for widgets that don't use stylesheet)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS['bg_dark']))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLORS['bg_medium']))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS['bg_light']))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(COLORS['bg_light']))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS['bg_medium']))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Link, QColor(COLORS['accent_blue']))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS['accent_blue']))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor('#FFFFFF'))

    app.setPalette(palette)


def main():
    """Main application function."""
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Autonomous Trading Bot Dashboard")
    app.setOrganizationName("Trading Bot")
    app.setApplicationVersion("1.0.0")

    # Setup style
    setup_application_style(app)

    # Show splash screen
    splash = create_splash_screen(app)

    # Create main window (this might take a moment)
    try:
        main_window = MainWindow()

        # Close splash and show main window
        QTimer.singleShot(1500, lambda: (splash.close(), main_window.show()))

    except Exception as e:
        splash.close()
        print(f"Error starting application: {e}")
        sys.exit(1)

    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
