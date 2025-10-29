"""
Modern Dark Theme Styles for the Trading Bot Dashboard.
Professional, clean, and easy on the eyes for long trading sessions.
"""

# Main color palette
COLORS = {
    'bg_dark': '#1E1E2E',        # Main background
    'bg_medium': '#2A2A3E',      # Card background
    'bg_light': '#3A3A52',       # Hover states
    'text_primary': '#E0E0E0',   # Main text
    'text_secondary': '#A0A0A0', # Secondary text
    'accent_blue': '#5E81AC',    # Primary accent
    'accent_green': '#A3BE8C',   # Success/profit
    'accent_red': '#BF616A',     # Error/loss
    'accent_yellow': '#EBCB8B',  # Warning
    'accent_purple': '#B48EAD',  # Info
    'border': '#4C4C6E',         # Borders
}

# Main application stylesheet
MAIN_STYLE = f"""
    /* Global */
    QWidget {{
        background-color: {COLORS['bg_dark']};
        color: {COLORS['text_primary']};
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 10pt;
    }}

    /* Main Window */
    QMainWindow {{
        background-color: {COLORS['bg_dark']};
    }}

    /* Tab Widget */
    QTabWidget::pane {{
        border: 1px solid {COLORS['border']};
        background-color: {COLORS['bg_dark']};
        border-radius: 5px;
    }}

    QTabBar::tab {{
        background-color: {COLORS['bg_medium']};
        color: {COLORS['text_secondary']};
        padding: 10px 20px;
        margin-right: 2px;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
    }}

    QTabBar::tab:selected {{
        background-color: {COLORS['accent_blue']};
        color: {COLORS['text_primary']};
        font-weight: bold;
    }}

    QTabBar::tab:hover:!selected {{
        background-color: {COLORS['bg_light']};
    }}

    /* Push Buttons */
    QPushButton {{
        background-color: {COLORS['bg_medium']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: bold;
    }}

    QPushButton:hover {{
        background-color: {COLORS['bg_light']};
        border: 1px solid {COLORS['accent_blue']};
    }}

    QPushButton:pressed {{
        background-color: {COLORS['accent_blue']};
    }}

    QPushButton:disabled {{
        background-color: {COLORS['bg_medium']};
        color: {COLORS['text_secondary']};
        border: 1px solid {COLORS['border']};
    }}

    /* Primary Button (Start, Save, etc) */
    QPushButton#primaryButton {{
        background-color: {COLORS['accent_green']};
        color: #FFFFFF;
        border: none;
    }}

    QPushButton#primaryButton:hover {{
        background-color: #8FA776;
    }}

    /* Danger Button (Stop, Delete, etc) */
    QPushButton#dangerButton {{
        background-color: {COLORS['accent_red']};
        color: #FFFFFF;
        border: none;
    }}

    QPushButton#dangerButton:hover {{
        background-color: #A05259;
    }}

    /* Labels */
    QLabel {{
        background-color: transparent;
        color: {COLORS['text_primary']};
    }}

    QLabel#titleLabel {{
        font-size: 16pt;
        font-weight: bold;
        color: {COLORS['text_primary']};
    }}

    QLabel#subtitleLabel {{
        font-size: 11pt;
        color: {COLORS['text_secondary']};
    }}

    QLabel#metricLabel {{
        font-size: 24pt;
        font-weight: bold;
        color: {COLORS['accent_blue']};
    }}

    QLabel#profitLabel {{
        font-size: 20pt;
        font-weight: bold;
        color: {COLORS['accent_green']};
    }}

    QLabel#lossLabel {{
        font-size: 20pt;
        font-weight: bold;
        color: {COLORS['accent_red']};
    }}

    /* Line Edits */
    QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {COLORS['bg_medium']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 3px;
        padding: 5px;
    }}

    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border: 1px solid {COLORS['accent_blue']};
    }}

    /* Combo Box */
    QComboBox {{
        background-color: {COLORS['bg_medium']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 3px;
        padding: 5px;
    }}

    QComboBox:hover {{
        border: 1px solid {COLORS['accent_blue']};
    }}

    QComboBox::drop-down {{
        border: none;
    }}

    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid {COLORS['text_secondary']};
        margin-right: 5px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {COLORS['bg_medium']};
        color: {COLORS['text_primary']};
        selection-background-color: {COLORS['accent_blue']};
        border: 1px solid {COLORS['border']};
    }}

    /* Tables */
    QTableWidget {{
        background-color: {COLORS['bg_medium']};
        alternate-background-color: {COLORS['bg_dark']};
        color: {COLORS['text_primary']};
        gridline-color: {COLORS['border']};
        border: 1px solid {COLORS['border']};
        border-radius: 5px;
    }}

    QTableWidget::item {{
        padding: 5px;
    }}

    QTableWidget::item:selected {{
        background-color: {COLORS['accent_blue']};
    }}

    QHeaderView::section {{
        background-color: {COLORS['bg_light']};
        color: {COLORS['text_primary']};
        padding: 8px;
        border: none;
        border-right: 1px solid {COLORS['border']};
        border-bottom: 1px solid {COLORS['border']};
        font-weight: bold;
    }}

    /* Scroll Bars */
    QScrollBar:vertical {{
        background-color: {COLORS['bg_dark']};
        width: 12px;
        border-radius: 6px;
    }}

    QScrollBar::handle:vertical {{
        background-color: {COLORS['bg_light']};
        border-radius: 6px;
        min-height: 20px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {COLORS['accent_blue']};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    QScrollBar:horizontal {{
        background-color: {COLORS['bg_dark']};
        height: 12px;
        border-radius: 6px;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {COLORS['bg_light']};
        border-radius: 6px;
        min-width: 20px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background-color: {COLORS['accent_blue']};
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* Group Box */
    QGroupBox {{
        border: 1px solid {COLORS['border']};
        border-radius: 5px;
        margin-top: 10px;
        padding-top: 10px;
        font-weight: bold;
        color: {COLORS['text_primary']};
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        color: {COLORS['accent_blue']};
    }}

    /* Progress Bar */
    QProgressBar {{
        border: 1px solid {COLORS['border']};
        border-radius: 5px;
        text-align: center;
        background-color: {COLORS['bg_medium']};
        color: {COLORS['text_primary']};
    }}

    QProgressBar::chunk {{
        background-color: {COLORS['accent_green']};
        border-radius: 4px;
    }}

    /* Status Bar */
    QStatusBar {{
        background-color: {COLORS['bg_medium']};
        color: {COLORS['text_secondary']};
        border-top: 1px solid {COLORS['border']};
    }}

    /* Tool Tip */
    QToolTip {{
        background-color: {COLORS['bg_light']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 3px;
        padding: 5px;
    }}

    /* Check Box */
    QCheckBox {{
        color: {COLORS['text_primary']};
        spacing: 5px;
    }}

    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {COLORS['border']};
        border-radius: 3px;
        background-color: {COLORS['bg_medium']};
    }}

    QCheckBox::indicator:checked {{
        background-color: {COLORS['accent_green']};
        border: 1px solid {COLORS['accent_green']};
    }}

    QCheckBox::indicator:hover {{
        border: 1px solid {COLORS['accent_blue']};
    }}

    /* Radio Button */
    QRadioButton {{
        color: {COLORS['text_primary']};
        spacing: 5px;
    }}

    QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {COLORS['border']};
        border-radius: 9px;
        background-color: {COLORS['bg_medium']};
    }}

    QRadioButton::indicator:checked {{
        background-color: {COLORS['accent_green']};
        border: 1px solid {COLORS['accent_green']};
    }}

    /* Menu Bar */
    QMenuBar {{
        background-color: {COLORS['bg_medium']};
        color: {COLORS['text_primary']};
        border-bottom: 1px solid {COLORS['border']};
    }}

    QMenuBar::item {{
        padding: 5px 10px;
        background-color: transparent;
    }}

    QMenuBar::item:selected {{
        background-color: {COLORS['bg_light']};
    }}

    QMenu {{
        background-color: {COLORS['bg_medium']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
    }}

    QMenu::item {{
        padding: 5px 20px;
    }}

    QMenu::item:selected {{
        background-color: {COLORS['accent_blue']};
    }}

    /* Card-style Frame */
    QFrame#card {{
        background-color: {COLORS['bg_medium']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 15px;
    }}
"""

def get_color(color_name: str) -> str:
    """Get color from palette."""
    return COLORS.get(color_name, '#FFFFFF')
