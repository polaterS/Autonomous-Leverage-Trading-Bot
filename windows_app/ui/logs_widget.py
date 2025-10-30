"""
Logs Viewer Widget - Real-time log monitoring with filtering.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                              QPushButton, QComboBox, QLabel, QLineEdit)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QTextCursor, QColor
from ..controllers.log_controller import LogController
from .styles import get_color


class LogsWidget(QWidget):
    """Real-time logs viewer with filtering capabilities."""

    def __init__(self, log_controller: LogController, parent=None):
        super().__init__(parent)
        self.log_controller = log_controller

        self.init_ui()

        # Auto-refresh timer (every 1 second) - will be started by MainWindow
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_logs)

        # Initial load (delayed - will load when tab is activated)
        # self.load_recent_logs()

    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()

        # === Controls ===
        controls_layout = QHBoxLayout()

        # Filter by level
        level_label = QLabel("Level:")
        self.level_combo = QComboBox()
        self.level_combo.addItems(['ALL', 'INFO', 'WARNING', 'ERROR', 'DEBUG'])
        self.level_combo.currentTextChanged.connect(self.filter_changed)

        # Search
        search_label = QLabel("Search:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search logs...")
        self.search_input.textChanged.connect(self.filter_changed)

        # Clear button
        clear_btn = QPushButton("🗑️ Clear Logs")
        clear_btn.clicked.connect(self.clear_logs)

        # Auto-scroll checkbox
        self.auto_scroll_btn = QPushButton("📜 Auto-scroll: ON")
        self.auto_scroll_btn.setCheckable(True)
        self.auto_scroll_btn.setChecked(True)
        self.auto_scroll_btn.clicked.connect(self.toggle_auto_scroll)

        controls_layout.addWidget(level_label)
        controls_layout.addWidget(self.level_combo)
        controls_layout.addWidget(search_label)
        controls_layout.addWidget(self.search_input)
        controls_layout.addStretch()
        controls_layout.addWidget(self.auto_scroll_btn)
        controls_layout.addWidget(clear_btn)

        # === Log Display ===
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFontFamily("Consolas, Monaco, monospace")
        self.log_display.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

        layout.addLayout(controls_layout)
        layout.addWidget(self.log_display)

        self.setLayout(layout)

        self.auto_scroll = True

    def load_recent_logs(self):
        """Load recent logs on startup."""
        logs = self.log_controller.get_recent_logs(500)
        self.log_display.clear()

        for log in logs:
            self.append_log_line(log)

    def refresh_logs(self):
        """Refresh logs (get new lines)."""
        new_logs = self.log_controller.get_new_logs()

        for log in new_logs:
            self.append_log_line(log)

    def append_log_line(self, log_line: str):
        """Append a log line with appropriate color."""
        # Parse log to get level
        parsed = self.log_controller.parse_log_line(log_line)
        level = parsed['level']

        # Apply filter
        selected_level = self.level_combo.currentText()
        search_text = self.search_input.text().lower()

        if selected_level != 'ALL' and level != selected_level:
            return

        if search_text and search_text not in log_line.lower():
            return

        # Color based on level
        color_map = {
            'INFO': get_color('text_primary'),
            'WARNING': get_color('accent_yellow'),
            'ERROR': get_color('accent_red'),
            'CRITICAL': get_color('accent_red'),
            'DEBUG': get_color('text_secondary'),
        }

        color = color_map.get(level, get_color('text_primary'))

        # Insert with color
        self.log_display.setTextColor(QColor(color))
        self.log_display.append(log_line)

        # Auto-scroll to bottom
        if self.auto_scroll:
            cursor = self.log_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.log_display.setTextCursor(cursor)

    def filter_changed(self):
        """Re-apply filters when changed."""
        # Reload all logs with new filter
        self.log_display.clear()
        self.load_recent_logs()

    def clear_logs(self):
        """Clear log display and file."""
        reply = QMessageBox.question(
            self,
            'Clear Logs',
            'Are you sure you want to clear all logs?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.log_controller.clear_log_file()
            self.log_display.clear()

    def toggle_auto_scroll(self):
        """Toggle auto-scroll feature."""
        self.auto_scroll = not self.auto_scroll
        if self.auto_scroll:
            self.auto_scroll_btn.setText("📜 Auto-scroll: ON")
        else:
            self.auto_scroll_btn.setText("📜 Auto-scroll: OFF")


# Import for message box
from PyQt6.QtWidgets import QMessageBox
