"""
Dashboard Widget - Real-time trading metrics and position monitoring.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QGridLayout, QPushButton, QGroupBox, QMessageBox)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from ..controllers.db_controller import DatabaseController
from ..controllers.bot_controller import BotController
from .styles import get_color
from datetime import datetime


class MetricCard(QFrame):
    """Reusable metric card component."""

    def __init__(self, title: str, value: str = "0.00", color: str = 'text_primary', parent=None):
        super().__init__(parent)
        self.setObjectName("card")

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title_label = QLabel(title)
        title_label.setObjectName("subtitleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Value
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricLabel")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setStyleSheet(f"color: {get_color(color)};")

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        self.setLayout(layout)

    def update_value(self, value: str, color: str = None):
        """Update the card value."""
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"color: {get_color(color)};")


class DashboardWidget(QWidget):
    """Main dashboard widget with real-time metrics."""

    def __init__(self, db_controller: DatabaseController, bot_controller: BotController, parent=None):
        super().__init__(parent)
        self.db = db_controller
        self.bot = bot_controller

        self.init_ui()

        # Auto-refresh timer (every 2 seconds)
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(2000)  # 2 seconds

        # Initial load
        self.refresh_data()

    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(20)

        # === Bot Status Section ===
        status_group = self.create_bot_status_section()
        layout.addWidget(status_group)

        # === Metrics Grid ===
        metrics_grid = self.create_metrics_grid()
        layout.addLayout(metrics_grid)

        # === Active Position Section ===
        self.position_group = self.create_position_section()
        layout.addWidget(self.position_group)

        # === Statistics Section ===
        stats_group = self.create_statistics_section()
        layout.addWidget(stats_group)

        layout.addStretch()
        self.setLayout(layout)

    def create_bot_status_section(self) -> QGroupBox:
        """Create bot status and control section."""
        group = QGroupBox("Bot Status & Control")
        layout = QHBoxLayout()

        # Status indicator
        self.status_label = QLabel("● OFFLINE")
        self.status_label.setObjectName("titleLabel")
        self.status_label.setStyleSheet(f"color: {get_color('accent_red')};")

        # Bot controls
        self.start_btn = QPushButton("▶ Start Bot")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.clicked.connect(self.start_bot)

        self.stop_btn = QPushButton("■ Stop Bot")
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.clicked.connect(self.stop_bot)
        self.stop_btn.setEnabled(False)

        self.restart_btn = QPushButton("↻ Restart")
        self.restart_btn.clicked.connect(self.restart_bot)

        # Bot info
        self.bot_info_label = QLabel("PID: N/A | CPU: 0% | Memory: 0 MB")
        self.bot_info_label.setObjectName("subtitleLabel")

        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.bot_info_label)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.restart_btn)

        group.setLayout(layout)
        return group

    def create_metrics_grid(self) -> QGridLayout:
        """Create main metrics grid."""
        grid = QGridLayout()
        grid.setSpacing(15)

        # Row 1: Capital and P&L
        self.capital_card = MetricCard("Current Capital", "$0.00", 'accent_blue')
        self.daily_pnl_card = MetricCard("Today's P&L", "$0.00", 'accent_green')
        self.total_pnl_card = MetricCard("Total P&L", "$0.00", 'accent_green')

        grid.addWidget(self.capital_card, 0, 0)
        grid.addWidget(self.daily_pnl_card, 0, 1)
        grid.addWidget(self.total_pnl_card, 0, 2)

        # Row 2: Trading Stats
        self.win_rate_card = MetricCard("Win Rate", "0%", 'accent_purple')
        self.total_trades_card = MetricCard("Total Trades", "0", 'text_primary')
        self.avg_win_card = MetricCard("Avg Win", "$0.00", 'accent_green')

        grid.addWidget(self.win_rate_card, 1, 0)
        grid.addWidget(self.total_trades_card, 1, 1)
        grid.addWidget(self.avg_win_card, 1, 2)

        return grid

    def create_position_section(self) -> QGroupBox:
        """Create active position monitoring section."""
        group = QGroupBox("Active Position")
        layout = QVBoxLayout()

        # Position status
        self.position_status = QLabel("No active position")
        self.position_status.setObjectName("subtitleLabel")
        self.position_status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Position details grid
        self.position_grid = QGridLayout()
        self.position_grid.setSpacing(10)

        # Labels will be populated when position is active
        self.position_labels = {}

        layout.addWidget(self.position_status)
        layout.addLayout(self.position_grid)

        group.setLayout(layout)
        return group

    def create_statistics_section(self) -> QGroupBox:
        """Create detailed statistics section."""
        group = QGroupBox("Performance Statistics")
        layout = QGridLayout()
        layout.setSpacing(10)

        # Create stat labels
        labels = [
            ("Winners:", "winners_label"),
            ("Losers:", "losers_label"),
            ("Avg Loss:", "avg_loss_label"),
            ("Best Trade:", "best_trade_label")
        ]

        for i, (title, key) in enumerate(labels):
            title_label = QLabel(title)
            title_label.setObjectName("subtitleLabel")

            value_label = QLabel("0")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

            self.position_labels[key] = value_label

            layout.addWidget(title_label, i // 2, (i % 2) * 2)
            layout.addWidget(value_label, i // 2, (i % 2) * 2 + 1)

        group.setLayout(layout)
        return group

    def refresh_data(self):
        """Refresh all dashboard data."""
        try:
            # Update bot status
            bot_status = self.bot.get_bot_status()
            if bot_status['running']:
                self.status_label.setText("● ONLINE")
                self.status_label.setStyleSheet(f"color: {get_color('accent_green')};")
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)

                self.bot_info_label.setText(
                    f"PID: {bot_status['pid']} | "
                    f"CPU: {bot_status['cpu_percent']:.1f}% | "
                    f"Memory: {bot_status['memory_mb']:.1f} MB"
                )
            else:
                self.status_label.setText("● OFFLINE")
                self.status_label.setStyleSheet(f"color: {get_color('accent_red')};")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.bot_info_label.setText("PID: N/A | CPU: 0% | Memory: 0 MB")

            # Update capital
            capital = self.db.get_current_capital()
            self.capital_card.update_value(f"${capital:,.2f}", 'accent_blue')

            # Update daily P&L
            daily_pnl = self.db.get_daily_pnl()
            pnl_color = 'accent_green' if daily_pnl >= 0 else 'accent_red'
            self.daily_pnl_card.update_value(f"${daily_pnl:+,.2f}", pnl_color)

            # Update statistics
            stats = self.db.get_statistics()
            self.total_pnl_card.update_value(
                f"${stats['total_pnl']:+,.2f}",
                'accent_green' if stats['total_pnl'] >= 0 else 'accent_red'
            )
            self.win_rate_card.update_value(f"{stats['win_rate']:.1f}%", 'accent_purple')
            self.total_trades_card.update_value(str(stats['total_trades']), 'text_primary')
            self.avg_win_card.update_value(f"${stats['avg_win']:.2f}", 'accent_green')

            # Update stats labels
            if 'winners_label' in self.position_labels:
                self.position_labels['winners_label'].setText(str(stats['winners']))
                self.position_labels['losers_label'].setText(str(stats['losers']))
                self.position_labels['avg_loss_label'].setText(f"${stats['avg_loss']:.2f}")

            # Update active position
            self.update_active_position()

        except Exception as e:
            print(f"Error refreshing dashboard: {e}")

    def update_active_position(self):
        """Update active position display."""
        position = self.db.get_active_position()

        if position:
            self.position_status.setText(f"🔵 {position['symbol']} {position['side']} {position['leverage']}x")

            # Clear existing grid
            for i in reversed(range(self.position_grid.count())):
                self.position_grid.itemAt(i).widget().setParent(None)

            # Position details
            entry_price = float(position['entry_price'])
            current_price = float(position.get('current_price', entry_price))
            unrealized_pnl = float(position.get('unrealized_pnl_usd', 0))

            details = [
                ("Entry:", f"${entry_price:.4f}"),
                ("Current:", f"${current_price:.4f}"),
                ("Quantity:", f"{float(position['quantity']):.6f}"),
                ("P&L:", f"${unrealized_pnl:+.2f}"),
                ("Stop-Loss:", f"${float(position['stop_loss_price']):.4f}"),
                ("Liquidation:", f"${float(position['liquidation_price']):.4f}"),
            ]

            for i, (label_text, value_text) in enumerate(details):
                label = QLabel(label_text)
                label.setObjectName("subtitleLabel")

                value = QLabel(value_text)
                value.setAlignment(Qt.AlignmentFlag.AlignRight)

                # Color P&L
                if label_text == "P&L:":
                    color = 'accent_green' if unrealized_pnl >= 0 else 'accent_red'
                    value.setStyleSheet(f"color: {get_color(color)}; font-weight: bold;")

                self.position_grid.addWidget(label, i // 2, (i % 2) * 2)
                self.position_grid.addWidget(value, i // 2, (i % 2) * 2 + 1)
        else:
            self.position_status.setText("No active position")

            # Clear grid
            for i in reversed(range(self.position_grid.count())):
                self.position_grid.itemAt(i).widget().setParent(None)

    def start_bot(self):
        """Start the trading bot."""
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Starting...")

        try:
            if self.bot.start_bot():
                QMessageBox.information(
                    self,
                    "Success",
                    "✅ Bot started successfully!\n\nThe trading bot is now running."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Failed",
                    "❌ Failed to start bot.\n\nPlease check:\n- Docker is running\n- Database is accessible\n- No existing bot instance"
                )
        finally:
            self.start_btn.setText("▶ Start Bot")
            self.refresh_data()

    def stop_bot(self):
        """Stop the trading bot."""
        # Confirm stop
        reply = QMessageBox.question(
            self,
            "Confirm Stop",
            "Are you sure you want to stop the trading bot?\n\n⚠️ Any active positions will remain open.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping...")

            try:
                if self.bot.stop_bot():
                    QMessageBox.information(
                        self,
                        "Success",
                        "✅ Bot stopped successfully!"
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Failed",
                        "❌ Failed to stop bot.\n\nPlease check the logs for details."
                    )
            finally:
                self.stop_btn.setText("■ Stop Bot")
                self.refresh_data()

    def restart_bot(self):
        """Restart the trading bot."""
        # Confirm restart
        reply = QMessageBox.question(
            self,
            "Confirm Restart",
            "Are you sure you want to restart the trading bot?\n\nThis will temporarily stop trading.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.restart_btn.setEnabled(False)
            self.restart_btn.setText("Restarting...")

            try:
                if self.bot.restart_bot():
                    QMessageBox.information(
                        self,
                        "Success",
                        "✅ Bot restarted successfully!"
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Failed",
                        "❌ Failed to restart bot.\n\nPlease try stopping and starting manually."
                    )
            finally:
                self.restart_btn.setText("↻ Restart")
                self.restart_btn.setEnabled(True)
                self.refresh_data()
