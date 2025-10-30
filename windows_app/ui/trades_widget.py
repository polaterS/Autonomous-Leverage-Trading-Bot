"""
Trade History Widget - View all completed trades.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
                              QTableWidgetItem, QPushButton, QLabel, QHeaderView)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor
from ..controllers.db_controller import DatabaseController
from .styles import get_color
from datetime import datetime


class TradesWidget(QWidget):
    """Trade history table with statistics."""

    def __init__(self, db_controller: DatabaseController, parent=None):
        super().__init__(parent)
        self.db = db_controller

        self.init_ui()

        # Auto-refresh timer - will be started by MainWindow
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)

        # Initial load (delayed - will load when tab is activated)
        # self.refresh_data()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()

        # Controls
        controls = QHBoxLayout()

        self.stats_label = QLabel("Total Trades: 0 | Winners: 0 | Losers: 0 | Win Rate: 0%")
        self.stats_label.setObjectName("subtitleLabel")

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_data)

        controls.addWidget(self.stats_label)
        controls.addStretch()
        controls.addWidget(refresh_btn)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            'Date/Time', 'Symbol', 'Side', 'Leverage', 'Entry', 'Exit',
            'Quantity', 'P&L $', 'P&L %', 'Duration'
        ])

        # Table settings
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        # Auto-resize columns
        header = self.table.horizontalHeader()
        for i in range(10):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

        layout.addLayout(controls)
        layout.addWidget(self.table)

        self.setLayout(layout)

    def refresh_data(self):
        """Refresh trade history."""
        try:
            trades = self.db.get_trade_history(100)
            stats = self.db.get_statistics()

            # Update stats
            self.stats_label.setText(
                f"Total Trades: {stats['total_trades']} | "
                f"Winners: {stats['winners']} | "
                f"Losers: {stats['losers']} | "
                f"Win Rate: {stats['win_rate']:.1f}%"
            )

            # Update table
            self.table.setRowCount(len(trades))

            for row, trade in enumerate(trades):
                # Date/Time
                exit_time = trade.get('exit_time', datetime.now())
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                self.table.setItem(row, 0, QTableWidgetItem(exit_time.strftime('%Y-%m-%d %H:%M')))

                # Symbol
                self.table.setItem(row, 1, QTableWidgetItem(trade['symbol']))

                # Side
                side_item = QTableWidgetItem(trade['side'])
                side_color = get_color('accent_green') if trade['side'] == 'LONG' else get_color('accent_red')
                side_item.setForeground(QColor(side_color))
                self.table.setItem(row, 2, side_item)

                # Leverage
                self.table.setItem(row, 3, QTableWidgetItem(f"{trade['leverage']}x"))

                # Entry/Exit prices
                self.table.setItem(row, 4, QTableWidgetItem(f"${float(trade['entry_price']):.4f}"))
                self.table.setItem(row, 5, QTableWidgetItem(f"${float(trade['exit_price']):.4f}"))

                # Quantity
                self.table.setItem(row, 6, QTableWidgetItem(f"{float(trade['quantity']):.6f}"))

                # P&L $
                pnl = float(trade['realized_pnl_usd'])
                pnl_item = QTableWidgetItem(f"${pnl:+.2f}")
                pnl_color = get_color('accent_green') if pnl >= 0 else get_color('accent_red')
                pnl_item.setForeground(QColor(pnl_color))
                self.table.setItem(row, 7, pnl_item)

                # P&L %
                pnl_pct = float(trade.get('pnl_percent', 0))
                pnl_pct_item = QTableWidgetItem(f"{pnl_pct:+.2f}%")
                pnl_pct_item.setForeground(QColor(pnl_color))
                self.table.setItem(row, 8, pnl_pct_item)

                # Duration
                duration_sec = trade.get('trade_duration_seconds', 0)
                duration_str = self.format_duration(duration_sec)
                self.table.setItem(row, 9, QTableWidgetItem(duration_str))

        except Exception as e:
            print(f"Error refreshing trades: {e}")

    @staticmethod
    def format_duration(seconds: int) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
