"""
Main Window - Trading Bot Dashboard Application.
"""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget,
                              QStatusBar, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QAction
from ..controllers.db_controller import DatabaseController
from ..controllers.bot_controller import BotController
from ..controllers.log_controller import LogController
from .dashboard_widget import DashboardWidget
from .logs_widget import LogsWidget
from .trades_widget import TradesWidget
from .charts_widget import ChartsWidget
from .config_widget import ConfigWidget
from .styles import MAIN_STYLE
import sys


class MainWindow(QMainWindow):
    """Main application window with tabbed interface."""

    def __init__(self):
        super().__init__()

        # Initialize controllers
        self.db = DatabaseController()
        self.bot = BotController()
        self.log = LogController()

        # Connect to database
        if not self.db.connect():
            QMessageBox.critical(
                self,
                "Database Error",
                "Failed to connect to database!\n\n"
                "Please ensure PostgreSQL is running and DATABASE_URL is correct."
            )
            sys.exit(1)

        self.init_ui()

        # Status bar timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_bar)
        self.status_timer.start(2000)  # Update every 2 seconds

        # Active tab tracking for performance
        self.current_tab_index = 0
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Start dashboard timer (first tab is active by default)
        self.dashboard_tab.refresh_timer.start(2000)

    def init_ui(self):
        """Initialize the main window UI."""
        self.setWindowTitle("🤖 Autonomous Trading Bot Dashboard")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)

        # Apply stylesheet
        self.setStyleSheet(MAIN_STYLE)

        # Create menu bar
        self.create_menu_bar()

        # Central widget with tabs
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Add tabs
        self.dashboard_tab = DashboardWidget(self.db, self.bot)
        self.logs_tab = LogsWidget(self.log)
        self.trades_tab = TradesWidget(self.db)
        self.charts_tab = ChartsWidget(self.db)
        self.config_tab = ConfigWidget(self.db)

        self.tabs.addTab(self.dashboard_tab, "📊 Dashboard")
        self.tabs.addTab(self.trades_tab, "📈 Trades")
        self.tabs.addTab(self.charts_tab, "📉 Charts")
        self.tabs.addTab(self.logs_tab, "📝 Logs")
        self.tabs.addTab(self.config_tab, "⚙️ Settings")

        main_layout.addWidget(self.tabs)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def create_menu_bar(self):
        """Create application menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        refresh_action = QAction("🔄 Refresh All", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_all)
        file_menu.addAction(refresh_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Bot menu
        bot_menu = menu_bar.addMenu("Bot")

        start_action = QAction("▶ Start Bot", self)
        start_action.triggered.connect(lambda: self.bot.start_bot())
        bot_menu.addAction(start_action)

        stop_action = QAction("■ Stop Bot", self)
        stop_action.triggered.connect(lambda: self.bot.stop_bot())
        bot_menu.addAction(stop_action)

        restart_action = QAction("↻ Restart Bot", self)
        restart_action.triggered.connect(lambda: self.bot.restart_bot())
        bot_menu.addAction(restart_action)

        # Help menu
        help_menu = menu_bar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)

    def on_tab_changed(self, index):
        """Handle tab change - stop inactive tab timers for performance."""
        self.current_tab_index = index

        # Stop all widget timers first
        self.dashboard_tab.refresh_timer.stop()
        self.trades_tab.refresh_timer.stop()
        self.charts_tab.refresh_timer.stop()
        self.logs_tab.refresh_timer.stop()

        # Start only active tab timer
        if index == 0:  # Dashboard
            self.dashboard_tab.refresh_timer.start(2000)
        elif index == 1:  # Trades
            self.trades_tab.refresh_timer.start(5000)
            self.trades_tab.refresh_data()  # Refresh immediately
        elif index == 2:  # Charts
            self.charts_tab.refresh_timer.start(30000)
            self.charts_tab.refresh_data()  # Refresh immediately
        elif index == 3:  # Logs
            self.logs_tab.refresh_timer.start(1000)
            # Load recent logs immediately
            if not self.logs_tab.log_display.toPlainText():
                self.logs_tab.load_recent_logs()

    def update_status_bar(self):
        """Update status bar with current information."""
        try:
            bot_status = self.bot.get_bot_status()
            capital = self.db.get_current_capital()
            daily_pnl = self.db.get_daily_pnl()

            status_text = (
                f"Bot: {'🟢 RUNNING' if bot_status['running'] else '🔴 STOPPED'} | "
                f"Capital: ${capital:,.2f} | "
                f"Today P&L: ${daily_pnl:+,.2f}"
            )

            self.status_bar.showMessage(status_text)

        except Exception as e:
            self.status_bar.showMessage(f"Error updating status: {e}")

    def refresh_all(self):
        """Refresh all widgets."""
        try:
            self.dashboard_tab.refresh_data()
            self.trades_tab.refresh_data()
            self.charts_tab.refresh_data()
            self.status_bar.showMessage("✅ All data refreshed", 3000)
        except Exception as e:
            self.status_bar.showMessage(f"Error refreshing: {e}", 5000)

    def show_about(self):
        """Show about dialog."""
        about_text = """
        <h2>Autonomous Trading Bot Dashboard</h2>
        <p><b>Version:</b> 1.0.0</p>
        <p><b>Description:</b> Professional trading bot monitoring and control interface.</p>
        <br>
        <p><b>Features:</b></p>
        <ul>
            <li>Real-time position monitoring</li>
            <li>Live trade history</li>
            <li>Performance charts</li>
            <li>Configuration management</li>
            <li>Bot process control</li>
        </ul>
        <br>
        <p><b>⚠️ Warning:</b> Leverage trading is extremely risky. Only trade with money you can afford to lose.</p>
        """

        QMessageBox.about(self, "About", about_text)

    def show_documentation(self):
        """Show documentation link."""
        QMessageBox.information(
            self,
            "Documentation",
            "📚 Documentation\n\n"
            "Please refer to README.md and UPGRADE_GUIDE.md in the project directory.\n\n"
            "For detailed instructions on using the bot and this GUI application."
        )

    def closeEvent(self, event):
        """Handle window close event."""
        if self.bot.is_running():
            reply = QMessageBox.question(
                self,
                'Exit Application',
                'The trading bot is still running!\n\n'
                'Stop the bot before exiting?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.bot.stop_bot()
                self.db.disconnect()
                event.accept()
            elif reply == QMessageBox.StandardButton.No:
                self.db.disconnect()
                event.accept()
            else:
                event.ignore()
        else:
            self.db.disconnect()
            event.accept()
