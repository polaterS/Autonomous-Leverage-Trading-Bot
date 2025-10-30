"""
Charts Widget - Performance visualization with PyQt6 Charts.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis, QBarSeries, QBarSet
from PyQt6.QtCore import QTimer, Qt, QDateTime
from PyQt6.QtGui import QPainter, QColor
from ..controllers.db_controller import DatabaseController
from .styles import get_color, COLORS


class ChartsWidget(QWidget):
    """Performance charts - P&L over time, win/loss distribution."""

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

        title = QLabel("Performance Charts")
        title.setObjectName("titleLabel")

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_data)

        controls.addWidget(title)
        controls.addStretch()
        controls.addWidget(refresh_btn)

        # === P&L Chart ===
        self.pnl_chart = self.create_pnl_chart()
        self.pnl_chart_view = QChartView(self.pnl_chart)
        self.pnl_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout.addLayout(controls)
        layout.addWidget(self.pnl_chart_view)

        self.setLayout(layout)

    def create_pnl_chart(self) -> QChart:
        """Create P&L over time line chart."""
        chart = QChart()
        chart.setTitle("Capital Over Time (Last 30 Days)")
        chart.setTheme(QChart.ChartTheme.ChartThemeDark)
        chart.setBackgroundBrush(QColor(COLORS['bg_dark']))

        # Configure chart appearance
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

        return chart

    def refresh_data(self):
        """Refresh chart data."""
        try:
            # Get P&L history
            pnl_history = self.db.get_pnl_history(30)

            if not pnl_history:
                return

            # Clear existing series
            self.pnl_chart.removeAllSeries()

            # Create series
            series = QLineSeries()
            series.setName("Capital")

            # Add data points
            for entry in reversed(pnl_history):  # Reverse to show oldest first
                date_obj = entry['date']
                capital = float(entry['ending_capital'])

                # Convert date to QDateTime
                qdatetime = QDateTime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0)
                series.append(qdatetime.toMSecsSinceEpoch(), capital)

            # Add series to chart
            self.pnl_chart.addSeries(series)

            # Create axes
            axis_x = QDateTimeAxis()
            axis_x.setFormat("MM/dd")
            axis_x.setTitleText("Date")
            axis_x.setLabelsColor(QColor(COLORS['text_primary']))

            axis_y = QValueAxis()
            axis_y.setTitleText("Capital ($)")
            axis_y.setLabelsColor(QColor(COLORS['text_primary']))

            # Attach axes
            self.pnl_chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
            self.pnl_chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)

            series.attachAxis(axis_x)
            series.attachAxis(axis_y)

        except Exception as e:
            print(f"Error refreshing charts: {e}")
