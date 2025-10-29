"""
Configuration Widget - Manage bot settings.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                              QPushButton, QGroupBox, QDoubleSpinBox, QSpinBox,
                              QCheckBox, QMessageBox, QLabel)
from PyQt6.QtCore import Qt
from ..controllers.db_controller import DatabaseController
from .styles import get_color


class ConfigWidget(QWidget):
    """Bot configuration panel."""

    def __init__(self, db_controller: DatabaseController, parent=None):
        super().__init__(parent)
        self.db = db_controller

        self.init_ui()
        self.load_config()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()

        # === Trading Parameters ===
        trading_group = QGroupBox("Trading Parameters")
        trading_layout = QFormLayout()

        self.max_leverage_spin = QSpinBox()
        self.max_leverage_spin.setRange(1, 10)
        self.max_leverage_spin.setValue(5)

        self.position_size_spin = QDoubleSpinBox()
        self.position_size_spin.setRange(0.1, 1.0)
        self.position_size_spin.setSingleStep(0.05)
        self.position_size_spin.setValue(0.80)
        self.position_size_spin.setSuffix(" (80%)")

        self.min_sl_spin = QDoubleSpinBox()
        self.min_sl_spin.setRange(0.01, 0.5)
        self.min_sl_spin.setSingleStep(0.01)
        self.min_sl_spin.setValue(0.05)
        self.min_sl_spin.setSuffix(" (5%)")

        self.max_sl_spin = QDoubleSpinBox()
        self.max_sl_spin.setRange(0.01, 0.5)
        self.max_sl_spin.setSingleStep(0.01)
        self.max_sl_spin.setValue(0.10)
        self.max_sl_spin.setSuffix(" (10%)")

        self.min_profit_spin = QDoubleSpinBox()
        self.min_profit_spin.setRange(0.5, 50.0)
        self.min_profit_spin.setSingleStep(0.5)
        self.min_profit_spin.setValue(2.50)
        self.min_profit_spin.setPrefix("$")

        self.min_confidence_spin = QDoubleSpinBox()
        self.min_confidence_spin.setRange(0.5, 1.0)
        self.min_confidence_spin.setSingleStep(0.05)
        self.min_confidence_spin.setValue(0.60)
        self.min_confidence_spin.setSuffix(" (60%)")

        trading_layout.addRow("Max Leverage:", self.max_leverage_spin)
        trading_layout.addRow("Position Size %:", self.position_size_spin)
        trading_layout.addRow("Min Stop-Loss %:", self.min_sl_spin)
        trading_layout.addRow("Max Stop-Loss %:", self.max_sl_spin)
        trading_layout.addRow("Min Profit USD:", self.min_profit_spin)
        trading_layout.addRow("Min AI Confidence:", self.min_confidence_spin)

        trading_group.setLayout(trading_layout)

        # === Risk Management ===
        risk_group = QGroupBox("Risk Management")
        risk_layout = QFormLayout()

        self.daily_loss_spin = QDoubleSpinBox()
        self.daily_loss_spin.setRange(0.01, 0.5)
        self.daily_loss_spin.setSingleStep(0.01)
        self.daily_loss_spin.setValue(0.10)
        self.daily_loss_spin.setSuffix(" (10%)")

        self.max_consecutive_spin = QSpinBox()
        self.max_consecutive_spin.setRange(1, 10)
        self.max_consecutive_spin.setValue(3)

        self.trading_enabled_check = QCheckBox("Trading Enabled")
        self.trading_enabled_check.setChecked(True)

        risk_layout.addRow("Daily Loss Limit %:", self.daily_loss_spin)
        risk_layout.addRow("Max Consecutive Losses:", self.max_consecutive_spin)
        risk_layout.addRow("Status:", self.trading_enabled_check)

        risk_group.setLayout(risk_layout)

        # === Buttons ===
        buttons_layout = QHBoxLayout()

        save_btn = QPushButton("💾 Save Configuration")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self.save_config)

        reset_btn = QPushButton("↻ Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)

        buttons_layout.addStretch()
        buttons_layout.addWidget(save_btn)
        buttons_layout.addWidget(reset_btn)

        # Assemble layout
        layout.addWidget(trading_group)
        layout.addWidget(risk_group)
        layout.addStretch()
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    def load_config(self):
        """Load configuration from database."""
        try:
            config = self.db.get_trading_config()
            if config:
                self.max_leverage_spin.setValue(config['max_leverage'])
                self.position_size_spin.setValue(float(config['position_size_percent']))
                self.min_sl_spin.setValue(float(config['min_stop_loss_percent']))
                self.max_sl_spin.setValue(float(config['max_stop_loss_percent']))
                self.min_profit_spin.setValue(float(config['min_profit_usd']))
                self.min_confidence_spin.setValue(float(config['min_ai_confidence']))
                self.daily_loss_spin.setValue(float(config['daily_loss_limit_percent']))
                self.max_consecutive_spin.setValue(config['max_consecutive_losses'])
                self.trading_enabled_check.setChecked(config['is_trading_enabled'])

        except Exception as e:
            print(f"Error loading config: {e}")

    def save_config(self):
        """Save configuration to database."""
        try:
            config_updates = {
                'max_leverage': self.max_leverage_spin.value(),
                'position_size_percent': self.position_size_spin.value(),
                'min_stop_loss_percent': self.min_sl_spin.value(),
                'max_stop_loss_percent': self.max_sl_spin.value(),
                'min_profit_usd': self.min_profit_spin.value(),
                'min_ai_confidence': self.min_confidence_spin.value(),
                'daily_loss_limit_percent': self.daily_loss_spin.value(),
                'max_consecutive_losses': self.max_consecutive_spin.value(),
                'is_trading_enabled': self.trading_enabled_check.isChecked()
            }

            if self.db.update_config(config_updates):
                QMessageBox.information(self, "Success", "Configuration saved successfully!")
            else:
                QMessageBox.warning(self, "Error", "Failed to save configuration.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving config: {e}")

    def reset_to_defaults(self):
        """Reset to default values."""
        reply = QMessageBox.question(
            self,
            'Reset Configuration',
            'Reset all settings to defaults?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.max_leverage_spin.setValue(5)
            self.position_size_spin.setValue(0.80)
            self.min_sl_spin.setValue(0.05)
            self.max_sl_spin.setValue(0.10)
            self.min_profit_spin.setValue(2.50)
            self.min_confidence_spin.setValue(0.60)
            self.daily_loss_spin.setValue(0.10)
            self.max_consecutive_spin.setValue(3)
            self.trading_enabled_check.setChecked(True)
