"""
Trading Configuration Model - Data validation and business logic.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional
from datetime import datetime


@dataclass
class TradingConfig:
    """Trading configuration model with validation."""

    id: int = 1
    initial_capital: Decimal = Decimal("100.00")
    current_capital: Decimal = Decimal("100.00")
    position_size_percent: Decimal = Decimal("0.80")
    min_stop_loss_percent: Decimal = Decimal("0.05")
    max_stop_loss_percent: Decimal = Decimal("0.10")
    min_profit_usd: Decimal = Decimal("2.50")
    max_leverage: int = 5
    min_ai_confidence: Decimal = Decimal("0.60")
    daily_loss_limit_percent: Decimal = Decimal("0.10")
    max_consecutive_losses: int = 3
    is_trading_enabled: bool = True
    last_updated: Optional[datetime] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid

        Returns:
            True if all validations pass
        """
        # Capital validation
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")

        if self.current_capital < 0:
            raise ValueError("Current capital cannot be negative")

        # Leverage validation
        if not 1 <= self.max_leverage <= 20:
            raise ValueError("Max leverage must be between 1 and 20")

        # Position size validation
        if not 0.01 <= self.position_size_percent <= 1.0:
            raise ValueError("Position size percent must be between 0.01 and 1.0")

        # Stop-loss validation
        if not 0.01 <= self.min_stop_loss_percent <= 0.5:
            raise ValueError("Min stop-loss must be between 0.01 and 0.5")

        if not 0.01 <= self.max_stop_loss_percent <= 0.5:
            raise ValueError("Max stop-loss must be between 0.01 and 0.5")

        if self.min_stop_loss_percent > self.max_stop_loss_percent:
            raise ValueError("Min stop-loss cannot exceed max stop-loss")

        # Profit validation
        if self.min_profit_usd < 0:
            raise ValueError("Min profit USD must be non-negative")

        # AI confidence validation
        if not 0.5 <= self.min_ai_confidence <= 1.0:
            raise ValueError("Min AI confidence must be between 0.5 and 1.0")

        # Risk management validation
        if not 0.01 <= self.daily_loss_limit_percent <= 0.5:
            raise ValueError("Daily loss limit must be between 0.01 and 0.5")

        if not 1 <= self.max_consecutive_losses <= 20:
            raise ValueError("Max consecutive losses must be between 1 and 20")

        return True

    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            'id': self.id,
            'initial_capital': float(self.initial_capital),
            'current_capital': float(self.current_capital),
            'position_size_percent': float(self.position_size_percent),
            'min_stop_loss_percent': float(self.min_stop_loss_percent),
            'max_stop_loss_percent': float(self.max_stop_loss_percent),
            'min_profit_usd': float(self.min_profit_usd),
            'max_leverage': self.max_leverage,
            'min_ai_confidence': float(self.min_ai_confidence),
            'daily_loss_limit_percent': float(self.daily_loss_limit_percent),
            'max_consecutive_losses': self.max_consecutive_losses,
            'is_trading_enabled': self.is_trading_enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TradingConfig':
        """Create instance from dictionary."""
        return cls(
            id=data.get('id', 1),
            initial_capital=Decimal(str(data.get('initial_capital', 100.00))),
            current_capital=Decimal(str(data.get('current_capital', 100.00))),
            position_size_percent=Decimal(str(data.get('position_size_percent', 0.80))),
            min_stop_loss_percent=Decimal(str(data.get('min_stop_loss_percent', 0.05))),
            max_stop_loss_percent=Decimal(str(data.get('max_stop_loss_percent', 0.10))),
            min_profit_usd=Decimal(str(data.get('min_profit_usd', 2.50))),
            max_leverage=data.get('max_leverage', 5),
            min_ai_confidence=Decimal(str(data.get('min_ai_confidence', 0.60))),
            daily_loss_limit_percent=Decimal(str(data.get('daily_loss_limit_percent', 0.10))),
            max_consecutive_losses=data.get('max_consecutive_losses', 3),
            is_trading_enabled=data.get('is_trading_enabled', True),
            last_updated=data.get('last_updated'),
        )

    def get_risk_level(self) -> str:
        """
        Determine risk level based on configuration.

        Returns:
            'LOW', 'MEDIUM', or 'HIGH'
        """
        risk_score = 0

        # Leverage contribution
        if self.max_leverage <= 3:
            risk_score += 1
        elif self.max_leverage <= 5:
            risk_score += 2
        else:
            risk_score += 3

        # Position size contribution
        if self.position_size_percent <= Decimal("0.5"):
            risk_score += 1
        elif self.position_size_percent <= Decimal("0.8"):
            risk_score += 2
        else:
            risk_score += 3

        # Stop-loss contribution (wider = riskier)
        avg_sl = (self.min_stop_loss_percent + self.max_stop_loss_percent) / 2
        if avg_sl <= Decimal("0.05"):
            risk_score += 1
        elif avg_sl <= Decimal("0.10"):
            risk_score += 2
        else:
            risk_score += 3

        # Determine level
        if risk_score <= 4:
            return "LOW"
        elif risk_score <= 7:
            return "MEDIUM"
        else:
            return "HIGH"

    def calculate_max_loss_per_trade(self) -> Decimal:
        """Calculate maximum potential loss per trade."""
        position_value = self.current_capital * self.position_size_percent
        max_loss_percent = self.max_stop_loss_percent
        leveraged_loss = position_value * max_loss_percent * self.max_leverage
        return leveraged_loss

    def is_safe_to_trade(self) -> tuple[bool, str]:
        """
        Check if it's safe to trade with current settings.

        Returns:
            (is_safe, reason)
        """
        if not self.is_trading_enabled:
            return False, "Trading is disabled"

        if self.current_capital <= 0:
            return False, "Insufficient capital"

        if self.current_capital < 10:
            return False, "Capital too low (minimum $10)"

        max_loss = self.calculate_max_loss_per_trade()
        if max_loss > self.current_capital * Decimal("0.20"):
            return False, f"Max loss per trade too high: ${float(max_loss):.2f}"

        return True, "Safe to trade"
