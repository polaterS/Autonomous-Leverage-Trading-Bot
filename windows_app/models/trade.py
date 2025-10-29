"""
Trade Model - Individual trade data with analytics.
"""

from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    """Trade model with analytics methods."""

    id: int
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    leverage: int
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    position_value_usd: Decimal
    realized_pnl_usd: Decimal
    pnl_percent: Decimal
    stop_loss_percent: Optional[Decimal] = None
    close_reason: Optional[str] = None
    trade_duration_seconds: Optional[int] = None
    ai_model_consensus: Optional[str] = None
    ai_confidence: Optional[Decimal] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    is_winner: bool = False

    def __post_init__(self):
        """Calculate derived fields."""
        self.is_winner = self.realized_pnl_usd > 0

    @property
    def duration_str(self) -> str:
        """Format trade duration as human-readable string."""
        if not self.trade_duration_seconds:
            return "N/A"

        seconds = self.trade_duration_seconds

        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes}m"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"

    @property
    def pnl_color(self) -> str:
        """Get color for P&L display."""
        return "green" if self.is_winner else "red"

    @property
    def side_color(self) -> str:
        """Get color for side display."""
        return "green" if self.side == "LONG" else "red"

    def get_roi(self) -> Decimal:
        """
        Calculate Return on Investment.

        Returns:
            ROI as decimal (e.g., 0.05 = 5%)
        """
        if self.position_value_usd == 0:
            return Decimal("0")

        return self.realized_pnl_usd / self.position_value_usd

    def get_risk_reward_ratio(self) -> Optional[Decimal]:
        """
        Calculate risk/reward ratio.

        Returns:
            Risk/reward ratio or None if stop-loss not set
        """
        if not self.stop_loss_percent:
            return None

        # How much was risked vs how much was gained
        risk_amount = self.position_value_usd * self.stop_loss_percent
        reward_amount = abs(self.realized_pnl_usd)

        if risk_amount == 0:
            return None

        return reward_amount / risk_amount

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'leverage': self.leverage,
            'entry_price': float(self.entry_price),
            'exit_price': float(self.exit_price),
            'quantity': float(self.quantity),
            'position_value_usd': float(self.position_value_usd),
            'realized_pnl_usd': float(self.realized_pnl_usd),
            'pnl_percent': float(self.pnl_percent),
            'stop_loss_percent': float(self.stop_loss_percent) if self.stop_loss_percent else None,
            'close_reason': self.close_reason,
            'trade_duration_seconds': self.trade_duration_seconds,
            'ai_model_consensus': self.ai_model_consensus,
            'ai_confidence': float(self.ai_confidence) if self.ai_confidence else None,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'is_winner': self.is_winner,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Trade':
        """Create instance from dictionary."""
        return cls(
            id=data['id'],
            symbol=data['symbol'],
            side=data['side'],
            leverage=data['leverage'],
            entry_price=Decimal(str(data['entry_price'])),
            exit_price=Decimal(str(data['exit_price'])),
            quantity=Decimal(str(data['quantity'])),
            position_value_usd=Decimal(str(data['position_value_usd'])),
            realized_pnl_usd=Decimal(str(data['realized_pnl_usd'])),
            pnl_percent=Decimal(str(data['pnl_percent'])),
            stop_loss_percent=Decimal(str(data['stop_loss_percent'])) if data.get('stop_loss_percent') else None,
            close_reason=data.get('close_reason'),
            trade_duration_seconds=data.get('trade_duration_seconds'),
            ai_model_consensus=data.get('ai_model_consensus'),
            ai_confidence=Decimal(str(data['ai_confidence'])) if data.get('ai_confidence') else None,
            entry_time=data.get('entry_time'),
            exit_time=data.get('exit_time'),
            is_winner=data.get('is_winner', False),
        )

    def get_summary(self) -> str:
        """Get trade summary string."""
        return (
            f"{self.symbol} {self.side} {self.leverage}x: "
            f"${float(self.realized_pnl_usd):+.2f} "
            f"({float(self.pnl_percent):+.2f}%) "
            f"in {self.duration_str}"
        )
