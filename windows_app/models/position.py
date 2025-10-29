"""
Active Position Model - Current open position data.
"""

from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Optional


@dataclass
class ActivePosition:
    """Active position model with real-time calculations."""

    id: int
    symbol: str
    side: str
    leverage: int
    entry_price: Decimal
    current_price: Decimal
    quantity: Decimal
    position_value_usd: Decimal
    stop_loss_price: Decimal
    stop_loss_percent: Decimal
    min_profit_target_usd: Decimal
    min_profit_price: Decimal
    liquidation_price: Decimal
    unrealized_pnl_usd: Decimal = Decimal("0")
    exchange_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    ai_model_consensus: Optional[str] = None
    ai_confidence: Optional[Decimal] = None
    entry_time: Optional[datetime] = None
    last_check_time: Optional[datetime] = None
    partial_close_executed: bool = False

    def calculate_unrealized_pnl(self, current_price: Optional[Decimal] = None) -> Decimal:
        """
        Calculate unrealized P&L.

        Args:
            current_price: Optional price to use (defaults to self.current_price)

        Returns:
            Unrealized P&L in USD
        """
        price = current_price if current_price else self.current_price

        if self.side == 'LONG':
            price_change_pct = (price - self.entry_price) / self.entry_price
        else:  # SHORT
            price_change_pct = (self.entry_price - price) / self.entry_price

        pnl = self.position_value_usd * price_change_pct * self.leverage
        return pnl

    def get_pnl_percent(self) -> Decimal:
        """Get P&L as percentage of position value."""
        if self.position_value_usd == 0:
            return Decimal("0")

        return (self.unrealized_pnl_usd / self.position_value_usd) * 100

    def get_distance_to_stop_loss(self) -> Decimal:
        """
        Get distance to stop-loss as percentage.

        Returns:
            Percentage distance (positive value)
        """
        if self.side == 'LONG':
            distance = (self.current_price - self.stop_loss_price) / self.current_price
        else:  # SHORT
            distance = (self.stop_loss_price - self.current_price) / self.current_price

        return abs(distance) * 100

    def get_distance_to_liquidation(self) -> Decimal:
        """
        Get distance to liquidation as percentage.

        Returns:
            Percentage distance (positive value)
        """
        if self.side == 'LONG':
            distance = (self.current_price - self.liquidation_price) / self.current_price
        else:  # SHORT
            distance = (self.liquidation_price - self.current_price) / self.current_price

        return abs(distance) * 100

    def is_in_profit(self) -> bool:
        """Check if position is currently profitable."""
        return self.unrealized_pnl_usd > 0

    def reached_min_profit(self) -> bool:
        """Check if minimum profit target reached."""
        return self.unrealized_pnl_usd >= self.min_profit_target_usd

    def is_near_stop_loss(self, threshold_percent: Decimal = Decimal("2.0")) -> bool:
        """
        Check if position is dangerously close to stop-loss.

        Args:
            threshold_percent: Warning threshold (default 2%)

        Returns:
            True if within threshold of stop-loss
        """
        return self.get_distance_to_stop_loss() < threshold_percent

    def is_near_liquidation(self, threshold_percent: Decimal = Decimal("5.0")) -> bool:
        """
        Check if position is dangerously close to liquidation.

        Args:
            threshold_percent: Warning threshold (default 5%)

        Returns:
            True if within threshold of liquidation
        """
        return self.get_distance_to_liquidation() < threshold_percent

    def get_status_summary(self) -> dict:
        """
        Get comprehensive status summary.

        Returns:
            Dictionary with status information
        """
        return {
            'symbol': self.symbol,
            'side': self.side,
            'leverage': self.leverage,
            'entry_price': float(self.entry_price),
            'current_price': float(self.current_price),
            'unrealized_pnl': float(self.unrealized_pnl_usd),
            'pnl_percent': float(self.get_pnl_percent()),
            'is_in_profit': self.is_in_profit(),
            'reached_min_profit': self.reached_min_profit(),
            'distance_to_sl': float(self.get_distance_to_stop_loss()),
            'distance_to_liq': float(self.get_distance_to_liquidation()),
            'is_near_stop_loss': self.is_near_stop_loss(),
            'is_near_liquidation': self.is_near_liquidation(),
        }

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'leverage': self.leverage,
            'entry_price': float(self.entry_price),
            'current_price': float(self.current_price),
            'quantity': float(self.quantity),
            'position_value_usd': float(self.position_value_usd),
            'stop_loss_price': float(self.stop_loss_price),
            'stop_loss_percent': float(self.stop_loss_percent),
            'min_profit_target_usd': float(self.min_profit_target_usd),
            'min_profit_price': float(self.min_profit_price),
            'liquidation_price': float(self.liquidation_price),
            'unrealized_pnl_usd': float(self.unrealized_pnl_usd),
            'exchange_order_id': self.exchange_order_id,
            'stop_loss_order_id': self.stop_loss_order_id,
            'ai_model_consensus': self.ai_model_consensus,
            'ai_confidence': float(self.ai_confidence) if self.ai_confidence else None,
            'partial_close_executed': self.partial_close_executed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ActivePosition':
        """Create instance from dictionary."""
        return cls(
            id=data['id'],
            symbol=data['symbol'],
            side=data['side'],
            leverage=data['leverage'],
            entry_price=Decimal(str(data['entry_price'])),
            current_price=Decimal(str(data.get('current_price', data['entry_price']))),
            quantity=Decimal(str(data['quantity'])),
            position_value_usd=Decimal(str(data['position_value_usd'])),
            stop_loss_price=Decimal(str(data['stop_loss_price'])),
            stop_loss_percent=Decimal(str(data['stop_loss_percent'])),
            min_profit_target_usd=Decimal(str(data['min_profit_target_usd'])),
            min_profit_price=Decimal(str(data['min_profit_price'])),
            liquidation_price=Decimal(str(data['liquidation_price'])),
            unrealized_pnl_usd=Decimal(str(data.get('unrealized_pnl_usd', 0))),
            exchange_order_id=data.get('exchange_order_id'),
            stop_loss_order_id=data.get('stop_loss_order_id'),
            ai_model_consensus=data.get('ai_model_consensus'),
            ai_confidence=Decimal(str(data['ai_confidence'])) if data.get('ai_confidence') else None,
            entry_time=data.get('entry_time'),
            last_check_time=data.get('last_check_time'),
            partial_close_executed=data.get('partial_close_executed', False),
        )
