"""
Risk Management System for the trading bot.
Enforces all safety rules and validates trades before execution.
"""

from typing import Dict, Any
from decimal import Decimal
from datetime import date
from src.config import get_settings
from src.database import get_db_client
from src.utils import setup_logging, calculate_liquidation_price

logger = setup_logging()


class RiskManager:
    """Comprehensive risk management system."""

    def __init__(self):
        self.settings = get_settings()

    async def validate_trade(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive trade validation against all risk rules.

        Args:
            trade_params: Dict with symbol, side, leverage, stop_loss_percent, current_price

        Returns:
            Dict with 'approved': bool, 'reason': str, 'adjusted_params': dict
        """
        symbol = trade_params['symbol']
        side = trade_params['side']
        leverage = trade_params['leverage']
        stop_loss_percent = trade_params['stop_loss_percent']
        current_price = Decimal(str(trade_params['current_price']))

        logger.info(f"Validating trade: {symbol} {side} {leverage}x with {stop_loss_percent}% stop-loss")

        # RULE 1: Stop-loss must be between 5-10%
        if stop_loss_percent < 5 or stop_loss_percent > 10:
            return {
                'approved': False,
                'reason': f'Stop-loss {stop_loss_percent}% outside required range (5-10%)',
                'adjusted_params': None
            }

        # RULE 2: Check if we have enough capital
        db = await get_db_client()
        current_capital = await db.get_current_capital()
        position_value = current_capital * self.settings.position_size_percent

        if position_value < 10:  # Minimum $10 position
            return {
                'approved': False,
                'reason': f'Insufficient capital: ${current_capital:.2f} (need at least $12.50)',
                'adjusted_params': None
            }

        # RULE 3: Leverage check
        if leverage > self.settings.max_leverage:
            # Adjust leverage down to max
            logger.warning(f"Leverage {leverage}x exceeds maximum, adjusting to {self.settings.max_leverage}x")
            leverage = self.settings.max_leverage
            trade_params['leverage'] = leverage

        if leverage < 2:
            leverage = 2
            trade_params['leverage'] = 2

        # RULE 4: Check if position already exists
        active_position = await db.get_active_position()
        if active_position:
            return {
                'approved': False,
                'reason': 'Position already open. Cannot open multiple positions.',
                'adjusted_params': None
            }

        # RULE 5: Daily loss limit check
        daily_pnl = await db.get_daily_pnl(date.today())
        max_daily_loss = current_capital * self.settings.daily_loss_limit_percent

        if daily_pnl < -max_daily_loss:
            await db.record_circuit_breaker(
                'daily_loss_limit',
                daily_pnl,
                -max_daily_loss,
                'Trading suspended until next day'
            )
            return {
                'approved': False,
                'reason': f'Daily loss limit reached: ${daily_pnl:.2f} (limit: ${-max_daily_loss:.2f})',
                'adjusted_params': None
            }

        # RULE 6: Consecutive losses check
        consecutive_losses = await db.get_consecutive_losses()
        if consecutive_losses >= self.settings.max_consecutive_losses:
            await db.record_circuit_breaker(
                'consecutive_losses',
                Decimal(consecutive_losses),
                Decimal(self.settings.max_consecutive_losses),
                'Trading paused after consecutive losses'
            )
            return {
                'approved': False,
                'reason': f'{consecutive_losses} consecutive losses - trading paused for review',
                'adjusted_params': None
            }

        # RULE 7: Maximum loss per trade check
        max_loss_per_trade = position_value * (Decimal(str(stop_loss_percent)) / 100) * leverage
        max_acceptable_loss = current_capital * Decimal("0.20")  # 20% of capital max

        if max_loss_per_trade > max_acceptable_loss:
            # Try to reduce leverage
            adjusted_leverage = int((max_acceptable_loss / position_value) / (Decimal(str(stop_loss_percent)) / 100))
            adjusted_leverage = max(2, min(adjusted_leverage, self.settings.max_leverage))

            if adjusted_leverage < 2:
                return {
                    'approved': False,
                    'reason': f'Potential loss ${max_loss_per_trade:.2f} exceeds 20% of capital even with minimum leverage',
                    'adjusted_params': None
                }

            logger.warning(f"Adjusting leverage from {leverage}x to {adjusted_leverage}x to limit risk")
            leverage = adjusted_leverage
            trade_params['leverage'] = adjusted_leverage

        # RULE 8: Liquidation distance must be safe
        liq_price = calculate_liquidation_price(current_price, leverage, side)
        liq_distance = abs(current_price - liq_price) / current_price

        if liq_distance < Decimal("0.10"):  # Must be at least 10% away
            return {
                'approved': False,
                'reason': f'Liquidation too close: {float(liq_distance)*100:.1f}% (need at least 10%)',
                'adjusted_params': None
            }

        # RULE 9: Minimum profit target validation
        min_profit_pct = self.settings.min_profit_usd / position_value

        if min_profit_pct > Decimal("0.15"):  # More than 15% needed
            return {
                'approved': False,
                'reason': f'Minimum profit target {float(min_profit_pct)*100:.1f}% too high for position size',
                'adjusted_params': None
            }

        # RULE 10: Trading enabled check
        config = await db.get_trading_config()
        if not config.get('is_trading_enabled', True):
            return {
                'approved': False,
                'reason': 'Trading is currently disabled',
                'adjusted_params': None
            }

        # All checks passed
        logger.info(f"✅ Trade validation passed for {symbol} {side} {leverage}x")

        return {
            'approved': True,
            'reason': 'All risk checks passed',
            'adjusted_params': trade_params
        }

    async def check_position_risk(self, position: Dict[str, Any], current_price: Decimal) -> Dict[str, Any]:
        """
        Check risk metrics for an open position.

        Returns:
            Dict with risk assessment and recommended actions
        """
        symbol = position['symbol']
        side = position['side']
        entry_price = Decimal(str(position['entry_price']))
        leverage = position['leverage']
        stop_loss_price = Decimal(str(position['stop_loss_price']))
        liquidation_price = Decimal(str(position['liquidation_price']))

        # Calculate distance to liquidation
        liq_distance = abs(current_price - liquidation_price) / current_price

        # Calculate distance to stop-loss
        sl_distance = abs(current_price - stop_loss_price) / current_price

        # Determine risk level
        if liq_distance < Decimal("0.05"):  # Less than 5%
            risk_level = 'CRITICAL'
            action = 'CLOSE_IMMEDIATELY'
            reason = f'Liquidation risk: only {float(liq_distance)*100:.2f}% away'

        elif liq_distance < Decimal("0.10"):  # Less than 10%
            risk_level = 'HIGH'
            action = 'MONITOR_CLOSELY'
            reason = f'Liquidation warning: {float(liq_distance)*100:.2f}% away'

        elif sl_distance < Decimal("0.01"):  # Stop-loss very close
            risk_level = 'HIGH'
            action = 'STOP_LOSS_IMMINENT'
            reason = f'Stop-loss about to trigger: {float(sl_distance)*100:.2f}% away'

        else:
            risk_level = 'NORMAL'
            action = 'CONTINUE'
            reason = 'Risk levels acceptable'

        return {
            'risk_level': risk_level,
            'action': action,
            'reason': reason,
            'liq_distance_pct': float(liq_distance) * 100,
            'sl_distance_pct': float(sl_distance) * 100
        }

    async def check_daily_limits(self) -> Dict[str, Any]:
        """
        Check if daily trading limits have been reached.

        Returns:
            Dict with 'can_trade': bool, 'reason': str
        """
        db = await get_db_client()

        # Check daily loss limit
        current_capital = await db.get_current_capital()
        daily_pnl = await db.get_daily_pnl(date.today())
        max_daily_loss = current_capital * self.settings.daily_loss_limit_percent

        if daily_pnl < -max_daily_loss:
            return {
                'can_trade': False,
                'reason': f'Daily loss limit reached: ${daily_pnl:.2f}',
                'type': 'daily_loss_limit'
            }

        # Check consecutive losses
        consecutive_losses = await db.get_consecutive_losses()
        if consecutive_losses >= self.settings.max_consecutive_losses:
            return {
                'can_trade': False,
                'reason': f'{consecutive_losses} consecutive losses',
                'type': 'consecutive_losses'
            }

        # Check trading enabled
        config = await db.get_trading_config()
        if not config.get('is_trading_enabled', True):
            return {
                'can_trade': False,
                'reason': 'Trading manually disabled',
                'type': 'manual_disable'
            }

        return {
            'can_trade': True,
            'reason': 'All limits OK',
            'type': None
        }

    async def calculate_optimal_position_size(
        self,
        symbol: str,
        entry_price: Decimal,
        stop_loss_percent: float,
        leverage: int
    ) -> Decimal:
        """
        Calculate optimal position size based on risk parameters.

        Returns:
            Position value in USD
        """
        db = await get_db_client()
        current_capital = await db.get_current_capital()

        # Use configured position size percentage
        position_value = current_capital * self.settings.position_size_percent

        # Ensure we don't risk more than acceptable per trade
        max_risk = current_capital * Decimal("0.15")  # Max 15% risk per trade
        stop_loss_decimal = Decimal(str(stop_loss_percent)) / 100
        max_position_from_risk = max_risk / (stop_loss_decimal * leverage)

        # Take the smaller of the two
        optimal_size = min(position_value, max_position_from_risk)

        # Ensure minimum viable position
        if optimal_size < 10:
            logger.warning(f"Calculated position size ${optimal_size:.2f} too small")
            return Decimal("0")

        logger.info(f"Optimal position size: ${optimal_size:.2f}")
        return optimal_size

    async def should_emergency_close(self, position: Dict[str, Any], current_price: Decimal) -> tuple[bool, str]:
        """
        Determine if position should be emergency closed.

        Returns:
            (should_close: bool, reason: str)
        """
        # Check liquidation distance
        liquidation_price = Decimal(str(position['liquidation_price']))
        liq_distance = abs(current_price - liquidation_price) / current_price

        if liq_distance < Decimal("0.05"):
            return True, f"EMERGENCY: Liquidation risk ({float(liq_distance)*100:.2f}% away)"

        # Check stop-loss
        stop_loss_price = Decimal(str(position['stop_loss_price']))
        side = position['side']

        if side == 'LONG' and current_price <= stop_loss_price * Decimal("1.001"):
            return True, "Stop-loss triggered"
        elif side == 'SHORT' and current_price >= stop_loss_price * Decimal("0.999"):
            return True, "Stop-loss triggered"

        return False, ""


# Singleton instance
_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Get or create risk manager instance."""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager
