"""
Position Monitor - Continuously monitors open positions.
Checks for stop-loss, take-profit, liquidation risk, and AI exit signals.
"""

import asyncio
from typing import Dict, Any
from decimal import Decimal
from datetime import datetime
from src.config import get_settings
from src.database import get_db_client
from src.exchange_client import get_exchange_client
from src.risk_manager import get_risk_manager
from src.trade_executor import get_trade_executor
from src.telegram_notifier import get_notifier
from src.ai_engine import get_ai_engine
from src.utils import setup_logging, calculate_pnl
from src.indicators import calculate_indicators, detect_market_regime

logger = setup_logging()


class PositionMonitor:
    """Monitors open positions and manages exit conditions."""

    def __init__(self):
        self.settings = get_settings()
        self.last_update_time = None
        self.last_ai_check_time = None

    async def check_position(self, position: Dict[str, Any]) -> None:
        """
        Main position monitoring function.
        Called every 60 seconds when a position is open.

        Args:
            position: Active position data from database
        """
        symbol = position['symbol']
        side = position['side']

        try:
            exchange = await get_exchange_client()
            db = await get_db_client()
            risk_manager = get_risk_manager()
            executor = get_trade_executor()
            notifier = get_notifier()

            # Get current price
            ticker = await exchange.fetch_ticker(symbol)
            current_price = Decimal(str(ticker['last']))

            # Calculate current P&L
            pnl_data = calculate_pnl(
                Decimal(str(position['entry_price'])),
                current_price,
                Decimal(str(position['quantity'])),
                side,
                position['leverage'],
                Decimal(str(position['position_value_usd']))
            )

            unrealized_pnl = Decimal(str(pnl_data['unrealized_pnl']))

            # Update position in database
            await db.update_position_price(position['id'], current_price, unrealized_pnl)

            logger.info(
                f"Position: {symbol} {side} | "
                f"Entry: ${float(position['entry_price']):.4f} | "
                f"Current: ${float(current_price):.4f} | "
                f"P&L: ${float(unrealized_pnl):+.2f}"
            )

            # === CRITICAL CHECK 1: Emergency close conditions ===
            should_emergency_close, emergency_reason = await risk_manager.should_emergency_close(position, current_price)

            if should_emergency_close:
                logger.critical(f"🚨 EMERGENCY CLOSE: {emergency_reason}")
                await notifier.send_alert('critical', f"Emergency closing position: {emergency_reason}")
                await executor.close_position(position, current_price, emergency_reason)
                return

            # === CRITICAL CHECK 2: Liquidation distance ===
            liquidation_price = Decimal(str(position['liquidation_price']))
            distance_to_liq = abs(current_price - liquidation_price) / current_price

            if distance_to_liq < Decimal("0.05"):  # Less than 5%
                logger.critical(f"🚨 LIQUIDATION RISK! Distance: {float(distance_to_liq)*100:.2f}%")
                await notifier.send_alert(
                    'critical',
                    f"🚨 LIQUIDATION RISK!\n"
                    f"{symbol} {side}\n"
                    f"Distance to liquidation: {float(distance_to_liq)*100:.2f}%\n"
                    f"Closing position immediately!"
                )
                await executor.close_position(position, current_price, "EMERGENCY - Liquidation risk")
                return

            # === CHECK 3: Take profit conditions ===
            min_profit_usd = Decimal(str(position['min_profit_target_usd']))

            if unrealized_pnl >= min_profit_usd:
                logger.info(f"✅ Minimum profit target reached: ${float(unrealized_pnl):.2f}")

                # Strategy 1: If profit is 2x minimum target, close immediately
                if unrealized_pnl >= (min_profit_usd * 2):
                    logger.info(f"🎉 Excellent profit achieved: ${float(unrealized_pnl):.2f} (2x target)")
                    await executor.close_position(
                        position,
                        current_price,
                        "Take profit - 2x minimum target achieved"
                    )
                    return

                # Strategy 2: If price moved significantly beyond min profit price
                min_profit_price = Decimal(str(position['min_profit_price']))

                if side == 'LONG':
                    price_beyond_target = (current_price - min_profit_price) / min_profit_price
                    if price_beyond_target > Decimal("0.02"):  # 2% beyond target
                        logger.info(f"📈 Strong move beyond target: {float(price_beyond_target)*100:.2f}%")
                        await executor.close_position(
                            position,
                            current_price,
                            "Take profit - strong move beyond target"
                        )
                        return
                else:  # SHORT
                    price_beyond_target = (min_profit_price - current_price) / min_profit_price
                    if price_beyond_target > Decimal("0.02"):
                        logger.info(f"📉 Strong move beyond target: {float(price_beyond_target)*100:.2f}%")
                        await executor.close_position(
                            position,
                            current_price,
                            "Take profit - strong move beyond target"
                        )
                        return

            # === CHECK 4: AI exit signal (every 5 minutes) ===
            should_check_ai = (
                self.last_ai_check_time is None or
                (datetime.now() - self.last_ai_check_time).total_seconds() >= 300
            )

            if should_check_ai and unrealized_pnl > -min_profit_usd:  # Don't ask AI when deep in loss
                logger.info("Requesting AI exit signal...")
                self.last_ai_check_time = datetime.now()

                try:
                    # Gather quick market data
                    market_data = await self._gather_quick_market_data(symbol, current_price)

                    # Get AI exit recommendation
                    ai_engine = get_ai_engine()
                    duration = (datetime.now() - position.get('entry_time', datetime.now())).total_seconds()

                    position_for_ai = {
                        **position,
                        'unrealized_pnl_usd': unrealized_pnl,
                        'duration': f"{int(duration//60)}m"
                    }

                    exit_signal = await ai_engine.get_exit_signal(symbol, market_data, position_for_ai)

                    if exit_signal.get('should_exit') and exit_signal.get('confidence', 0) >= 0.70:
                        logger.info(f"🤖 AI recommends exit: {exit_signal.get('reason')}")
                        await executor.close_position(
                            position,
                            current_price,
                            f"AI exit signal - {exit_signal.get('reason')}"
                        )
                        return

                except Exception as e:
                    logger.warning(f"Failed to get AI exit signal: {e}")

            # === CHECK 5: Send periodic updates (every 5 minutes) ===
            should_send_update = (
                self.last_update_time is None or
                (datetime.now() - self.last_update_time).total_seconds() >= 300
            )

            if should_send_update:
                self.last_update_time = datetime.now()
                await notifier.send_position_update(position, unrealized_pnl)

        except Exception as e:
            logger.error(f"Error in position monitor: {e}")
            # Don't raise - we'll try again next cycle

    async def _gather_quick_market_data(self, symbol: str, current_price: Decimal) -> Dict[str, Any]:
        """Gather minimal market data for AI exit signal."""
        try:
            exchange = await get_exchange_client()

            # Get OHLCV for indicators
            ohlcv_15m = await exchange.fetch_ohlcv(symbol, '15m', limit=50)

            # Calculate indicators
            indicators_15m = calculate_indicators(ohlcv_15m)

            # Market regime
            regime = detect_market_regime(ohlcv_15m)

            # Get funding rate
            funding_rate = await exchange.fetch_funding_rate(symbol)

            return {
                'current_price': float(current_price),
                'volume_24h': indicators_15m.get('volume', 0),
                'market_regime': regime,
                'funding_rate': funding_rate,
                'indicators': {
                    '15m': indicators_15m,
                    '1h': indicators_15m,  # Use same for simplicity in quick check
                    '4h': indicators_15m
                }
            }

        except Exception as e:
            logger.warning(f"Error gathering quick market data: {e}")
            return {
                'current_price': float(current_price),
                'volume_24h': 0,
                'market_regime': 'UNKNOWN',
                'funding_rate': {'rate': 0.0},
                'indicators': {
                    '15m': {'rsi': 50, 'macd': 0, 'macd_signal': 0},
                    '1h': {'rsi': 50, 'macd': 0, 'macd_signal': 0},
                    '4h': {'rsi': 50, 'macd': 0, 'macd_signal': 0}
                }
            }


# Singleton instance
_position_monitor: Optional[PositionMonitor] = None


def get_position_monitor() -> PositionMonitor:
    """Get or create position monitor instance."""
    global _position_monitor
    if _position_monitor is None:
        _position_monitor = PositionMonitor()
    return _position_monitor
