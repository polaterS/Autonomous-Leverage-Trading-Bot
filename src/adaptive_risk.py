"""
Adaptive Risk Management System
Dynamically adjusts stop-loss, take-profit, and other risk parameters based on:
- Historical win rate
- Symbol-specific performance
- Time-of-day patterns
- Market volatility
"""

from typing import Dict, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime, time, timedelta
from src.database import get_db_client
from src.utils import setup_logging
import asyncio

logger = setup_logging()


class AdaptiveRiskManager:
    """
    Adaptive risk management that learns from performance data.

    Features:
    1. Adaptive Stop-Loss: Tighter SL for low WR, wider for high WR
    2. Dynamic Take-Profit: Based on symbol volatility and success patterns
    3. Symbol-Specific Risk: Each coin gets custom risk params
    4. Time-Based Adjustments: Different risk levels for different hours
    5. Pattern-Based Learning: Learns which setups win/lose
    """

    def __init__(self):
        # Cache for performance metrics (updated every 5 minutes)
        self.performance_cache: Dict[str, Any] = {}
        self.last_cache_update: Optional[datetime] = None
        self.cache_ttl_seconds = 300  # 5 minutes

        # Symbol-specific learned parameters
        self.symbol_risk_params: Dict[str, Dict] = {}

        # Time-of-day performance tracking
        self.hourly_performance: Dict[int, Dict] = {}  # hour -> {wins, losses, avg_pnl}

    async def get_adaptive_stop_loss_percent(
        self,
        symbol: str,
        side: str,
        base_stop_loss: float = 10.0
    ) -> float:
        """
        DISABLED: Adaptive stop-loss based on win rate.

        USER REQUEST: Use fixed stop-loss only (1.5-2.5% from config.py)
        Exit strategy now relies on:
        - Fixed stop-loss: 1.5-2.5% (from config)
        - Fixed loss limit: -$1.50 to -$2.50 (from position_monitor.py)
        - Fixed profit target: $1.50-$2.50 (from position_monitor.py)

        OLD LOGIC (DISABLED):
        - Win rate-based SL adjustment (3-5% range)
        - Symbol-specific performance tracking
        - Adaptive tightening/widening based on recent results

        REASON FOR DISABLING:
        - Conflicted with user's fixed profit/loss targets
        - Made positions close too early (before -$1.50 loss limit)
        - User wants simple, predictable exit points

        Returns:
            base_stop_loss (from config, not adjusted)
        """
        # Return config-based stop-loss without adaptation
        logger.debug(f"📊 Adaptive SL DISABLED - using fixed config stop-loss: {base_stop_loss:.1f}%")
        return base_stop_loss

    async def get_adaptive_take_profit(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        leverage: int,
        position_value: Decimal = Decimal("50.0")  # Default $50 per position
    ) -> Optional[Decimal]:
        """
        Calculate adaptive take-profit price for $2.0 target profit.

        🎯 USER REQUESTED: $1.5-2.5 profit per trade
        This method calculates TP for $2.0 profit (middle of range)

        Args:
            symbol: Trading symbol
            side: LONG or SHORT
            entry_price: Entry price
            leverage: Leverage used
            position_value: Position value in USD (default $50)

        Returns:
            Take-profit price or None if no TP recommended
        """
        try:
            # 🎯 USER REQUESTED: $2.0 profit target
            # Calculate price move needed for $2.0 profit
            # Formula: $2.0 = position_value * price_move * leverage
            target_profit_usd = Decimal("2.0")
            leverage_decimal = Decimal(str(leverage))

            # Calculate required price move percentage
            # price_move = $2.0 / (position_value * leverage)
            tp_percent = float((target_profit_usd / (position_value * leverage_decimal)) * 100)

            # Safety bounds: 0.3% to 1.5% (reasonable for 6x-10x leverage with $50-60 positions)
            # With $57 position at 10x: $2.0 profit = 0.35% price move
            # With $57 position at 6x: $2.0 profit = 0.58% price move
            tp_percent = max(0.3, min(tp_percent, 1.5))

            # Calculate TP price
            if side == 'LONG':
                tp_price = entry_price * (1 + Decimal(str(tp_percent / 100)))
            else:  # SHORT
                tp_price = entry_price * (1 - Decimal(str(tp_percent / 100)))

            logger.info(
                f"🎯 ADAPTIVE TP: {symbol} {side} → {tp_percent:.2f}% "
                f"(target: $2.0 profit with {leverage}x) → ${float(tp_price):.4f}"
            )

            return tp_price

        except Exception as e:
            logger.error(f"Failed to calculate adaptive TP: {e}")
            return None

    async def get_time_based_risk_multiplier(self) -> float:
        """
        Get risk multiplier based on current time-of-day performance.

        Logic:
        - If current hour has high win rate → 1.0-1.2x multiplier (normal/aggressive)
        - If current hour has low win rate → 0.7-0.9x multiplier (defensive)

        Returns:
            Risk multiplier (0.7-1.2)
        """
        try:
            await self._update_hourly_performance()

            current_hour = datetime.now().hour
            hour_stats = self.hourly_performance.get(current_hour, {})

            hour_wr = hour_stats.get('win_rate', 60.0)
            total_trades = hour_stats.get('total', 0)

            # Need at least 10 trades to trust the data
            if total_trades < 10:
                return 1.0  # Neutral

            # Adaptive risk multiplier:
            # 70%+ WR → 1.1-1.2x (increase position size slightly)
            # 60-69% WR → 1.0x (standard)
            # 50-59% WR → 0.9x (slight reduction)
            # <50% WR → 0.7-0.8x (defensive, reduce risk)

            if hour_wr >= 70:
                multiplier = 1.15
                logger.info(f"⏰ TIME-BASED RISK: Hour {current_hour} performing well ({hour_wr:.0f}%) → {multiplier}x")
            elif hour_wr >= 60:
                multiplier = 1.0
            elif hour_wr >= 50:
                multiplier = 0.9
                logger.warning(f"⏰ TIME-BASED RISK: Hour {current_hour} weak ({hour_wr:.0f}%) → {multiplier}x")
            else:
                multiplier = 0.75
                logger.warning(f"⚠️ TIME-BASED RISK: Hour {current_hour} poor ({hour_wr:.0f}%) → {multiplier}x (defensive)")

            return multiplier

        except Exception as e:
            logger.error(f"Failed to calculate time-based multiplier: {e}")
            return 1.0

    async def _update_performance_cache(self):
        """Update performance cache if expired."""
        now = datetime.now()

        if (self.last_cache_update is None or
            (now - self.last_cache_update).total_seconds() >= self.cache_ttl_seconds):

            db = await get_db_client()

            # Get last 50 trades performance
            query = '''
                WITH recent_trades AS (
                    SELECT side, realized_pnl_usd
                    FROM trade_history
                    ORDER BY exit_time DESC
                    LIMIT 50
                )
                SELECT side,
                       COUNT(*) as total,
                       SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins
                FROM recent_trades
                GROUP BY side
            '''

            results = await db.pool.fetch(query)

            for row in results:
                side = row['side']
                total = row['total']
                wins = row['wins']
                wr = (wins / total * 100) if total > 0 else 60.0

                self.performance_cache[f'{side}_wr'] = wr
                self.performance_cache[f'{side}_total'] = total

            self.last_cache_update = now
            logger.debug("📊 Performance cache updated")

    async def _get_symbol_performance(self, symbol: str, side: str) -> Dict[str, float]:
        """Get symbol-specific performance metrics."""
        try:
            db = await get_db_client()

            query = '''
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                    AVG(CASE WHEN realized_pnl_usd > 0 THEN
                        ABS((exit_price - entry_price) / entry_price * 100)
                    ELSE NULL END) as avg_win_percent
                FROM trade_history
                WHERE symbol = $1 AND side = $2
                    AND exit_time >= NOW() - INTERVAL '30 days'
            '''

            row = await db.pool.fetchrow(query, symbol, side)

            if row and row['total'] > 0:
                return {
                    'win_rate': (row['wins'] / row['total'] * 100),
                    'total': row['total'],
                    'avg_win_percent': float(row['avg_win_percent'] or 3.0)
                }
            else:
                return {'win_rate': 60.0, 'total': 0, 'avg_win_percent': 3.0}

        except Exception as e:
            logger.error(f"Failed to get symbol performance: {e}")
            return {'win_rate': 60.0, 'total': 0, 'avg_win_percent': 3.0}

    async def _update_hourly_performance(self):
        """Update hourly performance statistics."""
        try:
            db = await get_db_client()

            # Get performance by hour (last 30 days)
            query = '''
                SELECT
                    EXTRACT(HOUR FROM exit_time AT TIME ZONE 'UTC+3') as hour,
                    COUNT(*) as total,
                    SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins
                FROM trade_history
                WHERE exit_time >= NOW() - INTERVAL '30 days'
                GROUP BY hour
                ORDER BY hour
            '''

            rows = await db.pool.fetch(query)

            for row in rows:
                hour = int(row['hour'])
                total = row['total']
                wins = row['wins']
                wr = (wins / total * 100) if total > 0 else 60.0

                self.hourly_performance[hour] = {
                    'total': total,
                    'wins': wins,
                    'win_rate': wr
                }

            logger.debug("⏰ Hourly performance updated")

        except Exception as e:
            logger.error(f"Failed to update hourly performance: {e}")


# Singleton instance
_adaptive_risk_manager: Optional[AdaptiveRiskManager] = None


def get_adaptive_risk_manager() -> AdaptiveRiskManager:
    """Get or create adaptive risk manager instance."""
    global _adaptive_risk_manager
    if _adaptive_risk_manager is None:
        _adaptive_risk_manager = AdaptiveRiskManager()
    return _adaptive_risk_manager
