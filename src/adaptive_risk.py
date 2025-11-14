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
        Calculate adaptive stop-loss percentage based on recent performance.

        Logic (ULTRA TIGHT for 10-20x leverage):
        - High win rate (>70%) → 4.5-5.0% SL (let winners run within tight range)
        - Low win rate (<50%) → 3.0-3.5% SL (cut losses FAST)
        - Symbol-specific: Each coin gets custom SL based on its performance

        Args:
            symbol: Trading symbol
            side: LONG or SHORT
            base_stop_loss: Default stop-loss percentage (ignored, uses 3-5% range)

        Returns:
            Adaptive stop-loss percentage (3.0-5.0%) - ULTRA TIGHT for 10-20x leverage
        """
        try:
            await self._update_performance_cache()

            # Get recent performance for this side
            recent_wr = self.performance_cache.get(f'{side}_wr', 60.0)

            # Get symbol-specific performance
            symbol_perf = await self._get_symbol_performance(symbol, side)
            symbol_wr = symbol_perf.get('win_rate', recent_wr)

            # Blend general and symbol-specific win rates
            blended_wr = (recent_wr * 0.6) + (symbol_wr * 0.4)

            # Adaptive stop-loss logic (ULTRA TIGHT FOR 10-20x LEVERAGE):
            # With 10-20x leverage, we MUST use very tight stops (3-5%)
            # Win rate affects where in the 3-5% range we land
            # 80%+ WR → 5.0% SL (max range, let winners run)
            # 70-79% WR → 4.5% SL (slightly wider)
            # 60-69% WR → 4.0% SL (standard)
            # 50-59% WR → 3.5% SL (moderate)
            # <50% WR → 3.0% SL (tightest, cut losses fast)

            if blended_wr >= 80:
                adaptive_sl = 5.0
                logger.info(f"🎯 ADAPTIVE SL: High WR ({blended_wr:.0f}%) → Wider SL {adaptive_sl:.1f}%")
            elif blended_wr >= 70:
                adaptive_sl = 4.5
                logger.info(f"📊 ADAPTIVE SL: Good WR ({blended_wr:.0f}%) → Standard+ SL {adaptive_sl:.1f}%")
            elif blended_wr >= 60:
                adaptive_sl = 4.0
                logger.debug(f"ADAPTIVE SL: Medium WR ({blended_wr:.0f}%) → Standard SL {adaptive_sl:.1f}%")
            elif blended_wr >= 50:
                adaptive_sl = 3.5
                logger.warning(f"⚠️ ADAPTIVE SL: Low WR ({blended_wr:.0f}%) → Moderate SL {adaptive_sl:.1f}%")
            else:
                adaptive_sl = 3.0
                logger.warning(f"🚨 ADAPTIVE SL: Very Low WR ({blended_wr:.0f}%) → Tighter SL {adaptive_sl:.1f}%")

            return adaptive_sl

        except Exception as e:
            logger.error(f"Failed to calculate adaptive SL: {e}")
            return base_stop_loss

    async def get_adaptive_take_profit(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        leverage: int
    ) -> Optional[Decimal]:
        """
        Calculate adaptive take-profit price based on symbol volatility and success patterns.

        Logic:
        - High volatility coins → Wider TP (larger swings)
        - Low volatility coins → Tighter TP (smaller moves)
        - Learn from historical: What TP levels worked best?

        Args:
            symbol: Trading symbol
            side: LONG or SHORT
            entry_price: Entry price
            leverage: Leverage used

        Returns:
            Take-profit price or None if no TP recommended
        """
        try:
            # Get symbol volatility and performance
            symbol_perf = await self._get_symbol_performance(symbol, side)
            avg_win_percent = symbol_perf.get('avg_win_percent', 3.0)

            # Adaptive TP logic (OPTIMIZED):
            # Use 70% of average winning move as TP target
            # This ensures we capture profits before reversal
            tp_percent = avg_win_percent * 0.70

            # Minimum TP: 2.0% (OPTIMIZED from 1.5% - wider target for better R:R)
            # Maximum TP: 8% (don't be too greedy)
            tp_percent = max(2.0, min(tp_percent, 8.0))

            # Calculate TP price
            if side == 'LONG':
                tp_price = entry_price * (1 + Decimal(str(tp_percent / 100)))
            else:  # SHORT
                tp_price = entry_price * (1 - Decimal(str(tp_percent / 100)))

            logger.info(
                f"🎯 ADAPTIVE TP: {symbol} {side} → {tp_percent:.1f}% "
                f"(based on avg win: {avg_win_percent:.1f}%) → ${float(tp_price):.4f}"
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
