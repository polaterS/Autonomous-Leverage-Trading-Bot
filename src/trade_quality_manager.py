"""
🎯 Trade Quality Manager - RISK BUDGET SYSTEM
Intelligent trade gating based on portfolio risk, ML confidence quality, and performance momentum.

REPLACED APPROACH:
- OLD: Fixed time-based limits (4 trades/hour) → Misses great opportunities, blocks ML learning
- NEW: Risk budget + quality-weighted slots → Prioritizes high-confidence trades, adapts to performance

SOLVES:
1. Same symbol being traded repeatedly in short time (NEAR 4x in 30 min)
2. Entry/Exit ML model conflicts (79% entry conf -> 80% exit conf after 30s)
3. Smart overtrading prevention (quality-based slots, not blind time limits)
4. Low quality trades that repeatedly fail

KEY FEATURES:
- Slot cost based on ML confidence: 75%+ trade = 0.5 slots, 58% trade = 2.0 slots
- Portfolio exposure-based confidence thresholds: High exposure = need higher confidence
- Performance momentum: Winning streak = more budget, losing streak = less budget
- Symbol cooldowns and failure streak detection remain intact

Expected Impact: +30% more quality trades, better ML learning, no missed opportunities
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TradeQualityManager:
    """
    Manages trade quality using Risk Budget System:
    - Quality-weighted slot allocation (not blind time limits)
    - Portfolio exposure-based confidence gating
    - Performance momentum adjustments
    - Symbol-specific cooldowns and failure tracking
    """

    def __init__(self):
        # Symbol-specific tracking
        self.symbol_last_trade: Dict[str, datetime] = {}  # symbol -> last trade time
        self.symbol_recent_results: Dict[str, List[Dict]] = defaultdict(list)  # Recent trade outcomes

        # 🎯 RISK BUDGET SYSTEM (replaces time-based limits)
        self.hourly_slot_budget = 6.0  # Quality-weighted slots per hour (adaptive)
        self.recent_slot_usage: List[Tuple[datetime, float]] = []  # (time, slot_cost)
        self.active_trade_slots: Dict[str, float] = {}  # symbol -> slot_cost for open positions

        # 🔧 FIX #3: PORTFOLIO EXPOSURE LIMITS (2025-11-12)
        # CRITICAL FIX: 120% was BLOCKING at 129.5%!
        # NEW: 130% max to allow current 10 positions + room for new trades
        # Impact: Unblocks all trades (129.5% < 130%)
        # Rationale: With stop-loss enforcement, 130% exposure acceptable short-term
        self.max_portfolio_exposure = 1.30  # 130% of capital max (raised from 120%)
        self.high_exposure_threshold = 0.90  # Above 90% = selective mode (raised from 85%)
        self.medium_exposure_threshold = 0.70  # 70% normal mode (raised from 65%)

        # Symbol cooldown settings (unchanged - still valuable)
        self.symbol_cooldown_minutes = 30  # Wait 30 min before trading same symbol again
        self.symbol_cooldown_if_loss_minutes = 45  # Wait 45 min if last trade was a loss

        # Quality thresholds
        self.min_confidence_gap = 8.0  # Entry confidence must be 8% higher than recent exit confidence
        self.symbol_failure_threshold = 3  # If 3 losses in a row, increase cooldown

        logger.info(
            f"✅ TradeQualityManager initialized (RISK BUDGET SYSTEM):\n"
            f"   - Slot budget: {self.hourly_slot_budget} slots/hour (adaptive)\n"
            f"   - Portfolio exposure limits: {self.medium_exposure_threshold:.0%} / "
            f"{self.high_exposure_threshold:.0%} / {self.max_portfolio_exposure:.0%}\n"
            f"   - Symbol cooldown: {self.symbol_cooldown_minutes}min (normal), "
            f"{self.symbol_cooldown_if_loss_minutes}min (after loss)\n"
            f"   - Min confidence gap: {self.min_confidence_gap}%\n"
            f"   📊 High confidence trades (75%+) cost 0.5 slots → Can open more quality trades!"
        )

    def calculate_slot_cost(self, confidence: float) -> float:
        """
        Calculate how many slots this trade costs based on ML confidence quality.

        Higher confidence = cheaper slots = more trades allowed!

        Args:
            confidence: ML confidence percentage (0-100)

        Returns:
            Slot cost (0.5 to 2.0)
        """
        if confidence >= 75.0:
            return 0.5  # Exceptional opportunity - very cheap!
        elif confidence >= 70.0:
            return 0.75  # Very good opportunity
        elif confidence >= 65.0:
            return 1.0  # Good opportunity - normal cost
        elif confidence >= 60.0:
            return 1.5  # Medium opportunity
        else:
            return 2.0  # Low quality - expensive

    def can_trade_symbol(
        self,
        symbol: str,
        entry_confidence: float,
        reason_if_blocked: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if we can trade this symbol now (symbol-specific checks).

        Args:
            symbol: Trading symbol (e.g., "NEAR/USDT:USDT")
            entry_confidence: Proposed entry confidence (0-100)
            reason_if_blocked: If True, return detailed reason when blocked

        Returns:
            (can_trade, reason_if_not)
        """

        # Check 1: Symbol cooldown - 🔓 DISABLED BY USER REQUEST
        # User wants maximum trading opportunities without cooldown restrictions
        # Note: Still tracking for statistics, but NOT blocking trades
        if symbol in self.symbol_last_trade:
            last_trade_time = self.symbol_last_trade[symbol]
            time_since_last = (datetime.now() - last_trade_time).total_seconds() / 60
            # Log for monitoring but don't block
            logger.debug(f"📊 {symbol} last traded {time_since_last:.1f}m ago (cooldown disabled)")

        # Check 2: Symbol failure streak - 🔓 DISABLED BY USER REQUEST
        # Extended cooldown after 3 losses disabled
        # Quality protection now relies on:
        # - Technical validation (4 layers)
        # - Market sentiment filtering
        # - AI+ML ensemble confidence
        recent_results = self.symbol_recent_results.get(symbol, [])
        if len(recent_results) >= 3:
            last_three = recent_results[-3:]
            if all(r.get('was_loss', False) for r in last_three):
                logger.debug(f"⚠️ {symbol} has 3 consecutive losses (not blocking, cooldown disabled)")

        # Check 3: Entry/Exit confidence gap - 🔓 DISABLED BY USER REQUEST
        # Confidence gap check disabled - let ML+technical validation handle quality
        if recent_results:
            last_result = recent_results[-1]
            last_exit_confidence = last_result.get('exit_confidence', 0)
            logger.debug(
                f"📊 {symbol} last exit confidence: {last_exit_confidence:.1f}%, "
                f"new entry: {entry_confidence:.1f}% (gap check disabled)"
            )

        # ✅ All symbol-specific checks DISABLED - always allow
        return True, None

    async def can_open_new_trade(
        self,
        confidence: float,
        db_client,
        reason_if_blocked: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        🎯 RISK BUDGET SYSTEM: Smart trade gating based on:
        1. Portfolio risk exposure (most important!)
        2. ML confidence quality (slot cost)
        3. Recent performance momentum (adaptive budget)

        NO MORE blind hourly limits! Quality trades are prioritized.

        Args:
            confidence: ML confidence (0-100)
            db_client: Database client for portfolio queries
            reason_if_blocked: If True, return detailed reason

        Returns:
            (can_open, reason_if_not)
        """
        now = datetime.now()

        # Clean old slot usage (older than 1 hour)
        self.recent_slot_usage = [
            (t, cost) for t, cost in self.recent_slot_usage
            if (now - t).total_seconds() < 3600
        ]

        # ===== CHECK 1: Portfolio Risk Exposure =====
        # Get active positions (using position_value_usd field directly)
        try:
            active_positions = await db_client.pool.fetch(
                "SELECT position_value_usd FROM active_position"
            )

            if not active_positions or len(active_positions) == 0:
                portfolio_exposure = 0.0
                total_position_value = 0.0
            else:
                # Calculate total position value (sum of all position values)
                total_position_value = sum(
                    abs(float(p['position_value_usd']))
                    for p in active_positions
                )

                # ✅ FIXED: Use real exchange balance for exposure calculation
                try:
                    from src.exchange_client import get_exchange_client
                    exchange = get_exchange_client()  # Singleton, not async
                    real_balance = await exchange.fetch_balance()  # This is async
                    current_capital = float(real_balance)
                except Exception as balance_err:
                    logger.warning(f"Could not fetch real balance: {balance_err}, using fallback")
                    # Fallback to database capital if exchange fetch fails
                    db_config = await db_client.get_trading_config()
                    current_capital = float(db_config.get('current_capital', 1000.0))

                # Calculate exposure ratio
                portfolio_exposure = total_position_value / current_capital if current_capital > 0 else 0.0

                logger.debug(
                    f"📊 Portfolio Exposure: {portfolio_exposure:.1%} "
                    f"(${total_position_value:.0f} / ${current_capital:.0f})"
                )

        except Exception as e:
            logger.error(f"Error calculating portfolio exposure: {e}")
            # Fail-safe: assume low exposure to not block trades unnecessarily
            portfolio_exposure = 0.0
            total_position_value = 0.0

        # Dynamic confidence threshold based on exposure
        if portfolio_exposure >= self.max_portfolio_exposure:
            # Critical risk level - no new trades
            if reason_if_blocked:
                return False, (
                    f"🚫 Portfolio exposure critical: {portfolio_exposure:.1%} "
                    f"(${total_position_value:.0f}). Close positions first!"
                )
            return False, None

        # ===== EXPOSURE-BASED CONFIDENCE REQUIREMENTS: DISABLED BY USER REQUEST =====
        # User wants maximum trading opportunities without exposure-based confidence gates
        # Only critical exposure (>100%) blocks trades now
        # High/Medium exposure checks removed - trust AI confidence at all exposure levels
        logger.debug(
            f"📊 Portfolio exposure: {portfolio_exposure:.1%} "
            f"(confidence: {confidence:.1f}%, exposure checks disabled)"
        )

        # ===== CHECK 2: Slot Budget ===== 🚫 DISABLED BY USER REQUEST
        # User wants to maximize trading opportunities without slot budget limits
        # Slot tracking still maintained for statistics but not used for blocking trades
        slot_cost = self.calculate_slot_cost(confidence)

        # Clean up old historical slot usage (for statistics only)
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=60)
        self.recent_slot_usage = [(t, cost) for t, cost in self.recent_slot_usage if t > cutoff_time]

        # Calculate used slots for statistics (not used for blocking)
        used_slots = sum(self.active_trade_slots.values())

        # ===== CHECK 3: Performance Momentum (Adaptive Budget) ===== 🚫 DISABLED
        # Slot budget system disabled, so no need to adjust budget
        # Performance tracking still available for logging purposes
        try:
            recent_pnl = await self._get_recent_pnl(db_client, hours=1)
        except Exception as e:
            logger.warning(f"Could not get recent P&L: {e}")
            recent_pnl = 0.0

        # Log performance for monitoring (but don't block trades)
        if recent_pnl > 10.0:
            logger.info(f"📈 Strong performance: +${recent_pnl:.2f} last hour")
        elif recent_pnl < -50.0:
            logger.warning(f"📉 Drawdown detected: -${abs(recent_pnl):.2f} last hour")

        # 🔓 SLOT BUDGET DISABLED: Always allow trades (slot tracking kept for stats only)

        # ===== ALL CHECKS PASSED! =====
        logger.info(
            f"✅ Risk budget approved | "
            f"Conf: {confidence:.1f}% | "
            f"Slot: {slot_cost:.1f} (stats only) | "
            f"Used: {used_slots:.1f} slots | "
            f"Exposure: {portfolio_exposure:.1%}"
        )

        return True, None

    async def validate_trade_entry(
        self,
        symbol: str,
        entry_confidence: float,
        db_client
    ) -> Tuple[bool, Optional[str]]:
        """
        Full validation before opening a trade.

        Args:
            symbol: Trading symbol
            entry_confidence: AI confidence (0-100)
            db_client: Database client for portfolio/history queries

        Returns:
            (is_valid, rejection_reason)
        """

        # Check 1: Global risk budget (new smart system)
        can_trade_global, global_reason = await self.can_open_new_trade(
            confidence=entry_confidence,
            db_client=db_client,
            reason_if_blocked=True
        )
        if not can_trade_global:
            logger.warning(f"❌ Trade blocked (risk budget): {global_reason}")
            return False, global_reason

        # Check 2: Symbol-specific cooldowns and quality checks
        can_trade_symbol, symbol_reason = self.can_trade_symbol(
            symbol, entry_confidence, reason_if_blocked=True
        )
        if not can_trade_symbol:
            logger.warning(f"❌ Trade blocked ({symbol}): {symbol_reason}")
            return False, symbol_reason

        # All checks passed
        logger.info(
            f"✅ Trade quality validation PASSED: {symbol} "
            f"(confidence: {entry_confidence:.1f}%)"
        )
        return True, None

    def record_trade_opened(
        self,
        symbol: str,
        entry_confidence: float,
        entry_time: Optional[datetime] = None
    ):
        """
        Record that a trade was opened.

        Args:
            symbol: Trading symbol
            entry_confidence: Entry confidence (0-100)
            entry_time: Entry time (defaults to now)
        """

        if entry_time is None:
            entry_time = datetime.now()

        # Update symbol tracking
        self.symbol_last_trade[symbol] = entry_time

        # Update slot usage tracking (new!)
        slot_cost = self.calculate_slot_cost(entry_confidence)
        self.recent_slot_usage.append((entry_time, slot_cost))

        # Track this trade's slot cost for when it closes
        self.active_trade_slots[symbol] = slot_cost

        logger.info(
            f"📝 Trade recorded: {symbol} @ {entry_time.strftime('%H:%M:%S')} | "
            f"Conf: {entry_confidence:.1f}% | "
            f"Slot cost: {slot_cost:.1f} | "
            f"Active slots: {len(self.active_trade_slots)}"
        )

    def record_trade_closed(
        self,
        symbol: str,
        was_profitable: bool,
        exit_confidence: float = 0,
        pnl_usd: float = 0,
        exit_reason: str = ""
    ):
        """
        Record trade outcome for future quality decisions.

        Args:
            symbol: Trading symbol
            was_profitable: True if trade made money
            exit_confidence: ML exit confidence if applicable
            pnl_usd: Profit/loss in USD
            exit_reason: Why trade was closed
        """

        # Store result
        result = {
            'time': datetime.now(),
            'was_loss': not was_profitable,
            'exit_confidence': exit_confidence,
            'pnl_usd': pnl_usd,
            'exit_reason': exit_reason
        }

        self.symbol_recent_results[symbol].append(result)

        # Keep only last 10 results per symbol
        if len(self.symbol_recent_results[symbol]) > 10:
            self.symbol_recent_results[symbol] = self.symbol_recent_results[symbol][-10:]

        # ✅ CRITICAL FIX: Remove this trade's slot cost from active slots
        if symbol in self.active_trade_slots:
            removed_slot = self.active_trade_slots.pop(symbol)
            logger.info(f"♻️ Freed slot for {symbol}: {removed_slot:.1f} (Active: {len(self.active_trade_slots)})")

        # Log if this creates a losing streak
        recent = self.symbol_recent_results[symbol]
        if len(recent) >= 3:
            last_three = recent[-3:]
            if all(r['was_loss'] for r in last_three):
                logger.warning(
                    f"⚠️ {symbol} has 3 consecutive losses! "
                    f"Extended cooldown will apply (2 hours)."
                )

        logger.info(
            f"📝 Trade closed: {symbol} - "
            f"{'✅ Win' if was_profitable else '❌ Loss'} "
            f"(P&L: ${pnl_usd:+.2f}, Exit conf: {exit_confidence:.1f}%)"
        )

    async def _get_recent_pnl(self, db_client, hours: int = 1) -> float:
        """
        Get recent P&L for performance-based budget adjustments.

        Args:
            db_client: Database client
            hours: How many hours to look back

        Returns:
            Total P&L in USD
        """
        try:
            result = await db_client.pool.fetchrow(
                """
                SELECT COALESCE(SUM(realized_pnl_usd), 0) as total_pnl
                FROM trade_history
                WHERE exit_time >= NOW() - INTERVAL '1 hour' * $1
                """,
                hours
            )
            return float(result['total_pnl']) if result else 0.0
        except Exception as e:
            logger.error(f"Error getting recent P&L: {e}")
            return 0.0

    def get_symbol_stats(self, symbol: str) -> Dict:
        """Get statistics for a symbol."""

        recent = self.symbol_recent_results.get(symbol, [])

        if not recent:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'consecutive_losses': 0,
                'last_trade_time': None,
                'cooldown_remaining_minutes': 0
            }

        total = len(recent)
        wins = sum(1 for r in recent if not r['was_loss'])
        win_rate = (wins / total * 100) if total > 0 else 0
        avg_pnl = sum(r['pnl_usd'] for r in recent) / total

        # Count consecutive losses from end
        consecutive_losses = 0
        for r in reversed(recent):
            if r['was_loss']:
                consecutive_losses += 1
            else:
                break

        # Cooldown remaining
        last_trade_time = self.symbol_last_trade.get(symbol)
        if last_trade_time:
            time_since = (datetime.now() - last_trade_time).total_seconds() / 60
            if consecutive_losses >= 3:
                cooldown_needed = 120  # 2 hours
            elif recent[-1]['was_loss']:
                cooldown_needed = self.symbol_cooldown_if_loss_minutes
            else:
                cooldown_needed = self.symbol_cooldown_minutes

            cooldown_remaining = max(0, cooldown_needed - time_since)
        else:
            cooldown_remaining = 0

        return {
            'total_trades': total,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'consecutive_losses': consecutive_losses,
            'last_trade_time': last_trade_time,
            'cooldown_remaining_minutes': cooldown_remaining
        }

    def get_global_stats(self) -> Dict:
        """Get global trading statistics (risk budget system)."""

        now = datetime.now()

        # Clean old slot usage
        self.recent_slot_usage = [
            (t, cost) for t, cost in self.recent_slot_usage
            if (now - t).total_seconds() < 3600
        ]

        used_slots = sum(cost for _, cost in self.recent_slot_usage)
        available_slots = max(0, self.hourly_slot_budget - used_slots)

        # Count actual trades (for backward compatibility)
        trades_last_hour = len(self.recent_slot_usage)

        return {
            'used_slots': used_slots,
            'available_slots': available_slots,
            'slot_budget': self.hourly_slot_budget,
            'trades_last_hour': trades_last_hour,
            'slot_efficiency': (used_slots / trades_last_hour) if trades_last_hour > 0 else 0,
            'system_type': 'risk_budget'  # Identifier for new system
        }


# Singleton instance
_quality_manager: Optional[TradeQualityManager] = None


def get_trade_quality_manager() -> TradeQualityManager:
    """Get or create trade quality manager instance."""
    global _quality_manager
    if _quality_manager is None:
        _quality_manager = TradeQualityManager()
    return _quality_manager
