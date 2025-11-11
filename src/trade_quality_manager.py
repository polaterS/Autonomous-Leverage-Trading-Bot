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

        # Portfolio risk thresholds
        self.max_portfolio_exposure = 0.70  # 70% of capital max
        self.high_exposure_threshold = 0.50  # Above 50% = selective mode
        self.medium_exposure_threshold = 0.30  # Above 30% = normal mode

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

        # Check 1: Symbol cooldown
        if symbol in self.symbol_last_trade:
            last_trade_time = self.symbol_last_trade[symbol]
            time_since_last = (datetime.now() - last_trade_time).total_seconds() / 60

            # Determine cooldown based on last trade result
            recent_results = self.symbol_recent_results.get(symbol, [])
            if recent_results and recent_results[-1].get('was_loss', False):
                cooldown_needed = self.symbol_cooldown_if_loss_minutes
            else:
                cooldown_needed = self.symbol_cooldown_minutes

            if time_since_last < cooldown_needed:
                if reason_if_blocked:
                    return False, (
                        f"❄️ Symbol cooldown: {symbol} was traded {time_since_last:.1f}m ago, "
                        f"need {cooldown_needed}m cooldown"
                    )
                return False, None

        # Check 2: Symbol failure streak (3+ losses in a row = extended cooldown)
        recent_results = self.symbol_recent_results.get(symbol, [])
        if len(recent_results) >= 3:
            last_three = recent_results[-3:]
            if all(r.get('was_loss', False) for r in last_three):
                # All last 3 trades were losses
                if symbol in self.symbol_last_trade:
                    last_trade_time = self.symbol_last_trade[symbol]
                    time_since_last = (datetime.now() - last_trade_time).total_seconds() / 60
                    extended_cooldown = 120  # 2 hours after 3 losses

                    if time_since_last < extended_cooldown:
                        if reason_if_blocked:
                            return False, (
                                f"🚫 {symbol} has 3 consecutive losses. "
                                f"Extended cooldown: {time_since_last:.1f}m / {extended_cooldown}m"
                            )
                        return False, None

        # Check 3: Entry/Exit confidence gap
        # If last trade exited due to high confidence ML signal,
        # new entry must have significantly higher confidence
        if recent_results:
            last_result = recent_results[-1]
            last_exit_confidence = last_result.get('exit_confidence', 0)

            if last_exit_confidence >= 75.0:  # Last exit had high confidence
                confidence_gap = entry_confidence - last_exit_confidence

                if confidence_gap < self.min_confidence_gap:
                    if reason_if_blocked:
                        return False, (
                            f"⚖️ Confidence gap too small: Entry {entry_confidence:.1f}% vs "
                            f"Last Exit {last_exit_confidence:.1f}% (need +{self.min_confidence_gap}% gap)"
                        )
                    return False, None

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
        # Get active positions
        try:
            active_positions = await db_client.pool.fetch(
                "SELECT quantity, entry_price FROM active_position"
            )

            if not active_positions or len(active_positions) == 0:
                portfolio_exposure = 0.0
                total_position_value = 0.0
            else:
                # Calculate total position value
                total_position_value = sum(
                    abs(float(p['quantity']) * float(p['entry_price']))
                    for p in active_positions
                )

                # Get current capital
                capital_result = await db_client.pool.fetchrow(
                    "SELECT capital FROM bot_state ORDER BY updated_at DESC LIMIT 1"
                )
                current_capital = float(capital_result['capital']) if capital_result else 1000.0

                # Calculate exposure ratio
                portfolio_exposure = total_position_value / current_capital if current_capital > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating portfolio exposure: {e}")
            # Fail-safe: assume medium exposure
            portfolio_exposure = 0.35
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

        elif portfolio_exposure >= self.high_exposure_threshold:
            # High risk - need high confidence
            required_confidence = 70.0
            if confidence < required_confidence:
                if reason_if_blocked:
                    return False, (
                        f"🎯 High exposure ({portfolio_exposure:.1%}) requires "
                        f"confidence ≥{required_confidence:.0f}% (got {confidence:.1f}%)"
                    )
                return False, None

        elif portfolio_exposure >= self.medium_exposure_threshold:
            # Medium risk - need decent confidence
            required_confidence = 62.0
            if confidence < required_confidence:
                if reason_if_blocked:
                    return False, (
                        f"🎯 Medium exposure ({portfolio_exposure:.1%}) requires "
                        f"confidence ≥{required_confidence:.0f}% (got {confidence:.1f}%)"
                    )
                return False, None
        # else: Low exposure (<30%), use config's min_ai_confidence (58%)

        # ===== CHECK 2: Slot Budget =====
        slot_cost = self.calculate_slot_cost(confidence)
        used_slots = sum(cost for _, cost in self.recent_slot_usage)

        # ===== CHECK 3: Performance Momentum (Adaptive Budget) =====
        try:
            recent_pnl = await self._get_recent_pnl(db_client, hours=1)
        except Exception as e:
            logger.warning(f"Could not get recent P&L: {e}")
            recent_pnl = 0.0

        # Adjust budget based on performance
        adjusted_budget = self.hourly_slot_budget

        if recent_pnl > 5.0:  # Winning streak
            adjusted_budget *= 1.3  # +30% more trades
            logger.info(
                f"📈 Performance bonus: +${recent_pnl:.2f} last hour → "
                f"Slot budget increased to {adjusted_budget:.1f}"
            )
        elif recent_pnl < -15.0:  # Losing streak
            adjusted_budget *= 0.7  # -30% trades (defensive)
            logger.warning(
                f"📉 Drawdown protection: -${abs(recent_pnl):.2f} last hour → "
                f"Slot budget reduced to {adjusted_budget:.1f}"
            )

        # Check if we have budget for this trade
        if used_slots + slot_cost > adjusted_budget:
            if reason_if_blocked:
                # Find when oldest slot expires
                if self.recent_slot_usage:
                    oldest_slot_time = min(t for t, _ in self.recent_slot_usage)
                    time_until_available = 60 - ((now - oldest_slot_time).total_seconds() / 60)
                else:
                    time_until_available = 0

                return False, (
                    f"🎰 Slot budget full: {used_slots:.1f}/{adjusted_budget:.1f} used. "
                    f"This trade costs {slot_cost:.1f} slots (conf: {confidence:.1f}%). "
                    f"Wait {time_until_available:.0f}m OR find better opportunity "
                    f"(≥70% = 0.75 slots, ≥75% = 0.5 slots)"
                )
            return False, None

        # ===== ALL CHECKS PASSED! =====
        logger.info(
            f"✅ Risk budget approved | "
            f"Conf: {confidence:.1f}% | "
            f"Slot: {slot_cost:.1f} | "
            f"Budget: {used_slots:.1f}→{used_slots+slot_cost:.1f}/{adjusted_budget:.1f} | "
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

        logger.info(
            f"📝 Trade recorded: {symbol} @ {entry_time.strftime('%H:%M:%S')} | "
            f"Conf: {entry_confidence:.1f}% | "
            f"Slot cost: {slot_cost:.1f}"
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
