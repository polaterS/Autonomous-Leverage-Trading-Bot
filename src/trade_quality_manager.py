"""
🎯 Trade Quality Manager
Prevents overtrading, bad repeating trades, and entry/exit conflicts

SOLVES:
1. Same symbol being traded repeatedly in short time (NEAR 4x in 30 min)
2. Entry/Exit ML model conflicts (79% entry conf -> 80% exit conf after 30s)
3. Overtrading (too many trades per hour)
4. Low quality trades that repeatedly fail

Expected Impact: -70% unnecessary trades, +$50-100 saved from fees/slippage
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
    Manages trade quality by tracking:
    - Symbol-specific cooldowns (prevent NEAR 4x in 30 min)
    - Trade frequency limits (max 3/hour globally)
    - Symbol performance history (learn which symbols fail)
    - Entry/Exit confidence gaps (prevent conflicts)
    """

    def __init__(self):
        # Symbol-specific tracking
        self.symbol_last_trade: Dict[str, datetime] = {}  # symbol -> last trade time
        self.symbol_recent_results: Dict[str, List[Dict]] = defaultdict(list)  # Recent trade outcomes

        # Global trade frequency
        self.recent_trades: List[datetime] = []  # All recent trade times

        # Symbol cooldown settings
        self.symbol_cooldown_minutes = 30  # Wait 30 min before trading same symbol again
        self.symbol_cooldown_if_loss_minutes = 45  # Wait 45 min if last trade was a loss

        # Global frequency limits
        self.max_trades_per_hour = 4  # Max 4 trades/hour (down from unlimited)
        self.max_trades_per_30min = 3  # Max 3 trades/30min

        # Quality thresholds
        self.min_confidence_gap = 8.0  # Entry confidence must be 8% higher than recent exit confidence
        self.symbol_failure_threshold = 3  # If 3 losses in a row, increase cooldown

        logger.info(
            f"✅ TradeQualityManager initialized:\n"
            f"   - Symbol cooldown: {self.symbol_cooldown_minutes}min (normal), "
            f"{self.symbol_cooldown_if_loss_minutes}min (after loss)\n"
            f"   - Max trades: {self.max_trades_per_hour}/hour, {self.max_trades_per_30min}/30min\n"
            f"   - Min confidence gap: {self.min_confidence_gap}%"
        )

    def can_trade_symbol(
        self,
        symbol: str,
        entry_confidence: float,
        reason_if_blocked: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if we can trade this symbol now.

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

    def can_trade_now(self, reason_if_blocked: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Check if we can open ANY trade right now (global frequency limit).

        Returns:
            (can_trade, reason_if_not)
        """

        now = datetime.now()

        # Clean old trades (older than 1 hour)
        self.recent_trades = [
            t for t in self.recent_trades
            if (now - t).total_seconds() < 3600
        ]

        # Check 1: Trades in last hour
        trades_last_hour = len(self.recent_trades)
        if trades_last_hour >= self.max_trades_per_hour:
            if reason_if_blocked:
                oldest_trade = min(self.recent_trades)
                time_until_available = 60 - ((now - oldest_trade).total_seconds() / 60)
                return False, (
                    f"🚦 Hourly limit reached: {trades_last_hour}/{self.max_trades_per_hour} trades. "
                    f"Wait {time_until_available:.1f}m"
                )
            return False, None

        # Check 2: Trades in last 30 minutes (prevent bursts)
        trades_last_30min = sum(
            1 for t in self.recent_trades
            if (now - t).total_seconds() < 1800
        )

        if trades_last_30min >= self.max_trades_per_30min:
            if reason_if_blocked:
                return False, (
                    f"🚦 Short-term limit: {trades_last_30min}/{self.max_trades_per_30min} "
                    f"trades in 30min. Cool down."
                )
            return False, None

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
            db_client: Database client for recent trade history

        Returns:
            (is_valid, rejection_reason)
        """

        # Check global frequency
        can_trade_global, global_reason = self.can_trade_now(reason_if_blocked=True)
        if not can_trade_global:
            logger.warning(f"❌ Trade blocked (global): {global_reason}")
            return False, global_reason

        # Check symbol-specific
        can_trade_symbol, symbol_reason = self.can_trade_symbol(
            symbol, entry_confidence, reason_if_blocked=True
        )
        if not can_trade_symbol:
            logger.warning(f"❌ Trade blocked ({symbol}): {symbol_reason}")
            return False, symbol_reason

        # All checks passed
        logger.info(
            f"✅ Trade quality check PASSED: {symbol} "
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
            entry_confidence: Entry confidence
            entry_time: Entry time (defaults to now)
        """

        if entry_time is None:
            entry_time = datetime.now()

        # Update symbol tracking
        self.symbol_last_trade[symbol] = entry_time

        # Update global frequency tracking
        self.recent_trades.append(entry_time)

        logger.info(
            f"📝 Recorded trade open: {symbol} @ {entry_time.strftime('%H:%M:%S')} "
            f"(confidence: {entry_confidence:.1f}%)"
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
            f"📝 Recorded trade close: {symbol} - "
            f"{'✅ Win' if was_profitable else '❌ Loss'} "
            f"(P&L: ${pnl_usd:+.2f}, Exit conf: {exit_confidence:.1f}%)"
        )

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
        """Get global trading statistics."""

        now = datetime.now()

        # Clean old trades
        self.recent_trades = [
            t for t in self.recent_trades
            if (now - t).total_seconds() < 3600
        ]

        trades_last_hour = len(self.recent_trades)
        trades_last_30min = sum(
            1 for t in self.recent_trades
            if (now - t).total_seconds() < 1800
        )

        return {
            'trades_last_hour': trades_last_hour,
            'trades_last_30min': trades_last_30min,
            'max_per_hour': self.max_trades_per_hour,
            'max_per_30min': self.max_trades_per_30min,
            'slots_available_hour': max(0, self.max_trades_per_hour - trades_last_hour),
            'slots_available_30min': max(0, self.max_trades_per_30min - trades_last_30min)
        }


# Singleton instance
_quality_manager: Optional[TradeQualityManager] = None


def get_trade_quality_manager() -> TradeQualityManager:
    """Get or create trade quality manager instance."""
    global _quality_manager
    if _quality_manager is None:
        _quality_manager = TradeQualityManager()
    return _quality_manager
