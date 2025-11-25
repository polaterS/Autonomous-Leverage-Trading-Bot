"""
= ADVANCED TRAILING STOP SYSTEM - Let Winners Run, Protect Profits


PHILOSOPHY: "Cut losses short, let winners run"

The difference between mediocre and excellent trading:
- Mediocre: Takes +1R profit, then watches it go to +5R
- Excellent: Uses trailing stops to secure gains while riding trends

4-STAGE ADAPTIVE TRAILING SYSTEM:
----------------------------------

Stage 1: BREAK-EVEN (+1.5R)
- When profit reaches +1.5R, move stop to break-even
- No loss possible anymore (except gap/slippage)

Stage 2: 50% TRAIL (+2.5R)
- When profit reaches +2.5R, trail at 50% of profit
- If price goes to +4R, stop at +2R
- Secures significant profit while allowing continuation

Stage 3: 25% TRAIL (+4R)
- When profit reaches +4R, trail at 25% of profit
- If price goes to +8R, stop at +6R
- Gives more room for big wins

Stage 4: ATR TRAIL (+6R)
- When profit exceeds +6R, trail based on ATR
- Adapts to market volatility
- Maximum room for explosive moves

MOMENTUM AWARENESS:
-------------------
- If momentum weakening  Tighten trailing stop
- If momentum strengthening  Loosen trailing stop
- Uses RSI, MACD divergence, volume decline


"""

import logging
from typing import Dict, Optional, Tuple
from decimal import Decimal
from enum import Enum

logger = logging.getLogger('trading_bot')


class TrailingStage(Enum):
    """Trailing stop stages"""
    INITIAL = "INITIAL"           # No trailing yet (< +1.5R)
    BREAK_EVEN = "BREAK_EVEN"     # +1.5R reached, stop at entry
    TRAIL_50PCT = "TRAIL_50PCT"   # +2.5R reached, trail 50%
    TRAIL_25PCT = "TRAIL_25PCT"   # +4R reached, trail 25%
    ATR_TRAIL = "ATR_TRAIL"       # +6R reached, ATR-based trailing


class AdvancedTrailingStop:
    """
    Professional 4-stage adaptive trailing stop system.

    Maximizes profit while protecting gains through progressive trailing.
    """

    def __init__(self):
        # Stage thresholds (in R multiples)
        self.break_even_threshold = 1.5    # Move to BE at +1.5R
        self.trail_50pct_threshold = 2.5   # Start 50% trail at +2.5R
        self.trail_25pct_threshold = 4.0   # Start 25% trail at +4R
        self.atr_trail_threshold = 6.0     # Start ATR trail at +6R

        # ATR trail multiplier
        self.atr_trail_multiplier = 1.5  # Trail 1.5x ATR behind current price

        # Momentum tightening parameters
        self.momentum_tighten_enabled = True
        self.momentum_loosen_enabled = True

        logger.info("= Advanced Trailing Stop System initialized")

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        initial_stop_loss: float,
        side: str,
        atr: float,
        indicators: Optional[Dict] = None,
        position_age_minutes: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Calculate adaptive trailing stop level.

        Args:
            entry_price: Original entry price
            current_price: Current market price
            initial_stop_loss: Initial stop loss price
            side: 'LONG' or 'SHORT'
            atr: Average True Range (volatility)
            indicators: Optional technical indicators for momentum analysis
            position_age_minutes: How long position has been open

        Returns:
            Dict with:
                - trailing_stop_price: Recommended stop loss price
                - stage: Current trailing stage
                - profit_r: Current profit in R multiples
                - locked_in_profit_r: Profit secured by trailing stop
                - should_tighten: Momentum suggests tightening
                - reasoning: Explanation
        """
        try:
            # Calculate current profit in R
            risk_amount = abs(entry_price - initial_stop_loss)
            if side == 'LONG':
                unrealized_profit = current_price - entry_price
            else:
                unrealized_profit = entry_price - current_price

            profit_r = unrealized_profit / risk_amount if risk_amount > 0 else 0

            # Determine current stage
            stage = self._determine_stage(profit_r)

            # Calculate trailing stop based on stage
            trailing_stop_price, locked_in_r = self._calculate_stage_stop(
                stage=stage,
                entry_price=entry_price,
                current_price=current_price,
                initial_stop_loss=initial_stop_loss,
                side=side,
                risk_amount=risk_amount,
                atr=atr
            )

            # Check momentum conditions
            should_tighten, should_loosen, momentum_reason = self._analyze_momentum(
                indicators, side, profit_r
            )

            # Apply momentum adjustments
            if should_tighten and self.momentum_tighten_enabled:
                trailing_stop_price = self._tighten_stop(
                    trailing_stop_price, current_price, side, atr
                )
                momentum_action = "TIGHTENED"
            elif should_loosen and self.momentum_loosen_enabled:
                trailing_stop_price = self._loosen_stop(
                    trailing_stop_price, current_price, side, atr
                )
                momentum_action = "LOOSENED"
            else:
                momentum_action = "NORMAL"

            # Ensure stop never moves against us
            trailing_stop_price = self._ensure_stop_never_worse(
                trailing_stop_price, initial_stop_loss, side
            )

            # Generate reasoning
            reasoning = self._generate_reasoning(
                stage, profit_r, locked_in_r, momentum_action, momentum_reason
            )

            result = {
                'trailing_stop_price': round(trailing_stop_price, 8),
                'stage': stage.value,
                'profit_r': round(profit_r, 2),
                'locked_in_profit_r': round(locked_in_r, 2),
                'should_tighten': should_tighten,
                'should_loosen': should_loosen,
                'momentum_action': momentum_action,
                'reasoning': reasoning,
                'entry_price': entry_price,
                'current_price': current_price,
                'side': side
            }

            self._log_trailing_stop(result)

            return result

        except Exception as e:
            logger.error(f"L Error calculating trailing stop: {e}", exc_info=True)
            # Fallback: Return initial stop loss
            return {
                'trailing_stop_price': initial_stop_loss,
                'stage': TrailingStage.INITIAL.value,
                'profit_r': 0,
                'locked_in_profit_r': 0,
                'should_tighten': False,
                'should_loosen': False,
                'momentum_action': 'ERROR',
                'reasoning': f'Error: {str(e)}',
                'entry_price': entry_price,
                'current_price': current_price,
                'side': side
            }

    def _determine_stage(self, profit_r: float) -> TrailingStage:
        """Determine which trailing stage we're in based on profit."""
        if profit_r >= self.atr_trail_threshold:
            return TrailingStage.ATR_TRAIL
        elif profit_r >= self.trail_25pct_threshold:
            return TrailingStage.TRAIL_25PCT
        elif profit_r >= self.trail_50pct_threshold:
            return TrailingStage.TRAIL_50PCT
        elif profit_r >= self.break_even_threshold:
            return TrailingStage.BREAK_EVEN
        else:
            return TrailingStage.INITIAL

    def _calculate_stage_stop(
        self,
        stage: TrailingStage,
        entry_price: float,
        current_price: float,
        initial_stop_loss: float,
        side: str,
        risk_amount: float,
        atr: float
    ) -> Tuple[float, float]:
        """
        Calculate trailing stop for current stage.

        Returns:
            Tuple of (stop_price, locked_in_profit_r)
        """
        if stage == TrailingStage.INITIAL:
            # No trailing yet
            return initial_stop_loss, 0.0

        elif stage == TrailingStage.BREAK_EVEN:
            # Move stop to break-even
            locked_in_r = 0.0
            return entry_price, locked_in_r

        elif stage == TrailingStage.TRAIL_50PCT:
            # Trail at 50% of current profit
            if side == 'LONG':
                current_profit = current_price - entry_price
                stop_price = entry_price + (current_profit * 0.5)
                locked_in_r = (stop_price - entry_price) / risk_amount
            else:  # SHORT
                current_profit = entry_price - current_price
                stop_price = entry_price - (current_profit * 0.5)
                locked_in_r = (entry_price - stop_price) / risk_amount

            return stop_price, locked_in_r

        elif stage == TrailingStage.TRAIL_25PCT:
            # Trail at 25% of current profit
            if side == 'LONG':
                current_profit = current_price - entry_price
                stop_price = entry_price + (current_profit * 0.75)  # Keep 75%, trail 25%
                locked_in_r = (stop_price - entry_price) / risk_amount
            else:  # SHORT
                current_profit = entry_price - current_price
                stop_price = entry_price - (current_profit * 0.75)
                locked_in_r = (entry_price - stop_price) / risk_amount

            return stop_price, locked_in_r

        elif stage == TrailingStage.ATR_TRAIL:
            # Trail based on ATR (volatility-adjusted)
            atr_distance = atr * self.atr_trail_multiplier

            if side == 'LONG':
                stop_price = current_price - atr_distance
                locked_in_r = (stop_price - entry_price) / risk_amount
            else:  # SHORT
                stop_price = current_price + atr_distance
                locked_in_r = (entry_price - stop_price) / risk_amount

            return stop_price, locked_in_r

        else:
            # Fallback
            return initial_stop_loss, 0.0

    def _analyze_momentum(
        self,
        indicators: Optional[Dict],
        side: str,
        profit_r: float
    ) -> Tuple[bool, bool, str]:
        """
        Analyze momentum to determine if trailing should tighten/loosen.

        Returns:
            Tuple of (should_tighten, should_loosen, reason)
        """
        if not indicators:
            return False, False, "No indicators available"

        should_tighten = False
        should_loosen = False
        reasons = []

        try:
            # RSI divergence check
            rsi = indicators.get('rsi', 50)
            if side == 'LONG':
                if profit_r > 2 and rsi > 70:
                    should_tighten = True
                    reasons.append("RSI overbought in LONG")
                elif rsi < 40:
                    should_loosen = True
                    reasons.append("RSI still strong in LONG")
            else:  # SHORT
                if profit_r > 2 and rsi < 30:
                    should_tighten = True
                    reasons.append("RSI oversold in SHORT")
                elif rsi > 60:
                    should_loosen = True
                    reasons.append("RSI still strong in SHORT")

            # MACD momentum
            macd_histogram = indicators.get('macd_histogram', 0)
            if side == 'LONG' and macd_histogram < 0:
                should_tighten = True
                reasons.append("MACD turning negative in LONG")
            elif side == 'SHORT' and macd_histogram > 0:
                should_tighten = True
                reasons.append("MACD turning positive in SHORT")

            # Volume decline (momentum weakening)
            volume_trend = indicators.get('volume_trend', 'normal')
            if volume_trend == 'low' and profit_r > 2:
                should_tighten = True
                reasons.append("Volume declining")

            # SuperTrend flip (early warning)
            supertrend_signal = indicators.get('supertrend_signal', 'NEUTRAL')
            if side == 'LONG' and supertrend_signal == 'SELL':
                should_tighten = True
                reasons.append("SuperTrend flipped bearish")
            elif side == 'SHORT' and supertrend_signal == 'BUY':
                should_tighten = True
                reasons.append("SuperTrend flipped bullish")

            # Strong momentum continuation signals
            if side == 'LONG':
                if rsi > 50 and macd_histogram > 0 and volume_trend == 'high':
                    should_loosen = True
                    reasons.append("Strong bullish momentum")
            else:
                if rsi < 50 and macd_histogram < 0 and volume_trend == 'high':
                    should_loosen = True
                    reasons.append("Strong bearish momentum")

            reason_str = ", ".join(reasons) if reasons else "Neutral momentum"
            return should_tighten, should_loosen, reason_str

        except Exception as e:
            logger.warning(f" Momentum analysis error: {e}")
            return False, False, "Error analyzing momentum"

    def _tighten_stop(
        self,
        current_stop: float,
        current_price: float,
        side: str,
        atr: float
    ) -> float:
        """Tighten trailing stop (move closer to current price)."""
        tighten_distance = atr * 0.5  # Move to within 0.5 ATR

        if side == 'LONG':
            tightened_stop = current_price - tighten_distance
            # Only move stop up, never down
            return max(current_stop, tightened_stop)
        else:  # SHORT
            tightened_stop = current_price + tighten_distance
            # Only move stop down, never up
            return min(current_stop, tightened_stop)

    def _loosen_stop(
        self,
        current_stop: float,
        current_price: float,
        side: str,
        atr: float
    ) -> float:
        """Loosen trailing stop (give more room)."""
        loosen_distance = atr * 2.0  # Give 2 ATR room

        if side == 'LONG':
            loosened_stop = current_price - loosen_distance
            # Only move stop down (loosen), not up
            return min(current_stop, loosened_stop)
        else:  # SHORT
            loosened_stop = current_price + loosen_distance
            # Only move stop up (loosen), not down
            return max(current_stop, loosened_stop)

    def _ensure_stop_never_worse(
        self,
        new_stop: float,
        old_stop: float,
        side: str
    ) -> float:
        """
        Ensure trailing stop never moves against us.

        For LONG: Stop can only move UP
        For SHORT: Stop can only move DOWN
        """
        if side == 'LONG':
            return max(new_stop, old_stop)  # Take higher stop
        else:
            return min(new_stop, old_stop)  # Take lower stop

    def _generate_reasoning(
        self,
        stage: TrailingStage,
        profit_r: float,
        locked_in_r: float,
        momentum_action: str,
        momentum_reason: str
    ) -> str:
        """Generate human-readable reasoning."""
        stage_descriptions = {
            TrailingStage.INITIAL: "Initial stop loss (no trailing yet)",
            TrailingStage.BREAK_EVEN: "Stop moved to break-even - Risk eliminated",
            TrailingStage.TRAIL_50PCT: "50% trailing - Securing half of profits",
            TrailingStage.TRAIL_25PCT: "25% trailing - Giving room for big wins",
            TrailingStage.ATR_TRAIL: "ATR-based trailing - Maximum room for explosive moves"
        }

        base_reason = stage_descriptions.get(stage, "Unknown stage")

        return f"{base_reason}\n" \
               f"Current Profit: +{profit_r:.2f}R\n" \
               f"Locked In: +{locked_in_r:.2f}R\n" \
               f"Momentum: {momentum_action} ({momentum_reason})"

    def _log_trailing_stop(self, result: Dict):
        """Log trailing stop calculation."""
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"= TRAILING STOP UPDATE - {result['side']}")
        logger.info("=" * 70)
        logger.info(f"= Entry: ${result['entry_price']:.2f}")
        logger.info(f"= Current: ${result['current_price']:.2f}")
        logger.info(f"= Trailing Stop: ${result['trailing_stop_price']:.2f}")
        logger.info("")
        logger.info(f"= Stage: {result['stage']}")
        logger.info(f"= Current Profit: +{result['profit_r']:.2f}R")
        logger.info(f"= Locked In: +{result['locked_in_profit_r']:.2f}R")
        logger.info(f" Momentum: {result['momentum_action']}")
        logger.info("")
        logger.info(f"= Reasoning:\n{result['reasoning']}")
        logger.info("=" * 70)
        logger.info("")


# Factory function
def calculate_trailing_stop(
    entry_price: float,
    current_price: float,
    initial_stop_loss: float,
    side: str,
    atr: float,
    indicators: Optional[Dict] = None,
    position_age_minutes: Optional[int] = None
) -> Dict[str, any]:
    """
    Convenience function to calculate trailing stop.

    Usage:
        trailing_data = calculate_trailing_stop(
            entry_price=50000,
            current_price=52000,
            initial_stop_loss=49500,
            side='LONG',
            atr=250,
            indicators=indicator_dict
        )

        new_stop_loss = trailing_data['trailing_stop_price']
        profit_locked = trailing_data['locked_in_profit_r']

        # Update position stop loss if trailing stop is better
        if (side == 'LONG' and new_stop_loss > current_stop_loss) or \
           (side == 'SHORT' and new_stop_loss < current_stop_loss):
            update_position_stop_loss(new_stop_loss)
    """
    system = AdvancedTrailingStop()
    return system.calculate_trailing_stop(
        entry_price=entry_price,
        current_price=current_price,
        initial_stop_loss=initial_stop_loss,
        side=side,
        atr=atr,
        indicators=indicators,
        position_age_minutes=position_age_minutes
    )
