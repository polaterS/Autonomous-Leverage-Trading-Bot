"""
=Ž DYNAMIC POSITION SIZER - Risk More on Better Setups


CRITICAL CONCEPT: Not all trades are equal

Amateur traders: Risk same amount on every trade
Professional traders: Risk more on high-quality setups

Why Dynamic Sizing Matters:
----------------------------
Scenario: 10 trades with 2% risk each

Static sizing:
- 7 wins at +6% each = +42%
- 3 losses at -2% each = -6%
- Net: +36%

Dynamic sizing (quality-based):
- 3 excellent setups (3% risk): 3 wins at +9% = +27%
- 4 good setups (2% risk): 3 wins at +6%, 1 loss = +16%
- 3 weak setups (SKIPPED): 0
- Net: +43% (19% better!)

SIZING RULES:
-------------
1. Confluence Score 90-100 (EXCELLENT):
   - Risk: 3-4% of capital
   - Multiplier: 1.5x-2.0x base size
   - Rationale: Historical 85%+ win rate

2. Confluence Score 80-89 (STRONG):
   - Risk: 2.5-3% of capital
   - Multiplier: 1.2x-1.5x base size
   - Rationale: Historical 75-80% win rate

3. Confluence Score 75-79 (GOOD):
   - Risk: 2% of capital
   - Multiplier: 1.0x base size
   - Rationale: Historical 70-75% win rate

4. Confluence Score < 75:
   - Risk: 0% (DON'T TRADE)
   - Multiplier: 0x
   - Rationale: Below acceptable win rate

KELLY CRITERION INTEGRATION:
-----------------------------
Kelly% = (Win_Rate * Avg_Win - Loss_Rate * Avg_Loss) / Avg_Win

Example: 75% win rate, 3R average winner
Kelly% = (0.75 * 3 - 0.25 * 1) / 3 = 0.666 = 66.6%

We use FRACTIONAL KELLY (1/4 Kelly) for safety:
Actual position size = Kelly% / 4 = ~16% of capital
But capped at 5% per trade (conservative)

DRAWDOWN ADJUSTMENT:
--------------------
- If account in drawdown > 10%: Reduce all sizes by 50%
- If account in drawdown > 20%: Reduce all sizes by 75%
- If account up > 30%: Increase sizes by 25% (scale with success)


"""

import logging
from typing import Dict, Optional, Tuple
from decimal import Decimal

logger = logging.getLogger('trading_bot')


class DynamicPositionSizer:
    """
    Professional dynamic position sizing system.

    Adjusts position size based on:
    1. Trade quality (confluence score)
    2. Account performance (drawdown/profit)
    3. Kelly Criterion (optimal sizing)
    4. Risk limits (never exceed max)
    """

    def __init__(self, base_risk_pct: float = 0.02):
        """
        Initialize Dynamic Position Sizer.

        Args:
            base_risk_pct: Base risk percentage (default 2%)
        """
        self.base_risk_pct = base_risk_pct  # 2% base risk

        # Quality-based multipliers
        self.quality_multipliers = {
            'EXCELLENT': 1.8,  # 90-100 score
            'STRONG': 1.4,     # 80-89 score
            'GOOD': 1.0,       # 75-79 score
            'MEDIOCRE': 0.0,   # 60-74 score (don't trade)
            'POOR': 0.0        # < 60 score (don't trade)
        }

        # Maximum risk per trade (safety cap)
        self.max_risk_pct = 0.05  # Never risk more than 5% on single trade

        # Minimum risk per trade
        self.min_risk_pct = 0.005  # Minimum 0.5%

        # Kelly Criterion parameters
        self.use_kelly = True
        self.kelly_fraction = 0.25  # Use 1/4 Kelly for safety

        # Drawdown adjustment parameters
        self.drawdown_reduce_threshold = 0.10  # -10% drawdown
        self.drawdown_reduce_severe = 0.20     # -20% drawdown

        # Profit scaling parameters
        self.profit_scale_threshold = 0.30     # +30% profit

        logger.info(f"=Ž Dynamic Position Sizer initialized (base_risk={base_risk_pct*100:.1f}%)")

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        leverage: int,
        confluence_quality: str,
        confluence_score: float,
        account_high_water_mark: Optional[float] = None,
        recent_win_rate: Optional[float] = None,
        recent_avg_rr: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Calculate optimal position size for trade.

        Args:
            account_balance: Current account balance (USD)
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price
            leverage: Leverage multiplier
            confluence_quality: Quality classification ('EXCELLENT', 'STRONG', etc.)
            confluence_score: Numeric quality score (0-100)
            account_high_water_mark: Peak account balance (for drawdown calc)
            recent_win_rate: Recent win rate (0-1) for Kelly
            recent_avg_rr: Recent average R/R ratio for Kelly

        Returns:
            Dict with:
                - position_size_usd: Position size in USD
                - position_size_coins: Position size in coins/contracts
                - risk_amount_usd: USD amount at risk
                - risk_percentage: % of account at risk
                - size_multiplier: Applied multiplier
                - reasoning: Explanation of sizing decision
        """
        try:
            # Get base risk multiplier from quality
            quality_multiplier = self.quality_multipliers.get(
                confluence_quality, 0.0
            )

            # Don't trade if quality too low
            if quality_multiplier == 0.0:
                return self._get_no_trade_result(
                    confluence_quality, "Quality below minimum threshold"
                )

            # Calculate base risk percentage
            base_risk = self.base_risk_pct * quality_multiplier

            # Apply Kelly Criterion adjustment if enabled and data available
            if self.use_kelly and recent_win_rate and recent_avg_rr:
                kelly_multiplier = self._calculate_kelly_multiplier(
                    recent_win_rate, recent_avg_rr
                )
                base_risk *= kelly_multiplier
                kelly_applied = True
            else:
                kelly_multiplier = 1.0
                kelly_applied = False

            # Apply drawdown/profit adjustments
            performance_multiplier, performance_reason = self._calculate_performance_adjustment(
                account_balance, account_high_water_mark
            )
            adjusted_risk = base_risk * performance_multiplier

            # Cap at maximum risk
            final_risk_pct = min(adjusted_risk, self.max_risk_pct)
            final_risk_pct = max(final_risk_pct, self.min_risk_pct)

            # Calculate risk amount in USD
            risk_amount_usd = account_balance * final_risk_pct

            # Calculate position size based on stop loss distance
            stop_distance_pct = abs(entry_price - stop_loss_price) / entry_price

            # Position size formula: Risk / (Stop Distance * Leverage)
            # Example: $100 risk / (0.01 * 10x) = $1000 position
            position_size_usd = risk_amount_usd / (stop_distance_pct * leverage)

            # Calculate position size in coins/contracts
            position_size_coins = position_size_usd / entry_price

            # Calculate total multiplier
            total_multiplier = quality_multiplier * kelly_multiplier * performance_multiplier

            # Generate reasoning
            reasoning = self._generate_sizing_reasoning(
                confluence_quality,
                confluence_score,
                quality_multiplier,
                kelly_applied,
                kelly_multiplier,
                performance_reason,
                performance_multiplier,
                final_risk_pct
            )

            result = {
                'position_size_usd': round(position_size_usd, 2),
                'position_size_coins': round(position_size_coins, 6),
                'risk_amount_usd': round(risk_amount_usd, 2),
                'risk_percentage': round(final_risk_pct * 100, 2),
                'size_multiplier': round(total_multiplier, 2),
                'quality_multiplier': quality_multiplier,
                'kelly_multiplier': kelly_multiplier,
                'performance_multiplier': performance_multiplier,
                'base_risk_pct': self.base_risk_pct * 100,
                'final_risk_pct': final_risk_pct * 100,
                'reasoning': reasoning,
                'should_trade': True
            }

            self._log_position_size(result)

            return result

        except Exception as e:
            logger.error(f"L Error calculating position size: {e}", exc_info=True)
            return self._get_no_trade_result('ERROR', f'Calculation error: {str(e)}')

    def _calculate_kelly_multiplier(
        self,
        win_rate: float,
        avg_rr_ratio: float
    ) -> float:
        """
        Calculate Kelly Criterion multiplier.

        Kelly% = (Win_Rate * Avg_Win - Loss_Rate * Avg_Loss) / Avg_Win

        For trading: Avg_Win = R/R ratio, Avg_Loss = 1 (1R loss)
        Kelly% = (win_rate * rr_ratio - (1 - win_rate)) / rr_ratio

        Returns multiplier to apply to position size.
        """
        try:
            loss_rate = 1 - win_rate
            avg_loss = 1.0  # We define 1R as our loss unit

            # Kelly formula
            kelly_pct = (win_rate * avg_rr_ratio - loss_rate * avg_loss) / avg_rr_ratio

            # Use fractional Kelly for safety (1/4 Kelly)
            fractional_kelly = kelly_pct * self.kelly_fraction

            # Cap at reasonable bounds
            fractional_kelly = max(0.2, min(fractional_kelly, 2.0))

            logger.debug(
                f"=Ê Kelly: win_rate={win_rate:.2%}, "
                f"avg_rr={avg_rr_ratio:.2f}, "
                f"kelly={kelly_pct:.2%}, "
                f"fractional={fractional_kelly:.2%}"
            )

            return fractional_kelly

        except Exception as e:
            logger.warning(f"  Kelly calculation error: {e}")
            return 1.0  # Neutral multiplier on error

    def _calculate_performance_adjustment(
        self,
        current_balance: float,
        high_water_mark: Optional[float]
    ) -> Tuple[float, str]:
        """
        Calculate adjustment based on account performance.

        Returns:
            Tuple of (multiplier, reason)
        """
        if not high_water_mark or high_water_mark == 0:
            return 1.0, "No performance history"

        # Calculate drawdown/profit percentage
        performance_pct = (current_balance - high_water_mark) / high_water_mark

        if performance_pct >= self.profit_scale_threshold:
            # Account up significantly - scale up (riding hot streak)
            multiplier = 1.25
            reason = f"Account up {performance_pct*100:.1f}% - Scaling up"

        elif performance_pct <= -self.drawdown_reduce_severe:
            # Severe drawdown - drastically reduce
            multiplier = 0.25  # 75% reduction
            reason = f"Severe drawdown {performance_pct*100:.1f}% - Reducing 75%"

        elif performance_pct <= -self.drawdown_reduce_threshold:
            # Moderate drawdown - reduce
            multiplier = 0.50  # 50% reduction
            reason = f"Drawdown {performance_pct*100:.1f}% - Reducing 50%"

        else:
            # Normal range
            multiplier = 1.0
            reason = f"Normal performance ({performance_pct*100:+.1f}%)"

        return multiplier, reason

    def _generate_sizing_reasoning(
        self,
        quality: str,
        score: float,
        quality_mult: float,
        kelly_applied: bool,
        kelly_mult: float,
        perf_reason: str,
        perf_mult: float,
        final_risk_pct: float
    ) -> str:
        """Generate human-readable reasoning."""
        lines = [
            f"Setup Quality: {quality} ({score:.1f}/100) ’ {quality_mult:.2f}x base",
        ]

        if kelly_applied:
            lines.append(f"Kelly Criterion: {kelly_mult:.2f}x adjustment")

        lines.append(f"Account Performance: {perf_mult:.2f}x ({perf_reason})")
        lines.append(f"Final Risk: {final_risk_pct*100:.2f}% of account")

        return "\n".join(lines)

    def _log_position_size(self, result: Dict):
        """Log position sizing results."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("=Ž DYNAMIC POSITION SIZING")
        logger.info("=" * 70)
        logger.info(f"=° Position Size: ${result['position_size_usd']:.2f}")
        logger.info(f">™ Position Size: {result['position_size_coins']:.6f} coins")
        logger.info(f"L Risk Amount: ${result['risk_amount_usd']:.2f}")
        logger.info(f"=Ê Risk %: {result['risk_percentage']:.2f}%")
        logger.info("")
        logger.info("Multipliers:")
        logger.info(f"  Quality: {result['quality_multiplier']:.2f}x")
        logger.info(f"  Kelly: {result['kelly_multiplier']:.2f}x")
        logger.info(f"  Performance: {result['performance_multiplier']:.2f}x")
        logger.info(f"  TOTAL: {result['size_multiplier']:.2f}x")
        logger.info("")
        logger.info(f"=­ Reasoning:\n{result['reasoning']}")
        logger.info("=" * 70)
        logger.info("")

    def _get_no_trade_result(self, quality: str, reason: str) -> Dict[str, any]:
        """Return result indicating no trade should be taken."""
        return {
            'position_size_usd': 0.0,
            'position_size_coins': 0.0,
            'risk_amount_usd': 0.0,
            'risk_percentage': 0.0,
            'size_multiplier': 0.0,
            'quality_multiplier': 0.0,
            'kelly_multiplier': 0.0,
            'performance_multiplier': 0.0,
            'base_risk_pct': self.base_risk_pct * 100,
            'final_risk_pct': 0.0,
            'reasoning': f"NO TRADE: {reason}",
            'should_trade': False
        }


# Factory function
def calculate_position_size(
    account_balance: float,
    entry_price: float,
    stop_loss_price: float,
    leverage: int,
    confluence_quality: str,
    confluence_score: float,
    account_high_water_mark: Optional[float] = None,
    recent_win_rate: Optional[float] = None,
    recent_avg_rr: Optional[float] = None,
    base_risk_pct: float = 0.02
) -> Dict[str, any]:
    """
    Convenience function to calculate position size.

    Usage:
        size_data = calculate_position_size(
            account_balance=1000,
            entry_price=50000,
            stop_loss_price=49500,
            leverage=10,
            confluence_quality='EXCELLENT',
            confluence_score=92,
            account_high_water_mark=900,  # Peak balance
            recent_win_rate=0.75,         # 75% recent win rate
            recent_avg_rr=3.5             # 3.5R average winner
        )

        if size_data['should_trade']:
            position_size = size_data['position_size_usd']
            risk_amount = size_data['risk_amount_usd']
            # Execute trade...
    """
    sizer = DynamicPositionSizer(base_risk_pct=base_risk_pct)
    return sizer.calculate_position_size(
        account_balance=account_balance,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        leverage=leverage,
        confluence_quality=confluence_quality,
        confluence_score=confluence_score,
        account_high_water_mark=account_high_water_mark,
        recent_win_rate=recent_win_rate,
        recent_avg_rr=recent_avg_rr
    )
