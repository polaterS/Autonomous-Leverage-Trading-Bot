"""
= DYNAMIC PROFIT TARGETS - Professional Edition


PROBLEM WITH FIXED TARGETS ($1.50):
------------------------------------
- $100 capital with 10x leverage = $1000 position
- $1.50 profit = 0.15% price movement needed
- BUT stop loss = 0.10% = ~$1-2 loss
- Risk/Reward becomes asymmetric: Risk $2 to make $1.50 = BAD

PROFESSIONAL APPROACH:
----------------------
Profit targets must be DYNAMIC based on:
1. ATR (Volatility) - More volatile = Bigger targets
2. Leverage - Higher leverage = More sensitive to small moves
3. Distance to Resistance - Don't set target beyond resistance
4. Risk/Reward Ratio - Minimum 3:1 for leveraged trading
5. Market Regime - Trending market = Let winners run longer

3-TIER PROFIT TAKING:
----------------------
TP1 (40% of position): Quick scalp, 1.5-2R
TP2 (40% of position): Main target, 3-4R
TP3 (20% of position): Let it run, trailing stop

This maximizes profit while securing gains.


"""

import logging
from typing import Dict, List, Tuple, Optional
from decimal import Decimal

logger = logging.getLogger('trading_bot')


class DynamicProfitTargetCalculator:
    """
    Professional dynamic profit target calculation.

    Calculates optimal profit targets based on market conditions,
    volatility, leverage, and risk parameters.
    """

    def __init__(self):
        # Base R multiples for 3-tier system
        self.tp1_r_multiple = 1.5  # Quick scalp
        self.tp2_r_multiple = 3.0  # Main target
        self.tp3_r_multiple = 5.0  # Let it run

        # Position allocation per tier
        self.tp1_allocation = 0.40  # 40% at TP1
        self.tp2_allocation = 0.40  # 40% at TP2
        self.tp3_allocation = 0.20  # 20% runner

        # Minimum R/R ratio for leveraged trading
        self.min_rr_ratio = 3.0

        # ATR multipliers for target calculation
        self.atr_multiplier_base = 2.0  # Base: 2x ATR for profit target

        logger.info("= Dynamic Profit Target Calculator initialized")

    def calculate_profit_targets(
        self,
        entry_price: float,
        stop_loss_price: float,
        side: str,
        atr: float,
        leverage: int,
        position_value_usd: float,
        resistance_price: Optional[float] = None,
        support_price: Optional[float] = None,
        market_regime: Optional[str] = None,
        volatility_level: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Calculate dynamic profit targets (3-tier system).

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            side: 'LONG' or 'SHORT'
            atr: Average True Range (volatility measure)
            leverage: Leverage multiplier
            position_value_usd: Position size in USD
            resistance_price: Next resistance level (for LONG)
            support_price: Next support level (for SHORT)
            market_regime: Current market regime
            volatility_level: Current volatility level

        Returns:
            Dict with:
                - tp1, tp2, tp3: Price levels
                - tp1_usd, tp2_usd, tp3_usd: USD profit amounts
                - tp1_allocation, tp2_allocation, tp3_allocation: % of position
                - total_potential_profit_usd: If all TPs hit
                - risk_usd: USD risk amount
                - rr_ratio: Overall risk/reward ratio
        """
        try:
            # Calculate risk amount
            risk_distance = abs(entry_price - stop_loss_price)
            risk_usd = (risk_distance / entry_price) * position_value_usd * leverage

            # Get ATR multiplier based on market conditions
            atr_multiplier = self._get_atr_multiplier(market_regime, volatility_level)

            # Calculate base target distance (ATR-based)
            target_distance_base = atr * atr_multiplier

            # Adjust for leverage (higher leverage = more sensitive)
            leverage_adjustment = 1.0 + (leverage / 50)  # 10x = 1.2x, 25x = 1.5x
            target_distance_adjusted = target_distance_base * leverage_adjustment

            # Calculate 3-tier targets
            tp1_distance = risk_distance * self.tp1_r_multiple
            tp2_distance = risk_distance * self.tp2_r_multiple
            tp3_distance = risk_distance * self.tp3_r_multiple

            # Ensure minimum ATR-based distance
            tp1_distance = max(tp1_distance, target_distance_adjusted * 0.5)
            tp2_distance = max(tp2_distance, target_distance_adjusted * 1.0)
            tp3_distance = max(tp3_distance, target_distance_adjusted * 1.5)

            # Calculate TP prices
            if side == 'LONG':
                tp1_price = entry_price + tp1_distance
                tp2_price = entry_price + tp2_distance
                tp3_price = entry_price + tp3_distance

                # Don't set target beyond resistance
                if resistance_price and resistance_price > entry_price:
                    resistance_buffer = resistance_price * 0.002  # 0.2% buffer before resistance
                    safe_resistance = resistance_price - resistance_buffer

                    # Adjust targets to not exceed resistance
                    if tp1_price > safe_resistance:
                        tp1_price = safe_resistance * 0.90  # TP1 at 90% to resistance
                    if tp2_price > safe_resistance:
                        tp2_price = safe_resistance  # TP2 at resistance
                    if tp3_price > safe_resistance:
                        # TP3 can go beyond resistance (breakout scenario)
                        tp3_price = resistance_price + (resistance_price * 0.01)  # +1% beyond

            else:  # SHORT
                tp1_price = entry_price - tp1_distance
                tp2_price = entry_price - tp2_distance
                tp3_price = entry_price - tp3_distance

                # Don't set target beyond support
                if support_price and support_price < entry_price:
                    support_buffer = support_price * 0.002  # 0.2% buffer before support
                    safe_support = support_price + support_buffer

                    # Adjust targets to not exceed support
                    if tp1_price < safe_support:
                        tp1_price = safe_support * 1.10  # TP1 at 110% to support
                    if tp2_price < safe_support:
                        tp2_price = safe_support  # TP2 at support
                    if tp3_price < safe_support:
                        # TP3 can go beyond support (breakdown scenario)
                        tp3_price = support_price - (support_price * 0.01)  # -1% beyond

            # Calculate USD profit for each tier
            tp1_price_change_pct = abs(tp1_price - entry_price) / entry_price
            tp2_price_change_pct = abs(tp2_price - entry_price) / entry_price
            tp3_price_change_pct = abs(tp3_price - entry_price) / entry_price

            tp1_usd = (tp1_price_change_pct * position_value_usd * leverage) * self.tp1_allocation
            tp2_usd = (tp2_price_change_pct * position_value_usd * leverage) * self.tp2_allocation
            tp3_usd = (tp3_price_change_pct * position_value_usd * leverage) * self.tp3_allocation

            total_potential_profit = tp1_usd + tp2_usd + tp3_usd

            # Calculate overall R/R ratio
            rr_ratio = total_potential_profit / risk_usd if risk_usd > 0 else 0

            # Validate minimum R/R
            if rr_ratio < self.min_rr_ratio:
                logger.warning(f" R/R ratio {rr_ratio:.2f} below minimum {self.min_rr_ratio}. Adjusting targets...")
                # Scale up targets to meet minimum R/R
                scale_factor = self.min_rr_ratio / rr_ratio
                if side == 'LONG':
                    tp1_price = entry_price + (tp1_distance * scale_factor)
                    tp2_price = entry_price + (tp2_distance * scale_factor)
                    tp3_price = entry_price + (tp3_distance * scale_factor)
                else:
                    tp1_price = entry_price - (tp1_distance * scale_factor)
                    tp2_price = entry_price - (tp2_distance * scale_factor)
                    tp3_price = entry_price - (tp3_distance * scale_factor)

                # Recalculate USD profits
                tp1_price_change_pct = abs(tp1_price - entry_price) / entry_price
                tp2_price_change_pct = abs(tp2_price - entry_price) / entry_price
                tp3_price_change_pct = abs(tp3_price - entry_price) / entry_price

                tp1_usd = (tp1_price_change_pct * position_value_usd * leverage) * self.tp1_allocation
                tp2_usd = (tp2_price_change_pct * position_value_usd * leverage) * self.tp2_allocation
                tp3_usd = (tp3_price_change_pct * position_value_usd * leverage) * self.tp3_allocation

                total_potential_profit = tp1_usd + tp2_usd + tp3_usd
                rr_ratio = total_potential_profit / risk_usd if risk_usd > 0 else 0

            result = {
                # Price levels
                'tp1_price': round(tp1_price, 8),
                'tp2_price': round(tp2_price, 8),
                'tp3_price': round(tp3_price, 8),

                # USD profits per tier
                'tp1_usd': round(tp1_usd, 2),
                'tp2_usd': round(tp2_usd, 2),
                'tp3_usd': round(tp3_usd, 2),

                # Allocations
                'tp1_allocation': self.tp1_allocation,
                'tp2_allocation': self.tp2_allocation,
                'tp3_allocation': self.tp3_allocation,

                # Risk/Reward
                'total_potential_profit_usd': round(total_potential_profit, 2),
                'risk_usd': round(risk_usd, 2),
                'rr_ratio': round(rr_ratio, 2),

                # Additional info
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'side': side,
                'meets_min_rr': rr_ratio >= self.min_rr_ratio
            }

            self._log_profit_targets(result)

            return result

        except Exception as e:
            logger.error(f"L Error calculating profit targets: {e}", exc_info=True)
            return self._get_fallback_targets(entry_price, stop_loss_price, side, position_value_usd, leverage)

    def _get_atr_multiplier(self, market_regime: Optional[str], volatility_level: Optional[str]) -> float:
        """
        Get ATR multiplier based on market conditions.

        Trending markets = Larger multiplier (let winners run)
        Ranging markets = Smaller multiplier (quick scalps)
        """
        multiplier = self.atr_multiplier_base

        # Adjust for market regime
        if market_regime:
            if 'TRENDING' in market_regime.upper():
                multiplier *= 1.5  # 50% larger targets in trending markets
            elif 'RANGING' in market_regime.upper():
                multiplier *= 0.7  # 30% smaller targets in ranging markets

        # Adjust for volatility
        if volatility_level:
            if 'HIGH' in volatility_level.upper() or 'EXTREME' in volatility_level.upper():
                multiplier *= 1.3  # Larger targets in high volatility
            elif 'LOW' in volatility_level.upper():
                multiplier *= 0.8  # Smaller targets in low volatility

        return multiplier

    def calculate_single_target(
        self,
        entry_price: float,
        stop_loss_price: float,
        side: str,
        min_rr_ratio: float = 3.0
    ) -> float:
        """
        Calculate single profit target with minimum R/R ratio.

        Simpler version for quick calculations.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            side: 'LONG' or 'SHORT'
            min_rr_ratio: Minimum risk/reward ratio (default 3.0)

        Returns:
            Target price
        """
        try:
            risk_distance = abs(entry_price - stop_loss_price)
            target_distance = risk_distance * min_rr_ratio

            if side == 'LONG':
                return entry_price + target_distance
            else:
                return entry_price - target_distance

        except Exception as e:
            logger.error(f"L Error calculating single target: {e}")
            return entry_price

    def should_take_trade(
        self,
        risk_usd: float,
        potential_profit_usd: float,
        min_rr_ratio: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Determine if trade should be taken based on R/R.

        Args:
            risk_usd: USD risk amount
            potential_profit_usd: USD potential profit
            min_rr_ratio: Minimum R/R ratio (uses default if None)

        Returns:
            Tuple of (should_take: bool, reason: str)
        """
        if min_rr_ratio is None:
            min_rr_ratio = self.min_rr_ratio

        if risk_usd <= 0:
            return False, "Invalid risk amount"

        rr_ratio = potential_profit_usd / risk_usd

        if rr_ratio >= min_rr_ratio:
            return True, f"R/R ratio {rr_ratio:.2f} meets minimum {min_rr_ratio}"
        else:
            return False, f"R/R ratio {rr_ratio:.2f} below minimum {min_rr_ratio}"

    def _log_profit_targets(self, result: Dict):
        """Log profit target calculation results."""
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"= DYNAMIC PROFIT TARGETS - {result['side']}")
        logger.info("=" * 70)
        logger.info(f"= Entry: ${result['entry_price']:.2f}")
        logger.info(f"= Stop Loss: ${result['stop_loss_price']:.2f}")
        logger.info(f"L Risk: ${result['risk_usd']:.2f}")
        logger.info("")
        logger.info(f"< TP1 ({int(result['tp1_allocation']*100)}%): ${result['tp1_price']:.2f}  +${result['tp1_usd']:.2f}")
        logger.info(f"< TP2 ({int(result['tp2_allocation']*100)}%): ${result['tp2_price']:.2f}  +${result['tp2_usd']:.2f}")
        logger.info(f"< TP3 ({int(result['tp3_allocation']*100)}%): ${result['tp3_price']:.2f}  +${result['tp3_usd']:.2f}")
        logger.info("")
        logger.info(f"= Total Potential: +${result['total_potential_profit_usd']:.2f}")
        logger.info(f"= Risk/Reward: {result['rr_ratio']:.2f}:1")
        logger.info(f" Meets Min R/R ({self.min_rr_ratio}:1): {result['meets_min_rr']}")
        logger.info("=" * 70)
        logger.info("")

    def _get_fallback_targets(
        self,
        entry_price: float,
        stop_loss_price: float,
        side: str,
        position_value_usd: float,
        leverage: int
    ) -> Dict[str, any]:
        """Return simple fallback targets on error."""
        risk_distance = abs(entry_price - stop_loss_price)

        if side == 'LONG':
            tp1 = entry_price + (risk_distance * 2.0)
            tp2 = entry_price + (risk_distance * 3.0)
            tp3 = entry_price + (risk_distance * 4.0)
        else:
            tp1 = entry_price - (risk_distance * 2.0)
            tp2 = entry_price - (risk_distance * 3.0)
            tp3 = entry_price - (risk_distance * 4.0)

        risk_usd = (risk_distance / entry_price) * position_value_usd * leverage

        return {
            'tp1_price': round(tp1, 8),
            'tp2_price': round(tp2, 8),
            'tp3_price': round(tp3, 8),
            'tp1_usd': round(risk_usd * 2.0 * 0.4, 2),
            'tp2_usd': round(risk_usd * 3.0 * 0.4, 2),
            'tp3_usd': round(risk_usd * 4.0 * 0.2, 2),
            'tp1_allocation': 0.40,
            'tp2_allocation': 0.40,
            'tp3_allocation': 0.20,
            'total_potential_profit_usd': round(risk_usd * 2.8, 2),  # Weighted average
            'risk_usd': round(risk_usd, 2),
            'rr_ratio': 2.8,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'side': side,
            'meets_min_rr': False
        }


# Factory function
def calculate_dynamic_targets(
    entry_price: float,
    stop_loss_price: float,
    side: str,
    atr: float,
    leverage: int,
    position_value_usd: float,
    resistance_price: Optional[float] = None,
    support_price: Optional[float] = None,
    market_regime: Optional[str] = None,
    volatility_level: Optional[str] = None
) -> Dict[str, any]:
    """
    Convenience function to calculate dynamic profit targets.

    Usage:
        targets = calculate_dynamic_targets(
            entry_price=50000,
            stop_loss_price=49500,
            side='LONG',
            atr=250,
            leverage=10,
            position_value_usd=100,
            resistance_price=52000,
            market_regime='TRENDING_UP'
        )

        # Execute partial exits at each TP
        if price >= targets['tp1_price']:
            close_percentage = targets['tp1_allocation']  # Close 40%
        # ...
    """
    calculator = DynamicProfitTargetCalculator()
    return calculator.calculate_profit_targets(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        side=side,
        atr=atr,
        leverage=leverage,
        position_value_usd=position_value_usd,
        resistance_price=resistance_price,
        support_price=support_price,
        market_regime=market_regime,
        volatility_level=volatility_level
    )
