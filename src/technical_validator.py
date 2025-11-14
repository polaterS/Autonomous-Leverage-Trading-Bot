"""
🎯 Technical Validator - Pre-Trade Technical Analysis Validation

CRITICAL: This module performs 4-layer technical validation BEFORE opening positions.
Previously, technical data was collected but NOT validated before trade execution.

VALIDATION LAYERS:
1. Support/Resistance Distance Check
2. Volume Surge Confirmation
3. Order Flow Alignment
4. Multi-Timeframe Confluence

Requirements: 50% of checks must pass for trade approval (2 out of 4).
"""

from typing import Dict, List, Tuple, Optional, Any
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class TechnicalValidator:
    """
    Pre-trade technical validation system.

    Validates trades against 4 technical criteria before execution.
    """

    def __init__(self):
        # Validation thresholds
        self.min_pass_rate = 0.50  # 50% of checks must pass (2 out of 4)
        self.max_sr_distance = 0.03  # Max 3% from support/resistance
        self.min_volume_surge = 1.2  # Min 1.2x average volume
        self.min_order_flow_strength = 5.0  # Min 5% order flow imbalance

        logger.info(
            f"✅ TechnicalValidator initialized:\n"
            f"   - Min pass rate: {self.min_pass_rate:.0%} (2/4 checks)\n"
            f"   - Max S/R distance: {self.max_sr_distance:.0%}\n"
            f"   - Min volume surge: {self.min_volume_surge}x\n"
            f"   - Min order flow: {self.min_order_flow_strength}%"
        )

    def validate_entry(
        self,
        symbol: str,
        side: str,
        current_price: Decimal,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive pre-trade technical validation.

        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            current_price: Current market price
            market_data: Optional market data dict with indicators, S/R, order flow, MTF

        Returns:
            Dict with:
                - valid: bool (True if passed)
                - score: float (0-1, percentage of checks passed)
                - checks: List of (name, passed, reason) tuples
                - reason: str (summary)
        """

        if not market_data:
            logger.warning(f"⚠️ No market_data provided for {symbol}, skipping technical validation")
            return {
                'valid': True,  # Don't block if data unavailable
                'score': 0.0,
                'checks': [],
                'reason': 'No market data available for validation'
            }

        checks: List[Tuple[str, bool, str]] = []

        # ========================================================================
        # CHECK 1: SUPPORT/RESISTANCE DISTANCE
        # ========================================================================
        sr_check = self._check_support_resistance(
            side, current_price, market_data
        )
        checks.append(sr_check)

        # ========================================================================
        # CHECK 2: VOLUME SURGE CONFIRMATION
        # ========================================================================
        volume_check = self._check_volume_surge(market_data)
        checks.append(volume_check)

        # ========================================================================
        # CHECK 3: ORDER FLOW ALIGNMENT
        # ========================================================================
        order_flow_check = self._check_order_flow(side, market_data)
        checks.append(order_flow_check)

        # ========================================================================
        # CHECK 4: MULTI-TIMEFRAME CONFLUENCE
        # ========================================================================
        mtf_check = self._check_multi_timeframe(side, market_data)
        checks.append(mtf_check)

        # ========================================================================
        # CALCULATE VALIDATION SCORE
        # ========================================================================
        passed = sum(1 for _, result, _ in checks if result)
        total = len(checks)
        score = passed / total if total > 0 else 0.0

        is_valid = score >= self.min_pass_rate

        # Log results
        if is_valid:
            logger.info(
                f"✅ Technical validation PASSED: {symbol} {side} "
                f"({passed}/{total} checks = {score:.0%})"
            )
            for name, result, reason in checks:
                status = "✅" if result else "❌"
                logger.debug(f"   {status} {name}: {reason}")
        else:
            logger.warning(
                f"❌ Technical validation FAILED: {symbol} {side} "
                f"({passed}/{total} checks = {score:.0%})"
            )
            for name, result, reason in checks:
                status = "✅" if result else "❌"
                logger.warning(f"   {status} {name}: {reason}")

        return {
            'valid': is_valid,
            'score': score,
            'checks': checks,
            'reason': f"Technical validation: {passed}/{total} checks passed ({score:.0%})"
        }

    def _check_support_resistance(
        self,
        side: str,
        current_price: Decimal,
        market_data: Dict[str, Any]
    ) -> Tuple[str, bool, str]:
        """
        Check distance to nearest support/resistance level.

        LONG: Should be near support (not near resistance)
        SHORT: Should be near resistance (not near support)
        """
        try:
            indicators = market_data.get('indicators', {})
            sr_levels = indicators.get('15m', {}).get('support_resistance', {})

            if not sr_levels:
                return ('support_resistance', True, 'No S/R data available (pass)')

            if side == 'LONG':
                # LONG: Check distance to nearest resistance (should have room)
                resistance = sr_levels.get('nearest_resistance')

                if resistance is None:
                    return ('support_resistance', True, 'No resistance detected (pass)')

                resistance = Decimal(str(resistance))
                distance_to_resistance = abs(resistance - current_price) / current_price

                if distance_to_resistance < self.max_sr_distance:
                    return (
                        'support_resistance',
                        False,
                        f"Too close to resistance: ${float(resistance):.4f} "
                        f"({float(distance_to_resistance)*100:.1f}% < {self.max_sr_distance*100:.0f}%)"
                    )
                else:
                    return (
                        'support_resistance',
                        True,
                        f"Good distance to resistance: {float(distance_to_resistance)*100:.1f}%"
                    )

            else:  # SHORT
                # SHORT: Check distance to nearest support (should have room)
                support = sr_levels.get('nearest_support')

                if support is None:
                    return ('support_resistance', True, 'No support detected (pass)')

                support = Decimal(str(support))
                distance_to_support = abs(current_price - support) / current_price

                if distance_to_support < self.max_sr_distance:
                    return (
                        'support_resistance',
                        False,
                        f"Too close to support: ${float(support):.4f} "
                        f"({float(distance_to_support)*100:.1f}% < {self.max_sr_distance*100:.0f}%)"
                    )
                else:
                    return (
                        'support_resistance',
                        True,
                        f"Good distance to support: {float(distance_to_support)*100:.1f}%"
                    )

        except Exception as e:
            logger.error(f"S/R check failed: {e}")
            return ('support_resistance', True, f'Check error (pass): {e}')

    def _check_volume_surge(
        self,
        market_data: Dict[str, Any]
    ) -> Tuple[str, bool, str]:
        """
        Check if volume is surging (indicates strong move).
        """
        try:
            indicators = market_data.get('indicators', {})
            volume_data = indicators.get('15m', {})

            current_volume = volume_data.get('volume')
            volume_sma = volume_data.get('volume_sma')

            if current_volume is None or volume_sma is None:
                return ('volume_surge', True, 'No volume data (pass)')

            volume_ratio = current_volume / volume_sma if volume_sma > 0 else 0

            if volume_ratio < self.min_volume_surge:
                return (
                    'volume_surge',
                    False,
                    f"Low volume: {volume_ratio:.2f}x average (need ≥{self.min_volume_surge}x)"
                )
            else:
                return (
                    'volume_surge',
                    True,
                    f"Volume surge confirmed: {volume_ratio:.2f}x average"
                )

        except Exception as e:
            logger.error(f"Volume check failed: {e}")
            return ('volume_surge', True, f'Check error (pass): {e}')

    def _check_order_flow(
        self,
        side: str,
        market_data: Dict[str, Any]
    ) -> Tuple[str, bool, str]:
        """
        Check if order flow supports the trade direction.

        LONG: Need positive order flow (buy pressure)
        SHORT: Need negative order flow (sell pressure)
        """
        try:
            order_flow = market_data.get('order_flow', {})
            imbalance = order_flow.get('weighted_imbalance', 0)

            if side == 'LONG':
                # LONG: Need buy pressure (positive imbalance)
                if imbalance < self.min_order_flow_strength:
                    return (
                        'order_flow',
                        False,
                        f"Weak buy pressure: {imbalance:+.1f}% (need ≥+{self.min_order_flow_strength}%)"
                    )
                else:
                    return (
                        'order_flow',
                        True,
                        f"Strong buy pressure: {imbalance:+.1f}%"
                    )

            else:  # SHORT
                # SHORT: Need sell pressure (negative imbalance)
                if imbalance > -self.min_order_flow_strength:
                    return (
                        'order_flow',
                        False,
                        f"Weak sell pressure: {imbalance:+.1f}% (need ≤-{self.min_order_flow_strength}%)"
                    )
                else:
                    return (
                        'order_flow',
                        True,
                        f"Strong sell pressure: {imbalance:+.1f}%"
                    )

        except Exception as e:
            logger.error(f"Order flow check failed: {e}")
            return ('order_flow', True, f'Check error (pass): {e}')

    def _check_multi_timeframe(
        self,
        side: str,
        market_data: Dict[str, Any]
    ) -> Tuple[str, bool, str]:
        """
        Check if multiple timeframes agree with trade direction.
        """
        try:
            mtf = market_data.get('multi_timeframe', {})

            if not mtf:
                return ('multi_timeframe', True, 'No MTF data (pass)')

            confluence = mtf.get('confluence_analysis', {})
            trading_bias = confluence.get('trading_bias', 'NEUTRAL')
            timeframe_agreement = mtf.get('timeframe_agreement', {})

            # Count agreeing timeframes
            agreeing_tfs = sum(
                1 for tf_data in timeframe_agreement.values()
                if tf_data.get('trend_direction') == side
            )
            total_tfs = len(timeframe_agreement)

            # Need at least 50% timeframe agreement
            agreement_ratio = agreeing_tfs / total_tfs if total_tfs > 0 else 0

            # Check if trading bias contradicts our direction
            if side == 'LONG' and trading_bias == 'SHORT_PREFERRED':
                return (
                    'multi_timeframe',
                    False,
                    f"MTF bias SHORT_PREFERRED conflicts with LONG ({agreeing_tfs}/{total_tfs} TFs agree)"
                )
            elif side == 'SHORT' and trading_bias == 'LONG_PREFERRED':
                return (
                    'multi_timeframe',
                    False,
                    f"MTF bias LONG_PREFERRED conflicts with SHORT ({agreeing_tfs}/{total_tfs} TFs agree)"
                )
            elif agreement_ratio < 0.5:
                return (
                    'multi_timeframe',
                    False,
                    f"Weak MTF agreement: {agreeing_tfs}/{total_tfs} timeframes ({agreement_ratio:.0%})"
                )
            else:
                return (
                    'multi_timeframe',
                    True,
                    f"MTF aligned: {agreeing_tfs}/{total_tfs} timeframes agree ({agreement_ratio:.0%})"
                )

        except Exception as e:
            logger.error(f"MTF check failed: {e}")
            return ('multi_timeframe', True, f'Check error (pass): {e}')


# Singleton instance
_technical_validator: Optional[TechnicalValidator] = None


def get_technical_validator() -> TechnicalValidator:
    """Get or create technical validator instance."""
    global _technical_validator
    if _technical_validator is None:
        _technical_validator = TechnicalValidator()
    return _technical_validator
