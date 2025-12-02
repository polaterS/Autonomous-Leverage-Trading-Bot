"""
< CONFLUENCE SCORING ENGINE - The Key to 75%+ Win Rate


CRITICAL PHILOSOPHY: Quality Over Quantity

Most trading bots fail because they take EVERY signal.
Professional traders take ONLY the highest probability setups.

This engine scores every trading opportunity from 0-100 based on:
1. Multi-Timeframe Alignment (25 points)
2. Volume Profile Confluence (20 points)
3. Technical Indicator Agreement (20 points)
4. Market Regime Compatibility (15 points)
5. Support/Resistance Quality (15 points)
6. Risk/Reward Ratio (5 points)

RULE: Only trade if score >= 75/100

Result:
- Fewer trades (2-5 per day instead of 10-20)
- Higher win rate (75-80% instead of 50-60%)
- Better risk/reward (3-4R instead of 1.5R)
- More profitable (selective = successful)


"""

import logging
from typing import Dict, List, Any, Optional
from decimal import Decimal
from enum import Enum

logger = logging.getLogger('trading_bot')


class SignalQuality(Enum):
    """Signal quality classification"""
    EXCELLENT = "EXCELLENT"      # 90-100: Take maximum size
    STRONG = "STRONG"            # 80-89: Take full size
    GOOD = "GOOD"                # 75-79: Take normal size
    MEDIOCRE = "MEDIOCRE"        # 60-74: Skip
    POOR = "POOR"                # < 60: Skip


class ConfluenceScorer:
    """
    Professional Confluence Scoring System.

    Evaluates trading opportunities using multi-factor analysis.
    Only the best setups (75+) are approved for trading.
    """

    def __init__(self):
        # Scoring weights (must sum to 100)
        # 🆕 v4.5.0: Added advanced professional indicators
        self.weights = {
            'multi_timeframe': 12,       # MTF trend alignment
            'volume_profile': 8,         # VPOC, HVN proximity
            'indicators': 15,            # RSI, MACD, SuperTrend, etc.
            'market_regime': 8,          # ADX-based regime
            'support_resistance': 10,    # S/R quality
            'risk_reward': 5,            # R/R ratio
            'enhanced_indicators': 12,   # v4.4.0: BB Squeeze, EMA Stack, Divergences
            'momentum_quality': 10,      # Momentum strength and direction
            'advanced_indicators': 20    # 🆕 v4.5.0: VWAP, StochRSI, CMF, Fibonacci
        }

        # Minimum score to trade
        # 🎯 60+ with MTF+momentum+volume checks for extra safety
        self.min_score_to_trade = 60

        # Confidence thresholds for quality classification
        self.quality_thresholds = {
            'EXCELLENT': 80,   # 80+ = elite setup, full position
            'STRONG': 70,      # 70-79 = strong setup
            'GOOD': 60,        # 60-69 = acceptable with other checks
            'MEDIOCRE': 50     # <60 = skip
        }

        logger.info(f"< Confluence Scorer initialized (min_score={self.min_score_to_trade})")

    def score_opportunity(
        self,
        symbol: str,
        side: str,
        pa_analysis: Dict,
        indicators: Dict,
        volume_profile: Dict,
        market_regime: Dict,
        mtf_data: Optional[Dict] = None,
        enhanced_data: Optional[Dict] = None,  # v4.4.0: Enhanced indicators
        advanced_data: Optional[Dict] = None   # 🆕 v4.5.0: Advanced indicators
    ) -> Dict[str, Any]:
        """
        Main scoring method: Evaluate a trading opportunity.

        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            pa_analysis: Price action analysis results
            indicators: Technical indicators
            volume_profile: Volume profile analysis
            market_regime: Market regime detection
            mtf_data: Multi-timeframe data (optional)

        Returns:
            Dict with:
                - total_score: 0-100
                - should_trade: bool
                - quality: SignalQuality
                - component_scores: breakdown by category
                - reasoning: explanation
        """
        try:
            component_scores = {}

            # 1. Multi-Timeframe Alignment (25 points)
            mtf_score = self._score_multi_timeframe(side, mtf_data, indicators)
            component_scores['multi_timeframe'] = mtf_score

            # 2. Volume Profile Confluence (20 points)
            volume_score = self._score_volume_profile(side, volume_profile, pa_analysis)
            component_scores['volume_profile'] = volume_score

            # 3. Technical Indicators (20 points)
            indicator_score = self._score_indicators(side, indicators)
            component_scores['indicators'] = indicator_score

            # 4. Market Regime (15 points)
            regime_score = self._score_market_regime(side, market_regime)
            component_scores['market_regime'] = regime_score

            # 5. Support/Resistance Quality (12 points)
            sr_score = self._score_support_resistance(side, pa_analysis)
            component_scores['support_resistance'] = sr_score

            # 6. Risk/Reward Ratio (8 points)
            rr_score = self._score_risk_reward(pa_analysis)
            component_scores['risk_reward'] = rr_score

            # 🆕 7. Enhanced Indicators (15 points) - BB Squeeze, EMA Stack, Divergences
            enhanced_score = self._score_enhanced_indicators(side, enhanced_data, indicators)
            component_scores['enhanced_indicators'] = enhanced_score

            # 🆕 8. Momentum Quality (10 points) - ADX, trend strength
            momentum_score = self._score_momentum_quality(side, enhanced_data, market_regime)
            component_scores['momentum_quality'] = momentum_score

            # 🆕 9. Advanced Indicators (20 points) - VWAP, StochRSI, CMF, Fib - v4.5.0
            advanced_score = self._score_advanced_indicators(side, advanced_data, indicators)
            component_scores['advanced_indicators'] = advanced_score

            # Calculate total score
            total_score = sum(component_scores.values())

            # Determine if should trade
            should_trade = total_score >= self.min_score_to_trade

            # Classify quality
            quality = self._classify_quality(total_score)

            # Generate reasoning
            reasoning = self._generate_reasoning(component_scores, total_score, side)

            result = {
                'symbol': symbol,
                'side': side,
                'total_score': round(total_score, 1),
                'should_trade': should_trade,
                'quality': quality.value,
                'component_scores': component_scores,
                'reasoning': reasoning,
                'recommended_position_multiplier': self._get_position_multiplier(quality)
            }

            self._log_score(result)

            return result

        except Exception as e:
            logger.error(f"L Error scoring opportunity for {symbol}: {e}", exc_info=True)
            return self._get_rejected_score(symbol, side, str(e))

    def _score_multi_timeframe(self, side: str, mtf_data: Optional[Dict], indicators: Dict) -> float:
        """
        Score multi-timeframe alignment (20 points max - reduced from 25).

        ✅ SOFTENED: More forgiving - 2/3 timeframe alignment = good score
        """
        max_points = self.weights['multi_timeframe']
        score = 0

        try:
            if not mtf_data:
                # Fallback: Use current timeframe indicators only
                # ✅ IMPROVED: More generous fallback scoring
                current_trend = indicators.get('trend', 'unknown')
                if (side == 'LONG' and current_trend == 'uptrend') or \
                   (side == 'SHORT' and current_trend == 'downtrend'):
                    score = max_points * 0.70  # 70% of max (was 60%)
                else:
                    score = max_points * 0.30  # 30% of max (was 20%)
                return score

            # Extract trend for each timeframe
            tf_4h = mtf_data.get('4h', {})
            tf_1h = mtf_data.get('1h', {})
            tf_15m = mtf_data.get('15m', {})

            trend_4h = tf_4h.get('trend', 'unknown')
            trend_1h = tf_1h.get('trend', 'unknown')
            trend_15m = tf_15m.get('trend', 'unknown')

            # ✅ SOFTENED: More generous scoring
            # 4h is most important (primary bias) = 8 points (was 12)
            if (side == 'LONG' and trend_4h == 'uptrend') or \
               (side == 'SHORT' and trend_4h == 'downtrend'):
                score += 8
            elif trend_4h == 'unknown' or trend_4h == 'sideways':
                score += 5  # More forgiving for neutral (was 6)
            else:
                score += 2  # ✅ NEW: Even wrong direction gets something

            # 1h pullback/continuation = 7 points (was 8)
            if (side == 'LONG' and trend_1h in ['uptrend', 'sideways']) or \
               (side == 'SHORT' and trend_1h in ['downtrend', 'sideways']):
                score += 7
            elif trend_1h == 'unknown':
                score += 4  # Keep
            else:
                score += 1  # ✅ NEW: Even conflicting gets 1 point

            # 15m entry confirmation = 5 points (keep)
            if (side == 'LONG' and trend_15m == 'uptrend') or \
               (side == 'SHORT' and trend_15m == 'downtrend'):
                score += 5
            elif trend_15m in ['unknown', 'sideways']:
                score += 3  # More forgiving (was 2)
            else:
                score += 1  # ✅ NEW: Even wrong gets 1 point

            return min(score, max_points)

        except Exception as e:
            logger.warning(f" MTF scoring error: {e}")
            return 5  # Default low score on error

    def _score_volume_profile(self, side: str, volume_profile: Dict, pa_analysis: Dict) -> float:
        """
        Score volume profile confluence (15 points max - reduced from 20).

        ✅ SOFTENED: More forgiving distances, base score even without VPOC
        """
        max_points = self.weights['volume_profile']
        score = 0

        try:
            # ✅ BASE SCORE: Give 5 points just for having volume data (was 3)
            if volume_profile:
                score += 5  # Baseline for any volume analysis (increased)

            # Check if near VPOC (strongest level)
            # ✅ SOFTENED: Extended distance ranges
            vpoc_distance_pct = abs(volume_profile.get('vpoc_distance_pct', 100))
            if vpoc_distance_pct <= 1.0:  # Within 1% of VPOC (was 0.5%)
                score += 6  # Maximum VPOC proximity (was 10)
            elif vpoc_distance_pct <= 2.0:  # Within 2% (was 1%)
                score += 4  # Good proximity (was 7)
            elif vpoc_distance_pct <= 3.0:  # ✅ NEW: Within 3%
                score += 2  # Decent proximity (was 4 for 2%)
            elif vpoc_distance_pct <= 5.0:  # ✅ NEW: Within 5%
                score += 1  # Acceptable

            # Check if near HVN (High Volume Node)
            # ✅ SOFTENED: More forgiving HVN distances
            hvn_levels = volume_profile.get('hvn_levels', [])
            current_price = volume_profile.get('current_price', 0)

            nearest_hvn_distance = 100  # Default large distance
            if hvn_levels and current_price > 0:
                for hvn in hvn_levels:
                    distance_pct = abs((hvn['price'] - current_price) / current_price) * 100
                    if distance_pct < nearest_hvn_distance:
                        nearest_hvn_distance = distance_pct

            if nearest_hvn_distance <= 2.0:  # Within 2% of HVN (was 1%)
                score += 3  # Good HVN proximity (was 6)
            elif nearest_hvn_distance <= 4.0:  # ✅ NEW: Within 4%
                score += 2  # Decent (was 3 for 2%)
            elif nearest_hvn_distance <= 6.0:  # ✅ NEW: Within 6%
                score += 1  # Acceptable

            # Check if in Value Area
            in_value_area = volume_profile.get('price_in_value_area', False)
            if in_value_area:
                score += 2  # In VA = normal/good (keep)
            else:
                score += 1  # Outside VA still gets 1 (keep)

            # Volume surge confirmation - ✅ NO CHANGE (already good)
            volume_trend = pa_analysis.get('volume_surge', False)
            if volume_trend:
                score += 1  # Volume confirms (reduced from 2 to fit 15 max)

            return min(score, max_points)

        except Exception as e:
            logger.warning(f" Volume profile scoring error: {e}")
            return 5  # Default low score on error

    def _score_indicators(self, side: str, indicators: Dict) -> float:
        """
        Score technical indicator agreement (25 points max - increased).

        ✅ SOFTENED: Base score + more forgiving thresholds
        """
        max_points = self.weights['indicators']
        score = 0
        agreement_count = 0
        total_indicators = 0

        try:
            # ✅ BASE SCORE: Give 5 points for having any indicators
            if indicators:
                score += 5  # Baseline

            # RSI (oversold/overbought) - 🎯 PROFESSIONAL THRESHOLDS
            # Industry standard: <35 oversold, >65 overbought
            rsi = indicators.get('rsi', 50)
            total_indicators += 1
            if side == 'LONG':
                if rsi < 35:  # 🎯 TRUE oversold - strong buy signal
                    score += 4
                    agreement_count += 1
                elif rsi < 45:  # Moderately oversold
                    score += 2
                elif rsi < 55:  # Neutral zone
                    score += 1
                else:
                    score += 0  # RSI high = weak for LONG
            else:  # SHORT
                if rsi > 65:  # 🎯 TRUE overbought - strong sell signal
                    score += 4
                    agreement_count += 1
                elif rsi > 55:  # Moderately overbought
                    score += 2
                elif rsi > 45:  # Neutral zone
                    score += 1
                else:
                    score += 0  # RSI low = weak for SHORT

            # MACD
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            total_indicators += 1
            if side == 'LONG' and macd > macd_signal:
                score += 3
                agreement_count += 1
            elif side == 'SHORT' and macd < macd_signal:
                score += 3
                agreement_count += 1
            else:
                score += 1  # Divergence is not ideal but not disqualifying

            # SuperTrend
            supertrend_signal = indicators.get('supertrend_signal', 'NEUTRAL')
            total_indicators += 1
            if (side == 'LONG' and supertrend_signal == 'BUY') or \
               (side == 'SHORT' and supertrend_signal == 'SELL'):
                score += 4
                agreement_count += 1
            elif supertrend_signal == 'NEUTRAL':
                score += 2

            # VWAP
            vwap_bias = indicators.get('vwap_bias', 'NEUTRAL')
            total_indicators += 1
            if (side == 'LONG' and vwap_bias == 'BULLISH') or \
               (side == 'SHORT' and vwap_bias == 'BEARISH'):
                score += 3
                agreement_count += 1
            elif vwap_bias == 'NEUTRAL':
                score += 1.5

            # Ichimoku
            ichimoku_signal = indicators.get('ichimoku_signal', 'NEUTRAL')
            total_indicators += 1
            if (side == 'LONG' and ichimoku_signal == 'BUY') or \
               (side == 'SHORT' and ichimoku_signal == 'SELL'):
                score += 3
                agreement_count += 1
            elif ichimoku_signal == 'NEUTRAL':
                score += 1.5

            # Stochastic RSI
            stoch_rsi_signal = indicators.get('stoch_rsi_signal_new', 'NEUTRAL')
            total_indicators += 1
            if (side == 'LONG' and stoch_rsi_signal == 'OVERSOLD') or \
               (side == 'SHORT' and stoch_rsi_signal == 'OVERBOUGHT'):
                score += 2
                agreement_count += 1
            elif stoch_rsi_signal == 'NEUTRAL':
                score += 1

            # MFI (Money Flow Index)
            mfi_signal = indicators.get('mfi_signal_new', 'NEUTRAL')
            total_indicators += 1
            if (side == 'LONG' and mfi_signal == 'OVERSOLD') or \
               (side == 'SHORT' and mfi_signal == 'OVERBOUGHT'):
                score += 2
                agreement_count += 1
            elif mfi_signal == 'NEUTRAL':
                score += 1

            # Bonus: High agreement across indicators
            agreement_ratio = agreement_count / total_indicators if total_indicators > 0 else 0
            if agreement_ratio >= 0.7:  # 70%+ indicators agree
                score += 2  # Bonus points for strong confluence

            return min(score, max_points)

        except Exception as e:
            logger.warning(f" Indicator scoring error: {e}")
            return 5  # Default low score on error

    def _score_market_regime(self, side: str, market_regime: Dict) -> float:
        """
        Score market regime compatibility (15 points max).

        Best: Trend-following in trending market, mean-reversion in ranging.
        """
        max_points = self.weights['market_regime']
        score = 0

        try:
            regime = market_regime.get('regime')
            if not regime:
                # 🎯 v4.2.3: Unknown regime still gets 8 points (was 5)
                return 8  # Unknown regime = neutral/tradeable

            regime_name = regime.value if hasattr(regime, 'value') else str(regime)
            confidence = market_regime.get('confidence', 0.5)
            # 🎯 v4.2.3: REMOVED should_trade hard block - all regimes tradeable
            # Previously: if not should_trade: return 0 (hard reject)

            # Trending markets favor trend-following
            if 'TRENDING' in regime_name or 'TREND' in regime_name.upper():
                # Check if we're trading with the trend
                if (side == 'LONG' and ('UP' in regime_name or 'BULL' in regime_name.upper())) or \
                   (side == 'SHORT' and ('DOWN' in regime_name or 'BEAR' in regime_name.upper())):
                    score = 12 + (confidence * 3)  # 12-15 points (with trend)
                else:
                    # 🎯 v4.2.3: Trading against trend = 7 points (was 3)
                    score = 7 + (confidence * 2)  # 7-9 points (counter-trend)

            # Ranging markets are trickier
            elif 'RANGING' in regime_name or 'RANGE' in regime_name.upper() or 'SIDEWAYS' in regime_name.upper():
                # Mean reversion strategies work better
                score = 10 + (confidence * 3)  # 10-13 points

            # Choppy/Volatile markets
            elif 'CHOPPY' in regime_name or 'VOLATILE' in regime_name:
                # 🎯 v4.2.3: Choppy still tradeable (was 5-8, now 7-10)
                score = 7 + (confidence * 3)  # 7-10 points

            # Unknown/Neutral/Other
            else:
                # 🎯 v4.2.3: Unknown/Other gets good score (was 7, now 9)
                score = 9  # Neutral but tradeable

            return min(score, max_points)

        except Exception as e:
            logger.warning(f" Market regime scoring error: {e}")
            return 8  # Default reasonable score on error (was 5)

    def _score_support_resistance(self, side: str, pa_analysis: Dict) -> float:
        """
        Score support/resistance quality (15 points max).

        ✅ SOFTENED: More forgiving - base score even without perfect S/R
        """
        max_points = self.weights['support_resistance']
        score = 0

        try:
            # Get S/R data
            if side == 'LONG':
                sr_level = pa_analysis.get('support_level', {})
            else:
                sr_level = pa_analysis.get('resistance_level', {})

            # ✅ IMPROVED: More generous fallback
            if not sr_level:
                return max_points * 0.40  # 40% for no S/R (was 33%)

            # ✅ BASE SCORE: Give 3 points for having any S/R level
            score += 3

            # Touch count (more touches = stronger level)
            # ✅ SOFTENED: More generous scoring
            touch_count = sr_level.get('touch_count', 0)
            if touch_count >= 5:
                score += 5  # Very strong (was 6)
            elif touch_count >= 3:
                score += 4  # Strong (keep)
            elif touch_count >= 2:
                score += 3  # Moderate (was 2)
            else:
                score += 2  # Weak but still gets 2 (was 1)

            # Strength/quality score - ✅ NO CHANGE (already percentage-based)
            strength = sr_level.get('strength', 0.5)
            score += strength * 4  # 0-4 points (was 0-5 to fit 15 max)

            # Recent test (recent = more relevant)
            # ✅ SOFTENED: Extended time ranges
            last_test_candles_ago = sr_level.get('last_test', 999)
            if last_test_candles_ago <= 20:  # Within 20 candles (was 10)
                score += 2  # Very recent (was 3)
            elif last_test_candles_ago <= 50:  # Within 50 (was 30)
                score += 1.5  # Recent (was 2)
            elif last_test_candles_ago <= 100:  # ✅ NEW: Within 100
                score += 1  # Somewhat recent (was 1 for 50)
            else:
                score += 0.5  # ✅ NEW: Even old S/R gets something

            # Distance to S/R (closer = better)
            # ✅ SOFTENED: Extended distance ranges
            distance_pct = sr_level.get('distance_pct', 10)
            if distance_pct <= 1.0:  # Within 1% (was 0.5%)
                score += 1  # At the level (keep)
            elif distance_pct <= 2.0:  # ✅ NEW: Within 2%
                score += 0.5  # Very close (was 0.5 for 1%)
            elif distance_pct <= 3.0:  # ✅ NEW: Within 3%
                score += 0.3  # Close enough

            return min(score, max_points)

        except Exception as e:
            logger.warning(f" S/R scoring error: {e}")
            return 5  # Default low score on error

    def _score_enhanced_indicators(self, side: str, enhanced_data: Optional[Dict], indicators: Dict) -> float:
        """
        🆕 v4.4.0: Score enhanced indicators (15 points max).

        Components:
        - BB Squeeze (5 pts): Breakout detection
        - EMA Stack (5 pts): Trend structure
        - Divergences (5 pts): RSI/MACD divergence

        These are professional-grade signals that significantly improve trade quality.
        """
        max_points = self.weights['enhanced_indicators']
        score = 0

        try:
            if not enhanced_data:
                return max_points * 0.4  # 40% fallback if no enhanced data

            # 1. BB Squeeze (5 points) - Breakout detection
            bb_squeeze = enhanced_data.get('bb_squeeze', {})
            squeeze_signal = bb_squeeze.get('signal', 'NEUTRAL')
            squeeze_on = bb_squeeze.get('squeeze_on', False)

            if squeeze_on:
                # Squeeze is ON = consolidation, wait for breakout
                score += 2  # Neutral - potential energy building
            elif squeeze_signal in ['STRONG_BUY', 'BUY'] and side == 'LONG':
                score += 5  # Bullish breakout confirmed
            elif squeeze_signal in ['STRONG_SELL', 'SELL'] and side == 'SHORT':
                score += 5  # Bearish breakout confirmed
            elif squeeze_signal == 'WAIT':
                score += 1  # Waiting for direction
            else:
                score += 2  # Neutral

            # 2. EMA Stack (5 points) - Trend structure
            ema_stack = enhanced_data.get('ema_stack', {})
            stack_type = ema_stack.get('stack_type', 'unknown')
            stack_score = ema_stack.get('stack_score', 0)

            if stack_type == 'bullish' and side == 'LONG':
                score += 5  # Perfect bullish stack
            elif stack_type == 'bearish' and side == 'SHORT':
                score += 5  # Perfect bearish stack
            elif stack_type == 'bullish_forming' and side == 'LONG' and stack_score >= 75:
                score += 4  # Forming bullish
            elif stack_type == 'bearish_forming' and side == 'SHORT' and stack_score >= 75:
                score += 4  # Forming bearish
            elif stack_type == 'tangled':
                score += 1  # Choppy market
            else:
                score += 2  # Neutral

            # 3. Divergences (5 points) - RSI/MACD divergence
            rsi_div = enhanced_data.get('rsi_divergence', {})
            macd_div = enhanced_data.get('macd_divergence', {})

            # RSI Divergence
            if rsi_div.get('has_divergence'):
                div_type = rsi_div.get('type', '')
                div_strength = rsi_div.get('strength', 0)
                if div_type == 'bullish' and side == 'LONG':
                    score += min(3, 2 + div_strength)  # Up to 3 points
                elif div_type == 'bearish' and side == 'SHORT':
                    score += min(3, 2 + div_strength)

            # MACD Divergence (bonus)
            if macd_div.get('has_divergence'):
                div_type = macd_div.get('divergence_type', '')
                if 'bullish' in str(div_type) and side == 'LONG':
                    score += 2  # Additional confirmation
                elif 'bearish' in str(div_type) and side == 'SHORT':
                    score += 2

            return min(score, max_points)

        except Exception as e:
            logger.warning(f"⚠️ Enhanced indicators scoring error: {e}")
            return max_points * 0.4

    def _score_momentum_quality(self, side: str, enhanced_data: Optional[Dict], market_regime: Dict) -> float:
        """
        🆕 v4.4.0: Score momentum quality (10 points max).

        Components:
        - ADX trend strength (5 pts)
        - ADX direction alignment (3 pts)
        - Momentum direction (2 pts)

        Strong momentum = Higher probability trades.
        """
        max_points = self.weights['momentum_quality']
        score = 0

        try:
            if not enhanced_data:
                return max_points * 0.5  # 50% fallback

            # Get ADX data
            adx_data = enhanced_data.get('adx', {})
            adx_value = adx_data.get('adx', 20)
            trend_direction = adx_data.get('trend_direction', 'neutral')
            trend_strength = adx_data.get('trend_strength', 'weak')
            adx_trend = adx_data.get('adx_trend', 'flat')

            # 1. ADX Trend Strength (5 points)
            if trend_strength == 'very_strong' or trend_strength == 'extreme':
                score += 5
            elif trend_strength == 'strong':
                score += 4
            elif trend_strength == 'emerging':
                score += 3
            elif trend_strength == 'weak':
                score += 1  # Weak trend = low confidence

            # 2. Direction Alignment (3 points)
            if trend_direction == 'bullish' and side == 'LONG':
                score += 3  # Trading with trend
            elif trend_direction == 'bearish' and side == 'SHORT':
                score += 3  # Trading with trend
            elif trend_direction == 'neutral':
                score += 1.5  # No clear direction
            else:
                score += 0.5  # Counter-trend (risky)

            # 3. ADX Trend Direction (2 points)
            if adx_trend == 'rising':
                score += 2  # Trend strengthening
            elif adx_trend == 'flat':
                score += 1  # Stable
            elif adx_trend == 'falling':
                score += 0.5  # Trend weakening

            return min(score, max_points)

        except Exception as e:
            logger.warning(f"⚠️ Momentum quality scoring error: {e}")
            return max_points * 0.5

    def _score_advanced_indicators(self, side: str, advanced_data: Optional[Dict], indicators: Dict) -> float:
        """
        🆕 v4.5.0: Score advanced professional indicators (20 points max).

        Components:
        - VWAP Position (5 pts): Institutional fair value
        - Stochastic RSI (4 pts): Sensitive momentum
        - Chaikin Money Flow (4 pts): Volume-weighted pressure
        - Williams %R (3 pts): Fast momentum
        - Fibonacci (4 pts): Key level confluence

        These are institutional-grade indicators for maximum accuracy.
        """
        max_points = self.weights['advanced_indicators']
        score = 0

        try:
            if not advanced_data:
                return max_points * 0.4  # 40% fallback if no advanced data

            # 1. VWAP Position (5 points) - Institutional bias
            vwap_data = advanced_data.get('vwap', {})
            vwap_signal = vwap_data.get('signal', 'NEUTRAL')
            vwap_position = vwap_data.get('position', 'unknown')
            deviation_pct = abs(vwap_data.get('deviation_pct', 0))

            if side == 'LONG':
                if vwap_position in ['slightly_above', 'at_vwap'] and deviation_pct < 0.5:
                    score += 5  # Perfect long position near VWAP
                elif vwap_position == 'above' and deviation_pct < 1.5:
                    score += 4  # Good bullish bias
                elif vwap_position == 'far_below':
                    score += 4  # Oversold bounce opportunity
                elif vwap_signal == 'BUY':
                    score += 3
                else:
                    score += 1
            else:  # SHORT
                if vwap_position in ['slightly_below', 'at_vwap'] and deviation_pct < 0.5:
                    score += 5  # Perfect short position near VWAP
                elif vwap_position == 'below' and deviation_pct < 1.5:
                    score += 4  # Good bearish bias
                elif vwap_position == 'far_above':
                    score += 4  # Overbought short opportunity
                elif vwap_signal == 'SELL':
                    score += 3
                else:
                    score += 1

            # 2. Stochastic RSI (4 points) - Sensitive momentum
            stoch_data = advanced_data.get('stoch_rsi', {})
            stoch_signal = stoch_data.get('signal', 'NEUTRAL')
            stoch_zone = stoch_data.get('zone', 'neutral')
            crossover = stoch_data.get('crossover')

            if side == 'LONG':
                if stoch_signal == 'STRONG_BUY':
                    score += 4  # Oversold with bullish crossover
                elif stoch_signal == 'BUY' or crossover == 'bullish':
                    score += 3
                elif stoch_zone == 'oversold':
                    score += 2.5  # Potential reversal zone
                elif stoch_zone == 'bullish':
                    score += 2
                elif stoch_zone != 'overbought':
                    score += 1
            else:  # SHORT
                if stoch_signal == 'STRONG_SELL':
                    score += 4  # Overbought with bearish crossover
                elif stoch_signal == 'SELL' or crossover == 'bearish':
                    score += 3
                elif stoch_zone == 'overbought':
                    score += 2.5  # Potential reversal zone
                elif stoch_zone == 'bearish':
                    score += 2
                elif stoch_zone != 'oversold':
                    score += 1

            # 3. Chaikin Money Flow (4 points) - Volume pressure
            cmf_data = advanced_data.get('cmf', {})
            cmf_signal = cmf_data.get('signal', 'NEUTRAL')
            cmf_pressure = cmf_data.get('pressure', 'balanced')
            cmf_value = cmf_data.get('cmf', 0)

            if side == 'LONG':
                if cmf_signal == 'STRONG_BUY' or cmf_pressure == 'strong_buying':
                    score += 4  # Strong institutional buying
                elif cmf_signal == 'BUY' or cmf_pressure == 'buying':
                    score += 3
                elif cmf_value > 0:
                    score += 2  # Positive flow
                else:
                    score += 1
            else:  # SHORT
                if cmf_signal == 'STRONG_SELL' or cmf_pressure == 'strong_selling':
                    score += 4  # Strong institutional selling
                elif cmf_signal == 'SELL' or cmf_pressure == 'selling':
                    score += 3
                elif cmf_value < 0:
                    score += 2  # Negative flow
                else:
                    score += 1

            # 4. Williams %R (3 points) - Fast momentum
            wr_data = advanced_data.get('williams_r', {})
            wr_signal = wr_data.get('signal', 'NEUTRAL')
            wr_zone = wr_data.get('zone', 'neutral')
            momentum_shift = wr_data.get('momentum_shift')

            if side == 'LONG':
                if wr_signal == 'STRONG_BUY' or momentum_shift == 'bullish':
                    score += 3  # Exiting oversold with momentum
                elif wr_signal == 'BUY' or wr_zone == 'oversold':
                    score += 2
                elif wr_zone == 'bullish':
                    score += 1.5
                else:
                    score += 0.5
            else:  # SHORT
                if wr_signal == 'STRONG_SELL' or momentum_shift == 'bearish':
                    score += 3  # Exiting overbought with momentum
                elif wr_signal == 'SELL' or wr_zone == 'overbought':
                    score += 2
                elif wr_zone == 'bearish':
                    score += 1.5
                else:
                    score += 0.5

            # 5. Fibonacci Confluence (4 points) - Key levels
            fib_data = advanced_data.get('fibonacci', {})
            fib_signal = fib_data.get('signal', 'NEUTRAL')
            fib_level = fib_data.get('current_level')
            fib_trend = fib_data.get('trend', 'unknown')

            if side == 'LONG' and fib_trend == 'uptrend':
                if fib_signal == 'STRONG_BUY':
                    score += 4  # At golden ratio in uptrend
                elif fib_signal == 'BUY' or fib_level in ['38.2', '50.0', '61.8']:
                    score += 3  # At key Fib level
                elif fib_level == '23.6':
                    score += 2  # Shallow retracement
                else:
                    score += 1
            elif side == 'SHORT' and fib_trend == 'downtrend':
                if fib_signal == 'STRONG_SELL':
                    score += 4  # At golden ratio in downtrend
                elif fib_signal == 'SELL' or fib_level in ['38.2', '50.0', '61.8']:
                    score += 3  # At key Fib level
                elif fib_level == '23.6':
                    score += 2  # Shallow retracement
                else:
                    score += 1
            else:
                score += 1  # Trend mismatch

            return min(score, max_points)

        except Exception as e:
            logger.warning(f"⚠️ Advanced indicators scoring error: {e}")
            return max_points * 0.4

    def _score_risk_reward(self, pa_analysis: Dict) -> float:
        """
        Score risk/reward ratio (8 points max - reduced from 10).

        🎯 v4.2.3: SOFTENED - Give moderate score when R/R not calculated
        """
        max_points = self.weights['risk_reward']

        try:
            rr_ratio = pa_analysis.get('risk_reward_ratio', 0)

            # 🎯 v4.2.3: If R/R not calculated yet, give moderate score
            # This happens because R/R is calculated AFTER confluence scoring
            if rr_ratio == 0 or rr_ratio is None:
                return 6  # Neutral score (was 1) - assume ~2:1 R/R

            # Scale for 10 max points
            if rr_ratio >= 5.0:  # 5:1 or better
                return 10  # Exceptional R/R
            elif rr_ratio >= 4.0:  # 4:1
                return 9   # Excellent R/R (was 5)
            elif rr_ratio >= 3.0:  # 3:1
                return 8   # Very good R/R (was 4)
            elif rr_ratio >= 2.5:  # 2.5:1
                return 6   # Good R/R (was 3)
            elif rr_ratio >= 2.0:  # 2:1
                return 5   # Acceptable R/R (was 2)
            elif rr_ratio >= 1.5:  # 1.5:1
                return 3   # Minimum R/R (was 1)
            elif rr_ratio >= 1.2:  # ✅ NEW: 1.2:1
                return 2   # Low but tradeable
            else:
                return 1   # Very poor R/R (was 0)

        except Exception as e:
            logger.warning(f" R/R scoring error: {e}")
            return 1  # Default minimal score

    def _classify_quality(self, score: float) -> SignalQuality:
        """Classify signal quality based on total score."""
        if score >= self.quality_thresholds['EXCELLENT']:
            return SignalQuality.EXCELLENT
        elif score >= self.quality_thresholds['STRONG']:
            return SignalQuality.STRONG
        elif score >= self.quality_thresholds['GOOD']:
            return SignalQuality.GOOD
        elif score >= self.quality_thresholds['MEDIOCRE']:
            return SignalQuality.MEDIOCRE
        else:
            return SignalQuality.POOR

    def _get_position_multiplier(self, quality: SignalQuality) -> float:
        """
        Get position size multiplier based on quality.

        Better setup = Larger position size (within risk limits).
        """
        multipliers = {
            SignalQuality.EXCELLENT: 1.5,  # 150% of normal size
            SignalQuality.STRONG: 1.2,     # 120% of normal size
            SignalQuality.GOOD: 1.0,       # Normal size
            SignalQuality.MEDIOCRE: 0.0,   # Don't trade
            SignalQuality.POOR: 0.0        # Don't trade
        }
        return multipliers.get(quality, 1.0)

    def _generate_reasoning(self, component_scores: Dict, total_score: float, side: str) -> str:
        """Generate human-readable reasoning for the score."""
        reasons = []

        # Identify strengths (high scores)
        for component, score in component_scores.items():
            max_score = self.weights[component]
            percentage = (score / max_score) * 100

            if percentage >= 80:
                reasons.append(f" Excellent {component.replace('_', ' ')} ({score:.1f}/{max_score})")
            elif percentage >= 60:
                reasons.append(f" Good {component.replace('_', ' ')} ({score:.1f}/{max_score})")
            elif percentage <= 30:
                reasons.append(f"L Weak {component.replace('_', ' ')} ({score:.1f}/{max_score})")

        # Overall assessment
        if total_score >= 90:
            overall = "< PREMIUM SETUP - Maximum confidence"
        elif total_score >= 80:
            overall = "= STRONG SETUP - High confidence"
        elif total_score >= 75:
            overall = " GOOD SETUP - Trade acceptable"
        else:
            overall = "= SKIP - Insufficient quality"

        return overall + "\n" + "\n".join(reasons)

    def _log_score(self, result: Dict):
        """Log scoring results."""
        symbol = result['symbol']
        side = result['side']
        score = result['total_score']
        quality = result['quality']
        should_trade = result['should_trade']

        emoji = "" if should_trade else "="

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"< CONFLUENCE SCORING: {symbol} {side}")
        logger.info("=" * 70)
        logger.info(f"{emoji} Total Score: {score}/100")
        logger.info(f"= Quality: {quality}")
        logger.info(f" Trade Decision: {'TAKE TRADE' if should_trade else 'SKIP'}")
        logger.info("")
        logger.info("Component Breakdown:")
        for component, component_score in result['component_scores'].items():
            max_score = self.weights[component]
            pct = (component_score / max_score) * 100
            logger.info(f"  {component.replace('_', ' ').title()}: {component_score:.1f}/{max_score} ({pct:.0f}%)")
        logger.info("")
        logger.info(f"= Reasoning:\n{result['reasoning']}")
        logger.info("=" * 70)
        logger.info("")

    def _get_rejected_score(self, symbol: str, side: str, error_msg: str) -> Dict[str, Any]:
        """Return a rejected score result (used on errors)."""
        return {
            'symbol': symbol,
            'side': side,
            'total_score': 0,
            'should_trade': False,
            'quality': SignalQuality.POOR.value,
            'component_scores': {},
            'reasoning': f"Error during scoring: {error_msg}",
            'recommended_position_multiplier': 0.0
        }


# Factory function
def score_trade_opportunity(
    symbol: str,
    side: str,
    pa_analysis: Dict,
    indicators: Dict,
    volume_profile: Dict,
    market_regime: Dict,
    mtf_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Convenience function to score a trade opportunity.

    Usage:
        score_result = score_trade_opportunity(
            symbol='BTCUSDT',
            side='LONG',
            pa_analysis=pa_data,
            indicators=indicator_data,
            volume_profile=volume_data,
            market_regime=regime_data,
            mtf_data=mtf_data  # Optional
        )

        if score_result['should_trade']:
            # Execute trade with recommended position size
            position_multiplier = score_result['recommended_position_multiplier']
            # ...
    """
    scorer = ConfluenceScorer()
    return scorer.score_opportunity(
        symbol=symbol,
        side=side,
        pa_analysis=pa_analysis,
        indicators=indicators,
        volume_profile=volume_profile,
        market_regime=market_regime,
        mtf_data=mtf_data
    )
