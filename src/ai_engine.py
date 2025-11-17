"""
AI Consensus Engine for trading analysis.
🤖 ML-ONLY MODE: Using ONLY our trained ML model (DeepSeek AI disabled)
Version: 5.0 - ULTRA AGGRESSIVE ($100 positions, 25-30x leverage, ±$1 targets)
"""

import asyncio
import openai
from typing import Dict, Any, Optional, List
from decimal import Decimal
from src.config import get_settings
from src.optimized_prompts import OPTIMIZED_SYSTEM_PROMPT, build_optimized_prompt
from src.utils import setup_logging, parse_ai_response
from src.database import get_db_client
from src.redis_client import get_redis_client
import json

logger = setup_logging()

# Version marker for deployment verification
PA_ONLY_VERSION = "6.3-PA-ONLY"  # PA-ONLY MODE: ML disabled, using only Support/Resistance + Trend + Volume
logger.info(f"📊 AI Engine initialized - Mode: {PA_ONLY_VERSION} (PA-ONLY: ML disabled, S/R+Trend+Volume, 20x leverage, 4-5% SL)")


class AIConsensusEngine:
    """Multi-model AI consensus engine for trading decisions."""

    def __init__(self):
        self.settings = get_settings()

        # Initialize AI clients
        # Qwen3-Max via OpenRouter (OpenAI-compatible API)
        self.qwen_client = openai.AsyncOpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        # DeepSeek uses OpenAI-compatible API
        self.deepseek_client = openai.AsyncOpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url="https://api.deepseek.com"
        )

    def _calculate_adaptive_cache_ttl(self, market_data: Dict[str, Any]) -> int:
        """
        Calculate adaptive cache TTL based on market volatility.

        High volatility = shorter cache (market changes fast)
        Low volatility = longer cache (market stable)

        Returns:
            Cache TTL in seconds
        """
        try:
            # Extract ATR% from volatility analysis
            volatility = market_data.get('volatility_analysis', {})
            atr_percent = volatility.get('atr_percent', 2.0)  # Default to normal

            # Adaptive TTL based on volatility
            if atr_percent >= 4.0:
                # EXTREME volatility - 2 minutes cache
                ttl = 120
                logger.debug(f"🔥 Extreme volatility (ATR {atr_percent:.2f}%) → Cache TTL: {ttl}s")
            elif atr_percent >= 3.0:
                # HIGH volatility - 3 minutes cache
                ttl = 180
                logger.debug(f"⚡ High volatility (ATR {atr_percent:.2f}%) → Cache TTL: {ttl}s")
            elif atr_percent >= 1.5:
                # NORMAL volatility - 5 minutes cache (default)
                ttl = 300
                logger.debug(f"📊 Normal volatility (ATR {atr_percent:.2f}%) → Cache TTL: {ttl}s")
            elif atr_percent >= 0.8:
                # LOW volatility - 8 minutes cache
                ttl = 480
                logger.debug(f"😴 Low volatility (ATR {atr_percent:.2f}%) → Cache TTL: {ttl}s")
            else:
                # VERY LOW volatility - 10 minutes cache
                ttl = 600
                logger.debug(f"💤 Very low volatility (ATR {atr_percent:.2f}%) → Cache TTL: {ttl}s")

            return ttl

        except Exception as e:
            logger.warning(f"Failed to calculate adaptive TTL: {e}, using default")
            return self.settings.ai_cache_ttl_seconds  # Fallback to default

    async def _analyze_with_retry(self, analyze_func, symbol: str, market_data: Dict[str, Any], model_name: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Retry wrapper for AI analysis with exponential backoff.

        Args:
            analyze_func: The analysis function to call
            symbol: Trading symbol
            market_data: Market data
            model_name: Name of the AI model for logging
            max_retries: Maximum number of retry attempts

        Returns:
            Analysis result or None if all attempts fail
        """
        for attempt in range(max_retries):
            try:
                result = await asyncio.wait_for(
                    analyze_func(symbol, market_data),
                    timeout=self.settings.ai_timeout_seconds
                )
                if result:
                    return result
            except asyncio.TimeoutError:
                logger.warning(f"{model_name} timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"{model_name} error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        logger.error(f"{model_name} failed after {max_retries} attempts")
        return None

    async def get_individual_analyses(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        market_sentiment: str = "NEUTRAL"  # 🎯 #7: Market sentiment parameter
    ) -> List[Dict[str, Any]]:
        """
        ❌ DISABLED: PA-ONLY MODE - No AI/ML models used

        USER REQUEST: "C: ML'Yİ KAPAT, SADECE PA KULLAN"

        This method previously called DeepSeek AI, but is now disabled.
        Market scanner will use Strategy fallback, which then calls PA-only consensus.

        Returns empty list to force Strategy-only mode.
        """
        logger.debug(f"📊 PA-ONLY MODE: Skipping AI/ML analysis for {symbol} (disabled by user)")

        # Return empty list - this forces market_scanner.py to use Strategy-only mode
        # which will then call get_consensus() for PA-only analysis
        return []

    async def get_consensus(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        📊 PA-ONLY MODE: Using ONLY Price Action analysis

        USER REQUEST: "C: ML'Yİ KAPAT, SADECE PA KULLAN
        ML model %63.7 accuracy ile çok kötü tahmin yapıyor.
        Sadece Price Action (Support/Resistance, Trend, Volume) kullansın."

        CHANGE: Disabled ML Predictor completely, using ONLY Price Action
        REASON: ML trained on bad data (59.7% win rate), making bad predictions

        Args:
            symbol: Trading symbol
            market_data: Market data dict with price, indicators, etc.

        Returns:
            PA-only analysis with action, confidence, and reasoning
        """
        from src.price_action_analyzer import PriceActionAnalyzer
        import pandas as pd

        logger.debug(f"📊 PA-ONLY mode for {symbol} (ML disabled by user request)")

        # Initialize PA analyzer
        pa_analyzer = PriceActionAnalyzer()

        # Get current price (market_scanner uses 'current_price' key)
        current_price = float(market_data.get('current_price', market_data.get('price', 0)))
        if current_price == 0:
            logger.error(f"❌ No price data for {symbol}")
            return {
                'action': 'hold',
                'side': None,
                'confidence': 0.0,
                'reasoning': 'No price data available',
                'suggested_leverage': 20,
                'stop_loss_percent': 4.0,
                'models_used': ['PA-ONLY'],
                'ensemble_method': 'pa_only',
                'risk_reward_ratio': 0.0,
                'price_action': None
            }

        # Get OHLCV data for PA analysis
        # PA needs historical candlestick data
        df = market_data.get('ohlcv_df')

        # 🔧 FIX: Fallback to creating DataFrame from ohlcv_15m if ohlcv_df is missing
        if df is None:
            ohlcv_15m = market_data.get('ohlcv_15m', [])
            if len(ohlcv_15m) >= 50:
                logger.debug(f"📊 {symbol} - Creating DataFrame from ohlcv_15m ({len(ohlcv_15m)} candles)")
                df = pd.DataFrame(
                    ohlcv_15m,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
            else:
                logger.warning(f"⚠️ Insufficient OHLCV data for {symbol} (need 50+ candles, got {len(ohlcv_15m)})")
                return {
                    'action': 'hold',
                    'side': None,
                    'confidence': 0.0,
                    'reasoning': 'Insufficient OHLCV data for PA analysis',
                    'suggested_leverage': 20,
                    'stop_loss_percent': 4.0,
                    'models_used': ['PA-ONLY'],
                    'ensemble_method': 'pa_only',
                    'risk_reward_ratio': 0.0,
                    'price_action': None
                }

        # Check if we have enough data for analysis
        # 🔧 USER REQUEST: Lowered from 100 to 50 candles since indicators consume first 20-50
        # We fetch 200 candles, indicators use ~50, leaving 150 usable candles
        if len(df) < 50:
            logger.warning(f"⚠️ Insufficient OHLCV data for {symbol} (need 50+ candles, got {len(df)})")
            return {
                'action': 'hold',
                'side': None,
                'confidence': 0.0,
                'reasoning': f'Insufficient OHLCV data ({len(df)} candles, need 50+)',
                'suggested_leverage': 20,
                'stop_loss_percent': 4.0,
                'models_used': ['PA-ONLY'],
                'ensemble_method': 'pa_only',
                'risk_reward_ratio': 0.0,
                'price_action': None
            }

        # Analyze Support/Resistance
        sr_analysis = pa_analyzer.analyze_support_resistance(df)
        trend = pa_analyzer.detect_trend(df)
        volume = pa_analyzer.analyze_volume(df)

        logger.info(
            f"📊 PA Analysis: {symbol} | "
            f"Trend: {trend['direction']} ({trend['strength']}) | "
            f"Volume: {volume['surge_ratio']:.1f}x | "
            f"S/R: {len(sr_analysis.get('support', []))} supports, {len(sr_analysis.get('resistance', []))} resistances"
        )

        # Determine PA signal based on trend + support/resistance
        pa_action = 'hold'
        pa_side = None
        pa_confidence = 0.0
        pa_reasoning = ''

        supports = [s['price'] for s in sr_analysis.get('support', [])]
        resistances = [r['price'] for r in sr_analysis.get('resistance', [])]

        # 🚫 CRITICAL FILTER #1: Check S/R strength
        # USER ISSUE: PEOPLE had S/R Strength 0% → Lost -$32.45
        support_count = len(supports)
        resistance_count = len(resistances)
        total_sr_levels = support_count + resistance_count

        if not supports or not resistances:
            pa_reasoning = 'Insufficient S/R levels for PA analysis'
        elif total_sr_levels < 3:
            # Need at least 3 S/R levels total for reliable analysis
            pa_reasoning = f'Weak S/R: Only {total_sr_levels} levels (need 3+) - REJECTED'
            logger.info(f"🚫 PA Filter: {pa_reasoning}")
        else:
            nearest_support = min(supports, key=lambda x: abs(x - current_price))
            nearest_resistance = min(resistances, key=lambda x: abs(x - current_price))

            dist_to_support = abs(current_price - nearest_support) / current_price
            dist_to_resistance = abs(current_price - nearest_resistance) / current_price

            # 🔧 FIX: STRICTER PA entry requirements (prevent too many bad trades)
            # OLD: 6% tolerance → Too many trades, 80% loss rate!
            # NEW: 3% tolerance → Only trade when VERY close to S/R

            # 🚫 CRITICAL FILTER: Reject SIDEWAYS trends (no direction = high risk!)
            # USER ISSUE: PEOPLE trade was SIDEWAYS with ADX 0.0 → Lost -$32.45
            adx = trend.get('adx', 0)
            if trend['direction'] == 'SIDEWAYS' or adx < 15:
                pa_reasoning = f"REJECTED: Trend {trend['direction']} (ADX: {adx:.1f}) - No clear direction!"
                logger.info(f"🚫 PA Filter: {pa_reasoning}")

            # LONG setup: Near support + uptrend
            elif dist_to_support <= 0.03 and trend['direction'] == 'UPTREND':
                pa_action = 'buy'
                pa_side = 'LONG'
                pa_confidence = 0.65  # Base 65% (higher than before)

                # Boost confidence for strong conditions
                if trend['strength'] == 'STRONG':
                    pa_confidence += 0.10
                if volume['is_surge']:
                    pa_confidence += 0.10
                if dist_to_support <= 0.015:  # Very close to support (1.5%)
                    pa_confidence += 0.05

                pa_reasoning = f"LONG: Support bounce ({dist_to_support*100:.1f}% away) in {trend['direction']} (ADX: {adx:.1f})"

            # SHORT setup: Near resistance + downtrend
            elif dist_to_resistance <= 0.03 and trend['direction'] == 'DOWNTREND':
                pa_action = 'sell'
                pa_side = 'SHORT'
                pa_confidence = 0.65  # Base 65% (higher than before)

                # Boost confidence for strong conditions
                if trend['strength'] == 'STRONG':
                    pa_confidence += 0.10
                if volume['is_surge']:
                    pa_confidence += 0.10
                if dist_to_resistance <= 0.015:  # Very close to resistance (1.5%)
                    pa_confidence += 0.05

                pa_reasoning = f"SHORT: Resistance rejection ({dist_to_resistance*100:.1f}% away) in {trend['direction']} (ADX: {adx:.1f})"

            else:
                pa_reasoning = (
                    f"No clear setup: "
                    f"Support {dist_to_support*100:.1f}% away, "
                    f"Resistance {dist_to_resistance*100:.1f}% away, "
                    f"Trend: {trend['direction']} (ADX: {adx:.1f})"
                )

        logger.info(f"📊 PA Signal: {pa_action.upper()} @ {pa_confidence:.1%} | {pa_reasoning}")

        # Return PA-only result
        return {
            'action': pa_action,
            'side': pa_side,
            'confidence': pa_confidence,
            'reasoning': f"PA-ONLY: {pa_reasoning}",
            'suggested_leverage': 20,  # User wants 20x
            'stop_loss_percent': 9.0,  # 🔧 FIX: 8-10% range (was 4-5%)
            'models_used': ['PA-ONLY'],
            'ensemble_method': 'pa_only',
            'risk_reward_ratio': 0.0,
            'price_action': {
                'trend': trend,
                'volume': volume,
                'support_resistance': sr_analysis,
                'nearest_support': nearest_support if supports else None,
                'nearest_resistance': nearest_resistance if resistances else None,
                'support_dist': dist_to_support if supports else 0,
                'resistance_dist': dist_to_resistance if resistances else 0
            }
        }

        # 🎯 AI+ML ENSEMBLE: Weighted combination (60% AI + 40% ML)
        ai_analysis = analyses[0]  # DeepSeek

        # Get action/side from each model
        ai_action = ai_analysis.get('action', 'hold')
        ai_side = ai_analysis.get('side')
        ai_confidence = ai_analysis.get('confidence', 0.0)

        ml_action = ml_prediction['action']
        ml_confidence = ml_prediction['confidence']

        # Determine ensemble action and confidence
        if ai_action == ml_action:
            # ✅ AGREEMENT: Both models agree - boost confidence
            ensemble_action = ai_action
            ensemble_side = ai_side
            ensemble_confidence = (ai_confidence * 0.60) + (ml_confidence * 0.40) + 0.05  # +5% agreement bonus
            ensemble_confidence = min(0.95, ensemble_confidence)  # Cap at 95%
            reasoning_suffix = "✅ AI+ML AGREE"
            logger.info(f"✅ CONSENSUS: AI and ML AGREE on {ensemble_action.upper()} ({ensemble_confidence:.1%})")

        elif ai_action == 'hold' and ml_action in ['buy', 'sell']:
            # ML suggests trade, AI says hold - use ML if confidence high
            if ml_confidence >= 0.65:
                ensemble_action = ml_action
                ensemble_side = ml_side
                ensemble_confidence = ml_confidence * 0.70  # Reduce confidence (no AI support)
                reasoning_suffix = "🤖 ML override (AI neutral)"
                logger.info(f"🤖 ML OVERRIDE: ML {ml_action.upper()} @ {ml_confidence:.1%}, AI hold")
            else:
                ensemble_action = 'hold'
                ensemble_side = None
                ensemble_confidence = (ai_confidence * 0.60) + (ml_confidence * 0.40)
                reasoning_suffix = "⏸️ ML weak, AI hold"
                logger.info(f"⏸️ HOLD: ML confidence too low ({ml_confidence:.1%}), AI hold")

        elif ml_action == 'hold' and ai_action in ['buy', 'sell']:
            # AI suggests trade, ML says hold - use AI if confidence high
            if ai_confidence >= 0.65:
                ensemble_action = ai_action
                ensemble_side = ai_side
                ensemble_confidence = ai_confidence * 0.70  # Reduce confidence (no ML support)
                reasoning_suffix = "🧠 AI override (ML neutral)"
                logger.info(f"🧠 AI OVERRIDE: AI {ai_action.upper()} @ {ai_confidence:.1%}, ML hold")
            else:
                ensemble_action = 'hold'
                ensemble_side = None
                ensemble_confidence = (ai_confidence * 0.60) + (ml_confidence * 0.40)
                reasoning_suffix = "⏸️ AI weak, ML hold"
                logger.info(f"⏸️ HOLD: AI confidence too low ({ai_confidence:.1%}), ML hold")

        else:
            # ❌ CONFLICT: Models disagree on direction (buy vs sell)
            # Use higher confidence, but reduce it significantly
            if ai_confidence > ml_confidence:
                ensemble_action = ai_action
                ensemble_side = ai_side
                ensemble_confidence = ai_confidence * 0.50  # 50% penalty for conflict
                reasoning_suffix = "⚠️ CONFLICT - Using AI (higher confidence)"
                logger.warning(
                    f"⚠️ CONFLICT: AI {ai_action.upper()} ({ai_confidence:.1%}) vs "
                    f"ML {ml_action.upper()} ({ml_confidence:.1%}) - Using AI with penalty"
                )
            else:
                ensemble_action = ml_action
                ensemble_side = ml_side
                ensemble_confidence = ml_confidence * 0.50  # 50% penalty for conflict
                reasoning_suffix = "⚠️ CONFLICT - Using ML (higher confidence)"
                logger.warning(
                    f"⚠️ CONFLICT: ML {ml_action.upper()} ({ml_confidence:.1%}) vs "
                    f"AI {ai_action.upper()} ({ai_confidence:.1%}) - Using ML with penalty"
                )

        # Build ensemble result
        consensus = {
            'action': ensemble_action,
            'side': ensemble_side,
            'confidence': ensemble_confidence,
            'reasoning': f"{ai_analysis.get('reasoning', '')} | {ml_prediction['reasoning']} | {reasoning_suffix}",
            'suggested_leverage': ai_analysis.get('suggested_leverage', 6),
            'stop_loss_percent': ai_analysis.get('stop_loss_percent', 4.0),
            'models_used': ['DeepSeek-AI (60%)', 'ML-Predictor (40%)'],
            'ensemble_method': 'ai_ml_weighted_ensemble',
            'risk_reward_ratio': ai_analysis.get('risk_reward_ratio', 0.0),

            # Include individual model outputs for transparency
            'ai_confidence': ai_confidence,
            'ai_action': ai_action,
            'ml_confidence': ml_confidence,
            'ml_action': ml_action,
            'ml_reasoning': ml_prediction['reasoning']
        }

        logger.info(
            f"✅ 🧠 AI+ML ENSEMBLE for {symbol}: {ensemble_action.upper()} "
            f"@ {ensemble_confidence:.1%} | "
            f"AI: {ai_action}@{ai_confidence:.1%} + ML: {ml_action}@{ml_confidence:.1%}"
        )

        return consensus

    async def _analyze_with_qwen(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get analysis from Qwen3-Max via OpenRouter using OPTIMIZED PROMPTS."""
        try:
            # 🎯 Use optimized prompt (27 indicators → 10 priority indicators)
            prompt = build_optimized_prompt(symbol, market_data)

            response = await self.qwen_client.chat.completions.create(
                model="qwen/qwen3-max",  # OpenRouter model identifier
                messages=[
                    {"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2048
            )

            content = response.choices[0].message.content
            analysis = parse_ai_response(content)

            if analysis:
                analysis['model_name'] = 'qwen3-max'
                logger.debug(f"Qwen3-Max analysis: {analysis.get('action')} ({analysis.get('confidence', 0):.0%})")
                return analysis
            else:
                logger.warning("Failed to parse Qwen3-Max response")
                return None

        except Exception as e:
            logger.error(f"Qwen3-Max API error: {e}")
            raise

    async def _analyze_with_deepseek(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get analysis from DeepSeek V3 using OPTIMIZED PROMPTS (10 priority indicators)."""
        try:
            # 🎯 Use optimized prompt (27 indicators → 10 priority indicators)
            prompt = build_optimized_prompt(symbol, market_data)

            response = await self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Increased from 0.3 for more varied responses
                max_tokens=2048,
                top_p=0.9  # Add sampling diversity
            )

            content = response.choices[0].message.content

            # DEBUG: Log raw AI response
            logger.info(f"🔍 DeepSeek RAW response for {symbol}: {content[:200]}...")

            analysis = parse_ai_response(content)

            if analysis:
                analysis['model_name'] = 'deepseek'
                logger.info(f"✅ DeepSeek parsed: action={analysis.get('action')}, confidence={analysis.get('confidence', 0):.2f}, side={analysis.get('side')}")
                return analysis
            else:
                logger.warning(f"❌ Failed to parse DeepSeek response for {symbol}")
                logger.warning(f"   Raw content: {content}")
                return None

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise

    async def get_exit_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        position: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get AI recommendation for exiting an existing position.
        Uses faster, cheaper model (DeepSeek) for position monitoring.

        Returns:
            Dict with 'should_exit', 'reason', 'confidence'
        """
        try:
            # Build exit analysis prompt
            prompt = f"""You are monitoring an open {position['side']} position for {symbol}.

POSITION DETAILS:
Entry Price: ${position['entry_price']:.4f}
Current Price: ${market_data['current_price']:.4f}
Leverage: {position['leverage']}x
Unrealized P&L: ${position.get('unrealized_pnl_usd', 0):.2f}
Stop-Loss: ${position['stop_loss_price']:.4f}
Time in Position: {position.get('duration', 'unknown')}

CURRENT MARKET DATA:
RSI (15m): {market_data['indicators']['15m']['rsi']:.1f}
MACD: {market_data['indicators']['15m']['macd']:.4f}
Market Regime: {market_data['market_regime']}

Should we exit this position now, or continue holding?

RESPONSE FORMAT (JSON only):
{{
    "should_exit": true|false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}"""

            # Use DeepSeek for cost efficiency
            response = await self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a professional trader monitoring positions. Be conservative - only recommend exit if there's clear reason."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=512
            )

            content = response.choices[0].message.content
            result = parse_ai_response(content)

            if result:
                return result
            else:
                return {'should_exit': False, 'confidence': 0.0, 'reason': 'Failed to parse AI response'}

        except Exception as e:
            logger.error(f"Error getting exit signal: {e}")
            return {'should_exit': False, 'confidence': 0.0, 'reason': f'API error: {e}'}

    def _validate_ai_response(self, analysis: Dict[str, Any]) -> bool:
        """
        Validate AI response for correctness and safety.

        Checks:
        - Required fields present
        - Confidence in valid range [0, 1]
        - Stop-loss in safe range [5%, 10%]
        - Leverage in allowed range [2, MAX_LEVERAGE]
        - Action is valid (buy/sell/hold)

        Returns:
            True if valid, False otherwise (logs error)
        """
        try:
            # Check required fields
            required_fields = ['action', 'confidence', 'side', 'stop_loss_percent', 'suggested_leverage']
            for field in required_fields:
                if field not in analysis:
                    logger.error(f"❌ AI validation failed: Missing required field '{field}'")
                    return False

            # Validate action
            if analysis['action'] not in ['buy', 'sell', 'hold']:
                logger.error(f"❌ AI validation failed: Invalid action '{analysis['action']}' (must be buy/sell/hold)")
                return False

            # Validate confidence range [0, 1]
            confidence = analysis['confidence']
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                logger.error(f"❌ AI validation failed: Confidence {confidence} out of range [0, 1]")
                return False

            # ⚠️ IMPORTANT: For "hold" action, skip stop-loss and leverage validation
            # (hold means no trade, so stop_loss=0 and leverage=0 is expected and valid)
            if analysis['action'] == 'hold':
                logger.debug(f"✅ AI validation passed: HOLD (confidence={confidence:.1%})")
                return True

            # Validate stop-loss percentage [5%, 20%] (for buy/sell only)
            stop_loss = analysis['stop_loss_percent']
            if not isinstance(stop_loss, (int, float)) or not (5 <= stop_loss <= 20):
                logger.error(f"❌ AI validation failed: Stop-loss {stop_loss}% out of safe range [5%, 20%]")
                return False

            # Validate leverage [2, MAX_LEVERAGE] (for buy/sell only)
            leverage = analysis['suggested_leverage']
            max_lev = self.settings.max_leverage
            if not isinstance(leverage, (int, float)) or not (2 <= leverage <= max_lev):
                logger.error(f"❌ AI validation failed: Leverage {leverage}x out of range [2, {max_lev}]")
                return False

            # Validate side (for buy/sell only)
            if analysis['action'] in ['buy', 'sell']:
                if analysis['side'] not in ['LONG', 'SHORT']:
                    logger.error(f"❌ AI validation failed: Invalid side '{analysis['side']}' (must be LONG/SHORT)")
                    return False

            # All checks passed
            logger.debug(f"✅ AI validation passed: {analysis.get('action')} confidence={confidence:.1%}")
            return True

        except Exception as e:
            logger.error(f"❌ AI validation exception: {e}")
            return False

    async def _get_ml_only_prediction(self, symbol: str, market_data: Dict[str, Any], ml_learner) -> Dict[str, Any]:
        """
        🧠 REAL ML MODE: Use trained GradientBoosting classifier.

        REPLACED: Hardcoded rules → Real sklearn ML model
        Features:
        - 40+ engineered numerical features
        - GradientBoosting classifier trained on historical trades
        - Calibrated probability estimates
        - Feature importance tracking
        - Online learning updates

        Args:
            symbol: Trading symbol
            market_data: Market indicators
            ml_learner: ML pattern learner instance (kept for backwards compatibility)

        Returns:
            ML-based prediction with confidence score
        """
        try:
            # 🚀 USE REAL ML PREDICTOR (GradientBoosting)
            from src.ml_predictor import get_ml_predictor
            ml_predictor = get_ml_predictor()

            # Get prediction for LONG position
            long_prediction = await ml_predictor.predict(market_data, side='LONG')

            # Get prediction for SHORT position
            short_prediction = await ml_predictor.predict(market_data, side='SHORT')

            # Choose direction with higher confidence
            if long_prediction['confidence'] > short_prediction['confidence']:
                action = 'buy'
                side = 'LONG'
                confidence = long_prediction['confidence']
                reasoning = long_prediction['reasoning']
            elif short_prediction['confidence'] > long_prediction['confidence']:
                action = 'sell'
                side = 'SHORT'
                confidence = short_prediction['confidence']
                reasoning = short_prediction['reasoning']
            else:
                # Equal confidence or both below threshold
                action = 'hold'
                side = None
                confidence = max(long_prediction['confidence'], short_prediction['confidence'])
                reasoning = "No clear directional signal"

            # CONSERVATIVE MODE: Adaptive leverage based on confidence
            # Max 10x leverage, wider stops for safety
            if confidence >= 0.85:
                leverage = 10  # Ultra high confidence (max)
                stop_loss = 20.0
            elif confidence >= 0.75:
                leverage = 7  # High confidence
                stop_loss = 16.0
            elif confidence >= 0.70:
                leverage = 5  # Acceptable confidence
                stop_loss = 14.0
            else:
                leverage = 3  # Low confidence (minimum)
                stop_loss = 12.0

            logger.info(
                f"🤖 REAL ML: {symbol} → {action.upper() if action != 'hold' else 'HOLD'} | "
                f"Confidence: {confidence:.1%} | "
                f"Model trained: {long_prediction.get('model_trained', False)} | "
                f"Samples: {long_prediction.get('training_samples', 0)}"
            )

            return {
                'action': action,
                'side': side,
                'confidence': confidence,
                'suggested_leverage': leverage,
                'stop_loss_percent': stop_loss,
                'risk_reward_ratio': 2.0,
                'reasoning': reasoning,
                'pattern_count': 0,  # Not used in real ML
                'models_used': ['GradientBoosting-ML'],
                'weighted_consensus': False,
                'ensemble_method': 'ml_sklearn',
                'model_trained': long_prediction.get('model_trained', False),
                'training_samples': long_prediction.get('training_samples', 0),
                'feature_importance': long_prediction.get('feature_importance')
            }

        except Exception as e:
            logger.error(f"❌ Real ML prediction failed: {e}")
            # Fallback to simple rule-based prediction
            indicators_1h = market_data.get('indicators', {}).get('1h', {})
            rsi_1h = indicators_1h.get('rsi', 50)
            trend_1h = indicators_1h.get('trend', 'neutral')

            # Simple fallback logic
            if rsi_1h < 35 and trend_1h == 'uptrend':
                action, side, confidence = 'buy', 'LONG', 0.45
            elif rsi_1h > 65 and trend_1h == 'downtrend':
                action, side, confidence = 'sell', 'SHORT', 0.45
            else:
                action, side, confidence = 'hold', None, 0.30

            return {
                'action': action,
                'side': side,
                'confidence': confidence,
                'suggested_leverage': 5,  # Conservative fallback
                'stop_loss_percent': 14.0,  # Conservative fallback
                'risk_reward_ratio': 2.0,
                'reasoning': f"Fallback: ML failed, using simple rules",
                'pattern_count': 0,
                'models_used': ['FALLBACK'],
                'weighted_consensus': False,
                'ensemble_method': 'fallback'
            }


# Singleton instance
_ai_engine: Optional[AIConsensusEngine] = None


def get_ai_engine() -> AIConsensusEngine:
    """Get or create AI engine instance."""
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIConsensusEngine()
    return _ai_engine
