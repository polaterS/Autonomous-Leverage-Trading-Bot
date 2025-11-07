"""
AI Consensus Engine for trading analysis.
🧠 PURE ML AUTONOMOUS MODE: AI models bypassed, using ML pattern learning only.
Version: 2.0 - ML-ONLY (No API calls)
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
ML_ONLY_VERSION = "2.0-ML-PURE"
logger.info(f"🧠 AI Engine initialized - Mode: {ML_ONLY_VERSION} (AI APIs bypassed)")


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
        🚫 DISABLED IN ML-ONLY MODE - This function should never be called.

        Returns empty list to force fallback to ML-ONLY mode.
        """
        logger.warning(f"⚠️ get_individual_analyses() called for {symbol} - ML-ONLY mode active, returning empty list")
        return []  # Force ML-ONLY fallback

        # 🚫 OLD AI CODE DISABLED BELOW (unreachable)
        """
        🎯 #2: MULTI-MODEL ENSEMBLE - Get analyses from BOTH Qwen3-Max AND DeepSeek.

        Uses weighted voting based on historical model performance per symbol and regime.

        Args:
            symbol: Trading symbol
            market_data: Market data dict with price, indicators, etc.
            market_sentiment: Market breadth sentiment (BULLISH_STRONG, BULLISH, NEUTRAL, BEARISH, BEARISH_STRONG)

        Returns:
            List of individual analyses with ML-adjusted confidence from both models
        """
        logger.info(f"🎯 Requesting AI analysis for {symbol} (DeepSeek only)...")

        # 🎯 SINGLE MODEL: DeepSeek only (Qwen3-Max disabled - no API key)
        deepseek_task = self._analyze_with_retry(
            self._analyze_with_deepseek,
            symbol,
            market_data,
            "DeepSeek-V3.2"
        )

        # Wait for DeepSeek to complete
        deepseek_analysis = await deepseek_task

        # Process valid analyses
        valid_analyses = []

        for analysis, model_name in [(deepseek_analysis, "DeepSeek-V3.2")]:
            if analysis is not None and self._validate_ai_response(analysis):
                # 🧠 ML ENHANCEMENT: Adjust confidence based on historical patterns + 🎯 #7: Market sentiment
                try:
                    from src.ml_pattern_learner import get_ml_learner
                    ml_learner = await get_ml_learner()
                    ml_result = await ml_learner.analyze_opportunity(
                        symbol, analysis, market_data, market_sentiment  # 🎯 #7: Pass sentiment
                    )

                    # Update analysis with ML insights
                    analysis['original_confidence'] = analysis['confidence']
                    analysis['confidence'] = ml_result['adjusted_confidence']
                    analysis['ml_adjustment'] = ml_result['ml_adjustment']
                    analysis['ml_score'] = ml_result['ml_score']
                    analysis['ml_reasoning'] = ml_result['reasoning']

                    if ml_result.get('regime_warning'):
                        analysis['regime_warning'] = ml_result['regime_warning']

                    logger.info(
                        f"🧠 {model_name} ML Enhanced: {symbol} confidence "
                        f"{analysis['original_confidence']:.1%} → {analysis['confidence']:.1%} "
                        f"(Δ {ml_result['ml_adjustment']:+.1%})"
                    )

                    # 🎯 ENHANCED ML OVERRIDE: Quality over quantity
                    # Threshold adjusted: 65% → 62% → 58% → 70% (based on ML stats: 60% conf = 0% win rate!)
                    # ML Data: 75% conf = 75% accuracy, 60% conf = 0% accuracy → Use 70% threshold
                    if analysis['confidence'] >= 0.70 and analysis['action'] == 'hold':
                        logger.info(
                            f"🔍 ML Override Check: {symbol} conf={analysis['confidence']:.1%}, "
                            f"action={analysis['action']}, attempting override..."
                        )

                        # 🛡️ SAFETY CHECK 1: Market Sentiment (NEW!)
                        # Don't trade against strong market sentiment
                        if market_sentiment == 'BEARISH_STRONG':
                            logger.warning(
                                f"⚠️ ML Override BLOCKED: {symbol} - STRONG BEARISH market sentiment "
                                f"(unsafe to open LONG positions)"
                            )
                            # Skip override - too risky in strong bearish market
                            # Keep action='hold'
                        elif market_sentiment == 'BULLISH_STRONG':
                            # Strong bullish OK for LONG, but check more
                            pass

                        # Continue with override checks only if market sentiment is acceptable
                        elif market_sentiment not in ['BEARISH_STRONG']:
                            # Determine direction from multiple sources (priority order)
                            multi_tf = market_data.get('multi_timeframe', {})
                            alignment = multi_tf.get('agreement', 'unknown')
                            trend_alignment_raw = multi_tf.get('trend_alignment', 'unknown')

                            indicators_1h = market_data.get('indicators', {}).get('1h', {})
                            indicators_4h = market_data.get('indicators', {}).get('4h', {})
                            trend_1h = indicators_1h.get('trend', 'unknown')
                            trend_4h = indicators_4h.get('trend', 'unknown')
                            rsi_1h = indicators_1h.get('rsi', 50)  # Default neutral

                            # 🛡️ SAFETY CHECK 2: Multi-Timeframe Trend (ENHANCED!)
                            # Count bearish/bullish timeframes
                            bearish_count = sum([
                                1 for t in [
                                    indicators_1h.get('trend'),
                                    indicators_4h.get('trend'),
                                    market_data.get('indicators', {}).get('15m', {}).get('trend')
                                ] if t in ['downtrend', 'bearish']
                            ])

                            bullish_count = sum([
                                1 for t in [
                                    indicators_1h.get('trend'),
                                    indicators_4h.get('trend'),
                                    market_data.get('indicators', {}).get('15m', {}).get('trend')
                                ] if t in ['uptrend', 'bullish']
                            ])

                            # If 3/4 timeframes bearish → BLOCK LONG override
                            if bearish_count >= 2 and 'bearish' in trend_alignment_raw.lower():
                                logger.warning(
                                    f"⚠️ ML Override BLOCKED: {symbol} - Multi-TF bearish "
                                    f"(bearish_count={bearish_count}, alignment={trend_alignment_raw})"
                                )
                                # Skip override
                            # If 3/4 timeframes bullish → BLOCK SHORT override
                            elif bullish_count >= 2 and 'bullish' in trend_alignment_raw.lower():
                                logger.warning(
                                    f"⚠️ ML Override BLOCKED: {symbol} - Multi-TF bullish "
                                    f"(bullish_count={bullish_count}, alignment={trend_alignment_raw}) "
                                    f"- SHORT override blocked"
                                )
                                # Skip SHORT override, but allow LONG
                                pass

                            direction = None
                            reason = ""
                            skip_reason = ""

                            # Priority 1: Strong Multi-timeframe alignment (3+ timeframes agree)
                            if 'strong_bullish' in alignment.lower() or bullish_count >= 3:
                                direction = 'buy'
                                reason = f"strong multi-TF bullish ({bullish_count}/4)"
                            elif 'strong_bearish' in alignment.lower() or bearish_count >= 3:
                                direction = 'sell'
                                reason = f"strong multi-TF bearish ({bearish_count}/4)"

                            # Priority 2: Moderate Multi-timeframe alignment (2+ timeframes)
                            elif 'bullish' in alignment.lower() and bullish_count >= 2:
                                direction = 'buy'
                                reason = f"multi-TF bullish ({bullish_count}/4)"
                            elif 'bearish' in alignment.lower() and bearish_count >= 2:
                                direction = 'sell'
                                reason = f"multi-TF bearish ({bearish_count}/4)"

                            # Priority 3: 4h trend (higher timeframe = more reliable)
                            elif trend_4h in ['uptrend', 'bullish']:
                                direction = 'buy'
                                reason = f"4h trend {trend_4h}"
                            elif trend_4h in ['downtrend', 'bearish']:
                                direction = 'sell'
                                reason = f"4h trend {trend_4h}"

                            # Priority 4: 1h trend
                            elif trend_1h in ['uptrend', 'bullish']:
                                direction = 'buy'
                                reason = f"1h trend {trend_1h}"
                            elif trend_1h in ['downtrend', 'bearish']:
                                direction = 'sell'
                                reason = f"1h trend {trend_1h}"

                            # Priority 5: RSI extremes (both LONG and SHORT opportunities)
                            elif rsi_1h < 30:  # Very oversold = buy opportunity
                                direction = 'buy'
                                reason = f"RSI very oversold ({rsi_1h:.0f})"
                            elif rsi_1h > 70:  # Very overbought = SHORT opportunity
                                direction = 'sell'
                                reason = f"RSI very overbought ({rsi_1h:.0f})"

                            # Priority 6: Moderate RSI (AGGRESSIVE: confidence >= 60% for faster ML learning)
                            elif analysis['confidence'] >= 0.60:
                                if rsi_1h < 45:  # Moderately oversold (relaxed from 40)
                                    direction = 'buy'
                                    reason = f"RSI oversold ({rsi_1h:.0f}, ML conf {analysis['confidence']:.0%})"
                                elif rsi_1h > 55:  # Moderately overbought (relaxed from 60)
                                    direction = 'sell'
                                    reason = f"RSI overbought ({rsi_1h:.0f}, ML conf {analysis['confidence']:.0%})"

                            # Priority 7: 15m trend (ULTRA AGGRESSIVE - confidence >= 50%)
                            elif analysis['confidence'] >= 0.50:
                                trend_15m = analysis.get('trend_15m', 'neutral')
                                if trend_15m in ['uptrend', 'bullish']:
                                    direction = 'buy'
                                    reason = f"15m trend {trend_15m} (ML conf {analysis['confidence']:.0%})"
                                elif trend_15m in ['downtrend', 'bearish']:
                                    direction = 'sell'
                                    reason = f"15m trend {trend_15m} (ML conf {analysis['confidence']:.0%})"

                            # 🛡️ SHORT TRADES CHECK: Block if disabled in settings
                            if direction == 'sell' and not self.settings.enable_short_trades:
                                logger.warning(
                                    f"⚠️ ML Override BLOCKED: {symbol} - SHORT signal detected but SHORT trades disabled in settings "
                                    f"(reason: {reason})"
                                )
                                direction = None  # Block SHORT if disabled

                            # Apply override if direction found (LONG or SHORT)
                            if direction == 'buy':
                                analysis['action'] = 'buy'
                                analysis['side'] = 'LONG'

                                # 🎯 Calculate appropriate stop-loss and leverage based on confidence
                                # FIXED: Always use 10% stop-loss for consistent $10 max loss on $100 position
                                conf = analysis['confidence']
                                if conf >= 0.90:
                                    analysis['suggested_leverage'] = 5
                                    analysis['stop_loss_percent'] = 10.0
                                elif conf >= 0.80:
                                    analysis['suggested_leverage'] = 4
                                    analysis['stop_loss_percent'] = 10.0
                                elif conf >= 0.70:
                                    analysis['suggested_leverage'] = 3
                                    analysis['stop_loss_percent'] = 10.0
                                else:  # 70-74% (threshold is now 70%)
                                    analysis['suggested_leverage'] = 2
                                    analysis['stop_loss_percent'] = 10.0

                                logger.info(
                                    f"🎯 ML Override SUCCESS: {symbol} 'hold' → 'buy' LONG "
                                    f"(ML conf: {analysis['confidence']:.1%}, {reason}, "
                                    f"{analysis['suggested_leverage']}x lev, {analysis['stop_loss_percent']}% SL)"
                                )
                            elif direction == 'sell':
                                analysis['action'] = 'sell'
                                analysis['side'] = 'SHORT'

                                # 🎯 Calculate appropriate stop-loss and leverage based on confidence
                                # FIXED: Always use 10% stop-loss for consistent $10 max loss on $100 position
                                conf = analysis['confidence']
                                if conf >= 0.90:
                                    analysis['suggested_leverage'] = 5
                                    analysis['stop_loss_percent'] = 10.0
                                elif conf >= 0.80:
                                    analysis['suggested_leverage'] = 4
                                    analysis['stop_loss_percent'] = 10.0
                                elif conf >= 0.70:
                                    analysis['suggested_leverage'] = 3
                                    analysis['stop_loss_percent'] = 10.0
                                else:  # 70-74% (threshold is now 70%)
                                    analysis['suggested_leverage'] = 2
                                    analysis['stop_loss_percent'] = 10.0

                                logger.info(
                                    f"🎯 ML Override SUCCESS: {symbol} 'hold' → 'sell' SHORT "
                                    f"(ML conf: {analysis['confidence']:.1%}, {reason}, "
                                    f"{analysis['suggested_leverage']}x lev, {analysis['stop_loss_percent']}% SL)"
                                )
                            else:
                                logger.info(
                                    f"⚠️ ML Override SKIPPED: {symbol} - No clear direction "
                                f"(TF: {alignment}, 1h: {trend_1h}, RSI: {rsi_1h:.0f})"
                            )

                except Exception as ml_error:
                    logger.warning(f"ML enhancement failed for {model_name} on {symbol}: {ml_error}")

                valid_analyses.append(analysis)
            elif analysis is None:
                logger.error(f"❌ {model_name} API failed for {symbol} (returned None)")
            else:
                logger.error(f"❌ {model_name} analysis INVALID for {symbol} (failed validation)")

        logger.info(f"✅ Got {len(valid_analyses)}/1 AI analysis for {symbol}")
        return valid_analyses

    async def get_consensus(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🎯 #2: Get WEIGHTED ENSEMBLE consensus from multiple AI models.

        Uses dynamic model weighting based on historical performance per symbol and regime.

        Args:
            symbol: Trading symbol
            market_data: Market data dict with price, indicators, etc.

        Returns:
            Weighted consensus analysis with action, confidence, and reasoning
        """
        # 🧠 ML-ONLY MODE: Skip cache entirely (cache may contain old AI analyses)
        # Check Redis cache first (much faster than PostgreSQL)
        # DISABLED: Cache may contain old AI-based predictions, we need fresh ML predictions
        # redis = await get_redis_client()
        # cached = await redis.get_cached_ai_analysis(symbol, '5m')
        # if cached:
        #     logger.debug(f"✅ Using cached AI analysis for {symbol} (from Redis)")
        #     return cached

        # 🧠 ML-ONLY MODE: Skip AI entirely, use pure ML pattern learning
        from src.ml_pattern_learner import get_ml_learner
        ml_learner = await get_ml_learner()

        # Determine market sentiment for ML context
        market_sentiment = market_data.get('market_sentiment', 'NEUTRAL')

        # 🎯 ALWAYS use ML-ONLY prediction (AI disabled for pure ML learning)
        logger.debug(f"🧠 Using ML-ONLY mode for {symbol} (AI bypassed)")

        # Get ML prediction directly
        ml_prediction = await self._get_ml_only_prediction(symbol, market_data, ml_learner)

        # Handle ML confidence check
        if ml_prediction['confidence'] >= 0.50:
            logger.info(
                f"🧠 ML-ONLY: {symbol} {ml_prediction['action']} @ {ml_prediction['confidence']:.0%} "
                f"(patterns: {ml_prediction['pattern_count']})"
            )
            return ml_prediction
        else:
            logger.info(f"🧠 ML-ONLY: {symbol} confidence too low ({ml_prediction['confidence']:.0%}) - holding")
            return {
                'action': 'hold',
                'side': None,
                'confidence': ml_prediction['confidence'],
                'weighted_consensus': False,
                'reason': f"ML confidence below threshold ({ml_prediction['confidence']:.0%} < 50%)",
                'suggested_leverage': 2,
                'stop_loss_percent': 10.0,
                'risk_reward_ratio': 0.0,
                'reasoning': 'ML confidence insufficient for trade',
                'models_used': ['ML-ONLY'],
                'ensemble_method': 'ml_pure'
            }

        # 🎯 OLD CODE REMOVED: AI ensemble voting (now using pure ML)
        # The rest of the AI ensemble code is kept but never executed
        # 🎯 #2: WEIGHTED ENSEMBLE VOTING (disabled - using ML-ONLY)
        market_regime = market_data.get('market_regime', 'UNKNOWN')
        consensus = ml_learner.calculate_weighted_ensemble(
            analyses, symbol, market_regime
        )

        # Add risk/reward ratio if not present
        if 'risk_reward_ratio' not in consensus:
            consensus['risk_reward_ratio'] = 0.0

        # Track which models were used
        consensus['models_used'] = [a.get('model_name', 'unknown') for a in analyses]

        # 🎯 ADAPTIVE CACHE TTL: Adjust based on market volatility
        adaptive_ttl = self._calculate_adaptive_cache_ttl(market_data)

        # Cache result in Redis (fast access with adaptive duration)
        await redis.cache_ai_analysis(
            symbol, '5m', consensus,
            adaptive_ttl
        )

        # Also cache in database for analysis/debugging
        db = await get_db_client()
        try:
            await db.set_ai_cache(
                symbol, 'consensus', '5m', consensus,
                adaptive_ttl
            )
        except Exception as e:
            logger.warning(f"Failed to cache in DB (non-critical): {e}")

        logger.info(f"💾 Cached weighted ensemble for {symbol} (TTL: {adaptive_ttl}s)")

        logger.info(
            f"✅ 🎯 WEIGHTED ENSEMBLE for {symbol}: {consensus['action'].upper()} "
            f"(confidence: {consensus['confidence']:.1%}, "
            f"strength: {consensus.get('consensus_strength', 0):.1%}, "
            f"models: {len(analyses)})"
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
        🧠 ML-ONLY MODE: Use learned patterns when AI models are unavailable.

        Uses:
        - Historical pattern matching
        - Bayesian feature analysis
        - Symbol-specific performance data
        - Market regime adaptation

        Args:
            symbol: Trading symbol
            market_data: Market indicators
            ml_learner: ML pattern learner instance

        Returns:
            ML-based prediction with confidence score
        """
        # Get technical indicators (already calculated in market_data)
        indicators_1h = market_data.get('indicators', {}).get('1h', {})
        indicators_4h = market_data.get('indicators', {}).get('4h', {})

        # Extract key features for pattern matching
        rsi_1h = indicators_1h.get('rsi', 50)
        trend_1h = indicators_1h.get('trend', 'neutral')
        trend_4h = indicators_4h.get('trend', 'neutral')

        # Build pattern features
        pattern_features = []

        # 1. Trend alignment
        if trend_1h == 'uptrend' and trend_4h == 'uptrend':
            pattern_features.append('strong_bullish_trend')
        elif trend_1h == 'downtrend' and trend_4h == 'downtrend':
            pattern_features.append('strong_bearish_trend')
        elif trend_1h == 'uptrend':
            pattern_features.append('bullish_trend')
        elif trend_1h == 'downtrend':
            pattern_features.append('bearish_trend')

        # 2. RSI signals
        if rsi_1h < 30:
            pattern_features.append('rsi_oversold')
        elif rsi_1h > 70:
            pattern_features.append('rsi_overbought')
        elif rsi_1h < 40:
            pattern_features.append('rsi_weak')
        elif rsi_1h > 60:
            pattern_features.append('rsi_strong')

        # 3. Market regime
        market_regime = market_data.get('market_regime', 'UNKNOWN')
        pattern_features.append(f'regime_{market_regime.lower()}')

        # Calculate confidence from Bayesian features
        confidence_scores = []
        for feature in pattern_features:
            if feature in ml_learner.bayesian_features:
                bayesian_wr = ml_learner.get_bayesian_win_rate(feature)
                weight = ml_learner.get_bayesian_weight(feature)
                confidence_scores.append(bayesian_wr * weight)

        # Base confidence from patterns
        if confidence_scores:
            base_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            # 🔥 DYNAMIC BASE CONFIDENCE based on pattern count
            # More patterns detected = Higher confidence (even without historical data)
            pattern_count = len(pattern_features)
            if pattern_count >= 5:
                base_confidence = 0.75  # 5+ patterns = Very strong setup
            elif pattern_count >= 4:
                base_confidence = 0.65  # 4 patterns = Strong setup
            elif pattern_count >= 3:
                base_confidence = 0.60  # 3 patterns = Good setup
            else:
                base_confidence = 0.50  # 2 or fewer = Neutral

        # 🚀 PATTERN COUNT BOOST: More patterns = Higher confidence
        # This works immediately, even before ML learns from data
        pattern_count = len(pattern_features)
        pattern_boost = min(0.15, pattern_count * 0.03)  # +3% per pattern, max +15%
        base_confidence = min(0.95, base_confidence + pattern_boost)

        # 🧠 ML PATTERN WIN RATE ADJUSTMENT: Boost/penalize based on historical pattern success
        pattern_wr_adjustment = 0.0
        pattern_wrs = []
        for pattern in pattern_features:
            wins = ml_learner.winning_patterns.get(pattern, 0)
            losses = ml_learner.losing_patterns.get(pattern, 0)
            total = wins + losses
            if total >= 5:  # Only use patterns with 5+ occurrences
                wr = wins / total
                pattern_wrs.append(wr)

        if pattern_wrs:
            avg_pattern_wr = sum(pattern_wrs) / len(pattern_wrs)
            # Adjust confidence based on pattern success rate
            # 85%+ WR → +10% confidence boost
            # 75-84% WR → +5% confidence boost
            # 60-74% WR → No change
            # 50-59% WR → -5% confidence penalty
            # <50% WR → -10% confidence penalty
            if avg_pattern_wr >= 0.85:
                pattern_wr_adjustment = 0.10
                logger.info(f"🟢 {symbol}: Excellent pattern WR ({avg_pattern_wr:.1%}) → +10% confidence boost")
            elif avg_pattern_wr >= 0.75:
                pattern_wr_adjustment = 0.05
                logger.info(f"🟢 {symbol}: Good pattern WR ({avg_pattern_wr:.1%}) → +5% confidence boost")
            elif avg_pattern_wr < 0.50:
                pattern_wr_adjustment = -0.10
                logger.warning(f"🔴 {symbol}: Poor pattern WR ({avg_pattern_wr:.1%}) → -10% confidence penalty")
            elif avg_pattern_wr < 0.60:
                pattern_wr_adjustment = -0.05
                logger.warning(f"🟡 {symbol}: Below-average pattern WR ({avg_pattern_wr:.1%}) → -5% confidence penalty")

        base_confidence = min(0.95, base_confidence + pattern_wr_adjustment)

        # 📸 SNAPSHOT-BASED ML PREDICTION: DISABLED to reduce API calls
        # Snapshot capture was causing 429 errors (too many API requests)
        # ML will still learn from entry/exit snapshots (trade_executor.py)
        # But won't use snapshot for live prediction to reduce API load
        snapshot_adjustment = 0.0

        base_confidence = min(0.95, base_confidence + snapshot_adjustment)

        # Adjust confidence based on symbol historical performance
        symbol_threshold = ml_learner.get_confidence_threshold(symbol)
        symbol_adjustment = (symbol_threshold - 0.6) / 0.4  # Normalize around 0.6

        final_confidence = base_confidence * (1 + symbol_adjustment * 0.2)  # ±20% adjustment

        # 🌍 MARKET REGIME ADAPTATION: Adjust confidence based on current market conditions
        regime = ml_learner.current_regime
        regime_adjustment = 0.0

        if regime == "HIGH_VOLATILITY_UNFAVORABLE":
            # Unfavorable conditions → Very conservative (raise confidence bar)
            regime_adjustment = 0.15  # +15% confidence required
            logger.warning(f"⚠️ {symbol}: Unfavorable high volatility regime → +15% confidence required")
        elif regime == "MIXED_CONDITIONS":
            # Mixed conditions → Slightly conservative
            regime_adjustment = 0.05  # +5% confidence required
            logger.info(f"📊 {symbol}: Mixed conditions regime → +5% confidence required")
        elif regime in ["BULL_TREND", "BEAR_TREND", "FAVORABLE_VOLATILE"]:
            # Favorable conditions → Slightly aggressive (lower confidence bar)
            regime_adjustment = -0.05  # -5% confidence required
            logger.info(f"✅ {symbol}: Favorable {regime} regime → -5% confidence required")

        final_confidence = max(0.50, min(0.95, final_confidence - regime_adjustment))  # Apply regime adjustment

        # Determine action based on strongest pattern
        action = 'hold'
        side = None
        leverage = 10  # Default aggressive leverage
        stop_loss = 10.0  # FIXED: Always 10% for consistent risk management

        # 🔥 ULTRA AGGRESSIVE ML LEARNING MODE - Trade on ANY bullish/bearish signal!
        # Goal: Generate more diverse training data across all coins
        pattern_count = len(pattern_features)

        # Count bullish vs bearish signals
        bullish_signals = sum(1 for p in pattern_features if 'bullish' in p or 'oversold' in p or 'weak' in p)
        bearish_signals = sum(1 for p in pattern_features if 'bearish' in p or 'overbought' in p or 'strong' in p)

        # AGGRESSIVE: Trade if confidence >= 45% AND any directional signal exists
        if final_confidence >= 0.45:
            if bullish_signals > bearish_signals and bullish_signals > 0:
                # ANY bullish pattern detected → BUY
                action = 'buy'
                side = 'LONG'
                # FIXED: Always 10% stop-loss for consistent $10 max loss with leverage adjustment
                if final_confidence >= 0.75 and pattern_count >= 4:
                    leverage = 30  # Ultra high confidence: 30x
                    stop_loss = 10.0
                elif final_confidence >= 0.65 and pattern_count >= 3:
                    leverage = 20  # High confidence: 20x
                    stop_loss = 10.0
                elif final_confidence >= 0.55:
                    leverage = 15  # Medium confidence: 15x
                    stop_loss = 10.0
                else:
                    leverage = 10  # Base confidence: 10x
                    stop_loss = 10.0

            elif bearish_signals > bullish_signals and bearish_signals > 0:
                # ANY bearish pattern detected → SELL
                action = 'sell'
                side = 'SHORT'
                # FIXED: Always 10% stop-loss for consistent $10 max loss with leverage adjustment
                if final_confidence >= 0.75 and pattern_count >= 4:
                    leverage = 30  # Ultra high confidence: 30x
                    stop_loss = 10.0
                elif final_confidence >= 0.65 and pattern_count >= 3:
                    leverage = 20  # High confidence: 20x
                    stop_loss = 10.0
                elif final_confidence >= 0.55:
                    leverage = 15  # Medium confidence: 15x
                    stop_loss = 10.0
                else:
                    leverage = 10  # Base confidence: 10x
                    stop_loss = 10.0

        return {
            'action': action,
            'side': side,
            'confidence': final_confidence,
            'suggested_leverage': leverage,
            'stop_loss_percent': stop_loss,
            'risk_reward_ratio': 2.0,
            'reasoning': f"ML-ONLY: {', '.join(pattern_features[:3])}",
            'pattern_count': len(pattern_features),
            'models_used': ['ML-ONLY'],
            'weighted_consensus': False,
            'ensemble_method': 'ml_only'
        }


# Singleton instance
_ai_engine: Optional[AIConsensusEngine] = None


def get_ai_engine() -> AIConsensusEngine:
    """Get or create AI engine instance."""
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIConsensusEngine()
    return _ai_engine
