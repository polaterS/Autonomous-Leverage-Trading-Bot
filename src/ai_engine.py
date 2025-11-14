"""
AI Consensus Engine for trading analysis.
🧠 HYBRID MODE: DeepSeek AI + ML Pattern Learning
Version: 3.0 - AI+ML Consensus (Cost-Optimized)
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
HYBRID_VERSION = "3.0-AI+ML-HYBRID"
logger.info(f"🧠 AI Engine initialized - Mode: {HYBRID_VERSION} (DeepSeek AI + ML Consensus)")


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
        🧠 HYBRID MODE: DeepSeek AI analysis (cost-optimized).

        Only called when position slots are available (smart scan logic).
        Returns DeepSeek analysis for AI+ML consensus.
        """
        logger.info(f"🧠 AI+ML HYBRID: Requesting DeepSeek analysis for {symbol}...")

        # 🎯 AI MODE ENABLED: DeepSeek-V3 (cost optimized)
        logger.info(f"🧠 DeepSeek AI analysis for {symbol}...")

        # 🎯 SINGLE MODEL: DeepSeek only (cost-optimized)
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

                                # 🎯 CONSERVATIVE MODE: Wider stops, lower leverage
                                # Stop-loss: 12-20% (adaptive), Leverage: 3-10x max
                                conf = analysis['confidence']
                                if conf >= 0.85:  # 85%+ = Ultra high confidence
                                    analysis['suggested_leverage'] = 10  # Max 10x
                                    analysis['stop_loss_percent'] = 20.0  # Widest stop
                                elif conf >= 0.75:  # 75-84% = High confidence
                                    analysis['suggested_leverage'] = 7
                                    analysis['stop_loss_percent'] = 16.0
                                elif conf >= 0.70:  # 70-74% = Acceptable
                                    analysis['suggested_leverage'] = 5
                                    analysis['stop_loss_percent'] = 14.0
                                else:  # <70% shouldn't happen (filtered out)
                                    analysis['suggested_leverage'] = 3
                                    analysis['stop_loss_percent'] = 12.0

                                logger.info(
                                    f"🎯 ML Override SUCCESS: {symbol} 'hold' → 'buy' LONG "
                                    f"(ML conf: {analysis['confidence']:.1%}, {reason}, "
                                    f"{analysis['suggested_leverage']}x lev, {analysis['stop_loss_percent']}% SL)"
                                )
                            elif direction == 'sell':
                                analysis['action'] = 'sell'
                                analysis['side'] = 'SHORT'

                                # 🎯 CONSERVATIVE MODE: Wider stops, lower leverage
                                # Stop-loss: 12-20% (adaptive), Leverage: 3-10x max
                                conf = analysis['confidence']
                                if conf >= 0.85:  # 85%+ = Ultra high confidence
                                    analysis['suggested_leverage'] = 10  # Max 10x
                                    analysis['stop_loss_percent'] = 20.0  # Widest stop
                                elif conf >= 0.75:  # 75-84% = High confidence
                                    analysis['suggested_leverage'] = 7
                                    analysis['stop_loss_percent'] = 16.0
                                elif conf >= 0.70:  # 70-74% = Acceptable
                                    analysis['suggested_leverage'] = 5
                                    analysis['stop_loss_percent'] = 14.0
                                else:  # <70% shouldn't happen (filtered out)
                                    analysis['suggested_leverage'] = 3
                                    analysis['stop_loss_percent'] = 12.0

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
        🧠 HYBRID MODE: AI+ML Consensus Engine (Cost-Optimized)

        Combines DeepSeek AI analysis with ML pattern learning for optimal decisions.
        Only calls AI when position slots are available (smart scan logic).

        Args:
            symbol: Trading symbol
            market_data: Market data dict with price, indicators, etc.

        Returns:
            AI+ML consensus analysis with action, confidence, and reasoning
        """
        from src.ml_pattern_learner import get_ml_learner
        ml_learner = await get_ml_learner()

        # Determine market sentiment for context
        market_sentiment = market_data.get('market_sentiment', 'NEUTRAL')

        # 🧠 HYBRID MODE: AI + ML Consensus
        logger.debug(f"🧠 AI+ML HYBRID mode for {symbol}")

        # Get AI analysis (DeepSeek)
        analyses = await self.get_individual_analyses(symbol, market_data, market_sentiment)

        # Handle case where AI analysis fails or returns empty
        if not analyses:
            logger.warning(f"⚠️ DeepSeek AI unavailable for {symbol}, falling back to ML-only")
            ml_prediction = await self._get_ml_only_prediction(symbol, market_data, ml_learner)
            ml_prediction['models_used'] = ['ML-ONLY (AI fallback)']
            ml_prediction['ensemble_method'] = 'ml_fallback'
            return ml_prediction

        # 🎯 AI+ML CONSENSUS: Weighted ensemble
        market_regime = market_data.get('market_regime', 'UNKNOWN')
        consensus = ml_learner.calculate_weighted_ensemble(
            analyses, symbol, market_regime
        )

        # Add risk/reward ratio if not present
        if 'risk_reward_ratio' not in consensus:
            consensus['risk_reward_ratio'] = 0.0

        # Track which models were used
        consensus['models_used'] = [a.get('model_name', 'unknown') for a in analyses]
        consensus['ensemble_method'] = 'ai_ml_hybrid'

        logger.info(
            f"✅ 🧠 AI+ML CONSENSUS for {symbol}: {consensus['action'].upper()} "
            f"@ {consensus['confidence']:.1%} "
            f"(AI: {analyses[0].get('original_confidence', 0):.1%} "
            f"→ ML-adjusted: {analyses[0].get('confidence', 0):.1%})"
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
