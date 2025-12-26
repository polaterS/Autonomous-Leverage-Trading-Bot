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
PA_ONLY_VERSION = "6.4-PA-ONLY"  # PA-ONLY MODE: ML disabled, using only Support/Resistance + Trend + Volume
logger.info(f"📊 AI Engine initialized - Mode: {PA_ONLY_VERSION} (PA-ONLY: ML disabled, S/R+Trend+Volume, uses Railway config for leverage/SL)")


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
            settings = get_settings()
            return {
                'action': 'hold',
                'side': None,
                'confidence': 0.0,
                'reasoning': 'No price data available',
                'suggested_leverage': settings.max_leverage,  # 🔧 FIX: Use config
                'stop_loss_percent': float(settings.max_stop_loss_percent),  # 🔧 FIX: Use config
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
                settings = get_settings()
                return {
                    'action': 'hold',
                    'side': None,
                    'confidence': 0.0,
                    'reasoning': 'Insufficient OHLCV data for PA analysis',
                    'suggested_leverage': settings.max_leverage,  # 🔧 FIX: Use config
                    'stop_loss_percent': float(settings.max_stop_loss_percent),  # 🔧 FIX: Use config
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
            settings = get_settings()
            return {
                'action': 'hold',
                'side': None,
                'confidence': 0.0,
                'reasoning': f'Insufficient OHLCV data ({len(df)} candles, need 50+)',
                'suggested_leverage': settings.max_leverage,  # 🔧 FIX: Use config
                'stop_loss_percent': float(settings.max_stop_loss_percent),  # 🔧 FIX: Use config
                'models_used': ['PA-ONLY'],
                'ensemble_method': 'pa_only',
                'risk_reward_ratio': 0.0,
                'price_action': None
            }

        # 🔥 USE ADVANCED PA ANALYZER instead of basic logic!
        # price_action_analyzer.should_enter_trade_v4_20251122() includes:
        # - S/R analysis (Multi-timeframe v4.0!)
        # - Trend analysis
        # - Volume analysis
        # - Candle patterns
        # - Order flow analysis
        # - Market structure
        # - RISK/REWARD filtering (minimum 2:1 ratio)
        # - Stop-loss optimization
        #
        # 🔥 RENAMED FUNCTION FOR BYTECODE CACHE BYPASS!
        # OLD: Basic PA logic here (only S/R + trend + volume)
        # NEW: Advanced PA analyzer with full confluence system

        # 🔥 v6.5: GET EXCHANGE FOR MULTI-TIMEFRAME S/R ANALYSIS!
        # Without exchange, only 15m S/R levels are used (missing 1w, 1d, 4h, 1h)
        from src.exchange_client import get_exchange_client
        try:
            exchange = await get_exchange_client()
        except Exception as e:
            logger.warning(f"⚠️ Could not get exchange client: {e} - using 15m S/R only")
            exchange = None

        # Call should_enter_trade_v4_20251122() for professional PA analysis
        pa_result = await pa_analyzer.should_enter_trade_v4_20251122(
            symbol=symbol,
            df=df,
            ml_signal='BUY',  # Check both LONG and SHORT
            ml_confidence=0.75,  # Base confidence
            current_price=current_price,
            exchange=exchange  # 🔥 v6.5: Pass exchange for multi-TF S/R!
        )

        # Check LONG first
        if pa_result['should_enter']:
            pa_action = 'buy'
            pa_side = 'LONG'
            pa_confidence = min(0.75 + (pa_result['confidence_boost'] / 100), 0.95)
            pa_reasoning = pa_result['reason']
            logger.info(f"✅ LONG setup found: {pa_reasoning}")
        else:
            # Check SHORT
            pa_result_short = await pa_analyzer.should_enter_trade_v4_20251122(
                symbol=symbol,
                df=df,
                ml_signal='SELL',  # Check SHORT
                ml_confidence=0.75,
                current_price=current_price,
                exchange=exchange  # 🔥 v6.5: Pass exchange for multi-TF S/R!
            )

            if pa_result_short['should_enter']:
                pa_action = 'sell'
                pa_side = 'SHORT'
                pa_confidence = min(0.75 + (pa_result_short['confidence_boost'] / 100), 0.95)
                pa_reasoning = pa_result_short['reason']
                logger.info(f"✅ SHORT setup found: {pa_reasoning}")
                pa_result = pa_result_short  # Use SHORT result
            else:
                # No setup found for either direction
                pa_action = 'hold'
                pa_side = None
                pa_confidence = 0.0
                pa_reasoning = pa_result.get('reason', 'No valid PA setup found')
                logger.info(f"🚫 No PA setup: {pa_reasoning}")

        # Get PA analysis data for metrics
        sr_analysis = pa_result['analysis']['support_resistance']
        trend = pa_result['analysis']['trend']
        volume = pa_result['analysis']['volume']

        supports = [s['price'] for s in sr_analysis.get('support', [])]
        resistances = [r['price'] for r in sr_analysis.get('resistance', [])]

        support_count = len(supports)
        resistance_count = len(resistances)
        total_sr_levels = support_count + resistance_count

        # Get nearest S/R levels
        nearest_support = min(supports, key=lambda x: abs(x - current_price)) if supports else None
        nearest_resistance = min(resistances, key=lambda x: abs(x - current_price)) if resistances else None

        dist_to_support = abs(current_price - nearest_support) / current_price if nearest_support else 0
        dist_to_resistance = abs(current_price - nearest_resistance) / current_price if nearest_resistance else 0

        adx = trend.get('adx', 0)

        logger.info(
            f"📊 PA Analysis: {symbol} | "
            f"Action: {pa_action.upper()} @ {pa_confidence:.1%} | "
            f"Trend: {trend['direction']} ({trend['strength']}, ADX {adx:.1f}) | "
            f"Volume: {volume['surge_ratio']:.1f}x | "
            f"S/R: {support_count} supports, {resistance_count} resistances | "
            f"R/R: {pa_result.get('rr_ratio', 0):.2f}:1"
        )

        # 🔧 FIX: Calculate missing PA metrics for telegram_notifier
        # Calculate S/R strength score (0-1 based on number of levels)
        # 6+ levels = 100% strength, scales linearly below that
        sr_strength_score = min(total_sr_levels / 6.0, 1.0)

        # Calculate swing lows/highs from sr_analysis
        swing_lows_count = len(sr_analysis.get('support', []))
        swing_highs_count = len(sr_analysis.get('resistance', []))

        # Calculate Risk/Reward ratios for both directions
        # R/R = (distance to target) / (distance to stop loss)
        # Target = opposite S/R level, Stop = entry +/- 2% (conservative)
        rr_long = 0.0
        rr_short = 0.0
        if supports and resistances and nearest_support is not None and nearest_resistance is not None:
            # LONG: Entry at support, target at resistance, stop 2% below support
            dist_to_resistance_abs = abs(nearest_resistance - current_price)
            stop_loss_dist_long = current_price * 0.02  # 2% stop
            rr_long = dist_to_resistance_abs / stop_loss_dist_long if stop_loss_dist_long > 0 else 0

            # SHORT: Entry at resistance, target at support, stop 2% above resistance
            dist_to_support_abs = abs(current_price - nearest_support)
            stop_loss_dist_short = current_price * 0.02  # 2% stop
            rr_short = dist_to_support_abs / stop_loss_dist_short if stop_loss_dist_short > 0 else 0

        # Calculate trend quality (0-1 based on ADX strength)
        # ADX < 20 = weak, 20-40 = moderate, 40+ = strong
        trend_quality = min(adx / 50.0, 1.0) if adx > 0 else 0.0

        # Return PA-only result
        # 🔧 FIX: Use config values instead of hardcoded (Railway env vars)
        from src.config import get_settings
        settings = get_settings()
        
        # 🗞️ v6.5: NEWS SENTIMENT INTEGRATION
        # Analyze news/social sentiment and adjust confidence
        news_sentiment = None
        news_adjustment = 0
        news_skip_reason = None
        
        if settings.enable_news_filter:
            try:
                from src.news_sentiment_analyzer import get_news_analyzer
                news_analyzer = get_news_analyzer()
                news_sentiment = await news_analyzer.analyze_sentiment(symbol)
                
                if news_sentiment:
                    news_adjustment = news_sentiment.confidence_adjustment
                    
                    # Check if we should skip trade due to very bearish news
                    if news_sentiment.should_skip_trade and pa_action != 'hold':
                        logger.warning(
                            f"🗞️ NEWS FILTER: Skipping {symbol} {pa_action.upper()} - "
                            f"{news_sentiment.skip_reason}"
                        )
                        news_skip_reason = news_sentiment.skip_reason
                        pa_action = 'hold'
                        pa_side = None
                        pa_confidence = 0.0
                        pa_reasoning = f"Skipped due to news: {news_sentiment.skip_reason}"
                    else:
                        # Apply confidence adjustment
                        original_confidence = pa_confidence
                        pa_confidence = max(0.0, min(0.95, pa_confidence + (news_adjustment / 100)))
                        
                        logger.info(
                            f"🗞️ News Sentiment: {news_sentiment.overall_sentiment.value} | "
                            f"F&G: {news_sentiment.fear_greed_index} | "
                            f"Adjustment: {news_adjustment:+d}% | "
                            f"Confidence: {original_confidence:.1%} → {pa_confidence:.1%}"
                        )
            except Exception as e:
                logger.warning(f"⚠️ News sentiment analysis failed: {e}")
        
        return {
            'action': pa_action,
            'side': pa_side,
            'confidence': pa_confidence,
            'reasoning': f"PA-ONLY: {pa_reasoning}",
            'suggested_leverage': settings.max_leverage,  # 🔧 FIX: Use Railway MAX_LEVERAGE (was hardcoded 20x)
            'stop_loss_percent': float(settings.max_stop_loss_percent),  # 🔧 FIX: Use Railway MAX_STOP_LOSS_PERCENT (was hardcoded 9%)
            'model_name': 'PA-ONLY',  # 🔥 FIX: Add model_name field so market_scanner doesn't show "unknown"
            'models_used': ['PA-ONLY'],
            'ensemble_method': 'pa_only',
            'risk_reward_ratio': 0.0,
            
            # 🗞️ v6.5: News sentiment data
            'news_sentiment': {
                'sentiment': news_sentiment.overall_sentiment.value if news_sentiment else 'neutral',
                'score': news_sentiment.sentiment_score if news_sentiment else 0.0,
                'fear_greed': news_sentiment.fear_greed_index if news_sentiment else 50,
                'adjustment': news_adjustment,
                'news_count': news_sentiment.news_count if news_sentiment else 0,
                'recommendation': news_sentiment.recommendation if news_sentiment else 'N/A',
                'skip_reason': news_skip_reason,
            } if settings.enable_news_filter else None,
            
            'price_action': {
                # Original fields
                'trend': trend,
                'volume': volume,
                'support_resistance': sr_analysis,
                'nearest_support': nearest_support if supports else None,
                'nearest_resistance': nearest_resistance if resistances else None,
                'support_dist': dist_to_support if supports else 0,
                'resistance_dist': dist_to_resistance if resistances else 0,

                # 🔥 NEW: Metrics for telegram_notifier (was missing - causing 0 values!)
                'sr_strength_score': sr_strength_score,
                'swing_lows_count': swing_lows_count,
                'swing_highs_count': swing_highs_count,
                'rr_long': rr_long,
                'rr_short': rr_short,
                'trend_direction': trend.get('direction', 'SIDEWAYS'),
                'trend_adx': trend.get('adx', 0),
                'trend_quality': trend_quality,
                'volume_surge': volume.get('is_surge', False),
                'volume_surge_ratio': volume.get('surge_ratio', 1.0)
            }
        }

        # Note: AI+ML ensemble code removed (PA-only mode active)

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
            # 🎯 USER UPDATE: Max 5x leverage, ultra conservative
            if confidence >= 0.85:
                leverage = 5  # Ultra high confidence (max)
                stop_loss = 18.0
            elif confidence >= 0.75:
                leverage = 4  # High confidence
                stop_loss = 16.0
            elif confidence >= 0.70:
                leverage = 4  # Acceptable confidence
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
