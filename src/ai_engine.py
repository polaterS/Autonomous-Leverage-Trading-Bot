"""
AI Consensus Engine for trading analysis.
Integrates Qwen3-Max and DeepSeek-V3.2 for 2-model consensus.
"""

import asyncio
import openai
from typing import Dict, Any, Optional, List
from decimal import Decimal
from src.config import get_settings, LEVERAGE_TRADING_SYSTEM_PROMPT, build_analysis_prompt
from src.utils import setup_logging, parse_ai_response
from src.database import get_db_client
from src.redis_client import get_redis_client
import json

logger = setup_logging()


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

    async def get_individual_analyses(self, symbol: str, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get individual analyses from DeepSeek model + ML ENHANCEMENT.
        (Qwen3-Max disabled due to API key issues)

        Args:
            symbol: Trading symbol
            market_data: Market data dict with price, indicators, etc.

        Returns:
            List of individual analyses with ML-adjusted confidence (currently just DeepSeek)
        """
        logger.info(f"Requesting AI analysis for {symbol} (DeepSeek + ML)...")

        # Get analysis from DeepSeek only
        analysis = await self._analyze_with_retry(
            self._analyze_with_deepseek,
            symbol,
            market_data,
            "DeepSeek-V3.2"
        )

        # Handle failures gracefully
        valid_analyses = []
        if analysis is not None:
            # 🧠 ML ENHANCEMENT: Adjust confidence based on historical patterns
            try:
                from src.ml_pattern_learner import get_ml_learner
                ml_learner = await get_ml_learner()
                ml_result = await ml_learner.analyze_opportunity(symbol, analysis, market_data)

                # Update analysis with ML insights
                analysis['original_confidence'] = analysis['confidence']
                analysis['confidence'] = ml_result['adjusted_confidence']
                analysis['ml_adjustment'] = ml_result['ml_adjustment']
                analysis['ml_score'] = ml_result['ml_score']
                analysis['ml_reasoning'] = ml_result['reasoning']

                if ml_result.get('regime_warning'):
                    analysis['regime_warning'] = ml_result['regime_warning']

                logger.info(f"🧠 ML Enhanced: {symbol} confidence {analysis['original_confidence']:.1%} → "
                          f"{analysis['confidence']:.1%} (Δ {ml_result['ml_adjustment']:+.1%})")

            except Exception as ml_error:
                logger.warning(f"ML enhancement failed for {symbol}: {ml_error} (continuing without ML)")

            valid_analyses.append(analysis)
        else:
            logger.error(f"DeepSeek analysis failed for {symbol}")

        logger.info(f"Got {len(valid_analyses)}/1 analyses for {symbol}")
        return valid_analyses

    async def get_consensus(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get AI consensus - require at least 2 out of 3 models to agree.
        Implements retry logic and graceful degradation.

        Args:
            symbol: Trading symbol
            market_data: Market data dict with price, indicators, etc.

        Returns:
            Consensus analysis with action, confidence, and reasoning
        """
        # Check Redis cache first (much faster than PostgreSQL)
        redis = await get_redis_client()
        cached = await redis.get_cached_ai_analysis(symbol, '5m')

        if cached:
            logger.debug(f"✅ Using cached AI analysis for {symbol} (from Redis)")
            return cached

        # Get analysis from DeepSeek only (Qwen3-Max disabled)
        logger.info(f"Requesting AI analysis for {symbol} (DeepSeek only)...")

        analysis = await self._analyze_with_retry(
            self._analyze_with_deepseek,
            symbol,
            market_data,
            "DeepSeek-V3.2"
        )

        # Handle failure
        if analysis is None:
            logger.warning(f"DeepSeek failed for {symbol}")
            return {
                'action': 'hold',
                'side': None,
                'confidence': 0.0,
                'consensus': False,
                'consensus_count': 0,
                'reason': 'AI analysis failed',
                'suggested_leverage': 2,
                'stop_loss_percent': 7.0,
                'risk_reward_ratio': 0.0,
                'reasoning': 'DeepSeek API unavailable',
                'models_used': []
            }

        # Single model - use its values directly
        consensus = {
            'action': analysis.get('action', 'hold'),
            'side': analysis.get('side'),
            'confidence': round(analysis.get('confidence', 0), 2),
            'consensus': True,  # Single model always "agrees" with itself
            'consensus_count': 1,
            'suggested_leverage': max(2, min(5, analysis.get('suggested_leverage', 3))),
            'stop_loss_percent': max(5.0, min(10.0, analysis.get('stop_loss_percent', 7.0))),
            'risk_reward_ratio': round(analysis.get('risk_reward_ratio', 0), 2),
            'reasoning': analysis.get('reasoning', 'N/A'),
            'models_used': ['deepseek'],
            'individual_analyses': [analysis]
        }

        # Cache result in Redis (fast access for next 5 minutes)
        redis = await get_redis_client()
        await redis.cache_ai_analysis(
            symbol, '5m', consensus,
            self.settings.ai_cache_ttl_seconds
        )

        # Also cache in database for analysis/debugging
        db = await get_db_client()
        try:
            await db.set_ai_cache(
                symbol, 'consensus', '5m', consensus,
                self.settings.ai_cache_ttl_seconds
            )
        except Exception as e:
            logger.warning(f"Failed to cache in DB (non-critical): {e}")

        logger.info(
            f"✅ AI Analysis for {symbol}: {consensus['action'].upper()} "
            f"(confidence: {consensus['confidence']:.1%}, confluence: {consensus.get('confluence_count', 0)} factors)"
        )

        return consensus

    async def _analyze_with_qwen(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get analysis from Qwen3-Max via OpenRouter."""
        try:
            prompt = build_analysis_prompt(symbol, market_data)

            response = await self.qwen_client.chat.completions.create(
                model="qwen/qwen3-max",  # OpenRouter model identifier
                messages=[
                    {"role": "system", "content": LEVERAGE_TRADING_SYSTEM_PROMPT},
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
        """Get analysis from DeepSeek V3."""
        try:
            prompt = build_analysis_prompt(symbol, market_data)

            response = await self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": LEVERAGE_TRADING_SYSTEM_PROMPT},
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


# Singleton instance
_ai_engine: Optional[AIConsensusEngine] = None


def get_ai_engine() -> AIConsensusEngine:
    """Get or create AI engine instance."""
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIConsensusEngine()
    return _ai_engine
