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
        Get individual analyses from both AI models WITHOUT requiring consensus.
        Returns all analyses for the "best opportunity" selection strategy.

        Args:
            symbol: Trading symbol
            market_data: Market data dict with price, indicators, etc.

        Returns:
            List of individual analyses from each model
        """
        logger.info(f"Requesting individual AI analyses for {symbol}...")

        # Get analysis from both models in parallel with retries
        tasks = [
            self._analyze_with_retry(self._analyze_with_qwen, symbol, market_data, "Qwen3-Max"),
            self._analyze_with_retry(self._analyze_with_deepseek, symbol, market_data, "DeepSeek-V3.2")
        ]

        analyses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle failures gracefully
        valid_analyses = []
        for i, analysis in enumerate(analyses):
            if isinstance(analysis, Exception):
                model_name = ['Qwen3-Max', 'DeepSeek-V3.2'][i]
                logger.error(f"{model_name} analysis failed: {analysis}")
            elif analysis is not None:
                valid_analyses.append(analysis)

        logger.info(f"Got {len(valid_analyses)}/2 analyses for {symbol}")
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
        # Check cache first
        db = await get_db_client()
        cached = await db.get_ai_cache(symbol, 'consensus', '5m')

        if cached:
            logger.debug(f"Using cached AI analysis for {symbol}")
            return cached['analysis_json']

        # Get analysis from both models in parallel with retries
        logger.info(f"Requesting AI analysis for {symbol}...")

        tasks = [
            self._analyze_with_retry(self._analyze_with_qwen, symbol, market_data, "Qwen3-Max"),
            self._analyze_with_retry(self._analyze_with_deepseek, symbol, market_data, "DeepSeek-V3.2")
        ]

        analyses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any failures gracefully
        valid_analyses = []
        for i, analysis in enumerate(analyses):
            if isinstance(analysis, Exception):
                model_name = ['Qwen3-Max', 'DeepSeek-V3.2'][i] if i < 2 else f'Model {i}'
                logger.error(f"{model_name} analysis failed: {analysis}")
            elif analysis is not None:
                valid_analyses.append(analysis)

        # For 2-model system, need both models to respond
        if len(valid_analyses) < 2:
            logger.warning(f"Insufficient AI responses for {symbol}: {len(valid_analyses)}/2")
            return {
                'action': 'hold',
                'side': None,
                'confidence': 0.0,
                'consensus': False,
                'consensus_count': len(valid_analyses),
                'reason': 'Insufficient AI responses - need both models',
                'suggested_leverage': 2,
                'stop_loss_percent': 7.0,
                'risk_reward_ratio': 0.0,
                'reasoning': 'Both AI models must respond for consensus',
                'models_used': [a.get('model_name', 'unknown') for a in valid_analyses]
            }

        # Count votes (2-model system: both must agree for buy/sell)
        buy_votes = sum(1 for a in valid_analyses if a.get('action') == 'buy')
        sell_votes = sum(1 for a in valid_analyses if a.get('action') == 'sell')
        hold_votes = sum(1 for a in valid_analyses if a.get('action') == 'hold')

        logger.info(f"AI Votes for {symbol}: Buy={buy_votes}, Sell={sell_votes}, Hold={hold_votes}")

        # Determine consensus (requires both models to agree)
        if buy_votes == 2:
            consensus_action = 'buy'
            consensus_side = 'LONG'
        elif sell_votes == 2:
            consensus_action = 'sell'
            consensus_side = 'SHORT'
        else:
            # If models disagree or both say hold, we hold
            consensus_action = 'hold'
            consensus_side = None

        # Get agreeing models (for 2-model system)
        agreeing_models = [a for a in valid_analyses if a.get('action') == consensus_action]

        # For 2-model system: if both agree on buy/sell, use their average
        # If they disagree (hold), use lower confidence
        if len(agreeing_models) == 2:
            # Both models agree - average their metrics
            avg_confidence = sum(a.get('confidence', 0) for a in agreeing_models) / 2
            avg_leverage = int(sum(a.get('suggested_leverage', 3) for a in agreeing_models) / 2)
            avg_stop_loss = sum(a.get('stop_loss_percent', 7.0) for a in agreeing_models) / 2
            avg_risk_reward = sum(a.get('risk_reward_ratio', 0) for a in agreeing_models) / 2
        else:
            # Models disagree - use conservative values
            avg_confidence = 0.0
            avg_leverage = 2
            avg_stop_loss = 7.0
            avg_risk_reward = 0.0

        # Combine reasoning from both models
        combined_reasoning = " | ".join([
            f"{a.get('model_name', 'AI')}: {a.get('reasoning', 'N/A')[:50]}"
            for a in valid_analyses
        ])

        # Build consensus response
        consensus = {
            'action': consensus_action,
            'side': consensus_side,
            'confidence': round(avg_confidence, 2),
            'consensus': len(agreeing_models) == 2,  # True only if both models agree
            'consensus_count': len(agreeing_models),
            'suggested_leverage': max(2, min(5, avg_leverage)),  # Clamp to 2-5
            'stop_loss_percent': max(5.0, min(10.0, avg_stop_loss)),  # Clamp to 5-10%
            'risk_reward_ratio': round(avg_risk_reward, 2),
            'reasoning': combined_reasoning,
            'models_used': [a.get('model_name', 'unknown') for a in valid_analyses],
            'individual_analyses': valid_analyses  # Store both analyses for reference
        }

        # Cache result
        await db.set_ai_cache(
            symbol, 'consensus', '5m', consensus,
            self.settings.ai_cache_ttl_seconds
        )

        logger.info(
            f"AI Consensus for {symbol}: {consensus_action.upper()} "
            f"(confidence: {avg_confidence:.1%}, models: {len(agreeing_models)})"
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
                temperature=0.3,
                max_tokens=2048
            )

            content = response.choices[0].message.content
            analysis = parse_ai_response(content)

            if analysis:
                analysis['model_name'] = 'deepseek'
                logger.debug(f"DeepSeek analysis: {analysis.get('action')} ({analysis.get('confidence', 0):.0%})")
                return analysis
            else:
                logger.warning("Failed to parse DeepSeek response")
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
