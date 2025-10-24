"""
Market Scanner - Scans multiple symbols for trading opportunities.
Gets AI consensus and executes best trades.
"""

import asyncio
from typing import Dict, Any, List, Optional
from decimal import Decimal
from src.config import get_settings
from src.exchange_client import get_exchange_client
from src.ai_engine import get_ai_engine
from src.risk_manager import get_risk_manager
from src.trade_executor import get_trade_executor
from src.telegram_notifier import get_notifier
from src.utils import setup_logging, is_bullish, is_bearish
from src.indicators import calculate_indicators, detect_market_regime

logger = setup_logging()


class MarketScanner:
    """Scans multiple symbols and identifies best trading opportunities."""

    def __init__(self):
        self.settings = get_settings()
        self.symbols = self.settings.trading_symbols

    async def scan_and_execute(self) -> None:
        """
        NEW STRATEGY: "Best Opportunity" Selection

        1. Scan all symbols
        2. Get INDIVIDUAL AI analyses from BOTH models for each symbol
        3. Collect ALL analyses (2 models × N symbols = 2N analyses)
        4. Select the SINGLE analysis with HIGHEST confidence
        5. Execute that trade if it meets minimum criteria

        No consensus required - just pick the best single opportunity!
        """
        logger.info("🔍 Starting market scan (Best Opportunity Strategy)...")
        notifier = get_notifier()

        await notifier.send_alert('info', '🔍 Market scan started...')

        all_analyses = []  # Collect ALL individual analyses

        for symbol in self.symbols:
            try:
                # Get market data
                market_data = await self.gather_market_data(symbol)

                if not market_data:
                    continue

                # Get INDIVIDUAL analyses from BOTH models (no consensus)
                ai_engine = get_ai_engine()
                individual_analyses = await ai_engine.get_individual_analyses(symbol, market_data)

                # Add each analysis to our collection
                for analysis in individual_analyses:
                    # Skip HOLD signals
                    if analysis.get('action') == 'hold':
                        logger.debug(
                            f"{symbol} ({analysis.get('model_name', 'AI')}): HOLD "
                            f"(confidence: {analysis.get('confidence', 0):.1%})"
                        )
                        continue

                    # Add to collection with metadata
                    all_analyses.append({
                        'symbol': symbol,
                        'analysis': analysis,
                        'market_data': market_data,
                        'model': analysis.get('model_name', 'unknown'),
                        'confidence': analysis.get('confidence', 0)
                    })

                    logger.info(
                        f"📊 {symbol} ({analysis.get('model_name', 'AI')}): "
                        f"{analysis.get('action', 'unknown').upper()} - "
                        f"Confidence: {analysis.get('confidence', 0):.1%}"
                    )

                # Rate limiting
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

        # Sort ALL analyses by confidence (highest first)
        all_analyses.sort(key=lambda x: x['confidence'], reverse=True)

        logger.info(f"📈 Total analyses collected: {len(all_analyses)}")

        if all_analyses:
            # Get the BEST analysis (highest confidence)
            best = all_analyses[0]

            logger.info(
                f"🎯 BEST OPPORTUNITY: {best['symbol']} "
                f"({best['model'].upper()}) - "
                f"{best['analysis']['action'].upper()} - "
                f"Confidence: {best['confidence']:.1%}"
            )

            # Check minimum confidence threshold
            if best['confidence'] >= float(self.settings.min_ai_confidence):
                await notifier.send_scan_result(
                    best['symbol'],
                    best['confidence'],
                    best['analysis']['action']
                )

                # Execute the best opportunity
                await self.execute_trade(best)
            else:
                await notifier.send_alert(
                    'info',
                    f"Best opportunity: {best['symbol']} ({best['model']}) - "
                    f"Confidence: {best['confidence']:.1%}\n"
                    f"Below minimum threshold ({float(self.settings.min_ai_confidence):.1%}). "
                    f"Waiting for better setup..."
                )
        else:
            logger.info("😐 No good opportunities found (all models recommend HOLD)")
            await notifier.send_alert(
                'info',
                '😐 No good opportunities found. All AI models recommend HOLD. '
                'Will scan again in 5 minutes.'
            )

    async def gather_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Gather comprehensive market data for AI analysis.

        Returns:
            Market data dict with price, indicators, regime, etc.
        """
        try:
            exchange = await get_exchange_client()

            # OHLCV data (multiple timeframes)
            ohlcv_15m = await exchange.fetch_ohlcv(symbol, '15m', limit=100)
            ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=100)
            ohlcv_4h = await exchange.fetch_ohlcv(symbol, '4h', limit=50)

            # Current ticker
            ticker = await exchange.fetch_ticker(symbol)

            # Funding rate
            funding_rate = await exchange.fetch_funding_rate(symbol)

            # Technical indicators
            indicators_15m = calculate_indicators(ohlcv_15m)
            indicators_1h = calculate_indicators(ohlcv_1h)
            indicators_4h = calculate_indicators(ohlcv_4h)

            # Market regime detection
            regime = detect_market_regime(ohlcv_1h)

            return {
                'symbol': symbol,
                'current_price': ticker['last'],
                'volume_24h': ticker.get('quoteVolume', 0),
                'ohlcv': {
                    '15m': ohlcv_15m[-20:],  # Last 20 candles
                    '1h': ohlcv_1h[-20:],
                    '4h': ohlcv_4h[-20:]
                },
                'indicators': {
                    '15m': indicators_15m,
                    '1h': indicators_1h,
                    '4h': indicators_4h
                },
                'funding_rate': funding_rate,
                'market_regime': regime
            }

        except Exception as e:
            logger.error(f"Error gathering market data for {symbol}: {e}")
            return None

    def calculate_opportunity_score(self, ai_analysis: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """
        Score opportunity from 0-100 based on multiple factors.

        Scoring breakdown:
        - AI Confidence: 40 points
        - AI Consensus strength: 20 points
        - Market regime appropriateness: 20 points
        - Risk/Reward ratio: 10 points
        - Technical alignment: 10 points
        """
        score = 0.0

        # AI Confidence (40 points)
        confidence = ai_analysis.get('confidence', 0)
        score += confidence * 40

        # AI Consensus strength (20 points)
        consensus_count = ai_analysis.get('consensus_count', 0)
        if consensus_count == 3:  # All 3 AIs agree
            score += 20
        elif consensus_count == 2:  # 2 out of 3
            score += 10

        # Market regime appropriateness (20 points)
        regime = market_data.get('market_regime', 'UNKNOWN')
        action = ai_analysis.get('action', 'hold')

        if regime == 'TRENDING' and action in ['buy', 'sell']:
            score += 20  # Excellent for leverage
        elif regime == 'RANGING':
            score += 10  # Moderate
        elif regime == 'VOLATILE':
            score += 5   # Risky

        # Risk/Reward ratio (10 points)
        rr_ratio = ai_analysis.get('risk_reward_ratio', 0)
        if rr_ratio >= 2.0:
            score += 10
        elif rr_ratio >= 1.5:
            score += 5

        # Technical alignment (10 points)
        # All timeframes should agree with the signal
        indicators = market_data.get('indicators', {})
        timeframe_agreement = 0

        for tf in ['15m', '1h', '4h']:
            tf_indicators = indicators.get(tf, {})
            if is_bullish(tf_indicators) and action == 'buy':
                timeframe_agreement += 1
            elif is_bearish(tf_indicators) and action == 'sell':
                timeframe_agreement += 1

        score += (timeframe_agreement / 3) * 10

        return min(score, 100)

    async def execute_trade(self, opportunity: Dict[str, Any]) -> None:
        """
        Execute the trade with proper risk management.

        Args:
            opportunity: Best opportunity dict with symbol, analysis, market_data
        """
        symbol = opportunity['symbol']
        analysis = opportunity['analysis']
        market_data = opportunity['market_data']

        logger.info(f"📊 Preparing to execute trade: {symbol} {analysis['side']}")

        # Validate with risk manager
        risk_manager = get_risk_manager()
        notifier = get_notifier()

        trade_params = {
            'symbol': symbol,
            'side': analysis['side'],
            'leverage': analysis['suggested_leverage'],
            'stop_loss_percent': analysis['stop_loss_percent'],
            'current_price': market_data['current_price']
        }

        validation = await risk_manager.validate_trade(trade_params)

        if not validation['approved']:
            logger.warning(f"❌ Trade rejected by risk manager: {validation['reason']}")
            await notifier.send_alert(
                'warning',
                f"Trade rejected by risk manager:\n{symbol} {analysis['side']}\n\n{validation['reason']}"
            )
            return

        # Use adjusted parameters if provided
        if validation.get('adjusted_params'):
            trade_params = validation['adjusted_params']
            logger.info("Using risk-adjusted trade parameters")

        # Execute trade
        executor = get_trade_executor()
        success = await executor.open_position(trade_params, analysis, market_data)

        if success:
            logger.info(f"✅ Trade executed successfully: {symbol} {analysis['side']}")
        else:
            logger.error(f"❌ Trade execution failed: {symbol}")


# Singleton instance
_market_scanner: Optional[MarketScanner] = None


def get_market_scanner() -> MarketScanner:
    """Get or create market scanner instance."""
    global _market_scanner
    if _market_scanner is None:
        _market_scanner = MarketScanner()
    return _market_scanner
