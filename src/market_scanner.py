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
from src.telegram_bot import get_telegram_bot
from src.database import get_db_client
from src.utils import setup_logging, is_bullish, is_bearish
from src.indicators import (
    calculate_indicators,
    calculate_multi_timeframe_indicators,
    detect_market_regime,
    detect_support_resistance_levels,
    calculate_volume_profile,
    calculate_fibonacci_levels,
    analyze_funding_rate_trend,
    detect_divergences,
    analyze_order_flow,
    detect_smart_money_concepts,
    calculate_volatility_bands,
    calculate_confluence_score,
    calculate_momentum_strength,
    calculate_btc_correlation,
    analyze_open_interest,
    estimate_liquidation_levels
)

logger = setup_logging()


class MarketScanner:
    """Scans multiple symbols and identifies best trading opportunities."""

    def __init__(self):
        self.settings = get_settings()
        self.symbols = self.settings.trading_symbols

    async def scan_and_execute(self) -> None:
        """
        ULTRA PROFESSIONAL STRATEGY: "Best Opportunity" with Market Breadth

        1. Scan all symbols
        2. Calculate market breadth (% bullish/bearish)
        3. Get INDIVIDUAL AI analyses for each symbol
        4. Calculate confluence scores
        5. Select the SINGLE analysis with HIGHEST confidence + confluence
        6. Execute that trade if it meets minimum criteria

        Elite institutional approach!
        """
        logger.info("🔍 Starting market scan (Ultra Professional Strategy)...")
        notifier = get_notifier()

        await notifier.send_alert('info', '🔍 Market scan started...')

        all_analyses = []  # Collect ALL individual analyses
        telegram_bot = await get_telegram_bot()
        total_symbols = len(self.symbols)
        scanned = 0

        # Market breadth counters
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        # 🚀 PARALLEL SCANNING: Analyze all symbols concurrently with rate limiting
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent scans
        scan_tasks = [self._scan_symbol_parallel(symbol, semaphore) for symbol in self.symbols]

        logger.info(f"⚡ Starting PARALLEL scan of {total_symbols} symbols (max 10 concurrent)...")
        scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)

        # Process results
        for result in scan_results:
            if isinstance(result, Exception):
                logger.error(f"Scan error: {result}")
                continue

            if result is None:
                continue  # Skipped symbol

            # Unpack result
            symbol, individual_analyses, market_data = result

            # Track market breadth and collect analyses
            for analysis in individual_analyses:
                action = analysis.get('action', 'hold')
                if action == 'buy':
                    bullish_count += 1
                elif action == 'sell':
                    bearish_count += 1
                else:
                    neutral_count += 1

                # Skip HOLD signals
                if action == 'hold':
                    logger.debug(
                        f"{symbol} ({analysis.get('model_name', 'AI')}): HOLD "
                        f"(confidence: {analysis.get('confidence', 0):.1%})"
                    )
                    continue

                # Calculate comprehensive opportunity score (akan önce market sentiment belirlenmeli)
                # Market sentiment henüz belirlenmedi, bu yüzden None geçiyoruz, sonra güncelleyeceğiz
                opportunity_score = 0  # Placeholder, will calculate after market sentiment

                # Calculate CONFLUENCE SCORE (multi-factor agreement)
                side = analysis.get('side', 'LONG')
                confluence = calculate_confluence_score(market_data, side)

                # Add to collection with metadata
                all_analyses.append({
                    'symbol': symbol,
                    'analysis': analysis,
                    'market_data': market_data,
                    'model': analysis.get('model_name', 'unknown'),
                    'confidence': analysis.get('confidence', 0),
                    'opportunity_score': opportunity_score,  # Will be updated later
                    'confluence': confluence,
                    'action': analysis.get('action', 'hold'),
                    'side': side
                })

                logger.info(
                    f"📊 {symbol} ({analysis.get('model_name', 'AI')}): "
                    f"{analysis.get('action', 'unknown').upper()} - "
                    f"Confidence: {analysis.get('confidence', 0):.1%} | "
                    f"Confluence: {confluence.get('score', 0):.0f}% ({confluence.get('confluence_count', 0)} factors)"
                )

        # Calculate MARKET BREADTH (market sentiment)
        total_scanned = bullish_count + bearish_count + neutral_count
        if total_scanned > 0:
            bullish_pct = (bullish_count / total_scanned) * 100
            bearish_pct = (bearish_count / total_scanned) * 100
            neutral_pct = (neutral_count / total_scanned) * 100
        else:
            bullish_pct = bearish_pct = neutral_pct = 0

        # Determine market sentiment
        if bullish_pct > 60:
            market_sentiment = "STRONG BULLISH"
        elif bullish_pct > 40:
            market_sentiment = "BULLISH"
        elif bearish_pct > 60:
            market_sentiment = "STRONG BEARISH"
        elif bearish_pct > 40:
            market_sentiment = "BEARISH"
        else:
            market_sentiment = "NEUTRAL/MIXED"

        logger.info(f"📊 MARKET BREADTH: {bullish_pct:.0f}% bullish, {bearish_pct:.0f}% bearish, {neutral_pct:.0f}% neutral")
        logger.info(f"🎯 MARKET SENTIMENT: {market_sentiment}")

        # 🎯 NOW calculate opportunity scores with market sentiment factor
        for opp in all_analyses:
            opp['opportunity_score'] = self.calculate_opportunity_score(
                opp['analysis'],
                opp['market_data'],
                market_sentiment,
                opp['side']
            )
            logger.debug(f"  {opp['symbol']}: Score = {opp['opportunity_score']:.1f}/100")

        # Sort ALL analyses by opportunity score (multi-factor) instead of just confidence
        # This considers: AI confidence, market regime, risk/reward, technical alignment
        all_analyses.sort(key=lambda x: x['opportunity_score'], reverse=True)

        logger.info(f"📈 Total analyses collected: {len(all_analyses)}")

        # Send scan summary with market breadth
        await telegram_bot.send_message(
            f"📊 <b>Market Taraması Tamamlandı!</b>\n\n"
            f"✅ Taranan coin: {total_symbols}\n"
            f"📈 Fırsat bulunan: {len(all_analyses)}\n\n"
            f"🎯 <b>Market Breadth:</b>\n"
            f"📈 Bullish: {bullish_pct:.0f}% ({bullish_count} coins)\n"
            f"📉 Bearish: {bearish_pct:.0f}% ({bearish_count} coins)\n"
            f"➖ Neutral: {neutral_pct:.0f}% ({neutral_count} coins)\n"
            f"💡 Sentiment: <b>{market_sentiment}</b>\n\n"
            f"En iyi fırsat analiz ediliyor...",
            parse_mode="HTML"
        )

        if all_analyses:
            # Get the BEST analysis (highest confidence)
            best = all_analyses[0]

            logger.info(
                f"🎯 BEST OPPORTUNITY: {best['symbol']} "
                f"({best['model'].upper()}) - "
                f"{best['analysis']['action'].upper()} - "
                f"Confidence: {best['confidence']:.1%} | "
                f"Score: {best['opportunity_score']:.1f}/100"
            )

            # Check both confidence AND opportunity score thresholds
            min_score = 65.0  # Minimum 65/100 score required
            if best['confidence'] >= float(self.settings.min_ai_confidence) and best['opportunity_score'] >= min_score:
                await notifier.send_scan_result(
                    best['symbol'],
                    best['confidence'],
                    best['analysis']['action']
                )

                # 🤖 FULLY AUTONOMOUS MODE: Execute trade automatically
                logger.info(f"🚀 AUTONOMOUS EXECUTION: Opening position automatically...")
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

    async def _scan_symbol_parallel(self, symbol: str, semaphore: asyncio.Semaphore):
        """
        Scan a single symbol in parallel with rate limiting.

        Returns:
            Tuple of (symbol, individual_analyses, market_data) or None if skipped
        """
        async with semaphore:  # Rate limiting
            try:
                logger.info(f"📨 Scanning {symbol}...")

                # Get market data
                market_data = await self.gather_market_data(symbol)

                if not market_data:
                    logger.warning(f"⚠️ {symbol} - Could not fetch data, skipping")
                    return None

                # Get AI analyses
                ai_engine = get_ai_engine()
                individual_analyses = await ai_engine.get_individual_analyses(symbol, market_data)

                logger.debug(f"✅ {symbol} - Scan complete")
                return (symbol, individual_analyses, market_data)

            except Exception as e:
                logger.error(f"❌ Error scanning {symbol}: {e}")
                return None

    async def gather_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Gather comprehensive market data for AI analysis.

        Returns:
            Market data dict with price, indicators, regime, etc.
        """
        try:
            exchange = await get_exchange_client()

            # OHLCV data (multiple timeframes) - ENHANCED with 5m
            ohlcv_5m = await exchange.fetch_ohlcv(symbol, '5m', limit=100)
            ohlcv_15m = await exchange.fetch_ohlcv(symbol, '15m', limit=100)
            ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=100)
            ohlcv_4h = await exchange.fetch_ohlcv(symbol, '4h', limit=50)

            # Current ticker
            ticker = await exchange.fetch_ticker(symbol)

            # Funding rate
            funding_rate = await exchange.fetch_funding_rate(symbol)

            # MULTI-TIMEFRAME CONFLUENCE ANALYSIS (Professional Edge!)
            mtf_analysis = calculate_multi_timeframe_indicators(
                ohlcv_5m, ohlcv_15m, ohlcv_1h, ohlcv_4h
            )

            # Technical indicators (legacy - kept for compatibility)
            indicators_15m = calculate_indicators(ohlcv_15m)
            indicators_1h = calculate_indicators(ohlcv_1h)
            indicators_4h = calculate_indicators(ohlcv_4h)

            # Market regime detection
            regime = detect_market_regime(ohlcv_1h)

            current_price = ticker['last']

            # ADVANCED INDICATORS (Institutional-Grade Analysis)
            # 1. Support/Resistance Levels (Key liquidity zones)
            support_resistance = detect_support_resistance_levels(ohlcv_1h, current_price)

            # 2. Volume Profile (POC and value areas)
            volume_profile = calculate_volume_profile(ohlcv_1h)

            # 3. Fibonacci Levels (Retracement and extension targets)
            fibonacci = calculate_fibonacci_levels(ohlcv_4h, current_price)

            # 4. Funding Rate Trend (Overleveraged position detection)
            # Fetch historical funding rates (last 8 hours)
            funding_history = []
            try:
                funding_history_data = await exchange.fetch_funding_rate_history(symbol, limit=8)
                funding_history = [f['fundingRate'] for f in funding_history_data if f.get('fundingRate') is not None]
            except:
                pass

            if not funding_history:
                funding_history = [funding_rate] if funding_rate else [0.0]

            funding_analysis = analyze_funding_rate_trend(funding_history)

            # 5. TIER 1 CRITICAL FEATURES (Professional Edge)

            # Divergence Detection (strongest reversal signals)
            divergence = detect_divergences(ohlcv_1h)

            # Order Flow Analysis (big money positioning)
            order_book = None
            recent_trades = []
            try:
                order_book = await exchange.fetch_order_book(symbol, limit=20)
                recent_trades = await exchange.fetch_trades(symbol, limit=50)
            except:
                pass

            order_flow = analyze_order_flow(order_book, recent_trades)

            # Smart Money Concepts (institutional edge)
            smart_money = detect_smart_money_concepts(ohlcv_4h, current_price)

            # Volatility Bands (adaptive risk management)
            volatility = calculate_volatility_bands(ohlcv_1h, current_price)

            # PHASE 2 CRITICAL FEATURES (ULTRA PROFESSIONAL)

            # Momentum Strength (rate of change analysis)
            momentum = calculate_momentum_strength(ohlcv_15m)

            # BTC Correlation Analysis (if not BTC itself)
            btc_correlation = {'correlation': 0.0, 'correlation_strength': 'n/a', 'independent_move': True, 'recommendation': 'N/A (BTC itself)'}
            if symbol != 'BTC/USDT:USDT':
                try:
                    # Fetch BTC data for correlation
                    btc_ohlcv = await exchange.fetch_ohlcv('BTC/USDT:USDT', '15m', limit=100)
                    btc_correlation = calculate_btc_correlation(symbol, ohlcv_15m, btc_ohlcv)
                except:
                    pass

            # PHASE 3 ULTRA PROFESSIONAL FEATURES (OPEN INTEREST & LIQUIDATIONS)

            # Open Interest Analysis (Trend Strength Confirmation)
            open_interest_data = {'trend_strength': 'unknown', 'signal': 'neutral', 'confidence_boost': 0.0, 'trading_implication': 'OI data not available'}
            try:
                # Fetch Open Interest history (last 24 hours, 1h intervals)
                oi_history = await exchange.fetch_open_interest_history(symbol, '1h', limit=24)

                if oi_history and len(oi_history) >= 3:
                    # Extract price history for the same period
                    price_history = [float(candle[4]) for candle in ohlcv_1h[-len(oi_history):]]

                    open_interest_data = analyze_open_interest(oi_history, price_history, current_price)
                    logger.debug(f"{symbol} OI Analysis: {open_interest_data.get('trend_strength', 'unknown')} - {open_interest_data.get('trading_implication', '')}")
            except Exception as e:
                logger.debug(f"{symbol} - Could not fetch Open Interest: {e}")

            # Liquidation Heatmap (Liquidity Magnet Detection)
            liquidation_map = estimate_liquidation_levels(current_price, ohlcv_1h, typical_leverage=10)

            return {
                'symbol': symbol,
                'current_price': current_price,
                'volume_24h': ticker.get('quoteVolume', 0),
                'ohlcv': {
                    '5m': ohlcv_5m[-20:],    # Last 20 candles
                    '15m': ohlcv_15m[-20:],
                    '1h': ohlcv_1h[-20:],
                    '4h': ohlcv_4h[-20:]
                },
                'indicators': {
                    '15m': indicators_15m,
                    '1h': indicators_1h,
                    '4h': indicators_4h
                },
                'funding_rate': funding_rate,
                'market_regime': regime,
                # MULTI-TIMEFRAME CONFLUENCE (ULTRA PROFESSIONAL!)
                'multi_timeframe': mtf_analysis,
                # ADVANCED INSTITUTIONAL INDICATORS
                'support_resistance': support_resistance,
                'volume_profile': volume_profile,
                'fibonacci': fibonacci,
                'funding_analysis': funding_analysis,
                # TIER 1 CRITICAL PROFESSIONAL FEATURES
                'divergence': divergence,
                'order_flow': order_flow,
                'smart_money': smart_money,
                'volatility': volatility,
                # PHASE 2 ULTRA PROFESSIONAL FEATURES
                'momentum': momentum,
                'btc_correlation': btc_correlation,
                # PHASE 3 ULTRA PROFESSIONAL FEATURES (OI & LIQUIDATIONS)
                'open_interest': open_interest_data,
                'liquidation_map': liquidation_map
            }

        except Exception as e:
            logger.error(f"Error gathering market data for {symbol}: {e}")
            return None

    def calculate_opportunity_score(
        self,
        ai_analysis: Dict[str, Any],
        market_data: Dict[str, Any],
        market_sentiment: str = "NEUTRAL/MIXED",
        trade_side: str = "LONG"
    ) -> float:
        """
        Score opportunity from 0-100 based on multiple factors.

        Improved scoring with weighted factors:
        - AI Confidence: 30 points (most important)
        - Market Regime Appropriateness: 20 points (critical for leverage)
        - Market Breadth Alignment: 15 points (NEW! Market sentiment factor)
        - Risk/Reward Ratio: 15 points (must be favorable)
        - Technical Alignment: 15 points (confirm the signal)
        - Volume Confirmation: 5 points (ensure liquidity)
        """
        score = 0.0

        # FACTOR 1: AI Confidence (30 points - reduced from 35)
        confidence = ai_analysis.get('confidence', 0)
        score += confidence * 30

        # FACTOR 2: Market Breadth Alignment (15 points - NEW!)
        # If market sentiment aligns with trade direction, boost score
        if market_sentiment == "STRONG BULLISH" and trade_side == "LONG":
            score += 15  # Perfect alignment
        elif market_sentiment == "BULLISH" and trade_side == "LONG":
            score += 12  # Good alignment
        elif market_sentiment == "STRONG BEARISH" and trade_side == "SHORT":
            score += 15  # Perfect alignment
        elif market_sentiment == "BEARISH" and trade_side == "SHORT":
            score += 12  # Good alignment
        elif market_sentiment == "NEUTRAL/MIXED":
            score += 7   # Neutral - no clear direction
        else:
            # Trading against market breadth (risky!)
            score += 0   # No points for counter-trend trades
            logger.warning(f"⚠️ Trading AGAINST market breadth: {market_sentiment} vs {trade_side}")

        # FACTOR 3: Market Regime Appropriateness (20 points - reduced from 25)
        regime = market_data.get('market_regime', 'UNKNOWN')
        action = ai_analysis.get('action', 'hold')
        side = ai_analysis.get('side')

        if regime == 'TRENDING' and action in ['buy', 'sell']:
            score += 20  # Perfect for leverage trading
        elif regime == 'RANGING' and action in ['buy', 'sell']:
            score += 10  # Moderate - mean reversion possible
        elif regime == 'VOLATILE':
            score += 4   # Risky - high slippage and whipsaw risk
        else:
            score += 0   # Unknown regime - no points

        # FACTOR 4: Risk/Reward Ratio (15 points - reduced from 20)
        rr_ratio = ai_analysis.get('risk_reward_ratio', 0)
        if rr_ratio >= 3.0:
            score += 15  # Excellent R:R
        elif rr_ratio >= 2.0:
            score += 12  # Good R:R
        elif rr_ratio >= 1.5:
            score += 8   # Acceptable R:R
        else:
            score += 0   # Poor R:R - no points

        # FACTOR 5: Technical Alignment (15 points)
        # Check if multiple timeframes agree with the signal
        indicators = market_data.get('indicators', {})
        timeframe_agreement = 0

        for tf in ['15m', '1h', '4h']:
            tf_indicators = indicators.get(tf, {})
            if is_bullish(tf_indicators) and action == 'buy':
                timeframe_agreement += 1
            elif is_bearish(tf_indicators) and action == 'sell':
                timeframe_agreement += 1

        # Award points based on alignment
        if timeframe_agreement == 3:
            score += 15  # All timeframes agree
        elif timeframe_agreement == 2:
            score += 10  # 2 out of 3 agree
        elif timeframe_agreement == 1:
            score += 5   # Only 1 timeframe agrees

        # FACTOR 6: Volume Confirmation (5 points)
        # High volume supports the trade thesis
        volume_trend = market_data.get('indicators', {}).get('15m', {}).get('volume_trend', 'normal')
        if volume_trend == 'high':
            score += 5  # Strong volume confirmation
        else:
            score += 2  # Normal volume

        # Cap at 100
        return min(score, 100)

    async def send_opportunity_for_selection(self, opportunity: Dict[str, Any]) -> None:
        """
        Send the best opportunity to telegram bot with multi-leverage options.

        Args:
            opportunity: Best opportunity dict with symbol, analysis, market_data, model
        """
        symbol = opportunity['symbol']
        analysis = opportunity['analysis']
        market_data = opportunity['market_data']
        model_name = opportunity.get('model', 'AI')

        logger.info(f"📱 Sending multi-leverage opportunity to Telegram: {symbol} {analysis['side']}")

        # Get current capital from database
        db = await get_db_client()
        config = await db.get_trading_config()
        capital = float(config.get('current_capital', 100))

        # Get telegram bot
        telegram_bot = await get_telegram_bot()

        # Send multi-leverage opportunity
        await telegram_bot.send_multi_leverage_opportunity(
            symbol=symbol,
            side=analysis['side'],
            current_price=float(market_data['current_price']),
            ai_confidence=analysis.get('confidence', 0),
            ai_models=[model_name],  # List of AI models that agreed
            capital=capital,
            analysis=analysis,
            market_data=market_data
        )

        logger.info(f"✅ Multi-leverage opportunity sent to Telegram. Waiting for user selection...")

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
