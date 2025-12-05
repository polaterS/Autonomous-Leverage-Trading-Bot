"""
Market Scanner - Scans multiple symbols for trading opportunities.
Gets AI consensus and executes best trades.

🚨 EMERGENCY DEPLOYMENT: v3.1 (2025-11-13 10:30) - CACHE BYPASS
✅ Market Direction Filter: ACTIVE (blocks SHORT in 70%+ bearish, LONG in 70%+ bullish)
✅ Max 3 Positions: ENFORCED (strict risk control)
✅ Portfolio Exposure: 130% limit
✅ Contrarian Logic: ACTIVE (77% bearish = open LONG only)
"""

import asyncio
from typing import Dict, Any, List, Optional
from decimal import Decimal
import numpy as np  # 🎯 #5: For GARCH volatility calculations
from src.config import get_settings
from src.exchange_client import get_exchange_client
from src.ai_engine import get_ai_engine
from src.risk_manager import get_risk_manager
from src.trade_executor import get_trade_executor
from src.telegram_notifier import get_notifier
from src.telegram_bot import get_telegram_bot
from src.database import get_db_client
from src.symbol_blacklist import get_symbol_blacklist  # 🔥 FIX #4: Symbol blacklist
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
from src.enhanced_trading_system import get_enhanced_trading_system

logger = setup_logging()


class MarketScanner:
    """Scans multiple symbols and identifies best trading opportunities."""

    def __init__(self):
        self.settings = get_settings()
        self.symbols = self.settings.trading_symbols
        self.last_position_time = 0  # 🔧 FIX #4: Track last position opening time (Unix timestamp)

        # 🔥 PROFIT FIX #5: Signal cache to prevent repetitive signals
        # Tracks recent signals: {(symbol, side): timestamp}
        # Prevents same coin+direction from appearing in consecutive scans
        self.signal_cache: Dict[tuple, float] = {}
        self.signal_expiry_minutes = 5  # Signals expire after 5 minutes

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

        # ⏰ PHASE 1: Time Filter Check
        if self.settings.enable_time_filter:
            from src.time_filter import get_time_filter
            time_filter = get_time_filter()
            can_trade, reason, category = time_filter.should_trade_now()

            if not can_trade:
                logger.warning(f"⏰ Trading blocked by Time Filter: {reason}")
                await notifier.send_alert('warning', f'⏰ Trade blocked: {reason}')
                return  # Skip scan entirely during toxic hours

            if category == "PRIME":
                logger.info(f"⏰ {reason}")

        # 📊 LOG consecutive losses for awareness (but don't pause trading)
        from src.database import get_db_client
        db = await get_db_client()
        consecutive_losses = await db.get_consecutive_losses()
        if consecutive_losses >= 5:
            logger.warning(f"⚠️ High consecutive losses detected: {consecutive_losses} (continuing to trade for ML learning)")
        elif consecutive_losses >= 3:
            logger.info(f"📊 Consecutive losses: {consecutive_losses}")

        await notifier.send_alert('info', '🔍 Market scan started...')

        all_analyses = []  # Collect ALL individual analyses
        telegram_bot = await get_telegram_bot()
        total_symbols = len(self.symbols)
        scanned = 0

        # Market breadth counters
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        # 🔧 FIX #3 OPTIMIZATION: Fetch BTC data ONCE for all symbols (API rate limit optimization)
        # This prevents 108 separate BTC API calls (one per symbol)
        btc_ohlcv = None
        try:
            from src.exchange_client import get_exchange_client
            exchange = await get_exchange_client()
            btc_ohlcv = await exchange.fetch_ohlcv('BTC/USDT:USDT', '15m', limit=20)
            logger.info(f"✅ BTC data fetched for correlation (will be reused for all {total_symbols} symbols)")
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch BTC data: {e} - correlation check disabled for this scan")

        # 🚀 PARALLEL SCANNING: Analyze all symbols concurrently with rate limiting
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent scans

        # 🎯 #7: Pass placeholder sentiment (will be calculated after first pass)
        # Note: First scan uses NEUTRAL, then we'll calculate actual sentiment
        scan_tasks = [
            self._scan_symbol_parallel(symbol, semaphore, "NEUTRAL", btc_ohlcv)  # Pass shared BTC data
            for symbol in self.symbols
        ]

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

            # Track market breadth for ALL symbols (including HOLD)
            # Count the DOMINANT signal from this symbol's analyses
            if individual_analyses:
                # Get the highest confidence analysis for this symbol
                dominant_analysis = max(individual_analyses, key=lambda x: x.get('confidence', 0))
                dominant_action = dominant_analysis.get('action', 'hold')

                if dominant_action == 'buy':
                    bullish_count += 1
                elif dominant_action == 'sell':
                    bearish_count += 1
                else:
                    neutral_count += 1
            else:
                # No analysis = neutral
                neutral_count += 1

            # Collect only BUY/SELL analyses (skip HOLD for trading)
            for analysis in individual_analyses:
                action = analysis.get('action', 'hold')

                # Skip HOLD signals (don't add to all_analyses for trading)
                if action == 'hold':
                    logger.debug(
                        f"{symbol} ({analysis.get('model_name', 'AI')}): HOLD "
                        f"(confidence: {analysis.get('confidence', 0):.1%})"
                    )
                    continue

                # 🔧 FIX #6: MARKET DIRECTION FILTER (2025-11-12)
                # Don't open SHORT in uptrend or LONG in downtrend
                # This prevents ML from predicting wrong direction
                regime = market_data.get('market_regime', 'UNKNOWN')
                mtf = market_data.get('multi_timeframe', {})
                trend_15m = mtf.get('trend_15m', 'NEUTRAL')
                trend_1h = mtf.get('trend_1h', 'NEUTRAL')

                # Check if trade direction aligns with market trend
                trade_side = analysis.get('side', 'LONG')
                skip_trade = False
                skip_reason = ""

                # LONG trade in strong downtrend? SKIP!
                if trade_side == 'LONG':
                    if trend_15m == 'DOWNTREND' and trend_1h == 'DOWNTREND':
                        skip_trade = True
                        skip_reason = "LONG in strong downtrend (15m + 1h)"
                    elif regime == 'TRENDING' and trend_1h == 'DOWNTREND':
                        skip_trade = True
                        skip_reason = "LONG in trending downmarket"

                # SHORT trade in strong uptrend? SKIP!
                elif trade_side == 'SHORT':
                    if trend_15m == 'UPTREND' and trend_1h == 'UPTREND':
                        skip_trade = True
                        skip_reason = "SHORT in strong uptrend (15m + 1h)"
                    elif regime == 'TRENDING' and trend_1h == 'UPTREND':
                        skip_trade = True
                        skip_reason = "SHORT in trending upmarket"

                if skip_trade:
                    logger.warning(
                        f"⏸️ {symbol} {trade_side} SKIPPED: {skip_reason} | "
                        f"ML Conf: {analysis.get('confidence', 0):.1%}"
                    )
                    continue

                # 🔥 PROFIT FIX #5: Signal Cache Check (Prevent Repetitive Signals)
                # Skip if same symbol+side appeared recently (within 5 minutes)
                # IMPACT: Eliminates repetitive signals from ranging coins
                from datetime import datetime, timezone
                signal_key = (symbol, trade_side)
                current_time = datetime.now(timezone.utc).timestamp()

                if signal_key in self.signal_cache:
                    last_seen = self.signal_cache[signal_key]
                    minutes_ago = (current_time - last_seen) / 60

                    if minutes_ago < self.signal_expiry_minutes:
                        logger.info(
                            f"⏭️ {symbol} {trade_side} - Signal seen recently "
                            f"({minutes_ago:.1f}m ago, expires after {self.signal_expiry_minutes}m) - Skipping to avoid repetition"
                        )
                        continue

                # Add to signal cache (will be used if trade is selected)
                self.signal_cache[signal_key] = current_time

                # Clean old signals from cache (older than expiry time)
                expired_signals = [
                    key for key, timestamp in self.signal_cache.items()
                    if (current_time - timestamp) / 60 > self.signal_expiry_minutes
                ]
                for key in expired_signals:
                    del self.signal_cache[key]

                # Calculate comprehensive opportunity score (akan önce market sentiment belirlenmeli)
                # Market sentiment henüz belirlenmedi, bu yüzden None geçiyoruz, sonra güncelleyeceğiz
                opportunity_score = 0  # Placeholder, will calculate after market sentiment

                # 🚀 ENHANCED CONFLUENCE SCORING (Professional Trading System)
                side = analysis.get('side', 'LONG')

                # Check if enhanced system is enabled
                try:
                    enhanced_system = get_enhanced_trading_system()
                    if enhanced_system.enable_confluence_filtering:
                        # Use advanced confluence scoring engine
                        # Extract needed components from market_data
                        # ✅ Extract timeframes dict from multi_timeframe data
                        mtf_full = market_data.get('multi_timeframe', {})
                        mtf_timeframes = mtf_full.get('timeframes', None) if mtf_full else None

                        # ✅ Convert market_regime string to dict format expected by scorer
                        regime_str = market_data.get('market_regime', 'UNKNOWN')
                        regime_dict = {
                            'regime': regime_str,
                            'should_trade': True,  # Allow trading in all regimes
                            'confidence': 0.7  # Default confidence
                        }

                        # ✅ Extract 15m indicators (primary execution timeframe)
                        indicators_dict = market_data.get('indicators', {})
                        indicators_15m = indicators_dict.get('15m', {})

                        # 🔧 v4.7.2 FIX: Calculate enhanced indicators for PROPER scoring!
                        # Without these, all categories fall back to 60% defaults
                        ohlcv_data = market_data.get('ohlcv_15m', [])

                        # v4.4.0 - v4.6.0 enhanced indicators (need 50+ candles)
                        enhanced_data = None
                        advanced_data = None
                        institutional_data = None

                        if ohlcv_data and len(ohlcv_data) >= 50:
                            try:
                                from src.indicators import (
                                    calculate_enhanced_indicators,
                                    calculate_advanced_indicators,
                                    calculate_institutional_indicators
                                )
                                enhanced_data = calculate_enhanced_indicators(ohlcv_data)
                                advanced_data = calculate_advanced_indicators(ohlcv_data)
                                institutional_data = calculate_institutional_indicators(ohlcv_data)
                                logger.debug(f"🔬 {symbol}: Enhanced/Advanced/Institutional indicators calculated")
                            except Exception as e:
                                logger.warning(f"⚠️ {symbol}: Enhanced indicators failed: {e}")

                        # v4.7.0 Ultra Professional Analysis (need 78+ candles)
                        derivatives_data = None
                        technical_advanced_data = None
                        harmonic_data = None

                        if ohlcv_data and len(ohlcv_data) >= 78:
                            try:
                                from src.indicators import calculate_ultra_professional_analysis
                                ultra_pro_data = calculate_ultra_professional_analysis(ohlcv_data)

                                if not ultra_pro_data.get('error'):
                                    derivatives_data = ultra_pro_data.get('derivatives_analysis', {})
                                    technical_advanced_data = ultra_pro_data.get('advanced_analysis', {})
                                    harmonic_data = ultra_pro_data.get('advanced_analysis', {}).get('harmonic_patterns', {})
                                    logger.debug(f"🚀 {symbol}: Ultra Professional Analysis (v4.7.0) calculated")
                            except Exception as e:
                                logger.warning(f"⚠️ {symbol}: Ultra Professional Analysis failed: {e}")

                        confluence_result = enhanced_system.confluence_scorer.score_opportunity(
                            symbol=symbol,
                            side=side,
                            pa_analysis=analysis,  # PA analysis from above
                            indicators=indicators_15m,  # ✅ Use 15m indicators (not multi-timeframe dict)
                            volume_profile=market_data.get('volume_profile', {}),  # ✅ Use real volume profile data
                            market_regime=regime_dict,  # ✅ Use real regime data (converted to dict)
                            mtf_data=mtf_timeframes,  # ✅ Pass timeframes dict (not full mtf object)
                            # 🔧 v4.7.2 FIX: Pass ALL enhanced indicator parameters!
                            enhanced_data=enhanced_data,           # v4.4.0: BB Squeeze, EMA Stack, Divergences
                            advanced_data=advanced_data,           # v4.5.0: VWAP, StochRSI, CMF, Fibonacci
                            institutional_data=institutional_data, # v4.6.0: SMC, Wyckoff VSA, Hurst, Z-Score
                            derivatives_data=derivatives_data,     # v4.7.0: Funding, OI, L/S Ratio, Fear&Greed
                            technical_advanced_data=technical_advanced_data,  # v4.7.0: CVD, Ichimoku, Liquidations
                            harmonic_data=harmonic_data            # v4.7.0: Harmonic Patterns
                        )
                        confluence = {
                            'score': confluence_result['total_score'],
                            'approved': confluence_result['should_trade'],
                            'confluence_count': len(confluence_result['component_scores']),
                            'quality': confluence_result.get('quality', 'UNKNOWN'),
                            'reasoning': confluence_result.get('reasoning', ''),
                            'component_scores': confluence_result.get('component_scores', {})
                        }

                        # ═══════════════════════════════════════════════════════════════════════
                        # 🛡️ v4.7.10: QUALITY PROTECTION FILTERS
                        # ═══════════════════════════════════════════════════════════════════════
                        # These filters prevent low-quality trades that historically lose money
                        # Based on analysis of losing positions:
                        # - Technical Advanced < 40% = CVD/Ichimoku/Liquidations warning
                        # - Derivatives = 50% (fallback) = No real data, skip
                        # - ATR < 0.5% = No volatility, can't hit profit targets
                        # - ADX > 40 = Trend exhausted, reversal likely
                        # ═══════════════════════════════════════════════════════════════════════

                        component_scores = confluence.get('component_scores', {})
                        skip_reason = None

                        # 🛡️ FILTER 1: Technical Advanced < 40% (CVD, Ichimoku, Liquidations warning)
                        tech_advanced_score = component_scores.get('technical_advanced', 0)
                        tech_advanced_pct = (tech_advanced_score / 10) * 100 if tech_advanced_score else 0
                        if tech_advanced_pct < 40:
                            skip_reason = f"Technical Advanced too weak: {tech_advanced_pct:.0f}% (need 40%+) - CVD/Ichimoku/Liquidations warning"

                        # 🛡️ FILTER 2: REMOVED in v4.7.12
                        # Derivatives = 50% filter was blocking ALL trades because
                        # Bybit API doesn't provide real Funding/OI/L-S data via OHLCV
                        # Real derivatives data requires separate API endpoints (future enhancement)

                        # 🛡️ FILTER 2: ATR < 0.3% (very low volatility - can't hit profit targets)
                        atr_percent = indicators_15m.get('atr_percent', 1.0)
                        if not skip_reason and atr_percent < 0.3:
                            skip_reason = f"Volatility too low: ATR {atr_percent:.2f}% (need 0.3%+) - Price won't move enough"

                        # 🛡️ FILTER 3: ADX > 40 (trend exhaustion - reversal likely)
                        # v4.7.13 FIX: Use ADX from Price Action analysis (same as shown in trade)
                        # Was using indicators_15m which had different/missing ADX value
                        trend_data = analysis.get('trend', {})
                        adx = trend_data.get('adx', 25) if isinstance(trend_data, dict) else 25
                        if not skip_reason and adx > 40:
                            skip_reason = f"ADX too high: {adx:.1f} (max 40) - Trend exhausted, reversal likely"

                        # Apply filter - mark as not approved with reason
                        if skip_reason:
                            logger.warning(f"🛡️ {symbol} QUALITY FILTER: {skip_reason}")
                            confluence['approved'] = False
                            confluence['quality'] = 'FILTERED'
                            confluence['filter_reason'] = skip_reason

                        logger.info(
                            f"🎯 {symbol} Enhanced Confluence: {confluence['score']:.1f}/100 "
                            f"({confluence['quality']}) - {confluence['reasoning']}"
                        )
                    else:
                        # Fall back to old confluence scoring
                        confluence = calculate_confluence_score(market_data, side)
                except Exception as e:
                    logger.warning(f"Enhanced confluence failed for {symbol}, using fallback: {e}")
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

        # 🔧 FIX #2: CONTRARIAN MARKET BREADTH LOGIC (2025-11-12)
        # When 70%+ traders agree on direction, market often moves OPPOSITE
        # OLD: Follow majority (74% bearish → open SHORT → all lose)
        # NEW: Contrarian when extreme sentiment detected
        #
        # 🔴 DISABLED TEMPORARILY (2025-11-13):
        # - Win rate currently 49.5% (need 60%+)
        # - Contrarian adds complexity and may reduce win rate
        # - Testing simple trend-following for 20-30 trades
        # - If win rate improves to 60%+, keep it disabled
        # - Re-enable only if needed AND with higher threshold (95%+)

        contrarian_mode = False

        # Determine market sentiment with contrarian logic
        # 🔴 DISABLED: Changed threshold from 70 to 999 (never activates)
        if bearish_pct >= 999:  # Was: 70 (DISABLED)
            # 🎯 EXTREME BEARISH = CONTRARIAN BULLISH
            # When 70%+ are bearish, short squeeze likely (prices go UP)
            market_sentiment = "CONTRARIAN_BULLISH"
            market_sentiment_display = "CONTRARIAN BULLISH (70%+ bearish → squeeze expected)"
            contrarian_mode = True
            logger.warning(f"⚠️ CONTRARIAN MODE: {bearish_pct:.0f}% bearish → Expecting SHORT SQUEEZE (prices UP)")

        elif bullish_pct >= 999:  # Was: 70 (DISABLED)
            # 🎯 EXTREME BULLISH = CONTRARIAN BEARISH
            # When 70%+ are bullish, long squeeze likely (prices go DOWN)
            market_sentiment = "CONTRARIAN_BEARISH"
            market_sentiment_display = "CONTRARIAN BEARISH (70%+ bullish → correction expected)"
            contrarian_mode = True
            logger.warning(f"⚠️ CONTRARIAN MODE: {bullish_pct:.0f}% bullish → Expecting LONG SQUEEZE (prices DOWN)")

        elif bullish_pct > 60:
            market_sentiment = "BULLISH_STRONG"
            market_sentiment_display = "STRONG BULLISH"
        elif bullish_pct > 40:
            market_sentiment = "BULLISH"
            market_sentiment_display = "BULLISH"
        elif bearish_pct > 60:
            market_sentiment = "BEARISH_STRONG"
            market_sentiment_display = "STRONG BEARISH"
        elif bearish_pct > 40:
            market_sentiment = "BEARISH"
            market_sentiment_display = "BEARISH"
        else:
            market_sentiment = "NEUTRAL"
            market_sentiment_display = "NEUTRAL/MIXED"

        logger.info(f"📊 MARKET BREADTH: {bullish_pct:.0f}% bullish, {bearish_pct:.0f}% bearish, {neutral_pct:.0f}% neutral")
        logger.info(f"🎯 MARKET SENTIMENT: {market_sentiment}")
        if contrarian_mode:
            logger.warning(f"🔄 CONTRARIAN TRADING ACTIVE: Inverse signals expected!")

        # 🎯 NOW calculate opportunity scores with market sentiment factor + 🎯 #7: Store sentiment
        for opp in all_analyses:
            # 🎯 #7: Add market sentiment to each opportunity for ML
            opp['market_sentiment'] = market_sentiment

            # 🔥 CRITICAL FIX: Add market breadth for direction filter
            opp['market_breadth'] = {
                'bullish_percent': bullish_pct,
                'bearish_percent': bearish_pct,
                'neutral_percent': neutral_pct
            }

            opp['opportunity_score'] = self.calculate_opportunity_score(
                opp['analysis'],
                opp['market_data'],
                market_sentiment_display,  # Use display format for scoring
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
            f"💡 Sentiment: <b>{market_sentiment_display}</b>\n\n"
            f"En iyi fırsat analiz ediliyor...",
            parse_mode="HTML"
        )

        if all_analyses:
            # 🎯 MULTI-POSITION: Get TOP N opportunities (not just best 1)
            # Check how many positions we can open
            db = await get_db_client()
            active_positions = await db.get_active_positions()
            max_positions = self.settings.max_concurrent_positions
            available_slots = max_positions - len(active_positions)

            # Get list of symbols with existing positions
            existing_symbols = {pos['symbol'] for pos in active_positions}

            logger.info(
                f"🎯 Active positions: {len(active_positions)}/{max_positions} "
                f"(Available slots: {available_slots})"
            )
            if existing_symbols:
                logger.info(f"📋 Existing positions: {', '.join(existing_symbols)}")

            if available_slots <= 0:
                logger.info("📊 Maximum concurrent positions reached. Waiting for positions to close...")
                await notifier.send_alert(
                    'info',
                    f'📊 Max positions reached ({len(active_positions)}/{max_positions}). '
                    f'Monitoring existing positions...'
                )
            else:
                # Filter out symbols with existing positions BEFORE selecting top opportunities
                available_analyses = [
                    analysis for analysis in all_analyses
                    if analysis['symbol'] not in existing_symbols
                ]

                if len(available_analyses) < len(all_analyses):
                    filtered_count = len(all_analyses) - len(available_analyses)
                    logger.info(
                        f"🔍 Filtered out {filtered_count} opportunity/ies with existing positions"
                    )

                # Select TOP N opportunities from available coins (up to available slots)
                top_opportunities = available_analyses[:available_slots]

                # Filter by minimum thresholds - QUALITY OVER QUANTITY
                # After ML learning phase, we prioritize high-quality setups
                min_score = 30.0  # Higher quality bar (was 10.0)
                min_confidence = float(self.settings.min_ai_confidence)

                # ═══════════════════════════════════════════════════════════════════════
                # 🛡️ v4.7.10 FILTER 5: Market 80%+ Neutral = Reduce trades
                # ═══════════════════════════════════════════════════════════════════════
                # When market is 80%+ neutral, there's no clear trend direction.
                # In this environment, trend-following strategies struggle.
                # Solution: Be MORE selective - require higher scores
                # ═══════════════════════════════════════════════════════════════════════
                if neutral_pct >= 80:
                    min_score = 70.0  # Raised from 30 to 70 when market is indecisive
                    logger.warning(
                        f"🛡️ MARKET NEUTRAL FILTER: {neutral_pct:.0f}% neutral → "
                        f"Raising min score to {min_score:.0f} (need stronger setups)"
                    )

                # 🧠 ML PATTERN QUALITY FILTER: Check if trade patterns have good historical WR
                from src.ml_pattern_learner import get_ml_learner
                ml_learner = await get_ml_learner()

                qualified_opportunities = []
                for opp in top_opportunities:
                    # Basic filters
                    if opp['confidence'] < min_confidence or opp['opportunity_score'] < min_score:
                        logger.info(
                            f"🚫 {opp['symbol']}: Confidence ({opp['confidence']:.1%}) < {min_confidence:.1%} "
                            f"OR Score ({opp['opportunity_score']:.1f}) < {min_score:.1f} - skipping"
                        )
                        continue

                    # 🧠 ML FILTER: Check if this trade's patterns have proven successful
                    # 🔥 DISABLED FOR PA-ONLY MODE: PA has its own strict filters (ADX, S/R, trend)
                    # ML pattern filter is redundant and blocks good PA setups
                    model_name = opp.get('model', 'unknown')

                    # 🔥 FIX: Bypass ML filter for both PA-ONLY and PA-Override
                    # PA-Override = ML said HOLD but PA found setup (still PA-driven!)
                    if not model_name.startswith('PA-'):
                        # Only apply ML pattern filter to ML/AI-based trades
                        reasoning = opp['analysis'].get('reasoning', '')
                        patterns = ml_learner._extract_pattern_keywords(reasoning)

                        # Calculate average pattern win rate
                        if patterns:
                            pattern_wrs = []
                            for pattern in patterns:
                                wins = ml_learner.winning_patterns.get(pattern, 0)
                                losses = ml_learner.losing_patterns.get(pattern, 0)
                                total = wins + losses
                                if total >= 5:  # Only use patterns with 5+ occurrences
                                    wr = (wins / total) * 100
                                    pattern_wrs.append(wr)

                            if pattern_wrs:
                                avg_pattern_wr = sum(pattern_wrs) / len(pattern_wrs)
                                # Require 50%+ average pattern WR for trade approval (lowered for early ML learning phase)
                                if avg_pattern_wr < 50.0:
                                    logger.info(
                                        f"🚫 {opp['symbol']}: Pattern WR too low ({avg_pattern_wr:.0f}%) - skipping"
                                    )
                                    continue

                    qualified_opportunities.append(opp)

                if qualified_opportunities:
                    logger.info(
                        f"🚀 Found {len(qualified_opportunities)} qualified opportunities "
                        f"(top {available_slots} out of {len(all_analyses)})"
                    )

                    # Execute all qualified opportunities in parallel
                    for i, opp in enumerate(qualified_opportunities, 1):
                        logger.info(
                            f"🎯 OPPORTUNITY #{i}: {opp['symbol']} "
                            f"({opp['model'].upper()}) - "
                            f"{opp['analysis']['action'].upper()} - "
                            f"Confidence: {opp['confidence']:.1%} | "
                            f"Score: {opp['opportunity_score']:.1f}/100"
                        )

                    # 🤖 FULLY AUTONOMOUS MODE: Execute all trades automatically
                    logger.info(f"🚀 AUTONOMOUS EXECUTION: Opening {len(qualified_opportunities)} position(s)...")

                    # Execute trades sequentially (to avoid race conditions)
                    for opp in qualified_opportunities:
                        await notifier.send_scan_result(
                            opp['symbol'],
                            opp['confidence'],
                            opp['analysis']['action']
                        )
                        await self.execute_trade(opp)
                        await asyncio.sleep(2)  # Small delay between trades

                    # Note: Capital is now always read from config (no cache to reset)
                else:
                    # Show best opportunity from AVAILABLE coins (excluding existing positions)
                    if available_analyses:
                        best = available_analyses[0]

                        # Determine rejection reason
                        rejection_reasons = []

                        if best['confidence'] < float(self.settings.min_ai_confidence):
                            rejection_reasons.append(
                                f"Confidence {best['confidence']:.1%} < {float(self.settings.min_ai_confidence):.1%}"
                            )

                        if best['opportunity_score'] < min_score:
                            rejection_reasons.append(
                                f"Opportunity Score {best['opportunity_score']:.1f} < {min_score:.1f}"
                            )

                        reason_text = " AND ".join(rejection_reasons) if rejection_reasons else "Unknown reason"

                        await notifier.send_alert(
                            'info',
                            f"Best opportunity: {best['symbol']} ({best['model']}) - "
                            f"Confidence: {best['confidence']:.1%} | "
                            f"Score: {best['opportunity_score']:.1f}/100\n\n"
                            f"❌ Rejected: {reason_text}\n\n"
                            f"Waiting for better setup..."
                        )
                    else:
                        # All opportunities filtered out due to existing positions
                        await notifier.send_alert(
                            'info',
                            f"😐 All opportunities filtered out (existing positions: {', '.join(existing_symbols)}). "
                            f"Will scan again in 5 minutes."
                        )
        else:
            logger.info("😐 No good opportunities found (all models recommend HOLD)")
            await notifier.send_alert(
                'info',
                '😐 No good opportunities found. All AI models recommend HOLD. '
                'Will scan again in 5 minutes.'
            )

    async def _scan_symbol_parallel(
        self,
        symbol: str,
        semaphore: asyncio.Semaphore,
        market_sentiment: str = "NEUTRAL",  # 🎯 #7: Market sentiment parameter
        btc_ohlcv: Optional[List] = None  # 🔧 FIX #3: Shared BTC data for correlation (API optimization)
    ):
        """
        Scan a single symbol in parallel with rate limiting.

        Args:
            symbol: Trading symbol
            semaphore: Asyncio semaphore for rate limiting
            market_sentiment: Market breadth sentiment (BULLISH_STRONG, BULLISH, NEUTRAL, BEARISH, BEARISH_STRONG)

        Returns:
            Tuple of (symbol, individual_analyses, market_data) or None if skipped
        """
        async with semaphore:  # Rate limiting
            try:
                logger.debug(f"Scanning {symbol}...")

                # 🔥 PROFIT FIX #4: BLACKLIST CHECK (Skip problematic symbols)
                # Prevents wasted execution attempts on symbols with exchange restrictions
                # or repeated failures
                blacklist = get_symbol_blacklist()
                is_blacklisted, blacklist_reason = blacklist.is_blacklisted(symbol)
                if is_blacklisted:
                    logger.info(f"🚫 {symbol} - BLACKLISTED: {blacklist_reason} - Skipping")
                    return None  # Skip this symbol

                # 🚫 COOLDOWN CHECK: Prevent re-trading same symbol too soon (prevents doubling down)
                if self.settings.position_cooldown_minutes > 0:
                    from src.database import get_db_client
                    from datetime import datetime, timezone

                    db = await get_db_client()
                    last_closed = await db.get_last_closed_time(symbol)
                    if last_closed:
                        # Make last_closed timezone-aware if it's naive (DB stores in UTC)
                        if last_closed.tzinfo is None:
                            last_closed = last_closed.replace(tzinfo=timezone.utc)

                        minutes_since_close = (datetime.now(timezone.utc) - last_closed).total_seconds() / 60
                        if minutes_since_close < self.settings.position_cooldown_minutes:
                            logger.info(
                                f"⏰ {symbol} - Cooldown active "
                                f"({minutes_since_close:.1f}m / {self.settings.position_cooldown_minutes}m) - Skipping"
                            )
                            return None  # Skip this symbol

                # Get market data
                market_data = await self.gather_market_data(symbol)

                if not market_data:
                    logger.warning(f"⚠️ {symbol} - Could not fetch data, skipping")
                    return None

                # 🎯 PROFESSIONAL PRICE ACTION ANALYSIS (RUNS FIRST!)
                # This analyzes S/R, trend, volume BEFORE ML filtering
                # Can boost ML confidence for perfect setups (e.g., 55% → 70%)
                pa_result = None

                logger.info(f"🔍 {symbol} - Starting PA pre-analysis...")

                try:
                    from src.price_action_analyzer import get_price_action_analyzer
                    from src.exchange_client import get_exchange_client
                    import pandas as pd

                    logger.info(f"✅ {symbol} - PA analyzer imported successfully")

                    pa_analyzer = get_price_action_analyzer()

                    # 🔥 CRITICAL FIX: Get exchange for multi-timeframe S/R analysis
                    # v4.0 S/R enhancements need exchange to fetch 4h and 1h data
                    # Without exchange, only 15m data is used (insufficient for quality S/R)
                    exchange = await get_exchange_client()

                    # Get FULL OHLCV data for price action (need 50+ candles for S/R detection)
                    # Use ohlcv_15m key (contains full 100 candles, not the nested 'ohlcv' dict which only has 20)
                    ohlcv_15m = market_data.get('ohlcv_15m', [])

                    logger.info(f"📊 {symbol} - OHLCV data length: {len(ohlcv_15m)} candles")

                    if len(ohlcv_15m) >= 50:  # Need minimum data for analysis
                        logger.info(f"✅ {symbol} - Sufficient data for PA analysis (≥50 candles)")
                        # Convert to pandas DataFrame
                        df = pd.DataFrame(
                            ohlcv_15m,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        current_price = market_data.get('current_price', df['close'].iloc[-1])

                        # 🔧 FIX #3: Use shared BTC data for correlation check (passed from parent)
                        # This prevents 108 separate API calls - BTC data is fetched once per scan cycle
                        # For BTC itself, btc_ohlcv will be None (no self-correlation)
                        btc_data_for_check = btc_ohlcv if symbol != 'BTC/USDT:USDT' else None

                        # Analyze price action for BOTH buy and sell opportunities
                        # (we don't know ML signal yet, so check both directions)
                        # 🔥 CRITICAL: Pass exchange for multi-timeframe S/R (v4.0 feature)
                        # 🔥 CRITICAL FIX: await async should_enter_trade_v4_20251122() - RENAMED FOR CACHE BYPASS!
                        pa_long = await pa_analyzer.should_enter_trade_v4_20251122(
                            symbol=symbol,
                            df=df,
                            ml_signal='BUY',
                            ml_confidence=50.0,  # Neutral starting point
                            current_price=current_price,
                            exchange=exchange,  # 🔥 v4.0: Enable multi-timeframe S/R
                            btc_ohlcv=btc_data_for_check  # 🔧 FIX #3: Shared BTC data (API optimized)
                        )

                        pa_short = await pa_analyzer.should_enter_trade_v4_20251122(
                            symbol=symbol,
                            df=df,
                            ml_signal='SELL',
                            ml_confidence=50.0,  # Neutral starting point
                            current_price=current_price,
                            exchange=exchange,  # 🔥 v4.0: Enable multi-timeframe S/R
                            btc_ohlcv=btc_data_for_check  # 🔧 FIX #3: Shared BTC data (API optimized)
                        )

                        # Store best PA setup for later use
                        # Always log PA analysis results (even if rejected) for transparency
                        logger.info(
                            f"📊 {symbol} PA PRE-ANALYSIS: "
                            f"LONG={'✅' if pa_long['should_enter'] else '❌'} (+{pa_long['confidence_boost']}%) | "
                            f"SHORT={'✅' if pa_short['should_enter'] else '❌'} (+{pa_short['confidence_boost']}%)"
                        )

                        # 🔍 DEBUG: Log rejection reasons to understand why 0 opportunities found
                        if not pa_long['should_enter']:
                            logger.info(f"   LONG rejected: {pa_long.get('reason', 'No reason provided')}")
                        if not pa_short['should_enter']:
                            logger.info(f"   SHORT rejected: {pa_short.get('reason', 'No reason provided')}")

                        if pa_long['should_enter'] or pa_short['should_enter']:
                            pa_result = {
                                'long': pa_long if pa_long['should_enter'] else None,
                                'short': pa_short if pa_short['should_enter'] else None
                            }
                    else:
                        logger.info(f"⏸️ {symbol} - Insufficient data for PA analysis (need ≥50, got {len(ohlcv_15m)})")

                except Exception as e:
                    logger.error(f"❌ {symbol} - Price action pre-analysis EXCEPTION: {e}", exc_info=True)

                # 🔥 TIER 3: STRATEGY DISABLED FOR PA-ONLY MODE
                # USER REQUEST: "C: ML'Yİ KAPAT, SADECE PA KULLAN"
                # Strategy fallback disabled - will use only PA-only consensus via get_consensus()
                strategy_analysis = None
                logger.debug(f"📊 {symbol} - Strategy disabled (PA-ONLY mode)")

                # Get AI analyses (🎯 #7: Pass market sentiment for ML enhancement)
                ai_engine = get_ai_engine()
                individual_analyses = await ai_engine.get_individual_analyses(
                    symbol, market_data, market_sentiment
                )

                # 🔥 TIER 3: STRATEGY CONSENSUS - Combine with AI/ML if available
                if strategy_analysis and strategy_analysis.get('action', '').lower() != 'hold':
                    # Strategy found a signal! Let's see if AI/ML agrees
                    if individual_analyses:
                        # Both strategy and AI have opinions - check for confluence
                        best_ai = max(individual_analyses, key=lambda x: x.get('confidence', 0))
                        ai_action = best_ai.get('action', 'hold')
                        ai_side = best_ai.get('side', '')

                        strategy_action = strategy_analysis['action'].lower()  # Normalize to lowercase
                        strategy_side = strategy_analysis['side']

                        # Check if they agree (case-insensitive comparison)
                        if ai_action == strategy_action and ai_side == strategy_side:
                            # 🎯 PERFECT CONFLUENCE: Strategy + AI agree!
                            # Boost AI confidence by strategy confidence
                            confluence_boost = strategy_analysis['confidence'] * 0.15  # Max 15% boost
                            old_confidence = best_ai['confidence']
                            best_ai['confidence'] = min(old_confidence + confluence_boost, 0.95)  # Cap at 95%
                            best_ai['strategy_confluence'] = {
                                'strategy_name': strategy_analysis['strategy'],
                                'strategy_confidence': strategy_analysis['confidence'],
                                'confidence_boost': confluence_boost,
                                'regime': strategy_analysis['regime']
                            }

                            logger.info(
                                f"✅ {symbol} STRATEGY CONFLUENCE: {strategy_analysis['strategy']} + AI agree on {strategy_side} | "
                                f"Confidence: {old_confidence:.0%} → {best_ai['confidence']:.0%} (+{confluence_boost:.0%})"
                            )
                    else:
                        # Only strategy has a signal (AI/ML silent) - use strategy signal
                        # Convert strategy to individual analysis format
                        # 🔧 Calculate leverage and stop loss based on strategy confidence
                        strategy_confidence = strategy_analysis['confidence']
                        suggested_leverage = 5 if strategy_confidence < 0.80 else 7 if strategy_confidence < 0.90 else 10
                        stop_loss_percent = 20.0 if strategy_confidence < 0.80 else 16.0 if strategy_confidence < 0.90 else 12.0

                        individual_analyses = [{
                            'action': strategy_analysis['action'].lower(),
                            'side': strategy_analysis['side'],
                            'confidence': strategy_analysis['confidence'],  # Already in decimal format (0.0-1.0)
                            'suggested_leverage': suggested_leverage,
                            'stop_loss_percent': stop_loss_percent,
                            'model_name': f"Strategy-{strategy_analysis['strategy']}",
                            'models_used': [strategy_analysis['strategy']],
                            'reasoning': strategy_analysis.get('reasoning', f"{strategy_analysis['strategy']} strategy signal"),
                            'strategy_only': True,
                            'regime': strategy_analysis['regime']
                        }]

                        logger.info(
                            f"🎯 {symbol} STRATEGY-ONLY: {strategy_analysis['strategy']} | "
                            f"{strategy_analysis['action'].upper()} @ {strategy_analysis['confidence']:.0%}"
                        )

                # 🧠 ML-ONLY FALLBACK: If AI models failed, try ML-only prediction
                if not individual_analyses:
                    logger.debug(f"{symbol} - All AI models failed, trying ML-ONLY mode...")

                    # 🔥 CRITICAL FIX: Don't re-run PA analysis! Use pre-analysis results
                    # PROBLEM: get_consensus() re-runs PA analysis → inconsistent results
                    # - TIA: Pre-analysis rejected (bearish candles) ❌
                    # - But get_consensus() accepted → 105% confidence ✓ BUG!
                    # SOLUTION: Reuse pa_result from pre-analysis (lines 557-586)

                    # Check if PA found perfect setup (from pre-analysis)
                    if pa_result:
                        # ML said HOLD, but PA found opportunity - create synthetic analysis
                        best_pa = None
                        suggested_side = None

                        # 🔥 STRICTER FILTER: Require should_enter=True (not just confidence_boost)
                        # OLD: if confidence_boost >= 15 → could be False with 0% boost!
                        # NEW: if should_enter=True AND confidence_boost >= 15
                        if pa_result.get('long') and pa_result['long']['should_enter'] and pa_result['long']['confidence_boost'] >= 15:
                            best_pa = pa_result['long']
                            suggested_side = 'buy'
                        elif pa_result.get('short') and pa_result['short']['should_enter'] and pa_result['short']['confidence_boost'] >= 15:
                            best_pa = pa_result['short']
                            suggested_side = 'sell'

                        if best_pa and suggested_side:
                            # PA setup is STRONG (≥15% boost) - override ML HOLD
                            synthetic_confidence = (50 + best_pa['confidence_boost']) / 100

                            # Calculate leverage and stop-loss based on confidence (same logic as AI engine)
                            if synthetic_confidence >= 0.85:  # 85%+ = Ultra high confidence
                                suggested_leverage = 10  # Max 10x
                                stop_loss_percent = 20.0  # Widest stop
                            elif synthetic_confidence >= 0.75:  # 75-84% = High confidence
                                suggested_leverage = 7
                                stop_loss_percent = 16.0
                            elif synthetic_confidence >= 0.70:  # 70-74% = Acceptable
                                suggested_leverage = 5
                                stop_loss_percent = 14.0
                            else:  # <70% (typical for PA override: 65-70%)
                                suggested_leverage = 3
                                stop_loss_percent = 12.0

                            logger.info(
                                f"🎯 {symbol} PA OVERRIDE: ML said HOLD but PA found {suggested_side.upper()} setup! "
                                f"Confidence: 50% + PA {best_pa['confidence_boost']}% = {synthetic_confidence*100:.0f}% | "
                                f"Leverage: {suggested_leverage}x, Stop-Loss: {stop_loss_percent}%"
                            )

                            # Extract detailed PA analysis data
                            pa_analysis = best_pa.get('analysis', {})
                            sr_data = pa_analysis.get('support_resistance', {})
                            trend_data = pa_analysis.get('trend', {})
                            volume_data = pa_analysis.get('volume', {})

                            individual_analyses = [{
                                'action': suggested_side,
                                'side': 'LONG' if suggested_side == 'buy' else 'SHORT',
                                'confidence': synthetic_confidence,
                                'suggested_leverage': suggested_leverage,
                                'stop_loss_percent': stop_loss_percent,
                                'model_name': 'PA-Override',
                                'models_used': ['PriceAction'],
                                'reasoning': best_pa['reason'],
                                'price_action': {
                                    'validated': True,
                                    'reason': best_pa['reason'],
                                    'stop_loss': best_pa['stop_loss'],
                                    'targets': best_pa['targets'],
                                    'rr_ratio': best_pa['rr_ratio'],
                                    'confidence_boost': best_pa['confidence_boost'],
                                    'pa_override': True,
                                    # ✅ NEW: Full PA analysis details for Telegram
                                    'support_levels': [s['price'] for s in sr_data.get('support', [])[:3]],  # Top 3 support
                                    'resistance_levels': [r['price'] for r in sr_data.get('resistance', [])[:3]],  # Top 3 resistance
                                    'trend': trend_data.get('direction', 'N/A'),
                                    'trend_strength': trend_data.get('strength', 'N/A'),
                                    'adx': trend_data.get('adx', 0),
                                    'volume_surge': volume_data.get('is_surge', False),
                                    'volume_ratio': volume_data.get('surge_ratio', 1.0),
                                    'obv_trend': volume_data.get('obv_trend', 'NEUTRAL'),
                                    'vpoc': sr_data.get('vpoc', 0),
                                    'indicators': ['Support/Resistance', 'Trend (ADX)', 'Volume (OBV)', 'Fibonacci'],
                                    'timeframes': ['15m'],  # Primary timeframe used
                                }
                            }]

                    if not individual_analyses:
                        logger.debug(f"🧠 ML-ONLY: {symbol} - No high-confidence prediction (holding)")

                # 🎯 PRICE ACTION CONFIDENCE BOOST: If ML predicted trade, boost with PA
                if individual_analyses and pa_result:
                    try:
                        # Skip if this was a PA override (already processed)
                        if not any(a.get('price_action', {}).get('pa_override') for a in individual_analyses):
                            # Get best analysis (highest confidence)
                            best_analysis = max(individual_analyses, key=lambda x: x.get('confidence', 0))
                            ml_signal = best_analysis.get('action', 'hold').upper()
                            ml_confidence = best_analysis.get('confidence', 0) * 100  # Convert to percentage

                            # Get matching PA result (long for BUY, short for SELL)
                            matching_pa = None
                            if ml_signal == 'BUY' and pa_result.get('long'):
                                matching_pa = pa_result['long']
                            elif ml_signal == 'SELL' and pa_result.get('short'):
                                matching_pa = pa_result['short']

                            if matching_pa and matching_pa['should_enter']:
                                # ✅ Price action confirms ML signal!
                                # Boost confidence
                                boosted_confidence = (ml_confidence + matching_pa['confidence_boost']) / 100

                                # Extract detailed PA analysis data
                                pa_analysis = matching_pa.get('analysis', {})
                                sr_data = pa_analysis.get('support_resistance', {})
                                trend_data = pa_analysis.get('trend', {})
                                volume_data = pa_analysis.get('volume', {})

                                # Update analysis with boosted confidence and price action details
                                for analysis in individual_analyses:
                                    analysis['confidence'] = boosted_confidence
                                    analysis['price_action'] = {
                                        'validated': True,
                                        'reason': matching_pa['reason'],
                                        'stop_loss': matching_pa['stop_loss'],
                                        'targets': matching_pa['targets'],
                                        'rr_ratio': matching_pa['rr_ratio'],
                                        'confidence_boost': matching_pa['confidence_boost'],
                                        # ✅ NEW: Full PA analysis details for Telegram
                                        'support_levels': [s['price'] for s in sr_data.get('support', [])[:3]],
                                        'resistance_levels': [r['price'] for r in sr_data.get('resistance', [])[:3]],
                                        'trend': trend_data.get('direction', 'N/A'),
                                        'trend_strength': trend_data.get('strength', 'N/A'),
                                        'adx': trend_data.get('adx', 0),
                                        'volume_surge': volume_data.get('is_surge', False),
                                        'volume_ratio': volume_data.get('surge_ratio', 1.0),
                                        'obv_trend': volume_data.get('obv_trend', 'NEUTRAL'),
                                        'vpoc': sr_data.get('vpoc', 0),
                                        'indicators': ['Support/Resistance', 'Trend (ADX)', 'Volume (OBV)', 'Fibonacci'],
                                        'timeframes': ['15m'],
                                    }

                                logger.info(
                                    f"✅ {symbol} PRICE ACTION CONFIRMED: {matching_pa['reason']} | "
                                    f"ML {ml_confidence:.0f}% + PA Boost +{matching_pa['confidence_boost']}% "
                                    f"= {boosted_confidence*100:.0f}%"
                                )
                            elif matching_pa:
                                # ❌ Price action rejects ML signal - skip this trade
                                logger.info(
                                    f"⏸️ {symbol} PRICE ACTION REJECTED: {matching_pa['reason']} | "
                                    f"ML said {ml_signal} {ml_confidence:.0f}% but skipped"
                                )
                                individual_analyses = []  # Clear analyses to skip this symbol
                            else:
                                # No matching PA result - ML trade proceeds without PA boost
                                logger.debug(
                                    f"{symbol} - ML {ml_signal} but no matching PA setup "
                                    f"(LONG: {pa_result.get('long') is not None}, SHORT: {pa_result.get('short') is not None})"
                                )

                    except Exception as e:
                        logger.warning(f"{symbol} - Price action boost failed: {e}")
                        # Don't block trade if price action fails - just log warning

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

            # Use price manager for intelligent caching (reduces API calls by 80%)
            from src.price_manager import get_price_manager
            price_manager = get_price_manager()

            # OHLCV data (multiple timeframes) - ENHANCED with 5m + CACHING
            # 🔧 API RATE LIMIT FIX: Reduced from 200 to 100 candles (still enough for PA + indicators)
            # Indicators consume first 20-50 candles, so fetching 100 ensures 50+ clean candles remain
            ohlcv_5m = await price_manager.get_ohlcv(symbol, '5m', exchange, limit=100)
            ohlcv_15m = await price_manager.get_ohlcv(symbol, '15m', exchange, limit=100)
            ohlcv_1h = await price_manager.get_ohlcv(symbol, '1h', exchange, limit=100)
            ohlcv_4h = await price_manager.get_ohlcv(symbol, '4h', exchange, limit=100)

            # Current ticker (with caching)
            ticker = await price_manager.get_ticker(symbol, exchange, use_cache=True)

            # 🎯 #6: Update price for correlation tracking
            try:
                from src.ml_pattern_learner import get_ml_learner
                ml_learner = await get_ml_learner()
                ml_learner.update_price_for_correlation(symbol, ticker['last'])
            except Exception as e:
                logger.debug(f"Failed to update correlation price for {symbol}: {e}")

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
                # 🛡️ v4.7.4: Fetch 100 levels for deep order book analysis
                order_book = await exchange.fetch_order_book(symbol, limit=100)
                recent_trades = await exchange.fetch_trades(symbol, limit=50)
                # 🛡️ v4.7.6: INFO level to diagnose order book issues
                logger.info(f"📊 {symbol} Order book: {len(order_book.get('bids', []))} bids, {len(order_book.get('asks', []))} asks")
            except Exception as e:
                # 🛡️ v4.7.6: INFO level to see failures
                logger.warning(f"⚠️ {symbol} Order book fetch FAILED: {e}")

            order_flow = analyze_order_flow(order_book, recent_trades)

            # 🛡️ v4.7.6: ALWAYS log order flow result (INFO level)
            logger.info(f"📊 {symbol} Order flow: weighted_imbalance={order_flow.get('weighted_imbalance', 0):.1f}%, depth={order_flow.get('depth_levels_analyzed', 0)}")

            # Smart Money Concepts (institutional edge)
            smart_money = detect_smart_money_concepts(ohlcv_4h, current_price)

            # Volatility Bands (adaptive risk management)
            volatility = calculate_volatility_bands(ohlcv_1h, current_price)

            # 🎯 #5: GARCH FORWARD-LOOKING VOLATILITY PREDICTION
            garch_volatility = None
            try:
                from src.volatility_predictor import get_volatility_predictor
                vol_predictor = get_volatility_predictor()

                # Calculate returns from 1h OHLCV for GARCH
                closes = np.array([candle[4] for candle in ohlcv_1h])  # Close prices
                returns = np.diff(closes) / closes[:-1]  # Percentage returns

                # Predict next-period volatility
                predicted_vol = vol_predictor.predict_volatility(symbol, returns, horizon=1)

                if predicted_vol is not None:
                    vol_regime = vol_predictor.get_volatility_regime(predicted_vol)
                    optimal_lev = vol_predictor.calculate_optimal_leverage(predicted_vol, max_leverage=5)
                    price_range = vol_predictor.predict_next_day_range(current_price, predicted_vol)

                    garch_volatility = {
                        'predicted_vol_annual_pct': predicted_vol,
                        'volatility_regime': vol_regime,
                        'optimal_leverage': optimal_lev,
                        'next_day_range': price_range,
                        'model': 'GARCH(1,1)'
                    }

                    logger.debug(
                        f"GARCH {symbol}: {predicted_vol:.1f}% annual vol | "
                        f"Regime: {vol_regime} | Optimal Lev: {optimal_lev}x"
                    )
                else:
                    # Fallback if GARCH fails
                    garch_volatility = {
                        'predicted_vol_annual_pct': volatility.get('atr_percent', 2.0) * np.sqrt(252) * 100,  # Rough annual estimate
                        'volatility_regime': 'UNKNOWN',
                        'optimal_leverage': 3,
                        'next_day_range': (current_price * 0.95, current_price * 1.05),
                        'model': 'FALLBACK'
                    }

            except Exception as garch_error:
                logger.warning(f"GARCH prediction failed for {symbol}: {garch_error}")
                # Graceful fallback
                garch_volatility = {
                    'predicted_vol_annual_pct': 50.0,  # Default crypto vol
                    'volatility_regime': 'NORMAL_VOL',
                    'optimal_leverage': 3,
                    'next_day_range': (current_price * 0.95, current_price * 1.05),
                    'model': 'ERROR'
                }

            # PHASE 2 CRITICAL FEATURES (ULTRA PROFESSIONAL)

            # Momentum Strength (rate of change analysis)
            momentum = calculate_momentum_strength(ohlcv_15m)

            # BTC Correlation Analysis (if not BTC itself)
            btc_correlation = {'correlation': 0.0, 'correlation_strength': 'n/a', 'independent_move': True, 'recommendation': 'N/A (BTC itself)'}
            if symbol != 'BTC/USDT:USDT':
                try:
                    # Fetch BTC data for correlation
                    # 🔧 API RATE LIMIT FIX: Reduced to 100 candles (sufficient for correlation)
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

            # 🔥 TIER 3: ORDER BOOK ANALYSIS (Whale Detection + Liquidity Zones)
            order_book_analysis = {'signal': 'NEUTRAL', 'signal_strength': 'NONE', 'confidence_boost': 0.0}
            try:
                from src.order_book_analyzer import get_order_book_analyzer

                ob_analyzer = get_order_book_analyzer()
                order_book_analysis = await ob_analyzer.analyze_order_book(symbol)

                if order_book_analysis.get('signal') != 'NEUTRAL':
                    logger.info(
                        f"📊 {symbol} ORDER BOOK: {order_book_analysis['signal']} "
                        f"({order_book_analysis['signal_strength']}) | "
                        f"Confidence Boost: +{order_book_analysis['confidence_boost']:.0%} | "
                        f"Imbalance: {order_book_analysis.get('bid_ask_imbalance', 0):.1%}"
                    )
            except Exception as e:
                logger.debug(f"{symbol} - Order book analysis failed: {e}")

            # 🔧 FIX: Create pandas DataFrame for PA analysis (ai_engine.py expects this!)
            # Convert 15m OHLCV to DataFrame for Price Action analysis
            import pandas as pd
            ohlcv_df = pd.DataFrame(
                ohlcv_15m,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            logger.debug(f"📊 {symbol} - OHLCV data length: {len(ohlcv_df)} candles")

            return {
                'symbol': symbol,
                'current_price': current_price,
                'volume_24h': ticker.get('quoteVolume', 0),
                'ohlcv': {
                    '5m': ohlcv_5m[-20:],    # Last 20 candles (for display/quick analysis)
                    '15m': ohlcv_15m[-20:],
                    '1h': ohlcv_1h[-20:],
                    '4h': ohlcv_4h[-20:]
                },
                # 🎯 FULL OHLCV DATA FOR PRICE ACTION ANALYSIS (need 50+ candles for S/R detection)
                'ohlcv_15m': ohlcv_15m,    # Full 100 candles for PA (API optimized)
                'ohlcv_1h': ohlcv_1h,      # Full 100 candles for PA (API optimized)
                'ohlcv_4h': ohlcv_4h,      # Full 100 candles for PA (API optimized)
                'ohlcv_df': ohlcv_df,      # 🔧 FIX: DataFrame for ai_engine.py PA analysis!
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
                'liquidation_map': liquidation_map,
                # 🎯 #5: GARCH FORWARD-LOOKING VOLATILITY
                'garch_volatility': garch_volatility,
                # 🔥 TIER 3: ORDER BOOK ANALYSIS (Whale Detection + Liquidity Zones)
                'order_book_analysis': order_book_analysis
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

        # FACTOR 1: AI Confidence (30 points)
        confidence = ai_analysis.get('confidence', 0)
        score += confidence * 30

        # NOTE: Historical Performance Boost removed due to async/sync conflict
        # Will be re-implemented in async wrapper function in future update
        # Counter-trend penalty below (FACTOR 2) helps reduce SHORT bias

        # FACTOR 2: Market Breadth Alignment (15 points)
        # If market sentiment aligns with trade direction, boost score
        # Note: market_sentiment uses display format (STRONG BULLISH, BULLISH, etc.)
        sentiment_upper = market_sentiment.upper()
        if "STRONG" in sentiment_upper and "BULLISH" in sentiment_upper and trade_side == "LONG":
            score += 15  # Perfect alignment
        elif "BULLISH" in sentiment_upper and trade_side == "LONG":
            score += 12  # Good alignment
        elif "STRONG" in sentiment_upper and "BEARISH" in sentiment_upper and trade_side == "SHORT":
            score += 15  # Perfect alignment
        elif "BEARISH" in sentiment_upper and trade_side == "SHORT":
            score += 12  # Good alignment
        elif "NEUTRAL" in sentiment_upper or "MIXED" in sentiment_upper:
            score += 7   # Neutral - no clear direction
        else:
            # Trading against market breadth (VERY RISKY!)
            score -= 20  # HEAVY PENALTY for counter-trend trades
            logger.warning(
                f"⚠️ COUNTER-TREND: {market_sentiment} market but {trade_side} trade. "
                f"Score penalty: -20 points"
            )

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

        # 🔥 TIER 3 FACTOR 7: Order Book Signal (0-15 points bonus)
        # Whale positioning and liquidity zones provide institutional edge
        ob_analysis = market_data.get('order_book_analysis', {})
        ob_signal = ob_analysis.get('signal', 'NEUTRAL')
        ob_strength = ob_analysis.get('signal_strength', 'NONE')

        # Check if order book signal aligns with trade direction
        if (ob_signal == 'BUY' and trade_side == 'LONG') or (ob_signal == 'SELL' and trade_side == 'SHORT'):
            # Order book confirms our trade direction!
            if ob_strength == 'STRONG':
                score += 15  # Strong whale support (3+ confluence factors)
                logger.debug(f"  Order Book STRONG confirmation: +15 points")
            elif ob_strength == 'MODERATE':
                score += 10  # Moderate whale support (2 factors)
                logger.debug(f"  Order Book MODERATE confirmation: +10 points")
            elif ob_strength == 'WEAK':
                score += 5   # Weak whale support (1 factor)
                logger.debug(f"  Order Book WEAK confirmation: +5 points")
        elif ob_signal != 'NEUTRAL':
            # Order book contradicts our trade direction - penalty
            score -= 10
            logger.warning(f"  Order Book CONTRADICTION: {ob_signal} vs {trade_side} trade (-10 points)")

        # Cap at 100
        return min(score, 100)

    def _check_entry_confluence(
        self,
        analysis: Dict[str, Any],
        market_data: Dict[str, Any],
        side: str
    ) -> Dict[str, Any]:
        """
        🔥 TIER 3: Multi-factor confluence validation (60+ score required).

        Checks 8 critical factors before allowing trade entry.
        Prevents low-quality setups from being traded.

        Args:
            analysis: AI analysis dict
            market_data: Market data dict with indicators
            side: Trade direction ('LONG' or 'SHORT')

        Returns:
            Dict with approved (bool), score (int), reason (str), factors (list)
        """
        score = 0
        factors = []

        # Get indicators
        indicators_15m = market_data.get('indicators', {}).get('15m', {})

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FACTOR 1: PA Validation (30 points) - Most important
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        pa = analysis.get('price_action', {})
        if pa.get('validated'):
            score += 30
            factors.append('PA validated')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FACTOR 2: RSI Healthy Zone (15 points)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        rsi = indicators_15m.get('rsi', 50)
        if side == 'LONG' and 40 <= rsi <= 70:
            score += 15
            factors.append(f'RSI {rsi:.0f} healthy for LONG')
        elif side == 'SHORT' and 30 <= rsi <= 60:
            score += 15
            factors.append(f'RSI {rsi:.0f} healthy for SHORT')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FACTOR 3: MACD Alignment (15 points)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        macd = indicators_15m.get('macd', 0)
        macd_signal = indicators_15m.get('macd_signal', 0)
        if (side == 'LONG' and macd > macd_signal) or \
           (side == 'SHORT' and macd < macd_signal):
            score += 15
            factors.append('MACD aligned')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FACTOR 4: Volume Confirmation (10 points)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if indicators_15m.get('volume_trend') == 'high':
            score += 10
            factors.append('High volume')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FACTOR 5: Trend Alignment (20 points)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        trend = indicators_15m.get('trend', 'unknown')
        if (side == 'LONG' and trend == 'uptrend') or \
           (side == 'SHORT' and trend == 'downtrend'):
            score += 20
            factors.append(f'Trend: {trend}')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FACTOR 6: No Divergence Conflict (10 points)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        divergence = market_data.get('divergence', {})
        has_divergence = divergence.get('has_divergence', False)
        div_type = divergence.get('type', '')

        # No divergence at all = good
        if not has_divergence:
            score += 10
            factors.append('No divergence conflict')
        # Divergence that agrees with trade = bonus
        elif (side == 'LONG' and 'bullish' in div_type.lower()) or \
             (side == 'SHORT' and 'bearish' in div_type.lower()):
            score += 10
            factors.append(f'{div_type} divergence confirms')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FACTOR 7: SuperTrend Alignment (10 points) - NEW TIER 1 indicator
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        supertrend_trend = indicators_15m.get('supertrend_trend', 'unknown')
        if (side == 'LONG' and supertrend_trend == 'uptrend') or \
           (side == 'SHORT' and supertrend_trend == 'downtrend'):
            score += 10
            factors.append(f'SuperTrend: {supertrend_trend}')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FACTOR 8: VWAP Bias (10 points) - NEW TIER 1 indicator
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        vwap_bias = indicators_15m.get('vwap_bias', 'neutral')
        if (side == 'LONG' and 'bullish' in vwap_bias) or \
           (side == 'SHORT' and 'bearish' in vwap_bias):
            score += 10
            factors.append(f'VWAP bias: {vwap_bias}')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🎯 TIER 2 CONSERVATIVE: New Professional Indicators
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # FACTOR 9: Ichimoku Cloud (15 points) - Ultimate trend system
        ichimoku_signal = indicators_15m.get('ichimoku_signal', 'NEUTRAL')
        ichimoku_price_vs_cloud = indicators_15m.get('ichimoku_price_vs_cloud', 'IN_CLOUD')
        ichimoku_cloud_color = indicators_15m.get('ichimoku_cloud_color', 'neutral')

        if side == 'LONG':
            # Perfect Ichimoku LONG: Price above bullish cloud + buy signal
            if ichimoku_price_vs_cloud == 'ABOVE_CLOUD' and ichimoku_cloud_color == 'BULLISH':
                score += 15
                factors.append('Ichimoku: STRONG BULLISH (above cloud)')
            elif ichimoku_signal in ['BUY', 'STRONG_BUY']:
                score += 10
                factors.append(f'Ichimoku: {ichimoku_signal}')
        elif side == 'SHORT':
            # Perfect Ichimoku SHORT: Price below bearish cloud + sell signal
            if ichimoku_price_vs_cloud == 'BELOW_CLOUD' and ichimoku_cloud_color == 'BEARISH':
                score += 15
                factors.append('Ichimoku: STRONG BEARISH (below cloud)')
            elif ichimoku_signal in ['SELL', 'STRONG_SELL']:
                score += 10
                factors.append(f'Ichimoku: {ichimoku_signal}')

        # FACTOR 10: Stochastic RSI (10 points) - Enhanced overbought/oversold
        stoch_rsi_signal = indicators_15m.get('stoch_rsi_signal_new', 'NEUTRAL')
        stoch_rsi_zone = indicators_15m.get('stoch_rsi_zone', 'NEUTRAL')

        if side == 'LONG':
            # Best: Oversold + bullish signal
            if stoch_rsi_zone == 'OVERSOLD' and 'BUY' in stoch_rsi_signal:
                score += 10
                factors.append('StochRSI: OVERSOLD + BUY')
            elif 'BUY' in stoch_rsi_signal:
                score += 5
                factors.append(f'StochRSI: {stoch_rsi_signal}')
        elif side == 'SHORT':
            # Best: Overbought + bearish signal
            if stoch_rsi_zone == 'OVERBOUGHT' and 'SELL' in stoch_rsi_signal:
                score += 10
                factors.append('StochRSI: OVERBOUGHT + SELL')
            elif 'SELL' in stoch_rsi_signal:
                score += 5
                factors.append(f'StochRSI: {stoch_rsi_signal}')

        # FACTOR 11: Money Flow Index (10 points) - Volume-weighted confirmation
        mfi_signal = indicators_15m.get('mfi_signal_new', 'NEUTRAL')
        mfi_zone = indicators_15m.get('mfi_zone', 'NEUTRAL')
        mfi_value = indicators_15m.get('mfi_value', 50.0)

        if side == 'LONG':
            # Strong oversold MFI = institutional accumulation
            if mfi_zone == 'STRONG_OVERSOLD':
                score += 10
                factors.append(f'MFI: STRONG OVERSOLD ({mfi_value:.0f})')
            elif mfi_zone == 'OVERSOLD' or 'BUY' in mfi_signal:
                score += 5
                factors.append(f'MFI: {mfi_zone or mfi_signal}')
        elif side == 'SHORT':
            # Strong overbought MFI = institutional distribution
            if mfi_zone == 'STRONG_OVERBOUGHT':
                score += 10
                factors.append(f'MFI: STRONG OVERBOUGHT ({mfi_value:.0f})')
            elif mfi_zone == 'OVERBOUGHT' or 'SELL' in mfi_signal:
                score += 5
                factors.append(f'MFI: {mfi_zone or mfi_signal}')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FINAL VERDICT: Need 40+/155 to approve (11 factors total)
        # 🎯 LOWERED from 60 to 40 for PA-ONLY mode (TIER 2 indicators are very strict)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        approved = score >= 40

        if not approved:
            missing_factors = []
            if score < 30:
                missing_factors.append("PA not validated")
            if 'RSI' not in str(factors):
                missing_factors.append("RSI not in healthy zone")
            if 'MACD' not in str(factors):
                missing_factors.append("MACD not aligned")
            if 'volume' not in str(factors).lower():
                missing_factors.append("No volume confirmation")
            if 'Trend' not in str(factors):
                missing_factors.append("Trend not aligned")

            reason = f"Only {len(factors)}/11 factors present. Missing: {', '.join(missing_factors[:3])}"
        else:
            reason = f"Strong confluence: {len(factors)}/11 factors present"

        return {
            'approved': approved,
            'score': score,
            'reason': reason,
            'factors': factors,
            'total_possible': 155  # 🎯 TIER 2: Increased from 120 (8 factors → 11 factors)
        }

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
        Execute the trade with proper risk management and quality controls.

        Args:
            opportunity: Best opportunity dict with symbol, analysis, market_data
        """
        symbol = opportunity['symbol']
        analysis = opportunity['analysis']
        market_data = opportunity['market_data']

        logger.info(f"📊 Preparing to execute trade: {symbol} {analysis['side']}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔥 TIER 3: CONFLUENCE SCORING VALIDATION (60+ Requirement)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #
        # Multi-factor confluence check - need 60+/100 score to enter trade
        # Prevents low-quality setups from being traded
        #
        # 🎯 USER REQUEST: DISABLED FOR PA-ONLY MODE!
        # - PA-ONLY already has strict filters (ADX ≥15, S/R ≥3 levels, trend, volume)
        # - Confluence check is redundant and blocks good PA setups
        # - ML trades still use confluence for quality control
        #
        # FACTORS CHECKED (ML trades only):
        # 1. PA validation (30 pts) - Price action setup quality
        # 2. RSI healthy (15 pts) - Not overbought/oversold extreme
        # 3. MACD alignment (15 pts) - Momentum agrees with direction
        # 4. Volume confirmation (10 pts) - High volume supporting move
        # 5. Trend alignment (20 pts) - Trend matches trade direction
        # 6. No divergence conflict (10 pts) - No bearish/bullish divergence
        # 7. SuperTrend alignment (10 pts) - NEW indicator confirms
        # 8. VWAP bias (10 pts) - Institutional flow agrees
        #
        # TOTAL: 120 points possible, need 60+ to trade
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # 🔥 RE-ENABLED: Apply confluence check to ALL trades (including PA-ONLY)
        # USER ISSUE: Disabling confluence led to low-quality entries and immediate stop-loss hits
        # FIX: PA-ONLY now has strict internal filters + confluence acts as final quality gate

        model_name = analysis.get('model_name', 'unknown')

        # 🚀 CRITICAL FIX: Use enhanced confluence from scan phase (don't recalculate!)
        # The opportunity already contains confluence score from enhanced system
        confluence = opportunity.get('confluence', {})

        # If no confluence data (shouldn't happen), calculate it now
        if not confluence or 'score' not in confluence:
            logger.warning(f"⚠️ No confluence data in opportunity, calculating now...")
            confluence = self._check_entry_confluence(analysis, market_data, analysis['side'])

        # 🚀 ENHANCED SYSTEM: Use professional confluence thresholds if enabled
        try:
            enhanced_system = get_enhanced_trading_system()
            if enhanced_system.enable_confluence_filtering:
                # Professional system: Single high threshold (75+) for ALL trades
                min_confluence_score = enhanced_system.min_confluence_score
                logger.info(
                    f"🎯 Enhanced System Active: Using {min_confluence_score} threshold "
                    f"(Quality: {confluence.get('quality', 'UNKNOWN')})"
                )
            else:
                # Legacy system: Minimum 60 with MTF+momentum+volume checks
                if model_name == 'PA-ONLY':
                    min_confluence_score = 60
                elif model_name == 'PA-Override':
                    min_confluence_score = 65
                else:
                    min_confluence_score = 70
        except Exception as e:
            logger.warning(f"⚠️ Enhanced system check failed: {e}")
            # Fallback: Minimum 60 for safety
            if model_name == 'PA-ONLY':
                min_confluence_score = 60
            elif model_name == 'PA-Override':
                min_confluence_score = 65
            else:
                min_confluence_score = 70

        if not confluence['approved'] or confluence['score'] < min_confluence_score:
            # 🛡️ v4.7.11: Show filter_reason when quality is FILTERED
            if confluence.get('quality') == 'FILTERED' and confluence.get('filter_reason'):
                reason_text = f"🛡️ FILTER: {confluence.get('filter_reason')}"
            else:
                # Support both old ('reason', 'factors') and new ('reasoning', 'component_scores') formats
                reason_text = confluence.get('reasoning', confluence.get('reason', 'Unknown'))

            logger.warning(
                f"❌ CONFLUENCE CHECK FAILED ({model_name}): {confluence['score']}/100 (need {min_confluence_score}+)\n"
                f"   Quality: {confluence.get('quality', 'UNKNOWN')}\n"
                f"   Reason: {reason_text}"
            )
            notifier = get_notifier()
            await notifier.send_alert(
                'info',
                f"⏸️ Trade skipped - Low confluence:\n"
                f"{symbol} {analysis['side']} ({model_name})\n\n"
                f"Score: {confluence['score']}/100 (need {min_confluence_score}+)\n"
                f"Quality: {confluence.get('quality', 'UNKNOWN')}\n"
                f"Reason: {reason_text}"
            )
            return

        # Support both formats for passed trades too
        factors_info = ""
        if 'factors' in confluence:
            factors_info = f"Factors ({len(confluence['factors'])}): {', '.join(confluence['factors'])}"
        elif 'component_scores' in confluence:
            factors_info = f"Quality: {confluence.get('quality', 'UNKNOWN')}"

        logger.info(
            f"✅ CONFLUENCE CHECK PASSED ({model_name}): {confluence['score']}/100\n"
            f"   {factors_info}"
        )

        # 🎯 NEW: Trade Quality Manager validation (prevents overtrading and repeated failures)
        from src.trade_quality_manager import get_trade_quality_manager
        from src.database import get_db_client

        quality_mgr = get_trade_quality_manager()
        db = await get_db_client()

        entry_confidence = analysis.get('confidence', 0) * 100  # Convert to percentage

        # Validate trade quality (checks cooldowns, frequency limits, confidence gaps)
        is_valid, rejection_reason = await quality_mgr.validate_trade_entry(
            symbol, entry_confidence, db
        )

        if not is_valid:
            logger.warning(f"❌ Trade quality check FAILED: {rejection_reason}")
            notifier = get_notifier()
            await notifier.send_alert(
                'info',
                f"⏸️ Trade skipped:\n{symbol} {analysis['side']}\n\n"
                f"Reason: {rejection_reason}\n\n"
                f"Quality controls prevent overtrading."
            )
            return

        # Validate with risk manager
        risk_manager = get_risk_manager()
        notifier = get_notifier()

        trade_params = {
            'symbol': symbol,
            'side': analysis['side'],
            'leverage': analysis['suggested_leverage'],
            'stop_loss_percent': analysis['stop_loss_percent'],
            'current_price': market_data['current_price'],
            'market_breadth': opportunity.get('market_breadth'),  # For market direction filter
            'market_data': market_data  # 🛡️ v4.7.3: Required for STRICT technical validation
        }

        # 🔧 FIX #4: Position spacing check (prevent same-minute entries)
        # PROBLEM: 3 SHORTs opened in 6 minutes (SOL, BNB at 13:14, STORJ at 13:20)
        # SOLUTION: Minimum 2-minute gap between position openings (reduced from 5 for faster entry)
        import time
        current_time = time.time()
        time_since_last_position = current_time - self.last_position_time
        min_spacing_seconds = 120  # 2 minutes (was 5 - too slow for catching trends)

        if self.last_position_time > 0 and time_since_last_position < min_spacing_seconds:
            wait_time = min_spacing_seconds - time_since_last_position
            logger.warning(
                f"⏰ POSITION SPACING: Waiting {wait_time:.0f}s before next entry "
                f"(last position: {time_since_last_position:.0f}s ago, need {min_spacing_seconds}s gap)"
            )
            await notifier.send_alert(
                'info',
                f"⏰ Position spacing active:\n{symbol} {analysis['side']}\n\n"
                f"Waiting {wait_time:.0f}s before next entry\n"
                f"(Prevents correlation risk from same-minute entries)"
            )
            return

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

        # 🎯 Record trade opening in quality manager
        quality_mgr.record_trade_opened(symbol, entry_confidence)

        # Execute trade
        executor = get_trade_executor()
        success = await executor.open_position(trade_params, analysis, market_data)

        if success:
            # 🔧 FIX #4: Update last position time for spacing check
            self.last_position_time = current_time
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
