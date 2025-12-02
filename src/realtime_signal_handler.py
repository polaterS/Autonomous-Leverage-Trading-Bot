"""
Real-Time Signal Handler - Instant trade execution from WebSocket signals.

This module handles signals from RealtimeTrendDetector and executes trades
IMMEDIATELY without waiting for periodic scans.

Flow:
1. RealtimeTrendDetector detects trend start via WebSocket
2. Emits signal to this handler
3. Handler validates signal (confluence, risk, etc.)
4. Executes trade INSTANTLY

This reduces entry delay from 3-8 minutes to < 1 second!
"""

import asyncio
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
from src.config import get_settings
from src.utils import setup_logging
from src.risk_manager import get_risk_manager
from src.trade_executor import get_trade_executor
from src.telegram_notifier import get_notifier
from src.exchange_client import get_exchange_client
from src.database import get_db_client
# 🆕 v4.4.0: Import enhanced indicators for professional confluence scoring
# 🆕 v4.5.0: Import advanced indicators (VWAP, StochRSI, CMF, Fibonacci)
from src.indicators import calculate_enhanced_indicators, calculate_advanced_indicators

logger = setup_logging()


class RealtimeSignalHandler:
    """
    Handles real-time signals and executes trades instantly.

    Validates signals against:
    - Risk management rules
    - Position limits
    - Confluence score (quick check)
    - Market conditions
    - 🔥 NEW: Market trend filter (BTC trend alignment)
    """

    def __init__(self):
        self.settings = get_settings()
        self.is_running = False

        # ═══════════════════════════════════════════════════════════════════════
        # 🚫 v4.5.1: INSTANT TRADING DISABLED
        # ═══════════════════════════════════════════════════════════════════════
        # Reason: Realtime signals tend to enter trades TOO LATE
        # - By the time signal is detected and validated, price has already peaked (for LONG)
        #   or bottomed (for SHORT), causing immediate losses
        # - Main trading loop with proper confluence analysis is more reliable
        # - Set to True to re-enable instant trading
        # ═══════════════════════════════════════════════════════════════════════
        self.ENABLE_INSTANT_TRADES = False  # 🚫 DISABLED - Use main trading loop only

        # Track recent executions to prevent duplicates
        self.recent_executions: Dict[str, float] = {}  # symbol -> timestamp
        self.execution_cooldown = 300  # 5 minutes between same-symbol trades

        # Minimum signal strength to consider
        self.min_signal_strength = 30

        # 🔥 NEW: Market trend cache (BTC trend determines market direction)
        self.market_trend: Dict[str, Any] = {
            'direction': 'NEUTRAL',  # 'BULLISH', 'BEARISH', 'NEUTRAL'
            'strength': 0,
            'last_update': None
        }
        self.market_trend_update_interval = 60  # Update every 60 seconds

        # Statistics
        self.stats = {
            'signals_received': 0,
            'signals_validated': 0,
            'trades_executed': 0,
            'trades_rejected': 0,
            'counter_trend_rejected': 0  # 🔥 NEW: Track counter-trend rejections
        }

    async def start(self):
        """Start the signal handler."""
        self.is_running = True
        # Update market trend on startup
        await self._update_market_trend()

        # 🚫 v4.5.1: Log instant trading status
        if self.ENABLE_INSTANT_TRADES:
            logger.info("✅ Real-Time Signal Handler started - INSTANT TRADING ENABLED")
        else:
            logger.info(
                "✅ Real-Time Signal Handler started - 🚫 INSTANT TRADING DISABLED | "
                "Signals will be logged but NO trades will be executed via realtime"
            )

    async def stop(self):
        """Stop the signal handler."""
        self.is_running = False
        logger.info("Real-Time Signal Handler stopped")

    async def _update_market_trend(self):
        """
        🔥 NEW: Update market trend based on BTC.

        BTC is the market leader - when BTC dumps, altcoins dump harder.
        When BTC pumps, altcoins pump (usually).

        This helps us avoid counter-trend trades that lose money.
        """
        try:
            exchange = await get_exchange_client()

            # Fetch BTC 15m candles for trend analysis
            btc_ohlcv = await exchange.fetch_ohlcv('BTC/USDT:USDT', '15m', limit=20)

            if not btc_ohlcv or len(btc_ohlcv) < 15:
                logger.warning("Could not fetch BTC data for market trend")
                return

            closes = [c[4] for c in btc_ohlcv]

            # Calculate EMAs
            ema_fast = self._quick_ema(closes, 5)
            ema_slow = self._quick_ema(closes, 13)

            # Calculate recent momentum (last 5 candles = 75 minutes)
            recent_change_pct = (closes[-1] - closes[-5]) / closes[-5] * 100

            # Determine market direction
            if ema_fast > ema_slow and recent_change_pct > 0.3:
                direction = 'BULLISH'
                strength = min(100, int(recent_change_pct * 30))
            elif ema_fast < ema_slow and recent_change_pct < -0.3:
                direction = 'BEARISH'
                strength = min(100, int(abs(recent_change_pct) * 30))
            else:
                direction = 'NEUTRAL'
                strength = 0

            self.market_trend = {
                'direction': direction,
                'strength': strength,
                'btc_price': closes[-1],
                'btc_change_pct': recent_change_pct,
                'last_update': datetime.now()
            }

            logger.info(
                f"📊 Market Trend Updated: {direction} | "
                f"BTC: ${closes[-1]:.2f} ({recent_change_pct:+.2f}%)"
            )

        except Exception as e:
            logger.warning(f"Failed to update market trend: {e}")

    async def handle_signal(self, signal: Dict[str, Any]):
        """
        Handle incoming signal from RealtimeTrendDetector.

        This is called INSTANTLY when a trend is detected.
        Must be fast but thorough in validation.

        Args:
            signal: {
                'symbol': 'BTC/USDT:USDT',
                'side': 'LONG' or 'SHORT',
                'trend': 'UP' or 'DOWN',
                'strength': 0-100,
                'trigger': 'EMA_CROSS' / 'MOMENTUM_SHIFT' / 'VOLUME_SPIKE',
                'price': Decimal,
                'timestamp': datetime,
                'source': 'REALTIME_WEBSOCKET'
            }
        """
        if not self.is_running:
            return

        self.stats['signals_received'] += 1
        symbol = signal['symbol']
        side = signal['side']
        strength = signal['strength']
        trigger = signal['trigger']
        price = signal['price']

        # ═══════════════════════════════════════════════════════════════════════
        # 🚫 v4.5.1: INSTANT TRADING DISABLED CHECK
        # ═══════════════════════════════════════════════════════════════════════
        if not self.ENABLE_INSTANT_TRADES:
            logger.info(
                f"📡 REALTIME SIGNAL (LOGGED ONLY - NO TRADE): {symbol} {side} | "
                f"Strength: {strength} | Trigger: {trigger} | "
                f"Price: ${float(price):.4f} | "
                f"⚠️ Instant trading DISABLED - signal logged for monitoring only"
            )
            return  # 🚫 Do NOT execute trade - just log and return
        # ═══════════════════════════════════════════════════════════════════════

        logger.info(
            f"📡 REALTIME SIGNAL RECEIVED: {symbol} {side} | "
            f"Strength: {strength} | Trigger: {trigger} | "
            f"Price: ${float(price):.4f}"
        )

        notifier = get_notifier()

        try:
            # === VALIDATION PHASE (must be fast!) ===

            # 0. 🔥 NEW: Update market trend periodically
            if (self.market_trend['last_update'] is None or
                (datetime.now() - self.market_trend['last_update']).seconds > self.market_trend_update_interval):
                await self._update_market_trend()

            # 1. Check signal strength
            if strength < self.min_signal_strength:
                logger.debug(f"Signal too weak: {strength} < {self.min_signal_strength}")
                self.stats['trades_rejected'] += 1
                return

            # 🎯 v4.2: Market trend - LOG ONLY, no blocking
            # Smart Confidence system in PA analyzer handles trend penalties
            market_direction = self.market_trend.get('direction', 'NEUTRAL')
            is_counter_trend = False

            if market_direction == 'BEARISH' and side == 'LONG':
                is_counter_trend = True
                logger.info(
                    f"⚠️ COUNTER-TREND NOTE: {symbol} LONG in BEARISH market | "
                    f"PA analyzer will apply confidence penalty"
                )
            elif market_direction == 'BULLISH' and side == 'SHORT':
                is_counter_trend = True
                logger.info(
                    f"⚠️ COUNTER-TREND NOTE: {symbol} SHORT in BULLISH market | "
                    f"PA analyzer will apply confidence penalty"
                )

            # 2. Check execution cooldown
            current_time = datetime.now().timestamp()
            last_execution = self.recent_executions.get(symbol, 0)

            if current_time - last_execution < self.execution_cooldown:
                remaining = int(self.execution_cooldown - (current_time - last_execution))
                logger.info(f"⏰ Execution cooldown for {symbol}: {remaining}s remaining")
                return

            # 3. Check if we can open more positions
            risk_manager = get_risk_manager()
            can_open = await risk_manager.can_open_position()

            if not can_open['can_open']:
                logger.info(f"❌ Cannot open position: {can_open.get('reason', 'Unknown')}")
                self.stats['trades_rejected'] += 1
                return

            # 4. Check if we already have a position in this symbol
            # 🔥 FIX: Get active positions from database instead of exchange
            db = await get_db_client()
            active_positions = await db.get_active_positions()
            existing_symbols = [p['symbol'] for p in active_positions]

            if symbol in existing_symbols:
                logger.info(f"⚠️ Already have position in {symbol}, skipping")
                return

            # 5. Quick market data fetch for confluence scoring
            market_data = await self._fetch_quick_market_data(symbol)

            if not market_data:
                logger.warning(f"Failed to fetch market data for {symbol}")
                self.stats['trades_rejected'] += 1
                return

            # 5.5 🎯 QUALITY FILTER: Volume confirmation required
            # Reject signals without sufficient volume (prevents false breakouts)
            current_volume = market_data.get('volume_24h', 0)
            avg_volume = market_data.get('avg_volume_24h', current_volume)

            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                MIN_VOLUME_RATIO = 1.2  # Require 20% above average volume

                if volume_ratio < MIN_VOLUME_RATIO:
                    logger.info(
                        f"❌ Volume too low for {symbol}: {volume_ratio:.2f}x (need {MIN_VOLUME_RATIO}x)"
                    )
                    self.stats['trades_rejected'] += 1
                    return

            # 6. Quick confluence check (simplified for speed)
            confluence_score = await self._quick_confluence_check(
                symbol, side, market_data, signal
            )

            # ═══════════════════════════════════════════════════════════════════
            # 🎯 HIGH-CERTAINTY TRADE VALIDATION SYSTEM
            # ═══════════════════════════════════════════════════════════════════
            # v4.2 SMART CONFIDENCE SYSTEM
            # For a trade to execute:
            # 1. Confluence >= 60 (basic quality signal)
            # 2. Smart PA confidence >= 55% (ML + PA adjustments)
            # 3. Multi-timeframe + momentum checks still apply
            # ═══════════════════════════════════════════════════════════════════

            # 🎯 v4.3.1: Use MIN_CONFLUENCE_SCORE from environment/config
            # This ensures env var MIN_CONFLUENCE_SCORE=75 is actually used!
            MIN_CERTAINTY_CONFLUENCE = self.settings.min_confluence_score  # From env (default 75)

            if confluence_score < MIN_CERTAINTY_CONFLUENCE:
                logger.info(
                    f"❌ HIGH-CERTAINTY REJECTED: {symbol} confluence {confluence_score:.1f} < {MIN_CERTAINTY_CONFLUENCE}"
                )
                self.stats['trades_rejected'] += 1
                return

            # 🎯 CHECK 2: MULTI-TIMEFRAME ALIGNMENT
            indicators = market_data.get('indicators', {})

            # Get EMA trends from different timeframes
            ema_15m_fast = indicators.get('15m', {}).get('ema_fast', 0)
            ema_15m_slow = indicators.get('15m', {}).get('ema_slow', 0)
            ema_1h_fast = indicators.get('1h', {}).get('ema_fast', 0)
            ema_1h_slow = indicators.get('1h', {}).get('ema_slow', 0)
            ema_4h_fast = indicators.get('4h', {}).get('ema_fast', 0)
            ema_4h_slow = indicators.get('4h', {}).get('ema_slow', 0)

            # Check alignment
            trend_15m = 'LONG' if ema_15m_fast > ema_15m_slow else 'SHORT'
            trend_1h = 'LONG' if ema_1h_fast > ema_1h_slow else 'SHORT'
            trend_4h = 'LONG' if ema_4h_fast > ema_4h_slow else 'SHORT'

            # All timeframes must agree with our trade direction
            mtf_aligned = (trend_15m == side and trend_1h == side and trend_4h == side)

            if not mtf_aligned and ema_15m_fast > 0:  # Only check if we have data
                logger.info(
                    f"❌ HIGH-CERTAINTY REJECTED: {symbol} MTF not aligned | "
                    f"4h:{trend_4h} 1h:{trend_1h} 15m:{trend_15m} vs Trade:{side}"
                )
                self.stats['trades_rejected'] += 1
                return

            # 🎯 CHECK 3: STRONG MOMENTUM (price actively moving)
            current_price = float(market_data.get('current_price', 0))
            price_change_1h = float(indicators.get('1h', {}).get('price_change_pct', 0))

            # For LONG: price should be rising (positive momentum)
            # For SHORT: price should be falling (negative momentum)
            MIN_MOMENTUM_PCT = 0.3  # At least 0.3% move in our direction

            momentum_ok = False
            if side == 'LONG' and price_change_1h > MIN_MOMENTUM_PCT:
                momentum_ok = True
            elif side == 'SHORT' and price_change_1h < -MIN_MOMENTUM_PCT:
                momentum_ok = True

            if not momentum_ok and price_change_1h != 0:
                logger.info(
                    f"❌ HIGH-CERTAINTY REJECTED: {symbol} weak momentum | "
                    f"1h change: {price_change_1h:.2f}% (need {MIN_MOMENTUM_PCT}%+ for {side})"
                )
                self.stats['trades_rejected'] += 1
                return

            # 🎯 CHECK 4: MINIMUM PROFIT POTENTIAL ($2-3 after fees)
            # Position size ~$1000 with 25x leverage from $40 margin
            # Need at least 0.3% price move for $3 profit (0.3% × $1000 = $3)
            # After fees (~$1): net $2 minimum
            position_value = 1000  # Approximate position value
            MIN_PROFIT_USD = 2.50  # Minimum $2.50 profit target
            FEES_USD = 1.00  # Approximate round-trip fees

            # Calculate required price move for minimum profit
            required_move_pct = ((MIN_PROFIT_USD + FEES_USD) / position_value) * 100

            # Check ATR (Average True Range) - indicates expected price movement
            atr_pct = float(indicators.get('15m', {}).get('atr_percent', 0))

            # ATR should be at least 1.5x our required move (room for profit)
            if atr_pct > 0 and atr_pct < required_move_pct * 1.5:
                logger.info(
                    f"❌ HIGH-CERTAINTY REJECTED: {symbol} low volatility | "
                    f"ATR: {atr_pct:.2f}% < {required_move_pct * 1.5:.2f}% needed for ${MIN_PROFIT_USD} profit"
                )
                self.stats['trades_rejected'] += 1
                return

            # ═══════════════════════════════════════════════════════════════════
            # ✅ ALL CHECKS PASSED - HIGH-CERTAINTY TRADE APPROVED
            # ═══════════════════════════════════════════════════════════════════

            self.stats['signals_validated'] += 1

            logger.info(
                f"✅ HIGH-CERTAINTY TRADE APPROVED: {symbol} {side} | "
                f"Confluence: {confluence_score:.1f}/100 | "
                f"MTF: ✓ | Momentum: {price_change_1h:.2f}% | ATR: {atr_pct:.2f}%"
            )

            # Send notification about instant entry
            await notifier.send_alert(
                'info',
                f"⚡ INSTANT ENTRY TRIGGERED!\n\n"
                f"Symbol: {symbol}\n"
                f"Side: {side}\n"
                f"Trigger: {trigger}\n"
                f"Signal Strength: {strength}\n"
                f"Confluence: {confluence_score:.1f}/100\n"
                f"Price: ${float(price):.4f}\n\n"
                f"🚀 Executing trade NOW..."
            )

            # Execute trade
            success = await self._execute_instant_trade(
                symbol=symbol,
                side=side,
                signal=signal,
                market_data=market_data,
                confluence_score=confluence_score
            )

            if success:
                self.recent_executions[symbol] = current_time
                self.stats['trades_executed'] += 1
                logger.info(f"✅ INSTANT TRADE EXECUTED: {symbol} {side}")
            else:
                self.stats['trades_rejected'] += 1
                logger.warning(f"❌ Trade execution failed: {symbol}")

        except Exception as e:
            logger.error(f"Error handling signal for {symbol}: {e}", exc_info=True)
            self.stats['trades_rejected'] += 1

    async def _fetch_quick_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch FULL market data for proper validation.

        🔥 FIX: Now includes multi-timeframe data, ATR, and price change calculations
        that are REQUIRED for the quality checks to work!
        """
        try:
            exchange = await get_exchange_client()

            # 🔥 Fetch MULTIPLE timeframes for proper MTF analysis
            ohlcv_15m = await exchange.fetch_ohlcv(symbol, '15m', limit=100)
            ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=50)
            ohlcv_4h = await exchange.fetch_ohlcv(symbol, '4h', limit=30)

            if not ohlcv_15m or len(ohlcv_15m) < 50:
                logger.warning(f"Insufficient 15m data for {symbol}")
                return None

            # Extract price data
            closes_15m = [c[4] for c in ohlcv_15m]
            highs_15m = [c[2] for c in ohlcv_15m]
            lows_15m = [c[3] for c in ohlcv_15m]
            volumes_15m = [c[5] for c in ohlcv_15m]
            current_price = closes_15m[-1]

            # Calculate 15m indicators
            ema_fast_15m = self._quick_ema(closes_15m, 9)
            ema_slow_15m = self._quick_ema(closes_15m, 21)
            rsi_15m = self._quick_rsi(closes_15m)
            atr_15m = self._calculate_atr(highs_15m, lows_15m, closes_15m, 14)
            atr_percent_15m = (atr_15m / current_price * 100) if current_price > 0 else 0

            # Price change over last 4 candles (1 hour in 15m timeframe)
            price_change_1h = ((closes_15m[-1] - closes_15m[-5]) / closes_15m[-5] * 100) if len(closes_15m) >= 5 else 0

            # Calculate 1h indicators
            closes_1h = [c[4] for c in ohlcv_1h] if ohlcv_1h else []
            ema_fast_1h = self._quick_ema(closes_1h, 9) if len(closes_1h) >= 9 else 0
            ema_slow_1h = self._quick_ema(closes_1h, 21) if len(closes_1h) >= 21 else 0

            # Calculate 4h indicators
            closes_4h = [c[4] for c in ohlcv_4h] if ohlcv_4h else []
            ema_fast_4h = self._quick_ema(closes_4h, 9) if len(closes_4h) >= 9 else 0
            ema_slow_4h = self._quick_ema(closes_4h, 21) if len(closes_4h) >= 21 else 0

            # Volume analysis
            avg_volume = sum(volumes_15m[-20:]) / 20 if len(volumes_15m) >= 20 else sum(volumes_15m) / len(volumes_15m)
            current_volume = volumes_15m[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            # Trend determination
            trend = 'UPTREND' if ema_fast_15m > ema_slow_15m else 'DOWNTREND' if ema_fast_15m < ema_slow_15m else 'NEUTRAL'

            return {
                'symbol': symbol,
                'current_price': current_price,
                'ohlcv': ohlcv_15m,
                'volume_24h': current_volume,
                'avg_volume_24h': avg_volume,
                'indicators': {
                    'rsi': rsi_15m,
                    'ema_9': ema_fast_15m,
                    'ema_21': ema_slow_15m,
                    'trend': trend,
                    'volume_ratio': volume_ratio,
                    # 🔥 Multi-timeframe data for MTF checks
                    '15m': {
                        'ema_fast': ema_fast_15m,
                        'ema_slow': ema_slow_15m,
                        'atr_percent': atr_percent_15m,
                        'rsi': rsi_15m
                    },
                    '1h': {
                        'ema_fast': ema_fast_1h,
                        'ema_slow': ema_slow_1h,
                        'price_change_pct': price_change_1h
                    },
                    '4h': {
                        'ema_fast': ema_fast_4h,
                        'ema_slow': ema_slow_4h
                    }
                },
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    def _calculate_atr(self, highs: list, lows: list, closes: list, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(highs) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_ranges.append(max(high_low, high_close, low_close))

        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0

        return sum(true_ranges[-period:]) / period

    def _quick_rsi(self, closes: list, period: int = 14) -> float:
        """Calculate RSI quickly."""
        if len(closes) < period + 1:
            return 50.0

        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _quick_ema(self, data: list, period: int) -> float:
        """Calculate EMA quickly."""
        if len(data) < period:
            return data[-1] if data else 0

        multiplier = 2 / (period + 1)
        ema = data[0]

        for price in data[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    async def _quick_confluence_check(
        self,
        symbol: str,
        side: str,
        market_data: Dict[str, Any],
        signal: Dict[str, Any]
    ) -> float:
        """
        🔥 ENHANCED confluence score with PA analysis and S/R levels.

        Now includes:
        - Multi-timeframe alignment check
        - Support/Resistance proximity analysis
        - Full indicator analysis
        - PA pattern recognition
        """
        score = 0.0
        indicators = market_data.get('indicators', {})

        # ═══════════════════════════════════════════════════════════════════
        # 1. MULTI-TIMEFRAME ALIGNMENT (up to 20 points)
        # ═══════════════════════════════════════════════════════════════════
        ema_15m_fast = indicators.get('15m', {}).get('ema_fast', 0)
        ema_15m_slow = indicators.get('15m', {}).get('ema_slow', 0)
        ema_1h_fast = indicators.get('1h', {}).get('ema_fast', 0)
        ema_1h_slow = indicators.get('1h', {}).get('ema_slow', 0)
        ema_4h_fast = indicators.get('4h', {}).get('ema_fast', 0)
        ema_4h_slow = indicators.get('4h', {}).get('ema_slow', 0)

        mtf_score = 0
        if ema_15m_fast > 0:
            trend_15m = 'LONG' if ema_15m_fast > ema_15m_slow else 'SHORT'
            trend_1h = 'LONG' if ema_1h_fast > ema_1h_slow else 'SHORT'
            trend_4h = 'LONG' if ema_4h_fast > ema_4h_slow else 'SHORT'

            # Count aligned timeframes
            aligned = sum([
                1 for t in [trend_15m, trend_1h, trend_4h]
                if t == side
            ])

            if aligned == 3:
                mtf_score = 20  # All aligned - perfect
            elif aligned == 2:
                mtf_score = 12  # 2 of 3 aligned
            elif aligned == 1:
                mtf_score = 5   # Only 1 aligned - weak
            else:
                mtf_score = 0   # Counter-trend on all TFs

        score += mtf_score

        # ═══════════════════════════════════════════════════════════════════
        # 2. PA ANALYSIS - Support/Resistance (up to 25 points)
        # ═══════════════════════════════════════════════════════════════════
        try:
            from src.pa_analyzer import get_pa_analyzer

            pa_analyzer = get_pa_analyzer()
            ohlcv = market_data.get('ohlcv', [])
            current_price = market_data.get('current_price', 0)

            if ohlcv and len(ohlcv) >= 50 and current_price > 0:
                # Get S/R levels
                sr_result = pa_analyzer.find_sr_levels_from_ohlcv(ohlcv)
                supports = sr_result.get('supports', [])
                resistances = sr_result.get('resistances', [])

                # Calculate distance to nearest S/R
                nearest_support = min([s for s in supports if s < current_price], default=0)
                nearest_resistance = min([r for r in resistances if r > current_price], default=current_price * 1.1)

                support_distance_pct = ((current_price - nearest_support) / current_price * 100) if nearest_support > 0 else 10
                resistance_distance_pct = ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance > 0 else 10

                # For LONG: Want to be near support, away from resistance
                # For SHORT: Want to be near resistance, away from support
                if side == 'LONG':
                    if support_distance_pct < 1.5 and resistance_distance_pct > 3:
                        score += 25  # Near support, good room to resistance
                    elif support_distance_pct < 3:
                        score += 15  # Reasonably close to support
                    elif resistance_distance_pct < 1.5:
                        score += 0   # Too close to resistance - bad LONG
                    else:
                        score += 8
                else:  # SHORT
                    if resistance_distance_pct < 1.5 and support_distance_pct > 3:
                        score += 25  # Near resistance, good room to support
                    elif resistance_distance_pct < 3:
                        score += 15  # Reasonably close to resistance
                    elif support_distance_pct < 1.5:
                        score += 0   # Too close to support - bad SHORT
                    else:
                        score += 8

                logger.info(
                    f"📊 PA S/R Check: {symbol} | Support: {support_distance_pct:.1f}% away | "
                    f"Resistance: {resistance_distance_pct:.1f}% away | {side} score: +{score - mtf_score}"
                )
            else:
                score += 8  # No PA data, neutral score
        except Exception as e:
            logger.warning(f"PA analysis failed for {symbol}: {e}")
            score += 8  # Fallback score

        # ═══════════════════════════════════════════════════════════════════
        # 3. RSI ANALYSIS (up to 15 points)
        # ═══════════════════════════════════════════════════════════════════
        rsi = indicators.get('rsi', 50)

        if side == 'LONG':
            if rsi < 30:  # Strongly oversold
                score += 15
            elif rsi < 40:
                score += 12
            elif rsi < 50:
                score += 8
            elif rsi > 70:  # Overbought - bad for LONG
                score += 0
            else:
                score += 5
        else:  # SHORT
            if rsi > 70:  # Strongly overbought
                score += 15
            elif rsi > 60:
                score += 12
            elif rsi > 50:
                score += 8
            elif rsi < 30:  # Oversold - bad for SHORT
                score += 0
            else:
                score += 5

        # ═══════════════════════════════════════════════════════════════════
        # 4. VOLUME CONFIRMATION (up to 15 points)
        # ═══════════════════════════════════════════════════════════════════
        volume_ratio = indicators.get('volume_ratio', 1)

        if volume_ratio > 2.5:
            score += 15  # Very high volume
        elif volume_ratio > 1.8:
            score += 12
        elif volume_ratio > 1.3:
            score += 8
        elif volume_ratio > 1.0:
            score += 5
        else:
            score += 2  # Low volume - weak signal

        # ═══════════════════════════════════════════════════════════════════
        # 5. MOMENTUM + ATR (up to 15 points)
        # ═══════════════════════════════════════════════════════════════════
        price_change = indicators.get('1h', {}).get('price_change_pct', 0)
        atr_pct = indicators.get('15m', {}).get('atr_percent', 0)

        # Momentum should match our direction
        if side == 'LONG' and price_change > 0.5:
            score += 8
        elif side == 'SHORT' and price_change < -0.5:
            score += 8
        elif abs(price_change) < 0.2:
            score += 2  # No clear momentum

        # ATR shows volatility potential
        if atr_pct > 1.5:
            score += 7  # Good volatility for profit
        elif atr_pct > 0.8:
            score += 5
        elif atr_pct > 0.4:
            score += 3
        else:
            score += 1  # Low volatility

        # ═══════════════════════════════════════════════════════════════════
        # 6. SIGNAL TRIGGER QUALITY (up to 10 points)
        # ═══════════════════════════════════════════════════════════════════
        trigger = signal.get('trigger', '')

        if 'EMA_CROSS' in trigger:
            score += 10  # Most reliable
        elif 'VOLUME_SPIKE' in trigger:
            score += 8
        elif 'MOMENTUM' in trigger:
            score += 6
        else:
            score += 3

        # ═══════════════════════════════════════════════════════════════════
        # 🆕 7. ENHANCED INDICATORS (up to 25 points) - v4.4.0
        # BB Squeeze, EMA Stack, ADX, Divergences
        # ═══════════════════════════════════════════════════════════════════
        enhanced_score = 0
        ohlcv = market_data.get('ohlcv', [])

        if ohlcv and len(ohlcv) >= 50:
            try:
                enhanced_data = calculate_enhanced_indicators(ohlcv)

                # BB Squeeze Analysis (up to 8 points)
                bb_squeeze = enhanced_data.get('bb_squeeze', {})
                squeeze_signal = bb_squeeze.get('signal', 'NEUTRAL')
                if squeeze_signal in ['STRONG_BUY', 'BUY'] and side == 'LONG':
                    enhanced_score += 8
                elif squeeze_signal in ['STRONG_SELL', 'SELL'] and side == 'SHORT':
                    enhanced_score += 8
                elif bb_squeeze.get('squeeze_on', False):
                    enhanced_score += 3  # Squeeze building - potential
                else:
                    enhanced_score += 2

                # EMA Stack Analysis (up to 8 points)
                ema_stack = enhanced_data.get('ema_stack', {})
                stack_type = ema_stack.get('stack_type', 'unknown')
                if stack_type == 'bullish' and side == 'LONG':
                    enhanced_score += 8
                elif stack_type == 'bearish' and side == 'SHORT':
                    enhanced_score += 8
                elif 'forming' in stack_type and ema_stack.get('stack_score', 0) >= 75:
                    enhanced_score += 5
                elif stack_type == 'tangled':
                    enhanced_score += 1
                else:
                    enhanced_score += 2

                # ADX Trend Strength (up to 5 points)
                adx = enhanced_data.get('adx', {})
                adx_value = adx.get('adx', 20)
                trend_direction = adx.get('trend_direction', 'neutral')

                if adx_value >= 25:
                    if (trend_direction == 'bullish' and side == 'LONG') or \
                       (trend_direction == 'bearish' and side == 'SHORT'):
                        enhanced_score += 5  # Strong trend in our direction
                    else:
                        enhanced_score += 1  # Counter-trend
                else:
                    enhanced_score += 2  # Weak trend

                # Divergence Detection (up to 4 points)
                rsi_div = enhanced_data.get('rsi_divergence', {})
                macd_div = enhanced_data.get('macd_divergence', {})

                if rsi_div.get('has_divergence'):
                    div_type = rsi_div.get('type', '')
                    if (div_type == 'bullish' and side == 'LONG') or \
                       (div_type == 'bearish' and side == 'SHORT'):
                        enhanced_score += 2

                if macd_div.get('has_divergence'):
                    div_type = macd_div.get('divergence_type', '')
                    if ('bullish' in str(div_type) and side == 'LONG') or \
                       ('bearish' in str(div_type) and side == 'SHORT'):
                        enhanced_score += 2

                logger.info(
                    f"🔬 Enhanced Indicators: {symbol} | BB: {squeeze_signal} | "
                    f"EMA: {stack_type} | ADX: {adx_value:.1f} ({trend_direction}) | "
                    f"Score: +{enhanced_score}"
                )

            except Exception as e:
                logger.warning(f"⚠️ Enhanced indicators failed: {e}")
                enhanced_score = 6  # Fallback

        else:
            enhanced_score = 6  # No OHLCV data

        score += enhanced_score

        # ═══════════════════════════════════════════════════════════════════
        # 🆕 8. ADVANCED INDICATORS (up to 20 points) - v4.5.0
        # VWAP, Stochastic RSI, Williams %R, CMF, Fibonacci
        # ═══════════════════════════════════════════════════════════════════
        advanced_score = 0

        if ohlcv and len(ohlcv) >= 50:
            try:
                advanced_data = calculate_advanced_indicators(ohlcv)

                # VWAP Analysis (up to 5 points)
                vwap_data = advanced_data.get('vwap', {})
                vwap_position = vwap_data.get('position', 'unknown')
                if side == 'LONG':
                    if vwap_position in ['slightly_above', 'at_vwap']:
                        advanced_score += 5  # Perfect long setup
                    elif vwap_position in ['above', 'far_below']:
                        advanced_score += 3
                    else:
                        advanced_score += 1
                else:
                    if vwap_position in ['slightly_below', 'at_vwap']:
                        advanced_score += 5  # Perfect short setup
                    elif vwap_position in ['below', 'far_above']:
                        advanced_score += 3
                    else:
                        advanced_score += 1

                # Stochastic RSI (up to 4 points)
                stoch_data = advanced_data.get('stoch_rsi', {})
                stoch_signal = stoch_data.get('signal', 'NEUTRAL')
                stoch_zone = stoch_data.get('zone', 'neutral')
                if (stoch_signal in ['STRONG_BUY', 'BUY'] and side == 'LONG') or \
                   (stoch_signal in ['STRONG_SELL', 'SELL'] and side == 'SHORT'):
                    advanced_score += 4
                elif (stoch_zone == 'oversold' and side == 'LONG') or \
                     (stoch_zone == 'overbought' and side == 'SHORT'):
                    advanced_score += 3
                else:
                    advanced_score += 1

                # Chaikin Money Flow (up to 4 points)
                cmf_data = advanced_data.get('cmf', {})
                cmf_pressure = cmf_data.get('pressure', 'balanced')
                if (cmf_pressure == 'strong_buying' and side == 'LONG') or \
                   (cmf_pressure == 'strong_selling' and side == 'SHORT'):
                    advanced_score += 4
                elif (cmf_pressure == 'buying' and side == 'LONG') or \
                     (cmf_pressure == 'selling' and side == 'SHORT'):
                    advanced_score += 3
                else:
                    advanced_score += 1

                # Williams %R (up to 3 points)
                wr_data = advanced_data.get('williams_r', {})
                wr_signal = wr_data.get('signal', 'NEUTRAL')
                if (wr_signal in ['STRONG_BUY', 'BUY'] and side == 'LONG') or \
                   (wr_signal in ['STRONG_SELL', 'SELL'] and side == 'SHORT'):
                    advanced_score += 3
                else:
                    advanced_score += 1

                # Fibonacci (up to 4 points)
                fib_data = advanced_data.get('fibonacci', {})
                fib_signal = fib_data.get('signal', 'NEUTRAL')
                fib_level = fib_data.get('current_level')
                if (fib_signal == 'STRONG_BUY' and side == 'LONG') or \
                   (fib_signal == 'STRONG_SELL' and side == 'SHORT'):
                    advanced_score += 4
                elif fib_level in ['38.2', '50.0', '61.8']:
                    advanced_score += 3
                else:
                    advanced_score += 1

                logger.info(
                    f"🔬 Advanced Indicators: {symbol} | VWAP: {vwap_position} | "
                    f"StochRSI: {stoch_zone} | CMF: {cmf_pressure} | "
                    f"Fib: {fib_level} | Score: +{advanced_score}"
                )

            except Exception as e:
                logger.warning(f"⚠️ Advanced indicators failed: {e}")
                advanced_score = 8  # Fallback

        else:
            advanced_score = 8  # No OHLCV data

        score += advanced_score

        final_score = min(100, score)

        logger.info(
            f"📊 Confluence Score: {symbol} {side} = {final_score:.1f}/100 | "
            f"MTF: {mtf_score}/20 | RSI: {rsi:.0f} | Vol: {volume_ratio:.1f}x | "
            f"ATR: {atr_pct:.1f}% | Enhanced: {enhanced_score}/25 | Advanced: {advanced_score}/20"
        )

        return final_score

    async def _execute_instant_trade(
        self,
        symbol: str,
        side: str,
        signal: Dict[str, Any],
        market_data: Dict[str, Any],
        confluence_score: float
    ) -> bool:
        """
        Execute trade instantly.

        Uses simplified parameters based on signal data.
        """
        try:
            notifier = get_notifier()
            risk_manager = get_risk_manager()
            executor = get_trade_executor()

            current_price = market_data['current_price']

            # Calculate stop-loss and take-profit
            # Use tighter stops for instant entries (we caught the trend early)
            if side == 'LONG':
                stop_loss_percent = 2.0  # 2% stop
                take_profit_percent = 4.0  # 4% target (2:1 R/R)
                stop_loss_price = current_price * (1 - stop_loss_percent / 100)
                take_profit_price = current_price * (1 + take_profit_percent / 100)
            else:  # SHORT
                stop_loss_percent = 2.0
                take_profit_percent = 4.0
                stop_loss_price = current_price * (1 + stop_loss_percent / 100)
                take_profit_price = current_price * (1 - take_profit_percent / 100)

            # Build trade params
            trade_params = {
                'symbol': symbol,
                'side': side,
                'confidence': signal['strength'] / 100,
                'leverage': self.settings.max_leverage,  # 🔥 FIX: Add required leverage param
                'take_profit_price': take_profit_price,
                'stop_loss_price': stop_loss_price,
                'stop_loss_percent': stop_loss_percent,
                'current_price': current_price,
                'market_breadth': None,
                'source': 'REALTIME_INSTANT'
            }

            # Validate with risk manager
            validation = await risk_manager.validate_trade(trade_params)

            if not validation['approved']:
                logger.warning(f"Trade rejected by risk manager: {validation['reason']}")
                await notifier.send_alert(
                    'warning',
                    f"⚡ Instant trade rejected:\n{symbol} {side}\n\n"
                    f"Reason: {validation['reason']}"
                )
                return False

            # Use adjusted params if provided
            if validation.get('adjusted_params'):
                trade_params = validation['adjusted_params']

            # Build analysis dict for executor
            analysis = {
                'action': 'buy' if side == 'LONG' else 'sell',
                'side': side,
                'confidence': signal['strength'] / 100,
                'stop_loss_percent': stop_loss_percent,
                'reasoning': f"Real-time {signal['trigger']} signal | Confluence: {confluence_score:.1f}",
                'model_name': 'REALTIME_DETECTOR'
            }

            # Execute!
            success = await executor.open_position(trade_params, analysis, market_data)

            if success:
                await notifier.send_alert(
                    'success',
                    f"⚡ INSTANT TRADE EXECUTED!\n\n"
                    f"Symbol: {symbol}\n"
                    f"Side: {side}\n"
                    f"Entry: ${current_price:.4f}\n"
                    f"Stop Loss: ${stop_loss_price:.4f} ({stop_loss_percent}%)\n"
                    f"Take Profit: ${take_profit_price:.4f} ({take_profit_percent}%)\n"
                    f"Confluence: {confluence_score:.1f}/100\n"
                    f"Trigger: {signal['trigger']}\n\n"
                    f"🚀 Position opened in < 1 second!"
                )

            return success

        except Exception as e:
            logger.error(f"Error executing instant trade: {e}", exc_info=True)
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            **self.stats,
            'is_running': self.is_running,
            'market_trend': self.market_trend.get('direction', 'UNKNOWN'),
            'btc_change': f"{self.market_trend.get('btc_change_pct', 0):.2f}%",
            'symbols_in_cooldown': len([
                s for s, ts in self.recent_executions.items()
                if datetime.now().timestamp() - ts < self.execution_cooldown
            ])
        }


# Singleton instance
_signal_handler: Optional[RealtimeSignalHandler] = None


async def get_signal_handler() -> RealtimeSignalHandler:
    """Get or create signal handler instance."""
    global _signal_handler
    if _signal_handler is None:
        _signal_handler = RealtimeSignalHandler()
    return _signal_handler
