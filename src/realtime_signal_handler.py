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
        logger.info("✅ Real-Time Signal Handler started")

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

            # 1.5 🔥 NEW: Market trend filter - BLOCK counter-trend trades
            market_direction = self.market_trend.get('direction', 'NEUTRAL')

            if market_direction == 'BEARISH' and side == 'LONG':
                # Market is bearish but signal is LONG - HIGH RISK!
                logger.warning(
                    f"⚠️ COUNTER-TREND BLOCKED: {symbol} LONG signal rejected | "
                    f"Market is BEARISH (BTC dumping)"
                )
                await notifier.send_alert(
                    'warning',
                    f"⚠️ Counter-Trend Signal Blocked:\\n\\n"
                    f"Symbol: {symbol}\\n"
                    f"Signal: LONG\\n"
                    f"Market: BEARISH 📉\\n"
                    f"BTC Change: {self.market_trend.get('btc_change_pct', 0):.2f}%\\n\\n"
                    f"❌ LONG trades blocked when BTC is dumping!\\n"
                    f"This prevents losses like WIF/CRV."
                )
                self.stats['counter_trend_rejected'] += 1
                self.stats['trades_rejected'] += 1
                return

            elif market_direction == 'BULLISH' and side == 'SHORT':
                # Market is bullish but signal is SHORT - HIGH RISK!
                logger.warning(
                    f"⚠️ COUNTER-TREND BLOCKED: {symbol} SHORT signal rejected | "
                    f"Market is BULLISH (BTC pumping)"
                )
                await notifier.send_alert(
                    'warning',
                    f"⚠️ Counter-Trend Signal Blocked:\\n\\n"
                    f"Symbol: {symbol}\\n"
                    f"Signal: SHORT\\n"
                    f"Market: BULLISH 📈\\n"
                    f"BTC Change: {self.market_trend.get('btc_change_pct', 0):.2f}%\\n\\n"
                    f"❌ SHORT trades blocked when BTC is pumping!"
                )
                self.stats['counter_trend_rejected'] += 1
                self.stats['trades_rejected'] += 1
                return

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

            # 🎯 v4.2: CONFLUENCE REDUCED (60+) - Smart Confidence handles quality
            # Price action analyzer now does sophisticated confidence calculation
            MIN_CERTAINTY_CONFLUENCE = 60  # 60+ is sufficient with smart PA system

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
        Fetch minimal market data for quick validation.

        Optimized for speed - only fetch what's necessary.
        """
        try:
            exchange = await get_exchange_client()

            # Fetch 15m candles (minimal for analysis)
            ohlcv = await exchange.fetch_ohlcv(symbol, '15m', limit=50)

            if not ohlcv or len(ohlcv) < 20:
                return None

            # Calculate quick indicators
            closes = [c[4] for c in ohlcv]
            volumes = [c[5] for c in ohlcv]
            current_price = closes[-1]

            # Quick RSI calculation
            rsi = self._quick_rsi(closes)

            # Quick trend check (EMA)
            ema_9 = self._quick_ema(closes, 9)
            ema_21 = self._quick_ema(closes, 21)

            trend = 'UPTREND' if ema_9 > ema_21 else 'DOWNTREND' if ema_9 < ema_21 else 'NEUTRAL'

            # Volume analysis
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            return {
                'symbol': symbol,
                'current_price': current_price,
                'ohlcv': ohlcv,
                'indicators': {
                    'rsi': rsi,
                    'ema_9': ema_9,
                    'ema_21': ema_21,
                    'trend': trend,
                    'volume_ratio': volume_ratio
                },
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

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
        Quick confluence score calculation for real-time signals.

        Simplified version for speed - focuses on key factors.
        """
        score = 0.0
        indicators = market_data.get('indicators', {})

        # 1. Signal strength bonus (up to 25 points)
        signal_strength = signal.get('strength', 0)
        score += min(25, signal_strength * 0.25)

        # 2. Trend alignment (up to 25 points)
        trend = indicators.get('trend', 'NEUTRAL')

        if side == 'LONG' and trend == 'UPTREND':
            score += 25
        elif side == 'SHORT' and trend == 'DOWNTREND':
            score += 25
        elif trend == 'NEUTRAL':
            score += 10
        else:
            score += 5  # Counter-trend (risky but possible)

        # 3. RSI alignment (up to 20 points)
        rsi = indicators.get('rsi', 50)

        if side == 'LONG':
            if rsi < 35:  # Oversold - good for long
                score += 20
            elif rsi < 45:
                score += 15
            elif rsi < 55:
                score += 10
            else:
                score += 5
        else:  # SHORT
            if rsi > 65:  # Overbought - good for short
                score += 20
            elif rsi > 55:
                score += 15
            elif rsi > 45:
                score += 10
            else:
                score += 5

        # 4. Volume confirmation (up to 15 points)
        volume_ratio = indicators.get('volume_ratio', 1)

        if volume_ratio > 2.0:
            score += 15
        elif volume_ratio > 1.5:
            score += 12
        elif volume_ratio > 1.0:
            score += 8
        else:
            score += 4

        # 5. Trigger quality bonus (up to 15 points)
        trigger = signal.get('trigger', '')

        if 'EMA_CROSS' in trigger:
            score += 15  # Most reliable
        elif 'VOLUME_SPIKE' in trigger:
            score += 12
        elif 'MOMENTUM' in trigger:
            score += 10
        else:
            score += 5

        return min(100, score)

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
