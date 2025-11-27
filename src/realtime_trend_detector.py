"""
Real-Time Trend Detector - Instant trend detection via WebSocket.

This module provides INSTANT trend detection by analyzing live candlestick data
from Binance WebSocket streams. When a trend starts, it triggers immediate entry.

Key Features:
- Real-time kline (candlestick) stream analysis
- Multi-timeframe trend detection (1m, 5m, 15m)
- Instant signal generation (sub-second latency)
- EMA crossover detection
- RSI divergence detection
- Volume spike detection
- Momentum shift detection

Architecture:
- WebSocket streams for all trading symbols (108 coins)
- Background analysis tasks running continuously
- Callback-based signal generation
- Direct integration with trade executor
"""

import asyncio
import json
import websockets
from typing import Dict, Any, Optional, Callable, List, Set
from decimal import Decimal
from datetime import datetime
from collections import deque
import numpy as np
from src.config import get_settings
from src.utils import setup_logging

logger = setup_logging()


class RealtimeTrendDetector:
    """
    Real-time trend detection using WebSocket kline streams.

    Detects trend changes INSTANTLY and triggers callbacks for immediate entry.
    """

    def __init__(self):
        self.settings = get_settings()
        self.is_running = False
        self._tasks: Set[asyncio.Task] = set()

        # WebSocket connections (one per symbol batch for efficiency)
        self.ws_connections: Dict[str, websockets.WebSocketClientProtocol] = {}

        # Kline data cache: {symbol: {timeframe: deque of candles}}
        # Keep last 50 candles per timeframe for analysis
        self.kline_cache: Dict[str, Dict[str, deque]] = {}

        # Signal callbacks
        self.signal_callbacks: List[Callable] = []

        # Trend state tracking: {symbol: {'trend': 'UP'/'DOWN'/'NEUTRAL', 'strength': 0-100}}
        self.trend_states: Dict[str, Dict[str, Any]] = {}

        # Signal cooldown: {symbol: last_signal_timestamp}
        # Prevent multiple signals for same coin within 60 seconds
        self.signal_cooldown: Dict[str, float] = {}
        self.cooldown_seconds = 60

        # Binance Futures WebSocket
        self.ws_base_url = "wss://fstream.binance.com"

        # Timeframes to monitor (1m for instant detection, 5m/15m for confirmation)
        self.timeframes = ['1m', '5m', '15m']

        # Detection parameters
        self.ema_fast = 9
        self.ema_slow = 21
        self.rsi_period = 14
        self.volume_spike_threshold = 2.0  # 2x average volume
        self.momentum_threshold = 0.5  # 0.5% price change threshold

    async def start(self, symbols: List[str]):
        """
        Start real-time trend detection for given symbols.

        Args:
            symbols: List of trading symbols (e.g., ['BTC/USDT:USDT', 'ETH/USDT:USDT'])
        """
        if self.is_running:
            logger.warning("Realtime trend detector already running")
            return

        self.is_running = True
        logger.info(f"🚀 Starting Real-Time Trend Detector for {len(symbols)} symbols...")

        # Initialize caches for all symbols
        for symbol in symbols:
            self.kline_cache[symbol] = {tf: deque(maxlen=50) for tf in self.timeframes}
            self.trend_states[symbol] = {
                'trend': 'NEUTRAL',
                'strength': 0,
                'last_update': None
            }

        # Connect to combined stream (more efficient than individual connections)
        # Binance allows up to 200 streams per connection
        # We'll batch symbols into groups of ~60 (60 symbols * 3 timeframes = 180 streams)
        batch_size = 60
        symbol_batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]

        for batch_idx, batch in enumerate(symbol_batches):
            task = asyncio.create_task(self._combined_stream(batch, batch_idx))
            self._tasks.add(task)
            task.add_done_callback(lambda t: self._tasks.discard(t))

        logger.info(f"✅ Real-Time Trend Detector started ({len(symbol_batches)} WebSocket connections)")

    async def stop(self):
        """Stop all WebSocket connections and background tasks."""
        self.is_running = False
        logger.info("Stopping Real-Time Trend Detector...")

        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Close all connections
        for batch_id, ws in list(self.ws_connections.items()):
            try:
                await ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket batch {batch_id}: {e}")

        self.ws_connections.clear()
        self.kline_cache.clear()
        self.trend_states.clear()
        logger.info("✅ Real-Time Trend Detector stopped")

    def register_signal_callback(self, callback: Callable):
        """
        Register a callback for trend signals.

        Callback signature: async def callback(signal: Dict[str, Any])
        Signal format:
        {
            'symbol': 'BTC/USDT:USDT',
            'side': 'LONG' or 'SHORT',
            'trend': 'UP' or 'DOWN',
            'strength': 0-100,
            'trigger': 'EMA_CROSS' / 'MOMENTUM_SHIFT' / 'VOLUME_SPIKE',
            'price': Decimal,
            'timestamp': datetime
        }
        """
        self.signal_callbacks.append(callback)
        logger.info(f"Registered signal callback (total: {len(self.signal_callbacks)})")

    def _to_binance_symbol(self, ccxt_symbol: str) -> str:
        """Convert CCXT format to Binance format: 'BTC/USDT:USDT' -> 'btcusdt'"""
        base = ccxt_symbol.split('/')[0].lower()
        quote = ccxt_symbol.split('/')[1].split(':')[0].lower()
        return base + quote

    def _from_binance_symbol(self, binance_symbol: str) -> str:
        """Convert Binance format to CCXT format: 'btcusdt' -> 'BTC/USDT:USDT'"""
        if binance_symbol.endswith('usdt'):
            base = binance_symbol[:-4].upper()
            return f"{base}/USDT:USDT"
        return binance_symbol.upper()

    async def _combined_stream(self, symbols: List[str], batch_idx: int):
        """
        Connect to Binance combined stream for multiple symbols and timeframes.

        More efficient than individual connections - single connection handles all.
        """
        reconnect_delay = 1.0

        while self.is_running:
            try:
                # Build stream names: symbol@kline_interval
                streams = []
                for symbol in symbols:
                    binance_symbol = self._to_binance_symbol(symbol)
                    for tf in self.timeframes:
                        streams.append(f"{binance_symbol}@kline_{tf}")

                # Combined stream URL
                stream_param = "/".join(streams)
                ws_url = f"{self.ws_base_url}/stream?streams={stream_param}"

                logger.info(f"🔌 Connecting to combined kline stream (batch {batch_idx}, {len(streams)} streams)...")

                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=10 * 1024 * 1024  # 10MB max message size
                ) as websocket:
                    self.ws_connections[f"batch_{batch_idx}"] = websocket
                    reconnect_delay = 1.0
                    logger.info(f"✅ Combined stream connected (batch {batch_idx})")

                    while self.is_running:
                        try:
                            message = await asyncio.wait_for(
                                websocket.recv(),
                                timeout=30.0
                            )

                            data = json.loads(message)
                            await self._handle_kline_update(data)

                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            try:
                                pong = await websocket.ping()
                                await asyncio.wait_for(pong, timeout=10)
                            except:
                                logger.warning(f"Ping failed for batch {batch_idx}, reconnecting...")
                                break

                        except websockets.exceptions.ConnectionClosed:
                            logger.warning(f"WebSocket closed for batch {batch_idx}, reconnecting...")
                            break

            except Exception as e:
                logger.error(f"WebSocket error for batch {batch_idx}: {e}")

            # Reconnect with exponential backoff
            if self.is_running:
                logger.info(f"Reconnecting batch {batch_idx} in {reconnect_delay:.1f}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60.0)

    async def _handle_kline_update(self, data: Dict[str, Any]):
        """
        Process kline update from WebSocket.

        Binance kline format:
        {
            "stream": "btcusdt@kline_1m",
            "data": {
                "e": "kline",
                "E": 1672515782136,
                "s": "BTCUSDT",
                "k": {
                    "t": 1672515780000,  # Kline start time
                    "T": 1672515839999,  # Kline close time
                    "s": "BTCUSDT",
                    "i": "1m",           # Interval
                    "o": "42000.00",     # Open
                    "c": "42100.00",     # Close
                    "h": "42150.00",     # High
                    "l": "41950.00",     # Low
                    "v": "100.5",        # Volume
                    "x": false,          # Is this kline closed?
                    ...
                }
            }
        }
        """
        try:
            if 'data' not in data:
                return

            kline_data = data['data']
            if kline_data.get('e') != 'kline':
                return

            k = kline_data['k']
            binance_symbol = k['s'].lower()
            symbol = self._from_binance_symbol(binance_symbol)
            timeframe = k['i']

            if symbol not in self.kline_cache:
                return

            # Create candle dict
            candle = {
                'timestamp': k['t'],
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v']),
                'closed': k['x']
            }

            # Update cache (only add closed candles to history, always update current)
            cache = self.kline_cache[symbol][timeframe]

            if k['x']:  # Candle closed - add to history
                # Check if this candle timestamp already exists
                if len(cache) == 0 or cache[-1]['timestamp'] != candle['timestamp']:
                    cache.append(candle)

                    # Analyze trend on closed candle
                    await self._analyze_trend(symbol, timeframe, candle)
            else:
                # Update current (unclosed) candle for real-time analysis
                await self._analyze_realtime(symbol, timeframe, candle)

        except Exception as e:
            logger.error(f"Error processing kline update: {e}", exc_info=True)

    async def _analyze_realtime(self, symbol: str, timeframe: str, current_candle: Dict):
        """
        Analyze real-time (unclosed) candle for instant signals.

        This is where we catch trend starts IMMEDIATELY.
        """
        # Only analyze 1m timeframe for real-time signals
        if timeframe != '1m':
            return

        cache = self.kline_cache[symbol]['1m']
        if len(cache) < self.ema_slow + 5:
            return  # Not enough data

        # Get closes including current candle
        closes = [c['close'] for c in cache] + [current_candle['close']]
        volumes = [c['volume'] for c in cache] + [current_candle['volume']]

        # Calculate EMAs
        ema_fast = self._calculate_ema(closes, self.ema_fast)
        ema_slow = self._calculate_ema(closes, self.ema_slow)

        if ema_fast is None or ema_slow is None:
            return

        # Previous EMAs (for crossover detection)
        prev_closes = closes[:-1]
        prev_ema_fast = self._calculate_ema(prev_closes, self.ema_fast)
        prev_ema_slow = self._calculate_ema(prev_closes, self.ema_slow)

        if prev_ema_fast is None or prev_ema_slow is None:
            return

        current_price = current_candle['close']

        # === TREND DETECTION SIGNALS ===
        signal = None
        trigger = None

        # 1. EMA Crossover (most reliable)
        if prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow:
            # Bullish crossover - LONG signal
            signal = 'LONG'
            trigger = 'EMA_CROSS_UP'
            strength = min(100, int((ema_fast - ema_slow) / ema_slow * 10000))

        elif prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow:
            # Bearish crossover - SHORT signal
            signal = 'SHORT'
            trigger = 'EMA_CROSS_DOWN'
            strength = min(100, int((ema_slow - ema_fast) / ema_fast * 10000))

        # 2. Volume Spike + Direction
        if signal is None and len(volumes) >= 20:
            avg_volume = np.mean(volumes[-20:-1])  # Last 20 candles excluding current
            current_volume = volumes[-1]

            if current_volume > avg_volume * self.volume_spike_threshold:
                # Volume spike detected - check direction
                price_change = (current_price - closes[-2]) / closes[-2] * 100

                if price_change > self.momentum_threshold:
                    signal = 'LONG'
                    trigger = 'VOLUME_SPIKE_UP'
                    strength = min(100, int(price_change * 20))
                elif price_change < -self.momentum_threshold:
                    signal = 'SHORT'
                    trigger = 'VOLUME_SPIKE_DOWN'
                    strength = min(100, int(abs(price_change) * 20))

        # 3. Strong Momentum Shift
        if signal is None and len(closes) >= 5:
            # Check last 5 candles for momentum
            recent_change = (current_price - closes[-5]) / closes[-5] * 100

            if recent_change > 1.5:  # 1.5% up in 5 candles
                signal = 'LONG'
                trigger = 'MOMENTUM_SHIFT_UP'
                strength = min(100, int(recent_change * 15))
            elif recent_change < -1.5:  # 1.5% down in 5 candles
                signal = 'SHORT'
                trigger = 'MOMENTUM_SHIFT_DOWN'
                strength = min(100, int(abs(recent_change) * 15))

        # === EMIT SIGNAL ===
        if signal:
            await self._emit_signal(
                symbol=symbol,
                side=signal,
                trend='UP' if signal == 'LONG' else 'DOWN',
                strength=strength,
                trigger=trigger,
                price=Decimal(str(current_price))
            )

    async def _analyze_trend(self, symbol: str, timeframe: str, closed_candle: Dict):
        """
        Analyze closed candle for trend confirmation.

        This provides additional confirmation for signals.
        """
        # Update trend state based on closed candles
        cache = self.kline_cache[symbol][timeframe]
        if len(cache) < self.ema_slow:
            return

        closes = [c['close'] for c in cache]

        # Calculate EMAs
        ema_fast = self._calculate_ema(closes, self.ema_fast)
        ema_slow = self._calculate_ema(closes, self.ema_slow)

        if ema_fast is None or ema_slow is None:
            return

        # Update trend state
        if ema_fast > ema_slow:
            trend = 'UP'
            strength = min(100, int((ema_fast - ema_slow) / ema_slow * 5000))
        elif ema_fast < ema_slow:
            trend = 'DOWN'
            strength = min(100, int((ema_slow - ema_fast) / ema_fast * 5000))
        else:
            trend = 'NEUTRAL'
            strength = 0

        self.trend_states[symbol] = {
            'trend': trend,
            'strength': strength,
            'timeframe': timeframe,
            'last_update': datetime.now()
        }

    def _calculate_ema(self, data: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return None

        multiplier = 2 / (period + 1)
        ema = data[0]

        for price in data[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    async def _emit_signal(
        self,
        symbol: str,
        side: str,
        trend: str,
        strength: int,
        trigger: str,
        price: Decimal
    ):
        """
        Emit trading signal to registered callbacks.

        Includes cooldown check to prevent signal spam.
        """
        # Check cooldown
        current_time = datetime.now().timestamp()
        last_signal = self.signal_cooldown.get(symbol, 0)

        if current_time - last_signal < self.cooldown_seconds:
            logger.debug(f"Signal cooldown active for {symbol} ({trigger})")
            return

        # Update cooldown
        self.signal_cooldown[symbol] = current_time

        # Create signal
        signal = {
            'symbol': symbol,
            'side': side,
            'trend': trend,
            'strength': strength,
            'trigger': trigger,
            'price': price,
            'timestamp': datetime.now(),
            'source': 'REALTIME_WEBSOCKET'
        }

        logger.info(
            f"🎯 REALTIME SIGNAL: {symbol} {side} | "
            f"Trigger: {trigger} | Strength: {strength} | "
            f"Price: ${float(price):.4f}"
        )

        # Call all registered callbacks
        for callback in self.signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}", exc_info=True)

    def get_trend_state(self, symbol: str) -> Dict[str, Any]:
        """Get current trend state for a symbol."""
        return self.trend_states.get(symbol, {
            'trend': 'NEUTRAL',
            'strength': 0,
            'last_update': None
        })

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            'is_running': self.is_running,
            'symbols_tracked': len(self.kline_cache),
            'active_connections': len(self.ws_connections),
            'signals_in_cooldown': sum(
                1 for ts in self.signal_cooldown.values()
                if datetime.now().timestamp() - ts < self.cooldown_seconds
            ),
            'trend_states': {
                'UP': sum(1 for s in self.trend_states.values() if s['trend'] == 'UP'),
                'DOWN': sum(1 for s in self.trend_states.values() if s['trend'] == 'DOWN'),
                'NEUTRAL': sum(1 for s in self.trend_states.values() if s['trend'] == 'NEUTRAL')
            }
        }


# Singleton instance
_trend_detector: Optional[RealtimeTrendDetector] = None


async def get_trend_detector() -> RealtimeTrendDetector:
    """Get or create trend detector instance."""
    global _trend_detector
    if _trend_detector is None:
        _trend_detector = RealtimeTrendDetector()
    return _trend_detector
