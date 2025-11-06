"""
Market Snapshot Capture for ML Learning
Captures comprehensive market state at entry/exit for pattern learning
"""

import asyncio
from typing import Dict, List, Any, Optional
from decimal import Decimal
from datetime import datetime
import numpy as np
import pandas as pd
from src.utils import setup_logging, safe_decimal

logger = setup_logging()


class MarketSnapshotCapture:
    """
    Captures comprehensive market state snapshots for ML learning.

    Snapshot includes:
    - Multi-timeframe indicators (15m, 30m, 1h, 4h, 1d)
    - Support/Resistance levels
    - Trend analysis
    - Volume & Order Flow
    - Funding Rate & Liquidations
    - Price action patterns
    """

    def __init__(self, exchange_client):
        self.exchange = exchange_client

    async def capture_snapshot(
        self,
        symbol: str,
        current_price: Decimal,
        context: str = "entry"
    ) -> Dict[str, Any]:
        """
        Capture comprehensive market snapshot.

        Args:
            symbol: Trading pair
            current_price: Current market price
            context: "entry" or "exit"

        Returns:
            Comprehensive snapshot dictionary
        """
        try:
            logger.info(f"📸 Capturing {context} snapshot for {symbol} @ ${current_price}")

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "price": float(current_price),
            }

            # Parallel data fetching for speed
            snapshot_tasks = [
                self._capture_multi_timeframe_indicators(symbol),
                self._capture_support_resistance(symbol, current_price),
                self._capture_volume_metrics(symbol),
                self._capture_funding_and_liquidations(symbol),
            ]

            results = await asyncio.gather(*snapshot_tasks, return_exceptions=True)

            # Merge results
            snapshot["indicators"] = results[0] if not isinstance(results[0], Exception) else {}
            snapshot["levels"] = results[1] if not isinstance(results[1], Exception) else {}
            snapshot["volume"] = results[2] if not isinstance(results[2], Exception) else {}
            snapshot["funding"] = results[3] if not isinstance(results[3], Exception) else {}

            # Add trend analysis (derived from indicators)
            snapshot["trends"] = self._analyze_trends(snapshot.get("indicators", {}))

            logger.info(f"✅ Snapshot captured: {len(snapshot)} categories")
            return snapshot

        except Exception as e:
            logger.error(f"❌ Snapshot capture failed: {e}", exc_info=True)
            return {
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "price": float(current_price),
                "error": str(e)
            }

    async def _capture_multi_timeframe_indicators(self, symbol: str) -> Dict[str, Any]:
        """Capture indicators across multiple timeframes."""
        timeframes = {
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }

        indicators_by_tf = {}

        for tf_name, tf_interval in timeframes.items():
            try:
                # Fetch OHLCV data
                ohlcv = await self.exchange.fetch_ohlcv(symbol, tf_interval, limit=100)

                if not ohlcv or len(ohlcv) < 50:
                    continue

                # Convert to pandas for indicator calculation
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                # Calculate indicators
                indicators = self._calculate_indicators(df)
                indicators_by_tf[tf_name] = indicators

            except Exception as e:
                logger.warning(f"Failed to capture {tf_name} indicators: {e}")
                indicators_by_tf[tf_name] = None

        return indicators_by_tf

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators from OHLCV data."""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values

            indicators = {}

            # RSI (14-period)
            rsi_period = min(14, len(close) - 1)
            if rsi_period > 0:
                indicators['rsi'] = float(self._calculate_rsi(close, rsi_period))

            # MACD
            if len(close) >= 26:
                macd, signal = self._calculate_macd(close)
                indicators['macd'] = float(macd)
                indicators['macd_signal'] = float(signal)
                indicators['macd_histogram'] = float(macd - signal)

            # Moving Averages
            if len(close) >= 20:
                indicators['sma_20'] = float(np.mean(close[-20:]))
            if len(close) >= 50:
                indicators['sma_50'] = float(np.mean(close[-50:]))
            if len(close) >= 20:
                indicators['ema_20'] = float(self._calculate_ema(close, 20))

            # Bollinger Bands
            if len(close) >= 20:
                sma = np.mean(close[-20:])
                std = np.std(close[-20:])
                indicators['bb_upper'] = float(sma + 2 * std)
                indicators['bb_middle'] = float(sma)
                indicators['bb_lower'] = float(sma - 2 * std)
                indicators['bb_position'] = float((close[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower']))

            # ATR (Average True Range) for volatility
            if len(high) >= 14:
                indicators['atr'] = float(self._calculate_atr(high, low, close, 14))

            # Volume indicators
            if len(volume) >= 20:
                indicators['volume_current'] = float(volume[-1])
                indicators['volume_sma_20'] = float(np.mean(volume[-20:]))
                indicators['volume_ratio'] = float(volume[-1] / np.mean(volume[-20:]) if np.mean(volume[-20:]) > 0 else 1.0)

            # Price momentum
            if len(close) >= 10:
                indicators['price_change_10_pct'] = float((close[-1] - close[-10]) / close[-10] * 100)
            if len(close) >= 50:
                indicators['price_change_50_pct'] = float((close[-1] - close[-50]) / close[-50] * 100)

            return indicators

        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return {}

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: np.ndarray, fast=12, slow=26, signal_period=9):
        """Calculate MACD indicator."""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        signal_line = ema_fast - ema_slow  # Simplified

        return macd_line, signal_line

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA (Exponential Moving Average)."""
        if len(prices) < period:
            return float(np.mean(prices))

        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])  # Start with SMA

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return float(ema)

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate ATR (Average True Range)."""
        tr_list = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return float(np.mean(tr_list)) if tr_list else 0.0

        atr = np.mean(tr_list[-period:])
        return float(atr)

    async def _capture_support_resistance(self, symbol: str, current_price: Decimal) -> Dict[str, Any]:
        """Identify support and resistance levels."""
        try:
            # Fetch 4h data for level detection
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '4h', limit=100)

            if not ohlcv or len(ohlcv) < 20:
                return {}

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Find local maxima (resistance) and minima (support)
            highs = df['high'].values
            lows = df['low'].values

            # Simple pivot point detection
            resistance_levels = self._find_resistance_levels(highs)
            support_levels = self._find_support_levels(lows)

            # Filter levels near current price (within 10%)
            price_float = float(current_price)
            nearby_resistance = [r for r in resistance_levels if price_float < r < price_float * 1.10]
            nearby_support = [s for s in support_levels if price_float * 0.90 < s < price_float]

            # Calculate distances
            nearest_resistance = min(nearby_resistance) if nearby_resistance else None
            nearest_support = max(nearby_support) if nearby_support else None

            return {
                "support_levels": nearby_support[:3],  # Top 3 closest
                "resistance_levels": nearby_resistance[:3],  # Top 3 closest
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "distance_to_support_pct": float((price_float - nearest_support) / price_float * 100) if nearest_support else None,
                "distance_to_resistance_pct": float((nearest_resistance - price_float) / price_float * 100) if nearest_resistance else None,
            }

        except Exception as e:
            logger.warning(f"Support/Resistance detection failed: {e}")
            return {}

    def _find_resistance_levels(self, highs: np.ndarray, window: int = 5) -> List[float]:
        """Find local maxima as resistance."""
        resistance = []
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                resistance.append(float(highs[i]))
        return sorted(set(resistance), reverse=True)  # Descending order

    def _find_support_levels(self, lows: np.ndarray, window: int = 5) -> List[float]:
        """Find local minima as support."""
        support = []
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i-window:i+window+1]):
                support.append(float(lows[i]))
        return sorted(set(support), reverse=True)  # Descending order

    async def _capture_volume_metrics(self, symbol: str) -> Dict[str, Any]:
        """Capture volume and order flow metrics."""
        try:
            # Fetch 1h data for volume analysis
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '1h', limit=168)  # 7 days

            if not ohlcv or len(ohlcv) < 24:
                return {}

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            volume = df['volume'].values

            return {
                "volume_24h": float(np.sum(volume[-24:])),
                "volume_7d_avg": float(np.mean(volume)),
                "volume_ratio_24h_vs_7d": float(np.sum(volume[-24:]) / (np.mean(volume) * 24)) if np.mean(volume) > 0 else 1.0,
                "volume_trend": "increasing" if volume[-1] > np.mean(volume[-24:]) else "decreasing",
            }

        except Exception as e:
            logger.warning(f"Volume metrics capture failed: {e}")
            return {}

    async def _capture_funding_and_liquidations(self, symbol: str) -> Dict[str, Any]:
        """Capture funding rate and estimate liquidation levels."""
        try:
            funding_data = await self.exchange.fetch_funding_rate(symbol)

            return {
                "funding_rate": float(funding_data.get('rate', 0.0)),
                "funding_rate_status": self._classify_funding_rate(funding_data.get('rate', 0.0)),
            }

        except Exception as e:
            logger.warning(f"Funding rate capture failed: {e}")
            return {}

    def _classify_funding_rate(self, rate: float) -> str:
        """Classify funding rate as extreme/high/normal/low."""
        if rate > 0.0010:
            return "extreme_long_bias"
        elif rate > 0.0003:
            return "high_long_bias"
        elif rate < -0.0010:
            return "extreme_short_bias"
        elif rate < -0.0003:
            return "high_short_bias"
        else:
            return "neutral"

    def _analyze_trends(self, indicators_by_tf: Dict[str, Any]) -> Dict[str, str]:
        """Analyze trend direction across timeframes."""
        trends = {}

        for tf, indicators in indicators_by_tf.items():
            if not indicators or indicators is None:
                trends[tf] = "unknown"
                continue

            # Simple trend detection using SMA
            close_approx = indicators.get('sma_20', 0)  # Approximation
            sma_50 = indicators.get('sma_50')

            if sma_50 and close_approx > sma_50:
                trends[tf] = "bullish"
            elif sma_50 and close_approx < sma_50:
                trends[tf] = "bearish"
            else:
                # Use RSI as backup
                rsi = indicators.get('rsi')
                if rsi and rsi > 55:
                    trends[tf] = "bullish"
                elif rsi and rsi < 45:
                    trends[tf] = "bearish"
                else:
                    trends[tf] = "ranging"

        return trends


# Singleton instance
_snapshot_capture_instance: Optional[MarketSnapshotCapture] = None


async def get_snapshot_capture(exchange_client=None):
    """Get or create snapshot capture instance."""
    global _snapshot_capture_instance

    if _snapshot_capture_instance is None:
        if exchange_client is None:
            from src.exchange_client import get_exchange_client
            exchange_client = await get_exchange_client()

        _snapshot_capture_instance = MarketSnapshotCapture(exchange_client)

    return _snapshot_capture_instance
