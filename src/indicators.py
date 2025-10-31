"""
Technical indicators calculation module.
Calculates RSI, MACD, Bollinger Bands, Moving Averages, and other indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from ta import trend, momentum, volatility, volume as vol_indicators


def calculate_indicators(ohlcv_data: List[List]) -> Dict[str, Any]:
    """
    Calculate comprehensive technical indicators from OHLCV data.

    Args:
        ohlcv_data: List of [timestamp, open, high, low, close, volume]

    Returns:
        Dict with all calculated indicators
    """
    if not ohlcv_data or len(ohlcv_data) < 20:
        return get_default_indicators()

    # Convert to DataFrame
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill any NaN values (using modern pandas syntax)
    df.ffill(inplace=True)

    try:
        indicators = {}

        # Current price
        indicators['close'] = float(df['close'].iloc[-1])

        # RSI (14 period)
        indicators['rsi'] = calculate_rsi(df['close'], period=14)

        # MACD
        macd_data = calculate_macd(df['close'])
        indicators['macd'] = macd_data['macd']
        indicators['macd_signal'] = macd_data['signal']
        indicators['macd_histogram'] = macd_data['histogram']

        # Bollinger Bands (20 period, 2 std)
        bb_data = calculate_bollinger_bands(df['close'], period=20, std_dev=2)
        indicators['bb_upper'] = bb_data['upper']
        indicators['bb_middle'] = bb_data['middle']
        indicators['bb_lower'] = bb_data['lower']

        # Moving Averages
        indicators['sma_20'] = calculate_sma(df['close'], period=20)
        indicators['sma_50'] = calculate_sma(df['close'], period=50)
        indicators['ema_12'] = calculate_ema(df['close'], period=12)
        indicators['ema_26'] = calculate_ema(df['close'], period=26)

        # Volume indicators
        indicators['volume'] = float(df['volume'].iloc[-1])
        indicators['volume_sma'] = float(df['volume'].rolling(window=20).mean().iloc[-1])
        indicators['volume_trend'] = 'high' if indicators['volume'] > indicators['volume_sma'] * 1.5 else 'normal'

        # ATR (Average True Range) for volatility
        indicators['atr'] = calculate_atr(df)

        # Trend determination
        indicators['trend'] = determine_trend(df['close'])

        return indicators

    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return get_default_indicators()


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index)."""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1])
    except:
        return 50.0  # Neutral RSI


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    try:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            'macd': float(macd_line.iloc[-1]),
            'signal': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1])
        }
    except:
        return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
    """Calculate Bollinger Bands."""
    try:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return {
            'upper': float(upper_band.iloc[-1]),
            'middle': float(sma.iloc[-1]),
            'lower': float(lower_band.iloc[-1])
        }
    except:
        current_price = float(prices.iloc[-1])
        return {
            'upper': current_price * 1.02,
            'middle': current_price,
            'lower': current_price * 0.98
        }


def calculate_sma(prices: pd.Series, period: int) -> float:
    """Calculate Simple Moving Average."""
    try:
        if len(prices) < period:
            return float(prices.mean())
        sma = prices.rolling(window=period).mean()
        return float(sma.iloc[-1])
    except:
        return float(prices.iloc[-1])


def calculate_ema(prices: pd.Series, period: int) -> float:
    """Calculate Exponential Moving Average."""
    try:
        if len(prices) < period:
            return float(prices.mean())
        ema = prices.ewm(span=period, adjust=False).mean()
        return float(ema.iloc[-1])
    except:
        return float(prices.iloc[-1])


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range (volatility indicator)."""
    try:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        atr = true_range.rolling(window=period).mean()
        return float(atr.iloc[-1])
    except:
        return 0.0


def determine_trend(prices: pd.Series) -> str:
    """
    Determine overall trend: 'uptrend', 'downtrend', or 'sideways'.

    Uses SMA comparison and price momentum.
    """
    try:
        if len(prices) < 50:
            return 'unknown'

        sma_20 = prices.rolling(window=20).mean().iloc[-1]
        sma_50 = prices.rolling(window=50).mean().iloc[-1]
        current_price = prices.iloc[-1]

        # Calculate recent momentum
        price_change = (current_price - prices.iloc[-10]) / prices.iloc[-10]

        if sma_20 > sma_50 and current_price > sma_20 and price_change > 0.01:
            return 'uptrend'
        elif sma_20 < sma_50 and current_price < sma_20 and price_change < -0.01:
            return 'downtrend'
        else:
            return 'sideways'

    except:
        return 'unknown'


def detect_market_regime(ohlcv_data: List[List]) -> str:
    """
    Detect market regime: 'TRENDING', 'RANGING', or 'VOLATILE'.

    Args:
        ohlcv_data: List of OHLCV candles

    Returns:
        Market regime string
    """
    if not ohlcv_data or len(ohlcv_data) < 20:
        return 'UNKNOWN'

    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    try:
        # Calculate ATR for volatility
        atr = calculate_atr(df)
        avg_price = df['close'].mean()
        atr_percent = (atr / avg_price) * 100

        # Calculate trend strength using ADX-like logic
        price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
        abs_change = abs(price_change)

        # High volatility regime
        if atr_percent > 3.0:
            return 'VOLATILE'

        # Trending regime (strong directional movement)
        elif abs_change > 0.05:  # 5% change
            return 'TRENDING'

        # Ranging regime (sideways movement)
        else:
            return 'RANGING'

    except Exception as e:
        return 'UNKNOWN'


def detect_support_resistance_levels(ohlcv_data: List[List], current_price: float) -> Dict[str, Any]:
    """
    Detect key support and resistance levels using pivot points and swing highs/lows.

    Institutional traders watch these levels for:
    - Entry/exit points
    - Stop-loss placement
    - Liquidation cluster zones

    Args:
        ohlcv_data: List of [timestamp, open, high, low, close, volume]
        current_price: Current market price

    Returns:
        Dict with support/resistance levels and distance percentages
    """
    if not ohlcv_data or len(ohlcv_data) < 20:
        return {
            'nearest_support': current_price * 0.98,
            'nearest_resistance': current_price * 1.02,
            'support_distance_pct': 2.0,
            'resistance_distance_pct': 2.0,
            'key_levels': []
        }

    try:
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Find swing highs (resistance) and swing lows (support)
        # A swing high is a high that's higher than N highs before and after it
        swing_period = 5

        swing_highs = []
        swing_lows = []

        for i in range(swing_period, len(df) - swing_period):
            # Check if this is a swing high
            is_swing_high = True
            is_swing_low = True

            for j in range(1, swing_period + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i - j] or df['high'].iloc[i] <= df['high'].iloc[i + j]:
                    is_swing_high = False
                if df['low'].iloc[i] >= df['low'].iloc[i - j] or df['low'].iloc[i] >= df['low'].iloc[i + j]:
                    is_swing_low = False

            if is_swing_high:
                swing_highs.append(float(df['high'].iloc[i]))
            if is_swing_low:
                swing_lows.append(float(df['low'].iloc[i]))

        # Also add recent highs/lows
        recent_high = float(df['high'].tail(20).max())
        recent_low = float(df['low'].tail(20).min())

        if recent_high not in swing_highs:
            swing_highs.append(recent_high)
        if recent_low not in swing_lows:
            swing_lows.append(recent_low)

        # Find nearest support and resistance
        supports_below = [s for s in swing_lows if s < current_price]
        resistances_above = [r for r in swing_highs if r > current_price]

        nearest_support = max(supports_below) if supports_below else current_price * 0.97
        nearest_resistance = min(resistances_above) if resistances_above else current_price * 1.03

        # Calculate distances
        support_distance_pct = abs(current_price - nearest_support) / current_price * 100
        resistance_distance_pct = abs(nearest_resistance - current_price) / current_price * 100

        # Combine all key levels
        all_levels = sorted(set(swing_highs + swing_lows))

        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance_pct': support_distance_pct,
            'resistance_distance_pct': resistance_distance_pct,
            'key_levels': all_levels[-10:],  # Last 10 key levels
            'swing_highs': swing_highs[-5:],  # Last 5 swing highs
            'swing_lows': swing_lows[-5:]  # Last 5 swing lows
        }

    except Exception as e:
        return {
            'nearest_support': current_price * 0.98,
            'nearest_resistance': current_price * 1.02,
            'support_distance_pct': 2.0,
            'resistance_distance_pct': 2.0,
            'key_levels': []
        }


def calculate_volume_profile(ohlcv_data: List[List]) -> Dict[str, Any]:
    """
    Calculate Volume Profile to identify high-volume price levels (value areas).

    Institutional concept: Price tends to return to high-volume nodes (fair value).
    - POC (Point of Control): Price level with highest volume
    - Value Area High/Low: 70% of volume distribution

    Args:
        ohlcv_data: List of OHLCV candles

    Returns:
        Dict with POC, value areas, and volume distribution
    """
    if not ohlcv_data or len(ohlcv_data) < 20:
        return {
            'poc': 0.0,  # Point of Control
            'value_area_high': 0.0,
            'value_area_low': 0.0,
            'high_volume_nodes': []
        }

    try:
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Create price bins (1% intervals)
        min_price = df['close'].min()
        max_price = df['close'].max()
        num_bins = 50

        price_bins = np.linspace(min_price, max_price, num_bins)
        volume_at_price = np.zeros(num_bins - 1)

        # Distribute volume to price bins
        for i in range(len(df)):
            price = df['close'].iloc[i]
            volume = df['volume'].iloc[i]

            # Find which bin this price falls into
            bin_idx = np.digitize(price, price_bins) - 1
            if 0 <= bin_idx < len(volume_at_price):
                volume_at_price[bin_idx] += volume

        # Find Point of Control (highest volume)
        poc_idx = np.argmax(volume_at_price)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2

        # Find Value Area (70% of total volume)
        total_volume = np.sum(volume_at_price)
        target_volume = total_volume * 0.70

        # Start from POC and expand
        va_indices = [poc_idx]
        accumulated_volume = volume_at_price[poc_idx]

        left_idx = poc_idx - 1
        right_idx = poc_idx + 1

        while accumulated_volume < target_volume and (left_idx >= 0 or right_idx < len(volume_at_price)):
            left_vol = volume_at_price[left_idx] if left_idx >= 0 else 0
            right_vol = volume_at_price[right_idx] if right_idx < len(volume_at_price) else 0

            if left_vol > right_vol and left_idx >= 0:
                va_indices.append(left_idx)
                accumulated_volume += left_vol
                left_idx -= 1
            elif right_idx < len(volume_at_price):
                va_indices.append(right_idx)
                accumulated_volume += right_vol
                right_idx += 1
            else:
                break

        va_indices = sorted(va_indices)
        value_area_low = price_bins[va_indices[0]]
        value_area_high = price_bins[va_indices[-1] + 1]

        # Find high volume nodes (above 80th percentile)
        volume_threshold = np.percentile(volume_at_price, 80)
        high_volume_nodes = []

        for i, vol in enumerate(volume_at_price):
            if vol >= volume_threshold:
                node_price = (price_bins[i] + price_bins[i + 1]) / 2
                high_volume_nodes.append(float(node_price))

        return {
            'poc': float(poc_price),
            'value_area_high': float(value_area_high),
            'value_area_low': float(value_area_low),
            'high_volume_nodes': high_volume_nodes
        }

    except Exception as e:
        return {
            'poc': 0.0,
            'value_area_high': 0.0,
            'value_area_low': 0.0,
            'high_volume_nodes': []
        }


def calculate_fibonacci_levels(ohlcv_data: List[List], current_price: float) -> Dict[str, Any]:
    """
    Calculate Fibonacci retracement and extension levels.

    Institutional traders use Fibonacci levels for:
    - Identifying potential reversal zones
    - Setting profit targets
    - Finding confluence with other indicators

    Key Levels: 0.236, 0.382, 0.5, 0.618, 0.786 (retracements)
                1.272, 1.618 (extensions)

    Args:
        ohlcv_data: List of OHLCV candles
        current_price: Current market price

    Returns:
        Dict with Fibonacci levels and nearest level
    """
    if not ohlcv_data or len(ohlcv_data) < 20:
        return {
            'trend': 'unknown',
            'swing_high': current_price,
            'swing_low': current_price,
            'fib_levels': {},
            'nearest_fib_level': 0.5,
            'nearest_fib_price': current_price
        }

    try:
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Find recent swing high and swing low
        swing_high = float(df['high'].tail(50).max())
        swing_low = float(df['low'].tail(50).min())

        # Determine trend direction
        recent_close = float(df['close'].iloc[-1])
        trend = 'uptrend' if recent_close > (swing_high + swing_low) / 2 else 'downtrend'

        # Calculate Fibonacci levels
        diff = swing_high - swing_low

        if trend == 'uptrend':
            # Retracement from high to low
            fib_levels = {
                '0.0': swing_high,
                '0.236': swing_high - (diff * 0.236),
                '0.382': swing_high - (diff * 0.382),
                '0.5': swing_high - (diff * 0.5),
                '0.618': swing_high - (diff * 0.618),
                '0.786': swing_high - (diff * 0.786),
                '1.0': swing_low,
                '1.272': swing_low - (diff * 0.272),
                '1.618': swing_low - (diff * 0.618)
            }
        else:
            # Retracement from low to high
            fib_levels = {
                '0.0': swing_low,
                '0.236': swing_low + (diff * 0.236),
                '0.382': swing_low + (diff * 0.382),
                '0.5': swing_low + (diff * 0.5),
                '0.618': swing_low + (diff * 0.618),
                '0.786': swing_low + (diff * 0.786),
                '1.0': swing_high,
                '1.272': swing_high + (diff * 0.272),
                '1.618': swing_high + (diff * 0.618)
            }

        # Find nearest Fibonacci level to current price
        distances = {level: abs(price - current_price) for level, price in fib_levels.items()}
        nearest_level = min(distances, key=distances.get)
        nearest_price = fib_levels[nearest_level]

        return {
            'trend': trend,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'fib_levels': {k: float(v) for k, v in fib_levels.items()},
            'nearest_fib_level': float(nearest_level),
            'nearest_fib_price': float(nearest_price),
            'nearest_distance_pct': abs(current_price - nearest_price) / current_price * 100
        }

    except Exception as e:
        return {
            'trend': 'unknown',
            'swing_high': current_price,
            'swing_low': current_price,
            'fib_levels': {},
            'nearest_fib_level': 0.5,
            'nearest_fib_price': current_price
        }


def analyze_funding_rate_trend(funding_rate_history: List[float]) -> Dict[str, Any]:
    """
    Analyze funding rate trends to detect overleveraged positions.

    Crypto-specific edge:
    - Positive funding: Longs pay shorts → Overleveraged longs (SHORT opportunity)
    - Negative funding: Shorts pay longs → Overleveraged shorts (LONG opportunity)
    - Trend matters: Rising funding = increasing leverage = squeeze risk

    Args:
        funding_rate_history: List of recent funding rates (as decimals)

    Returns:
        Dict with trend analysis and trading implications
    """
    if not funding_rate_history or len(funding_rate_history) < 3:
        return {
            'current_rate': 0.0,
            'avg_rate': 0.0,
            'trend': 'neutral',
            'trading_implication': 'neutral',
            'risk_level': 'low'
        }

    try:
        current_rate = funding_rate_history[-1]
        avg_rate = np.mean(funding_rate_history)

        # Calculate trend (linear regression slope)
        x = np.arange(len(funding_rate_history))
        y = np.array(funding_rate_history)

        # Simple linear fit
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        # Determine trend
        if slope > 0.0001:
            trend = 'rising'
        elif slope < -0.0001:
            trend = 'falling'
        else:
            trend = 'stable'

        # Trading implications
        if current_rate > 0.05:  # 0.05% = high positive funding
            if trend == 'rising':
                trading_implication = 'strong_short_opportunity'
                risk_level = 'high'
            else:
                trading_implication = 'short_opportunity'
                risk_level = 'moderate'
        elif current_rate < -0.05:  # High negative funding
            if trend == 'falling':
                trading_implication = 'strong_long_opportunity'
                risk_level = 'high'
            else:
                trading_implication = 'long_opportunity'
                risk_level = 'moderate'
        else:
            trading_implication = 'neutral'
            risk_level = 'low'

        return {
            'current_rate': float(current_rate),
            'avg_rate': float(avg_rate),
            'trend': trend,
            'trading_implication': trading_implication,
            'risk_level': risk_level,
            'slope': float(slope)
        }

    except Exception as e:
        return {
            'current_rate': 0.0,
            'avg_rate': 0.0,
            'trend': 'neutral',
            'trading_implication': 'neutral',
            'risk_level': 'low'
        }


def get_default_indicators() -> Dict[str, Any]:
    """Return default indicators when calculation fails."""
    return {
        'close': 0.0,
        'rsi': 50.0,
        'macd': 0.0,
        'macd_signal': 0.0,
        'macd_histogram': 0.0,
        'bb_upper': 0.0,
        'bb_middle': 0.0,
        'bb_lower': 0.0,
        'sma_20': 0.0,
        'sma_50': 0.0,
        'ema_12': 0.0,
        'ema_26': 0.0,
        'volume': 0.0,
        'volume_sma': 0.0,
        'volume_trend': 'normal',
        'atr': 0.0,
        'trend': 'unknown'
    }
