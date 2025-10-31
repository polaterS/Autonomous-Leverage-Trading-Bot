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


def detect_divergences(ohlcv_data: List[List]) -> Dict[str, Any]:
    """
    Detect bullish and bearish divergences between price and indicators.

    PROFESSIONAL DIVERGENCE DETECTION:
    - Bullish Divergence: Price makes lower low, RSI/MACD makes higher low → REVERSAL UP
    - Bearish Divergence: Price makes higher high, RSI/MACD makes lower high → REVERSAL DOWN

    These are among THE STRONGEST reversal signals in technical analysis!

    Args:
        ohlcv_data: List of OHLCV candles

    Returns:
        Dict with divergence signals and strength
    """
    if not ohlcv_data or len(ohlcv_data) < 30:
        return {
            'has_divergence': False,
            'type': None,
            'strength': 0,
            'indicator': None,
            'details': 'Insufficient data'
        }

    try:
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate RSI and MACD
        rsi_values = []
        for i in range(14, len(df)):
            prices_slice = df['close'].iloc[i-14:i+1]
            delta = prices_slice.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(float(rsi.iloc[-1]))

        # Get recent price lows and highs (last 20 candles)
        recent_data = df.tail(20)
        recent_rsi = rsi_values[-20:] if len(rsi_values) >= 20 else rsi_values

        if len(recent_rsi) < 10:
            return {
                'has_divergence': False,
                'type': None,
                'strength': 0,
                'indicator': None,
                'details': 'Insufficient RSI data'
            }

        # Find price lows and highs
        price_lows = []
        price_highs = []
        rsi_lows = []
        rsi_highs = []

        for i in range(2, len(recent_data) - 2):
            # Local low (lower than 2 candles before and after)
            if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i-2] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+2]):
                price_lows.append((i, float(recent_data['low'].iloc[i])))
                if i < len(recent_rsi):
                    rsi_lows.append((i, recent_rsi[i]))

            # Local high
            if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i-2] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+2]):
                price_highs.append((i, float(recent_data['high'].iloc[i])))
                if i < len(recent_rsi):
                    rsi_highs.append((i, recent_rsi[i]))

        # Check for BULLISH divergence (price lower low, RSI higher low)
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            last_price_low = price_lows[-1][1]
            prev_price_low = price_lows[-2][1]
            last_rsi_low = rsi_lows[-1][1]
            prev_rsi_low = rsi_lows[-2][1]

            # Bullish divergence: price makes lower low, RSI makes higher low
            if last_price_low < prev_price_low and last_rsi_low > prev_rsi_low:
                strength = min((last_rsi_low - prev_rsi_low) / 10, 1.0)  # Normalize to 0-1
                return {
                    'has_divergence': True,
                    'type': 'bullish',
                    'strength': float(strength),
                    'indicator': 'RSI',
                    'details': f'Price: {prev_price_low:.4f}→{last_price_low:.4f}, RSI: {prev_rsi_low:.1f}→{last_rsi_low:.1f}'
                }

        # Check for BEARISH divergence (price higher high, RSI lower high)
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            last_price_high = price_highs[-1][1]
            prev_price_high = price_highs[-2][1]
            last_rsi_high = rsi_highs[-1][1]
            prev_rsi_high = rsi_highs[-2][1]

            # Bearish divergence: price makes higher high, RSI makes lower high
            if last_price_high > prev_price_high and last_rsi_high < prev_rsi_high:
                strength = min((prev_rsi_high - last_rsi_high) / 10, 1.0)
                return {
                    'has_divergence': True,
                    'type': 'bearish',
                    'strength': float(strength),
                    'indicator': 'RSI',
                    'details': f'Price: {prev_price_high:.4f}→{last_price_high:.4f}, RSI: {prev_rsi_high:.1f}→{last_rsi_high:.1f}'
                }

        return {
            'has_divergence': False,
            'type': None,
            'strength': 0,
            'indicator': None,
            'details': 'No divergence detected'
        }

    except Exception as e:
        return {
            'has_divergence': False,
            'type': None,
            'strength': 0,
            'indicator': None,
            'details': f'Error: {e}'
        }


def analyze_order_flow(order_book: Dict[str, Any], recent_trades: List[Dict]) -> Dict[str, Any]:
    """
    Analyze order flow and market depth for institutional positioning.

    ORDER FLOW = Where big money is positioned!
    - Bid/Ask imbalance: More bids = buyers in control (bullish)
    - Large order walls: Support/Resistance levels where institutions sit
    - Trade flow: Are large trades buying or selling?

    Args:
        order_book: {'bids': [[price, size], ...], 'asks': [[price, size], ...]}
        recent_trades: List of recent trades with size and side

    Returns:
        Dict with order flow analysis
    """
    try:
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return {
                'imbalance': 0.0,
                'imbalance_ratio': 1.0,
                'signal': 'neutral',
                'large_bid_wall': None,
                'large_ask_wall': None,
                'buy_pressure': 0.5
            }

        bids = order_book.get('bids', [])[:20]  # Top 20 bids
        asks = order_book.get('asks', [])[:20]  # Top 20 asks

        if not bids or not asks:
            return {
                'imbalance': 0.0,
                'imbalance_ratio': 1.0,
                'signal': 'neutral',
                'large_bid_wall': None,
                'large_ask_wall': None,
                'buy_pressure': 0.5
            }

        # Calculate total bid and ask volume
        total_bid_volume = sum(float(bid[1]) for bid in bids)
        total_ask_volume = sum(float(ask[1]) for ask in asks)

        # Bid/Ask imbalance
        if total_ask_volume > 0:
            imbalance_ratio = total_bid_volume / total_ask_volume
        else:
            imbalance_ratio = 1.0

        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) * 100

        # Determine signal
        if imbalance > 20:  # More than 20% more bids
            signal = 'strong_bullish'
        elif imbalance > 10:
            signal = 'bullish'
        elif imbalance < -20:
            signal = 'strong_bearish'
        elif imbalance < -10:
            signal = 'bearish'
        else:
            signal = 'neutral'

        # Detect large order walls (institutional orders)
        avg_bid_size = total_bid_volume / len(bids) if bids else 0
        avg_ask_size = total_ask_volume / len(asks) if asks else 0

        large_bid_wall = None
        large_ask_wall = None

        # Find bid walls (3x larger than average)
        for bid in bids[:5]:  # Check top 5 levels
            if float(bid[1]) > avg_bid_size * 3:
                large_bid_wall = {'price': float(bid[0]), 'size': float(bid[1])}
                break

        # Find ask walls
        for ask in asks[:5]:
            if float(ask[1]) > avg_ask_size * 3:
                large_ask_wall = {'price': float(ask[0]), 'size': float(ask[1])}
                break

        # Analyze recent trades for buy/sell pressure
        buy_pressure = 0.5  # Default neutral
        if recent_trades and len(recent_trades) > 0:
            buy_volume = sum(float(t.get('amount', 0)) for t in recent_trades if t.get('side') == 'buy')
            sell_volume = sum(float(t.get('amount', 0)) for t in recent_trades if t.get('side') == 'sell')
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                buy_pressure = buy_volume / total_volume

        return {
            'imbalance': float(imbalance),
            'imbalance_ratio': float(imbalance_ratio),
            'signal': signal,
            'large_bid_wall': large_bid_wall,
            'large_ask_wall': large_ask_wall,
            'buy_pressure': float(buy_pressure),
            'total_bid_volume': float(total_bid_volume),
            'total_ask_volume': float(total_ask_volume)
        }

    except Exception as e:
        return {
            'imbalance': 0.0,
            'imbalance_ratio': 1.0,
            'signal': 'neutral',
            'large_bid_wall': None,
            'large_ask_wall': None,
            'buy_pressure': 0.5
        }


def detect_smart_money_concepts(ohlcv_data: List[List], current_price: float) -> Dict[str, Any]:
    """
    Detect Smart Money Concepts (SMC) - follow institutional traders, not retail!

    SMART MONEY CONCEPTS:
    1. ORDER BLOCKS: Where institutions entered (high-volume rejection zones)
    2. FAIR VALUE GAPS (FVG): Price imbalances that tend to be filled
    3. LIQUIDITY GRABS: Stop hunts before reversals (fake breakouts)

    Args:
        ohlcv_data: List of OHLCV candles
        current_price: Current market price

    Returns:
        Dict with Smart Money signals
    """
    if not ohlcv_data or len(ohlcv_data) < 20:
        return {
            'order_blocks': [],
            'fair_value_gaps': [],
            'liquidity_grab_detected': False,
            'market_structure': 'unknown',
            'smart_money_signal': 'neutral'
        }

    try:
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        order_blocks = []
        fair_value_gaps = []

        # Detect ORDER BLOCKS (last 20 candles)
        # Order block = High volume candle followed by strong rejection
        recent_df = df.tail(20)
        avg_volume = recent_df['volume'].mean()

        for i in range(1, len(recent_df) - 1):
            candle_volume = recent_df['volume'].iloc[i]
            candle_close = recent_df['close'].iloc[i]
            candle_open = recent_df['open'].iloc[i]
            next_close = recent_df['close'].iloc[i + 1]

            # Bullish order block: High volume down candle followed by reversal up
            if (candle_volume > avg_volume * 1.5 and
                candle_close < candle_open and  # Red candle
                next_close > candle_close):      # Reversal

                order_blocks.append({
                    'type': 'bullish',
                    'price_high': float(recent_df['high'].iloc[i]),
                    'price_low': float(recent_df['low'].iloc[i]),
                    'distance_pct': abs(current_price - float(recent_df['low'].iloc[i])) / current_price * 100
                })

            # Bearish order block: High volume up candle followed by reversal down
            elif (candle_volume > avg_volume * 1.5 and
                  candle_close > candle_open and  # Green candle
                  next_close < candle_close):      # Reversal

                order_blocks.append({
                    'type': 'bearish',
                    'price_high': float(recent_df['high'].iloc[i]),
                    'price_low': float(recent_df['low'].iloc[i]),
                    'distance_pct': abs(current_price - float(recent_df['high'].iloc[i])) / current_price * 100
                })

        # Detect FAIR VALUE GAPS (FVG)
        # FVG = Gap between candles that hasn't been filled (imbalance)
        for i in range(len(recent_df) - 2):
            candle1_high = recent_df['high'].iloc[i]
            candle1_low = recent_df['low'].iloc[i]
            candle3_high = recent_df['high'].iloc[i + 2]
            candle3_low = recent_df['low'].iloc[i + 2]

            # Bullish FVG: Gap up (candle 3 low > candle 1 high)
            if candle3_low > candle1_high:
                gap_size = (candle3_low - candle1_high) / candle1_high * 100
                if gap_size > 0.5:  # At least 0.5% gap
                    fair_value_gaps.append({
                        'type': 'bullish',
                        'gap_top': float(candle3_low),
                        'gap_bottom': float(candle1_high),
                        'gap_size_pct': float(gap_size),
                        'filled': current_price < candle3_low
                    })

            # Bearish FVG: Gap down (candle 3 high < candle 1 low)
            elif candle3_high < candle1_low:
                gap_size = (candle1_low - candle3_high) / candle3_high * 100
                if gap_size > 0.5:
                    fair_value_gaps.append({
                        'type': 'bearish',
                        'gap_top': float(candle1_low),
                        'gap_bottom': float(candle3_high),
                        'gap_size_pct': float(gap_size),
                        'filled': current_price > candle1_low
                    })

        # Detect LIQUIDITY GRAB (stop hunt before reversal)
        liquidity_grab_detected = False
        last_10 = df.tail(10)

        # Check if price recently spiked beyond previous high/low then reversed (stop hunt)
        if len(last_10) >= 5:
            recent_high = last_10['high'].iloc[-5:].max()
            previous_high = last_10['high'].iloc[-10:-5].max()
            current_close = last_10['close'].iloc[-1]

            # Bearish liquidity grab: Spike above previous high, then drop
            if recent_high > previous_high * 1.005 and current_close < recent_high * 0.995:
                liquidity_grab_detected = True

        # Determine overall smart money signal
        bullish_signals = sum(1 for ob in order_blocks if ob['type'] == 'bullish')
        bearish_signals = sum(1 for ob in order_blocks if ob['type'] == 'bearish')

        if bullish_signals > bearish_signals:
            smart_money_signal = 'bullish'
        elif bearish_signals > bullish_signals:
            smart_money_signal = 'bearish'
        else:
            smart_money_signal = 'neutral'

        return {
            'order_blocks': order_blocks[-3:],  # Last 3 order blocks
            'fair_value_gaps': fair_value_gaps[-3:],  # Last 3 FVGs
            'liquidity_grab_detected': liquidity_grab_detected,
            'smart_money_signal': smart_money_signal,
            'order_block_count': len(order_blocks)
        }

    except Exception as e:
        return {
            'order_blocks': [],
            'fair_value_gaps': [],
            'liquidity_grab_detected': False,
            'smart_money_signal': 'neutral',
            'order_block_count': 0
        }


def calculate_volatility_bands(ohlcv_data: List[List], current_price: float) -> Dict[str, Any]:
    """
    Calculate ATR-based dynamic volatility bands for adaptive risk management.

    VOLATILITY BANDS = Adaptive Stop-Loss and Entry Ranges
    - High volatility → Wider stops needed, avoid tight entries
    - Low volatility → Tighter stops, better for scalping
    - Volatility breakouts → Strong trend signals

    Args:
        ohlcv_data: List of OHLCV candles
        current_price: Current market price

    Returns:
        Dict with volatility analysis
    """
    if not ohlcv_data or len(ohlcv_data) < 20:
        return {
            'atr': 0.0,
            'atr_percent': 0.0,
            'volatility_level': 'unknown',
            'upper_band': current_price * 1.02,
            'lower_band': current_price * 0.98,
            'breakout_detected': False
        }

    try:
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate ATR (14-period)
        atr = calculate_atr(df, period=14)
        atr_percent = (atr / current_price) * 100

        # Determine volatility level
        if atr_percent < 1.0:
            volatility_level = 'very_low'
            multiplier = 1.5
        elif atr_percent < 2.0:
            volatility_level = 'low'
            multiplier = 2.0
        elif atr_percent < 3.0:
            volatility_level = 'normal'
            multiplier = 2.5
        elif atr_percent < 5.0:
            volatility_level = 'high'
            multiplier = 3.0
        else:
            volatility_level = 'extreme'
            multiplier = 4.0

        # Calculate volatility bands (current_price ± ATR * multiplier)
        upper_band = current_price + (atr * multiplier)
        lower_band = current_price - (atr * multiplier)

        # Detect volatility breakout (ATR increasing rapidly)
        recent_atr = []
        for i in range(max(0, len(df) - 20), len(df)):
            if i >= 14:
                period_atr = calculate_atr(df.iloc[:i+1], period=14)
                recent_atr.append(period_atr)

        breakout_detected = False
        if len(recent_atr) >= 5:
            atr_change = (recent_atr[-1] - recent_atr[-5]) / recent_atr[-5]
            if atr_change > 0.3:  # 30% increase in ATR
                breakout_detected = True

        return {
            'atr': float(atr),
            'atr_percent': float(atr_percent),
            'volatility_level': volatility_level,
            'upper_band': float(upper_band),
            'lower_band': float(lower_band),
            'band_width_pct': float((upper_band - lower_band) / current_price * 100),
            'breakout_detected': breakout_detected,
            'recommended_stop_pct': float(min(atr_percent * 2, 10.0))  # 2x ATR or max 10%
        }

    except Exception as e:
        return {
            'atr': 0.0,
            'atr_percent': 0.0,
            'volatility_level': 'unknown',
            'upper_band': current_price * 1.02,
            'lower_band': current_price * 0.98,
            'breakout_detected': False
        }


def calculate_confluence_score(market_data: Dict[str, Any], side: str) -> Dict[str, Any]:
    """
    Calculate confluence score - how many factors agree on direction.

    CONFLUENCE = Multiple independent factors confirming the same direction
    Higher confluence = Higher confidence!

    Args:
        market_data: Complete market data with all indicators
        side: 'LONG' or 'SHORT'

    Returns:
        Dict with confluence score and factor breakdown
    """
    try:
        bullish_factors = []
        bearish_factors = []

        # Factor 1: Divergence
        divergence = market_data.get('divergence', {})
        if divergence.get('has_divergence'):
            if divergence.get('type') == 'bullish':
                bullish_factors.append(f"Bullish divergence ({divergence.get('indicator')})")
            elif divergence.get('type') == 'bearish':
                bearish_factors.append(f"Bearish divergence ({divergence.get('indicator')})")

        # Factor 2: Support/Resistance
        sr = market_data.get('support_resistance', {})
        support_dist = sr.get('support_distance_pct', 100)
        resistance_dist = sr.get('resistance_distance_pct', 100)

        if support_dist < 2:  # Within 2% of support
            bullish_factors.append(f"At support (${sr.get('nearest_support', 0):.2f})")
        if resistance_dist < 2:  # Within 2% of resistance
            bearish_factors.append(f"At resistance (${sr.get('nearest_resistance', 0):.2f})")

        # Factor 3: Order Flow
        order_flow = market_data.get('order_flow', {})
        signal = order_flow.get('signal', 'neutral')
        if signal in ['strong_bullish', 'bullish']:
            bullish_factors.append(f"Order flow {signal} ({order_flow.get('imbalance', 0):.1f}%)")
        elif signal in ['strong_bearish', 'bearish']:
            bearish_factors.append(f"Order flow {signal} ({order_flow.get('imbalance', 0):.1f}%)")

        # Factor 4: Smart Money
        smart_money = market_data.get('smart_money', {})
        sm_signal = smart_money.get('smart_money_signal', 'neutral')
        if sm_signal == 'bullish':
            bullish_factors.append(f"Smart money bullish ({smart_money.get('order_block_count', 0)} order blocks)")
        elif sm_signal == 'bearish':
            bearish_factors.append(f"Smart money bearish ({smart_money.get('order_block_count', 0)} order blocks)")

        # Factor 5: Volume Profile (POC)
        volume_profile = market_data.get('volume_profile', {})
        poc = volume_profile.get('poc', 0)
        current_price = market_data.get('current_price', 0)
        if poc > 0 and current_price > 0:
            poc_distance = abs(current_price - poc) / current_price * 100
            if poc_distance < 1:  # Within 1% of POC
                bullish_factors.append(f"At POC/Fair Value (${poc:.2f})")

        # Factor 6: Fibonacci
        fibonacci = market_data.get('fibonacci', {})
        fib_distance = fibonacci.get('nearest_distance_pct', 100)
        if fib_distance < 1:  # Within 1% of Fib level
            fib_level = fibonacci.get('nearest_fib_level', 0)
            if fib_level in [0.382, 0.5, 0.618]:  # Key retracement levels
                bullish_factors.append(f"At Fibonacci {fib_level} level")

        # Factor 7: Funding Rate
        funding = market_data.get('funding_analysis', {})
        funding_impl = funding.get('trading_implication', 'neutral')
        if 'long_opportunity' in funding_impl:
            bullish_factors.append(f"Funding rate favorable for LONG")
        elif 'short_opportunity' in funding_impl:
            bearish_factors.append(f"Funding rate favorable for SHORT")

        # Factor 8: Volatility Breakout
        volatility = market_data.get('volatility', {})
        if volatility.get('breakout_detected'):
            # Breakout is directional - check trend
            trend = market_data.get('indicators', {}).get('1h', {}).get('trend', 'unknown')
            if trend == 'uptrend':
                bullish_factors.append("Volatility breakout (uptrend)")
            elif trend == 'downtrend':
                bearish_factors.append("Volatility breakout (downtrend)")

        # Factor 9: Multi-timeframe RSI
        rsi_15m = market_data.get('indicators', {}).get('15m', {}).get('rsi', 50)
        rsi_1h = market_data.get('indicators', {}).get('1h', {}).get('rsi', 50)
        rsi_4h = market_data.get('indicators', {}).get('4h', {}).get('rsi', 50)

        if rsi_15m < 35 and rsi_1h < 40:
            bullish_factors.append(f"Oversold (RSI 15m:{rsi_15m:.0f}, 1h:{rsi_1h:.0f})")
        if rsi_15m > 65 and rsi_1h > 60:
            bearish_factors.append(f"Overbought (RSI 15m:{rsi_15m:.0f}, 1h:{rsi_1h:.0f})")

        # Calculate confluence score
        if side == 'LONG':
            confluence_count = len(bullish_factors)
            conflicting_count = len(bearish_factors)
            factors = bullish_factors
        else:  # SHORT
            confluence_count = len(bearish_factors)
            conflicting_count = len(bullish_factors)
            factors = bearish_factors

        total_factors = confluence_count + conflicting_count
        if total_factors > 0:
            confluence_score = (confluence_count / total_factors) * 100
        else:
            confluence_score = 50  # Neutral

        return {
            'score': float(confluence_score),
            'confluence_count': confluence_count,
            'conflicting_count': conflicting_count,
            'factors': factors,
            'strength': 'strong' if confluence_score >= 75 else 'moderate' if confluence_score >= 60 else 'weak'
        }

    except Exception as e:
        return {
            'score': 50.0,
            'confluence_count': 0,
            'conflicting_count': 0,
            'factors': [],
            'strength': 'unknown'
        }


def calculate_momentum_strength(ohlcv_data: List[List]) -> Dict[str, Any]:
    """
    Calculate momentum strength and rate of change across timeframes.

    MOMENTUM STRENGTH = Is momentum accelerating or decelerating?
    - Accelerating = Strong trend, ride it!
    - Decelerating = Momentum fading, be cautious!

    Args:
        ohlcv_data: List of OHLCV candles

    Returns:
        Dict with momentum analysis
    """
    if not ohlcv_data or len(ohlcv_data) < 50:
        return {
            'momentum_direction': 'neutral',
            'momentum_strength': 0.0,
            'is_accelerating': False,
            'roc_1h': 0.0,
            'roc_4h': 0.0,
            'roc_12h': 0.0
        }

    try:
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        current_price = df['close'].iloc[-1]

        # Calculate Rate of Change (ROC) for different periods
        # 1h = ~4 candles (15m timeframe)
        # 4h = ~16 candles
        # 12h = ~48 candles

        if len(df) >= 4:
            price_1h_ago = df['close'].iloc[-4]
            roc_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100
        else:
            roc_1h = 0

        if len(df) >= 16:
            price_4h_ago = df['close'].iloc[-16]
            roc_4h = ((current_price - price_4h_ago) / price_4h_ago) * 100
        else:
            roc_4h = 0

        if len(df) >= 48:
            price_12h_ago = df['close'].iloc[-48]
            roc_12h = ((current_price - price_12h_ago) / price_12h_ago) * 100
        else:
            roc_12h = 0

        # Determine momentum direction
        avg_roc = (roc_1h + roc_4h + roc_12h) / 3
        if avg_roc > 1:
            momentum_direction = 'bullish'
        elif avg_roc < -1:
            momentum_direction = 'bearish'
        else:
            momentum_direction = 'neutral'

        # Check if momentum is accelerating
        # Accelerating = short-term ROC > long-term ROC
        is_accelerating = False
        if momentum_direction == 'bullish':
            is_accelerating = roc_1h > roc_4h and roc_4h > roc_12h
        elif momentum_direction == 'bearish':
            is_accelerating = roc_1h < roc_4h and roc_4h < roc_12h

        # Momentum strength (0-100)
        momentum_strength = min(abs(avg_roc) * 10, 100)

        return {
            'momentum_direction': momentum_direction,
            'momentum_strength': float(momentum_strength),
            'is_accelerating': is_accelerating,
            'roc_1h': float(roc_1h),
            'roc_4h': float(roc_4h),
            'roc_12h': float(roc_12h)
        }

    except Exception as e:
        return {
            'momentum_direction': 'neutral',
            'momentum_strength': 0.0,
            'is_accelerating': False,
            'roc_1h': 0.0,
            'roc_4h': 0.0,
            'roc_12h': 0.0
        }


def calculate_btc_correlation(symbol: str, symbol_ohlcv: List[List], btc_ohlcv: List[List]) -> Dict[str, Any]:
    """
    Calculate correlation between symbol and BTC.

    BTC CORRELATION = Does this coin follow BTC?
    - High correlation (>0.8): Trade BTC instead, not this coin
    - Low correlation (<0.4): Independent move possible!
    - Negative correlation: Inverse relationship (rare but valuable)

    Args:
        symbol: Trading symbol
        symbol_ohlcv: Symbol's OHLCV data
        btc_ohlcv: BTC's OHLCV data

    Returns:
        Dict with correlation analysis
    """
    if not symbol_ohlcv or not btc_ohlcv or len(symbol_ohlcv) < 20 or len(btc_ohlcv) < 20:
        return {
            'correlation': 0.5,
            'correlation_strength': 'unknown',
            'independent_move': False,
            'recommendation': 'Insufficient data'
        }

    try:
        # Get last 20 closes for both
        symbol_df = pd.DataFrame(symbol_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        btc_df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        symbol_df['close'] = pd.to_numeric(symbol_df['close'], errors='coerce')
        btc_df['close'] = pd.to_numeric(btc_df['close'], errors='coerce')

        # Get last 20 candles
        symbol_prices = symbol_df['close'].tail(20).pct_change().dropna()
        btc_prices = btc_df['close'].tail(20).pct_change().dropna()

        # Ensure same length
        min_len = min(len(symbol_prices), len(btc_prices))
        if min_len < 10:
            return {
                'correlation': 0.5,
                'correlation_strength': 'unknown',
                'independent_move': False,
                'recommendation': 'Insufficient data'
            }

        symbol_prices = symbol_prices.tail(min_len)
        btc_prices = btc_prices.tail(min_len)

        # Calculate Pearson correlation
        correlation = symbol_prices.corr(btc_prices)

        # Determine strength
        abs_corr = abs(correlation)
        if abs_corr > 0.8:
            correlation_strength = 'very_high'
            independent_move = False
            recommendation = "High BTC correlation - consider trading BTC instead"
        elif abs_corr > 0.6:
            correlation_strength = 'high'
            independent_move = False
            recommendation = "Follows BTC closely - watch BTC direction"
        elif abs_corr > 0.4:
            correlation_strength = 'moderate'
            independent_move = False
            recommendation = "Moderate BTC correlation"
        else:
            correlation_strength = 'low'
            independent_move = True
            recommendation = "Low BTC correlation - independent move possible!"

        return {
            'correlation': float(correlation),
            'correlation_strength': correlation_strength,
            'independent_move': independent_move,
            'recommendation': recommendation
        }

    except Exception as e:
        return {
            'correlation': 0.5,
            'correlation_strength': 'unknown',
            'independent_move': False,
            'recommendation': f'Error: {e}'
        }


def calculate_multi_timeframe_indicators(
    ohlcv_5m: List[List],
    ohlcv_15m: List[List],
    ohlcv_1h: List[List],
    ohlcv_4h: List[List]
) -> Dict[str, Any]:
    """
    Calculate indicators across multiple timeframes for confluence analysis.

    MULTI-TIMEFRAME CONFLUENCE = Professional trading edge!
    - 5m: Micro structure, precise entries
    - 15m: Primary execution timeframe
    - 1h: Trend bias filter
    - 4h: Macro trend and major support/resistance

    Args:
        ohlcv_5m: 5-minute OHLCV data
        ohlcv_15m: 15-minute OHLCV data
        ohlcv_1h: 1-hour OHLCV data
        ohlcv_4h: 4-hour OHLCV data

    Returns:
        Dict with multi-timeframe analysis and confluence signals
    """
    try:
        # Calculate indicators for each timeframe
        indicators_5m = calculate_indicators(ohlcv_5m)
        indicators_15m = calculate_indicators(ohlcv_15m)
        indicators_1h = calculate_indicators(ohlcv_1h)
        indicators_4h = calculate_indicators(ohlcv_4h)

        # Analyze timeframe alignment
        current_price = indicators_15m['close']

        # Check trend alignment
        trend_5m = indicators_5m.get('trend', 'unknown')
        trend_15m = indicators_15m.get('trend', 'unknown')
        trend_1h = indicators_1h.get('trend', 'unknown')
        trend_4h = indicators_4h.get('trend', 'unknown')

        # Count bullish vs bearish timeframes
        bullish_count = sum([
            1 for t in [trend_5m, trend_15m, trend_1h, trend_4h]
            if t == 'uptrend'
        ])
        bearish_count = sum([
            1 for t in [trend_5m, trend_15m, trend_1h, trend_4h]
            if t == 'downtrend'
        ])

        # Determine overall trend alignment
        if bullish_count >= 3:
            trend_alignment = 'strong_bullish'
            alignment_score = (bullish_count / 4) * 100
        elif bearish_count >= 3:
            trend_alignment = 'strong_bearish'
            alignment_score = (bearish_count / 4) * 100
        elif bullish_count == 2 and bearish_count == 2:
            trend_alignment = 'conflicted'
            alignment_score = 50
        else:
            trend_alignment = 'mixed'
            alignment_score = 60

        # Check EMA50 alignment (price above/below EMA50 on higher timeframes)
        ema50_1h = indicators_1h.get('sma_50', current_price)
        ema50_4h = indicators_4h.get('sma_50', current_price)

        price_above_ema50_1h = current_price > ema50_1h
        price_above_ema50_4h = current_price > ema50_4h

        # Check RSI alignment for overbought/oversold
        rsi_5m = indicators_5m.get('rsi', 50)
        rsi_15m = indicators_15m.get('rsi', 50)
        rsi_1h = indicators_1h.get('rsi', 50)
        rsi_4h = indicators_4h.get('rsi', 50)

        # Oversold across timeframes (potential long)
        oversold_count = sum([
            1 for rsi in [rsi_15m, rsi_1h, rsi_4h]
            if rsi < 35
        ])

        # Overbought across timeframes (potential short)
        overbought_count = sum([
            1 for rsi in [rsi_15m, rsi_1h, rsi_4h]
            if rsi > 65
        ])

        # Higher timeframe bias (1h + 4h must agree for high-confidence trades)
        higher_tf_bullish = (
            price_above_ema50_1h and
            price_above_ema50_4h and
            rsi_1h > 45 and
            rsi_4h > 45
        )

        higher_tf_bearish = (
            not price_above_ema50_1h and
            not price_above_ema50_4h and
            rsi_1h < 55 and
            rsi_4h < 55
        )

        # Trading recommendations based on confluence
        if trend_alignment == 'strong_bullish' and higher_tf_bullish:
            trading_bias = 'LONG_ONLY'
            confidence_multiplier = 1.3
            recommendation = "✅ STRONG BULLISH CONFLUENCE - All timeframes aligned for LONG"
        elif trend_alignment == 'strong_bearish' and higher_tf_bearish:
            trading_bias = 'SHORT_ONLY'
            confidence_multiplier = 1.3
            recommendation = "✅ STRONG BEARISH CONFLUENCE - All timeframes aligned for SHORT"
        elif trend_alignment == 'conflicted':
            trading_bias = 'AVOID'
            confidence_multiplier = 0.5
            recommendation = "⚠️ CONFLICTING TIMEFRAMES - Avoid trading, wait for clarity"
        elif higher_tf_bullish and trend_1h == 'uptrend':
            trading_bias = 'LONG_PREFERRED'
            confidence_multiplier = 1.1
            recommendation = "LONG PREFERRED - Higher timeframes bullish, 1h confirms"
        elif higher_tf_bearish and trend_1h == 'downtrend':
            trading_bias = 'SHORT_PREFERRED'
            confidence_multiplier = 1.1
            recommendation = "SHORT PREFERRED - Higher timeframes bearish, 1h confirms"
        else:
            trading_bias = 'NEUTRAL'
            confidence_multiplier = 0.8
            recommendation = "NEUTRAL - Mixed signals, reduce position size"

        return {
            'timeframes': {
                '5m': indicators_5m,
                '15m': indicators_15m,
                '1h': indicators_1h,
                '4h': indicators_4h
            },
            'confluence_analysis': {
                'trend_alignment': trend_alignment,
                'alignment_score': float(alignment_score),
                'bullish_timeframes': bullish_count,
                'bearish_timeframes': bearish_count,
                'higher_tf_bullish': higher_tf_bullish,
                'higher_tf_bearish': higher_tf_bearish,
                'trading_bias': trading_bias,
                'confidence_multiplier': float(confidence_multiplier),
                'recommendation': recommendation
            },
            'rsi_analysis': {
                '5m': float(rsi_5m),
                '15m': float(rsi_15m),
                '1h': float(rsi_1h),
                '4h': float(rsi_4h),
                'oversold_timeframes': oversold_count,
                'overbought_timeframes': overbought_count
            },
            'trend_summary': {
                '5m': trend_5m,
                '15m': trend_15m,
                '1h': trend_1h,
                '4h': trend_4h
            },
            'ema50_analysis': {
                'price': float(current_price),
                'ema50_1h': float(ema50_1h),
                'ema50_4h': float(ema50_4h),
                'above_1h': price_above_ema50_1h,
                'above_4h': price_above_ema50_4h
            }
        }

    except Exception as e:
        return {
            'error': str(e),
            'timeframes': {},
            'confluence_analysis': {
                'trend_alignment': 'unknown',
                'trading_bias': 'AVOID',
                'confidence_multiplier': 0.5,
                'recommendation': f'Error calculating multi-timeframe analysis: {e}'
            }
        }


def analyze_open_interest(
    oi_history: List[Dict[str, Any]],
    price_history: List[float],
    current_price: float
) -> Dict[str, Any]:
    """
    Analyze Open Interest and its relationship with price for trend strength.

    OPEN INTEREST = Total Outstanding Contracts
    - OI rising + Price rising = STRONG BULLISH (new longs entering)
    - OI rising + Price falling = STRONG BEARISH (new shorts entering)
    - OI falling + Price rising = WEAK RALLY (longs closing)
    - OI falling + Price falling = WEAK DUMP (shorts closing)

    Args:
        oi_history: List of OI data points with 'openInterest' and 'timestamp'
        price_history: Corresponding price data
        current_price: Current market price

    Returns:
        Dict with OI analysis and trading implications
    """
    try:
        if not oi_history or len(oi_history) < 3:
            return {
                'current_oi': 0.0,
                'oi_change_pct': 0.0,
                'price_change_pct': 0.0,
                'trend_strength': 'unknown',
                'signal': 'neutral',
                'trading_implication': 'Insufficient OI data',
                'confidence_boost': 0.0
            }

        # Extract OI values
        oi_values = [float(d.get('openInterest', 0)) for d in oi_history]

        if len(oi_values) < 2 or len(price_history) < 2:
            return {
                'current_oi': oi_values[-1] if oi_values else 0.0,
                'oi_change_pct': 0.0,
                'price_change_pct': 0.0,
                'trend_strength': 'unknown',
                'signal': 'neutral',
                'trading_implication': 'Insufficient data',
                'confidence_boost': 0.0
            }

        current_oi = oi_values[-1]
        prev_oi = oi_values[0]

        # Calculate OI change
        oi_change_pct = ((current_oi - prev_oi) / prev_oi) * 100 if prev_oi > 0 else 0

        # Calculate price change
        price_change_pct = ((current_price - price_history[0]) / price_history[0]) * 100 if price_history[0] > 0 else 0

        # Determine trend strength and signal
        oi_rising = oi_change_pct > 2  # OI increased by >2%
        oi_falling = oi_change_pct < -2  # OI decreased by >2%
        price_rising = price_change_pct > 0.5  # Price increased
        price_falling = price_change_pct < -0.5  # Price decreased

        # Analyze combinations
        if oi_rising and price_rising:
            trend_strength = 'STRONG_BULLISH'
            signal = 'strong_buy'
            trading_implication = '✅ STRONG UPTREND - New longs entering, high conviction buyers'
            confidence_boost = 0.15  # +15% confidence
        elif oi_rising and price_falling:
            trend_strength = 'STRONG_BEARISH'
            signal = 'strong_sell'
            trading_implication = '✅ STRONG DOWNTREND - New shorts entering, high conviction sellers'
            confidence_boost = 0.15
        elif oi_falling and price_rising:
            trend_strength = 'WEAK_BULLISH'
            signal = 'weak_buy'
            trading_implication = '⚠️ WEAK RALLY - Longs closing positions, reversal risk'
            confidence_boost = -0.10  # -10% confidence (risky)
        elif oi_falling and price_falling:
            trend_strength = 'WEAK_BEARISH'
            signal = 'weak_sell'
            trading_implication = '⚠️ WEAK DUMP - Shorts closing positions, bounce possible'
            confidence_boost = -0.10
        else:
            trend_strength = 'NEUTRAL'
            signal = 'neutral'
            trading_implication = 'OI and price moving sideways, no clear trend'
            confidence_boost = 0.0

        return {
            'current_oi': float(current_oi),
            'oi_change_pct': float(oi_change_pct),
            'price_change_pct': float(price_change_pct),
            'trend_strength': trend_strength,
            'signal': signal,
            'trading_implication': trading_implication,
            'confidence_boost': float(confidence_boost),
            'oi_rising': oi_rising,
            'price_rising': price_rising
        }

    except Exception as e:
        return {
            'current_oi': 0.0,
            'oi_change_pct': 0.0,
            'price_change_pct': 0.0,
            'trend_strength': 'unknown',
            'signal': 'neutral',
            'trading_implication': f'Error: {e}',
            'confidence_boost': 0.0
        }


def estimate_liquidation_levels(
    current_price: float,
    ohlcv_data: List[List],
    typical_leverage: int = 10
) -> Dict[str, Any]:
    """
    Estimate liquidation cluster zones based on recent price action.

    LIQUIDATION HEATMAP = Where leveraged positions get liquidated
    - Long liquidations cluster BELOW recent support (stop hunts)
    - Short liquidations cluster ABOVE recent resistance (stop hunts)
    - Price tends to move TOWARD liquidation clusters (liquidity magnet)

    Args:
        current_price: Current market price
        ohlcv_data: Recent OHLCV data
        typical_leverage: Typical leverage used (default 10x)

    Returns:
        Dict with estimated liquidation zones
    """
    try:
        if not ohlcv_data or len(ohlcv_data) < 20:
            return {
                'long_liquidation_zones': [],
                'short_liquidation_zones': [],
                'nearest_long_liq': current_price * 0.90,
                'nearest_short_liq': current_price * 1.10,
                'magnet_direction': 'none',
                'trading_implication': 'Insufficient data for liquidation estimation'
            }

        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Find recent swing lows (long liquidation zones)
        swing_period = 5
        swing_lows = []
        swing_highs = []

        for i in range(swing_period, len(df) - swing_period):
            is_swing_low = True
            is_swing_high = True

            for j in range(1, swing_period + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i - j] or df['low'].iloc[i] >= df['low'].iloc[i + j]:
                    is_swing_low = False
                if df['high'].iloc[i] <= df['high'].iloc[i - j] or df['high'].iloc[i] <= df['high'].iloc[i + j]:
                    is_swing_high = False

            if is_swing_low:
                swing_lows.append(float(df['low'].iloc[i]))
            if is_swing_high:
                swing_highs.append(float(df['high'].iloc[i]))

        # Estimate long liquidation zones (below swing lows)
        # At 10x leverage, ~9% move liquidates position
        liquidation_buffer = 0.09 / typical_leverage * 10  # Scale based on leverage

        long_liq_zones = []
        for low in swing_lows[-5:]:  # Last 5 swing lows
            liq_price = low * (1 - liquidation_buffer)
            if liq_price < current_price:
                long_liq_zones.append({
                    'price': float(liq_price),
                    'distance_pct': abs(current_price - liq_price) / current_price * 100
                })

        # Estimate short liquidation zones (above swing highs)
        short_liq_zones = []
        for high in swing_highs[-5:]:  # Last 5 swing highs
            liq_price = high * (1 + liquidation_buffer)
            if liq_price > current_price:
                short_liq_zones.append({
                    'price': float(liq_price),
                    'distance_pct': abs(liq_price - current_price) / current_price * 100
                })

        # Find nearest liquidation zones
        nearest_long_liq = min([z['price'] for z in long_liq_zones]) if long_liq_zones else current_price * 0.90
        nearest_short_liq = min([z['price'] for z in short_liq_zones]) if short_liq_zones else current_price * 1.10

        # Determine magnet direction (which cluster is closer)
        long_liq_distance = abs(current_price - nearest_long_liq) / current_price * 100
        short_liq_distance = abs(nearest_short_liq - current_price) / current_price * 100

        if long_liq_distance < 3 and long_liq_distance < short_liq_distance:
            magnet_direction = 'downward'
            trading_implication = '⚠️ DOWNWARD MAGNET - Long liquidations nearby, possible dump to liquidate longs'
        elif short_liq_distance < 3 and short_liq_distance < long_liq_distance:
            magnet_direction = 'upward'
            trading_implication = '⚠️ UPWARD MAGNET - Short liquidations nearby, possible pump to liquidate shorts'
        else:
            magnet_direction = 'balanced'
            trading_implication = 'Liquidation clusters balanced, no strong magnet effect'

        return {
            'long_liquidation_zones': long_liq_zones[:3],  # Top 3 zones
            'short_liquidation_zones': short_liq_zones[:3],
            'nearest_long_liq': float(nearest_long_liq),
            'nearest_short_liq': float(nearest_short_liq),
            'long_liq_distance_pct': float(long_liq_distance),
            'short_liq_distance_pct': float(short_liq_distance),
            'magnet_direction': magnet_direction,
            'trading_implication': trading_implication
        }

    except Exception as e:
        return {
            'long_liquidation_zones': [],
            'short_liquidation_zones': [],
            'nearest_long_liq': current_price * 0.90,
            'nearest_short_liq': current_price * 1.10,
            'magnet_direction': 'none',
            'trading_implication': f'Error: {e}'
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
