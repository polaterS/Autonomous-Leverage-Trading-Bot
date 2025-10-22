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

    # Fill any NaN values
    df.fillna(method='ffill', inplace=True)

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
