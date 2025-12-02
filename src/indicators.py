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

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔥 TIER 1 CRITICAL INDICATORS - Professional Edge
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Stochastic RSI - REMOVED (replaced by TIER 2 enhanced version below)
        # stoch_rsi = calculate_stochastic_rsi(df['close'])
        # indicators['stoch_rsi_k'] = stoch_rsi['k_line']
        # indicators['stoch_rsi_d'] = stoch_rsi['d_line']
        # indicators['stoch_rsi_signal'] = stoch_rsi['signal']
        # indicators['stoch_rsi_overbought'] = stoch_rsi['overbought']
        # indicators['stoch_rsi_oversold'] = stoch_rsi['oversold']

        # SuperTrend (superior trend indicator)
        supertrend = calculate_supertrend(df)
        indicators['supertrend'] = supertrend['supertrend']
        indicators['supertrend_direction'] = supertrend['direction']
        indicators['supertrend_signal'] = supertrend['signal']
        indicators['supertrend_trend'] = supertrend['trend']
        indicators['price_above_supertrend'] = supertrend['price_above']

        # VWAP (institutional pivot)
        vwap = calculate_vwap(df)
        indicators['vwap'] = vwap['vwap']
        indicators['vwap_distance_pct'] = vwap['distance_pct']
        indicators['vwap_bias'] = vwap['bias']
        indicators['above_vwap'] = vwap['above_vwap']

        # MFI - REMOVED (replaced by TIER 2 enhanced version below)
        # mfi = calculate_mfi(df)
        # indicators['mfi'] = mfi['mfi']
        # indicators['mfi_signal'] = mfi['signal']
        # indicators['mfi_overbought'] = mfi['overbought']
        # indicators['mfi_oversold'] = mfi['oversold']

        # ATR as percentage of price (for volatility-adjusted stops)
        current_price = indicators['close']
        indicators['atr_percent'] = (indicators['atr'] / current_price) * 100 if current_price > 0 else 0

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🎯 TIER 2 CONSERVATIVE INDICATORS - New Professional Additions
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Ichimoku Cloud (Ultimate trend system) - with fallback
        try:
            ichimoku = calculate_ichimoku(ohlcv_data)
            indicators['ichimoku_conversion'] = ichimoku['conversion_line']
            indicators['ichimoku_base'] = ichimoku['base_line']
            indicators['ichimoku_span_a'] = ichimoku['leading_span_a']
            indicators['ichimoku_span_b'] = ichimoku['leading_span_b']
            indicators['ichimoku_cloud_color'] = ichimoku['cloud_color']
            indicators['ichimoku_price_vs_cloud'] = ichimoku['price_vs_cloud']
            indicators['ichimoku_signal'] = ichimoku['signal']
            indicators['ichimoku_trend_strength'] = ichimoku['trend_strength']
        except Exception as e:
            # Fallback: Neutral Ichimoku if calculation fails
            import logging
            logger = logging.getLogger('trading_bot')
            logger.warning(f"⚠️ Ichimoku calculation failed, using NEUTRAL fallback: {e}")
            indicators['ichimoku_conversion'] = current_price
            indicators['ichimoku_base'] = current_price
            indicators['ichimoku_span_a'] = current_price
            indicators['ichimoku_span_b'] = current_price
            indicators['ichimoku_cloud_color'] = 'NEUTRAL'
            indicators['ichimoku_price_vs_cloud'] = 'IN_CLOUD'
            indicators['ichimoku_signal'] = 'NEUTRAL'
            indicators['ichimoku_trend_strength'] = 0.0

        # Stochastic RSI (Enhanced - more sensitive oversold/overbought) - with fallback
        try:
            stoch_rsi_new = calculate_stochastic_rsi(ohlcv_data)
            indicators['stoch_rsi_value'] = stoch_rsi_new['stoch_rsi']
            indicators['stoch_rsi_k_new'] = stoch_rsi_new['k']
            indicators['stoch_rsi_d_new'] = stoch_rsi_new['d']
            indicators['stoch_rsi_signal_new'] = stoch_rsi_new['signal']
            indicators['stoch_rsi_zone'] = stoch_rsi_new['zone']
        except Exception as e:
            # Fallback: Neutral StochRSI if calculation fails
            import logging
            logger = logging.getLogger('trading_bot')
            logger.warning(f"⚠️ StochRSI calculation failed, using NEUTRAL fallback: {e}")
            indicators['stoch_rsi_value'] = 50.0
            indicators['stoch_rsi_k_new'] = 50.0
            indicators['stoch_rsi_d_new'] = 50.0
            indicators['stoch_rsi_signal_new'] = 'NEUTRAL'
            indicators['stoch_rsi_zone'] = 'NEUTRAL'

        # Money Flow Index (Volume-weighted RSI - institutional activity) - with fallback
        try:
            mfi_new = calculate_mfi(ohlcv_data)
            indicators['mfi_value'] = mfi_new['mfi']
            indicators['mfi_signal_new'] = mfi_new['signal']
            indicators['mfi_zone'] = mfi_new['zone']
            indicators['mfi_buying_pressure'] = mfi_new['buying_pressure']
        except Exception as e:
            # Fallback: Neutral MFI if calculation fails
            import logging
            logger = logging.getLogger('trading_bot')
            logger.warning(f"⚠️ MFI calculation failed, using NEUTRAL fallback: {e}")
            indicators['mfi_value'] = 50.0
            indicators['mfi_signal_new'] = 'NEUTRAL'
            indicators['mfi_zone'] = 'NEUTRAL'
            indicators['mfi_buying_pressure'] = False

        return indicators

    except Exception as e:
        # Log error with logger instead of print
        import logging
        logger = logging.getLogger('trading_bot')
        logger.error(f"❌ Error calculating indicators: {e}", exc_info=True)
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔥 TIER 1 CRITICAL INDICATORS - Professional Trading Edge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_stochastic_rsi(prices: pd.Series, period: int = 14, k: int = 3, d: int = 3) -> Dict[str, float]:
    """
    Calculate Stochastic RSI - More sensitive overbought/oversold indicator than regular RSI.

    🎯 USAGE:
    - StochRSI K crosses D from below + < 20 → STRONG BUY signal
    - StochRSI K crosses D from above + > 80 → STRONG SELL signal
    - More responsive than RSI for catching reversals

    Args:
        prices: Series of closing prices
        period: RSI calculation period (default 14)
        k: K line smoothing period (default 3)
        d: D line smoothing period (default 3)

    Returns:
        Dict with k_line, d_line, crossover signal
    """
    try:
        if len(prices) < period + k + d:
            return {'k_line': 50.0, 'd_line': 50.0, 'signal': 'neutral'}

        # Calculate RSI first
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        # Calculate Stochastic of RSI
        rsi_min = rsi.rolling(window=period).min()
        rsi_max = rsi.rolling(window=period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100

        # Smooth with K and D periods
        k_line = stoch_rsi.rolling(window=k).mean()
        d_line = k_line.rolling(window=d).mean()

        # Detect crossover signals
        k_val = float(k_line.iloc[-1])
        d_val = float(d_line.iloc[-1])
        k_prev = float(k_line.iloc[-2]) if len(k_line) > 1 else k_val
        d_prev = float(d_line.iloc[-2]) if len(d_line) > 1 else d_val

        signal = 'neutral'
        if k_val > d_val and k_prev <= d_prev:
            if k_val < 20:
                signal = 'strong_buy'  # Oversold crossover
            else:
                signal = 'buy'  # Regular bullish crossover
        elif k_val < d_val and k_prev >= d_prev:
            if k_val > 80:
                signal = 'strong_sell'  # Overbought crossover
            else:
                signal = 'sell'  # Regular bearish crossover

        return {
            'k_line': k_val,
            'd_line': d_val,
            'signal': signal,
            'overbought': k_val > 80,
            'oversold': k_val < 20
        }

    except Exception as e:
        print(f"StochRSI calculation error: {e}")
        return {'k_line': 50.0, 'd_line': 50.0, 'signal': 'neutral'}


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict[str, Any]:
    """
    Calculate SuperTrend - Superior trend indicator for crypto markets.

    🎯 USAGE:
    - Price above SuperTrend = STRONG UPTREND → BUY bias
    - Price below SuperTrend = STRONG DOWNTREND → SELL bias
    - Trend change = Reversal signal (more reliable than MACD for crypto)

    Args:
        df: DataFrame with high, low, close columns
        period: ATR calculation period (default 10)
        multiplier: ATR multiplier for bands (default 3.0)

    Returns:
        Dict with supertrend value, direction, and signal
    """
    try:
        if len(df) < period + 10:
            return {
                'supertrend': float(df['close'].iloc[-1]),
                'direction': 0,
                'signal': 'neutral',
                'trend': 'unknown'
            }

        # Calculate ATR
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Calculate basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        # Initialize SuperTrend series
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        # Calculate SuperTrend with proper logic
        for i in range(period, len(df)):
            if i == period:
                # Initialize
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                prev_supertrend = supertrend.iloc[i-1]
                prev_direction = direction.iloc[i-1]

                if close.iloc[i] > prev_supertrend:
                    # Uptrend
                    supertrend.iloc[i] = max(lower_band.iloc[i], prev_supertrend)
                    direction.iloc[i] = 1
                elif close.iloc[i] < prev_supertrend:
                    # Downtrend
                    supertrend.iloc[i] = min(upper_band.iloc[i], prev_supertrend)
                    direction.iloc[i] = -1
                else:
                    # Continue previous trend
                    supertrend.iloc[i] = prev_supertrend
                    direction.iloc[i] = prev_direction

        # Get final values
        final_st = float(supertrend.iloc[-1])
        final_dir = int(direction.iloc[-1])
        prev_dir = int(direction.iloc[-2]) if len(direction) > 1 else final_dir
        current_price = float(close.iloc[-1])

        # Detect trend change
        signal = 'neutral'
        if final_dir == 1 and prev_dir == -1:
            signal = 'buy'  # Trend reversed to up
        elif final_dir == -1 and prev_dir == 1:
            signal = 'sell'  # Trend reversed to down

        return {
            'supertrend': final_st,
            'direction': final_dir,
            'signal': signal,
            'trend': 'uptrend' if final_dir == 1 else 'downtrend',
            'price_above': current_price > final_st
        }

    except Exception as e:
        print(f"SuperTrend calculation error: {e}")
        return {
            'supertrend': float(df['close'].iloc[-1]),
            'direction': 0,
            'signal': 'neutral',
            'trend': 'unknown'
        }


def calculate_vwap(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate VWAP (Volume Weighted Average Price) - Institutional pivot level.

    🎯 USAGE:
    - Price > VWAP = Bullish bias (institutions buying)
    - Price < VWAP = Bearish bias (institutions selling)
    - VWAP acts as dynamic support/resistance
    - Distance from VWAP indicates strength of move

    Args:
        df: DataFrame with high, low, close, volume columns

    Returns:
        Dict with VWAP value, distance, and bias
    """
    try:
        if len(df) < 10:
            return {
                'vwap': float(df['close'].iloc[-1]),
                'distance_pct': 0.0,
                'bias': 'neutral'
            }

        # Typical price (HL2C) weighted by volume
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

        vwap_val = float(vwap.iloc[-1])
        current_price = float(df['close'].iloc[-1])
        distance_pct = ((current_price - vwap_val) / vwap_val) * 100

        # Determine bias
        if distance_pct > 1.0:
            bias = 'strong_bullish'
        elif distance_pct > 0.3:
            bias = 'bullish'
        elif distance_pct < -1.0:
            bias = 'strong_bearish'
        elif distance_pct < -0.3:
            bias = 'bearish'
        else:
            bias = 'neutral'

        return {
            'vwap': vwap_val,
            'distance_pct': distance_pct,
            'bias': bias,
            'above_vwap': current_price > vwap_val
        }

    except Exception as e:
        print(f"VWAP calculation error: {e}")
        return {
            'vwap': float(df['close'].iloc[-1]),
            'distance_pct': 0.0,
            'bias': 'neutral'
        }


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
    """
    Calculate Money Flow Index (MFI) - Volume-weighted RSI.

    🎯 USAGE:
    - MFI > 80 = Overbought (high buying pressure, reversal risk)
    - MFI < 20 = Oversold (high selling pressure, bounce opportunity)
    - MFI divergence with price = Strong reversal signal
    - More reliable than RSI because it includes volume

    Args:
        df: DataFrame with high, low, close, volume columns
        period: Calculation period (default 14)

    Returns:
        Dict with MFI value and signal
    """
    try:
        if len(df) < period + 10:
            return {
                'mfi': 50.0,
                'signal': 'neutral',
                'overbought': False,
                'oversold': False
            }

        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Calculate money flow
        money_flow = typical_price * df['volume']

        # Positive and negative money flow
        price_diff = typical_price.diff()
        positive_flow = money_flow.where(price_diff > 0, 0)
        negative_flow = money_flow.where(price_diff < 0, 0)

        # Calculate money flow ratio
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfr = positive_mf / (negative_mf + 1e-10)  # Avoid division by zero
        mfi = 100 - (100 / (1 + mfr))

        mfi_val = float(mfi.iloc[-1])

        # Determine signal
        if mfi_val > 80:
            signal = 'overbought'
        elif mfi_val < 20:
            signal = 'oversold'
        elif mfi_val > 70:
            signal = 'weakening'  # Approaching overbought
        elif mfi_val < 30:
            signal = 'strengthening'  # Approaching oversold
        else:
            signal = 'neutral'

        return {
            'mfi': mfi_val,
            'signal': signal,
            'overbought': mfi_val > 80,
            'oversold': mfi_val < 20
        }

    except Exception as e:
        print(f"MFI calculation error: {e}")
        return {
            'mfi': 50.0,
            'signal': 'neutral',
            'overbought': False,
            'oversold': False
        }


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

        # 🎯 #3: DEEP ORDER BOOK ANALYSIS (20 → 100 levels)
        # Institutional iceberg orders hidden in deeper levels!
        bids = order_book.get('bids', [])[:100]  # Top 100 bids (was 20)
        asks = order_book.get('asks', [])[:100]  # Top 100 asks (was 20)

        if not bids or not asks:
            return {
                'imbalance': 0.0,
                'imbalance_ratio': 1.0,
                'signal': 'neutral',
                'large_bid_wall': None,
                'large_ask_wall': None,
                'buy_pressure': 0.5
            }

        # 🎯 #3: DEPTH-WEIGHTED VOLUME CALCULATION
        # Exponential decay: deeper levels = less weight
        # Weight = exp(-i / 10) where i is the level index
        import math

        def calculate_depth_weighted_volume(orders):
            """Calculate exponentially depth-weighted volume"""
            weighted_volume = 0.0
            for i, order in enumerate(orders):
                volume = float(order[1])
                weight = math.exp(-i / 10)  # Decay factor
                weighted_volume += volume * weight
            return weighted_volume

        # STANDARD (for backward compatibility)
        total_bid_volume = sum(float(bid[1]) for bid in bids[:20])  # Top 20 only
        total_ask_volume = sum(float(ask[1]) for ask in asks[:20])

        # 🎯 #3: DEPTH-WEIGHTED (institutional positioning)
        weighted_bid_volume = calculate_depth_weighted_volume(bids)
        weighted_ask_volume = calculate_depth_weighted_volume(asks)

        # Bid/Ask imbalance (STANDARD - Top 20 only)
        if total_ask_volume > 0:
            imbalance_ratio = total_bid_volume / total_ask_volume
        else:
            imbalance_ratio = 1.0

        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) * 100

        # 🎯 #3: DEPTH-WEIGHTED IMBALANCE (Institutional positioning!)
        # This reveals hidden iceberg orders in deeper levels
        if weighted_ask_volume > 0:
            weighted_imbalance_ratio = weighted_bid_volume / weighted_ask_volume
        else:
            weighted_imbalance_ratio = 1.0

        weighted_imbalance = (weighted_bid_volume - weighted_ask_volume) / (weighted_bid_volume + weighted_ask_volume) * 100

        # Determine signal using WEIGHTED imbalance (more accurate!)
        if weighted_imbalance > 20:  # More than 20% more bids (weighted)
            signal = 'strong_bullish'
        elif weighted_imbalance > 10:
            signal = 'bullish'
        elif weighted_imbalance < -20:
            signal = 'strong_bearish'
        elif weighted_imbalance < -10:
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
            # STANDARD metrics (Top 20, backward compatible)
            'imbalance': float(imbalance),
            'imbalance_ratio': float(imbalance_ratio),
            'signal': signal,
            'large_bid_wall': large_bid_wall,
            'large_ask_wall': large_ask_wall,
            'buy_pressure': float(buy_pressure),
            'total_bid_volume': float(total_bid_volume),
            'total_ask_volume': float(total_ask_volume),

            # 🎯 #3: DEPTH-WEIGHTED metrics (100 levels, institutional positioning)
            'weighted_imbalance': float(weighted_imbalance),
            'weighted_imbalance_ratio': float(weighted_imbalance_ratio),
            'weighted_bid_volume': float(weighted_bid_volume),
            'weighted_ask_volume': float(weighted_ask_volume),
            'depth_levels_analyzed': len(bids)  # Should be ~100
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

        # ⚖️ WEIGHTED CONFLUENCE SCORING (Not all factors are equal!)
        # Weights based on historical performance and institutional importance
        FACTOR_WEIGHTS = {
            'divergence': 3.0,           # Strongest reversal signal
            'order_flow': 2.5,           # Institutional positioning
            'smart_money': 2.0,          # Order blocks & FVG
            'open_interest': 2.0,        # Leverage positioning
            'liquidation': 1.5,          # Price magnet effect
            'support_resistance': 1.5,   # Key levels
            'multi_timeframe': 1.5,      # Trend confirmation
            'volatility': 1.0,           # Market state
            'funding': 1.0,              # Overleveraged positioning
            'volume_profile': 0.8,       # Fair value
            'fibonacci': 0.5,            # Weaker signal
            'rsi_oversold': 0.5          # Weak alone
        }

        # Assign weights to factors based on keywords
        def get_factor_weight(factor_text: str) -> float:
            factor_lower = factor_text.lower()
            if 'divergence' in factor_lower:
                return FACTOR_WEIGHTS['divergence']
            elif 'order flow' in factor_lower:
                return FACTOR_WEIGHTS['order_flow']
            elif 'smart money' in factor_lower or 'order block' in factor_lower:
                return FACTOR_WEIGHTS['smart_money']
            elif 'open interest' in factor_lower or 'oi' in factor_lower:
                return FACTOR_WEIGHTS['open_interest']
            elif 'liquidation' in factor_lower:
                return FACTOR_WEIGHTS['liquidation']
            elif 'support' in factor_lower or 'resistance' in factor_lower:
                return FACTOR_WEIGHTS['support_resistance']
            elif 'timeframe' in factor_lower or 'aligned' in factor_lower:
                return FACTOR_WEIGHTS['multi_timeframe']
            elif 'volatility' in factor_lower or 'breakout' in factor_lower:
                return FACTOR_WEIGHTS['volatility']
            elif 'funding' in factor_lower:
                return FACTOR_WEIGHTS['funding']
            elif 'poc' in factor_lower or 'fair value' in factor_lower:
                return FACTOR_WEIGHTS['volume_profile']
            elif 'fibonacci' in factor_lower or 'fib' in factor_lower:
                return FACTOR_WEIGHTS['fibonacci']
            elif 'oversold' in factor_lower or 'overbought' in factor_lower:
                return FACTOR_WEIGHTS['rsi_oversold']
            else:
                return 1.0  # Default weight

        # Calculate weighted scores
        bullish_weighted = sum(get_factor_weight(f) for f in bullish_factors)
        bearish_weighted = sum(get_factor_weight(f) for f in bearish_factors)

        # Determine confluence for requested side
        if side == 'LONG':
            confluence_count = len(bullish_factors)
            conflicting_count = len(bearish_factors)
            factors = bullish_factors
            side_weight = bullish_weighted
            opposite_weight = bearish_weighted
        else:  # SHORT
            confluence_count = len(bearish_factors)
            conflicting_count = len(bullish_factors)
            factors = bearish_factors
            side_weight = bearish_weighted
            opposite_weight = bullish_weighted

        # Weighted confluence score (0-100)
        total_weight = side_weight + opposite_weight
        if total_weight > 0:
            confluence_score = (side_weight / total_weight) * 100
        else:
            confluence_score = 50  # Neutral

        # Determine strength based on weighted score
        if confluence_score >= 80:
            strength = 'very_strong'
        elif confluence_score >= 70:
            strength = 'strong'
        elif confluence_score >= 60:
            strength = 'moderate'
        elif confluence_score >= 50:
            strength = 'weak'
        else:
            strength = 'very_weak'

        return {
            'score': float(confluence_score),
            'confluence_count': confluence_count,
            'conflicting_count': conflicting_count,
            'factors': factors,
            'strength': strength,
            'weighted_score': float(side_weight),  # NEW: Absolute weighted score
            'total_weight': float(total_weight)    # NEW: Total weight for reference
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
        # 🧪 TEST MODE: Loosened from >= 3 to >= 2 (accept 2/4 timeframe agreement)
        if bullish_count >= 2:
            trend_alignment = 'strong_bullish'
            alignment_score = (bullish_count / 4) * 100
        elif bearish_count >= 2:
            trend_alignment = 'strong_bearish'
            alignment_score = (bearish_count / 4) * 100
        else:
            trend_alignment = 'mixed'
            alignment_score = 50

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
        'trend': 'unknown',
        # 🎯 TIER 2 CONSERVATIVE: New indicator defaults
        'ichimoku_conversion': 0.0,
        'ichimoku_base': 0.0,
        'ichimoku_span_a': 0.0,
        'ichimoku_span_b': 0.0,
        'ichimoku_cloud_color': 'neutral',
        'ichimoku_price_vs_cloud': 'IN_CLOUD',
        'ichimoku_signal': 'NEUTRAL',
        'ichimoku_trend_strength': 'WEAK',
        'stoch_rsi_value': 50.0,
        'stoch_rsi_k_new': 50.0,
        'stoch_rsi_d_new': 50.0,
        'stoch_rsi_signal_new': 'NEUTRAL',
        'stoch_rsi_zone': 'NEUTRAL',
        'mfi_value': 50.0,
        'mfi_signal_new': 'NEUTRAL',
        'mfi_zone': 'NEUTRAL',
        'mfi_buying_pressure': 0.5
    }
"""
🎯 CONSERVATIVE STRATEGY: New Professional Indicators
Added for improved trade quality and win rate.

Indicators:
1. Ichimoku Cloud - Ultimate trend indicator
2. Stochastic RSI - Better overbought/oversold detection
3. Money Flow Index (MFI) - Volume-weighted RSI
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


def calculate_ichimoku(ohlcv: List) -> Dict[str, Any]:
    """
    Calculate Ichimoku Cloud indicator (Ultimate trend system).

    Components:
    - Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    - Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    - Leading Span A (Senkou Span A): (Conversion + Base) / 2, shifted 26 periods ahead
    - Leading Span B (Senkou Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
    - Lagging Span (Chikou Span): Close, shifted 26 periods back

    Signals:
    - Price above cloud = STRONG BULLISH
    - Price below cloud = STRONG BEARISH
    - Price in cloud = NEUTRAL/CONSOLIDATION
    - Conversion crosses Base UP = BUY signal
    - Conversion crosses Base DOWN = SELL signal
    - Cloud color (Span A > Span B) = Bullish cloud
    - Cloud color (Span A < Span B) = Bearish cloud

    Args:
        ohlcv: List of OHLCV data [timestamp, open, high, low, close, volume]

    Returns:
        Dict with Ichimoku components and signals
    """
    try:
        if len(ohlcv) < 52:  # Need minimum 52 periods for full calculation
            return {
                'conversion_line': 0.0,
                'base_line': 0.0,
                'leading_span_a': 0.0,
                'leading_span_b': 0.0,
                'lagging_span': 0.0,
                'cloud_color': 'neutral',
                'price_vs_cloud': 'IN_CLOUD',
                'signal': 'NEUTRAL',
                'trend_strength': 'WEAK'
            }

        # Convert to numpy arrays
        highs = np.array([candle[2] for candle in ohlcv])
        lows = np.array([candle[3] for candle in ohlcv])
        closes = np.array([candle[4] for candle in ohlcv])

        # Conversion Line (Tenkan-sen): 9-period
        conversion_high = np.max(highs[-9:])
        conversion_low = np.min(lows[-9:])
        conversion_line = (conversion_high + conversion_low) / 2

        # Base Line (Kijun-sen): 26-period
        base_high = np.max(highs[-26:])
        base_low = np.min(lows[-26:])
        base_line = (base_high + base_low) / 2

        # Leading Span A (Senkou Span A): (Conversion + Base) / 2
        leading_span_a = (conversion_line + base_line) / 2

        # Leading Span B (Senkou Span B): 52-period
        span_b_high = np.max(highs[-52:])
        span_b_low = np.min(lows[-52:])
        leading_span_b = (span_b_high + span_b_low) / 2

        # Lagging Span (Chikou Span): Current close
        lagging_span = closes[-1]

        # Current price
        current_price = closes[-1]

        # Determine cloud color
        if leading_span_a > leading_span_b:
            cloud_color = 'BULLISH'  # Green cloud
        elif leading_span_a < leading_span_b:
            cloud_color = 'BEARISH'  # Red cloud
        else:
            cloud_color = 'NEUTRAL'

        # Determine price position relative to cloud
        cloud_top = max(leading_span_a, leading_span_b)
        cloud_bottom = min(leading_span_a, leading_span_b)

        if current_price > cloud_top:
            price_vs_cloud = 'ABOVE_CLOUD'  # Bullish
            trend_strength = 'STRONG'
        elif current_price < cloud_bottom:
            price_vs_cloud = 'BELOW_CLOUD'  # Bearish
            trend_strength = 'STRONG'
        else:
            price_vs_cloud = 'IN_CLOUD'  # Neutral/consolidation
            trend_strength = 'WEAK'

        # Determine signal based on multiple factors
        signal = 'NEUTRAL'

        # Check conversion-base crossover (if we have previous values)
        if len(ohlcv) >= 53:
            prev_conversion_high = np.max(highs[-10:-1])
            prev_conversion_low = np.min(lows[-10:-1])
            prev_conversion = (prev_conversion_high + prev_conversion_low) / 2

            prev_base_high = np.max(highs[-27:-1])
            prev_base_low = np.min(lows[-27:-1])
            prev_base = (prev_base_high + prev_base_low) / 2

            # Bullish cross: Conversion crosses above Base
            if conversion_line > base_line and prev_conversion <= prev_base:
                signal = 'STRONG_BUY'
            # Bearish cross: Conversion crosses below Base
            elif conversion_line < base_line and prev_conversion >= prev_base:
                signal = 'STRONG_SELL'
            # No cross, use position signals
            elif price_vs_cloud == 'ABOVE_CLOUD' and cloud_color == 'BULLISH':
                signal = 'BUY'
            elif price_vs_cloud == 'BELOW_CLOUD' and cloud_color == 'BEARISH':
                signal = 'SELL'

        return {
            'conversion_line': float(conversion_line),
            'base_line': float(base_line),
            'leading_span_a': float(leading_span_a),
            'leading_span_b': float(leading_span_b),
            'lagging_span': float(lagging_span),
            'cloud_color': cloud_color,
            'price_vs_cloud': price_vs_cloud,
            'signal': signal,
            'trend_strength': trend_strength,
            'cloud_thickness': float(abs(leading_span_a - leading_span_b)),  # Thicker cloud = stronger support/resistance
        }

    except Exception as e:
        return {
            'conversion_line': 0.0,
            'base_line': 0.0,
            'leading_span_a': 0.0,
            'leading_span_b': 0.0,
            'lagging_span': 0.0,
            'cloud_color': 'neutral',
            'price_vs_cloud': 'IN_CLOUD',
            'signal': 'NEUTRAL',
            'trend_strength': 'WEAK',
            'error': str(e)
        }


def calculate_stochastic_rsi(ohlcv: List, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Dict[str, Any]:
    """
    Calculate Stochastic RSI (More sensitive than regular RSI).

    StochRSI applies the Stochastic oscillator formula to RSI values.

    Formula:
    1. Calculate RSI
    2. StochRSI = (RSI - RSI_min) / (RSI_max - RSI_min)
    3. %K = SMA(StochRSI, smooth_k)
    4. %D = SMA(%K, smooth_d)

    Signals:
    - StochRSI < 20 = Oversold (potential BUY)
    - StochRSI > 80 = Overbought (potential SELL)
    - %K crosses %D upward = BUY signal
    - %K crosses %D downward = SELL signal

    Args:
        ohlcv: List of OHLCV data
        period: RSI period (default 14)
        smooth_k: %K smoothing period (default 3)
        smooth_d: %D smoothing period (default 3)

    Returns:
        Dict with StochRSI, %K, %D values and signal
    """
    try:
        if len(ohlcv) < period + smooth_k + smooth_d + 10:
            return {
                'stoch_rsi': 50.0,
                'k': 50.0,
                'd': 50.0,
                'signal': 'NEUTRAL',
                'zone': 'NEUTRAL'
            }

        # Calculate RSI first
        closes = np.array([candle[4] for candle in ohlcv])

        # Calculate price changes
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Calculate average gains and losses
        avg_gains = pd.Series(gains).rolling(window=period).mean().values
        avg_losses = pd.Series(losses).rolling(window=period).mean().values

        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        # Remove NaN values
        rsi = rsi[~np.isnan(rsi)]

        if len(rsi) < period:
            return {
                'stoch_rsi': 50.0,
                'k': 50.0,
                'd': 50.0,
                'signal': 'NEUTRAL',
                'zone': 'NEUTRAL'
            }

        # Calculate Stochastic RSI
        stoch_rsi_values = []
        for i in range(period, len(rsi)):
            rsi_window = rsi[i-period:i]
            rsi_min = np.min(rsi_window)
            rsi_max = np.max(rsi_window)

            if rsi_max == rsi_min:
                stoch_rsi_values.append(50.0)
            else:
                stoch_value = ((rsi[i] - rsi_min) / (rsi_max - rsi_min)) * 100
                stoch_rsi_values.append(stoch_value)

        if len(stoch_rsi_values) < smooth_k + smooth_d:
            return {
                'stoch_rsi': 50.0,
                'k': 50.0,
                'd': 50.0,
                'signal': 'NEUTRAL',
                'zone': 'NEUTRAL'
            }

        # Calculate %K (smoothed StochRSI)
        k_values = pd.Series(stoch_rsi_values).rolling(window=smooth_k).mean().values

        # Calculate %D (smoothed %K)
        d_values = pd.Series(k_values).rolling(window=smooth_d).mean().values

        # Get current values
        current_stoch_rsi = stoch_rsi_values[-1]
        current_k = k_values[-1] if not np.isnan(k_values[-1]) else 50.0
        current_d = d_values[-1] if not np.isnan(d_values[-1]) else 50.0

        # Determine zone
        if current_stoch_rsi < 20:
            zone = 'OVERSOLD'
        elif current_stoch_rsi > 80:
            zone = 'OVERBOUGHT'
        else:
            zone = 'NEUTRAL'

        # Determine signal
        signal = 'NEUTRAL'

        if len(k_values) >= 2 and len(d_values) >= 2:
            prev_k = k_values[-2]
            prev_d = d_values[-2]

            # Bullish crossover: %K crosses above %D
            if current_k > current_d and prev_k <= prev_d:
                if zone == 'OVERSOLD':
                    signal = 'STRONG_BUY'  # Best signal - oversold + bullish cross
                else:
                    signal = 'BUY'
            # Bearish crossover: %K crosses below %D
            elif current_k < current_d and prev_k >= prev_d:
                if zone == 'OVERBOUGHT':
                    signal = 'STRONG_SELL'  # Best signal - overbought + bearish cross
                else:
                    signal = 'SELL'

        return {
            'stoch_rsi': float(current_stoch_rsi),
            'k': float(current_k),
            'd': float(current_d),
            'signal': signal,
            'zone': zone
        }

    except Exception as e:
        return {
            'stoch_rsi': 50.0,
            'k': 50.0,
            'd': 50.0,
            'signal': 'NEUTRAL',
            'zone': 'NEUTRAL',
            'error': str(e)
        }


def calculate_mfi(ohlcv: List, period: int = 14) -> Dict[str, Any]:
    """
    Calculate Money Flow Index (Volume-weighted RSI).

    MFI is similar to RSI but incorporates volume, making it more reliable
    for detecting overbought/oversold conditions with institutional activity.

    Formula:
    1. Typical Price = (High + Low + Close) / 3
    2. Money Flow = Typical Price × Volume
    3. Positive Money Flow = Sum of money flow when price increases
    4. Negative Money Flow = Sum of money flow when price decreases
    5. Money Ratio = Positive MF / Negative MF
    6. MFI = 100 - (100 / (1 + Money Ratio))

    Signals:
    - MFI < 20 = Oversold + Strong buying opportunity
    - MFI > 80 = Overbought + Strong selling opportunity
    - MFI 20-40 = Mild oversold
    - MFI 60-80 = Mild overbought
    - Divergence with price = Reversal warning

    Args:
        ohlcv: List of OHLCV data [timestamp, open, high, low, close, volume]
        period: MFI period (default 14)

    Returns:
        Dict with MFI value, signal, and zone
    """
    try:
        if len(ohlcv) < period + 2:
            return {
                'mfi': 50.0,
                'signal': 'NEUTRAL',
                'zone': 'NEUTRAL',
                'buying_pressure': 0.5
            }

        # Extract OHLCV data
        highs = np.array([candle[2] for candle in ohlcv])
        lows = np.array([candle[3] for candle in ohlcv])
        closes = np.array([candle[4] for candle in ohlcv])
        volumes = np.array([candle[5] for candle in ohlcv])

        # Calculate Typical Price
        typical_prices = (highs + lows + closes) / 3

        # Calculate Money Flow
        money_flow = typical_prices * volumes

        # Calculate positive and negative money flow
        positive_mf = []
        negative_mf = []

        for i in range(1, len(typical_prices)):
            if typical_prices[i] > typical_prices[i-1]:
                positive_mf.append(money_flow[i])
                negative_mf.append(0)
            elif typical_prices[i] < typical_prices[i-1]:
                positive_mf.append(0)
                negative_mf.append(money_flow[i])
            else:
                positive_mf.append(0)
                negative_mf.append(0)

        # Calculate MFI for the period
        mfi_values = []

        for i in range(period, len(positive_mf) + 1):
            pos_sum = sum(positive_mf[i-period:i])
            neg_sum = sum(negative_mf[i-period:i])

            if neg_sum == 0:
                mfi = 100.0
            else:
                money_ratio = pos_sum / neg_sum
                mfi = 100 - (100 / (1 + money_ratio))

            mfi_values.append(mfi)

        if not mfi_values:
            return {
                'mfi': 50.0,
                'signal': 'NEUTRAL',
                'zone': 'NEUTRAL',
                'buying_pressure': 0.5
            }

        current_mfi = mfi_values[-1]

        # Calculate buying pressure (0.0 - 1.0)
        buying_pressure = current_mfi / 100.0

        # Determine zone
        if current_mfi < 20:
            zone = 'STRONG_OVERSOLD'
            signal = 'STRONG_BUY'
        elif current_mfi < 40:
            zone = 'OVERSOLD'
            signal = 'BUY'
        elif current_mfi > 80:
            zone = 'STRONG_OVERBOUGHT'
            signal = 'STRONG_SELL'
        elif current_mfi > 60:
            zone = 'OVERBOUGHT'
            signal = 'SELL'
        else:
            zone = 'NEUTRAL'
            signal = 'NEUTRAL'

        return {
            'mfi': float(current_mfi),
            'signal': signal,
            'zone': zone,
            'buying_pressure': float(buying_pressure)
        }

    except Exception as e:
        return {
            'mfi': 50.0,
            'signal': 'NEUTRAL',
            'zone': 'NEUTRAL',
            'buying_pressure': 0.5,
            'error': str(e)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🆕 v4.4.0 - NEW PROFESSIONAL INDICATORS FOR CONFLUENCE SCORING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def calculate_bb_squeeze(ohlcv: List, bb_period: int = 20, bb_std: float = 2.0,
                         kc_period: int = 20, kc_mult: float = 1.5) -> Dict[str, Any]:
    """
    Detect Bollinger Band Squeeze (TTM Squeeze concept).

    SQUEEZE = Low volatility → Breakout imminent!
    When BB is INSIDE Keltner Channel = Squeeze is ON (consolidation)
    When BB expands OUTSIDE KC = Squeeze fires (breakout)

    Professional traders use this to:
    - Identify consolidation periods before big moves
    - Time entries right before breakouts
    - Avoid trading during low volatility chop

    Args:
        ohlcv: List of OHLCV data
        bb_period: Bollinger Band period (default 20)
        bb_std: Bollinger Band standard deviation (default 2.0)
        kc_period: Keltner Channel period (default 20)
        kc_mult: Keltner Channel ATR multiplier (default 1.5)

    Returns:
        Dict with squeeze status, momentum, and signal
    """
    try:
        if len(ohlcv) < max(bb_period, kc_period) + 20:
            return {
                'squeeze_on': False,
                'squeeze_off': False,
                'momentum': 0.0,
                'momentum_direction': 'neutral',
                'signal': 'NEUTRAL',
                'bars_in_squeeze': 0
            }

        # Extract OHLC data
        highs = np.array([candle[2] for candle in ohlcv], dtype=float)
        lows = np.array([candle[3] for candle in ohlcv], dtype=float)
        closes = np.array([candle[4] for candle in ohlcv], dtype=float)

        # Calculate Bollinger Bands
        bb_sma = pd.Series(closes).rolling(window=bb_period).mean()
        bb_std_val = pd.Series(closes).rolling(window=bb_period).std()
        bb_upper = bb_sma + (bb_std * bb_std_val)
        bb_lower = bb_sma - (bb_std * bb_std_val)

        # Calculate Keltner Channel (using ATR)
        tr = np.maximum(highs[1:] - lows[1:],
                       np.maximum(np.abs(highs[1:] - closes[:-1]),
                                 np.abs(lows[1:] - closes[:-1])))
        tr = np.insert(tr, 0, highs[0] - lows[0])
        atr = pd.Series(tr).rolling(window=kc_period).mean()

        kc_middle = pd.Series(closes).rolling(window=kc_period).mean()
        kc_upper = kc_middle + (kc_mult * atr)
        kc_lower = kc_middle - (kc_mult * atr)

        # Squeeze detection: BB inside KC = Squeeze ON
        squeeze_on = (bb_lower.iloc[-1] > kc_lower.iloc[-1]) and (bb_upper.iloc[-1] < kc_upper.iloc[-1])
        squeeze_off = not squeeze_on

        # Count bars in squeeze
        bars_in_squeeze = 0
        for i in range(len(closes) - 1, -1, -1):
            if pd.notna(bb_lower.iloc[i]) and pd.notna(kc_lower.iloc[i]):
                if (bb_lower.iloc[i] > kc_lower.iloc[i]) and (bb_upper.iloc[i] < kc_upper.iloc[i]):
                    bars_in_squeeze += 1
                else:
                    break

        # Calculate momentum (Linear Regression of price - average)
        # Simplified: Use price vs middle band
        momentum = (closes[-1] - kc_middle.iloc[-1]) / kc_middle.iloc[-1] * 100

        # Momentum direction
        if len(closes) >= 2:
            prev_momentum = (closes[-2] - kc_middle.iloc[-2]) / kc_middle.iloc[-2] * 100
            if momentum > prev_momentum:
                momentum_direction = 'increasing'
            elif momentum < prev_momentum:
                momentum_direction = 'decreasing'
            else:
                momentum_direction = 'flat'
        else:
            momentum_direction = 'neutral'

        # Signal generation
        signal = 'NEUTRAL'
        if squeeze_off and bars_in_squeeze >= 5:
            # Squeeze just fired after being on for 5+ bars
            if momentum > 0:
                signal = 'STRONG_BUY'  # Breakout UP
            else:
                signal = 'STRONG_SELL'  # Breakout DOWN
        elif squeeze_on:
            signal = 'WAIT'  # Wait for squeeze to fire
        elif momentum > 0.5:
            signal = 'BUY'
        elif momentum < -0.5:
            signal = 'SELL'

        return {
            'squeeze_on': squeeze_on,
            'squeeze_off': squeeze_off,
            'momentum': float(momentum),
            'momentum_direction': momentum_direction,
            'signal': signal,
            'bars_in_squeeze': bars_in_squeeze,
            'bb_width_pct': float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_sma.iloc[-1] * 100),
            'kc_width_pct': float((kc_upper.iloc[-1] - kc_lower.iloc[-1]) / kc_middle.iloc[-1] * 100)
        }

    except Exception as e:
        return {
            'squeeze_on': False,
            'squeeze_off': False,
            'momentum': 0.0,
            'momentum_direction': 'neutral',
            'signal': 'NEUTRAL',
            'bars_in_squeeze': 0,
            'error': str(e)
        }


def calculate_ema_stack(ohlcv: List, periods: List[int] = None) -> Dict[str, Any]:
    """
    Analyze EMA Stack/Fan for trend strength.

    EMA STACK = Multiple EMAs in perfect order
    - Bullish: EMA8 > EMA21 > EMA34 > EMA55 > EMA89 (all stacked perfectly)
    - Bearish: EMA8 < EMA21 < EMA34 < EMA55 < EMA89 (inverse stack)
    - Tangled: EMAs crossing = No clear trend, choppy market

    Professional use:
    - Perfect stack = Strong trend, ride it with trailing stop
    - Tangled EMAs = Avoid trading, wait for clarity
    - Stack forming = Early trend entry opportunity

    Args:
        ohlcv: List of OHLCV data
        periods: EMA periods to use (default: [8, 21, 34, 55, 89] - Fibonacci)

    Returns:
        Dict with stack status, trend strength, and signal
    """
    try:
        if periods is None:
            periods = [8, 21, 34, 55, 89]  # Fibonacci-based EMAs

        if len(ohlcv) < max(periods) + 10:
            return {
                'stack_type': 'unknown',
                'trend_strength': 0.0,
                'is_perfect_stack': False,
                'stack_score': 0,
                'signal': 'NEUTRAL',
                'ema_values': {}
            }

        closes = np.array([candle[4] for candle in ohlcv], dtype=float)
        current_price = closes[-1]

        # Calculate all EMAs
        ema_values = {}
        for period in periods:
            ema = pd.Series(closes).ewm(span=period, adjust=False).mean()
            ema_values[f'ema_{period}'] = float(ema.iloc[-1])

        # Check stack order
        ema_list = [ema_values[f'ema_{p}'] for p in periods]

        # Count correct orderings
        bullish_order = 0
        bearish_order = 0

        for i in range(len(ema_list) - 1):
            if ema_list[i] > ema_list[i + 1]:
                bullish_order += 1
            elif ema_list[i] < ema_list[i + 1]:
                bearish_order += 1

        total_pairs = len(ema_list) - 1

        # Determine stack type
        if bullish_order == total_pairs:
            stack_type = 'bullish'
            is_perfect_stack = True
            stack_score = 100
        elif bearish_order == total_pairs:
            stack_type = 'bearish'
            is_perfect_stack = True
            stack_score = 100
        elif bullish_order > bearish_order:
            stack_type = 'bullish_forming'
            is_perfect_stack = False
            stack_score = int((bullish_order / total_pairs) * 100)
        elif bearish_order > bullish_order:
            stack_type = 'bearish_forming'
            is_perfect_stack = False
            stack_score = int((bearish_order / total_pairs) * 100)
        else:
            stack_type = 'tangled'
            is_perfect_stack = False
            stack_score = 0

        # Calculate trend strength (0-100)
        # Based on: stack quality + price position relative to EMAs
        above_count = sum(1 for ema in ema_list if current_price > ema)
        below_count = sum(1 for ema in ema_list if current_price < ema)

        if above_count == len(ema_list):
            price_position = 'above_all'
            trend_strength = stack_score * 1.0
        elif below_count == len(ema_list):
            price_position = 'below_all'
            trend_strength = stack_score * 1.0
        else:
            price_position = 'mixed'
            trend_strength = stack_score * 0.7

        # Generate signal
        if stack_type == 'bullish' and is_perfect_stack:
            signal = 'STRONG_BUY'
        elif stack_type == 'bearish' and is_perfect_stack:
            signal = 'STRONG_SELL'
        elif stack_type == 'bullish_forming' and stack_score >= 75:
            signal = 'BUY'
        elif stack_type == 'bearish_forming' and stack_score >= 75:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'

        return {
            'stack_type': stack_type,
            'trend_strength': float(trend_strength),
            'is_perfect_stack': is_perfect_stack,
            'stack_score': stack_score,
            'signal': signal,
            'ema_values': ema_values,
            'price_position': price_position,
            'bullish_pairs': bullish_order,
            'bearish_pairs': bearish_order
        }

    except Exception as e:
        return {
            'stack_type': 'unknown',
            'trend_strength': 0.0,
            'is_perfect_stack': False,
            'stack_score': 0,
            'signal': 'NEUTRAL',
            'ema_values': {},
            'error': str(e)
        }


def calculate_adx(ohlcv: List, period: int = 14) -> Dict[str, Any]:
    """
    Calculate ADX (Average Directional Index) - THE trend strength indicator.

    ADX measures STRENGTH of trend, not direction!
    - ADX < 20 = Weak/No trend (ranging market)
    - ADX 20-25 = Trend emerging
    - ADX 25-50 = Strong trend
    - ADX 50-75 = Very strong trend
    - ADX > 75 = Extremely strong (rare, often exhaustion)

    DI+ and DI- indicate direction:
    - DI+ > DI- = Bullish
    - DI- > DI+ = Bearish

    Professional use:
    - ADX rising = Trend strengthening, hold position
    - ADX falling = Trend weakening, tighten stops
    - ADX < 20 = Avoid trend-following strategies

    Args:
        ohlcv: List of OHLCV data
        period: ADX period (default 14)

    Returns:
        Dict with ADX, DI+, DI-, trend info
    """
    try:
        if len(ohlcv) < period * 2 + 10:
            return {
                'adx': 20.0,
                'di_plus': 0.0,
                'di_minus': 0.0,
                'trend_strength': 'weak',
                'trend_direction': 'neutral',
                'signal': 'NEUTRAL'
            }

        highs = np.array([candle[2] for candle in ohlcv], dtype=float)
        lows = np.array([candle[3] for candle in ohlcv], dtype=float)
        closes = np.array([candle[4] for candle in ohlcv], dtype=float)

        # Calculate True Range
        tr = np.zeros(len(ohlcv))
        tr[0] = highs[0] - lows[0]
        for i in range(1, len(ohlcv)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)

        # Calculate Directional Movement
        dm_plus = np.zeros(len(ohlcv))
        dm_minus = np.zeros(len(ohlcv))

        for i in range(1, len(ohlcv)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move

        # Smooth the values using Wilder's smoothing
        atr = pd.Series(tr).rolling(window=period).mean()
        smooth_dm_plus = pd.Series(dm_plus).rolling(window=period).mean()
        smooth_dm_minus = pd.Series(dm_minus).rolling(window=period).mean()

        # Calculate DI+ and DI-
        di_plus = 100 * smooth_dm_plus / (atr + 1e-10)
        di_minus = 100 * smooth_dm_minus / (atr + 1e-10)

        # Calculate DX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)

        # Calculate ADX (smoothed DX)
        adx = pd.Series(dx).rolling(window=period).mean()

        # Get current values
        current_adx = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 20.0
        current_di_plus = float(di_plus.iloc[-1]) if not np.isnan(di_plus.iloc[-1]) else 0.0
        current_di_minus = float(di_minus.iloc[-1]) if not np.isnan(di_minus.iloc[-1]) else 0.0

        # Determine trend strength
        if current_adx < 20:
            trend_strength = 'weak'
        elif current_adx < 25:
            trend_strength = 'emerging'
        elif current_adx < 50:
            trend_strength = 'strong'
        elif current_adx < 75:
            trend_strength = 'very_strong'
        else:
            trend_strength = 'extreme'

        # Determine trend direction
        if current_di_plus > current_di_minus:
            trend_direction = 'bullish'
        elif current_di_minus > current_di_plus:
            trend_direction = 'bearish'
        else:
            trend_direction = 'neutral'

        # ADX trend (rising or falling)
        if len(adx) >= 5:
            adx_slope = (adx.iloc[-1] - adx.iloc[-5]) / 5
            if adx_slope > 0.5:
                adx_trend = 'rising'
            elif adx_slope < -0.5:
                adx_trend = 'falling'
            else:
                adx_trend = 'flat'
        else:
            adx_trend = 'unknown'

        # Generate signal
        signal = 'NEUTRAL'
        if current_adx >= 25:
            if trend_direction == 'bullish' and adx_trend != 'falling':
                signal = 'BUY' if current_adx < 50 else 'STRONG_BUY'
            elif trend_direction == 'bearish' and adx_trend != 'falling':
                signal = 'SELL' if current_adx < 50 else 'STRONG_SELL'

        return {
            'adx': current_adx,
            'di_plus': current_di_plus,
            'di_minus': current_di_minus,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'adx_trend': adx_trend,
            'signal': signal,
            'di_diff': current_di_plus - current_di_minus
        }

    except Exception as e:
        return {
            'adx': 20.0,
            'di_plus': 0.0,
            'di_minus': 0.0,
            'trend_strength': 'weak',
            'trend_direction': 'neutral',
            'signal': 'NEUTRAL',
            'error': str(e)
        }


def calculate_macd_divergence(ohlcv: List) -> Dict[str, Any]:
    """
    Detect MACD Divergence - Professional reversal indicator.

    MACD Divergence = Price and MACD moving in opposite directions
    - Bullish Divergence: Price lower low + MACD higher low → Reversal UP
    - Bearish Divergence: Price higher high + MACD lower high → Reversal DOWN
    - Hidden Bullish: Price higher low + MACD lower low → Trend continuation UP
    - Hidden Bearish: Price lower high + MACD higher high → Trend continuation DOWN

    Args:
        ohlcv: List of OHLCV data

    Returns:
        Dict with divergence type, strength, and signal
    """
    try:
        if len(ohlcv) < 50:
            return {
                'has_divergence': False,
                'divergence_type': None,
                'strength': 0.0,
                'signal': 'NEUTRAL'
            }

        closes = np.array([candle[4] for candle in ohlcv], dtype=float)
        lows = np.array([candle[3] for candle in ohlcv], dtype=float)
        highs = np.array([candle[2] for candle in ohlcv], dtype=float)

        # Calculate MACD
        ema_12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
        ema_26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        # Find price lows/highs and MACD lows/highs in last 20 bars
        lookback = min(20, len(closes) - 5)
        recent_closes = closes[-lookback:]
        recent_lows = lows[-lookback:]
        recent_highs = highs[-lookback:]
        recent_macd = histogram.iloc[-lookback:].values

        # Find swing lows (for bullish divergence)
        price_lows = []
        macd_lows = []
        for i in range(2, len(recent_lows) - 2):
            if (recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and
                recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]):
                price_lows.append((i, recent_lows[i]))
                macd_lows.append((i, recent_macd[i]))

        # Find swing highs (for bearish divergence)
        price_highs = []
        macd_highs = []
        for i in range(2, len(recent_highs) - 2):
            if (recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i-2] and
                recent_highs[i] > recent_highs[i+1] and recent_highs[i] > recent_highs[i+2]):
                price_highs.append((i, recent_highs[i]))
                macd_highs.append((i, recent_macd[i]))

        # Check for BULLISH divergence (price lower low, MACD higher low)
        if len(price_lows) >= 2 and len(macd_lows) >= 2:
            last_price_low = price_lows[-1][1]
            prev_price_low = price_lows[-2][1]
            last_macd_low = macd_lows[-1][1]
            prev_macd_low = macd_lows[-2][1]

            if last_price_low < prev_price_low and last_macd_low > prev_macd_low:
                strength = min(abs(last_macd_low - prev_macd_low) / (abs(prev_macd_low) + 1e-10), 1.0)
                return {
                    'has_divergence': True,
                    'divergence_type': 'bullish',
                    'strength': float(strength),
                    'signal': 'STRONG_BUY',
                    'details': f'Price: {prev_price_low:.4f}→{last_price_low:.4f}, MACD: {prev_macd_low:.4f}→{last_macd_low:.4f}'
                }

            # Hidden bullish (price higher low, MACD lower low)
            if last_price_low > prev_price_low and last_macd_low < prev_macd_low:
                strength = min(abs(last_macd_low - prev_macd_low) / (abs(prev_macd_low) + 1e-10), 1.0) * 0.7
                return {
                    'has_divergence': True,
                    'divergence_type': 'hidden_bullish',
                    'strength': float(strength),
                    'signal': 'BUY',
                    'details': 'Hidden bullish - trend continuation'
                }

        # Check for BEARISH divergence (price higher high, MACD lower high)
        if len(price_highs) >= 2 and len(macd_highs) >= 2:
            last_price_high = price_highs[-1][1]
            prev_price_high = price_highs[-2][1]
            last_macd_high = macd_highs[-1][1]
            prev_macd_high = macd_highs[-2][1]

            if last_price_high > prev_price_high and last_macd_high < prev_macd_high:
                strength = min(abs(last_macd_high - prev_macd_high) / (abs(prev_macd_high) + 1e-10), 1.0)
                return {
                    'has_divergence': True,
                    'divergence_type': 'bearish',
                    'strength': float(strength),
                    'signal': 'STRONG_SELL',
                    'details': f'Price: {prev_price_high:.4f}→{last_price_high:.4f}, MACD: {prev_macd_high:.4f}→{last_macd_high:.4f}'
                }

            # Hidden bearish (price lower high, MACD higher high)
            if last_price_high < prev_price_high and last_macd_high > prev_macd_high:
                strength = min(abs(last_macd_high - prev_macd_high) / (abs(prev_macd_high) + 1e-10), 1.0) * 0.7
                return {
                    'has_divergence': True,
                    'divergence_type': 'hidden_bearish',
                    'strength': float(strength),
                    'signal': 'SELL',
                    'details': 'Hidden bearish - trend continuation'
                }

        return {
            'has_divergence': False,
            'divergence_type': None,
            'strength': 0.0,
            'signal': 'NEUTRAL'
        }

    except Exception as e:
        return {
            'has_divergence': False,
            'divergence_type': None,
            'strength': 0.0,
            'signal': 'NEUTRAL',
            'error': str(e)
        }


def calculate_enhanced_indicators(ohlcv: List) -> Dict[str, Any]:
    """
    Calculate ALL enhanced indicators for confluence scoring.

    This function combines all professional indicators into a single call
    for efficient use in the confluence scoring system.

    Args:
        ohlcv: List of OHLCV data

    Returns:
        Dict with all enhanced indicator values
    """
    try:
        result = {
            'bb_squeeze': calculate_bb_squeeze(ohlcv),
            'ema_stack': calculate_ema_stack(ohlcv),
            'adx': calculate_adx(ohlcv),
            'macd_divergence': calculate_macd_divergence(ohlcv),
            'rsi_divergence': detect_divergences(ohlcv),  # Already exists
        }

        # Add overall confluence signals
        signals = []

        # BB Squeeze signal
        if result['bb_squeeze'].get('signal') in ['STRONG_BUY', 'BUY']:
            signals.append(('bullish', 'bb_squeeze', result['bb_squeeze'].get('signal')))
        elif result['bb_squeeze'].get('signal') in ['STRONG_SELL', 'SELL']:
            signals.append(('bearish', 'bb_squeeze', result['bb_squeeze'].get('signal')))

        # EMA Stack signal
        if result['ema_stack'].get('signal') in ['STRONG_BUY', 'BUY']:
            signals.append(('bullish', 'ema_stack', result['ema_stack'].get('signal')))
        elif result['ema_stack'].get('signal') in ['STRONG_SELL', 'SELL']:
            signals.append(('bearish', 'ema_stack', result['ema_stack'].get('signal')))

        # ADX signal
        if result['adx'].get('signal') in ['STRONG_BUY', 'BUY']:
            signals.append(('bullish', 'adx', result['adx'].get('signal')))
        elif result['adx'].get('signal') in ['STRONG_SELL', 'SELL']:
            signals.append(('bearish', 'adx', result['adx'].get('signal')))

        # MACD Divergence signal
        if result['macd_divergence'].get('signal') in ['STRONG_BUY', 'BUY']:
            signals.append(('bullish', 'macd_div', result['macd_divergence'].get('signal')))
        elif result['macd_divergence'].get('signal') in ['STRONG_SELL', 'SELL']:
            signals.append(('bearish', 'macd_div', result['macd_divergence'].get('signal')))

        # RSI Divergence signal
        if result['rsi_divergence'].get('type') == 'bullish':
            signals.append(('bullish', 'rsi_div', 'BUY'))
        elif result['rsi_divergence'].get('type') == 'bearish':
            signals.append(('bearish', 'rsi_div', 'SELL'))

        # Count signals
        bullish_count = sum(1 for s in signals if s[0] == 'bullish')
        bearish_count = sum(1 for s in signals if s[0] == 'bearish')

        result['signal_summary'] = {
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'total_signals': len(signals),
            'signals': signals,
            'overall_bias': 'bullish' if bullish_count > bearish_count else ('bearish' if bearish_count > bullish_count else 'neutral')
        }

        return result

    except Exception as e:
        return {
            'bb_squeeze': {'signal': 'NEUTRAL'},
            'ema_stack': {'signal': 'NEUTRAL'},
            'adx': {'signal': 'NEUTRAL'},
            'macd_divergence': {'signal': 'NEUTRAL'},
            'rsi_divergence': {'has_divergence': False},
            'signal_summary': {
                'bullish_signals': 0,
                'bearish_signals': 0,
                'total_signals': 0,
                'signals': [],
                'overall_bias': 'neutral'
            },
            'error': str(e)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🆕 v4.5.0 - ADVANCED PROFESSIONAL INDICATORS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def calculate_vwap(ohlcv: List, include_bands: bool = True) -> Dict[str, Any]:
    """
    Calculate VWAP (Volume Weighted Average Price) - Institutional Trading Standard.

    VWAP = Σ(Price × Volume) / Σ(Volume)

    Professional Use:
    - Price > VWAP = Bullish bias (institutions buying)
    - Price < VWAP = Bearish bias (institutions selling)
    - Touches/bounces off VWAP = High probability trades
    - VWAP acts as dynamic support/resistance

    Why VWAP is critical for futures:
    - Shows where the "fair value" is for the session
    - Large traders use VWAP to evaluate their fill quality
    - Deviations from VWAP often mean-revert

    Args:
        ohlcv: List of OHLCV data
        include_bands: Whether to calculate standard deviation bands

    Returns:
        Dict with VWAP, bands, signal
    """
    try:
        if len(ohlcv) < 10:
            return {
                'vwap': 0.0,
                'upper_band_1': 0.0,
                'lower_band_1': 0.0,
                'upper_band_2': 0.0,
                'lower_band_2': 0.0,
                'deviation_pct': 0.0,
                'signal': 'NEUTRAL',
                'position': 'unknown'
            }

        # Extract data
        highs = np.array([candle[2] for candle in ohlcv], dtype=float)
        lows = np.array([candle[3] for candle in ohlcv], dtype=float)
        closes = np.array([candle[4] for candle in ohlcv], dtype=float)
        volumes = np.array([candle[5] for candle in ohlcv], dtype=float)

        # Typical Price = (High + Low + Close) / 3
        typical_price = (highs + lows + closes) / 3

        # VWAP = Cumulative(TP * Volume) / Cumulative(Volume)
        cumulative_tp_vol = np.cumsum(typical_price * volumes)
        cumulative_vol = np.cumsum(volumes)
        vwap = cumulative_tp_vol / (cumulative_vol + 1e-10)

        current_vwap = float(vwap[-1])
        current_price = closes[-1]

        # Calculate deviation from VWAP
        deviation_pct = ((current_price - current_vwap) / current_vwap) * 100

        # Calculate VWAP bands (standard deviation)
        if include_bands:
            squared_diff = (typical_price - vwap) ** 2
            variance = np.cumsum(squared_diff * volumes) / (cumulative_vol + 1e-10)
            std_dev = np.sqrt(variance)
            current_std = float(std_dev[-1])

            upper_band_1 = current_vwap + current_std
            lower_band_1 = current_vwap - current_std
            upper_band_2 = current_vwap + (2 * current_std)
            lower_band_2 = current_vwap - (2 * current_std)
        else:
            upper_band_1 = lower_band_1 = upper_band_2 = lower_band_2 = current_vwap

        # Determine position relative to VWAP
        if current_price > upper_band_2:
            position = 'far_above'
        elif current_price > upper_band_1:
            position = 'above'
        elif current_price > current_vwap:
            position = 'slightly_above'
        elif current_price < lower_band_2:
            position = 'far_below'
        elif current_price < lower_band_1:
            position = 'below'
        elif current_price < current_vwap:
            position = 'slightly_below'
        else:
            position = 'at_vwap'

        # Generate signal
        # Mean reversion from extremes + trend following near VWAP
        signal = 'NEUTRAL'

        if position == 'far_above':
            signal = 'SELL'  # Overextended, mean reversion likely
        elif position == 'far_below':
            signal = 'BUY'  # Oversold, bounce likely
        elif position == 'slightly_above' and deviation_pct < 0.5:
            signal = 'BUY'  # Just above VWAP, bullish bias
        elif position == 'slightly_below' and deviation_pct > -0.5:
            signal = 'SELL'  # Just below VWAP, bearish bias

        return {
            'vwap': current_vwap,
            'upper_band_1': float(upper_band_1),
            'lower_band_1': float(lower_band_1),
            'upper_band_2': float(upper_band_2),
            'lower_band_2': float(lower_band_2),
            'deviation_pct': float(deviation_pct),
            'signal': signal,
            'position': position,
            'current_price': float(current_price)
        }

    except Exception as e:
        return {
            'vwap': 0.0,
            'upper_band_1': 0.0,
            'lower_band_1': 0.0,
            'upper_band_2': 0.0,
            'lower_band_2': 0.0,
            'deviation_pct': 0.0,
            'signal': 'NEUTRAL',
            'position': 'unknown',
            'error': str(e)
        }


def calculate_stochastic_rsi(ohlcv: List, rsi_period: int = 14,
                              stoch_period: int = 14, k_smooth: int = 3,
                              d_smooth: int = 3) -> Dict[str, Any]:
    """
    Calculate Stochastic RSI - RSI + Stochastic combined for sensitivity.

    Stoch RSI = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)

    Why Stochastic RSI is better than plain RSI:
    - More sensitive to price changes
    - Better at detecting momentum shifts
    - K/D crossovers provide entry signals
    - Better overbought/oversold detection

    Professional Use:
    - %K crossing above %D = Bullish
    - %K crossing below %D = Bearish
    - Both above 80 = Overbought (look for short)
    - Both below 20 = Oversold (look for long)

    Args:
        ohlcv: OHLCV data
        rsi_period: RSI lookback period
        stoch_period: Stochastic lookback period
        k_smooth: %K smoothing
        d_smooth: %D smoothing (signal line)

    Returns:
        Dict with Stoch RSI values and signal
    """
    try:
        if len(ohlcv) < rsi_period + stoch_period + 10:
            return {
                'k': 50.0,
                'd': 50.0,
                'zone': 'neutral',
                'crossover': None,
                'signal': 'NEUTRAL'
            }

        closes = np.array([candle[4] for candle in ohlcv], dtype=float)

        # Calculate RSI first
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = pd.Series(gains).rolling(window=rsi_period).mean()
        avg_loss = pd.Series(losses).rolling(window=rsi_period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)

        # Calculate Stochastic of RSI
        rsi_series = pd.Series(rsi)
        lowest_rsi = rsi_series.rolling(window=stoch_period).min()
        highest_rsi = rsi_series.rolling(window=stoch_period).max()

        stoch_rsi = (rsi_series - lowest_rsi) / (highest_rsi - lowest_rsi + 1e-10) * 100

        # Smooth to get %K
        k_line = stoch_rsi.rolling(window=k_smooth).mean()

        # %D is SMA of %K
        d_line = k_line.rolling(window=d_smooth).mean()

        current_k = float(k_line.iloc[-1]) if not np.isnan(k_line.iloc[-1]) else 50.0
        current_d = float(d_line.iloc[-1]) if not np.isnan(d_line.iloc[-1]) else 50.0
        prev_k = float(k_line.iloc[-2]) if len(k_line) > 1 and not np.isnan(k_line.iloc[-2]) else current_k
        prev_d = float(d_line.iloc[-2]) if len(d_line) > 1 and not np.isnan(d_line.iloc[-2]) else current_d

        # Determine zone
        if current_k > 80 and current_d > 80:
            zone = 'overbought'
        elif current_k < 20 and current_d < 20:
            zone = 'oversold'
        elif current_k > 50:
            zone = 'bullish'
        else:
            zone = 'bearish'

        # Detect crossovers
        crossover = None
        if prev_k <= prev_d and current_k > current_d:
            crossover = 'bullish'  # %K crossed above %D
        elif prev_k >= prev_d and current_k < current_d:
            crossover = 'bearish'  # %K crossed below %D

        # Generate signal
        signal = 'NEUTRAL'
        if zone == 'oversold' and crossover == 'bullish':
            signal = 'STRONG_BUY'  # Oversold + bullish crossover
        elif zone == 'overbought' and crossover == 'bearish':
            signal = 'STRONG_SELL'  # Overbought + bearish crossover
        elif crossover == 'bullish':
            signal = 'BUY'
        elif crossover == 'bearish':
            signal = 'SELL'
        elif zone == 'oversold':
            signal = 'BUY'  # Prepare for reversal
        elif zone == 'overbought':
            signal = 'SELL'  # Prepare for reversal

        return {
            'k': current_k,
            'd': current_d,
            'zone': zone,
            'crossover': crossover,
            'signal': signal,
            'k_d_diff': current_k - current_d,
            'prev_k': prev_k,
            'prev_d': prev_d
        }

    except Exception as e:
        return {
            'k': 50.0,
            'd': 50.0,
            'zone': 'neutral',
            'crossover': None,
            'signal': 'NEUTRAL',
            'error': str(e)
        }


def calculate_williams_r(ohlcv: List, period: int = 14) -> Dict[str, Any]:
    """
    Calculate Williams %R - Fast momentum oscillator.

    Williams %R = ((Highest High - Close) / (Highest High - Lowest Low)) × -100

    Range: -100 to 0
    - Above -20 = Overbought
    - Below -80 = Oversold

    Professional Use:
    - More responsive than RSI (faster signals)
    - Good for catching reversals early
    - Best used with confirmation from other indicators
    - Divergences are powerful reversal signals

    Args:
        ohlcv: OHLCV data
        period: Lookback period

    Returns:
        Dict with Williams %R value and signal
    """
    try:
        if len(ohlcv) < period + 5:
            return {
                'williams_r': -50.0,
                'zone': 'neutral',
                'signal': 'NEUTRAL'
            }

        highs = np.array([candle[2] for candle in ohlcv], dtype=float)
        lows = np.array([candle[3] for candle in ohlcv], dtype=float)
        closes = np.array([candle[4] for candle in ohlcv], dtype=float)

        # Calculate Williams %R
        highest_high = pd.Series(highs).rolling(window=period).max()
        lowest_low = pd.Series(lows).rolling(window=period).min()

        williams_r = ((highest_high - closes) / (highest_high - lowest_low + 1e-10)) * -100

        current_wr = float(williams_r.iloc[-1]) if not np.isnan(williams_r.iloc[-1]) else -50.0
        prev_wr = float(williams_r.iloc[-2]) if len(williams_r) > 1 and not np.isnan(williams_r.iloc[-2]) else current_wr

        # Determine zone
        if current_wr > -20:
            zone = 'overbought'
        elif current_wr < -80:
            zone = 'oversold'
        elif current_wr > -50:
            zone = 'bullish'
        else:
            zone = 'bearish'

        # Detect momentum shift
        momentum_shift = None
        if prev_wr < -80 and current_wr > -80:
            momentum_shift = 'bullish'  # Exiting oversold
        elif prev_wr > -20 and current_wr < -20:
            momentum_shift = 'bearish'  # Exiting overbought

        # Generate signal
        signal = 'NEUTRAL'
        if zone == 'oversold' and momentum_shift == 'bullish':
            signal = 'STRONG_BUY'
        elif zone == 'overbought' and momentum_shift == 'bearish':
            signal = 'STRONG_SELL'
        elif zone == 'oversold':
            signal = 'BUY'
        elif zone == 'overbought':
            signal = 'SELL'

        return {
            'williams_r': current_wr,
            'zone': zone,
            'momentum_shift': momentum_shift,
            'signal': signal,
            'prev_williams_r': prev_wr
        }

    except Exception as e:
        return {
            'williams_r': -50.0,
            'zone': 'neutral',
            'signal': 'NEUTRAL',
            'error': str(e)
        }


def calculate_chaikin_money_flow(ohlcv: List, period: int = 20) -> Dict[str, Any]:
    """
    Calculate Chaikin Money Flow (CMF) - Volume-weighted buying/selling pressure.

    CMF = SUM(Money Flow Volume, period) / SUM(Volume, period)
    where Money Flow Volume = Money Flow Multiplier × Volume
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)

    Range: -1 to +1
    - CMF > 0.05 = Buying pressure (accumulation)
    - CMF < -0.05 = Selling pressure (distribution)
    - CMF > 0.25 = Strong buying
    - CMF < -0.25 = Strong selling

    Professional Use:
    - Confirms trend direction with volume
    - Divergences between CMF and price = reversal signals
    - CMF crossing zero = potential trend change

    Args:
        ohlcv: OHLCV data
        period: Lookback period

    Returns:
        Dict with CMF value and signal
    """
    try:
        if len(ohlcv) < period + 5:
            return {
                'cmf': 0.0,
                'zone': 'neutral',
                'pressure': 'balanced',
                'signal': 'NEUTRAL'
            }

        highs = np.array([candle[2] for candle in ohlcv], dtype=float)
        lows = np.array([candle[3] for candle in ohlcv], dtype=float)
        closes = np.array([candle[4] for candle in ohlcv], dtype=float)
        volumes = np.array([candle[5] for candle in ohlcv], dtype=float)

        # Calculate Money Flow Multiplier
        # MFM = ((Close - Low) - (High - Close)) / (High - Low)
        mfm = ((closes - lows) - (highs - closes)) / (highs - lows + 1e-10)

        # Money Flow Volume = MFM × Volume
        mfv = mfm * volumes

        # CMF = SUM(MFV, period) / SUM(Volume, period)
        sum_mfv = pd.Series(mfv).rolling(window=period).sum()
        sum_vol = pd.Series(volumes).rolling(window=period).sum()

        cmf = sum_mfv / (sum_vol + 1e-10)

        current_cmf = float(cmf.iloc[-1]) if not np.isnan(cmf.iloc[-1]) else 0.0
        prev_cmf = float(cmf.iloc[-2]) if len(cmf) > 1 and not np.isnan(cmf.iloc[-2]) else current_cmf

        # Determine pressure
        if current_cmf > 0.25:
            pressure = 'strong_buying'
        elif current_cmf > 0.05:
            pressure = 'buying'
        elif current_cmf < -0.25:
            pressure = 'strong_selling'
        elif current_cmf < -0.05:
            pressure = 'selling'
        else:
            pressure = 'balanced'

        # Determine zone
        if current_cmf > 0:
            zone = 'accumulation'
        else:
            zone = 'distribution'

        # Detect zero crossover
        zero_cross = None
        if prev_cmf < 0 and current_cmf > 0:
            zero_cross = 'bullish'
        elif prev_cmf > 0 and current_cmf < 0:
            zero_cross = 'bearish'

        # Generate signal
        signal = 'NEUTRAL'
        if pressure == 'strong_buying':
            signal = 'STRONG_BUY'
        elif pressure == 'buying' or zero_cross == 'bullish':
            signal = 'BUY'
        elif pressure == 'strong_selling':
            signal = 'STRONG_SELL'
        elif pressure == 'selling' or zero_cross == 'bearish':
            signal = 'SELL'

        return {
            'cmf': current_cmf,
            'zone': zone,
            'pressure': pressure,
            'zero_cross': zero_cross,
            'signal': signal,
            'prev_cmf': prev_cmf
        }

    except Exception as e:
        return {
            'cmf': 0.0,
            'zone': 'neutral',
            'pressure': 'balanced',
            'signal': 'NEUTRAL',
            'error': str(e)
        }


def calculate_atr_volatility_regime(ohlcv: List, atr_period: int = 14) -> Dict[str, Any]:
    """
    Calculate ATR-based Volatility Regime - Essential for position sizing and stop placement.

    ATR% = (ATR / Close) × 100

    Volatility Regimes:
    - ATR% < 1.5 = Low volatility (tight stops, larger size)
    - ATR% 1.5-3.0 = Normal volatility
    - ATR% 3.0-5.0 = High volatility (wider stops, smaller size)
    - ATR% > 5.0 = Extreme volatility (reduce size significantly)

    Professional Use:
    - Low volatility often precedes breakouts
    - High volatility = wider stops, protect capital
    - Adjust position size inverse to volatility
    - Best entries occur in low-moderate volatility

    Args:
        ohlcv: OHLCV data
        atr_period: ATR calculation period

    Returns:
        Dict with ATR%, regime, and recommendations
    """
    try:
        if len(ohlcv) < atr_period + 5:
            return {
                'atr': 0.0,
                'atr_pct': 0.0,
                'regime': 'normal',
                'stop_multiplier': 1.0,
                'size_multiplier': 1.0,
                'signal': 'NEUTRAL'
            }

        highs = np.array([candle[2] for candle in ohlcv], dtype=float)
        lows = np.array([candle[3] for candle in ohlcv], dtype=float)
        closes = np.array([candle[4] for candle in ohlcv], dtype=float)

        # Calculate True Range
        tr = np.zeros(len(ohlcv))
        tr[0] = highs[0] - lows[0]
        for i in range(1, len(ohlcv)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)

        # ATR (Average True Range)
        atr = pd.Series(tr).rolling(window=atr_period).mean()
        current_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0
        current_close = closes[-1]

        # ATR as percentage of price
        atr_pct = (current_atr / current_close) * 100

        # Historical ATR% for context
        atr_series = atr / pd.Series(closes) * 100
        avg_atr_pct = float(atr_series.iloc[-20:].mean()) if len(atr_series) >= 20 else atr_pct

        # Determine regime and multipliers
        if atr_pct < 1.5:
            regime = 'low'
            stop_multiplier = 0.8  # Tighter stops
            size_multiplier = 1.3  # Larger size (lower risk per trade)
            signal = 'BREAKOUT_WATCH'  # Low vol often precedes breakout
        elif atr_pct < 3.0:
            regime = 'normal'
            stop_multiplier = 1.0
            size_multiplier = 1.0
            signal = 'NEUTRAL'
        elif atr_pct < 5.0:
            regime = 'high'
            stop_multiplier = 1.3  # Wider stops
            size_multiplier = 0.7  # Smaller size
            signal = 'CAUTION'
        else:
            regime = 'extreme'
            stop_multiplier = 1.5  # Much wider stops
            size_multiplier = 0.4  # Much smaller size
            signal = 'HIGH_RISK'

        # Volatility trend (increasing or decreasing)
        if len(atr) >= 5:
            recent_atr = float(atr.iloc[-1])
            older_atr = float(atr.iloc[-5])
            if recent_atr > older_atr * 1.2:
                vol_trend = 'expanding'
            elif recent_atr < older_atr * 0.8:
                vol_trend = 'contracting'
            else:
                vol_trend = 'stable'
        else:
            vol_trend = 'unknown'

        return {
            'atr': current_atr,
            'atr_pct': float(atr_pct),
            'avg_atr_pct': float(avg_atr_pct),
            'regime': regime,
            'stop_multiplier': stop_multiplier,
            'size_multiplier': size_multiplier,
            'vol_trend': vol_trend,
            'signal': signal,
            'recommended_stop_atr': current_atr * stop_multiplier
        }

    except Exception as e:
        return {
            'atr': 0.0,
            'atr_pct': 0.0,
            'regime': 'normal',
            'stop_multiplier': 1.0,
            'size_multiplier': 1.0,
            'signal': 'NEUTRAL',
            'error': str(e)
        }


def calculate_fibonacci_confluence(ohlcv: List, lookback: int = 100) -> Dict[str, Any]:
    """
    Calculate Fibonacci Retracement Confluence - Professional level analysis.

    Key Fibonacci Levels:
    - 23.6% - Shallow retracement (strong trend)
    - 38.2% - Moderate retracement (healthy pullback)
    - 50.0% - Mid retracement (psychological level)
    - 61.8% - Golden ratio (best entry for continuation)
    - 78.6% - Deep retracement (potential reversal)

    Professional Use:
    - 38.2%-61.8% retracements = Best continuation entries
    - Price at Fib level + S/R = High probability trade
    - Multiple Fib levels clustering = Strong zone

    Args:
        ohlcv: OHLCV data
        lookback: Period to find swing high/low

    Returns:
        Dict with Fibonacci levels, current position, and signal
    """
    try:
        if len(ohlcv) < lookback:
            lookback = len(ohlcv)

        if lookback < 20:
            return {
                'levels': {},
                'current_level': None,
                'trend': 'unknown',
                'signal': 'NEUTRAL'
            }

        recent_ohlcv = ohlcv[-lookback:]
        highs = np.array([candle[2] for candle in recent_ohlcv], dtype=float)
        lows = np.array([candle[3] for candle in recent_ohlcv], dtype=float)
        closes = np.array([candle[4] for candle in recent_ohlcv], dtype=float)

        current_price = closes[-1]
        swing_high = np.max(highs)
        swing_low = np.min(lows)
        swing_range = swing_high - swing_low

        # Find swing high/low positions
        swing_high_idx = np.argmax(highs)
        swing_low_idx = np.argmin(lows)

        # Determine trend based on which came first
        if swing_low_idx < swing_high_idx:
            trend = 'uptrend'
            # Calculate retracements from high
            fib_levels = {
                '0.0': swing_high,
                '23.6': swing_high - (0.236 * swing_range),
                '38.2': swing_high - (0.382 * swing_range),
                '50.0': swing_high - (0.500 * swing_range),
                '61.8': swing_high - (0.618 * swing_range),
                '78.6': swing_high - (0.786 * swing_range),
                '100.0': swing_low
            }
        else:
            trend = 'downtrend'
            # Calculate retracements from low
            fib_levels = {
                '0.0': swing_low,
                '23.6': swing_low + (0.236 * swing_range),
                '38.2': swing_low + (0.382 * swing_range),
                '50.0': swing_low + (0.500 * swing_range),
                '61.8': swing_low + (0.618 * swing_range),
                '78.6': swing_low + (0.786 * swing_range),
                '100.0': swing_high
            }

        # Find which Fib level price is near (within 0.5%)
        tolerance_pct = 0.005  # 0.5%
        current_level = None
        distance_to_nearest = float('inf')

        for level_name, level_price in fib_levels.items():
            distance = abs(current_price - level_price) / level_price
            if distance < tolerance_pct and distance < distance_to_nearest:
                current_level = level_name
                distance_to_nearest = distance

        # Calculate distance to golden ratio (61.8%)
        golden_level = fib_levels['61.8']
        distance_to_golden = ((current_price - golden_level) / golden_level) * 100

        # Generate signal
        signal = 'NEUTRAL'

        if trend == 'uptrend':
            # In uptrend, look for pullback entries
            if current_level in ['38.2', '50.0', '61.8']:
                signal = 'BUY'  # At key retracement level
                if current_level == '61.8':
                    signal = 'STRONG_BUY'  # Golden ratio
            elif current_level == '78.6':
                signal = 'SELL'  # Deep retracement, might reverse
        else:  # downtrend
            # In downtrend, look for rally entries
            if current_level in ['38.2', '50.0', '61.8']:
                signal = 'SELL'  # At key retracement level
                if current_level == '61.8':
                    signal = 'STRONG_SELL'  # Golden ratio
            elif current_level == '78.6':
                signal = 'BUY'  # Deep retracement, might reverse

        return {
            'levels': {k: float(v) for k, v in fib_levels.items()},
            'current_level': current_level,
            'trend': trend,
            'swing_high': float(swing_high),
            'swing_low': float(swing_low),
            'distance_to_golden': float(distance_to_golden),
            'signal': signal,
            'current_price': float(current_price)
        }

    except Exception as e:
        return {
            'levels': {},
            'current_level': None,
            'trend': 'unknown',
            'signal': 'NEUTRAL',
            'error': str(e)
        }


def calculate_advanced_indicators(ohlcv: List) -> Dict[str, Any]:
    """
    Calculate ALL advanced indicators for confluence scoring.

    This combines all v4.5.0 professional indicators:
    1. VWAP - Institutional fair value
    2. Stochastic RSI - Momentum with sensitivity
    3. Williams %R - Fast reversal detection
    4. Chaikin Money Flow - Volume-weighted pressure
    5. ATR Volatility Regime - Risk management
    6. Fibonacci Confluence - Key levels

    Args:
        ohlcv: List of OHLCV data

    Returns:
        Dict with all advanced indicator values and summary
    """
    try:
        result = {
            'vwap': calculate_vwap(ohlcv),
            'stoch_rsi': calculate_stochastic_rsi(ohlcv),
            'williams_r': calculate_williams_r(ohlcv),
            'cmf': calculate_chaikin_money_flow(ohlcv),
            'atr_regime': calculate_atr_volatility_regime(ohlcv),
            'fibonacci': calculate_fibonacci_confluence(ohlcv),
        }

        # Count signals
        signals = []

        # VWAP signal
        vwap_sig = result['vwap'].get('signal', 'NEUTRAL')
        if vwap_sig in ['STRONG_BUY', 'BUY']:
            signals.append(('bullish', 'vwap', vwap_sig))
        elif vwap_sig in ['STRONG_SELL', 'SELL']:
            signals.append(('bearish', 'vwap', vwap_sig))

        # Stochastic RSI signal
        stoch_sig = result['stoch_rsi'].get('signal', 'NEUTRAL')
        if stoch_sig in ['STRONG_BUY', 'BUY']:
            signals.append(('bullish', 'stoch_rsi', stoch_sig))
        elif stoch_sig in ['STRONG_SELL', 'SELL']:
            signals.append(('bearish', 'stoch_rsi', stoch_sig))

        # Williams %R signal
        wr_sig = result['williams_r'].get('signal', 'NEUTRAL')
        if wr_sig in ['STRONG_BUY', 'BUY']:
            signals.append(('bullish', 'williams_r', wr_sig))
        elif wr_sig in ['STRONG_SELL', 'SELL']:
            signals.append(('bearish', 'williams_r', wr_sig))

        # CMF signal
        cmf_sig = result['cmf'].get('signal', 'NEUTRAL')
        if cmf_sig in ['STRONG_BUY', 'BUY']:
            signals.append(('bullish', 'cmf', cmf_sig))
        elif cmf_sig in ['STRONG_SELL', 'SELL']:
            signals.append(('bearish', 'cmf', cmf_sig))

        # Fibonacci signal
        fib_sig = result['fibonacci'].get('signal', 'NEUTRAL')
        if fib_sig in ['STRONG_BUY', 'BUY']:
            signals.append(('bullish', 'fibonacci', fib_sig))
        elif fib_sig in ['STRONG_SELL', 'SELL']:
            signals.append(('bearish', 'fibonacci', fib_sig))

        # Count
        bullish_count = sum(1 for s in signals if s[0] == 'bullish')
        bearish_count = sum(1 for s in signals if s[0] == 'bearish')

        # Get risk assessment from ATR regime
        atr_regime = result['atr_regime'].get('regime', 'normal')
        size_multiplier = result['atr_regime'].get('size_multiplier', 1.0)

        result['signal_summary'] = {
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'total_signals': len(signals),
            'signals': signals,
            'overall_bias': 'bullish' if bullish_count > bearish_count else ('bearish' if bearish_count > bullish_count else 'neutral'),
            'risk_regime': atr_regime,
            'recommended_size_mult': size_multiplier
        }

        return result

    except Exception as e:
        return {
            'vwap': {'signal': 'NEUTRAL'},
            'stoch_rsi': {'signal': 'NEUTRAL'},
            'williams_r': {'signal': 'NEUTRAL'},
            'cmf': {'signal': 'NEUTRAL'},
            'atr_regime': {'regime': 'normal', 'size_multiplier': 1.0},
            'fibonacci': {'signal': 'NEUTRAL'},
            'signal_summary': {
                'bullish_signals': 0,
                'bearish_signals': 0,
                'total_signals': 0,
                'signals': [],
                'overall_bias': 'neutral',
                'risk_regime': 'normal',
                'recommended_size_mult': 1.0
            },
            'error': str(e)
        }
