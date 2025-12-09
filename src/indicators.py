"""
Technical indicators calculation module.
Calculates RSI, MACD, Bollinger Bands, Moving Averages, and other indicators.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
from ta import trend, momentum, volatility, volume as vol_indicators

# 🛡️ v4.7.5: Global logger for order flow and other functions
logger = logging.getLogger('trading_bot')


def calculate_indicators(ohlcv_data: List[List]) -> Dict[str, Any]:
    """
    Calculate comprehensive technical indicators from OHLCV data.

    Args:
        ohlcv_data: List of [timestamp, open, high, low, close, volume]

    Returns:
        Dict with all calculated indicators
    """
    # Handle both list and DataFrame input
    if isinstance(ohlcv_data, pd.DataFrame):
        df = ohlcv_data.copy()
        if df.empty or len(df) < 20:
            return get_default_indicators()
    else:
        # List input
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

        # 🔧 v5.0.17 FIX: Add EMA 20/50 for Level-Based Trading trend filter
        # These were MISSING - causing trend detection to fail!
        indicators['ema_20'] = calculate_ema(df['close'], period=20)
        indicators['ema_50'] = calculate_ema(df['close'], period=50)

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

        # 🔧 v5.0.17 FIX: Add ADX + DI for Level-Based Trading trend filter
        # These were MISSING - causing ADX/trend filter to use default values (25)!
        try:
            adx_data = calculate_adx_simple(df)
            indicators['adx'] = adx_data['adx']
            indicators['plus_di'] = adx_data['plus_di']
            indicators['minus_di'] = adx_data['minus_di']
        except Exception as e:
            logger.warning(f"ADX calculation failed, using defaults: {e}")
            indicators['adx'] = 25.0
            indicators['plus_di'] = 25.0
            indicators['minus_di'] = 25.0

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


def calculate_adx_simple(df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
    """
    🔧 v5.0.17: Simple ADX calculation for DataFrame input.

    Returns ADX, +DI, -DI for trend strength and direction filtering.

    ADX interpretation:
    - ADX < 20: Weak/No trend (ranging market)
    - ADX 20-25: Trend emerging
    - ADX 25-50: Strong trend
    - ADX > 50: Very strong trend (often exhaustion)

    DI interpretation:
    - +DI > -DI: Bullish trend
    - -DI > +DI: Bearish trend
    """
    try:
        if len(df) < period + 1:
            return {'adx': 25.0, 'plus_di': 25.0, 'minus_di': 25.0}

        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate True Range
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        # Only keep positive directional movement
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smooth with EMA
        tr_smooth = tr.ewm(span=period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()

        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)

        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.ewm(span=period, adjust=False).mean()

        return {
            'adx': float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 25.0,
            'plus_di': float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 25.0,
            'minus_di': float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 25.0
        }
    except Exception as e:
        logger.warning(f"ADX simple calculation error: {e}")
        return {'adx': 25.0, 'plus_di': 25.0, 'minus_di': 25.0}


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
            logger.debug("⚠️ Order flow: No order_book data available")
            return {
                'imbalance': 0.0,
                'imbalance_ratio': 1.0,
                'signal': 'neutral',
                'large_bid_wall': None,
                'large_ask_wall': None,
                'buy_pressure': 0.5,
                # 🛡️ v4.7.4: Add weighted_imbalance to all returns
                'weighted_imbalance': 0.0,
                'weighted_imbalance_ratio': 1.0,
                'weighted_bid_volume': 0.0,
                'weighted_ask_volume': 0.0,
                'depth_levels_analyzed': 0
            }

        # 🎯 #3: DEEP ORDER BOOK ANALYSIS (20 → 100 levels)
        # Institutional iceberg orders hidden in deeper levels!
        bids = order_book.get('bids', [])[:100]  # Top 100 bids (was 20)
        asks = order_book.get('asks', [])[:100]  # Top 100 asks (was 20)

        if not bids or not asks:
            logger.debug("⚠️ Order flow: Empty bids or asks in order_book")
            return {
                'imbalance': 0.0,
                'imbalance_ratio': 1.0,
                'signal': 'neutral',
                'large_bid_wall': None,
                'large_ask_wall': None,
                'buy_pressure': 0.5,
                # 🛡️ v4.7.4: Add weighted_imbalance to all returns
                'weighted_imbalance': 0.0,
                'weighted_imbalance_ratio': 1.0,
                'weighted_bid_volume': 0.0,
                'weighted_ask_volume': 0.0,
                'depth_levels_analyzed': 0
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
        logger.warning(f"⚠️ Order flow analysis failed: {e}")
        return {
            'imbalance': 0.0,
            'imbalance_ratio': 1.0,
            'signal': 'neutral',
            'large_bid_wall': None,
            'large_ask_wall': None,
            'buy_pressure': 0.5,
            # 🛡️ v4.7.4: Add weighted_imbalance to exception return
            'weighted_imbalance': 0.0,
            'weighted_imbalance_ratio': 1.0,
            'weighted_bid_volume': 0.0,
            'weighted_ask_volume': 0.0,
            'depth_levels_analyzed': 0
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


def calculate_vwap_advanced(ohlcv: List, include_bands: bool = True) -> Dict[str, Any]:
    """
    Calculate VWAP (Volume Weighted Average Price) - Institutional Trading Standard.

    NOTE: This is the v4.5.0 advanced version with bands and detailed analysis.
    The original calculate_vwap() is used by the main indicator system.

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


def calculate_stochastic_rsi_advanced(ohlcv: List, rsi_period: int = 14,
                                       stoch_period: int = 14, k_smooth: int = 3,
                                       d_smooth: int = 3) -> Dict[str, Any]:
    """
    Calculate Stochastic RSI - RSI + Stochastic combined for sensitivity.

    NOTE: This is the v4.5.0 advanced version with detailed analysis.
    The original calculate_stochastic_rsi() is used by the main indicator system.

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
            'vwap': calculate_vwap_advanced(ohlcv),
            'stoch_rsi': calculate_stochastic_rsi_advanced(ohlcv),
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


# ═══════════════════════════════════════════════════════════════════════════════
# 🏛️ v4.6.0: INSTITUTIONAL GRADE INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════
# Smart Money Concepts (SMC), Wyckoff VSA, Market Structure, Statistical Edge
# These are the indicators used by professional institutional traders
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_order_blocks(ohlcv: List, lookback: int = 50) -> Dict[str, Any]:
    """
    🏛️ SMART MONEY CONCEPT: Order Blocks Detection

    Order blocks represent zones where institutional traders placed large orders.
    - Bullish OB: Last bearish candle before explosive up move
    - Bearish OB: Last bullish candle before explosive down move

    When price returns to these zones, it often reverses.

    Args:
        ohlcv: OHLCV data
        lookback: How far back to look for order blocks

    Returns:
        Dict with bullish/bearish order blocks and proximity analysis
    """
    try:
        if len(ohlcv) < lookback:
            return {'signal': 'NEUTRAL', 'bullish_obs': [], 'bearish_obs': [], 'nearest_ob': None}

        opens = [c[1] for c in ohlcv]
        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        closes = [c[4] for c in ohlcv]
        volumes = [c[5] for c in ohlcv]

        current_price = closes[-1]
        avg_volume = sum(volumes[-20:]) / 20

        bullish_obs = []  # Zones to buy from
        bearish_obs = []  # Zones to sell from

        # Scan for order blocks
        for i in range(2, min(lookback, len(ohlcv) - 2)):
            idx = len(ohlcv) - 1 - i

            if idx < 1:
                break

            # Current candle stats
            candle_body = abs(closes[idx] - opens[idx])
            candle_range = highs[idx] - lows[idx]
            is_bullish = closes[idx] > opens[idx]
            is_bearish = closes[idx] < opens[idx]

            # Next candle (the impulse move)
            next_body = abs(closes[idx + 1] - opens[idx + 1])
            next_range = highs[idx + 1] - lows[idx + 1]
            next_is_bullish = closes[idx + 1] > opens[idx + 1]
            next_is_bearish = closes[idx + 1] < opens[idx + 1]

            # Check for high volume on impulse
            volume_ratio = volumes[idx + 1] / avg_volume if avg_volume > 0 else 1

            # BULLISH ORDER BLOCK: Bearish candle followed by strong bullish impulse
            if is_bearish and next_is_bullish:
                # Impulse must be at least 1.5x the OB candle body
                if next_body > candle_body * 1.5 and volume_ratio > 1.2:
                    # OB zone is the body of the bearish candle
                    ob_high = max(opens[idx], closes[idx])
                    ob_low = min(opens[idx], closes[idx])
                    ob_mid = (ob_high + ob_low) / 2

                    # Check if price is above this OB (valid zone)
                    if current_price > ob_high:
                        distance_pct = (current_price - ob_mid) / current_price * 100
                        bullish_obs.append({
                            'high': ob_high,
                            'low': ob_low,
                            'mid': ob_mid,
                            'strength': volume_ratio,
                            'distance_pct': distance_pct,
                            'age_candles': i
                        })

            # BEARISH ORDER BLOCK: Bullish candle followed by strong bearish impulse
            if is_bullish and next_is_bearish:
                if next_body > candle_body * 1.5 and volume_ratio > 1.2:
                    ob_high = max(opens[idx], closes[idx])
                    ob_low = min(opens[idx], closes[idx])
                    ob_mid = (ob_high + ob_low) / 2

                    # Check if price is below this OB (valid zone)
                    if current_price < ob_low:
                        distance_pct = (ob_mid - current_price) / current_price * 100
                        bearish_obs.append({
                            'high': ob_high,
                            'low': ob_low,
                            'mid': ob_mid,
                            'strength': volume_ratio,
                            'distance_pct': distance_pct,
                            'age_candles': i
                        })

        # Find nearest OB to current price
        nearest_ob = None
        nearest_distance = float('inf')

        for ob in bullish_obs:
            if ob['distance_pct'] < nearest_distance:
                nearest_distance = ob['distance_pct']
                nearest_ob = {'type': 'BULLISH', **ob}

        for ob in bearish_obs:
            if ob['distance_pct'] < nearest_distance:
                nearest_distance = ob['distance_pct']
                nearest_ob = {'type': 'BEARISH', **ob}

        # Generate signal based on proximity to OB
        signal = 'NEUTRAL'

        if nearest_ob:
            if nearest_ob['type'] == 'BULLISH' and nearest_distance < 1.5:
                # Price near bullish OB - expect bounce up
                signal = 'BUY' if nearest_distance < 0.5 else 'WATCH_LONG'
            elif nearest_ob['type'] == 'BEARISH' and nearest_distance < 1.5:
                # Price near bearish OB - expect rejection down
                signal = 'SELL' if nearest_distance < 0.5 else 'WATCH_SHORT'

        return {
            'signal': signal,
            'bullish_obs': bullish_obs[:3],  # Top 3 nearest
            'bearish_obs': bearish_obs[:3],
            'nearest_ob': nearest_ob,
            'total_bullish': len(bullish_obs),
            'total_bearish': len(bearish_obs)
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'bullish_obs': [], 'bearish_obs': [], 'error': str(e)}


def calculate_fair_value_gaps(ohlcv: List, min_gap_pct: float = 0.1) -> Dict[str, Any]:
    """
    🏛️ SMART MONEY CONCEPT: Fair Value Gaps (FVG) Detection

    FVGs are price imbalances where price moved so fast that it left gaps.
    Price statistically returns to fill these gaps ~70% of the time.

    FVG Pattern (Bullish):
    - Candle 1 high < Candle 3 low = Gap (imbalance)

    Args:
        ohlcv: OHLCV data
        min_gap_pct: Minimum gap size as percentage

    Returns:
        Dict with bullish/bearish FVGs and fill probability
    """
    try:
        if len(ohlcv) < 20:
            return {'signal': 'NEUTRAL', 'bullish_fvgs': [], 'bearish_fvgs': []}

        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        closes = [c[4] for c in ohlcv]

        current_price = closes[-1]

        bullish_fvgs = []  # Gaps below price (support zones)
        bearish_fvgs = []  # Gaps above price (resistance zones)

        # Scan for FVGs (need 3 candles)
        for i in range(2, len(ohlcv) - 1):
            # Bullish FVG: Candle 1 high < Candle 3 low
            if highs[i-2] < lows[i]:
                gap_size = lows[i] - highs[i-2]
                gap_pct = gap_size / current_price * 100

                if gap_pct >= min_gap_pct:
                    fvg_mid = (lows[i] + highs[i-2]) / 2

                    # Only track unfilled FVGs (price above the gap)
                    if current_price > lows[i]:
                        distance_pct = (current_price - fvg_mid) / current_price * 100
                        bullish_fvgs.append({
                            'high': lows[i],
                            'low': highs[i-2],
                            'mid': fvg_mid,
                            'gap_pct': gap_pct,
                            'distance_pct': distance_pct,
                            'age_candles': len(ohlcv) - 1 - i
                        })

            # Bearish FVG: Candle 1 low > Candle 3 high
            if lows[i-2] > highs[i]:
                gap_size = lows[i-2] - highs[i]
                gap_pct = gap_size / current_price * 100

                if gap_pct >= min_gap_pct:
                    fvg_mid = (lows[i-2] + highs[i]) / 2

                    # Only track unfilled FVGs (price below the gap)
                    if current_price < highs[i]:
                        distance_pct = (fvg_mid - current_price) / current_price * 100
                        bearish_fvgs.append({
                            'high': lows[i-2],
                            'low': highs[i],
                            'mid': fvg_mid,
                            'gap_pct': gap_pct,
                            'distance_pct': distance_pct,
                            'age_candles': len(ohlcv) - 1 - i
                        })

        # Sort by distance (nearest first)
        bullish_fvgs.sort(key=lambda x: x['distance_pct'])
        bearish_fvgs.sort(key=lambda x: x['distance_pct'])

        # Generate signal
        signal = 'NEUTRAL'
        nearest_fvg = None

        # Check nearest bullish FVG
        if bullish_fvgs and bullish_fvgs[0]['distance_pct'] < 2.0:
            nearest_fvg = {'type': 'BULLISH', **bullish_fvgs[0]}
            if bullish_fvgs[0]['distance_pct'] < 0.5:
                signal = 'BUY'  # Price very close to bullish FVG
            else:
                signal = 'WATCH_LONG'

        # Check nearest bearish FVG
        if bearish_fvgs and bearish_fvgs[0]['distance_pct'] < 2.0:
            if not nearest_fvg or bearish_fvgs[0]['distance_pct'] < nearest_fvg['distance_pct']:
                nearest_fvg = {'type': 'BEARISH', **bearish_fvgs[0]}
                if bearish_fvgs[0]['distance_pct'] < 0.5:
                    signal = 'SELL'
                else:
                    signal = 'WATCH_SHORT'

        return {
            'signal': signal,
            'bullish_fvgs': bullish_fvgs[:5],  # Top 5 nearest
            'bearish_fvgs': bearish_fvgs[:5],
            'nearest_fvg': nearest_fvg,
            'total_bullish': len(bullish_fvgs),
            'total_bearish': len(bearish_fvgs),
            'fill_probability': 0.70  # Statistical probability
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'bullish_fvgs': [], 'bearish_fvgs': [], 'error': str(e)}


def calculate_liquidity_sweeps(ohlcv: List, lookback: int = 30) -> Dict[str, Any]:
    """
    🏛️ SMART MONEY CONCEPT: Liquidity Sweep Detection

    Liquidity sweeps occur when price spikes through obvious highs/lows
    (where retail traders place stops) and then reverses.

    This is how smart money hunts retail stop losses.

    Args:
        ohlcv: OHLCV data
        lookback: Period to look for swing points

    Returns:
        Dict with detected liquidity sweeps and reversal probability
    """
    try:
        if len(ohlcv) < lookback:
            return {'signal': 'NEUTRAL', 'recent_sweeps': [], 'sweep_detected': False}

        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        closes = [c[4] for c in ohlcv]

        current_price = closes[-1]

        # Find swing highs and lows (liquidity pools)
        swing_highs = []
        swing_lows = []

        for i in range(2, len(ohlcv) - 2):
            # Swing high: Higher than 2 candles on each side
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append({'price': highs[i], 'index': i})

            # Swing low: Lower than 2 candles on each side
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append({'price': lows[i], 'index': i})

        recent_sweeps = []

        # Check recent candles for liquidity sweeps
        for i in range(-5, 0):
            if len(ohlcv) + i < 1:
                continue

            candle_high = highs[i]
            candle_low = lows[i]
            candle_close = closes[i]
            candle_open = ohlcv[i][1]

            # Check if this candle swept a swing high (bearish sweep)
            for sh in swing_highs:
                if sh['index'] < len(ohlcv) + i - 1:  # Swing high is before this candle
                    # Wick above swing high but close below
                    if candle_high > sh['price'] and candle_close < sh['price']:
                        sweep_size = (candle_high - sh['price']) / sh['price'] * 100
                        if sweep_size > 0.1:  # At least 0.1% sweep
                            recent_sweeps.append({
                                'type': 'BEARISH_SWEEP',
                                'swept_level': sh['price'],
                                'sweep_high': candle_high,
                                'close': candle_close,
                                'candles_ago': abs(i),
                                'sweep_size_pct': sweep_size
                            })

            # Check if this candle swept a swing low (bullish sweep)
            for sl in swing_lows:
                if sl['index'] < len(ohlcv) + i - 1:
                    # Wick below swing low but close above
                    if candle_low < sl['price'] and candle_close > sl['price']:
                        sweep_size = (sl['price'] - candle_low) / sl['price'] * 100
                        if sweep_size > 0.1:
                            recent_sweeps.append({
                                'type': 'BULLISH_SWEEP',
                                'swept_level': sl['price'],
                                'sweep_low': candle_low,
                                'close': candle_close,
                                'candles_ago': abs(i),
                                'sweep_size_pct': sweep_size
                            })

        # Generate signal based on recent sweeps
        signal = 'NEUTRAL'
        sweep_detected = len(recent_sweeps) > 0

        if recent_sweeps:
            # Most recent sweep
            latest_sweep = min(recent_sweeps, key=lambda x: x['candles_ago'])

            if latest_sweep['type'] == 'BULLISH_SWEEP' and latest_sweep['candles_ago'] <= 3:
                signal = 'STRONG_BUY'  # Just swept lows - reversal likely
            elif latest_sweep['type'] == 'BEARISH_SWEEP' and latest_sweep['candles_ago'] <= 3:
                signal = 'STRONG_SELL'  # Just swept highs - reversal likely

        return {
            'signal': signal,
            'recent_sweeps': recent_sweeps,
            'sweep_detected': sweep_detected,
            'total_swing_highs': len(swing_highs),
            'total_swing_lows': len(swing_lows),
            'reversal_probability': 0.65 if sweep_detected else 0.0
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'recent_sweeps': [], 'sweep_detected': False, 'error': str(e)}


def calculate_market_structure(ohlcv: List) -> Dict[str, Any]:
    """
    🏛️ SMART MONEY CONCEPT: Market Structure Analysis

    Detects Break of Structure (BOS) and Change of Character (CHoCH):
    - BOS: Continuation signal (HH in uptrend, LL in downtrend)
    - CHoCH: Reversal signal (first LL in uptrend, first HH in downtrend)

    This is the foundation of all SMC trading.

    Args:
        ohlcv: OHLCV data

    Returns:
        Dict with market structure, BOS/CHoCH signals
    """
    try:
        if len(ohlcv) < 30:
            return {'signal': 'NEUTRAL', 'structure': 'UNKNOWN', 'bos': None, 'choch': None}

        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        closes = [c[4] for c in ohlcv]

        current_price = closes[-1]

        # Find swing points
        swing_highs = []
        swing_lows = []

        for i in range(3, len(ohlcv) - 3):
            # Swing high
            if highs[i] >= max(highs[i-3:i]) and highs[i] >= max(highs[i+1:i+4]):
                swing_highs.append({'price': highs[i], 'index': i})

            # Swing low
            if lows[i] <= min(lows[i-3:i]) and lows[i] <= min(lows[i+1:i+4]):
                swing_lows.append({'price': lows[i], 'index': i})

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'signal': 'NEUTRAL', 'structure': 'INSUFFICIENT_DATA', 'bos': None, 'choch': None}

        # Analyze structure (last 4 swing points)
        recent_highs = swing_highs[-4:] if len(swing_highs) >= 4 else swing_highs
        recent_lows = swing_lows[-4:] if len(swing_lows) >= 4 else swing_lows

        # Determine current structure
        hh_count = 0  # Higher highs
        ll_count = 0  # Lower lows
        lh_count = 0  # Lower highs
        hl_count = 0  # Higher lows

        for i in range(1, len(recent_highs)):
            if recent_highs[i]['price'] > recent_highs[i-1]['price']:
                hh_count += 1
            else:
                lh_count += 1

        for i in range(1, len(recent_lows)):
            if recent_lows[i]['price'] > recent_lows[i-1]['price']:
                hl_count += 1
            else:
                ll_count += 1

        # Determine structure
        structure = 'RANGING'
        if hh_count >= 2 and hl_count >= 1:
            structure = 'BULLISH'  # Higher highs + Higher lows
        elif ll_count >= 2 and lh_count >= 1:
            structure = 'BEARISH'  # Lower lows + Lower highs

        # Detect BOS and CHoCH
        bos = None
        choch = None

        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            last_high = recent_highs[-1]['price']
            prev_high = recent_highs[-2]['price']
            last_low = recent_lows[-1]['price']
            prev_low = recent_lows[-2]['price']

            # BOS (Break of Structure) - Continuation
            if structure == 'BULLISH' and current_price > last_high:
                bos = {'type': 'BULLISH_BOS', 'level': last_high, 'current': current_price}
            elif structure == 'BEARISH' and current_price < last_low:
                bos = {'type': 'BEARISH_BOS', 'level': last_low, 'current': current_price}

            # CHoCH (Change of Character) - Reversal
            if structure == 'BULLISH' and current_price < last_low:
                choch = {'type': 'BEARISH_CHOCH', 'level': last_low, 'current': current_price}
            elif structure == 'BEARISH' and current_price > last_high:
                choch = {'type': 'BULLISH_CHOCH', 'level': last_high, 'current': current_price}

        # Generate signal
        signal = 'NEUTRAL'

        if choch:
            # CHoCH is a strong reversal signal
            if choch['type'] == 'BULLISH_CHOCH':
                signal = 'STRONG_BUY'
            else:
                signal = 'STRONG_SELL'
        elif bos:
            # BOS is a continuation signal
            if bos['type'] == 'BULLISH_BOS':
                signal = 'BUY'
            else:
                signal = 'SELL'
        elif structure == 'BULLISH':
            signal = 'BUY'
        elif structure == 'BEARISH':
            signal = 'SELL'

        return {
            'signal': signal,
            'structure': structure,
            'bos': bos,
            'choch': choch,
            'swing_highs': len(swing_highs),
            'swing_lows': len(swing_lows),
            'hh_count': hh_count,
            'hl_count': hl_count,
            'lh_count': lh_count,
            'll_count': ll_count
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'structure': 'ERROR', 'error': str(e)}


def calculate_wyckoff_vsa(ohlcv: List) -> Dict[str, Any]:
    """
    🏛️ WYCKOFF VOLUME SPREAD ANALYSIS (VSA)

    Analyzes the relationship between price spread and volume to detect:
    - Accumulation (smart money buying)
    - Distribution (smart money selling)
    - Stopping volume (trend exhaustion)
    - Climax (panic buying/selling)
    - Effort vs Result (volume/price divergence)

    This is what institutional traders use to see "smart money" activity.

    Args:
        ohlcv: OHLCV data

    Returns:
        Dict with VSA analysis and signals
    """
    try:
        if len(ohlcv) < 30:
            return {'signal': 'NEUTRAL', 'phase': 'UNKNOWN', 'patterns': []}

        opens = [c[1] for c in ohlcv]
        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        closes = [c[4] for c in ohlcv]
        volumes = [c[5] for c in ohlcv]

        current_price = closes[-1]

        # Calculate averages
        avg_volume = sum(volumes[-20:]) / 20
        avg_spread = sum(highs[i] - lows[i] for i in range(-20, 0)) / 20

        patterns = []

        # Analyze recent candles for VSA patterns
        for i in range(-10, 0):
            spread = highs[i] - lows[i]
            body = abs(closes[i] - opens[i])
            volume = volumes[i]

            vol_ratio = volume / avg_volume if avg_volume > 0 else 1
            spread_ratio = spread / avg_spread if avg_spread > 0 else 1

            is_up = closes[i] > opens[i]
            is_down = closes[i] < opens[i]

            # Close position in range
            if spread > 0:
                close_position = (closes[i] - lows[i]) / spread
            else:
                close_position = 0.5

            pattern = None

            # === CLIMAX PATTERNS ===
            # Selling Climax: Wide spread, ultra-high volume, close near high (reversal up)
            if is_down and vol_ratio > 2.0 and spread_ratio > 1.5 and close_position > 0.6:
                pattern = {
                    'type': 'SELLING_CLIMAX',
                    'candles_ago': abs(i),
                    'signal': 'STRONG_BUY',
                    'description': 'Panic selling absorbed by smart money'
                }

            # Buying Climax: Wide spread, ultra-high volume, close near low (reversal down)
            elif is_up and vol_ratio > 2.0 and spread_ratio > 1.5 and close_position < 0.4:
                pattern = {
                    'type': 'BUYING_CLIMAX',
                    'candles_ago': abs(i),
                    'signal': 'STRONG_SELL',
                    'description': 'FOMO buying, smart money distributing'
                }

            # === STOPPING VOLUME ===
            # Stopping Volume (Down): High volume, closes off lows (absorption)
            elif is_down and vol_ratio > 1.5 and close_position > 0.5:
                pattern = {
                    'type': 'STOPPING_VOLUME_DOWN',
                    'candles_ago': abs(i),
                    'signal': 'BUY',
                    'description': 'Selling being absorbed, downtrend exhaustion'
                }

            # Stopping Volume (Up): High volume, closes off highs
            elif is_up and vol_ratio > 1.5 and close_position < 0.5:
                pattern = {
                    'type': 'STOPPING_VOLUME_UP',
                    'candles_ago': abs(i),
                    'signal': 'SELL',
                    'description': 'Buying being absorbed, uptrend exhaustion'
                }

            # === NO DEMAND / NO SUPPLY ===
            # No Demand: Up bar with low volume and narrow spread
            elif is_up and vol_ratio < 0.5 and spread_ratio < 0.7:
                pattern = {
                    'type': 'NO_DEMAND',
                    'candles_ago': abs(i),
                    'signal': 'SELL',
                    'description': 'Weak rally, no buying interest'
                }

            # No Supply: Down bar with low volume and narrow spread
            elif is_down and vol_ratio < 0.5 and spread_ratio < 0.7:
                pattern = {
                    'type': 'NO_SUPPLY',
                    'candles_ago': abs(i),
                    'signal': 'BUY',
                    'description': 'Weak decline, no selling pressure'
                }

            # === EFFORT VS RESULT ===
            # High effort, low result (up) - Distribution
            elif is_up and vol_ratio > 1.5 and spread_ratio < 0.5:
                pattern = {
                    'type': 'EFFORT_NO_RESULT_UP',
                    'candles_ago': abs(i),
                    'signal': 'SELL',
                    'description': 'High volume but small up move - selling into strength'
                }

            # High effort, low result (down) - Accumulation
            elif is_down and vol_ratio > 1.5 and spread_ratio < 0.5:
                pattern = {
                    'type': 'EFFORT_NO_RESULT_DOWN',
                    'candles_ago': abs(i),
                    'signal': 'BUY',
                    'description': 'High volume but small down move - buying the dip'
                }

            if pattern:
                patterns.append(pattern)

        # Determine overall phase
        buy_signals = sum(1 for p in patterns if p['signal'] in ['BUY', 'STRONG_BUY'])
        sell_signals = sum(1 for p in patterns if p['signal'] in ['SELL', 'STRONG_SELL'])

        phase = 'NEUTRAL'
        if buy_signals > sell_signals + 1:
            phase = 'ACCUMULATION'
        elif sell_signals > buy_signals + 1:
            phase = 'DISTRIBUTION'
        elif buy_signals > 0 and sell_signals > 0:
            phase = 'TRANSITION'

        # Generate signal based on most recent strong pattern
        signal = 'NEUTRAL'
        for pattern in sorted(patterns, key=lambda x: x['candles_ago']):
            if pattern['signal'] in ['STRONG_BUY', 'STRONG_SELL']:
                signal = pattern['signal']
                break
            elif pattern['signal'] in ['BUY', 'SELL'] and signal == 'NEUTRAL':
                signal = pattern['signal']

        return {
            'signal': signal,
            'phase': phase,
            'patterns': patterns,
            'total_patterns': len(patterns),
            'buy_patterns': buy_signals,
            'sell_patterns': sell_signals,
            'avg_volume': avg_volume,
            'current_volume_ratio': volumes[-1] / avg_volume if avg_volume > 0 else 1
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'phase': 'ERROR', 'patterns': [], 'error': str(e)}


def calculate_hurst_exponent(ohlcv: List, min_window: int = 10, max_window: int = 50) -> Dict[str, Any]:
    """
    🏛️ STATISTICAL EDGE: Hurst Exponent

    Measures the "memory" of the time series:
    - H < 0.5: Mean-reverting (use oscillator strategies)
    - H = 0.5: Random walk (avoid trading)
    - H > 0.5: Trending (use momentum strategies)

    This tells you WHETHER to trade and WHICH strategy to use.

    Args:
        ohlcv: OHLCV data
        min_window: Minimum window for R/S calculation
        max_window: Maximum window for R/S calculation

    Returns:
        Dict with Hurst exponent and regime classification
    """
    try:
        if len(ohlcv) < max_window + 10:
            return {'signal': 'NEUTRAL', 'hurst': 0.5, 'regime': 'RANDOM'}

        closes = [c[4] for c in ohlcv]

        # Calculate log returns
        returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0:
                returns.append(np.log(closes[i] / closes[i-1]))

        returns = np.array(returns)

        if len(returns) < max_window:
            return {'signal': 'NEUTRAL', 'hurst': 0.5, 'regime': 'RANDOM'}

        # Calculate R/S for different window sizes
        rs_values = []
        window_sizes = []

        for window in range(min_window, min(max_window + 1, len(returns) // 2)):
            # Split returns into windows
            n_windows = len(returns) // window

            if n_windows < 2:
                continue

            rs_sum = 0
            for w in range(n_windows):
                start = w * window
                end = start + window
                window_returns = returns[start:end]

                # Mean-adjusted cumulative sum
                mean_return = np.mean(window_returns)
                cumsum = np.cumsum(window_returns - mean_return)

                # Range
                R = np.max(cumsum) - np.min(cumsum)

                # Standard deviation
                S = np.std(window_returns, ddof=1)

                if S > 0:
                    rs_sum += R / S

            rs_avg = rs_sum / n_windows

            if rs_avg > 0:
                rs_values.append(np.log(rs_avg))
                window_sizes.append(np.log(window))

        # Linear regression to find Hurst exponent
        if len(rs_values) < 3:
            return {'signal': 'NEUTRAL', 'hurst': 0.5, 'regime': 'RANDOM'}

        x = np.array(window_sizes)
        y = np.array(rs_values)

        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)

        hurst = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)

        # Clamp to valid range
        hurst = max(0, min(1, hurst))

        # Determine regime
        if hurst < 0.4:
            regime = 'MEAN_REVERTING'
            signal = 'CONTRARIAN'  # Use mean reversion strategies
            description = 'Price tends to revert - use oscillator strategies'
        elif hurst > 0.6:
            regime = 'TRENDING'
            signal = 'MOMENTUM'  # Use trend following
            description = 'Strong trend persistence - use momentum strategies'
        else:
            regime = 'RANDOM'
            signal = 'NEUTRAL'  # Market is random, be cautious
            description = 'Random walk - reduce position size or avoid'

        return {
            'signal': signal,
            'hurst': round(hurst, 3),
            'regime': regime,
            'description': description,
            'tradeable': hurst < 0.4 or hurst > 0.6,
            'confidence': abs(hurst - 0.5) * 2  # 0-1 scale
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'hurst': 0.5, 'regime': 'ERROR', 'error': str(e)}


def calculate_zscore(ohlcv: List, period: int = 20) -> Dict[str, Any]:
    """
    🏛️ STATISTICAL EDGE: Z-Score Analysis

    Measures how many standard deviations price is from its mean.
    - Z > 2: Extremely overbought (sell opportunity)
    - Z > 1: Overbought (cautious on longs)
    - Z < -1: Oversold (cautious on shorts)
    - Z < -2: Extremely oversold (buy opportunity)

    Best combined with Hurst exponent - use Z-score when H < 0.5 (mean-reverting).

    Args:
        ohlcv: OHLCV data
        period: Lookback period for mean/std calculation

    Returns:
        Dict with Z-score and deviation analysis
    """
    try:
        if len(ohlcv) < period + 5:
            return {'signal': 'NEUTRAL', 'zscore': 0, 'deviation': 'NORMAL'}

        closes = [c[4] for c in ohlcv]
        current_price = closes[-1]

        # Calculate rolling mean and std
        lookback_prices = closes[-period:]
        mean_price = sum(lookback_prices) / len(lookback_prices)

        variance = sum((p - mean_price) ** 2 for p in lookback_prices) / len(lookback_prices)
        std_price = variance ** 0.5

        if std_price == 0:
            return {'signal': 'NEUTRAL', 'zscore': 0, 'deviation': 'NO_VOLATILITY'}

        # Calculate Z-score
        zscore = (current_price - mean_price) / std_price

        # Determine deviation level
        if zscore > 2.5:
            deviation = 'EXTREME_OVERBOUGHT'
            signal = 'STRONG_SELL'
            description = 'Price 2.5+ std above mean - high probability reversal down'
        elif zscore > 2.0:
            deviation = 'OVERBOUGHT'
            signal = 'SELL'
            description = 'Price 2+ std above mean - likely to revert'
        elif zscore > 1.0:
            deviation = 'SLIGHTLY_OVERBOUGHT'
            signal = 'NEUTRAL'
            description = 'Price elevated but within normal range'
        elif zscore < -2.5:
            deviation = 'EXTREME_OVERSOLD'
            signal = 'STRONG_BUY'
            description = 'Price 2.5+ std below mean - high probability reversal up'
        elif zscore < -2.0:
            deviation = 'OVERSOLD'
            signal = 'BUY'
            description = 'Price 2+ std below mean - likely to revert'
        elif zscore < -1.0:
            deviation = 'SLIGHTLY_OVERSOLD'
            signal = 'NEUTRAL'
            description = 'Price depressed but within normal range'
        else:
            deviation = 'NORMAL'
            signal = 'NEUTRAL'
            description = 'Price near mean - no statistical edge'

        # Calculate reversion probability based on Z-score
        reversion_prob = min(0.95, 0.5 + abs(zscore) * 0.15)

        return {
            'signal': signal,
            'zscore': round(zscore, 3),
            'deviation': deviation,
            'description': description,
            'mean_price': round(mean_price, 4),
            'std_price': round(std_price, 4),
            'current_price': current_price,
            'distance_from_mean_pct': round((current_price - mean_price) / mean_price * 100, 2),
            'reversion_probability': round(reversion_prob, 2)
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'zscore': 0, 'deviation': 'ERROR', 'error': str(e)}


def calculate_session_analysis(ohlcv: List, current_hour: int = None) -> Dict[str, Any]:
    """
    🏛️ TIME-BASED ANALYSIS: Trading Session Analysis

    Analyzes performance across different trading sessions:
    - Asian Session (00:00-08:00 UTC): Lower volatility, range-bound
    - European Session (08:00-16:00 UTC): Increasing volatility
    - US Session (14:00-22:00 UTC): Highest volatility, major moves

    Crypto tends to be more volatile during US hours.

    Args:
        ohlcv: OHLCV data with timestamps
        current_hour: Current UTC hour (0-23)

    Returns:
        Dict with session analysis and optimal trading times
    """
    try:
        from datetime import datetime

        if current_hour is None:
            current_hour = datetime.utcnow().hour

        # Define sessions
        if 0 <= current_hour < 8:
            session = 'ASIAN'
            volatility_expectation = 'LOW'
            optimal_strategy = 'RANGE_TRADING'
            signal_weight = 0.8  # Reduce signal weight in low vol
        elif 8 <= current_hour < 14:
            session = 'EUROPEAN'
            volatility_expectation = 'MEDIUM'
            optimal_strategy = 'BREAKOUT_WATCH'
            signal_weight = 1.0
        elif 14 <= current_hour < 22:
            session = 'US'
            volatility_expectation = 'HIGH'
            optimal_strategy = 'MOMENTUM'
            signal_weight = 1.2  # Increase weight during high vol
        else:
            session = 'LATE_US'
            volatility_expectation = 'MEDIUM'
            optimal_strategy = 'CAUTION'
            signal_weight = 0.9

        # Analyze historical session performance if we have timestamp data
        session_stats = {
            'asian_avg_move': 0,
            'european_avg_move': 0,
            'us_avg_move': 0
        }

        if ohlcv and len(ohlcv) > 20:
            # Calculate average moves
            moves = []
            for i in range(1, len(ohlcv)):
                move = abs(ohlcv[i][4] - ohlcv[i-1][4]) / ohlcv[i-1][4] * 100
                moves.append(move)

            if moves:
                avg_move = sum(moves) / len(moves)
                session_stats['overall_avg_move'] = round(avg_move, 3)

        return {
            'signal': 'NEUTRAL',  # Session analysis modifies other signals
            'session': session,
            'volatility_expectation': volatility_expectation,
            'optimal_strategy': optimal_strategy,
            'signal_weight': signal_weight,
            'current_hour_utc': current_hour,
            'session_stats': session_stats,
            'recommendation': f"Current session: {session} - {optimal_strategy} recommended"
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'session': 'UNKNOWN', 'error': str(e)}


def calculate_institutional_indicators(ohlcv: List) -> Dict[str, Any]:
    """
    🏛️ v4.6.0: MASTER FUNCTION - All Institutional Grade Indicators Combined

    Combines all professional indicators into a single comprehensive analysis:
    - Smart Money Concepts (Order Blocks, FVG, Liquidity Sweeps)
    - Market Structure (BOS, CHoCH)
    - Wyckoff VSA (Volume Spread Analysis)
    - Statistical Edge (Hurst Exponent, Z-Score)
    - Session Analysis

    Returns:
        Dict with complete institutional analysis and combined signals
    """
    try:
        if len(ohlcv) < 50:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0,
                'components': {},
                'error': 'Insufficient data for institutional analysis'
            }

        # Calculate all components
        order_blocks = calculate_order_blocks(ohlcv)
        fair_value_gaps = calculate_fair_value_gaps(ohlcv)
        liquidity_sweeps = calculate_liquidity_sweeps(ohlcv)
        market_structure = calculate_market_structure(ohlcv)
        wyckoff_vsa = calculate_wyckoff_vsa(ohlcv)
        hurst = calculate_hurst_exponent(ohlcv)
        zscore = calculate_zscore(ohlcv)
        session = calculate_session_analysis(ohlcv)

        result = {
            'order_blocks': order_blocks,
            'fair_value_gaps': fair_value_gaps,
            'liquidity_sweeps': liquidity_sweeps,
            'market_structure': market_structure,
            'wyckoff_vsa': wyckoff_vsa,
            'hurst': hurst,
            'zscore': zscore,
            'session': session
        }

        # Aggregate signals
        signals = []

        # Weight signals by importance
        signal_weights = {
            'market_structure': 2.0,   # Most important - structure defines bias
            'liquidity_sweeps': 1.8,   # Very strong reversal signal
            'wyckoff_vsa': 1.5,        # Smart money activity
            'order_blocks': 1.3,       # Key zones
            'fair_value_gaps': 1.2,    # Imbalance zones
            'zscore': 1.0,             # Statistical edge
            'hurst': 0.8               # Strategy selector (not direct signal)
        }

        for name, weight in signal_weights.items():
            component = result.get(name, {})
            sig = component.get('signal', 'NEUTRAL')

            if sig == 'STRONG_BUY':
                signals.append(('bullish', name, 2 * weight))
            elif sig == 'BUY':
                signals.append(('bullish', name, 1 * weight))
            elif sig == 'STRONG_SELL':
                signals.append(('bearish', name, 2 * weight))
            elif sig == 'SELL':
                signals.append(('bearish', name, 1 * weight))

        # Calculate weighted score
        bullish_score = sum(s[2] for s in signals if s[0] == 'bullish')
        bearish_score = sum(s[2] for s in signals if s[0] == 'bearish')

        # Apply session weight
        session_weight = session.get('signal_weight', 1.0)
        bullish_score *= session_weight
        bearish_score *= session_weight

        # Apply Hurst modifier
        # If trending regime, boost momentum signals
        # If mean-reverting, boost contrarian signals
        hurst_regime = hurst.get('regime', 'RANDOM')
        hurst_value = hurst.get('hurst', 0.5)

        if hurst_regime == 'TRENDING':
            # Boost momentum (structure-aligned) signals
            if market_structure.get('structure') == 'BULLISH':
                bullish_score *= 1.2
            elif market_structure.get('structure') == 'BEARISH':
                bearish_score *= 1.2
        elif hurst_regime == 'MEAN_REVERTING':
            # Boost contrarian signals
            zscore_val = zscore.get('zscore', 0)
            if zscore_val < -1.5:
                bullish_score *= 1.3
            elif zscore_val > 1.5:
                bearish_score *= 1.3

        # Determine final signal
        total_score = bullish_score + bearish_score

        if total_score == 0:
            final_signal = 'NEUTRAL'
            confidence = 0
        else:
            score_diff = bullish_score - bearish_score
            confidence = abs(score_diff) / total_score * 100

            if score_diff > 3:
                final_signal = 'STRONG_BUY'
            elif score_diff > 1:
                final_signal = 'BUY'
            elif score_diff < -3:
                final_signal = 'STRONG_SELL'
            elif score_diff < -1:
                final_signal = 'SELL'
            else:
                final_signal = 'NEUTRAL'

        result['signal'] = final_signal
        result['confidence'] = round(confidence, 1)
        result['bullish_score'] = round(bullish_score, 2)
        result['bearish_score'] = round(bearish_score, 2)
        result['signals_breakdown'] = signals

        # Add key insights
        insights = []

        if liquidity_sweeps.get('sweep_detected'):
            insights.append(f"🔥 LIQUIDITY SWEEP detected - reversal likely")

        if market_structure.get('choch'):
            insights.append(f"⚠️ CHANGE OF CHARACTER - trend reversal signal")
        elif market_structure.get('bos'):
            insights.append(f"📈 BREAK OF STRUCTURE - trend continuation")

        if wyckoff_vsa.get('phase') == 'ACCUMULATION':
            insights.append("🏛️ ACCUMULATION phase - smart money buying")
        elif wyckoff_vsa.get('phase') == 'DISTRIBUTION':
            insights.append("🏛️ DISTRIBUTION phase - smart money selling")

        if hurst.get('tradeable'):
            insights.append(f"📊 Hurst={hurst.get('hurst', 0.5):.2f} - {hurst.get('regime')} market")

        if abs(zscore.get('zscore', 0)) > 2:
            insights.append(f"📏 Z-Score={zscore.get('zscore', 0):.2f} - {zscore.get('deviation')}")

        result['insights'] = insights

        return result

    except Exception as e:
        return {
            'signal': 'NEUTRAL',
            'confidence': 0,
            'error': str(e)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 v4.7.0: ULTRA PROFESSIONAL TRADING INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1: Derivatives Analysis (Funding Rate, OI, L/S Ratio, Fear & Greed)
# Tier 2: Advanced Analysis (CVD, Liquidation Levels, Ichimoku Cloud)
# Tier 3: Pattern Recognition (Harmonic Patterns)
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_funding_rate_signal(funding_rate: float = None, funding_history: List[float] = None) -> Dict[str, Any]:
    """
    💰 TIER 1: Funding Rate Analysis

    Funding Rate = Premium/Discount between perpetual and spot price
    - Positive: Longs pay Shorts (too many longs = bearish signal)
    - Negative: Shorts pay Longs (too many shorts = bullish signal)

    Extreme funding rates predict reversals with ~85% accuracy.

    Args:
        funding_rate: Current funding rate (e.g., 0.0001 = 0.01%)
        funding_history: List of recent funding rates

    Returns:
        Dict with signal and analysis
    """
    try:
        if funding_rate is None:
            return {'signal': 'NEUTRAL', 'funding_rate': 0, 'interpretation': 'No data'}

        # Convert to percentage for easier interpretation
        funding_pct = funding_rate * 100

        # Determine signal based on funding rate thresholds
        if funding_pct >= 0.1:
            signal = 'STRONG_SELL'  # Extreme positive - too many longs
            interpretation = 'EXTREME LONG CROWDING - High reversal probability'
            contrarian_strength = min(100, funding_pct * 500)
        elif funding_pct >= 0.05:
            signal = 'SELL'
            interpretation = 'High long bias - Consider contrarian short'
            contrarian_strength = funding_pct * 400
        elif funding_pct >= 0.01:
            signal = 'SLIGHT_BEARISH'
            interpretation = 'Moderate long bias - Normal market'
            contrarian_strength = funding_pct * 200
        elif funding_pct <= -0.1:
            signal = 'STRONG_BUY'  # Extreme negative - too many shorts
            interpretation = 'EXTREME SHORT CROWDING - High reversal probability'
            contrarian_strength = min(100, abs(funding_pct) * 500)
        elif funding_pct <= -0.05:
            signal = 'BUY'
            interpretation = 'High short bias - Consider contrarian long'
            contrarian_strength = abs(funding_pct) * 400
        elif funding_pct <= -0.01:
            signal = 'SLIGHT_BULLISH'
            interpretation = 'Moderate short bias - Normal market'
            contrarian_strength = abs(funding_pct) * 200
        else:
            signal = 'NEUTRAL'
            interpretation = 'Balanced funding - No edge'
            contrarian_strength = 0

        # Analyze funding history trend if available
        funding_trend = 'STABLE'
        if funding_history and len(funding_history) >= 3:
            recent_avg = sum(funding_history[-3:]) / 3
            older_avg = sum(funding_history[:3]) / 3 if len(funding_history) >= 6 else recent_avg

            if recent_avg > older_avg * 1.5:
                funding_trend = 'INCREASING'  # Longs increasing
            elif recent_avg < older_avg * 0.5:
                funding_trend = 'DECREASING'  # Shorts increasing

        return {
            'signal': signal,
            'funding_rate': funding_rate,
            'funding_pct': round(funding_pct, 4),
            'interpretation': interpretation,
            'contrarian_strength': round(contrarian_strength, 1),
            'funding_trend': funding_trend,
            'is_extreme': abs(funding_pct) >= 0.05
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'funding_rate': 0, 'error': str(e)}


def calculate_open_interest_signal(
    current_oi: float,
    oi_history: List[float] = None,
    price_history: List[float] = None
) -> Dict[str, Any]:
    """
    📊 TIER 1: Open Interest Analysis

    OI = Total number of outstanding derivative contracts

    Price ↑ + OI ↑ = Strong bullish (new money entering longs)
    Price ↑ + OI ↓ = Weak rally (short covering)
    Price ↓ + OI ↑ = Strong bearish (new money entering shorts)
    Price ↓ + OI ↓ = Weak decline (long liquidation)

    Args:
        current_oi: Current open interest value
        oi_history: Historical OI values
        price_history: Historical price values (aligned with OI)

    Returns:
        Dict with OI analysis and signal
    """
    try:
        if not oi_history or len(oi_history) < 5:
            return {'signal': 'NEUTRAL', 'current_oi': current_oi, 'interpretation': 'Insufficient data'}

        # Calculate OI change
        oi_change_pct = (current_oi - oi_history[-5]) / oi_history[-5] * 100 if oi_history[-5] > 0 else 0

        # Calculate price change if available
        price_change_pct = 0
        if price_history and len(price_history) >= 5:
            price_change_pct = (price_history[-1] - price_history[-5]) / price_history[-5] * 100

        # Determine signal based on OI + Price relationship
        signal = 'NEUTRAL'
        interpretation = ''
        strength = 0

        if oi_change_pct > 5:  # OI increasing significantly
            if price_change_pct > 1:
                signal = 'STRONG_BUY'
                interpretation = 'NEW LONGS ENTERING - Strong bullish momentum'
                strength = min(100, oi_change_pct * 2 + price_change_pct * 5)
            elif price_change_pct < -1:
                signal = 'STRONG_SELL'
                interpretation = 'NEW SHORTS ENTERING - Strong bearish pressure'
                strength = min(100, oi_change_pct * 2 + abs(price_change_pct) * 5)
            else:
                signal = 'NEUTRAL'
                interpretation = 'OI rising but price flat - Building positions'
                strength = 30

        elif oi_change_pct < -5:  # OI decreasing significantly
            if price_change_pct > 1:
                signal = 'SLIGHT_BEARISH'
                interpretation = 'SHORT COVERING RALLY - Weak, may reverse'
                strength = 40
            elif price_change_pct < -1:
                signal = 'SLIGHT_BULLISH'
                interpretation = 'LONG LIQUIDATION - Capitulation, bounce possible'
                strength = 50
            else:
                signal = 'NEUTRAL'
                interpretation = 'Positions closing - Lower conviction'
                strength = 20
        else:
            signal = 'NEUTRAL'
            interpretation = 'Stable OI - No significant change'
            strength = 10

        # Calculate OI trend
        oi_trend = 'STABLE'
        if len(oi_history) >= 10:
            recent_oi = sum(oi_history[-3:]) / 3
            older_oi = sum(oi_history[-10:-7]) / 3
            if recent_oi > older_oi * 1.1:
                oi_trend = 'RISING'
            elif recent_oi < older_oi * 0.9:
                oi_trend = 'FALLING'

        return {
            'signal': signal,
            'current_oi': current_oi,
            'oi_change_pct': round(oi_change_pct, 2),
            'price_change_pct': round(price_change_pct, 2),
            'interpretation': interpretation,
            'strength': round(strength, 1),
            'oi_trend': oi_trend,
            'is_significant': abs(oi_change_pct) > 5
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'current_oi': 0, 'error': str(e)}


def calculate_long_short_ratio_signal(
    long_short_ratio: float,
    ratio_history: List[float] = None
) -> Dict[str, Any]:
    """
    📈 TIER 1: Long/Short Ratio Analysis

    L/S Ratio = Long accounts / Short accounts (or positions)

    - Ratio > 2.0: Too many longs → Contrarian SHORT
    - Ratio > 1.5: High long bias → Caution on longs
    - Ratio ≈ 1.0: Balanced market
    - Ratio < 0.7: High short bias → Caution on shorts
    - Ratio < 0.5: Too many shorts → Contrarian LONG

    Extreme ratios predict reversals (contrarian indicator).

    Args:
        long_short_ratio: Current ratio (e.g., 1.5 means 1.5 longs per short)
        ratio_history: Historical ratio values

    Returns:
        Dict with analysis and contrarian signal
    """
    try:
        if long_short_ratio is None or long_short_ratio <= 0:
            return {'signal': 'NEUTRAL', 'ratio': 0, 'interpretation': 'Invalid data'}

        ratio = long_short_ratio

        # Determine contrarian signal
        if ratio >= 2.5:
            signal = 'STRONG_SELL'  # Extreme long crowding
            interpretation = 'EXTREME LONG CROWDING - High probability reversal down'
            crowd_sentiment = 'EXTREME_BULLISH'
            contrarian_confidence = 90
        elif ratio >= 2.0:
            signal = 'SELL'
            interpretation = 'Heavy long bias - Contrarian short opportunity'
            crowd_sentiment = 'VERY_BULLISH'
            contrarian_confidence = 75
        elif ratio >= 1.5:
            signal = 'SLIGHT_BEARISH'
            interpretation = 'Moderate long bias - Be cautious on longs'
            crowd_sentiment = 'BULLISH'
            contrarian_confidence = 50
        elif ratio <= 0.4:
            signal = 'STRONG_BUY'  # Extreme short crowding
            interpretation = 'EXTREME SHORT CROWDING - High probability reversal up'
            crowd_sentiment = 'EXTREME_BEARISH'
            contrarian_confidence = 90
        elif ratio <= 0.5:
            signal = 'BUY'
            interpretation = 'Heavy short bias - Contrarian long opportunity'
            crowd_sentiment = 'VERY_BEARISH'
            contrarian_confidence = 75
        elif ratio <= 0.7:
            signal = 'SLIGHT_BULLISH'
            interpretation = 'Moderate short bias - Be cautious on shorts'
            crowd_sentiment = 'BEARISH'
            contrarian_confidence = 50
        else:
            signal = 'NEUTRAL'
            interpretation = 'Balanced positioning - No contrarian edge'
            crowd_sentiment = 'NEUTRAL'
            contrarian_confidence = 20

        # Analyze ratio trend
        ratio_trend = 'STABLE'
        if ratio_history and len(ratio_history) >= 5:
            recent_ratio = sum(ratio_history[-3:]) / 3
            older_ratio = sum(ratio_history[:3]) / 3 if len(ratio_history) >= 6 else recent_ratio

            if recent_ratio > older_ratio * 1.2:
                ratio_trend = 'LONGS_INCREASING'
            elif recent_ratio < older_ratio * 0.8:
                ratio_trend = 'SHORTS_INCREASING'

        # Calculate percentages
        long_pct = ratio / (ratio + 1) * 100
        short_pct = 100 - long_pct

        return {
            'signal': signal,
            'ratio': round(ratio, 3),
            'long_pct': round(long_pct, 1),
            'short_pct': round(short_pct, 1),
            'interpretation': interpretation,
            'crowd_sentiment': crowd_sentiment,
            'contrarian_confidence': contrarian_confidence,
            'ratio_trend': ratio_trend,
            'is_extreme': ratio >= 2.0 or ratio <= 0.5
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'ratio': 0, 'error': str(e)}


def calculate_fear_greed_index(ohlcv: List, volatility_data: Dict = None, volume_data: Dict = None) -> Dict[str, Any]:
    """
    😱 TIER 1: Fear & Greed Index (Crypto-specific calculation)

    Composite index 0-100:
    - 0-25: Extreme Fear → BUY (Contrarian)
    - 25-45: Fear → Watch for LONG
    - 45-55: Neutral
    - 55-75: Greed → Watch for SHORT
    - 75-100: Extreme Greed → SELL (Contrarian)

    Components:
    1. Volatility (25%): High vol = Fear
    2. Market Momentum (25%): Based on price trend
    3. Volume (25%): High volume can indicate fear/greed
    4. Price Distance from MA (25%): Far from mean = extreme sentiment

    Args:
        ohlcv: OHLCV data
        volatility_data: Optional volatility metrics
        volume_data: Optional volume metrics

    Returns:
        Dict with Fear & Greed index and interpretation
    """
    try:
        if len(ohlcv) < 30:
            return {'signal': 'NEUTRAL', 'index': 50, 'sentiment': 'NEUTRAL', 'interpretation': 'Insufficient data'}

        closes = [c[4] for c in ohlcv]
        volumes = [c[5] for c in ohlcv]
        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]

        current_price = closes[-1]

        # 1. VOLATILITY COMPONENT (25%)
        # High volatility = Fear, Low volatility = Complacency/Greed
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        volatility = np.std(returns[-14:]) * 100  # 14-day volatility
        avg_volatility = np.std(returns) * 100

        if volatility > avg_volatility * 1.5:
            volatility_score = 20  # High vol = Fear
        elif volatility > avg_volatility * 1.2:
            volatility_score = 35
        elif volatility < avg_volatility * 0.5:
            volatility_score = 80  # Low vol = Complacency
        elif volatility < avg_volatility * 0.8:
            volatility_score = 65
        else:
            volatility_score = 50

        # 2. MOMENTUM COMPONENT (25%)
        # Strong uptrend = Greed, Strong downtrend = Fear
        ma_20 = sum(closes[-20:]) / 20
        ma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else ma_20

        price_vs_ma20 = (current_price - ma_20) / ma_20 * 100
        price_vs_ma50 = (current_price - ma_50) / ma_50 * 100

        momentum_score = 50 + (price_vs_ma20 * 2) + (price_vs_ma50 * 1)
        momentum_score = max(0, min(100, momentum_score))

        # 3. VOLUME COMPONENT (25%)
        # Spike in volume during decline = Fear, Spike during rally = Greed
        avg_volume = sum(volumes[-20:]) / 20
        recent_volume = sum(volumes[-3:]) / 3
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        price_change_3d = (closes[-1] - closes[-4]) / closes[-4] * 100

        if volume_ratio > 1.5 and price_change_3d < -3:
            volume_score = 20  # Panic selling
        elif volume_ratio > 1.5 and price_change_3d > 3:
            volume_score = 80  # FOMO buying
        elif volume_ratio > 1.2:
            volume_score = 55 if price_change_3d > 0 else 45
        else:
            volume_score = 50

        # 4. PRICE DISTANCE FROM MEAN (25%)
        # Far above mean = Greed, Far below = Fear
        sma_30 = sum(closes[-30:]) / 30
        distance_pct = (current_price - sma_30) / sma_30 * 100

        distance_score = 50 + (distance_pct * 3)
        distance_score = max(0, min(100, distance_score))

        # CALCULATE FINAL INDEX
        fear_greed_index = (
            volatility_score * 0.25 +
            momentum_score * 0.25 +
            volume_score * 0.25 +
            distance_score * 0.25
        )

        fear_greed_index = round(max(0, min(100, fear_greed_index)), 1)

        # Determine sentiment and signal
        if fear_greed_index <= 20:
            sentiment = 'EXTREME_FEAR'
            signal = 'STRONG_BUY'  # Contrarian
            interpretation = '😱 EXTREME FEAR - Strong contrarian buy signal'
        elif fear_greed_index <= 35:
            sentiment = 'FEAR'
            signal = 'BUY'
            interpretation = '😰 FEAR - Consider buying on weakness'
        elif fear_greed_index <= 45:
            sentiment = 'SLIGHT_FEAR'
            signal = 'SLIGHT_BULLISH'
            interpretation = '😟 Slight fear - Market uncertain'
        elif fear_greed_index <= 55:
            sentiment = 'NEUTRAL'
            signal = 'NEUTRAL'
            interpretation = '😐 Neutral - No sentiment edge'
        elif fear_greed_index <= 65:
            sentiment = 'SLIGHT_GREED'
            signal = 'SLIGHT_BEARISH'
            interpretation = '🙂 Slight greed - Market optimistic'
        elif fear_greed_index <= 80:
            sentiment = 'GREED'
            signal = 'SELL'
            interpretation = '🤑 GREED - Consider taking profits'
        else:
            sentiment = 'EXTREME_GREED'
            signal = 'STRONG_SELL'  # Contrarian
            interpretation = '🚀😵 EXTREME GREED - Strong contrarian sell signal'

        return {
            'signal': signal,
            'index': fear_greed_index,
            'sentiment': sentiment,
            'interpretation': interpretation,
            'components': {
                'volatility': round(volatility_score, 1),
                'momentum': round(momentum_score, 1),
                'volume': round(volume_score, 1),
                'price_distance': round(distance_score, 1)
            },
            'is_extreme': fear_greed_index <= 25 or fear_greed_index >= 75
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'index': 50, 'sentiment': 'ERROR', 'error': str(e)}


def calculate_cvd(ohlcv: List, lookback: int = 50) -> Dict[str, Any]:
    """
    📈 TIER 2: Cumulative Volume Delta (CVD) Analysis

    CVD estimates buying vs selling pressure from OHLCV data.

    Delta = (Close - Low) / (High - Low) * Volume  (buying pressure)
         - (High - Close) / (High - Low) * Volume  (selling pressure)

    Price ↑ + CVD ↑ = Genuine buying, trend strong
    Price ↑ + CVD ↓ = Fake rally, distribution, reversal likely
    Price ↓ + CVD ↑ = Accumulation, reversal likely
    Price ↓ + CVD ↓ = Genuine selling, trend strong

    Args:
        ohlcv: OHLCV data
        lookback: Period for analysis

    Returns:
        Dict with CVD analysis and divergence detection
    """
    try:
        if len(ohlcv) < lookback:
            return {'signal': 'NEUTRAL', 'cvd': 0, 'interpretation': 'Insufficient data'}

        # Calculate delta for each candle
        deltas = []
        cumulative_delta = 0
        cvd_values = []

        for candle in ohlcv[-lookback:]:
            high = candle[2]
            low = candle[3]
            close = candle[4]
            volume = candle[5]

            candle_range = high - low
            if candle_range > 0:
                # Buy pressure: how much of the candle closed toward the high
                buy_pressure = ((close - low) / candle_range) * volume
                # Sell pressure: how much of the candle closed toward the low
                sell_pressure = ((high - close) / candle_range) * volume
                delta = buy_pressure - sell_pressure
            else:
                delta = 0

            deltas.append(delta)
            cumulative_delta += delta
            cvd_values.append(cumulative_delta)

        # Current CVD and trend
        current_cvd = cvd_values[-1] if cvd_values else 0
        cvd_change = cvd_values[-1] - cvd_values[-10] if len(cvd_values) >= 10 else 0

        # Price change over same period
        closes = [c[4] for c in ohlcv[-lookback:]]
        price_change = closes[-1] - closes[-10] if len(closes) >= 10 else 0
        price_change_pct = (price_change / closes[-10] * 100) if len(closes) >= 10 and closes[-10] > 0 else 0

        # Detect divergence
        divergence = 'NONE'
        signal = 'NEUTRAL'
        interpretation = ''

        if price_change > 0 and cvd_change > 0:
            signal = 'BUY'
            divergence = 'NONE'
            interpretation = 'CONFIRMED UPTREND - Price and CVD rising together'
            strength = min(80, abs(price_change_pct) * 5 + 40)
        elif price_change > 0 and cvd_change < 0:
            signal = 'SELL'  # Bearish divergence
            divergence = 'BEARISH'
            interpretation = '⚠️ BEARISH DIVERGENCE - Price up but selling pressure, reversal likely'
            strength = 70
        elif price_change < 0 and cvd_change > 0:
            signal = 'BUY'  # Bullish divergence
            divergence = 'BULLISH'
            interpretation = '⚠️ BULLISH DIVERGENCE - Price down but buying pressure, reversal likely'
            strength = 70
        elif price_change < 0 and cvd_change < 0:
            signal = 'SELL'
            divergence = 'NONE'
            interpretation = 'CONFIRMED DOWNTREND - Price and CVD falling together'
            strength = min(80, abs(price_change_pct) * 5 + 40)
        else:
            signal = 'NEUTRAL'
            interpretation = 'No clear signal'
            strength = 30

        # Calculate recent delta trend (last 5 candles)
        recent_delta_sum = sum(deltas[-5:])
        delta_trend = 'BUYING' if recent_delta_sum > 0 else 'SELLING' if recent_delta_sum < 0 else 'NEUTRAL'

        return {
            'signal': signal,
            'cvd': round(current_cvd, 2),
            'cvd_change': round(cvd_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'divergence': divergence,
            'interpretation': interpretation,
            'delta_trend': delta_trend,
            'recent_delta': round(recent_delta_sum, 2),
            'strength': round(strength, 1),
            'is_divergent': divergence != 'NONE'
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'cvd': 0, 'error': str(e)}


def calculate_liquidation_levels(
    ohlcv: List,
    current_price: float,
    leverage_levels: List[int] = [5, 10, 20, 25, 50, 100]
) -> Dict[str, Any]:
    """
    🔥 TIER 2: Liquidation Level Estimation

    Estimates where leveraged positions would get liquidated.
    Price tends to hunt these levels (liquidity hunting by smart money).

    Liquidation Price (Long) = Entry * (1 - 1/Leverage)
    Liquidation Price (Short) = Entry * (1 + 1/Leverage)

    Args:
        ohlcv: OHLCV data to find recent entry clusters
        current_price: Current market price
        leverage_levels: Common leverage levels to analyze

    Returns:
        Dict with liquidation clusters and hunting probability
    """
    try:
        if len(ohlcv) < 20:
            return {'signal': 'NEUTRAL', 'levels': [], 'interpretation': 'Insufficient data'}

        closes = [c[4] for c in ohlcv]
        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        volumes = [c[5] for c in ohlcv]

        # Find significant price levels (high volume areas = likely entry points)
        avg_volume = sum(volumes) / len(volumes)
        significant_levels = []

        for i in range(len(ohlcv)):
            if volumes[i] > avg_volume * 1.5:
                significant_levels.append(closes[i])

        # Also add recent swing highs/lows
        for i in range(2, len(highs) - 2):
            if highs[i] > max(highs[i-2:i]) and highs[i] > max(highs[i+1:i+3]):
                significant_levels.append(highs[i])
            if lows[i] < min(lows[i-2:i]) and lows[i] < min(lows[i+1:i+3]):
                significant_levels.append(lows[i])

        # Calculate liquidation levels for each significant price
        long_liquidations = []  # Levels below current price
        short_liquidations = []  # Levels above current price

        for entry_price in significant_levels:
            for lev in leverage_levels:
                # Long liquidation (below entry)
                long_liq = entry_price * (1 - 0.9 / lev)  # 90% of margin lost
                # Short liquidation (above entry)
                short_liq = entry_price * (1 + 0.9 / lev)

                if long_liq < current_price:
                    long_liquidations.append({
                        'level': long_liq,
                        'entry': entry_price,
                        'leverage': lev,
                        'distance_pct': (current_price - long_liq) / current_price * 100
                    })

                if short_liq > current_price:
                    short_liquidations.append({
                        'level': short_liq,
                        'entry': entry_price,
                        'leverage': lev,
                        'distance_pct': (short_liq - current_price) / current_price * 100
                    })

        # Cluster liquidation levels
        def cluster_levels(levels, tolerance_pct=0.5):
            if not levels:
                return []

            sorted_levels = sorted(levels, key=lambda x: x['level'])
            clusters = []
            current_cluster = [sorted_levels[0]]

            for lev in sorted_levels[1:]:
                if abs(lev['level'] - current_cluster[-1]['level']) / current_cluster[-1]['level'] * 100 < tolerance_pct:
                    current_cluster.append(lev)
                else:
                    if len(current_cluster) >= 2:
                        avg_level = sum(l['level'] for l in current_cluster) / len(current_cluster)
                        clusters.append({
                            'level': avg_level,
                            'count': len(current_cluster),
                            'avg_leverage': sum(l['leverage'] for l in current_cluster) / len(current_cluster)
                        })
                    current_cluster = [lev]

            if len(current_cluster) >= 2:
                avg_level = sum(l['level'] for l in current_cluster) / len(current_cluster)
                clusters.append({
                    'level': avg_level,
                    'count': len(current_cluster),
                    'avg_leverage': sum(l['leverage'] for l in current_cluster) / len(current_cluster)
                })

            return clusters

        long_clusters = cluster_levels(long_liquidations)
        short_clusters = cluster_levels(short_liquidations)

        # Find nearest liquidation cluster
        nearest_long = None
        nearest_short = None

        if long_clusters:
            nearest_long = max(long_clusters, key=lambda x: x['level'])
            nearest_long['distance_pct'] = (current_price - nearest_long['level']) / current_price * 100

        if short_clusters:
            nearest_short = min(short_clusters, key=lambda x: x['level'])
            nearest_short['distance_pct'] = (nearest_short['level'] - current_price) / current_price * 100

        # Determine which side is more likely to be hunted
        signal = 'NEUTRAL'
        interpretation = ''
        hunt_target = None

        if nearest_long and nearest_short:
            if nearest_long['distance_pct'] < nearest_short['distance_pct']:
                if nearest_long['distance_pct'] < 3:
                    signal = 'SELL'  # Price likely to dip to hunt longs
                    hunt_target = 'LONG_LIQUIDATIONS'
                    interpretation = f"⚠️ Long liquidations at ${nearest_long['level']:.2f} ({nearest_long['distance_pct']:.1f}% away) - Dip likely"
            else:
                if nearest_short['distance_pct'] < 3:
                    signal = 'BUY'  # Price likely to spike to hunt shorts
                    hunt_target = 'SHORT_LIQUIDATIONS'
                    interpretation = f"⚠️ Short liquidations at ${nearest_short['level']:.2f} ({nearest_short['distance_pct']:.1f}% away) - Spike likely"

        if not interpretation:
            interpretation = 'No immediate liquidation hunting expected'

        return {
            'signal': signal,
            'nearest_long_liq': nearest_long,
            'nearest_short_liq': nearest_short,
            'long_clusters': long_clusters[:3],  # Top 3
            'short_clusters': short_clusters[:3],
            'hunt_target': hunt_target,
            'interpretation': interpretation,
            'current_price': current_price
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'levels': [], 'error': str(e)}


def calculate_ichimoku_cloud(ohlcv: List) -> Dict[str, Any]:
    """
    ⛩️ TIER 2: Ichimoku Cloud Analysis (Japanese Institutional Standard)

    Complete trading system in one indicator:
    - Tenkan-sen (Conversion, 9): Short-term momentum
    - Kijun-sen (Base, 26): Medium-term momentum
    - Senkou Span A (Leading A): (Tenkan + Kijun) / 2, shifted 26 periods
    - Senkou Span B (Leading B): 52-period midpoint, shifted 26 periods
    - Chikou Span (Lagging): Close shifted back 26 periods

    Signals:
    - Price above cloud = Bullish
    - Price below cloud = Bearish
    - Price in cloud = Neutral/Transition
    - TK Cross above Kijun = Buy
    - TK Cross below Kijun = Sell

    Args:
        ohlcv: OHLCV data (need at least 78 candles for full calculation)

    Returns:
        Dict with complete Ichimoku analysis
    """
    try:
        if len(ohlcv) < 78:
            return {'signal': 'NEUTRAL', 'cloud_position': 'UNKNOWN', 'interpretation': 'Need 78+ candles'}

        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        closes = [c[4] for c in ohlcv]

        current_price = closes[-1]

        # Calculate Tenkan-sen (9-period)
        def period_midpoint(highs, lows, period, index):
            h = max(highs[index-period+1:index+1])
            l = min(lows[index-period+1:index+1])
            return (h + l) / 2

        tenkan = period_midpoint(highs, lows, 9, len(ohlcv)-1)

        # Calculate Kijun-sen (26-period)
        kijun = period_midpoint(highs, lows, 26, len(ohlcv)-1)

        # Calculate Senkou Span A (current, not shifted for simplicity)
        senkou_a = (tenkan + kijun) / 2

        # Calculate Senkou Span B (52-period)
        senkou_b = period_midpoint(highs, lows, 52, len(ohlcv)-1)

        # Chikou Span (current close vs price 26 periods ago)
        chikou = closes[-1]
        price_26_ago = closes[-27] if len(closes) > 26 else closes[0]

        # Determine cloud bounds
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        cloud_thickness = cloud_top - cloud_bottom
        cloud_thickness_pct = (cloud_thickness / current_price) * 100

        # Determine cloud color (future trend)
        cloud_color = 'GREEN' if senkou_a > senkou_b else 'RED'

        # Analyze price position relative to cloud
        if current_price > cloud_top:
            cloud_position = 'ABOVE_CLOUD'
            position_strength = min(100, ((current_price - cloud_top) / cloud_top * 100) * 10 + 50)
        elif current_price < cloud_bottom:
            cloud_position = 'BELOW_CLOUD'
            position_strength = min(100, ((cloud_bottom - current_price) / cloud_bottom * 100) * 10 + 50)
        else:
            cloud_position = 'IN_CLOUD'
            position_strength = 30

        # TK Cross analysis
        tk_cross = 'NONE'
        if tenkan > kijun:
            tk_cross = 'BULLISH'  # Tenkan above Kijun
        elif tenkan < kijun:
            tk_cross = 'BEARISH'

        # Chikou analysis (current vs 26 periods ago)
        chikou_signal = 'BULLISH' if chikou > price_26_ago else 'BEARISH'

        # Generate overall signal
        bullish_factors = 0
        bearish_factors = 0

        if cloud_position == 'ABOVE_CLOUD':
            bullish_factors += 2
        elif cloud_position == 'BELOW_CLOUD':
            bearish_factors += 2

        if tk_cross == 'BULLISH':
            bullish_factors += 1.5
        elif tk_cross == 'BEARISH':
            bearish_factors += 1.5

        if cloud_color == 'GREEN':
            bullish_factors += 1
        else:
            bearish_factors += 1

        if chikou_signal == 'BULLISH':
            bullish_factors += 0.5
        else:
            bearish_factors += 0.5

        # Final signal
        if bullish_factors >= 4:
            signal = 'STRONG_BUY'
            interpretation = '⛩️ STRONG BULLISH - All Ichimoku factors aligned'
        elif bullish_factors >= 3:
            signal = 'BUY'
            interpretation = '⛩️ BULLISH - Most factors support uptrend'
        elif bearish_factors >= 4:
            signal = 'STRONG_SELL'
            interpretation = '⛩️ STRONG BEARISH - All Ichimoku factors aligned'
        elif bearish_factors >= 3:
            signal = 'SELL'
            interpretation = '⛩️ BEARISH - Most factors support downtrend'
        elif cloud_position == 'IN_CLOUD':
            signal = 'NEUTRAL'
            interpretation = '⛩️ IN CLOUD - Transition period, wait for breakout'
        else:
            signal = 'NEUTRAL'
            interpretation = '⛩️ Mixed signals - No clear direction'

        return {
            'signal': signal,
            'tenkan': round(tenkan, 4),
            'kijun': round(kijun, 4),
            'senkou_a': round(senkou_a, 4),
            'senkou_b': round(senkou_b, 4),
            'cloud_top': round(cloud_top, 4),
            'cloud_bottom': round(cloud_bottom, 4),
            'cloud_position': cloud_position,
            'cloud_color': cloud_color,
            'cloud_thickness_pct': round(cloud_thickness_pct, 2),
            'tk_cross': tk_cross,
            'chikou_signal': chikou_signal,
            'interpretation': interpretation,
            'position_strength': round(position_strength, 1),
            'bullish_factors': bullish_factors,
            'bearish_factors': bearish_factors
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'cloud_position': 'ERROR', 'error': str(e)}


def calculate_harmonic_patterns(ohlcv: List, tolerance: float = 0.02) -> Dict[str, Any]:
    """
    🦋 TIER 3: Harmonic Pattern Detection

    Fibonacci-based geometric patterns with precise reversal zones:

    1. GARTLEY (222): Most reliable
       - XA: Impulse move
       - AB: 61.8% retracement of XA
       - BC: 38.2%-88.6% retracement of AB
       - CD: 78.6% retracement of XA (PRZ)

    2. BUTTERFLY: Extended pattern
       - AB: 78.6% retracement of XA
       - BC: 38.2%-88.6% retracement of AB
       - CD: 127%-161.8% extension of XA (PRZ)

    3. BAT: Deep retracement
       - AB: 38.2%-50% retracement of XA
       - BC: 38.2%-88.6% retracement of AB
       - CD: 88.6% retracement of XA (PRZ)

    4. CRAB: Most extreme
       - AB: 38.2%-61.8% retracement of XA
       - BC: 38.2%-88.6% retracement of AB
       - CD: 161.8% extension of XA (PRZ)

    PRZ = Potential Reversal Zone (high probability entry)

    Args:
        ohlcv: OHLCV data
        tolerance: Fibonacci ratio tolerance (default 2%)

    Returns:
        Dict with detected patterns and PRZ levels
    """
    try:
        if len(ohlcv) < 50:
            return {'signal': 'NEUTRAL', 'patterns': [], 'interpretation': 'Insufficient data'}

        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        closes = [c[4] for c in ohlcv]

        current_price = closes[-1]

        # Find swing points (XABCD)
        def find_swing_points(highs, lows, min_swing=5):
            """Find significant swing highs and lows."""
            swing_highs = []
            swing_lows = []

            for i in range(min_swing, len(highs) - min_swing):
                # Swing high
                if highs[i] == max(highs[i-min_swing:i+min_swing+1]):
                    swing_highs.append({'index': i, 'price': highs[i]})
                # Swing low
                if lows[i] == min(lows[i-min_swing:i+min_swing+1]):
                    swing_lows.append({'index': i, 'price': lows[i]})

            return swing_highs, swing_lows

        swing_highs, swing_lows = find_swing_points(highs, lows)

        def check_ratio(actual, expected, tolerance):
            """Check if actual ratio is within tolerance of expected."""
            return abs(actual - expected) <= tolerance

        def check_ratio_range(actual, min_ratio, max_ratio, tolerance):
            """Check if actual ratio is within range."""
            return (min_ratio - tolerance) <= actual <= (max_ratio + tolerance)

        detected_patterns = []

        # Combine and sort all swing points
        all_swings = []
        for sh in swing_highs:
            all_swings.append({'index': sh['index'], 'price': sh['price'], 'type': 'HIGH'})
        for sl in swing_lows:
            all_swings.append({'index': sl['index'], 'price': sl['price'], 'type': 'LOW'})

        all_swings.sort(key=lambda x: x['index'])

        # Need at least 5 swing points for XABCD
        if len(all_swings) < 5:
            return {'signal': 'NEUTRAL', 'patterns': [], 'interpretation': 'Not enough swing points'}

        # Check last 5 swing points for patterns
        for i in range(len(all_swings) - 4):
            X = all_swings[i]
            A = all_swings[i + 1]
            B = all_swings[i + 2]
            C = all_swings[i + 3]
            D = all_swings[i + 4]

            # Ensure alternating highs and lows
            if X['type'] == A['type'] or A['type'] == B['type'] or B['type'] == C['type'] or C['type'] == D['type']:
                continue

            # Calculate ratios
            XA = abs(A['price'] - X['price'])
            AB = abs(B['price'] - A['price'])
            BC = abs(C['price'] - B['price'])
            CD = abs(D['price'] - C['price'])

            if XA == 0:
                continue

            AB_XA = AB / XA
            BC_AB = BC / AB if AB > 0 else 0
            CD_XA = CD / XA if XA > 0 else 0
            CD_BC = CD / BC if BC > 0 else 0

            pattern_type = None
            pattern_direction = 'BULLISH' if D['type'] == 'LOW' else 'BEARISH'

            # GARTLEY Pattern
            if check_ratio(AB_XA, 0.618, tolerance):
                if check_ratio_range(BC_AB, 0.382, 0.886, tolerance):
                    if check_ratio(CD_XA, 0.786, tolerance):
                        pattern_type = 'GARTLEY'

            # BUTTERFLY Pattern
            if check_ratio(AB_XA, 0.786, tolerance):
                if check_ratio_range(BC_AB, 0.382, 0.886, tolerance):
                    if check_ratio_range(CD_XA, 1.27, 1.618, tolerance):
                        pattern_type = 'BUTTERFLY'

            # BAT Pattern
            if check_ratio_range(AB_XA, 0.382, 0.5, tolerance):
                if check_ratio_range(BC_AB, 0.382, 0.886, tolerance):
                    if check_ratio(CD_XA, 0.886, tolerance):
                        pattern_type = 'BAT'

            # CRAB Pattern
            if check_ratio_range(AB_XA, 0.382, 0.618, tolerance):
                if check_ratio_range(BC_AB, 0.382, 0.886, tolerance):
                    if check_ratio(CD_XA, 1.618, tolerance * 2):  # More tolerance for extreme
                        pattern_type = 'CRAB'

            if pattern_type:
                # Calculate PRZ (Potential Reversal Zone)
                prz_center = D['price']
                prz_range = XA * 0.05  # 5% of XA as PRZ range

                # Check if current price is near PRZ
                distance_to_prz = abs(current_price - prz_center) / current_price * 100

                detected_patterns.append({
                    'type': pattern_type,
                    'direction': pattern_direction,
                    'X': X['price'],
                    'A': A['price'],
                    'B': B['price'],
                    'C': C['price'],
                    'D': D['price'],
                    'prz_center': prz_center,
                    'prz_high': prz_center + prz_range,
                    'prz_low': prz_center - prz_range,
                    'distance_to_prz_pct': round(distance_to_prz, 2),
                    'is_active': distance_to_prz < 3,
                    'completion_index': D['index']
                })

        # Sort by recency and activity
        detected_patterns.sort(key=lambda x: (-x['is_active'], -x['completion_index']))

        # Generate signal based on most recent active pattern
        signal = 'NEUTRAL'
        interpretation = 'No harmonic patterns detected'
        active_pattern = None

        if detected_patterns:
            # Find most recent active pattern
            for pattern in detected_patterns:
                if pattern['is_active']:
                    active_pattern = pattern
                    break

            if not active_pattern and detected_patterns:
                active_pattern = detected_patterns[0]

            if active_pattern:
                if active_pattern['is_active']:
                    if active_pattern['direction'] == 'BULLISH':
                        signal = 'STRONG_BUY'
                        interpretation = f"🦋 ACTIVE {active_pattern['type']} (BULLISH) - PRZ at ${active_pattern['prz_center']:.4f}"
                    else:
                        signal = 'STRONG_SELL'
                        interpretation = f"🦋 ACTIVE {active_pattern['type']} (BEARISH) - PRZ at ${active_pattern['prz_center']:.4f}"
                else:
                    if active_pattern['direction'] == 'BULLISH':
                        signal = 'BUY'
                        interpretation = f"🦋 {active_pattern['type']} forming (BULLISH) - Watch for PRZ at ${active_pattern['prz_center']:.4f}"
                    else:
                        signal = 'SELL'
                        interpretation = f"🦋 {active_pattern['type']} forming (BEARISH) - Watch for PRZ at ${active_pattern['prz_center']:.4f}"

        return {
            'signal': signal,
            'patterns': detected_patterns[:3],  # Top 3 patterns
            'active_pattern': active_pattern,
            'total_patterns': len(detected_patterns),
            'interpretation': interpretation,
            'has_active': any(p['is_active'] for p in detected_patterns)
        }

    except Exception as e:
        return {'signal': 'NEUTRAL', 'patterns': [], 'error': str(e)}


def calculate_derivatives_analysis(
    ohlcv: List,
    funding_rate: float = None,
    open_interest: float = None,
    oi_history: List[float] = None,
    long_short_ratio: float = None
) -> Dict[str, Any]:
    """
    🚀 v4.7.0: MASTER FUNCTION - Complete Derivatives Analysis

    Combines all Tier 1 indicators:
    - Funding Rate Analysis
    - Open Interest Analysis
    - Long/Short Ratio Analysis
    - Fear & Greed Index

    Args:
        ohlcv: OHLCV data
        funding_rate: Current funding rate
        open_interest: Current OI value
        oi_history: Historical OI values
        long_short_ratio: Current L/S ratio

    Returns:
        Dict with complete derivatives analysis
    """
    try:
        closes = [c[4] for c in ohlcv] if ohlcv else []

        # Calculate each component
        funding_analysis = calculate_funding_rate_signal(funding_rate)
        oi_analysis = calculate_open_interest_signal(
            open_interest or 0,
            oi_history,
            closes[-len(oi_history):] if oi_history and closes else None
        )
        ls_analysis = calculate_long_short_ratio_signal(long_short_ratio)
        fear_greed = calculate_fear_greed_index(ohlcv) if ohlcv else {'signal': 'NEUTRAL', 'index': 50}

        # Aggregate signals
        signals = {
            'funding': funding_analysis.get('signal', 'NEUTRAL'),
            'oi': oi_analysis.get('signal', 'NEUTRAL'),
            'ls_ratio': ls_analysis.get('signal', 'NEUTRAL'),
            'fear_greed': fear_greed.get('signal', 'NEUTRAL')
        }

        # Calculate composite score
        signal_scores = {
            'STRONG_BUY': 2, 'BUY': 1, 'SLIGHT_BULLISH': 0.5,
            'NEUTRAL': 0,
            'SLIGHT_BEARISH': -0.5, 'SELL': -1, 'STRONG_SELL': -2
        }

        total_score = sum(signal_scores.get(s, 0) for s in signals.values())

        # Determine final signal
        if total_score >= 3:
            final_signal = 'STRONG_BUY'
            interpretation = '🚀 DERIVATIVES ULTRA BULLISH - All factors aligned for long'
        elif total_score >= 1.5:
            final_signal = 'BUY'
            interpretation = '📈 Derivatives favor long positions'
        elif total_score <= -3:
            final_signal = 'STRONG_SELL'
            interpretation = '🔻 DERIVATIVES ULTRA BEARISH - All factors aligned for short'
        elif total_score <= -1.5:
            final_signal = 'SELL'
            interpretation = '📉 Derivatives favor short positions'
        else:
            final_signal = 'NEUTRAL'
            interpretation = '➖ Mixed derivatives signals'

        return {
            'signal': final_signal,
            'composite_score': round(total_score, 1),
            'interpretation': interpretation,
            'funding': funding_analysis,
            'open_interest': oi_analysis,
            'long_short_ratio': ls_analysis,
            'fear_greed': fear_greed,
            'signals_breakdown': signals
        }

    except Exception as e:
        return {
            'signal': 'NEUTRAL',
            'composite_score': 0,
            'error': str(e)
        }


def calculate_advanced_analysis(ohlcv: List) -> Dict[str, Any]:
    """
    🔬 v4.7.0: MASTER FUNCTION - Complete Advanced Analysis (Tier 2 + 3)

    Combines:
    - CVD (Cumulative Volume Delta)
    - Liquidation Levels
    - Ichimoku Cloud
    - Harmonic Patterns

    Args:
        ohlcv: OHLCV data

    Returns:
        Dict with complete advanced analysis
    """
    try:
        if len(ohlcv) < 78:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0,
                'error': 'Insufficient data (need 78+ candles)'
            }

        current_price = ohlcv[-1][4]

        # Calculate each component
        cvd_analysis = calculate_cvd(ohlcv)
        liquidation_analysis = calculate_liquidation_levels(ohlcv, current_price)
        ichimoku_analysis = calculate_ichimoku_cloud(ohlcv)
        harmonic_analysis = calculate_harmonic_patterns(ohlcv)

        # Aggregate signals with weights
        signal_weights = {
            'cvd': 1.5,        # Volume delta important
            'ichimoku': 2.0,   # Complete system
            'harmonic': 1.5,   # Precise reversals
            'liquidation': 1.0  # Market mechanics
        }

        signal_scores = {
            'STRONG_BUY': 2, 'BUY': 1, 'SLIGHT_BULLISH': 0.5,
            'NEUTRAL': 0,
            'SLIGHT_BEARISH': -0.5, 'SELL': -1, 'STRONG_SELL': -2
        }

        weighted_score = 0
        total_weight = sum(signal_weights.values())

        weighted_score += signal_scores.get(cvd_analysis.get('signal', 'NEUTRAL'), 0) * signal_weights['cvd']
        weighted_score += signal_scores.get(ichimoku_analysis.get('signal', 'NEUTRAL'), 0) * signal_weights['ichimoku']
        weighted_score += signal_scores.get(harmonic_analysis.get('signal', 'NEUTRAL'), 0) * signal_weights['harmonic']
        weighted_score += signal_scores.get(liquidation_analysis.get('signal', 'NEUTRAL'), 0) * signal_weights['liquidation']

        normalized_score = weighted_score / total_weight

        # Determine final signal
        if normalized_score >= 1.2:
            final_signal = 'STRONG_BUY'
            interpretation = '🔬 ADVANCED ANALYSIS ULTRA BULLISH'
        elif normalized_score >= 0.5:
            final_signal = 'BUY'
            interpretation = '🔬 Advanced analysis favors long'
        elif normalized_score <= -1.2:
            final_signal = 'STRONG_SELL'
            interpretation = '🔬 ADVANCED ANALYSIS ULTRA BEARISH'
        elif normalized_score <= -0.5:
            final_signal = 'SELL'
            interpretation = '🔬 Advanced analysis favors short'
        else:
            final_signal = 'NEUTRAL'
            interpretation = '🔬 Mixed advanced signals'

        # Calculate confidence based on agreement
        signals_list = [
            cvd_analysis.get('signal', 'NEUTRAL'),
            ichimoku_analysis.get('signal', 'NEUTRAL'),
            harmonic_analysis.get('signal', 'NEUTRAL'),
            liquidation_analysis.get('signal', 'NEUTRAL')
        ]

        bullish_count = sum(1 for s in signals_list if 'BUY' in s)
        bearish_count = sum(1 for s in signals_list if 'SELL' in s)

        confidence = max(bullish_count, bearish_count) / len(signals_list) * 100

        # Key insights
        insights = []

        if cvd_analysis.get('is_divergent'):
            insights.append(f"⚠️ CVD DIVERGENCE: {cvd_analysis.get('divergence')}")

        if ichimoku_analysis.get('cloud_position') in ['ABOVE_CLOUD', 'BELOW_CLOUD']:
            insights.append(f"⛩️ ICHIMOKU: {ichimoku_analysis.get('cloud_position')}")

        if harmonic_analysis.get('has_active'):
            pattern = harmonic_analysis.get('active_pattern', {})
            insights.append(f"🦋 HARMONIC: Active {pattern.get('type', 'pattern')} ({pattern.get('direction', '')})")

        if liquidation_analysis.get('hunt_target'):
            insights.append(f"🎯 LIQUIDATION HUNT: {liquidation_analysis.get('hunt_target')}")

        return {
            'signal': final_signal,
            'normalized_score': round(normalized_score, 2),
            'confidence': round(confidence, 1),
            'interpretation': interpretation,
            'insights': insights,
            'cvd': cvd_analysis,
            'liquidation': liquidation_analysis,
            'ichimoku': ichimoku_analysis,
            'harmonic': harmonic_analysis
        }

    except Exception as e:
        return {
            'signal': 'NEUTRAL',
            'confidence': 0,
            'error': str(e)
        }


def calculate_ultra_professional_analysis(
    ohlcv: List,
    funding_rate: float = None,
    open_interest: float = None,
    oi_history: List[float] = None,
    long_short_ratio: float = None
) -> Dict[str, Any]:
    """
    🏆 v4.7.0: ULTIMATE MASTER FUNCTION - Complete Professional Analysis

    Combines ALL Tier 1, 2, and 3 indicators:

    TIER 1 (Derivatives):
    - Funding Rate Analysis
    - Open Interest Analysis
    - Long/Short Ratio Analysis
    - Fear & Greed Index

    TIER 2 (Advanced):
    - CVD (Cumulative Volume Delta)
    - Liquidation Levels
    - Ichimoku Cloud

    TIER 3 (Pattern Recognition):
    - Harmonic Patterns (Gartley, Butterfly, Bat, Crab)

    Args:
        ohlcv: OHLCV data
        funding_rate: Current funding rate
        open_interest: Current OI value
        oi_history: Historical OI values
        long_short_ratio: Current L/S ratio

    Returns:
        Dict with complete ultra professional analysis
    """
    try:
        if len(ohlcv) < 78:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0,
                'total_score': 0,
                'error': 'Insufficient data (need 78+ candles)'
            }

        # Calculate all component analyses
        derivatives_data = calculate_derivatives_analysis(
            ohlcv, funding_rate, open_interest, oi_history, long_short_ratio
        )
        advanced_data = calculate_advanced_analysis(ohlcv)

        # Weight the two main components
        derivatives_weight = 0.4  # 40% weight
        advanced_weight = 0.6     # 60% weight

        # Convert signals to scores
        signal_scores = {
            'STRONG_BUY': 100, 'BUY': 70, 'SLIGHT_BULLISH': 55,
            'NEUTRAL': 50,
            'SLIGHT_BEARISH': 45, 'SELL': 30, 'STRONG_SELL': 0
        }

        deriv_score = signal_scores.get(derivatives_data.get('signal', 'NEUTRAL'), 50)
        adv_score = signal_scores.get(advanced_data.get('signal', 'NEUTRAL'), 50)

        # Calculate composite score (0-100)
        composite_score = (deriv_score * derivatives_weight) + (adv_score * advanced_weight)

        # Determine final signal
        if composite_score >= 80:
            final_signal = 'STRONG_BUY'
            interpretation = '🏆 ULTRA PROFESSIONAL: ALL SYSTEMS BULLISH'
        elif composite_score >= 65:
            final_signal = 'BUY'
            interpretation = '🏆 Professional analysis favors LONG'
        elif composite_score <= 20:
            final_signal = 'STRONG_SELL'
            interpretation = '🏆 ULTRA PROFESSIONAL: ALL SYSTEMS BEARISH'
        elif composite_score <= 35:
            final_signal = 'SELL'
            interpretation = '🏆 Professional analysis favors SHORT'
        else:
            final_signal = 'NEUTRAL'
            interpretation = '🏆 Mixed professional signals - WAIT'

        # Collect all insights
        all_insights = []

        # Derivatives insights
        if derivatives_data.get('funding', {}).get('is_extreme'):
            all_insights.append(f"💰 EXTREME FUNDING: {derivatives_data['funding'].get('funding_pct', 0):.3f}%")

        if derivatives_data.get('open_interest', {}).get('is_significant'):
            all_insights.append(f"📊 OI SIGNAL: {derivatives_data['open_interest'].get('interpretation', '')}")

        if derivatives_data.get('long_short_ratio', {}).get('is_extreme'):
            all_insights.append(f"📈 L/S EXTREME: {derivatives_data['long_short_ratio'].get('ratio', 0):.2f}")

        if derivatives_data.get('fear_greed', {}).get('is_extreme'):
            all_insights.append(f"😱 F&G: {derivatives_data['fear_greed'].get('sentiment', '')} ({derivatives_data['fear_greed'].get('index', 50)})")

        # Advanced insights
        all_insights.extend(advanced_data.get('insights', []))

        return {
            'signal': final_signal,
            'composite_score': round(composite_score, 1),
            'interpretation': interpretation,
            'insights': all_insights[:5],  # Top 5 insights
            'derivatives_analysis': derivatives_data,
            'advanced_analysis': advanced_data,
            'components': {
                'derivatives_signal': derivatives_data.get('signal', 'NEUTRAL'),
                'derivatives_score': deriv_score,
                'advanced_signal': advanced_data.get('signal', 'NEUTRAL'),
                'advanced_score': adv_score
            }
        }

    except Exception as e:
        return {
            'signal': 'NEUTRAL',
            'composite_score': 50,
            'error': str(e)
        }
