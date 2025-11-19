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
