"""
Professional Prediction Chart Generator - TradingView Style
Creates ultra-professional charts with Entry, TP, SL, and Trend predictions.

Inspired by professional trading chart screenshots - clean, minimal, institutional-grade.
"""

import mplfinance as mpf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import io
from src.utils import setup_logging
from src.indicators import (
    calculate_indicators,
    detect_support_resistance_levels,
    calculate_atr
)

logger = setup_logging()


class PredictionChartGenerator:
    """
    Generates professional prediction charts with:
    - Entry point marker
    - Take Profit levels (TP1, TP2, TP3)
    - Stop Loss level
    - Trend lines (ascending/descending channels)
    - Support/Resistance zones
    - Prediction arrow (big yellow arrow showing direction)
    """

    def __init__(self):
        """Initialize with TradingView dark theme."""
        # TradingView Dark Theme Colors
        self.colors = {
            'background': '#131722',
            'grid': '#1E222D',
            'text': '#D1D4DC',
            'bull': '#26A69A',
            'bear': '#EF5350',
            'support': '#4CAF50',
            'resistance': '#F44336',
            'entry': '#FFD700',  # Gold for entry
            'tp': '#00E676',  # Bright green for TP
            'sl': '#FF1744',  # Bright red for SL
            'trend_up': '#26A69A',
            'trend_down': '#EF5350',
            'ema_fast': '#2962FF',
            'ema_slow': '#FF6D00',
            'prediction_arrow': '#FFEB3B',  # Yellow for prediction
            'channel': '#FFFFFF',  # White for channel lines
        }

        # Chart style
        self.style = mpf.make_mpf_style(
            base_mpf_style='charles',
            rc={
                'figure.facecolor': self.colors['background'],
                'axes.facecolor': self.colors['background'],
                'axes.edgecolor': self.colors['grid'],
                'axes.labelcolor': self.colors['text'],
                'xtick.color': self.colors['text'],
                'ytick.color': self.colors['text'],
                'grid.color': self.colors['grid'],
                'grid.alpha': 0.3,
            },
            marketcolors=mpf.make_marketcolors(
                up=self.colors['bull'],
                down=self.colors['bear'],
                edge='inherit',
                wick='inherit',
                volume='#26A69A80',
            )
        )

    def detect_channel(
        self,
        df: pd.DataFrame,
        lookback: int = 15
    ) -> Dict[str, Any]:
        """
        Detect price channel (ascending/descending/horizontal).

        Returns channel type and line parameters for drawing.
        """
        try:
            highs = df['high'].values
            lows = df['low'].values
            n = len(df)

            # Find swing points
            swing_highs = []
            swing_lows = []

            for i in range(lookback, n - 5):
                # Swing high
                if highs[i] == max(highs[max(0, i-lookback):min(n, i+lookback+1)]):
                    swing_highs.append((i, highs[i]))
                # Swing low
                if lows[i] == min(lows[max(0, i-lookback):min(n, i+lookback+1)]):
                    swing_lows.append((i, lows[i]))

            channel_info = {
                'type': 'unknown',
                'upper_line': None,
                'lower_line': None,
                'slope': 0
            }

            # Fit upper trendline (resistance)
            if len(swing_highs) >= 2:
                recent_highs = swing_highs[-4:] if len(swing_highs) >= 4 else swing_highs
                x_h = np.array([p[0] for p in recent_highs])
                y_h = np.array([p[1] for p in recent_highs])
                if len(x_h) >= 2:
                    coeffs_h = np.polyfit(x_h, y_h, 1)
                    channel_info['upper_line'] = (coeffs_h[0], coeffs_h[1])

            # Fit lower trendline (support)
            if len(swing_lows) >= 2:
                recent_lows = swing_lows[-4:] if len(swing_lows) >= 4 else swing_lows
                x_l = np.array([p[0] for p in recent_lows])
                y_l = np.array([p[1] for p in recent_lows])
                if len(x_l) >= 2:
                    coeffs_l = np.polyfit(x_l, y_l, 1)
                    channel_info['lower_line'] = (coeffs_l[0], coeffs_l[1])

            # Determine channel type
            if channel_info['upper_line'] and channel_info['lower_line']:
                avg_slope = (channel_info['upper_line'][0] + channel_info['lower_line'][0]) / 2
                channel_info['slope'] = avg_slope

                if avg_slope > 0.0001:
                    channel_info['type'] = 'ascending'
                elif avg_slope < -0.0001:
                    channel_info['type'] = 'descending'
                else:
                    channel_info['type'] = 'horizontal'

            return channel_info

        except Exception as e:
            logger.error(f"Error detecting channel: {e}")
            return {'type': 'unknown', 'upper_line': None, 'lower_line': None, 'slope': 0}

    def calculate_entry_tp_sl(
        self,
        df: pd.DataFrame,
        indicators: Dict,
        direction: str  # 'LONG' or 'SHORT'
    ) -> Dict[str, float]:
        """
        Calculate Entry, TP levels, and SL based on technical analysis.

        Uses ATR for volatility-based levels and S/R for targets.
        """
        current_price = float(df['close'].iloc[-1])
        atr = indicators.get('atr', current_price * 0.02)
        atr_pct = (atr / current_price) * 100

        # Get S/R levels - reset index to include timestamp column
        df_reset = df.reset_index()
        # Convert timestamp to milliseconds for detect_support_resistance_levels
        df_reset['timestamp'] = df_reset['timestamp'].astype('int64') // 10**6
        sr_data = detect_support_resistance_levels(df_reset.values.tolist(), current_price)
        supports = sr_data.get('swing_lows', [current_price * 0.97])
        resistances = sr_data.get('swing_highs', [current_price * 1.03])

        # Filter to get nearest levels
        nearest_support = min([s for s in supports if s < current_price], default=current_price * 0.97)
        nearest_resistance = max([r for r in resistances if r > current_price], default=current_price * 1.03)

        if direction == 'LONG':
            # Entry: Current price (or slightly better)
            entry = current_price

            # Stop Loss: Below nearest support or ATR-based
            sl_atr = current_price - (atr * 2)
            sl_support = nearest_support * 0.995
            sl = max(sl_atr, sl_support)  # Take the tighter stop

            # Take Profit levels based on R:R ratios
            risk = entry - sl
            tp1 = entry + (risk * 1.5)  # 1.5:1 R:R
            tp2 = entry + (risk * 2.5)  # 2.5:1 R:R
            tp3 = min(nearest_resistance, entry + (risk * 4))  # 4:1 R:R or resistance

        else:  # SHORT
            entry = current_price

            # Stop Loss: Above nearest resistance or ATR-based
            sl_atr = current_price + (atr * 2)
            sl_resistance = nearest_resistance * 1.005
            sl = min(sl_atr, sl_resistance)

            # Take Profit levels
            risk = sl - entry
            tp1 = entry - (risk * 1.5)
            tp2 = entry - (risk * 2.5)
            tp3 = max(nearest_support, entry - (risk * 4))

        return {
            'entry': entry,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'risk_pct': abs(entry - sl) / entry * 100,
            'reward_pct': abs(tp2 - entry) / entry * 100
        }

    def determine_prediction(
        self,
        indicators: Dict,
        channel_info: Dict
    ) -> Dict[str, Any]:
        """
        Determine prediction direction based on indicators and channel.

        Returns prediction with confidence score.
        """
        score = 0  # Positive = LONG, Negative = SHORT
        reasons = []

        # RSI
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            score += 2
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            score -= 2
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi < 45:
            score += 1
            reasons.append(f"RSI bullish ({rsi:.1f})")
        elif rsi > 55:
            score -= 1
            reasons.append(f"RSI bearish ({rsi:.1f})")

        # MACD
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal:
            score += 1
            reasons.append("MACD bullish cross")
        else:
            score -= 1
            reasons.append("MACD bearish")

        # SuperTrend
        supertrend_signal = indicators.get('supertrend_signal', 'neutral')
        if supertrend_signal == 'buy':
            score += 2
            reasons.append("SuperTrend BUY")
        elif supertrend_signal == 'sell':
            score -= 2
            reasons.append("SuperTrend SELL")

        # Channel type
        if channel_info['type'] == 'descending':
            # Descending channel at bottom = potential breakout UP
            score += 1
            reasons.append("Descending channel (breakout potential)")
        elif channel_info['type'] == 'ascending':
            score += 1
            reasons.append("Ascending channel (trend continuation)")

        # EMA trend
        ema_20 = indicators.get('ema_20', 0)
        ema_50 = indicators.get('ema_50', 0)
        close = indicators.get('close', 0)

        if close > ema_20 > ema_50:
            score += 1
            reasons.append("Price above EMAs (bullish)")
        elif close < ema_20 < ema_50:
            score -= 1
            reasons.append("Price below EMAs (bearish)")

        # Determine direction and confidence
        if score >= 2:
            direction = 'LONG'
            confidence = min(90, 50 + (score * 10))
        elif score <= -2:
            direction = 'SHORT'
            confidence = min(90, 50 + (abs(score) * 10))
        else:
            direction = 'NEUTRAL'
            confidence = 40

        return {
            'direction': direction,
            'confidence': confidence,
            'score': score,
            'reasons': reasons
        }

    async def generate_prediction_chart(
        self,
        symbol: str,
        ohlcv_data: List[List],
        timeframe: str = '15m',
        width: int = 14,
        height: int = 10
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Generate professional prediction chart with Entry/TP/SL.

        Returns:
            Tuple of (PNG image bytes, prediction data dict)
        """
        try:
            logger.info(f"Generating prediction chart for {symbol}...")

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Istanbul')
            df.set_index('timestamp', inplace=True)

            # Calculate indicators
            indicators = calculate_indicators(ohlcv_data)
            current_price = float(df['close'].iloc[-1])

            # Add EMAs
            df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()

            # Detect channel
            channel_info = self.detect_channel(df)

            # Get prediction
            prediction = self.determine_prediction(indicators, channel_info)
            direction = prediction['direction']

            # Calculate Entry/TP/SL
            if direction != 'NEUTRAL':
                levels = self.calculate_entry_tp_sl(df, indicators, direction)
            else:
                # Default levels for neutral
                levels = {
                    'entry': current_price,
                    'sl': current_price * 0.97,
                    'tp1': current_price * 1.02,
                    'tp2': current_price * 1.04,
                    'tp3': current_price * 1.06,
                    'risk_pct': 3,
                    'reward_pct': 4
                }

            # Build chart
            apds = []

            # EMAs
            apds.append(mpf.make_addplot(df['EMA12'], color=self.colors['ema_fast'], width=1.2))
            apds.append(mpf.make_addplot(df['EMA26'], color=self.colors['ema_slow'], width=1.2))

            # Create figure
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=self.style,
                volume=True,
                addplot=apds,
                title=f"\n{symbol} - {timeframe} PREDICTION",
                ylabel='Price (USDT)',
                ylabel_lower='Volume',
                figsize=(width, height),
                panel_ratios=(4, 1),
                returnfig=True,
                warn_too_much_data=len(df) + 1,
            )

            ax_main = axes[0]
            n = len(df)

            # Draw channel lines (white, like in the reference images)
            if channel_info['upper_line']:
                slope, intercept = channel_info['upper_line']
                x_range = range(n)
                y_values = [slope * x + intercept for x in x_range]
                ax_main.plot(x_range, y_values, color='#FFFFFF', linewidth=2, alpha=0.8)

            if channel_info['lower_line']:
                slope, intercept = channel_info['lower_line']
                x_range = range(n)
                y_values = [slope * x + intercept for x in x_range]
                ax_main.plot(x_range, y_values, color='#FFFFFF', linewidth=2, alpha=0.8)

            # Draw Entry line
            ax_main.axhline(
                y=levels['entry'],
                color=self.colors['entry'],
                linestyle='-',
                linewidth=2,
                alpha=0.9
            )
            ax_main.text(
                n + 1, levels['entry'],
                f" ENTRY ${levels['entry']:.4f}",
                color=self.colors['entry'],
                fontsize=10,
                fontweight='bold',
                va='center'
            )

            # Draw Stop Loss line
            ax_main.axhline(
                y=levels['sl'],
                color=self.colors['sl'],
                linestyle='--',
                linewidth=2,
                alpha=0.9
            )
            ax_main.text(
                n + 1, levels['sl'],
                f" SL ${levels['sl']:.4f} ({levels['risk_pct']:.1f}%)",
                color=self.colors['sl'],
                fontsize=9,
                fontweight='bold',
                va='center'
            )

            # Draw TP levels
            for i, (tp_name, tp_val) in enumerate([('TP1', levels['tp1']), ('TP2', levels['tp2']), ('TP3', levels['tp3'])]):
                ax_main.axhline(
                    y=tp_val,
                    color=self.colors['tp'],
                    linestyle=':',
                    linewidth=1.5,
                    alpha=0.7
                )
                ax_main.text(
                    n + 1, tp_val,
                    f" {tp_name} ${tp_val:.4f}",
                    color=self.colors['tp'],
                    fontsize=9,
                    va='center'
                )

            # Draw big prediction arrow (like in reference images)
            if direction != 'NEUTRAL':
                arrow_start_x = n - 10
                arrow_start_y = current_price

                if direction == 'LONG':
                    arrow_end_y = levels['tp2']
                    arrow_color = self.colors['prediction_arrow']
                else:
                    arrow_end_y = levels['tp2']
                    arrow_color = self.colors['prediction_arrow']

                # Draw thick arrow
                ax_main.annotate(
                    '',
                    xy=(n + 5, arrow_end_y),
                    xytext=(arrow_start_x, arrow_start_y),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color=arrow_color,
                        lw=4,
                        mutation_scale=30
                    )
                )

            # Add prediction info box
            direction_emoji = "" if direction == 'LONG' else "" if direction == 'SHORT' else ""
            pred_text = f"{direction_emoji} {direction}\nConfidence: {prediction['confidence']}%"

            box_color = self.colors['bull'] if direction == 'LONG' else self.colors['bear'] if direction == 'SHORT' else '#888888'

            ax_main.text(
                0.02, 0.98,
                pred_text,
                transform=ax_main.transAxes,
                fontsize=14,
                fontweight='bold',
                color='white',
                va='top',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=box_color,
                    edgecolor='white',
                    linewidth=2,
                    alpha=0.9
                )
            )

            # Add current price
            price_text = f"${current_price:.4f}"
            ax_main.text(
                0.98, 0.98,
                price_text,
                transform=ax_main.transAxes,
                fontsize=12,
                fontweight='bold',
                color=self.colors['text'],
                va='top',
                ha='right',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=self.colors['background'],
                    edgecolor=self.colors['text'],
                    alpha=0.9
                )
            )

            # Add R:R info
            rr_ratio = levels['reward_pct'] / levels['risk_pct'] if levels['risk_pct'] > 0 else 0
            rr_text = f"R:R = 1:{rr_ratio:.1f}"
            ax_main.text(
                0.02, 0.88,
                rr_text,
                transform=ax_main.transAxes,
                fontsize=10,
                color=self.colors['text'],
                va='top'
            )

            # Add channel type
            channel_text = f"Channel: {channel_info['type'].upper()}"
            ax_main.text(
                0.02, 0.82,
                channel_text,
                transform=ax_main.transAxes,
                fontsize=10,
                color=self.colors['text'],
                va='top'
            )

            # Add timestamp
            turkey_tz = timezone(timedelta(hours=3))
            turkey_time = datetime.now(turkey_tz)
            timestamp_text = f"Generated: {turkey_time.strftime('%Y-%m-%d %H:%M:%S')} (Turkey Time)"
            fig.text(
                0.99, 0.01,
                timestamp_text,
                ha='right',
                va='bottom',
                fontsize=8,
                color=self.colors['text'],
                alpha=0.6
            )

            # Finalize
            fig.patch.set_facecolor(self.colors['background'])
            plt.tight_layout()

            # Save to bytes
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format='png',
                dpi=150,
                facecolor=self.colors['background'],
                edgecolor='none',
                bbox_inches='tight'
            )
            buf.seek(0)
            plt.close(fig)

            # Prepare prediction data
            prediction_data = {
                'symbol': symbol,
                'direction': direction,
                'confidence': prediction['confidence'],
                'entry': levels['entry'],
                'sl': levels['sl'],
                'tp1': levels['tp1'],
                'tp2': levels['tp2'],
                'tp3': levels['tp3'],
                'risk_pct': levels['risk_pct'],
                'reward_pct': levels['reward_pct'],
                'rr_ratio': rr_ratio,
                'channel_type': channel_info['type'],
                'reasons': prediction['reasons'],
                'indicators': {
                    'rsi': indicators.get('rsi', 50),
                    'macd': indicators.get('macd', 0),
                    'supertrend': indicators.get('supertrend_signal', 'neutral')
                }
            }

            logger.info(f"Prediction chart generated: {direction} with {prediction['confidence']}% confidence")
            return buf.read(), prediction_data

        except Exception as e:
            logger.error(f"Error generating prediction chart: {e}")
            raise


# Singleton instance
_prediction_generator: Optional[PredictionChartGenerator] = None


def get_prediction_generator() -> PredictionChartGenerator:
    """Get or create prediction generator instance."""
    global _prediction_generator
    if _prediction_generator is None:
        _prediction_generator = PredictionChartGenerator()
    return _prediction_generator
