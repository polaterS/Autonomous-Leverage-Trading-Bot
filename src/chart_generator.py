"""
Ultra Professional TradingView-Style Chart Generator
Creates high-quality charts with support/resistance, trend lines, and indicators.
"""

import mplfinance as mpf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import setup_logging
from src.indicators import (
    calculate_indicators,
    detect_support_resistance_levels,
    detect_market_regime
)

logger = setup_logging()


class TradingViewChartGenerator:
    """
    Generates ultra-professional TradingView-style charts with:
    - Support/Resistance levels
    - Trend lines
    - Technical indicators (RSI, MACD, Volume)
    - Moving averages
    - High-resolution export
    """

    def __init__(self):
        """Initialize chart generator with TradingView-style theme."""
        # TradingView Dark Theme Colors
        self.colors = {
            'background': '#131722',
            'grid': '#1E222D',
            'text': '#D1D4DC',
            'bull': '#26A69A',  # Green candles
            'bear': '#EF5350',  # Red candles
            'support': '#4CAF50',  # Green support lines
            'resistance': '#F44336',  # Red resistance lines
            'trend_up': '#00E676',  # Bright green trend
            'trend_down': '#FF1744',  # Bright red trend
            'ema_fast': '#2962FF',  # Blue
            'ema_slow': '#FF6D00',  # Orange
            'ema_long': '#FFD600',  # Yellow
            'volume': '#26A69A80',  # Semi-transparent green
            'rsi': '#9C27B0',  # Purple
            'macd': '#2196F3',  # Blue
            'macd_signal': '#FF9800',  # Orange
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
                volume=self.colors['volume'],
            )
        )

    def detect_trend_lines(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        Detect trend lines by connecting swing highs/lows.

        Args:
            df: DataFrame with OHLCV data
            lookback: Number of candles to look back for swing points

        Returns:
            Dict with 'uptrend' and 'downtrend' line parameters (slope, intercept)
        """
        try:
            highs = df['high'].values
            lows = df['low'].values

            # Find swing highs (local maxima)
            swing_highs = []
            for i in range(lookback, len(highs) - lookback):
                if highs[i] == max(highs[i-lookback:i+lookback+1]):
                    swing_highs.append((i, highs[i]))

            # Find swing lows (local minima)
            swing_lows = []
            for i in range(lookback, len(lows) - lookback):
                if lows[i] == min(lows[i-lookback:i+lookback+1]):
                    swing_lows.append((i, lows[i]))

            # Fit trend lines
            uptrend_line = None
            downtrend_line = None

            # Uptrend line (connect rising lows)
            if len(swing_lows) >= 2:
                # Take last 3 swing lows
                recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
                x_coords = [point[0] for point in recent_lows]
                y_coords = [point[1] for point in recent_lows]

                # Fit line using numpy polyfit
                if len(x_coords) >= 2:
                    coeffs = np.polyfit(x_coords, y_coords, 1)
                    slope, intercept = coeffs[0], coeffs[1]

                    # Only use if upward sloping
                    if slope > 0:
                        uptrend_line = (slope, intercept)

            # Downtrend line (connect falling highs)
            if len(swing_highs) >= 2:
                # Take last 3 swing highs
                recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
                x_coords = [point[0] for point in recent_highs]
                y_coords = [point[1] for point in recent_highs]

                # Fit line using numpy polyfit
                if len(x_coords) >= 2:
                    coeffs = np.polyfit(x_coords, y_coords, 1)
                    slope, intercept = coeffs[0], coeffs[1]

                    # Only use if downward sloping
                    if slope < 0:
                        downtrend_line = (slope, intercept)

            return {
                'uptrend': uptrend_line,
                'downtrend': downtrend_line
            }

        except Exception as e:
            logger.error(f"Error detecting trend lines: {e}")
            return {'uptrend': None, 'downtrend': None}

    async def generate_chart(
        self,
        symbol: str,
        ohlcv_data: List[List],
        timeframe: str = '15m',
        show_indicators: bool = True,
        width: int = 14,
        height: int = 10
    ) -> bytes:
        """
        Generate ultra-professional TradingView-style chart.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            ohlcv_data: OHLCV data from exchange
            timeframe: Chart timeframe
            show_indicators: Whether to show RSI/MACD panels
            width: Chart width in inches
            height: Chart height in inches

        Returns:
            PNG image as bytes
        """
        try:
            logger.info(f"🎨 Generating ultra-pro chart for {symbol} ({timeframe})...")

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Calculate indicators (not used in chart, we calculate them directly)
            # indicators_15m = calculate_indicators(ohlcv_data)

            # Add EMAs to DataFrame
            df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

            # Calculate RSI and MACD
            df['RSI'] = self._calculate_rsi(df['close'], 14)
            macd, signal, hist = self._calculate_macd(df['close'])
            df['MACD'] = macd
            df['MACD_signal'] = signal
            df['MACD_hist'] = hist

            # Detect support/resistance levels
            current_price = float(df['close'].iloc[-1])
            support_resistance = detect_support_resistance_levels(ohlcv_data, current_price)
            support_levels = support_resistance.get('swing_lows', [])
            resistance_levels = support_resistance.get('swing_highs', [])

            logger.info(f"🔍 Support levels (swing_lows): {support_levels[:3]}")
            logger.info(f"🔍 Resistance levels (swing_highs): {resistance_levels[:3]}")

            # Detect trend lines
            trend_lines = self.detect_trend_lines(df)
            logger.info(f"🔍 Trend lines: uptrend={trend_lines['uptrend'] is not None}, downtrend={trend_lines['downtrend'] is not None}")

            # Build additional plots
            apds = []

            # EMAs on main chart
            ema_plots = [
                mpf.make_addplot(df['EMA12'], color=self.colors['ema_fast'], width=1.5, label='EMA 12'),
                mpf.make_addplot(df['EMA26'], color=self.colors['ema_slow'], width=1.5, label='EMA 26'),
                mpf.make_addplot(df['EMA50'], color=self.colors['ema_long'], width=1.5, label='EMA 50'),
            ]
            apds.extend(ema_plots)

            # RSI panel
            if show_indicators:
                rsi_plot = mpf.make_addplot(
                    df['RSI'],
                    panel=1,
                    color=self.colors['rsi'],
                    ylabel='RSI',
                    width=1.5,
                    ylim=(0, 100)
                )
                apds.append(rsi_plot)

                # RSI overbought/oversold lines
                rsi_upper = pd.Series([70] * len(df), index=df.index)
                rsi_lower = pd.Series([30] * len(df), index=df.index)
                apds.append(mpf.make_addplot(rsi_upper, panel=1, color='#666', linestyle='--', width=0.8))
                apds.append(mpf.make_addplot(rsi_lower, panel=1, color='#666', linestyle='--', width=0.8))

                # MACD panel
                macd_plot = mpf.make_addplot(
                    df['MACD'],
                    panel=2,
                    color=self.colors['macd'],
                    ylabel='MACD',
                    width=1.5
                )
                macd_signal_plot = mpf.make_addplot(
                    df['MACD_signal'],
                    panel=2,
                    color=self.colors['macd_signal'],
                    width=1.5
                )
                # MACD histogram
                colors_hist = [self.colors['bull'] if val >= 0 else self.colors['bear'] for val in df['MACD_hist']]
                macd_hist_plot = mpf.make_addplot(
                    df['MACD_hist'],
                    panel=2,
                    type='bar',
                    color=colors_hist,
                    alpha=0.4,
                    width=0.7
                )
                apds.extend([macd_plot, macd_signal_plot, macd_hist_plot])

            # Create figure
            panel_ratios = (4, 1, 1) if show_indicators else (1,)

            fig, axes = mpf.plot(
                df,
                type='candle',
                style=self.style,
                volume=True,
                addplot=apds,
                title=f"\n{symbol} - {timeframe} Chart",
                ylabel='Price (USDT)',
                ylabel_lower='Volume',
                figsize=(width, height),
                panel_ratios=panel_ratios,
                returnfig=True,
                warn_too_much_data=len(df) + 1,  # Suppress warning
            )

            # Get main axis (price chart)
            ax_main = axes[0]

            # Draw support levels (with z-order to appear on top) - ULTRA VISIBLE
            for i, level in enumerate(support_levels[:3]):  # Top 3 support levels
                ax_main.axhline(
                    y=level,
                    color='#00FF41',  # Bright neon green
                    linestyle='--',
                    linewidth=4,  # Thicker
                    alpha=1.0,  # Fully opaque
                    zorder=10,
                    label=f'SUPPORT ${level:.2f}' if i == 0 else ''
                )
                # Add price label on the right - BIGGER AND BOLDER
                ax_main.text(
                    len(df) + 2,
                    level,
                    f' SUPPORT: ${level:.2f}',
                    color='#00FF41',
                    fontsize=12,
                    va='center',
                    fontweight='bold',
                    zorder=11,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#00FF41', edgecolor='#00FF41', linewidth=2, alpha=0.4)
                )

            # Draw resistance levels (with z-order to appear on top) - ULTRA VISIBLE
            for i, level in enumerate(resistance_levels[:3]):  # Top 3 resistance levels
                ax_main.axhline(
                    y=level,
                    color='#FF1744',  # Bright neon red
                    linestyle='--',
                    linewidth=4,  # Thicker
                    alpha=1.0,  # Fully opaque
                    zorder=10,
                    label=f'RESISTANCE ${level:.2f}' if i == 0 else ''
                )
                # Add price label on the right - BIGGER AND BOLDER
                ax_main.text(
                    len(df) + 2,
                    level,
                    f' RESISTANCE: ${level:.2f}',
                    color='#FF1744',
                    fontsize=12,
                    va='center',
                    fontweight='bold',
                    zorder=11,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#FF1744', edgecolor='#FF1744', linewidth=2, alpha=0.4)
                )

            # Draw uptrend line (with z-order) - ULTRA VISIBLE
            if trend_lines['uptrend']:
                slope, intercept = trend_lines['uptrend']
                x_range = range(len(df))
                y_values = [slope * x + intercept for x in x_range]
                ax_main.plot(
                    x_range,
                    y_values,
                    color='#00E676',  # Bright neon green
                    linestyle='-',
                    linewidth=5,  # Much thicker
                    alpha=1.0,  # Fully opaque
                    zorder=9,
                    label='📈 UPTREND'
                )

            # Draw downtrend line (with z-order) - ULTRA VISIBLE
            if trend_lines['downtrend']:
                slope, intercept = trend_lines['downtrend']
                x_range = range(len(df))
                y_values = [slope * x + intercept for x in x_range]
                ax_main.plot(
                    x_range,
                    y_values,
                    color='#FF1744',  # Bright neon red
                    linestyle='-',
                    linewidth=5,  # Much thicker
                    alpha=1.0,  # Fully opaque
                    zorder=9,
                    label='📉 DOWNTREND'
                )

            # Add legend
            ax_main.legend(loc='upper left', fontsize=8, framealpha=0.8)

            # Add current price box
            current_price = df['close'].iloc[-1]
            price_change = ((current_price - df['open'].iloc[0]) / df['open'].iloc[0]) * 100
            price_color = self.colors['bull'] if price_change >= 0 else self.colors['bear']

            # Price info box
            price_text = f"${current_price:.4f}  {price_change:+.2f}%"
            ax_main.text(
                0.02, 0.98,
                price_text,
                transform=ax_main.transAxes,
                fontsize=14,
                fontweight='bold',
                color=price_color,
                va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['background'], edgecolor=price_color, linewidth=2, alpha=0.9)
            )

            # Add timestamp
            timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            fig.text(
                0.99, 0.01,
                timestamp_text,
                ha='right',
                va='bottom',
                fontsize=8,
                color=self.colors['text'],
                alpha=0.6
            )

            # Improve overall styling
            fig.patch.set_facecolor(self.colors['background'])
            plt.tight_layout()

            # Save to bytes buffer with high DPI
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format='png',
                dpi=150,  # High resolution
                facecolor=self.colors['background'],
                edgecolor='none',
                bbox_inches='tight'
            )
            buf.seek(0)

            # Clean up
            plt.close(fig)

            logger.info(f"✅ Chart generated successfully for {symbol}")
            return buf.read()

        except Exception as e:
            logger.error(f"❌ Error generating chart for {symbol}: {e}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist


# Singleton instance
_chart_generator: Optional[TradingViewChartGenerator] = None


def get_chart_generator() -> TradingViewChartGenerator:
    """Get or create chart generator instance."""
    global _chart_generator
    if _chart_generator is None:
        _chart_generator = TradingViewChartGenerator()
    return _chart_generator
