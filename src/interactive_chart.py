"""
Interactive HTML Chart Generator using Plotly
Creates browser-viewable interactive charts with zoom, pan, hover tooltips
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from src.utils import setup_logging
from src.indicators import detect_support_resistance_levels

logger = setup_logging()


async def generate_interactive_html_chart(
    symbol: str,
    ohlcv_data: List[List],
    support_levels: List[float],
    resistance_levels: List[float]
) -> str:
    """
    Generate interactive HTML chart using Plotly.

    Args:
        symbol: Trading symbol
        ohlcv_data: OHLCV data
        support_levels: Support price levels
        resistance_levels: Resistance price levels

    Returns:
        HTML string of interactive chart
    """
    try:
        logger.info(f"📊 Generating interactive HTML chart for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Calculate indicators
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Create subplots (4 rows: price, volume, RSI, MACD)
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} - 15m Chart', 'Volume', 'RSI', 'MACD'),
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#26A69A',
                decreasing_line_color='#EF5350'
            ),
            row=1, col=1
        )

        # EMAs
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['EMA12'],
                name='EMA 12',
                line=dict(color='#2962FF', width=2)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['EMA26'],
                name='EMA 26',
                line=dict(color='#FF6D00', width=2)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['EMA50'],
                name='EMA 50',
                line=dict(color='#FFD600', width=2)
            ),
            row=1, col=1
        )

        # Support levels
        for level in support_levels[:3]:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="#4CAF50",
                line_width=2,
                annotation_text=f"S: ${level:.2f}",
                annotation_position="right",
                row=1, col=1
            )

        # Resistance levels
        for level in resistance_levels[:3]:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="#F44336",
                line_width=2,
                annotation_text=f"R: ${level:.2f}",
                annotation_position="right",
                row=1, col=1
            )

        # Volume
        colors = ['#26A69A' if row['close'] >= row['open'] else '#EF5350' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df['timestamp'], y=df['volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )

        # RSI
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['RSI'],
                name='RSI',
                line=dict(color='#9C27B0', width=2)
            ),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dot", line_color="gray", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="gray", row=3, col=1)

        # MACD
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['MACD'],
                name='MACD',
                line=dict(color='#2196F3', width=2)
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['MACD_signal'],
                name='Signal',
                line=dict(color='#FF9800', width=2)
            ),
            row=4, col=1
        )
        # MACD histogram
        hist_colors = ['#26A69A' if val >= 0 else '#EF5350' for val in df['MACD_hist']]
        fig.add_trace(
            go.Bar(
                x=df['timestamp'], y=df['MACD_hist'],
                name='Histogram',
                marker_color=hist_colors
            ),
            row=4, col=1
        )

        # Update layout - TradingView dark theme
        fig.update_layout(
            template='plotly_dark',
            title=f"{symbol} - Interactive Chart",
            xaxis_rangeslider_visible=False,
            height=1000,
            hovermode='x unified',
            font=dict(family="Arial", size=12, color="#D1D4DC"),
            paper_bgcolor='#131722',
            plot_bgcolor='#131722',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#1E222D')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#1E222D')

        # Generate HTML
        html_content = fig.to_html(
            include_plotlyjs='cdn',
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
            }
        )

        logger.info(f"✅ Interactive HTML chart generated for {symbol}")
        return html_content

    except Exception as e:
        logger.error(f"❌ Error generating interactive chart: {e}")
        raise
