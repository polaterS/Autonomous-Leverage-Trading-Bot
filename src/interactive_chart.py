"""
Ultra Premium Interactive HTML Chart Generator
TradingView Pro+ style with full interactivity
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from src.utils import setup_logging
from src.indicators import detect_support_resistance_levels

logger = setup_logging()


# Premium Theme Colors
THEME = {
    'bg_dark': '#0a0e17',
    'bg_chart': '#0f1318',
    'bg_panel': '#141920',
    'grid': 'rgba(42, 46, 57, 0.4)',
    'border': '#2a2e39',
    'text_bright': '#d1d4dc',
    'text_dim': '#787b86',
    'text_muted': '#4a4e59',
    'candle_up': '#089981',
    'candle_down': '#f23645',
    'support': '#00e676',
    'support_bg': 'rgba(0, 230, 118, 0.1)',
    'resistance': '#ff5252',
    'resistance_bg': 'rgba(255, 82, 82, 0.1)',
    'ema_12': '#2962ff',
    'ema_26': '#ff6d00',
    'ema_50': '#ab47bc',
    'vol_up': 'rgba(8, 153, 129, 0.5)',
    'vol_down': 'rgba(242, 54, 69, 0.5)',
}


def detect_trend_lines(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Optional[Tuple[float, float]]]:
    """Detect trend lines by connecting swing highs/lows."""
    try:
        highs = df['high'].values
        lows = df['low'].values

        swing_highs = []
        for i in range(lookback, len(highs) - lookback):
            if highs[i] == max(highs[i-lookback:i+lookback+1]):
                swing_highs.append((i, highs[i]))

        swing_lows = []
        for i in range(lookback, len(lows) - lookback):
            if lows[i] == min(lows[i-lookback:i+lookback+1]):
                swing_lows.append((i, lows[i]))

        uptrend_line = None
        downtrend_line = None

        if len(swing_lows) >= 2:
            recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
            x_coords = [p[0] for p in recent_lows]
            y_coords = [p[1] for p in recent_lows]
            if len(x_coords) >= 2:
                coeffs = np.polyfit(x_coords, y_coords, 1)
                if coeffs[0] > 0:
                    uptrend_line = (coeffs[0], coeffs[1])

        if len(swing_highs) >= 2:
            recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
            x_coords = [p[0] for p in recent_highs]
            y_coords = [p[1] for p in recent_highs]
            if len(x_coords) >= 2:
                coeffs = np.polyfit(x_coords, y_coords, 1)
                if coeffs[0] < 0:
                    downtrend_line = (coeffs[0], coeffs[1])

        return {'uptrend': uptrend_line, 'downtrend': downtrend_line}
    except Exception as e:
        logger.error(f"Trend line detection error: {e}")
        return {'uptrend': None, 'downtrend': None}


async def generate_interactive_html_chart(
    symbol: str,
    ohlcv_data: List[List],
    support_levels: List[float],
    resistance_levels: List[float]
) -> str:
    """
    Generate ultra-premium interactive HTML chart.
    
    Features:
    - TradingView Pro+ dark theme
    - Smooth candlesticks with glow
    - Interactive zoom/pan
    - Crosshair on hover
    - Premium S/R levels
    """
    try:
        logger.info(f"📊 Generating PREMIUM interactive chart: {symbol}")

        # Prepare DataFrame
        df = pd.DataFrame(
            ohlcv_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Istanbul')

        # Calculate EMAs
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        df['EMA50'] = df['close'].ewm(span=50).mean()

        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.8, 0.2]
        )

        # ═══════════════════════════════════════
        # CANDLESTICKS
        # ═══════════════════════════════════════
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing=dict(line=dict(color=THEME['candle_up'], width=1), fillcolor=THEME['candle_up']),
                decreasing=dict(line=dict(color=THEME['candle_down'], width=1), fillcolor=THEME['candle_down']),
                showlegend=False
            ),
            row=1, col=1
        )

        # ═══════════════════════════════════════
        # EMAs (Smooth lines)
        # ═══════════════════════════════════════
        emas = [
            ('EMA12', THEME['ema_12'], 'EMA 12'),
            ('EMA26', THEME['ema_26'], 'EMA 26'),
            ('EMA50', THEME['ema_50'], 'EMA 50'),
        ]
        for col, color, name in emas:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'], y=df[col],
                    name=name,
                    line=dict(color=color, width=1.5, shape='spline'),
                    opacity=0.85,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

        # ═══════════════════════════════════════
        # SUPPORT LEVELS (Premium glow effect)
        # ═══════════════════════════════════════
        for level in support_levels[:3]:
            # Glow effect (wider transparent line)
            fig.add_hline(
                y=level,
                line=dict(color='rgba(0, 230, 118, 0.2)', width=6),
                row=1, col=1
            )
            # Main line
            fig.add_hline(
                y=level,
                line=dict(color=THEME['support'], width=1.5, dash='dot'),
                annotation=dict(
                    text=f"<b>S</b> ${level:,.2f}",
                    font=dict(size=10, color=THEME['support']),
                    bgcolor=THEME['support_bg'],
                    bordercolor=THEME['support'],
                    borderwidth=1,
                    borderpad=3
                ),
                annotation_position="right",
                row=1, col=1
            )

        # ═══════════════════════════════════════
        # RESISTANCE LEVELS (Premium glow effect)
        # ═══════════════════════════════════════
        for level in resistance_levels[:3]:
            # Glow effect
            fig.add_hline(
                y=level,
                line=dict(color='rgba(255, 82, 82, 0.2)', width=6),
                row=1, col=1
            )
            # Main line
            fig.add_hline(
                y=level,
                line=dict(color=THEME['resistance'], width=1.5, dash='dot'),
                annotation=dict(
                    text=f"<b>R</b> ${level:,.2f}",
                    font=dict(size=10, color=THEME['resistance']),
                    bgcolor=THEME['resistance_bg'],
                    bordercolor=THEME['resistance'],
                    borderwidth=1,
                    borderpad=3
                ),
                annotation_position="right",
                row=1, col=1
            )

        # ═══════════════════════════════════════
        # TREND LINES
        # ═══════════════════════════════════════
        trend_lines = detect_trend_lines(df)
        
        if trend_lines['uptrend']:
            slope, intercept = trend_lines['uptrend']
            y_values = [slope * x + intercept for x in range(len(df))]
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'], y=y_values,
                    name='Uptrend',
                    line=dict(color=THEME['candle_up'], width=2),
                    opacity=0.7,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

        if trend_lines['downtrend']:
            slope, intercept = trend_lines['downtrend']
            y_values = [slope * x + intercept for x in range(len(df))]
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'], y=y_values,
                    name='Downtrend',
                    line=dict(color=THEME['candle_down'], width=2),
                    opacity=0.7,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

        # ═══════════════════════════════════════
        # VOLUME BARS
        # ═══════════════════════════════════════
        vol_colors = [
            THEME['vol_up'] if c >= o else THEME['vol_down']
            for c, o in zip(df['close'], df['open'])
        ]
        fig.add_trace(
            go.Bar(
                x=df['timestamp'], y=df['volume'],
                name='Volume',
                marker=dict(color=vol_colors, line=dict(width=0)),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )

        # ═══════════════════════════════════════
        # CURRENT PRICE ANNOTATION
        # ═══════════════════════════════════════
        current_price = df['close'].iloc[-1]
        price_change = (current_price - df['open'].iloc[0]) / df['open'].iloc[0] * 100
        price_color = THEME['candle_up'] if price_change >= 0 else THEME['candle_down']
        
        fig.add_annotation(
            x=0.01, y=0.98,
            xref='paper', yref='paper',
            text=f"<b>${current_price:,.2f}</b>  <span style='color:{price_color}'>{price_change:+.2f}%</span>",
            showarrow=False,
            font=dict(size=18, color=THEME['text_bright'], family='Arial Black'),
            xanchor='left', yanchor='top',
            bgcolor=THEME['bg_panel'],
            bordercolor=price_color,
            borderwidth=2,
            borderpad=8
        )

        # ═══════════════════════════════════════
        # LAYOUT (Ultra Premium)
        # ═══════════════════════════════════════
        fig.update_layout(
            title=dict(
                text=f"<b>{symbol}</b> <span style='color:{THEME['text_dim']}'>15m</span>",
                font=dict(size=20, color=THEME['text_bright'], family='Arial Black'),
                x=0.5, xanchor='center', y=0.98
            ),
            
            paper_bgcolor=THEME['bg_dark'],
            plot_bgcolor=THEME['bg_chart'],
            
            font=dict(family='Arial', color=THEME['text_dim']),
            
            legend=dict(
                orientation='h',
                yanchor='bottom', y=1.01,
                xanchor='center', x=0.5,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=10, color=THEME['text_dim'])
            ),
            
            height=800,
            margin=dict(l=10, r=80, t=60, b=10),
            
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor=THEME['bg_panel'],
                font_size=11,
                font_family='Arial',
                bordercolor=THEME['border']
            ),
            
            dragmode='pan',
            xaxis_rangeslider_visible=True,
            xaxis_rangeslider_thickness=0.04
        )

        # Axis styling
        axis_style = dict(
            gridcolor=THEME['grid'],
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            linecolor=THEME['border'],
            tickfont=dict(color=THEME['text_muted'], size=9)
        )

        fig.update_xaxes(**axis_style, row=1, col=1)
        fig.update_xaxes(**axis_style, showgrid=False, row=2, col=1)
        
        fig.update_yaxes(
            **axis_style,
            side='right',
            tickformat='$,.2f',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor=THEME['text_muted'],
            spikethickness=1,
            row=1, col=1
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            side='right',
            tickfont=dict(color=THEME['text_muted'], size=8),
            row=2, col=1
        )

        # Timestamp watermark
        tz = timezone(timedelta(hours=3))
        now = datetime.now(tz)
        fig.add_annotation(
            x=0.99, y=0.01,
            xref='paper', yref='paper',
            text=f"🕐 {now.strftime('%H:%M:%S')} UTC+3",
            showarrow=False,
            font=dict(size=9, color=THEME['text_muted']),
            xanchor='right', yanchor='bottom',
            opacity=0.7
        )

        # ═══════════════════════════════════════
        # GENERATE HTML
        # ═══════════════════════════════════════
        html_content = fig.to_html(
            include_plotlyjs='cdn',
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'doubleClick': 'reset',
                'responsive': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{symbol.replace("/", "_")}_chart',
                    'height': 1000,
                    'width': 1600,
                    'scale': 2
                }
            }
        )

        # Add custom CSS for premium feel
        custom_css = """
        <style>
            body {
                background: linear-gradient(135deg, #0a0e17 0%, #0f1318 50%, #141920 100%);
                margin: 0;
                padding: 10px;
                font-family: 'Arial', sans-serif;
            }
            .js-plotly-plot {
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            }
        </style>
        """
        html_content = html_content.replace('<head>', f'<head>{custom_css}')

        logger.info(f"✅ Premium interactive chart ready: {symbol}")
        return html_content

    except Exception as e:
        logger.error(f"❌ Interactive chart error: {e}")
        raise
