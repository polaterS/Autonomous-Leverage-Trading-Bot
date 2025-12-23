"""
Premium TradingView-Style Chart Generator
Ultra-minimalist, professional, dark theme charts.
Inspired by TradingView's clean aesthetic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta, timezone
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import setup_logging
from src.indicators import (
    detect_support_resistance_levels,
    detect_market_regime
)

logger = setup_logging()


class PremiumChartGenerator:
    """
    Generates ultra-premium TradingView-style charts with:
    - Clean, minimalist dark theme
    - Gradient backgrounds
    - Professional candlesticks
    - Elegant support/resistance levels
    - Smooth EMA lines
    - Premium typography
    """

    def __init__(self):
        """Initialize with TradingView premium dark theme."""
        # Premium Dark Theme - TradingView inspired
        self.colors = {
            # Background gradient (dark blue to black)
            'bg_primary': '#0d1117',
            'bg_secondary': '#161b22',
            'bg_tertiary': '#1c2128',
            
            # Grid & borders
            'grid': '#21262d',
            'border': '#30363d',
            
            # Text colors
            'text_primary': '#e6edf3',
            'text_secondary': '#8b949e',
            'text_muted': '#484f58',
            
            # Candle colors - TradingView style
            'bull_body': '#26a69a',
            'bull_wick': '#26a69a',
            'bear_body': '#ef5350',
            'bear_wick': '#ef5350',
            
            # Support/Resistance
            'support': '#00c853',
            'resistance': '#ff1744',
            'support_zone': 'rgba(0, 200, 83, 0.1)',
            'resistance_zone': 'rgba(255, 23, 68, 0.1)',
            
            # EMAs - Gradient blues
            'ema_fast': '#2196f3',      # Blue
            'ema_medium': '#ff9800',    # Orange
            'ema_slow': '#9c27b0',      # Purple
            
            # Volume
            'volume_up': 'rgba(38, 166, 154, 0.5)',
            'volume_down': 'rgba(239, 83, 80, 0.5)',
            
            # Price labels
            'price_up': '#26a69a',
            'price_down': '#ef5350',
            
            # Accent
            'accent': '#58a6ff',
            'accent_glow': 'rgba(88, 166, 255, 0.3)',
        }

    def _prepare_dataframe(self, ohlcv_data: List[List]) -> pd.DataFrame:
        """Convert OHLCV data to DataFrame with Turkey timezone."""
        df = pd.DataFrame(
            ohlcv_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Istanbul')
        df.set_index('timestamp', inplace=True)
        
        # Calculate EMAs
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Candle colors
        df['color'] = np.where(df['close'] >= df['open'], 
                               self.colors['bull_body'], 
                               self.colors['bear_body'])
        df['volume_color'] = np.where(df['close'] >= df['open'],
                                       self.colors['volume_up'],
                                       self.colors['volume_down'])
        
        return df

    async def generate_premium_chart(
        self,
        symbol: str,
        ohlcv_data: List[List],
        timeframe: str = '15m',
        width: int = 1400,
        height: int = 900
    ) -> bytes:
        """
        Generate ultra-premium TradingView-style chart.
        
        Returns PNG image as bytes.
        """
        try:
            logger.info(f"🎨 Generating PREMIUM chart for {symbol}...")
            
            df = self._prepare_dataframe(ohlcv_data)
            
            # Detect S/R levels
            current_price = float(df['close'].iloc[-1])
            sr_data = detect_support_resistance_levels(ohlcv_data, current_price)
            support_levels = sr_data.get('swing_lows', [])[:3]
            resistance_levels = sr_data.get('swing_highs', [])[:3]
            
            # Create figure with subplots (main chart + volume)
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.8, 0.2],
                subplot_titles=None
            )
            
            # === CANDLESTICK CHART ===
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    increasing=dict(
                        line=dict(color=self.colors['bull_body'], width=1),
                        fillcolor=self.colors['bull_body']
                    ),
                    decreasing=dict(
                        line=dict(color=self.colors['bear_body'], width=1),
                        fillcolor=self.colors['bear_body']
                    ),
                    name='Price',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # === EMA LINES (Smooth & Elegant) ===
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['EMA12'],
                    mode='lines',
                    name='EMA 12',
                    line=dict(color=self.colors['ema_fast'], width=1.5),
                    opacity=0.9
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['EMA26'],
                    mode='lines',
                    name='EMA 26',
                    line=dict(color=self.colors['ema_medium'], width=1.5),
                    opacity=0.9
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['EMA50'],
                    mode='lines',
                    name='EMA 50',
                    line=dict(color=self.colors['ema_slow'], width=1.5),
                    opacity=0.9
                ),
                row=1, col=1
            )
            
            # === SUPPORT LEVELS (Premium Style) ===
            for level in support_levels:
                # Main line
                fig.add_hline(
                    y=level,
                    line=dict(
                        color=self.colors['support'],
                        width=1.5,
                        dash='dot'
                    ),
                    opacity=0.8,
                    row=1, col=1
                )
                # Price label annotation
                fig.add_annotation(
                    x=df.index[-1],
                    y=level,
                    text=f"  S ${level:,.2f}",
                    showarrow=False,
                    font=dict(
                        size=10,
                        color=self.colors['support'],
                        family='Arial Black'
                    ),
                    xanchor='left',
                    yanchor='middle',
                    bgcolor='rgba(0, 200, 83, 0.15)',
                    bordercolor=self.colors['support'],
                    borderwidth=1,
                    borderpad=3
                )
            
            # === RESISTANCE LEVELS (Premium Style) ===
            for level in resistance_levels:
                # Main line
                fig.add_hline(
                    y=level,
                    line=dict(
                        color=self.colors['resistance'],
                        width=1.5,
                        dash='dot'
                    ),
                    opacity=0.8,
                    row=1, col=1
                )
                # Price label annotation
                fig.add_annotation(
                    x=df.index[-1],
                    y=level,
                    text=f"  R ${level:,.2f}",
                    showarrow=False,
                    font=dict(
                        size=10,
                        color=self.colors['resistance'],
                        family='Arial Black'
                    ),
                    xanchor='left',
                    yanchor='middle',
                    bgcolor='rgba(255, 23, 68, 0.15)',
                    bordercolor=self.colors['resistance'],
                    borderwidth=1,
                    borderpad=3
                )
            
            # === CURRENT PRICE LINE ===
            price_color = self.colors['price_up'] if df['close'].iloc[-1] >= df['open'].iloc[-1] else self.colors['price_down']
            fig.add_hline(
                y=current_price,
                line=dict(color=price_color, width=2, dash='solid'),
                opacity=1,
                row=1, col=1
            )
            
            # Current price annotation (right side)
            price_change = ((current_price - df['open'].iloc[0]) / df['open'].iloc[0]) * 100
            fig.add_annotation(
                x=df.index[-1],
                y=current_price,
                text=f"  ${current_price:,.2f}  {price_change:+.2f}%",
                showarrow=False,
                font=dict(
                    size=12,
                    color='white',
                    family='Arial Black'
                ),
                xanchor='left',
                yanchor='middle',
                bgcolor=price_color,
                borderpad=5
            )
            
            # === VOLUME BARS ===
            colors_volume = [self.colors['volume_up'] if c >= o else self.colors['volume_down'] 
                           for c, o in zip(df['close'], df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    marker=dict(
                        color=colors_volume,
                        line=dict(width=0)
                    ),
                    name='Volume',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # === LAYOUT (Premium Dark Theme) ===
            fig.update_layout(
                # Title
                title=dict(
                    text=f'<b>{symbol}</b> <span style="color:{self.colors["text_secondary"]}">{timeframe}</span>',
                    font=dict(
                        size=24,
                        color=self.colors['text_primary'],
                        family='Arial Black'
                    ),
                    x=0.02,
                    y=0.98
                ),
                
                # Background
                paper_bgcolor=self.colors['bg_primary'],
                plot_bgcolor=self.colors['bg_secondary'],
                
                # Font
                font=dict(
                    family='Arial, sans-serif',
                    color=self.colors['text_secondary']
                ),
                
                # Legend
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(size=11, color=self.colors['text_secondary'])
                ),
                
                # Margins
                margin=dict(l=60, r=120, t=80, b=40),
                
                # Size
                width=width,
                height=height,
                
                # Hover
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor=self.colors['bg_tertiary'],
                    font_size=12,
                    font_family='Arial'
                ),
                
                # Range slider off
                xaxis_rangeslider_visible=False
            )
            
            # === X-AXIS STYLING ===
            fig.update_xaxes(
                gridcolor=self.colors['grid'],
                gridwidth=1,
                showgrid=True,
                zeroline=False,
                linecolor=self.colors['border'],
                tickfont=dict(color=self.colors['text_muted'], size=10),
                row=1, col=1
            )
            fig.update_xaxes(
                gridcolor=self.colors['grid'],
                showgrid=True,
                zeroline=False,
                linecolor=self.colors['border'],
                tickfont=dict(color=self.colors['text_muted'], size=10),
                row=2, col=1
            )
            
            # === Y-AXIS STYLING ===
            fig.update_yaxes(
                gridcolor=self.colors['grid'],
                gridwidth=1,
                showgrid=True,
                zeroline=False,
                linecolor=self.colors['border'],
                tickfont=dict(color=self.colors['text_muted'], size=10),
                side='right',
                tickformat='$,.2f',
                row=1, col=1
            )
            fig.update_yaxes(
                gridcolor=self.colors['grid'],
                showgrid=False,
                zeroline=False,
                linecolor=self.colors['border'],
                tickfont=dict(color=self.colors['text_muted'], size=9),
                side='right',
                row=2, col=1
            )
            
            # === TIMESTAMP WATERMARK ===
            turkey_tz = timezone(timedelta(hours=3))
            turkey_time = datetime.now(turkey_tz)
            
            fig.add_annotation(
                x=0.99,
                y=0.01,
                xref='paper',
                yref='paper',
                text=f'{turkey_time.strftime("%Y-%m-%d %H:%M")} UTC+3',
                showarrow=False,
                font=dict(size=9, color=self.colors['text_muted']),
                xanchor='right',
                yanchor='bottom',
                opacity=0.6
            )
            
            # === EXPORT TO PNG ===
            img_bytes = fig.to_image(
                format='png',
                width=width,
                height=height,
                scale=2  # 2x resolution for crisp display
            )
            
            logger.info(f"✅ Premium chart generated for {symbol}")
            return img_bytes
            
        except Exception as e:
            logger.error(f"❌ Error generating premium chart: {e}")
            raise

    async def generate_position_chart(
        self,
        symbol: str,
        ohlcv_data: List[List],
        entry_price: float,
        stop_loss: float,
        take_profit: Optional[float] = None,
        side: str = 'LONG',
        timeframe: str = '15m',
        width: int = 1400,
        height: int = 900
    ) -> bytes:
        """
        Generate premium chart with position entry, SL, and TP levels.
        Perfect for trade notifications.
        """
        try:
            logger.info(f"🎨 Generating POSITION chart for {symbol}...")
            
            df = self._prepare_dataframe(ohlcv_data)
            current_price = float(df['close'].iloc[-1])
            
            # Create figure
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.8, 0.2]
            )
            
            # === CANDLESTICK ===
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    increasing=dict(
                        line=dict(color=self.colors['bull_body'], width=1),
                        fillcolor=self.colors['bull_body']
                    ),
                    decreasing=dict(
                        line=dict(color=self.colors['bear_body'], width=1),
                        fillcolor=self.colors['bear_body']
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # === EMAs ===
            for ema_col, color, name in [
                ('EMA12', self.colors['ema_fast'], 'EMA 12'),
                ('EMA26', self.colors['ema_medium'], 'EMA 26'),
                ('EMA50', self.colors['ema_slow'], 'EMA 50')
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[ema_col],
                        mode='lines', name=name,
                        line=dict(color=color, width=1.5),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
            
            # === ENTRY PRICE LINE ===
            entry_color = '#2196f3'  # Blue
            fig.add_hline(
                y=entry_price,
                line=dict(color=entry_color, width=2, dash='solid'),
                row=1, col=1
            )
            fig.add_annotation(
                x=df.index[-1], y=entry_price,
                text=f"  ▶ ENTRY ${entry_price:,.2f}",
                showarrow=False,
                font=dict(size=11, color='white', family='Arial Black'),
                xanchor='left', yanchor='middle',
                bgcolor=entry_color,
                borderpad=4
            )
            
            # === STOP LOSS LINE ===
            sl_color = '#ff1744'  # Red
            fig.add_hline(
                y=stop_loss,
                line=dict(color=sl_color, width=2, dash='dash'),
                row=1, col=1
            )
            sl_pct = abs((stop_loss - entry_price) / entry_price * 100)
            fig.add_annotation(
                x=df.index[-1], y=stop_loss,
                text=f"  🛑 SL ${stop_loss:,.2f} (-{sl_pct:.1f}%)",
                showarrow=False,
                font=dict(size=11, color='white', family='Arial Black'),
                xanchor='left', yanchor='middle',
                bgcolor=sl_color,
                borderpad=4
            )
            
            # === TAKE PROFIT LINE (if provided) ===
            if take_profit:
                tp_color = '#00c853'  # Green
                fig.add_hline(
                    y=take_profit,
                    line=dict(color=tp_color, width=2, dash='dash'),
                    row=1, col=1
                )
                tp_pct = abs((take_profit - entry_price) / entry_price * 100)
                fig.add_annotation(
                    x=df.index[-1], y=take_profit,
                    text=f"  🎯 TP ${take_profit:,.2f} (+{tp_pct:.1f}%)",
                    showarrow=False,
                    font=dict(size=11, color='white', family='Arial Black'),
                    xanchor='left', yanchor='middle',
                    bgcolor=tp_color,
                    borderpad=4
                )
            
            # === CURRENT PRICE ===
            price_color = self.colors['price_up'] if current_price >= entry_price else self.colors['price_down']
            pnl_pct = (current_price - entry_price) / entry_price * 100
            if side == 'SHORT':
                pnl_pct = -pnl_pct
            
            fig.add_annotation(
                x=df.index[-1], y=current_price,
                text=f"  NOW ${current_price:,.2f} ({pnl_pct:+.2f}%)",
                showarrow=False,
                font=dict(size=10, color='white', family='Arial'),
                xanchor='left', yanchor='middle',
                bgcolor=price_color,
                borderpad=3,
                opacity=0.9
            )
            
            # === VOLUME ===
            colors_vol = [self.colors['volume_up'] if c >= o else self.colors['volume_down']
                         for c, o in zip(df['close'], df['open'])]
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'],
                       marker=dict(color=colors_vol, line=dict(width=0)),
                       showlegend=False),
                row=2, col=1
            )
            
            # === LAYOUT ===
            side_emoji = "🟢" if side == 'LONG' else "🔴"
            fig.update_layout(
                title=dict(
                    text=f'{side_emoji} <b>{symbol}</b> <span style="color:{self.colors["text_secondary"]}">{side} Position</span>',
                    font=dict(size=22, color=self.colors['text_primary'], family='Arial Black'),
                    x=0.02, y=0.98
                ),
                paper_bgcolor=self.colors['bg_primary'],
                plot_bgcolor=self.colors['bg_secondary'],
                font=dict(family='Arial', color=self.colors['text_secondary']),
                legend=dict(
                    orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(size=10)
                ),
                margin=dict(l=60, r=140, t=80, b=40),
                width=width, height=height,
                hovermode='x unified',
                xaxis_rangeslider_visible=False
            )
            
            # Axis styling
            for row in [1, 2]:
                fig.update_xaxes(
                    gridcolor=self.colors['grid'], showgrid=True,
                    zeroline=False, linecolor=self.colors['border'],
                    tickfont=dict(color=self.colors['text_muted'], size=10),
                    row=row, col=1
                )
            
            fig.update_yaxes(
                gridcolor=self.colors['grid'], showgrid=True,
                zeroline=False, linecolor=self.colors['border'],
                tickfont=dict(color=self.colors['text_muted'], size=10),
                side='right', tickformat='$,.2f',
                row=1, col=1
            )
            fig.update_yaxes(
                showgrid=False, zeroline=False,
                tickfont=dict(color=self.colors['text_muted'], size=9),
                side='right', row=2, col=1
            )
            
            # Timestamp
            turkey_tz = timezone(timedelta(hours=3))
            turkey_time = datetime.now(turkey_tz)
            fig.add_annotation(
                x=0.99, y=0.01, xref='paper', yref='paper',
                text=f'{turkey_time.strftime("%Y-%m-%d %H:%M")} UTC+3',
                showarrow=False,
                font=dict(size=9, color=self.colors['text_muted']),
                xanchor='right', yanchor='bottom', opacity=0.6
            )
            
            # Export
            img_bytes = fig.to_image(format='png', width=width, height=height, scale=2)
            
            logger.info(f"✅ Position chart generated for {symbol}")
            return img_bytes
            
        except Exception as e:
            logger.error(f"❌ Error generating position chart: {e}")
            raise


# Singleton
_premium_generator: Optional[PremiumChartGenerator] = None


def get_premium_chart_generator() -> PremiumChartGenerator:
    """Get or create premium chart generator instance."""
    global _premium_generator
    if _premium_generator is None:
        _premium_generator = PremiumChartGenerator()
    return _premium_generator
