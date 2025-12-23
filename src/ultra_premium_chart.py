"""
Ultra Premium Chart Generator
TradingView Pro+ style with gradient backgrounds, glow effects, and premium aesthetics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import setup_logging
from src.indicators import detect_support_resistance_levels

logger = setup_logging()


class UltraPremiumChart:
    """
    Ultra-premium TradingView Pro+ style charts.
    Features:
    - Gradient dark backgrounds
    - Glow effects on key levels
    - Premium typography
    - Smooth animations-ready design
    - Professional color palette
    """

    def __init__(self):
        # Ultra Premium Color Palette
        self.theme = {
            # Backgrounds (gradient effect simulated)
            'bg_dark': '#0a0e17',
            'bg_chart': '#0f1318',
            'bg_panel': '#141920',
            
            # Borders & Grid
            'grid': 'rgba(42, 46, 57, 0.5)',
            'border': '#2a2e39',
            
            # Text
            'text_bright': '#d1d4dc',
            'text_dim': '#787b86',
            'text_muted': '#4a4e59',
            
            # Candles - TradingView exact colors
            'candle_up': '#089981',
            'candle_down': '#f23645',
            'candle_up_wick': '#089981',
            'candle_down_wick': '#f23645',
            
            # Support/Resistance with glow
            'support': '#00e676',
            'support_glow': 'rgba(0, 230, 118, 0.2)',
            'resistance': '#ff5252',
            'resistance_glow': 'rgba(255, 82, 82, 0.2)',
            
            # EMAs - Premium gradient
            'ema_12': '#2962ff',
            'ema_26': '#ff6d00',
            'ema_50': '#ab47bc',
            
            # Volume
            'vol_up': 'rgba(8, 153, 129, 0.5)',
            'vol_down': 'rgba(242, 54, 69, 0.5)',
            
            # Special
            'entry': '#2196f3',
            'stop_loss': '#f44336',
            'take_profit': '#4caf50',
            'current': '#ffeb3b',
        }

    def _create_df(self, ohlcv: List[List]) -> pd.DataFrame:
        """Prepare DataFrame with indicators."""
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df['ts'] = df['ts'].dt.tz_convert('Europe/Istanbul')
        df.set_index('ts', inplace=True)
        
        # EMAs
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        
        return df

    async def generate(
        self,
        symbol: str,
        ohlcv: List[List],
        timeframe: str = '15m',
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        side: Optional[str] = None,
        width: int = 1600,
        height: int = 1000
    ) -> bytes:
        """
        Generate ultra-premium chart.
        
        Args:
            symbol: Trading pair
            ohlcv: OHLCV data
            timeframe: Chart timeframe
            entry_price: Position entry (optional)
            stop_loss: Stop loss level (optional)
            take_profit: Take profit level (optional)
            side: LONG or SHORT (optional)
            width: Image width
            height: Image height
            
        Returns:
            PNG bytes
        """
        try:
            logger.info(f"🎨 Generating ULTRA PREMIUM chart: {symbol}")
            
            df = self._create_df(ohlcv)
            current = float(df['close'].iloc[-1])
            
            # S/R levels
            sr = detect_support_resistance_levels(ohlcv, current)
            supports = sr.get('swing_lows', [])[:2]
            resistances = sr.get('swing_highs', [])[:2]
            
            # Create figure
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.82, 0.18]
            )
            
            # ═══════════════════════════════════════
            # CANDLESTICKS
            # ═══════════════════════════════════════
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                increasing=dict(
                    line=dict(color=self.theme['candle_up'], width=1),
                    fillcolor=self.theme['candle_up']
                ),
                decreasing=dict(
                    line=dict(color=self.theme['candle_down'], width=1),
                    fillcolor=self.theme['candle_down']
                ),
                showlegend=False,
                name='Price'
            ), row=1, col=1)
            
            # ═══════════════════════════════════════
            # EMAs (Smooth lines)
            # ═══════════════════════════════════════
            emas = [
                ('ema12', self.theme['ema_12'], 'EMA 12', 1.2),
                ('ema26', self.theme['ema_26'], 'EMA 26', 1.2),
                ('ema50', self.theme['ema_50'], 'EMA 50', 1.5),
            ]
            for col, color, name, w in emas:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=w, shape='spline'),
                    opacity=0.85
                ), row=1, col=1)
            
            # ═══════════════════════════════════════
            # SUPPORT LEVELS (with glow effect)
            # ═══════════════════════════════════════
            for lvl in supports:
                # Glow (wider, transparent)
                fig.add_hline(
                    y=lvl,
                    line=dict(color=self.theme['support_glow'], width=8),
                    row=1, col=1
                )
                # Main line
                fig.add_hline(
                    y=lvl,
                    line=dict(color=self.theme['support'], width=1.5, dash='dot'),
                    row=1, col=1
                )
                # Label
                fig.add_annotation(
                    x=df.index[-1], y=lvl,
                    text=f"<b>S</b> ${lvl:,.2f}",
                    showarrow=False,
                    font=dict(size=10, color=self.theme['support'], family='Arial'),
                    xanchor='left',
                    bgcolor='rgba(0, 230, 118, 0.1)',
                    bordercolor=self.theme['support'],
                    borderwidth=1,
                    borderpad=3
                )
            
            # ═══════════════════════════════════════
            # RESISTANCE LEVELS (with glow effect)
            # ═══════════════════════════════════════
            for lvl in resistances:
                # Glow
                fig.add_hline(
                    y=lvl,
                    line=dict(color=self.theme['resistance_glow'], width=8),
                    row=1, col=1
                )
                # Main line
                fig.add_hline(
                    y=lvl,
                    line=dict(color=self.theme['resistance'], width=1.5, dash='dot'),
                    row=1, col=1
                )
                # Label
                fig.add_annotation(
                    x=df.index[-1], y=lvl,
                    text=f"<b>R</b> ${lvl:,.2f}",
                    showarrow=False,
                    font=dict(size=10, color=self.theme['resistance'], family='Arial'),
                    xanchor='left',
                    bgcolor='rgba(255, 82, 82, 0.1)',
                    bordercolor=self.theme['resistance'],
                    borderwidth=1,
                    borderpad=3
                )
            
            # ═══════════════════════════════════════
            # POSITION LEVELS (if trading)
            # ═══════════════════════════════════════
            if entry_price:
                # Entry line
                fig.add_hline(
                    y=entry_price,
                    line=dict(color=self.theme['entry'], width=2),
                    row=1, col=1
                )
                fig.add_annotation(
                    x=df.index[-1], y=entry_price,
                    text=f"<b>▶ ENTRY</b> ${entry_price:,.2f}",
                    showarrow=False,
                    font=dict(size=11, color='white', family='Arial Black'),
                    xanchor='left',
                    bgcolor=self.theme['entry'],
                    borderpad=5
                )
            
            if stop_loss:
                sl_pct = abs((stop_loss - (entry_price or current)) / (entry_price or current) * 100)
                fig.add_hline(
                    y=stop_loss,
                    line=dict(color=self.theme['stop_loss'], width=2, dash='dash'),
                    row=1, col=1
                )
                fig.add_annotation(
                    x=df.index[-1], y=stop_loss,
                    text=f"<b>🛑 SL</b> ${stop_loss:,.2f} <span style='font-size:9px'>(-{sl_pct:.1f}%)</span>",
                    showarrow=False,
                    font=dict(size=11, color='white', family='Arial Black'),
                    xanchor='left',
                    bgcolor=self.theme['stop_loss'],
                    borderpad=5
                )
            
            if take_profit:
                tp_pct = abs((take_profit - (entry_price or current)) / (entry_price or current) * 100)
                fig.add_hline(
                    y=take_profit,
                    line=dict(color=self.theme['take_profit'], width=2, dash='dash'),
                    row=1, col=1
                )
                fig.add_annotation(
                    x=df.index[-1], y=take_profit,
                    text=f"<b>🎯 TP</b> ${take_profit:,.2f} <span style='font-size:9px'>(+{tp_pct:.1f}%)</span>",
                    showarrow=False,
                    font=dict(size=11, color='white', family='Arial Black'),
                    xanchor='left',
                    bgcolor=self.theme['take_profit'],
                    borderpad=5
                )
            
            # ═══════════════════════════════════════
            # CURRENT PRICE (Highlighted)
            # ═══════════════════════════════════════
            price_change = (current - df['open'].iloc[0]) / df['open'].iloc[0] * 100
            is_up = price_change >= 0
            price_color = self.theme['candle_up'] if is_up else self.theme['candle_down']
            
            # Current price box (premium style)
            fig.add_annotation(
                x=0.01, y=0.97,
                xref='paper', yref='paper',
                text=f"<b>${current:,.2f}</b>  <span style='color:{price_color}'>{price_change:+.2f}%</span>",
                showarrow=False,
                font=dict(size=20, color=self.theme['text_bright'], family='Arial Black'),
                xanchor='left', yanchor='top',
                bgcolor=self.theme['bg_panel'],
                bordercolor=price_color,
                borderwidth=2,
                borderpad=8
            )
            
            # ═══════════════════════════════════════
            # VOLUME BARS
            # ═══════════════════════════════════════
            vol_colors = [
                self.theme['vol_up'] if c >= o else self.theme['vol_down']
                for c, o in zip(df['close'], df['open'])
            ]
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['volume'],
                marker=dict(color=vol_colors, line=dict(width=0)),
                showlegend=False,
                name='Volume'
            ), row=2, col=1)
            
            # ═══════════════════════════════════════
            # LAYOUT (Ultra Premium)
            # ═══════════════════════════════════════
            title_text = f"<b>{symbol}</b>"
            if side:
                side_color = self.theme['candle_up'] if side == 'LONG' else self.theme['candle_down']
                side_emoji = "🟢" if side == 'LONG' else "🔴"
                title_text = f"{side_emoji} <b>{symbol}</b> <span style='color:{side_color}'>{side}</span>"
            
            fig.update_layout(
                title=dict(
                    text=title_text + f" <span style='color:{self.theme['text_dim']};font-size:14px'>{timeframe}</span>",
                    font=dict(size=26, color=self.theme['text_bright'], family='Arial Black'),
                    x=0.5, y=0.98,
                    xanchor='center'
                ),
                
                # Premium dark background
                paper_bgcolor=self.theme['bg_dark'],
                plot_bgcolor=self.theme['bg_chart'],
                
                font=dict(family='Arial', color=self.theme['text_dim']),
                
                # Legend
                legend=dict(
                    orientation='h',
                    yanchor='bottom', y=1.01,
                    xanchor='center', x=0.5,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(size=10, color=self.theme['text_dim'])
                ),
                
                margin=dict(l=10, r=130, t=70, b=30),
                width=width,
                height=height,
                
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor=self.theme['bg_panel'],
                    font_size=11,
                    font_family='Arial',
                    bordercolor=self.theme['border']
                ),
                
                xaxis_rangeslider_visible=False
            )
            
            # Axis styling
            axis_style = dict(
                gridcolor=self.theme['grid'],
                gridwidth=1,
                zeroline=False,
                linecolor=self.theme['border'],
                tickfont=dict(color=self.theme['text_muted'], size=9)
            )
            
            fig.update_xaxes(**axis_style, showgrid=True, row=1, col=1)
            fig.update_xaxes(**axis_style, showgrid=False, row=2, col=1)
            
            fig.update_yaxes(
                **axis_style,
                showgrid=True,
                side='right',
                tickformat='$,.2f',
                row=1, col=1
            )
            fig.update_yaxes(
                showgrid=False,
                zeroline=False,
                side='right',
                tickfont=dict(color=self.theme['text_muted'], size=8),
                row=2, col=1
            )
            
            # Timestamp watermark
            tz = timezone(timedelta(hours=3))
            now = datetime.now(tz)
            fig.add_annotation(
                x=0.99, y=0.005,
                xref='paper', yref='paper',
                text=f"🕐 {now.strftime('%H:%M:%S')} UTC+3",
                showarrow=False,
                font=dict(size=9, color=self.theme['text_muted']),
                xanchor='right', yanchor='bottom',
                opacity=0.7
            )
            
            # Export high-res PNG
            img = fig.to_image(format='png', width=width, height=height, scale=2)
            
            logger.info(f"✅ Ultra premium chart ready: {symbol}")
            return img
            
        except Exception as e:
            logger.error(f"❌ Chart generation failed: {e}")
            raise


# Singleton
_ultra_chart: Optional[UltraPremiumChart] = None

def get_ultra_premium_chart() -> UltraPremiumChart:
    global _ultra_chart
    if _ultra_chart is None:
        _ultra_chart = UltraPremiumChart()
    return _ultra_chart
