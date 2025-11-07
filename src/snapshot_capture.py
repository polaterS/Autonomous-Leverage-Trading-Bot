"""
Market Snapshot System - Captures comprehensive market data for ML learning

This module captures all technical indicators, market conditions, and order book data
at the moment of trade entry/exit. This data is stored in entry_snapshot and exit_snapshot
JSONB columns for advanced ML pattern learning.
"""

import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


async def capture_market_snapshot(
    exchange_client,
    symbol: str,
    current_price: Decimal,
    indicators: Dict[str, Any],
    side: Optional[str] = None,
    snapshot_type: str = "entry"
) -> Dict[str, Any]:
    """
    Capture comprehensive market snapshot for ML learning.

    Args:
        exchange_client: Exchange client for fetching additional data
        symbol: Trading symbol (e.g., 'BTC/USDT')
        current_price: Current market price
        indicators: Technical indicators dict from market analysis
        side: Trade side ('LONG' or 'SHORT'), if applicable
        snapshot_type: 'entry' or 'exit'

    Returns:
        Complete market snapshot as JSONB-ready dict
    """
    try:
        logger.info(f"📸 Capturing {snapshot_type} snapshot for {symbol}...")

        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'snapshot_type': snapshot_type,
            'symbol': symbol,
            'side': side,

            # 💰 PRICE DATA
            'price': {
                'current': float(current_price),
                'change_24h_percent': indicators.get('price_change_24h_percent', 0.0)
            },

            # 📊 TECHNICAL INDICATORS
            'indicators': {
                # RSI (Relative Strength Index)
                'rsi': indicators.get('rsi', 50.0),
                'rsi_category': _categorize_rsi(indicators.get('rsi', 50.0)),

                # MACD (Moving Average Convergence Divergence)
                'macd': indicators.get('macd', 0.0),
                'macd_signal': indicators.get('macd_signal', 0.0),
                'macd_histogram': indicators.get('macd_histogram', 0.0),
                'macd_trend': 'bullish' if indicators.get('macd', 0) > indicators.get('macd_signal', 0) else 'bearish',

                # Bollinger Bands
                'bb_upper': indicators.get('bb_upper', 0.0),
                'bb_middle': indicators.get('bb_middle', 0.0),
                'bb_lower': indicators.get('bb_lower', 0.0),
                'bb_position': _calculate_bb_position(current_price, indicators),
                'bb_width_percent': _calculate_bb_width(indicators),

                # Moving Averages
                'sma_20': indicators.get('sma_20', 0.0),
                'sma_50': indicators.get('sma_50', 0.0),
                'ema_12': indicators.get('ema_12', 0.0),
                'ema_26': indicators.get('ema_26', 0.0),
                'price_to_sma20_percent': _calculate_price_distance(current_price, indicators.get('sma_20', current_price)),
                'price_to_sma50_percent': _calculate_price_distance(current_price, indicators.get('sma_50', current_price)),

                # Stochastic Oscillator
                'stoch_k': indicators.get('stoch_k', 50.0),
                'stoch_d': indicators.get('stoch_d', 50.0),
                'stoch_category': _categorize_stochastic(indicators.get('stoch_k', 50.0)),

                # ADX (Average Directional Index)
                'adx': indicators.get('adx', 25.0),
                'adx_trend_strength': _categorize_adx(indicators.get('adx', 25.0)),

                # ATR (Average True Range)
                'atr': indicators.get('atr', 0.0),
                'atr_percent': indicators.get('atr_percent', 0.0)
            },

            # 📈 VOLUME DATA
            'volume': {
                'current': indicators.get('volume', 0.0),
                'avg_24h': indicators.get('volume_24h_avg', 0.0),
                'volume_ratio': _calculate_volume_ratio(indicators),
                'volume_category': _categorize_volume(indicators)
            },

            # 🎯 SUPPORT & RESISTANCE LEVELS
            'levels': {
                'support_levels': indicators.get('support_levels', []),
                'resistance_levels': indicators.get('resistance_levels', []),
                'nearest_support': _find_nearest_level(current_price, indicators.get('support_levels', []), 'below'),
                'nearest_resistance': _find_nearest_level(current_price, indicators.get('resistance_levels', []), 'above'),
                'support_distance_percent': _calculate_level_distance(current_price, indicators.get('support_levels', []), 'below'),
                'resistance_distance_percent': _calculate_level_distance(current_price, indicators.get('resistance_levels', []), 'above')
            },

            # 📐 TREND ANALYSIS
            'trend': {
                'short_term': indicators.get('trend_short', 'neutral'),  # Based on EMA12/26
                'medium_term': indicators.get('trend_medium', 'neutral'),  # Based on SMA20/50
                'long_term': indicators.get('trend_long', 'neutral'),  # Based on price action
                'overall_direction': indicators.get('trend_direction', 'neutral'),
                'trend_strength': indicators.get('trend_strength', 'weak')
            }
        }

        # 📚 ORDER BOOK DATA (DISABLED to reduce API calls)
        # Each snapshot fetching order book = 1 API call
        # With 10 positions, this is 10+ extra calls per minute
        # Binance limit: 6000/min weight, order book = 5-10 weight
        snapshot['order_book'] = None

        # 💸 FUNDING RATE (DISABLED to reduce API calls)
        # Funding rate rarely changes (8h intervals)
        # Not critical for ML learning
        snapshot['funding'] = None

        # 🔥 LIQUIDATION DATA (DISABLED - not implemented)
        snapshot['liquidations'] = None

        # 📊 MARKET REGIME
        snapshot['market_regime'] = _detect_market_regime(indicators, snapshot)

        logger.info(f"✅ Snapshot captured: RSI={snapshot['indicators']['rsi']:.1f}, "
                   f"Volume ratio={snapshot['volume']['volume_ratio']:.2f}x, "
                   f"Trend={snapshot['trend']['overall_direction']}")

        return snapshot

    except Exception as e:
        logger.error(f"❌ Failed to capture market snapshot for {symbol}: {e}", exc_info=True)
        # Return minimal snapshot on error
        return {
            'timestamp': datetime.now().isoformat(),
            'snapshot_type': snapshot_type,
            'symbol': symbol,
            'side': side,
            'error': str(e),
            'price': {'current': float(current_price)},
            'indicators': {},
            'volume': {},
            'levels': {},
            'trend': {}
        }


# ============================================================================
# HELPER FUNCTIONS - Categorization & Calculation
# ============================================================================

def _categorize_rsi(rsi: float) -> str:
    """Categorize RSI value"""
    if rsi <= 20:
        return 'extremely_oversold'
    elif rsi <= 30:
        return 'oversold'
    elif rsi <= 40:
        return 'weak'
    elif rsi <= 60:
        return 'neutral'
    elif rsi <= 70:
        return 'strong'
    elif rsi <= 80:
        return 'overbought'
    else:
        return 'extremely_overbought'


def _categorize_stochastic(stoch_k: float) -> str:
    """Categorize Stochastic value"""
    if stoch_k <= 20:
        return 'oversold'
    elif stoch_k <= 40:
        return 'weak'
    elif stoch_k <= 60:
        return 'neutral'
    elif stoch_k <= 80:
        return 'strong'
    else:
        return 'overbought'


def _categorize_adx(adx: float) -> str:
    """Categorize ADX trend strength"""
    if adx < 20:
        return 'weak_or_ranging'
    elif adx < 25:
        return 'developing'
    elif adx < 50:
        return 'strong'
    else:
        return 'very_strong'


def _categorize_volume(indicators: Dict) -> str:
    """Categorize volume relative to average"""
    ratio = _calculate_volume_ratio(indicators)

    if ratio >= 3.0:
        return 'extremely_high'
    elif ratio >= 2.0:
        return 'very_high'
    elif ratio >= 1.5:
        return 'high'
    elif ratio >= 0.8:
        return 'normal'
    elif ratio >= 0.5:
        return 'low'
    else:
        return 'very_low'


def _calculate_volume_ratio(indicators: Dict) -> float:
    """Calculate current volume / 24h average"""
    current_vol = indicators.get('volume', 0.0)
    avg_vol = indicators.get('volume_24h_avg', 1.0)

    if avg_vol == 0:
        return 1.0

    return current_vol / avg_vol


def _calculate_bb_position(price: Decimal, indicators: Dict) -> str:
    """Calculate price position within Bollinger Bands"""
    bb_upper = Decimal(str(indicators.get('bb_upper', 0)))
    bb_middle = Decimal(str(indicators.get('bb_middle', 0)))
    bb_lower = Decimal(str(indicators.get('bb_lower', 0)))

    if bb_upper == 0 or bb_lower == 0:
        return 'unknown'

    if price >= bb_upper:
        return 'above_upper'
    elif price >= bb_middle:
        return 'upper_half'
    elif price >= bb_lower:
        return 'lower_half'
    else:
        return 'below_lower'


def _calculate_bb_width(indicators: Dict) -> float:
    """Calculate Bollinger Band width as percentage"""
    bb_upper = indicators.get('bb_upper', 0)
    bb_lower = indicators.get('bb_lower', 0)
    bb_middle = indicators.get('bb_middle', 1)

    if bb_middle == 0:
        return 0.0

    width = ((bb_upper - bb_lower) / bb_middle) * 100
    return round(width, 2)


def _calculate_price_distance(price: Decimal, ma_value: float) -> float:
    """Calculate percentage distance between price and moving average"""
    if ma_value == 0:
        return 0.0

    ma_decimal = Decimal(str(ma_value))
    distance = ((price - ma_decimal) / ma_decimal) * 100
    return round(float(distance), 2)


def _find_nearest_level(price: Decimal, levels: List[float], direction: str) -> Optional[float]:
    """Find nearest support/resistance level"""
    if not levels:
        return None

    price_float = float(price)

    if direction == 'below':
        # Find nearest support (below current price)
        below_levels = [l for l in levels if l < price_float]
        return max(below_levels) if below_levels else None
    else:  # 'above'
        # Find nearest resistance (above current price)
        above_levels = [l for l in levels if l > price_float]
        return min(above_levels) if above_levels else None


def _calculate_level_distance(price: Decimal, levels: List[float], direction: str) -> Optional[float]:
    """Calculate percentage distance to nearest support/resistance"""
    nearest = _find_nearest_level(price, levels, direction)

    if nearest is None:
        return None

    distance = ((float(price) - nearest) / nearest) * 100
    return round(abs(distance), 2)


def _detect_market_regime(indicators: Dict, snapshot: Dict) -> Dict[str, str]:
    """Detect current market regime from indicators"""
    rsi = indicators.get('rsi', 50.0)
    adx = indicators.get('adx', 25.0)
    volume_ratio = _calculate_volume_ratio(indicators)

    # Volatility
    if snapshot['indicators']['bb_width_percent'] > 10:
        volatility = 'high'
    elif snapshot['indicators']['bb_width_percent'] > 5:
        volatility = 'normal'
    else:
        volatility = 'low'

    # Trend vs Range
    if adx >= 25:
        market_type = 'trending'
    else:
        market_type = 'ranging'

    # Direction
    if rsi > 60 and snapshot['trend']['overall_direction'] == 'bullish':
        direction = 'bullish'
    elif rsi < 40 and snapshot['trend']['overall_direction'] == 'bearish':
        direction = 'bearish'
    else:
        direction = 'neutral'

    return {
        'volatility': volatility,
        'market_type': market_type,
        'direction': direction,
        'combined': f"{volatility}_{market_type}_{direction}"
    }


# ============================================================================
# EXCHANGE DATA FETCHING
# ============================================================================

async def _fetch_order_book_data(exchange_client, symbol: str) -> Dict[str, Any]:
    """Fetch order book depth and calculate imbalance"""
    try:
        # Fetch order book (top 20 levels)
        order_book = await exchange_client.fetch_order_book(symbol, limit=20)

        # Calculate bid/ask volumes
        bid_volume = sum(bid[1] for bid in order_book['bids'][:10])
        ask_volume = sum(ask[1] for ask in order_book['asks'][:10])

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            imbalance = 0.0
        else:
            imbalance = (bid_volume - ask_volume) / total_volume

        # Calculate spread
        best_bid = order_book['bids'][0][0] if order_book['bids'] else 0
        best_ask = order_book['asks'][0][0] if order_book['asks'] else 0
        spread_percent = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0

        return {
            'bid_volume': round(bid_volume, 4),
            'ask_volume': round(ask_volume, 4),
            'imbalance': round(imbalance, 4),  # -1 (all asks) to +1 (all bids)
            'spread_percent': round(spread_percent, 4),
            'best_bid': best_bid,
            'best_ask': best_ask
        }

    except Exception as e:
        logger.debug(f"Order book fetch failed: {e}")
        return None


async def _fetch_funding_rate(exchange_client, symbol: str) -> Dict[str, Any]:
    """Fetch futures funding rate"""
    try:
        # Only for futures contracts
        if '/USDT' not in symbol:
            return None

        # Fetch funding rate from exchange
        funding_info = await exchange_client.fetch_funding_rate(symbol)

        if not funding_info:
            return None

        funding_rate = funding_info.get('fundingRate', 0.0)

        # Categorize funding rate
        if funding_rate > 0.001:  # > 0.1%
            category = 'very_bullish'
        elif funding_rate > 0.0005:  # > 0.05%
            category = 'bullish'
        elif funding_rate > -0.0005:
            category = 'neutral'
        elif funding_rate > -0.001:
            category = 'bearish'
        else:
            category = 'very_bearish'

        return {
            'rate': round(funding_rate * 100, 4),  # Convert to percentage
            'rate_8h': round(funding_rate * 3 * 100, 4),  # 8-hour rate
            'category': category,
            'next_funding_time': funding_info.get('fundingTimestamp')
        }

    except Exception as e:
        logger.debug(f"Funding rate fetch failed: {e}")
        return None


async def _fetch_liquidation_clusters(exchange_client, symbol: str, current_price: Decimal) -> Dict[str, Any]:
    """
    Fetch liquidation heatmap data (if exchange supports it).

    NOTE: Most exchanges don't provide direct liquidation APIs.
    This is a placeholder for future integration with services like:
    - Coinglass API
    - Binance liquidation streams
    - Custom liquidation tracking
    """
    try:
        # TODO: Integrate with liquidation data provider
        # For now, return None (not implemented)
        return None

    except Exception as e:
        logger.debug(f"Liquidation data fetch failed: {e}")
        return None
