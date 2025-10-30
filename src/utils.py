"""
Utility functions and logging setup for the trading bot.
"""

import logging
import colorlog
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any
import json
import time


class TurkeyTimeFormatter(colorlog.ColoredFormatter):
    """Custom formatter that uses Turkey Time (UTC+3)."""

    def formatTime(self, record, datefmt=None):
        """Override formatTime to use Turkey Time (UTC+3)."""
        # Convert timestamp to Turkey Time (UTC+3)
        dt = datetime.fromtimestamp(record.created)
        turkey_offset = timedelta(hours=3)
        turkey_time = dt + turkey_offset

        if datefmt:
            return turkey_time.strftime(datefmt)
        else:
            return turkey_time.strftime('%Y-%m-%d %H:%M:%S')


def setup_logging(debug: bool = False) -> logging.Logger:
    """Set up colored logging with appropriate level."""

    log_level = logging.DEBUG if debug else logging.INFO

    # Create color formatter with Turkey Time
    formatter = TurkeyTimeFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    # Set up handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Create logger
    logger = logging.getLogger('trading_bot')
    logger.setLevel(log_level)
    logger.addHandler(handler)

    # Suppress noisy libraries
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    return logger


def calculate_liquidation_price(
    entry_price: Decimal,
    leverage: int,
    side: str,
    maintenance_margin_rate: Decimal = Decimal("0.004")
) -> Decimal:
    """
    Calculate liquidation price for a leveraged position.

    Args:
        entry_price: Entry price of the position
        leverage: Leverage multiplier
        side: 'LONG' or 'SHORT'
        maintenance_margin_rate: Exchange maintenance margin rate (default 0.4%)

    Returns:
        Liquidation price
    """
    # Simplified liquidation calculation (Binance-like)
    # Actual formula: Liq Price = Entry Price × (1 ± (1 - Maintenance Margin Rate) / Leverage)

    leverage_decimal = Decimal(str(leverage))

    if side == 'LONG':
        # For longs, liquidation is below entry
        liq_price = entry_price * (1 - (1 - maintenance_margin_rate) / leverage_decimal)
    else:  # SHORT
        # For shorts, liquidation is above entry
        liq_price = entry_price * (1 + (1 - maintenance_margin_rate) / leverage_decimal)

    return liq_price


def calculate_position_size(
    capital: Decimal,
    position_size_percent: Decimal,
    entry_price: Decimal,
    leverage: int
) -> tuple[Decimal, Decimal]:
    """
    Calculate position quantity and value.

    Returns:
        (quantity, position_value_usd)
    """
    # Calculate position value (before leverage)
    position_value = capital * position_size_percent

    # Calculate quantity
    leverage_decimal = Decimal(str(leverage))
    quantity = (position_value * leverage_decimal) / entry_price

    return quantity, position_value


def calculate_stop_loss_price(
    entry_price: Decimal,
    stop_loss_percent: Decimal,
    side: str
) -> Decimal:
    """Calculate stop-loss price."""
    sl_decimal = stop_loss_percent / 100

    if side == 'LONG':
        # For longs, stop-loss is below entry
        sl_price = entry_price * (1 - sl_decimal)
    else:  # SHORT
        # For shorts, stop-loss is above entry
        sl_price = entry_price * (1 + sl_decimal)

    return sl_price


def calculate_min_profit_price(
    entry_price: Decimal,
    min_profit_usd: Decimal,
    position_value: Decimal,
    leverage: int,
    side: str
) -> Decimal:
    """Calculate the price needed to achieve minimum profit target."""

    # Calculate required price change percentage
    # profit = position_value * price_change_pct * leverage
    # price_change_pct = profit / (position_value * leverage)
    leverage_decimal = Decimal(str(leverage))
    required_price_change_pct = min_profit_usd / (position_value * leverage_decimal)

    if side == 'LONG':
        # For longs, profit comes from price increase
        target_price = entry_price * (1 + required_price_change_pct)
    else:  # SHORT
        # For shorts, profit comes from price decrease
        target_price = entry_price * (1 - required_price_change_pct)

    return target_price


def calculate_pnl(
    entry_price: Decimal,
    current_price: Decimal,
    quantity: Decimal,
    side: str,
    leverage: int,
    position_value: Decimal
) -> Dict[str, Any]:
    """
    Calculate comprehensive P&L metrics.

    Returns dict with:
        - unrealized_pnl: Dollar amount
        - pnl_percent: Percentage of position value
        - leveraged_pnl_percent: Percentage including leverage effect
        - price_change_pct: Raw price change
    """

    # Price change percentage
    if side == 'LONG':
        price_change_pct = (current_price - entry_price) / entry_price
    else:  # SHORT
        price_change_pct = (entry_price - current_price) / entry_price

    # Leveraged P&L
    leverage_decimal = Decimal(str(leverage))
    unrealized_pnl = position_value * price_change_pct * leverage_decimal

    return {
        'unrealized_pnl': unrealized_pnl,
        'pnl_percent': float(price_change_pct * 100),
        'leveraged_pnl_percent': float(price_change_pct * leverage_decimal * 100),
        'price_change_pct': float(price_change_pct)
    }


def format_duration(seconds: int) -> str:
    """Format duration in a human-readable way."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def parse_ai_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse AI response JSON with error handling.

    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        # Try to find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')

        if start_idx == -1 or end_idx == -1:
            return None

        json_str = response_text[start_idx:end_idx + 1]
        return json.loads(json_str)

    except Exception as e:
        logging.getLogger('trading_bot').error(f"Failed to parse AI response: {e}")
        return None


def is_bullish(indicators: Dict[str, float]) -> bool:
    """
    Determine if technical indicators are bullish.

    Checks:
    - RSI between 40-70 (not overbought)
    - MACD above signal line
    - Price above SMA
    """
    rsi = indicators.get('rsi', 50)
    macd = indicators.get('macd', 0)
    macd_signal = indicators.get('macd_signal', 0)
    price = indicators.get('close', 0)
    sma = indicators.get('sma_20', price)

    conditions = [
        40 < rsi < 70,  # Not oversold or overbought
        macd > macd_signal,  # MACD bullish
        price > sma  # Price above SMA
    ]

    # Require at least 2 out of 3 conditions
    return sum(conditions) >= 2


def is_bearish(indicators: Dict[str, float]) -> bool:
    """
    Determine if technical indicators are bearish.

    Checks:
    - RSI between 30-60 (not oversold)
    - MACD below signal line
    - Price below SMA
    """
    rsi = indicators.get('rsi', 50)
    macd = indicators.get('macd', 0)
    macd_signal = indicators.get('macd_signal', 0)
    price = indicators.get('close', 0)
    sma = indicators.get('sma_20', price)

    conditions = [
        30 < rsi < 60,  # Not oversold or overbought
        macd < macd_signal,  # MACD bearish
        price < sma  # Price below SMA
    ]

    # Require at least 2 out of 3 conditions
    return sum(conditions) >= 2


def safe_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """Safely convert value to Decimal."""
    try:
        if value is None:
            return default
        return Decimal(str(value))
    except:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        if value is None:
            return default
        return float(value)
    except:
        return default
