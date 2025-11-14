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

    # Create logger
    logger = logging.getLogger('trading_bot')

    # If logger already has handlers, don't add more (prevent duplicates)
    if logger.handlers:
        return logger

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
    leverage: int,
    fixed_position_usd: Optional[Decimal] = None
) -> tuple[Decimal, Decimal]:
    """
    Calculate position quantity and value.

    Args:
        capital: Current capital
        position_size_percent: Position size as % of capital (used if fixed_position_usd is None)
        entry_price: Entry price
        leverage: Leverage multiplier
        fixed_position_usd: If set, uses this fixed USD amount instead of percentage

    Returns:
        (quantity, position_value_usd)
    """
    # FIXED USD SIZING: Use fixed amount if specified (e.g., always $100)
    if fixed_position_usd is not None:
        position_value = fixed_position_usd
        logger = logging.getLogger('trading_bot')
        logger.info(f"💰 Using FIXED position size: ${float(position_value):.2f} (not {float(position_size_percent*100):.1f}% of capital)")
    else:
        # PERCENTAGE SIZING: Calculate position value as % of capital
        position_value = capital * position_size_percent

    # Calculate quantity
    # 🔴 CRITICAL FIX: Don't multiply by leverage here!
    # Leverage is already applied by Binance when opening the position.
    # We only need: quantity = margin / price (NOT margin * leverage / price)
    #
    # Example:
    # - Margin (our capital): $10
    # - Entry price: $0.52
    # - Leverage: 10x
    #
    # WRONG (old): quantity = ($10 × 10) / $0.52 = 192 ADA
    #              → Binance sees: 192 × $0.52 × 10x = $998 position! ❌
    #
    # CORRECT: quantity = $10 / $0.52 = 19.2 ADA
    #          → Binance sees: 19.2 × $0.52 × 10x = $100 position! ✅
    quantity = position_value / entry_price

    return quantity, position_value


def calculate_stop_loss_price(
    entry_price: Decimal,
    stop_loss_percent: Decimal,
    side: str,
    leverage: Optional[int] = None,
    position_value: Optional[Decimal] = None
) -> Decimal:
    """
    Calculate stop-loss price based on maximum USD loss (accounting for leverage).

    CRITICAL FIX: The stop-loss must account for leverage!
    - Without leverage consideration: 10% price stop = 10% loss
    - With 10x leverage: 10% price stop = 100% loss (LIQUIDATION!)

    The correct formula:
    - Max USD loss = position_value * stop_loss_percent / 100
    - Price stop % = (Max USD loss / position_value) / leverage

    Args:
        entry_price: Entry price
        stop_loss_percent: Maximum loss as % of position value (e.g., 10 for 10%)
        side: 'LONG' or 'SHORT'
        leverage: Position leverage (if None, uses old simple calculation)
        position_value: Position value in USD (if None, uses old simple calculation)

    Returns:
        Stop-loss price
    """
    # NEW: Leverage-adjusted stop-loss calculation
    if leverage is not None and position_value is not None and leverage > 1:
        # Calculate price move % that results in the desired USD loss
        # Example: $100 position, 10% max loss = $10, 9x leverage
        # Price move % = $10 / $100 / 9 = 1.11% (not 10%!)
        max_usd_loss_percent = stop_loss_percent / 100  # e.g., 0.10 for 10%
        price_move_percent = max_usd_loss_percent / Decimal(str(leverage))

        if side == 'LONG':
            # For longs, stop-loss is below entry
            sl_price = entry_price * (1 - price_move_percent)
        else:  # SHORT
            # For shorts, stop-loss is above entry
            sl_price = entry_price * (1 + price_move_percent)

        logger = logging.getLogger('trading_bot')
        logger.info(
            f"🎯 LEVERAGE-ADJUSTED STOP-LOSS: {side} {leverage}x | "
            f"Max loss: {float(stop_loss_percent):.1f}% (${float(position_value * max_usd_loss_percent):.2f}) | "
            f"Price stop: {float(price_move_percent * 100):.2f}% | "
            f"Entry: ${float(entry_price):.4f} → Stop: ${float(sl_price):.4f}"
        )

        return sl_price

    # OLD: Simple percentage-based calculation (fallback for legacy code)
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
    position_value: Decimal,
    include_fees: bool = True
) -> Dict[str, Any]:
    """
    Calculate comprehensive P&L metrics WITH trading fees.

    Args:
        entry_price: Position entry price
        current_price: Current market price
        quantity: Position quantity
        side: 'LONG' or 'SHORT'
        leverage: Position leverage
        position_value: Position value in USD
        include_fees: Include Binance trading fees (default: True)

    Returns dict with:
        - unrealized_pnl: NET dollar amount (after fees)
        - gross_pnl: GROSS dollar amount (before fees)
        - total_fees: Entry + estimated exit fees
        - pnl_percent: Percentage of position value
        - leveraged_pnl_percent: Percentage including leverage effect
        - price_change_pct: Raw price change
    """

    # Price change percentage
    if side == 'LONG':
        price_change_pct = (current_price - entry_price) / entry_price
    else:  # SHORT
        price_change_pct = (entry_price - current_price) / entry_price

    # Leveraged P&L (GROSS - before fees)
    leverage_decimal = Decimal(str(leverage))
    gross_pnl = position_value * price_change_pct * leverage_decimal

    # Calculate trading fees (Binance futures taker: 0.05%)
    total_fees = Decimal("0")
    if include_fees:
        taker_fee_rate = Decimal("0.0005")  # 0.05%

        # Entry fee
        entry_notional = quantity * entry_price
        entry_fee = entry_notional * taker_fee_rate

        # Exit fee (estimated at current price)
        exit_notional = quantity * current_price
        exit_fee = exit_notional * taker_fee_rate

        total_fees = entry_fee + exit_fee

    # NET P&L (after fees)
    unrealized_pnl = gross_pnl - total_fees

    return {
        'unrealized_pnl': unrealized_pnl,  # NET (after fees)
        'gross_pnl': gross_pnl,  # GROSS (before fees)
        'total_fees': total_fees,
        'pnl_percent': float((unrealized_pnl / position_value) * 100),
        'leveraged_pnl_percent': float(price_change_pct * leverage_decimal * 100),
        'price_change_pct': float(price_change_pct)
    }


def calculate_profit_targets(
    entry_price: Decimal,
    side: str,
    position_value: Decimal,
    leverage: int,
    market_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Decimal]:
    """
    Calculate scaled profit targets for partial exit strategy.

    🎯 STRATEGY:
    - Target 1: Close 50% of position for guaranteed profit ($8)
    - Target 2: Trail remaining 50% for bigger gains ($12-15)
    - After T1 hit: Move stop-loss to breakeven (zero risk)

    Args:
        entry_price: Entry price of position
        side: 'LONG' or 'SHORT'
        position_value: Position value in USD
        leverage: Leverage used
        market_data: Optional market data with S/R levels and ATR

    Returns:
        Dict with profit_target_1, profit_target_2, target_1_profit_usd, target_2_profit_usd
    """

    # Calculate ATR-based targets (if market data available)
    atr = None
    if market_data:
        indicators = market_data.get('indicators', {})
        atr_data = indicators.get('15m', {}).get('volatility', {})
        atr = atr_data.get('atr')

    # Fallback ATR if not provided (1.5% of price)
    if not atr:
        atr = entry_price * Decimal("0.015")

    atr = Decimal(str(atr))

    if side == 'LONG':
        # TARGET 1: First resistance OR 1.5x ATR (aim for $8 profit)
        # With $100 position at 10-14x leverage, need ~0.6-0.8% price move for $8
        target_1_distance_pct = Decimal("0.008")  # 0.8% = $8 profit on $100 with 10x

        # Check if resistance is nearby
        if market_data:
            sr = market_data.get('indicators', {}).get('15m', {}).get('support_resistance', {})
            resistance = sr.get('nearest_resistance')

            if resistance:
                resistance = Decimal(str(resistance))
                # Use resistance if it's 0.8-3% away (good target zone)
                distance_to_resistance = (resistance - entry_price) / entry_price

                if Decimal("0.008") <= distance_to_resistance <= Decimal("0.03"):
                    # Use resistance as target (but leave 1% room for safety)
                    profit_target_1 = resistance * Decimal("0.99")
                else:
                    # Resistance too far/close, use calculated target
                    profit_target_1 = entry_price * (Decimal("1") + target_1_distance_pct)
            else:
                profit_target_1 = entry_price * (Decimal("1") + target_1_distance_pct)
        else:
            profit_target_1 = entry_price * (Decimal("1") + target_1_distance_pct)

        # TARGET 2: 2.5x ATR OR next resistance (aim for $12-15)
        target_2_distance_pct = Decimal("0.015")  # 1.5% = $15 profit on $100 with 10x
        profit_target_2 = entry_price * (Decimal("1") + target_2_distance_pct)

    else:  # SHORT
        # TARGET 1: First support OR 1.5x ATR (aim for $8 profit)
        target_1_distance_pct = Decimal("0.008")  # 0.8%

        # Check if support is nearby
        if market_data:
            sr = market_data.get('indicators', {}).get('15m', {}).get('support_resistance', {})
            support = sr.get('nearest_support')

            if support:
                support = Decimal(str(support))
                distance_to_support = (entry_price - support) / entry_price

                if Decimal("0.008") <= distance_to_support <= Decimal("0.03"):
                    # Use support as target (leave 1% room)
                    profit_target_1 = support * Decimal("1.01")
                else:
                    profit_target_1 = entry_price * (Decimal("1") - target_1_distance_pct)
            else:
                profit_target_1 = entry_price * (Decimal("1") - target_1_distance_pct)
        else:
            profit_target_1 = entry_price * (Decimal("1") - target_1_distance_pct)

        # TARGET 2: 2.5x ATR OR next support
        target_2_distance_pct = Decimal("0.015")  # 1.5%
        profit_target_2 = entry_price * (Decimal("1") - target_2_distance_pct)

    # Calculate expected profit in USD for each target
    # Target 1: 50% of position
    price_change_1 = abs(profit_target_1 - entry_price) / entry_price
    target_1_profit_usd = (position_value * Decimal("0.5")) * price_change_1 * Decimal(str(leverage))

    # Target 2: Remaining 50% (total profit if both hit)
    price_change_2 = abs(profit_target_2 - entry_price) / entry_price
    target_2_total_profit_usd = (
        target_1_profit_usd +  # Profit from T1 (already closed)
        (position_value * Decimal("0.5")) * price_change_2 * Decimal(str(leverage))  # Profit from T2
    )

    return {
        'profit_target_1': profit_target_1,  # First target (close 50%)
        'profit_target_2': profit_target_2,  # Second target (close remaining 50%)
        'target_1_profit_usd': target_1_profit_usd,  # Expected profit at T1
        'target_2_profit_usd': target_2_total_profit_usd,  # Total profit if T2 hit
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
