"""
Unit Tests for Utility Functions.

Tests critical utility calculations:
- Position size calculation
- P&L calculation
- Stop-loss price calculation
- Liquidation price calculation
- Duration formatting
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from src.utils import (
    calculate_position_size,
    calculate_pnl,
    calculate_stop_loss_price,
    calculate_liquidation_price,
    calculate_min_profit_price,
    format_duration,
    is_bullish,
    is_bearish
)


# ==================== POSITION SIZE TESTS ====================

def test_calculate_position_size_80_percent():
    """Test position size with 80% allocation."""
    capital = Decimal("100.00")
    position_size_percent = Decimal("0.80")
    entry_price = Decimal("50000")
    leverage = 3

    quantity, position_value = calculate_position_size(
        capital, position_size_percent, entry_price, leverage
    )

    assert position_value == Decimal("80.00")
    assert quantity == Decimal("80.00") / Decimal("50000")


def test_calculate_position_size_different_leverage():
    """Test position size with different leverage doesn't affect calculation."""
    capital = Decimal("100.00")
    position_size_percent = Decimal("0.80")
    entry_price = Decimal("50000")

    # Leverage doesn't affect position_value calculation
    _, value_3x = calculate_position_size(capital, position_size_percent, entry_price, 3)
    _, value_5x = calculate_position_size(capital, position_size_percent, entry_price, 5)

    assert value_3x == value_5x == Decimal("80.00")


# ==================== P&L CALCULATION TESTS ====================

def test_calculate_pnl_long_profit():
    """Test P&L calculation for profitable LONG."""
    entry_price = Decimal("50000")
    current_price = Decimal("52000")  # +4%
    quantity = Decimal("0.001")
    side = "LONG"
    leverage = 5
    position_value = Decimal("50.00")

    result = calculate_pnl(entry_price, current_price, quantity, side, leverage, position_value)

    # Price change: +4%, with 5x leverage = +20%
    expected_pnl = position_value * Decimal("0.04") * 5
    assert abs(result['unrealized_pnl'] - expected_pnl) < Decimal("0.01")


def test_calculate_pnl_short_profit():
    """Test P&L calculation for profitable SHORT."""
    entry_price = Decimal("50000")
    current_price = Decimal("48000")  # -4%
    quantity = Decimal("0.001")
    side = "SHORT"
    leverage = 3
    position_value = Decimal("50.00")

    result = calculate_pnl(entry_price, current_price, quantity, side, leverage, position_value)

    # Price change: -4%, SHORT profits from downmove, 3x leverage = +12%
    expected_pnl = position_value * Decimal("0.04") * 3
    assert abs(result['unrealized_pnl'] - expected_pnl) < Decimal("0.01")


def test_calculate_pnl_long_loss():
    """Test P&L calculation for losing LONG."""
    entry_price = Decimal("50000")
    current_price = Decimal("47500")  # -5%
    quantity = Decimal("0.001")
    side = "LONG"
    leverage = 4
    position_value = Decimal("50.00")

    result = calculate_pnl(entry_price, current_price, quantity, side, leverage, position_value)

    # Price change: -5%, with 4x leverage = -20%
    expected_pnl = position_value * Decimal("-0.05") * 4
    assert abs(result['unrealized_pnl'] - expected_pnl) < Decimal("0.01")
    assert result['unrealized_pnl'] < 0


# ==================== STOP-LOSS PRICE TESTS ====================

def test_stop_loss_long_5_percent():
    """Test stop-loss price for LONG with 5% SL."""
    entry_price = Decimal("50000")
    stop_loss_percent = Decimal("0.05")
    side = "LONG"

    sl_price = calculate_stop_loss_price(entry_price, stop_loss_percent, side)

    assert sl_price == Decimal("47500")  # 50000 * 0.95


def test_stop_loss_short_7_percent():
    """Test stop-loss price for SHORT with 7% SL."""
    entry_price = Decimal("50000")
    stop_loss_percent = Decimal("0.07")
    side = "SHORT"

    sl_price = calculate_stop_loss_price(entry_price, stop_loss_percent, side)

    assert sl_price == Decimal("53500")  # 50000 * 1.07


# ==================== LIQUIDATION PRICE TESTS ====================

def test_liquidation_price_long_5x():
    """Test liquidation price for LONG with 5x leverage."""
    entry_price = Decimal("50000")
    leverage = 5
    side = "LONG"

    liq_price = calculate_liquidation_price(entry_price, leverage, side)

    # Rough estimate: entry * (1 - 0.8/leverage)
    # For 5x: 50000 * (1 - 0.8/5) = 50000 * 0.84 = 42000
    assert liq_price < entry_price
    assert liq_price > entry_price * Decimal("0.8")


def test_liquidation_price_short_3x():
    """Test liquidation price for SHORT with 3x leverage."""
    entry_price = Decimal("50000")
    leverage = 3
    side = "SHORT"

    liq_price = calculate_liquidation_price(entry_price, leverage, side)

    # For SHORT, liquidation is above entry
    assert liq_price > entry_price


# ==================== MIN PROFIT PRICE TESTS ====================

def test_min_profit_price_long():
    """Test minimum profit target price for LONG."""
    entry_price = Decimal("50000")
    min_profit_usd = Decimal("2.50")
    position_value = Decimal("80.00")
    leverage = 5
    side = "LONG"

    min_price = calculate_min_profit_price(
        entry_price, min_profit_usd, position_value, leverage, side
    )

    # Should be slightly above entry price
    assert min_price > entry_price


def test_min_profit_price_short():
    """Test minimum profit target price for SHORT."""
    entry_price = Decimal("50000")
    min_profit_usd = Decimal("2.50")
    position_value = Decimal("80.00")
    leverage = 3
    side = "SHORT"

    min_price = calculate_min_profit_price(
        entry_price, min_profit_usd, position_value, leverage, side
    )

    # Should be slightly below entry price for SHORT
    assert min_price < entry_price


# ==================== DURATION FORMATTING TESTS ====================

def test_format_duration_minutes():
    """Test duration formatting for minutes."""
    seconds = 125  # 2 minutes 5 seconds

    formatted = format_duration(seconds)

    assert formatted == "2m"


def test_format_duration_hours():
    """Test duration formatting for hours."""
    seconds = 3725  # 1 hour 2 minutes 5 seconds

    formatted = format_duration(seconds)

    assert formatted == "1h 2m"


def test_format_duration_days():
    """Test duration formatting for days."""
    seconds = 90125  # 1 day 1 hour 2 minutes

    formatted = format_duration(seconds)

    assert formatted == "1d 1h"


# ==================== INDICATOR HELPERS TESTS ====================

def test_is_bullish_rsi_oversold_macd_positive():
    """Test bullish detection with RSI oversold and MACD positive."""
    indicators = {
        'rsi': 35,  # Oversold
        'macd': 10,
        'macd_signal': 5  # MACD > signal
    }

    assert is_bullish(indicators) is True


def test_is_bearish_rsi_overbought_macd_negative():
    """Test bearish detection with RSI overbought and MACD negative."""
    indicators = {
        'rsi': 72,  # Overbought
        'macd': -10,
        'macd_signal': -5  # MACD < signal
    }

    assert is_bearish(indicators) is True


def test_not_bullish_neutral():
    """Test neutral indicators don't trigger bullish."""
    indicators = {
        'rsi': 50,  # Neutral
        'macd': 0,
        'macd_signal': 0
    }

    assert is_bullish(indicators) is False


# ==================== EDGE CASES ====================

def test_position_size_very_small_capital():
    """Test position size with very small capital."""
    capital = Decimal("10.00")
    position_size_percent = Decimal("0.80")
    entry_price = Decimal("50000")
    leverage = 3

    quantity, position_value = calculate_position_size(
        capital, position_size_percent, entry_price, leverage
    )

    assert position_value == Decimal("8.00")
    assert quantity > 0


def test_pnl_zero_change():
    """Test P&L when price hasn't changed."""
    entry_price = Decimal("50000")
    current_price = Decimal("50000")
    quantity = Decimal("0.001")
    side = "LONG"
    leverage = 5
    position_value = Decimal("50.00")

    result = calculate_pnl(entry_price, current_price, quantity, side, leverage, position_value)

    assert result['unrealized_pnl'] == Decimal("0")


# ==================== RUN TESTS ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.utils", "--cov-report=term-missing"])
