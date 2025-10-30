"""
Unit Tests for Risk Manager.

Tests critical risk management functionality:
- Trade validation (stop-loss, leverage, capital checks)
- Circuit breakers (daily loss, consecutive losses)
- Position size calculations
- Emergency close conditions
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import date
from src.risk_manager import RiskManager
from src.database import DatabaseClient


class MockDatabaseClient:
    """Mock database for testing."""

    def __init__(self):
        self.trading_config = {
            'current_capital': Decimal("100.00"),
            'max_leverage': 5,
            'min_stop_loss_percent': Decimal("0.05"),
            'max_stop_loss_percent': Decimal("0.10"),
            'daily_loss_limit_percent': Decimal("0.10"),
            'max_consecutive_losses': 3,
            'is_trading_enabled': True
        }
        self.consecutive_losses = 0
        self.daily_pnl = Decimal("0")

    async def get_trading_config(self):
        return self.trading_config

    async def get_consecutive_losses(self):
        return self.consecutive_losses

    async def get_daily_pnl(self, target_date=None):
        return self.daily_pnl

    async def get_active_position(self):
        return None


@pytest.fixture
def mock_db():
    """Create mock database."""
    return MockDatabaseClient()


@pytest.fixture
def risk_manager(mock_db, monkeypatch):
    """Create risk manager with mocked database."""
    rm = RiskManager()

    # Monkey-patch the get_db_client to return our mock
    async def mock_get_db():
        return mock_db

    import src.risk_manager
    monkeypatch.setattr(src.risk_manager, 'get_db_client', mock_get_db)

    return rm


# ==================== TRADE VALIDATION TESTS ====================

@pytest.mark.asyncio
async def test_valid_trade_approved(risk_manager, mock_db):
    """Test that a valid trade is approved."""
    trade_params = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'LONG',
        'leverage': 3,
        'stop_loss_percent': 0.07,
        'current_price': Decimal("50000")
    }

    result = await risk_manager.validate_trade(trade_params)

    assert result['approved'] is True
    assert 'reason' not in result or result['reason'] == ''


@pytest.mark.asyncio
async def test_stop_loss_too_low_rejected(risk_manager, mock_db):
    """Test that stop-loss below minimum is rejected."""
    trade_params = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'LONG',
        'leverage': 3,
        'stop_loss_percent': 0.03,  # Below 5% minimum
        'current_price': Decimal("50000")
    }

    result = await risk_manager.validate_trade(trade_params)

    assert result['approved'] is False
    assert 'stop-loss' in result['reason'].lower()


@pytest.mark.asyncio
async def test_stop_loss_too_high_adjusted(risk_manager, mock_db):
    """Test that stop-loss above maximum is adjusted."""
    trade_params = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'LONG',
        'leverage': 3,
        'stop_loss_percent': 0.15,  # Above 10% maximum
        'current_price': Decimal("50000")
    }

    result = await risk_manager.validate_trade(trade_params)

    # Should either reject or adjust
    if result['approved']:
        assert result['adjusted_params']['stop_loss_percent'] <= 0.10
    else:
        assert 'stop-loss' in result['reason'].lower()


@pytest.mark.asyncio
async def test_leverage_too_high_rejected(risk_manager, mock_db):
    """Test that leverage above maximum is rejected."""
    trade_params = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'LONG',
        'leverage': 10,  # Above 5x maximum
        'stop_loss_percent': 0.07,
        'current_price': Decimal("50000")
    }

    result = await risk_manager.validate_trade(trade_params)

    assert result['approved'] is False
    assert 'leverage' in result['reason'].lower()


@pytest.mark.asyncio
async def test_insufficient_capital_rejected(risk_manager, mock_db):
    """Test that trade is rejected if capital too low."""
    mock_db.trading_config['current_capital'] = Decimal("5.00")  # Very low

    trade_params = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'LONG',
        'leverage': 3,
        'stop_loss_percent': 0.07,
        'current_price': Decimal("50000")
    }

    result = await risk_manager.validate_trade(trade_params)

    assert result['approved'] is False
    assert 'capital' in result['reason'].lower()


# ==================== CIRCUIT BREAKER TESTS ====================

@pytest.mark.asyncio
async def test_daily_loss_limit_triggered(risk_manager, mock_db):
    """Test that daily loss limit triggers circuit breaker."""
    mock_db.daily_pnl = Decimal("-12.00")  # 12% loss on $100 capital

    trade_params = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'LONG',
        'leverage': 3,
        'stop_loss_percent': 0.07,
        'current_price': Decimal("50000")
    }

    result = await risk_manager.validate_trade(trade_params)

    assert result['approved'] is False
    assert 'daily loss limit' in result['reason'].lower()


@pytest.mark.asyncio
async def test_consecutive_losses_triggered(risk_manager, mock_db):
    """Test that consecutive loss limit triggers circuit breaker."""
    mock_db.consecutive_losses = 3  # At limit

    trade_params = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'LONG',
        'leverage': 3,
        'stop_loss_percent': 0.07,
        'current_price': Decimal("50000")
    }

    result = await risk_manager.validate_trade(trade_params)

    assert result['approved'] is False
    assert 'consecutive losses' in result['reason'].lower()


@pytest.mark.asyncio
async def test_trading_disabled_rejected(risk_manager, mock_db):
    """Test that trades are rejected when trading is disabled."""
    mock_db.trading_config['is_trading_enabled'] = False

    trade_params = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'LONG',
        'leverage': 3,
        'stop_loss_percent': 0.07,
        'current_price': Decimal("50000")
    }

    result = await risk_manager.validate_trade(trade_params)

    assert result['approved'] is False
    assert 'disabled' in result['reason'].lower()


# ==================== EMERGENCY CLOSE TESTS ====================

@pytest.mark.asyncio
async def test_emergency_close_liquidation_risk(risk_manager):
    """Test emergency close when close to liquidation."""
    position = {
        'leverage': 5,
        'entry_price': Decimal("50000"),
        'liquidation_price': Decimal("40000"),  # 20% away
        'side': 'LONG'
    }

    # Current price very close to liquidation
    current_price = Decimal("40500")  # Only 1.25% away

    should_close, reason = await risk_manager.should_emergency_close(
        position, current_price
    )

    assert should_close is True
    assert 'liquidation' in reason.lower()


@pytest.mark.asyncio
async def test_no_emergency_close_safe_distance(risk_manager):
    """Test no emergency close when liquidation distance is safe."""
    position = {
        'leverage': 3,
        'entry_price': Decimal("50000"),
        'liquidation_price': Decimal("35000"),  # 30% away
        'side': 'LONG'
    }

    # Current price safely above liquidation
    current_price = Decimal("48000")  # 27% away

    should_close, reason = await risk_manager.should_emergency_close(
        position, current_price
    )

    assert should_close is False


# ==================== POSITION SIZE TESTS ====================

def test_calculate_position_size_valid():
    """Test position size calculation."""
    from src.utils import calculate_position_size

    capital = Decimal("100.00")
    position_size_percent = Decimal("0.80")  # 80%
    entry_price = Decimal("50000")
    leverage = 3

    quantity, position_value = calculate_position_size(
        capital, position_size_percent, entry_price, leverage
    )

    # Position value should be 80% of capital
    assert position_value == Decimal("80.00")

    # Quantity = position_value / entry_price
    expected_quantity = Decimal("80.00") / Decimal("50000")
    assert abs(quantity - expected_quantity) < Decimal("0.000001")


def test_calculate_stop_loss_price_long():
    """Test stop-loss calculation for LONG position."""
    from src.utils import calculate_stop_loss_price

    entry_price = Decimal("50000")
    stop_loss_percent = Decimal("0.05")  # 5%
    side = 'LONG'

    sl_price = calculate_stop_loss_price(entry_price, stop_loss_percent, side)

    # Should be 5% below entry for LONG
    expected_sl = entry_price * (Decimal("1") - stop_loss_percent)
    assert sl_price == expected_sl
    assert sl_price == Decimal("47500")


def test_calculate_stop_loss_price_short():
    """Test stop-loss calculation for SHORT position."""
    from src.utils import calculate_stop_loss_price

    entry_price = Decimal("50000")
    stop_loss_percent = Decimal("0.05")  # 5%
    side = 'SHORT'

    sl_price = calculate_stop_loss_price(entry_price, stop_loss_percent, side)

    # Should be 5% above entry for SHORT
    expected_sl = entry_price * (Decimal("1") + stop_loss_percent)
    assert sl_price == expected_sl
    assert sl_price == Decimal("52500")


# ==================== RUN TESTS ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
