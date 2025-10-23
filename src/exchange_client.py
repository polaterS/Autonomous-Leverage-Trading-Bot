"""
Exchange client wrapper for Binance Futures using CCXT.
Handles all exchange interactions including order execution and market data.
"""

import ccxt.async_support as ccxt
import asyncio
from typing import Dict, Any, List, Optional
from decimal import Decimal
from src.config import get_settings
from src.utils import setup_logging, safe_decimal, safe_float
import pandas as pd

logger = setup_logging()


class ExchangeClient:
    """Async wrapper for CCXT Binance Futures exchange."""

    def __init__(self, paper_trading: bool = None):
        self.settings = get_settings()
        self.paper_trading = paper_trading if paper_trading is not None else self.settings.use_paper_trading

        # Initialize CCXT exchange
        self.exchange = ccxt.binance({
            'apiKey': self.settings.binance_api_key,
            'secret': self.settings.binance_secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Use futures
                'adjustForTimeDifference': True,
            }
        })

        if self.paper_trading:
            logger.warning("PAPER TRADING MODE - No real orders will be executed")
            self.exchange.set_sandbox_mode(True)  # Use testnet if available

        self.paper_balance = Decimal("100.00")  # Virtual balance for paper trading

    async def initialize(self):
        """Initialize exchange connection and load markets."""
        try:
            await self.exchange.load_markets()
            logger.info(f"Exchange initialized: {self.exchange.id}")

            if not self.paper_trading:
                # Test connection
                balance = await self.exchange.fetch_balance()
                logger.info(f"Account balance loaded: ${balance.get('USDT', {}).get('free', 0):.2f} USDT")

        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise

    async def close(self):
        """Close exchange connection."""
        await self.exchange.close()
        logger.info("Exchange connection closed")

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data for a symbol.

        Returns:
            Dict with 'last', 'bid', 'ask', 'high', 'low', 'volume', etc.
        """
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """
        Fetch OHLCV candlestick data.

        Returns:
            List of [timestamp, open, high, low, close, volume]
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} {timeframe}: {e}")
            raise

    async def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current funding rate for perpetual futures.

        Returns:
            Dict with 'rate', 'timestamp', etc.
        """
        try:
            funding_rate_data = await self.exchange.fetch_funding_rate(symbol)
            # CCXT returns 'fundingRate' (camelCase), normalize to 'rate'
            return {
                'rate': funding_rate_data.get('fundingRate', 0.0),
                'timestamp': funding_rate_data.get('timestamp'),
                'datetime': funding_rate_data.get('datetime')
            }
        except Exception as e:
            logger.warning(f"Error fetching funding rate for {symbol}: {e}")
            # Return default if not available
            return {'rate': 0.0, 'timestamp': None}

    async def fetch_balance(self) -> Decimal:
        """Get current USDT balance."""
        if self.paper_trading:
            return self.paper_balance

        try:
            balance = await self.exchange.fetch_balance()
            usdt_balance = safe_decimal(balance.get('USDT', {}).get('free', 0))
            return usdt_balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol."""
        if self.paper_trading:
            logger.info(f"[PAPER] Set leverage for {symbol}: {leverage}x")
            return

        try:
            await self.exchange.set_leverage(leverage, symbol)
            logger.info(f"Leverage set for {symbol}: {leverage}x")
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            raise

    async def set_margin_mode(self, symbol: str, margin_mode: str = 'isolated') -> None:
        """Set margin mode (isolated or cross)."""
        if self.paper_trading:
            logger.info(f"[PAPER] Set margin mode for {symbol}: {margin_mode}")
            return

        try:
            await self.exchange.set_margin_mode(margin_mode, symbol)
            logger.info(f"Margin mode set for {symbol}: {margin_mode}")
        except Exception as e:
            # Margin mode might already be set
            logger.warning(f"Could not set margin mode for {symbol}: {e}")

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a market order.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT:USDT')
            side: 'buy' or 'sell'
            amount: Amount to trade
            params: Additional parameters

        Returns:
            Order info dict
        """
        if self.paper_trading:
            # Simulate order for paper trading
            ticker = await self.fetch_ticker(symbol)
            price = safe_decimal(ticker['last'])

            import time
            order = {
                'id': f'paper_{int(time.time() * 1000)}',
                'symbol': symbol,
                'type': 'market',
                'side': side,
                'amount': float(amount),
                'price': float(price),
                'average': float(price),
                'filled': float(amount),
                'remaining': 0,
                'status': 'closed',
                'timestamp': int(time.time() * 1000),
                'info': {'paper_trade': True}
            }

            logger.info(f"[PAPER] Market {side} order: {amount} {symbol} @ ${price:.4f}")
            return order

        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=float(amount),
                params=params or {}
            )

            logger.info(f"Market {side} order executed: {amount} {symbol} @ ${order.get('average', 0):.4f}")
            return order

        except Exception as e:
            logger.error(f"Error creating market order: {e}")
            raise

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        stop_price: Decimal
    ) -> Dict[str, Any]:
        """
        Create a stop-loss order.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell' (opposite of position)
            amount: Amount to trade
            stop_price: Trigger price for stop-loss

        Returns:
            Order info dict
        """
        if self.paper_trading:
            import time
            logger.info(f"[PAPER] Stop-loss order: {side} {amount} {symbol} @ ${stop_price:.4f}")
            return {
                'id': f'paper_sl_{int(time.time() * 1000)}',
                'symbol': symbol,
                'type': 'stop_market',
                'side': side,
                'stopPrice': float(stop_price),
                'status': 'open'
            }

        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type='stop_market',
                side=side,
                amount=float(amount),
                params={'stopPrice': float(stop_price)}
            )

            logger.info(f"Stop-loss order created: {side} {amount} {symbol} @ ${stop_price:.4f}")
            return order

        except Exception as e:
            logger.error(f"Error creating stop-loss order: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> None:
        """Cancel an open order."""
        if self.paper_trading:
            logger.info(f"[PAPER] Cancelled order: {order_id}")
            return

        try:
            await self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order cancelled: {order_id}")
        except Exception as e:
            logger.warning(f"Error cancelling order {order_id}: {e}")

    async def fetch_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch current position for a symbol.

        Returns:
            Position dict or None
        """
        if self.paper_trading:
            return None  # Paper trading doesn't track exchange positions

        try:
            positions = await self.exchange.fetch_positions([symbol])
            for pos in positions:
                if pos['symbol'] == symbol and float(pos.get('contracts', 0)) > 0:
                    return pos
            return None

        except Exception as e:
            logger.warning(f"Error fetching position for {symbol}: {e}")
            return None

    async def close_position(self, symbol: str, side: str, amount: Decimal) -> Dict[str, Any]:
        """
        Close a position completely.

        Args:
            symbol: Trading pair
            side: Position side ('LONG' or 'SHORT')
            amount: Amount to close

        Returns:
            Order info
        """
        # Determine close order side (opposite of position)
        close_side = 'sell' if side == 'LONG' else 'buy'

        return await self.create_market_order(symbol, close_side, amount)

    async def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """Get market information (price precision, min amount, etc.)."""
        try:
            market = self.exchange.market(symbol)
            return {
                'precision': {
                    'price': market.get('precision', {}).get('price', 4),
                    'amount': market.get('precision', {}).get('amount', 3)
                },
                'limits': {
                    'amount': market.get('limits', {}).get('amount', {}),
                    'cost': market.get('limits', {}).get('cost', {})
                }
            }
        except Exception as e:
            logger.warning(f"Error getting market info for {symbol}: {e}")
            return {
                'precision': {'price': 4, 'amount': 3},
                'limits': {}
            }

    def update_paper_balance(self, pnl: Decimal) -> None:
        """Update paper trading balance."""
        self.paper_balance += pnl
        logger.info(f"[PAPER] Balance updated: ${self.paper_balance:.2f} (P&L: ${pnl:+.2f})")


# Singleton instance
_exchange_client: Optional[ExchangeClient] = None


async def get_exchange_client() -> ExchangeClient:
    """Get or create exchange client instance."""
    global _exchange_client
    if _exchange_client is None:
        _exchange_client = ExchangeClient()
        await _exchange_client.initialize()
    return _exchange_client
