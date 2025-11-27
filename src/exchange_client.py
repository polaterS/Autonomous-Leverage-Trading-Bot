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
                'defaultType': 'swap',  # Use USDT-M Perpetual Swaps (fapi)
                'adjustForTimeDifference': True,
                # 🔥 Suppress warning when fetching all open orders without symbol
                # This is intentional for orphan orders cleanup
                'warnOnFetchOpenOrdersWithoutSymbol': False,
            }
        })

        # Paper trading - REAL SIMULATION (not sandbox mode)
        if self.paper_trading:
            logger.warning("=" * 60)
            logger.warning("PAPER TRADING MODE - Full Market Data, Simulated Orders")
            logger.warning("Real market prices will be used, but NO real orders executed")
            logger.warning("=" * 60)

        # Paper trading state
        self.paper_balance = self.settings.initial_capital
        self.paper_positions = {}  # Track simulated positions
        self.paper_orders = {}  # Track simulated orders

        # Slippage simulation for paper trading
        self.paper_slippage_percent = Decimal("0.05")  # 0.05% average slippage

    async def initialize(self):
        """Initialize exchange connection and load markets."""
        try:
            await self.exchange.load_markets()
            logger.info(f"Exchange initialized: {self.exchange.id}")

            if not self.paper_trading:
                # Test connection
                try:
                    balance = await self.exchange.fetch_balance()
                    logger.info(f"Account balance loaded: ${balance.get('USDT', {}).get('free', 0):.2f} USDT")
                except Exception as balance_err:
                    error_msg = str(balance_err)

                    # Check if it's the -2015 error (Futures account not activated)
                    if '-2015' in error_msg or 'permissions' in error_msg.lower():
                        logger.error("=" * 60)
                        logger.error("BINANCE FUTURES ACCOUNT NOT ACTIVATED")
                        logger.error("=" * 60)
                        logger.error("Your API key has 'Enable Futures' permission, but your")
                        logger.error("Binance account doesn't have Futures trading activated yet.")
                        logger.error("")
                        logger.error("To activate:")
                        logger.error("1. Go to https://www.binance.com/en/futures/BTCUSDT")
                        logger.error("2. Click 'Open Now' and agree to terms")
                        logger.error("3. Complete any required verification")
                        logger.error("4. Restart this bot")
                        logger.error("=" * 60)

                    raise Exception(f"Futures account not accessible: {balance_err}")

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
            logger.debug(f"[PAPER] Balance: ${self.paper_balance:.2f}")
            return self.paper_balance

        try:
            balance = await self.exchange.fetch_balance()
            usdt_balance = safe_decimal(balance.get('USDT', {}).get('free', 0))
            return usdt_balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise

    async def fetch_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Fetch all open orders for a symbol (or all symbols if None).

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT'). If None, fetches all open orders.

        Returns:
            List of open orders
        """
        if self.paper_trading:
            logger.debug(f"[PAPER] Fetching open orders for {symbol or 'all symbols'}")
            if symbol:
                return [order for order in self.paper_orders.values() if order['symbol'] == symbol]
            return list(self.paper_orders.values())

        try:
            open_orders = await self.exchange.fetch_open_orders(symbol)
            logger.debug(f"Fetched {len(open_orders)} open order(s) for {symbol or 'all symbols'}")
            return open_orders
        except Exception as e:
            logger.error(f"Error fetching open orders for {symbol}: {e}")
            return []  # Return empty list on error (non-critical)

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
        Create a market order with slippage simulation for paper trading.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT:USDT')
            side: 'buy' or 'sell'
            amount: Amount to trade
            params: Additional parameters

        Returns:
            Order info dict with actual execution price (including slippage)
        """
        if self.paper_trading:
            # Get real market price
            ticker = await self.fetch_ticker(symbol)
            market_price = safe_decimal(ticker['last'])

            # Simulate realistic Binance Futures slippage (0.01-0.02% - very tight)
            import random
            slippage = Decimal(str(random.uniform(0.01, 0.02))) / 100

            # Buying increases price slightly, selling decreases it
            if side == 'buy':
                execution_price = market_price * (1 + slippage)
            else:  # sell
                execution_price = market_price * (1 - slippage)

            import time
            order_id = f'paper_{int(time.time() * 1000)}_{random.randint(1000, 9999)}'

            order = {
                'id': order_id,
                'symbol': symbol,
                'type': 'market',
                'side': side,
                'amount': float(amount),
                'price': float(market_price),  # Market price at time of order
                'average': float(execution_price),  # Actual execution price with slippage
                'filled': float(amount),
                'remaining': 0,
                'status': 'closed',
                'timestamp': int(time.time() * 1000),
                'info': {
                    'paper_trade': True,
                    'market_price': float(market_price),
                    'slippage_percent': float(slippage * 100),
                    'slippage_cost': float(abs(execution_price - market_price) * amount)
                }
            }

            logger.info(
                f"[PAPER] Market {side}: {amount:.6f} {symbol} @ "
                f"${execution_price:.4f} (slippage: {slippage*100:.3f}%, "
                f"cost: ${abs(execution_price - market_price) * amount:.2f})"
            )
            return order

        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=float(amount),
                params=params or {}
            )

            # Log real order execution
            execution_price = safe_decimal(order.get('average', order.get('price', 0)))
            logger.info(f"Market {side} order executed: {amount} {symbol} @ ${execution_price:.4f}")
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
        Create a stop-loss order with retry logic.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell' (opposite of position)
            amount: Amount to trade
            stop_price: Trigger price for stop-loss

        Returns:
            Order info dict

        Raises:
            Exception if order placement fails after retries
        """
        if self.paper_trading:
            import time
            import random
            order_id = f'paper_sl_{int(time.time() * 1000)}_{random.randint(1000, 9999)}'

            # Store paper stop-loss order for monitoring
            self.paper_orders[order_id] = {
                'id': order_id,
                'symbol': symbol,
                'type': 'stop_market',
                'side': side,
                'amount': float(amount),
                'stopPrice': float(stop_price),
                'status': 'open',
                'timestamp': int(time.time() * 1000)
            }

            logger.info(f"[PAPER] Stop-loss order placed: {side} {amount:.6f} {symbol} @ ${stop_price:.4f}")
            return self.paper_orders[order_id]

        # Real trading - retry logic for stop-loss placement
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # 🔥 CRITICAL FIX: Binance Futures requires specific params for stop-loss
                # - reduceOnly: true (must close position, not open new one)
                # - stopPrice: trigger price
                # - closePosition: false (we specify amount, not close entire position)
                order = await self.exchange.create_order(
                    symbol=symbol,
                    type='stop_market',
                    side=side,
                    amount=float(amount),
                    params={
                        'stopPrice': float(stop_price),
                        'reduceOnly': True,  # 🔥 CRITICAL: Must be True for Binance Futures
                        'closePosition': False  # We specify exact amount
                    }
                )

                logger.info(f"✅ Stop-loss order created: {side} {amount} {symbol} @ ${stop_price:.4f} (Order ID: {order.get('id', 'N/A')})")
                return order

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                logger.warning(f"⚠️ Stop-loss placement attempt {attempt + 1}/{max_retries} failed: {e}")

                # Log detailed error for debugging
                if 'reduce' in error_msg or 'position' in error_msg:
                    logger.error("💥 REDUCE_ONLY error - Binance rejected stop-loss!")

                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

        # All retries failed
        logger.error(f"🚨 CRITICAL: Stop-loss placement failed after {max_retries} attempts: {last_error}")

        # Send critical alert
        try:
            from src.telegram_notifier import get_notifier
            notifier = get_notifier()
            await notifier.send_alert(
                'critical',
                f"🚨 STOP-LOSS PLACEMENT FAILED\n\n"
                f"Symbol: {symbol}\n"
                f"Side: {side}\n"
                f"Amount: {amount}\n"
                f"Stop Price: ${float(stop_price):.4f}\n\n"
                f"Error: {last_error}\n\n"
                f"⚠️ POSITION IS UNPROTECTED!\n"
                f"Manual intervention required!"
            )
        except:
            pass

        raise Exception(f"Stop-loss order placement failed: {last_error}")

    async def create_take_profit_order(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        take_profit_price: Decimal
    ) -> Dict[str, Any]:
        """
        Create a take-profit limit order.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell' (opposite of position)
            amount: Amount to trade
            take_profit_price: Target price for take-profit

        Returns:
            Order info dict

        Raises:
            Exception if order placement fails
        """
        if self.paper_trading:
            import time
            import random
            order_id = f'paper_tp_{int(time.time() * 1000)}_{random.randint(1000, 9999)}'

            # Store paper take-profit order for monitoring
            self.paper_orders[order_id] = {
                'id': order_id,
                'symbol': symbol,
                'type': 'limit',
                'side': side,
                'amount': float(amount),
                'price': float(take_profit_price),
                'status': 'open',
                'timestamp': int(time.time() * 1000)
            }

            logger.info(f"[PAPER] Take-profit order placed: {side} {amount:.6f} {symbol} @ ${take_profit_price:.4f}")
            return self.paper_orders[order_id]

        # Real trading - place limit order at take-profit price
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=float(amount),
                price=float(take_profit_price)
            )

            logger.info(f"Take-profit order created: {side} {amount} {symbol} @ ${take_profit_price:.4f}")
            return order

        except Exception as e:
            logger.error(f"Take-profit placement failed: {e}")
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
