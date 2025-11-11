"""
🌐 WebSocket Price Feed Manager
Real-time price streaming from Binance to eliminate rate limit errors

SOLVES:
- Binance 429 rate limit errors (2,400 requests/minute)
- REST API polling overhead (~40 calls/second)
- Price data latency

FEATURES:
- Real-time WebSocket price streaming for all active positions
- Automatic reconnection with exponential backoff
- Failover to REST API if WebSocket unavailable
- Thread-safe price cache with TTL
- Multi-symbol subscription management

Expected Impact: -80% API calls, zero rate limit errors, <100ms price updates
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from decimal import Decimal
import ccxtpro
from threading import Lock

logger = logging.getLogger(__name__)


class WebSocketPriceFeed:
    """
    Manages real-time WebSocket price feeds from Binance.

    Features:
    - Multi-symbol price streaming
    - Automatic reconnection
    - Price cache with thread safety
    - Failover to REST API
    """

    def __init__(self, exchange_id: str = "binance"):
        """
        Initialize WebSocket price feed manager.

        Args:
            exchange_id: Exchange name (default: binance)
        """
        self.exchange_id = exchange_id
        self.exchange = None  # ccxtpro exchange instance

        # Price cache: {symbol: {"price": Decimal, "timestamp": datetime}}
        self.price_cache: Dict[str, Dict] = {}
        self.cache_lock = Lock()

        # Subscription management
        self.subscribed_symbols: Set[str] = set()
        self.subscription_lock = Lock()

        # Connection state
        self.is_connected = False
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

        # Settings
        self.price_ttl_seconds = 30  # Price expires after 30 seconds
        self.reconnect_delay_base = 2  # Exponential backoff base (seconds)

        logger.info(f"✅ WebSocketPriceFeed initialized for {exchange_id}")

    async def start(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Start WebSocket connection to exchange.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet (default: False)
        """
        if self.is_running:
            logger.warning("WebSocket already running")
            return

        try:
            # Try multiple import methods for ccxtpro
            try:
                # Method 1: Direct import
                if self.exchange_id == 'binance':
                    from ccxtpro import binance as BinancePro
                    exchange_class = BinancePro
                else:
                    exchange_class = getattr(ccxtpro, self.exchange_id)
            except (ImportError, AttributeError) as import_error:
                # Method 2: Try as module import
                logger.warning(f"Direct import failed: {import_error}, trying module import...")
                try:
                    import importlib
                    ccxtpro_module = importlib.import_module('ccxtpro')
                    exchange_class = getattr(ccxtpro_module, self.exchange_id)
                except Exception as module_error:
                    logger.error(f"Module import also failed: {module_error}")
                    raise ValueError(
                        f"ccxtpro.{self.exchange_id} not available. "
                        f"WebSocket disabled, will use REST API only."
                    )

            # Initialize exchange
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'watchOrderBook': {'limit': 1},
                }
            })

            if testnet:
                self.exchange.set_sandbox_mode(True)

            self.is_running = True
            self.is_connected = True
            self.reconnect_attempts = 0

            logger.info(f"✅ WebSocket connection established to {self.exchange_id}")

        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket: {e}")
            self.is_running = False
            self.is_connected = False
            # Don't raise - let REST API take over
            logger.warning("⚠️ WebSocket unavailable, REST API cache will handle all requests")

    async def stop(self):
        """Stop WebSocket connection and cleanup."""
        if not self.is_running:
            return

        self.is_running = False
        self.is_connected = False

        if self.exchange:
            try:
                await self.exchange.close()
                logger.info(f"✅ WebSocket connection closed")
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")

        self.exchange = None
        self.subscribed_symbols.clear()
        self.price_cache.clear()

    async def subscribe_symbols(self, symbols: List[str]):
        """
        Subscribe to price updates for multiple symbols.

        Args:
            symbols: List of trading symbols (e.g., ["BTC/USDT:USDT", "ETH/USDT:USDT"])
        """
        if not self.is_running or not self.exchange:
            logger.warning("WebSocket not running, cannot subscribe")
            return

        new_symbols = set(symbols) - self.subscribed_symbols

        if not new_symbols:
            logger.debug("All symbols already subscribed")
            return

        with self.subscription_lock:
            for symbol in new_symbols:
                try:
                    # Start watching ticker for this symbol
                    asyncio.create_task(self._watch_ticker(symbol))
                    self.subscribed_symbols.add(symbol)
                    logger.info(f"📡 Subscribed to {symbol} price feed")

                except Exception as e:
                    logger.error(f"❌ Failed to subscribe to {symbol}: {e}")

        logger.info(
            f"✅ WebSocket subscriptions active: {len(self.subscribed_symbols)} symbols"
        )

    async def unsubscribe_symbol(self, symbol: str):
        """
        Unsubscribe from price updates for a symbol.

        Args:
            symbol: Trading symbol to unsubscribe
        """
        with self.subscription_lock:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)

                # Remove from cache
                with self.cache_lock:
                    if symbol in self.price_cache:
                        del self.price_cache[symbol]

                logger.info(f"📡 Unsubscribed from {symbol} price feed")

    async def _watch_ticker(self, symbol: str):
        """
        Watch ticker updates for a specific symbol.

        Args:
            symbol: Trading symbol to watch
        """
        while self.is_running and symbol in self.subscribed_symbols:
            try:
                # Watch ticker (blocks until new data arrives)
                ticker = await self.exchange.watch_ticker(symbol)

                if ticker and 'last' in ticker:
                    current_price = Decimal(str(ticker['last']))

                    # Update cache
                    with self.cache_lock:
                        self.price_cache[symbol] = {
                            'price': current_price,
                            'timestamp': datetime.now(),
                            'bid': Decimal(str(ticker.get('bid', 0))),
                            'ask': Decimal(str(ticker.get('ask', 0))),
                            'volume': Decimal(str(ticker.get('quoteVolume', 0)))
                        }

                    logger.debug(
                        f"💹 {symbol} price update: ${current_price:.4f} "
                        f"(bid: ${ticker.get('bid', 0):.4f}, ask: ${ticker.get('ask', 0):.4f})"
                    )

            except Exception as e:
                if self.is_running and symbol in self.subscribed_symbols:
                    logger.error(f"❌ Error watching {symbol} ticker: {e}")

                    # Try to reconnect
                    await self._handle_connection_error()

                    # Wait before retrying
                    await asyncio.sleep(5)
                else:
                    # Symbol was unsubscribed or websocket stopped
                    break

    async def _handle_connection_error(self):
        """Handle WebSocket connection errors with exponential backoff."""
        if not self.is_connected:
            return  # Already handling reconnection

        self.is_connected = False
        self.reconnect_attempts += 1

        if self.reconnect_attempts > self.max_reconnect_attempts:
            logger.error(
                f"❌ Max reconnection attempts ({self.max_reconnect_attempts}) reached. "
                f"WebSocket disabled."
            )
            await self.stop()
            return

        # Exponential backoff
        delay = min(
            self.reconnect_delay_base ** self.reconnect_attempts,
            300  # Max 5 minutes
        )

        logger.warning(
            f"⚠️ WebSocket disconnected. Reconnecting in {delay}s "
            f"(attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})"
        )

        await asyncio.sleep(delay)

        # Try to reconnect
        try:
            if self.exchange:
                await self.exchange.close()

            # Re-initialize exchange (same logic as start())
            if self.exchange_id == 'binance':
                from ccxtpro import binance as BinancePro
                self.exchange = BinancePro({
                    'apiKey': self.exchange.apiKey,
                    'secret': self.exchange.secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'watchOrderBook': {'limit': 1},
                    }
                })
            else:
                exchange_class = getattr(ccxtpro, self.exchange_id)
                self.exchange = exchange_class({
                    'apiKey': self.exchange.apiKey,
                    'secret': self.exchange.secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'watchOrderBook': {'limit': 1},
                    }
                })

            self.is_connected = True
            self.reconnect_attempts = 0

            logger.info(f"✅ WebSocket reconnected successfully")

            # Re-subscribe to all symbols
            symbols_to_resubscribe = list(self.subscribed_symbols)
            self.subscribed_symbols.clear()
            await self.subscribe_symbols(symbols_to_resubscribe)

        except Exception as e:
            logger.error(f"❌ Reconnection failed: {e}")
            await self._handle_connection_error()  # Retry

    def get_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current price for a symbol from cache.

        Args:
            symbol: Trading symbol

        Returns:
            Current price or None if not available/expired
        """
        with self.cache_lock:
            if symbol not in self.price_cache:
                return None

            data = self.price_cache[symbol]
            timestamp = data['timestamp']
            price = data['price']

            # Check if price is still fresh
            age_seconds = (datetime.now() - timestamp).total_seconds()

            if age_seconds > self.price_ttl_seconds:
                logger.debug(
                    f"⏰ Cached price for {symbol} expired "
                    f"({age_seconds:.1f}s > {self.price_ttl_seconds}s)"
                )
                return None

            return price

    def get_price_with_spread(self, symbol: str) -> Optional[Dict]:
        """
        Get price with bid/ask spread.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with 'price', 'bid', 'ask', 'spread_percent', 'age_seconds'
            or None if not available
        """
        with self.cache_lock:
            if symbol not in self.price_cache:
                return None

            data = self.price_cache[symbol]
            timestamp = data['timestamp']

            # Check freshness
            age_seconds = (datetime.now() - timestamp).total_seconds()

            if age_seconds > self.price_ttl_seconds:
                return None

            price = data['price']
            bid = data.get('bid', price)
            ask = data.get('ask', price)

            spread_percent = ((ask - bid) / price * 100) if price > 0 else 0

            return {
                'price': price,
                'bid': bid,
                'ask': ask,
                'spread_percent': spread_percent,
                'age_seconds': age_seconds,
                'volume': data.get('volume', Decimal('0'))
            }

    def is_price_available(self, symbol: str) -> bool:
        """
        Check if fresh price is available for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            True if price is available and fresh
        """
        return self.get_price(symbol) is not None

    def get_stats(self) -> Dict:
        """Get WebSocket feed statistics."""
        with self.cache_lock:
            fresh_prices = 0
            stale_prices = 0

            now = datetime.now()

            for symbol, data in self.price_cache.items():
                age_seconds = (now - data['timestamp']).total_seconds()
                if age_seconds <= self.price_ttl_seconds:
                    fresh_prices += 1
                else:
                    stale_prices += 1

        return {
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'subscribed_symbols': len(self.subscribed_symbols),
            'fresh_prices': fresh_prices,
            'stale_prices': stale_prices,
            'reconnect_attempts': self.reconnect_attempts,
            'price_ttl_seconds': self.price_ttl_seconds
        }


# Singleton instance
_ws_price_feed: Optional[WebSocketPriceFeed] = None


def get_websocket_price_feed() -> WebSocketPriceFeed:
    """Get or create WebSocket price feed instance."""
    global _ws_price_feed
    if _ws_price_feed is None:
        _ws_price_feed = WebSocketPriceFeed()
    return _ws_price_feed
