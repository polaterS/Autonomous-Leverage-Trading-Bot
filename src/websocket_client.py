"""
WebSocket client for real-time price feeds from Binance Futures.
Much more efficient than REST API polling for price monitoring.
"""

import asyncio
import json
from typing import Dict, Any, Optional, Callable, Set
from decimal import Decimal
from src.config import get_settings
from src.utils import setup_logging
from src.redis_client import get_redis_client

logger = setup_logging()


class WebSocketPriceClient:
    """
    Real-time price feed via Binance WebSocket.
    Subscribes to price updates for active positions and scanned symbols.
    """

    def __init__(self):
        self.settings = get_settings()
        self.ws_url = "wss://fstream.binance.com/ws"
        self.websocket = None
        self.subscribed_symbols: Set[str] = set()
        self.price_cache: Dict[str, Decimal] = {}
        self.callbacks: Dict[str, list] = {}  # symbol -> list of callbacks
        self.is_running = False

    async def connect(self):
        """Connect to Binance WebSocket (will be implemented with actual WS library)."""
        try:
            # Note: This is a simplified version. In production, use 'websockets' library
            # pip install websockets
            import websockets

            self.websocket = await websockets.connect(self.ws_url)
            self.is_running = True
            logger.info("✅ WebSocket connected to Binance")

            # Start message handler
            asyncio.create_task(self._handle_messages())

        except ImportError:
            logger.warning(
                "WebSockets library not installed. WebSocket features disabled. "
                "Install with: pip install websockets"
            )
            self.websocket = None

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self.websocket = None

    async def close(self):
        """Close WebSocket connection."""
        self.is_running = False
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed")

    async def subscribe_symbol(self, symbol: str, callback: Optional[Callable] = None):
        """
        Subscribe to real-time price updates for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            callback: Optional callback function to call on price update
        """
        if not self.websocket:
            logger.warning("WebSocket not connected, cannot subscribe")
            return

        # Convert CCXT format to Binance format
        # 'BTC/USDT:USDT' -> 'btcusdt'
        binance_symbol = symbol.split('/')[0].lower() + symbol.split('/')[1].split(':')[0].lower()

        if binance_symbol not in self.subscribed_symbols:
            # Subscribe to ticker stream
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": [f"{binance_symbol}@ticker"],
                "id": len(self.subscribed_symbols) + 1
            }

            try:
                await self.websocket.send(json.dumps(subscribe_message))
                self.subscribed_symbols.add(binance_symbol)
                logger.info(f"📊 Subscribed to WebSocket: {symbol}")

            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol}: {e}")

        # Register callback
        if callback:
            if symbol not in self.callbacks:
                self.callbacks[symbol] = []
            self.callbacks[symbol].append(callback)

    async def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from a symbol."""
        if not self.websocket:
            return

        binance_symbol = symbol.split('/')[0].lower() + symbol.split('/')[1].split(':')[0].lower()

        if binance_symbol in self.subscribed_symbols:
            unsubscribe_message = {
                "method": "UNSUBSCRIBE",
                "params": [f"{binance_symbol}@ticker"],
                "id": 999
            }

            try:
                await self.websocket.send(json.dumps(unsubscribe_message))
                self.subscribed_symbols.remove(binance_symbol)
                logger.info(f"🔕 Unsubscribed from WebSocket: {symbol}")

            except Exception as e:
                logger.error(f"Failed to unsubscribe from {symbol}: {e}")

        # Remove callbacks
        if symbol in self.callbacks:
            del self.callbacks[symbol]

    async def get_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get cached price from WebSocket feed.
        Falls back to REST API if WebSocket not available.

        Args:
            symbol: Trading symbol

        Returns:
            Current price or None
        """
        # Try cache first
        if symbol in self.price_cache:
            return self.price_cache[symbol]

        # If WebSocket not available, return None (caller will use REST API)
        return None

    async def _handle_messages(self):
        """Handle incoming WebSocket messages."""
        if not self.websocket:
            return

        logger.info("WebSocket message handler started")

        while self.is_running:
            try:
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=30.0  # 30s timeout
                )

                data = json.loads(message)

                # Handle ticker updates
                if 'e' in data and data['e'] == '24hrTicker':
                    await self._handle_ticker_update(data)

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await self.websocket.ping()
                except:
                    logger.warning("WebSocket ping failed, reconnecting...")
                    await self._reconnect()

            except Exception as e:
                logger.error(f"WebSocket message handler error: {e}")
                await asyncio.sleep(5)

    async def _handle_ticker_update(self, data: Dict[str, Any]):
        """Process ticker update from WebSocket."""
        try:
            binance_symbol = data['s'].lower()  # e.g., 'btcusdt'
            price = Decimal(data['c'])  # Current close price

            # Convert back to CCXT format for consistency
            # This is simplified - in production, maintain a symbol mapping
            base = binance_symbol[:-4].upper()  # 'BTC'
            quote = 'USDT'
            symbol = f"{base}/{quote}:{quote}"

            # Update cache
            self.price_cache[symbol] = price

            # Cache in Redis for other components
            redis = await get_redis_client()
            await redis.cache_market_data(
                symbol,
                {'price': float(price), 'timestamp': data.get('E')},
                ttl_seconds=10  # Very short TTL for real-time data
            )

            # Call registered callbacks
            if symbol in self.callbacks:
                for callback in self.callbacks[symbol]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(symbol, price)
                        else:
                            callback(symbol, price)
                    except Exception as e:
                        logger.error(f"Callback error for {symbol}: {e}")

            logger.debug(f"📊 WS Price update: {symbol} = ${float(price):.4f}")

        except Exception as e:
            logger.error(f"Failed to process ticker update: {e}")

    async def _reconnect(self):
        """Reconnect WebSocket."""
        logger.warning("Reconnecting WebSocket...")

        try:
            if self.websocket:
                await self.websocket.close()

            await asyncio.sleep(5)
            await self.connect()

            # Resubscribe to all symbols
            symbols_to_resubscribe = list(self.subscribed_symbols)
            self.subscribed_symbols.clear()

            for binance_symbol in symbols_to_resubscribe:
                # Convert back to CCXT format
                base = binance_symbol[:-4].upper()
                symbol = f"{base}/USDT:USDT"
                await self.subscribe_symbol(symbol)

            logger.info("✅ WebSocket reconnected and resubscribed")

        except Exception as e:
            logger.error(f"WebSocket reconnection failed: {e}")


# Singleton instance
_ws_client: Optional[WebSocketPriceClient] = None


async def get_ws_client() -> WebSocketPriceClient:
    """Get or create WebSocket client instance."""
    global _ws_client
    if _ws_client is None:
        _ws_client = WebSocketPriceClient()
        await _ws_client.connect()
    return _ws_client
