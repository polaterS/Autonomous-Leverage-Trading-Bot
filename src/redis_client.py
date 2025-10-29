"""
Redis cache client for high-performance caching.
Used for AI analysis results, market data, and temporary state.
"""

import redis.asyncio as redis
import json
from typing import Optional, Dict, Any
from datetime import timedelta
from src.config import get_settings
from src.utils import setup_logging

logger = setup_logging()


class RedisClient:
    """Async Redis client for caching."""

    def __init__(self):
        self.settings = get_settings()
        self.redis: Optional[redis.Redis] = None

    async def connect(self):
        """Connect to Redis server."""
        try:
            self.redis = await redis.from_url(
                self.settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                socket_connect_timeout=5,
                socket_keepalive=True
            )

            # Test connection
            await self.redis.ping()
            logger.info("✅ Redis connected")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Don't raise - gracefully degrade to no caching
            self.redis = None

    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 300
    ) -> bool:
        """
        Set a value in Redis with TTL.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl_seconds: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            return False

        try:
            json_value = json.dumps(value)
            await self.redis.setex(key, timedelta(seconds=ttl_seconds), json_value)
            logger.debug(f"Cache SET: {key} (TTL: {ttl_seconds}s)")
            return True

        except Exception as e:
            logger.warning(f"Redis SET error for {key}: {e}")
            return False

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from Redis.

        Args:
            key: Cache key

        Returns:
            Cached value (JSON deserialized) or None
        """
        if not self.redis:
            return None

        try:
            value = await self.redis.get(key)

            if value is None:
                logger.debug(f"Cache MISS: {key}")
                return None

            logger.debug(f"Cache HIT: {key}")
            return json.loads(value)

        except Exception as e:
            logger.warning(f"Redis GET error for {key}: {e}")
            return None

    async def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        if not self.redis:
            return False

        try:
            await self.redis.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return True

        except Exception as e:
            logger.warning(f"Redis DELETE error for {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        if not self.redis:
            return False

        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.warning(f"Redis EXISTS error for {key}: {e}")
            return False

    async def set_hash(
        self,
        key: str,
        field: str,
        value: Any,
        ttl_seconds: int = 300
    ) -> bool:
        """
        Set a field in a Redis hash.

        Args:
            key: Hash key
            field: Field name
            value: Value to cache
            ttl_seconds: TTL for the entire hash

        Returns:
            True if successful
        """
        if not self.redis:
            return False

        try:
            json_value = json.dumps(value)
            await self.redis.hset(key, field, json_value)
            await self.redis.expire(key, ttl_seconds)
            return True

        except Exception as e:
            logger.warning(f"Redis HSET error for {key}.{field}: {e}")
            return False

    async def get_hash(self, key: str, field: str) -> Optional[Any]:
        """Get a field from a Redis hash."""
        if not self.redis:
            return None

        try:
            value = await self.redis.hget(key, field)

            if value is None:
                return None

            return json.loads(value)

        except Exception as e:
            logger.warning(f"Redis HGET error for {key}.{field}: {e}")
            return None

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter."""
        if not self.redis:
            return None

        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.warning(f"Redis INCR error for {key}: {e}")
            return None

    async def flush_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Args:
            pattern: Pattern to match (e.g., "ai_cache:*")

        Returns:
            Number of keys deleted
        """
        if not self.redis:
            return 0

        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info(f"Flushed {deleted} keys matching '{pattern}'")
                return deleted

            return 0

        except Exception as e:
            logger.warning(f"Redis flush pattern error for '{pattern}': {e}")
            return 0

    # Specialized caching methods

    async def cache_ai_analysis(
        self,
        symbol: str,
        timeframe: str,
        analysis: Dict[str, Any],
        ttl_seconds: int = 300
    ) -> bool:
        """
        Cache AI analysis result.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '5m', '1h')
            analysis: Analysis dict
            ttl_seconds: Cache TTL

        Returns:
            True if cached successfully
        """
        key = f"ai_cache:{symbol}:{timeframe}"
        return await self.set(key, analysis, ttl_seconds)

    async def get_cached_ai_analysis(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached AI analysis.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Cached analysis or None
        """
        key = f"ai_cache:{symbol}:{timeframe}"
        return await self.get(key)

    async def cache_market_data(
        self,
        symbol: str,
        data: Dict[str, Any],
        ttl_seconds: int = 60
    ) -> bool:
        """Cache market data (price, indicators, etc)."""
        key = f"market:{symbol}"
        return await self.set(key, data, ttl_seconds)

    async def get_cached_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data."""
        key = f"market:{symbol}"
        return await self.get(key)


# Singleton instance
_redis_client: Optional[RedisClient] = None


async def get_redis_client() -> RedisClient:
    """Get or create Redis client instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
        await _redis_client.connect()
    return _redis_client
