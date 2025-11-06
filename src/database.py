"""
Database client for the Autonomous Leverage Trading Bot.
Handles all database operations with connection pooling.
"""

import asyncpg
import asyncio
from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime, date
from src.config import get_settings
from src.utils import setup_logging

logger = setup_logging()


class DatabaseClient:
    """Async database client with connection pooling."""

    def __init__(self):
        self.settings = get_settings()
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Create database connection pool."""
        try:
            # Railway requires SSL for public connections
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE  # For Railway public URL

            self.pool = await asyncpg.create_pool(
                self.settings.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60,
                ssl=ssl_context  # Add SSL support for Railway
            )
            logger.info("Database connection pool created")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    # Trading Configuration Methods

    async def get_trading_config(self) -> Dict[str, Any]:
        """Get current trading configuration."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM trading_config WHERE id = 1")
            return dict(row) if row else {}

    async def update_capital(self, new_capital: Decimal) -> None:
        """Update current capital."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE trading_config SET current_capital = $1, last_updated = NOW() WHERE id = 1",
                new_capital
            )
        logger.info(f"Capital updated to ${new_capital:.2f}")

    async def get_current_capital(self) -> Decimal:
        """Get current capital."""
        config = await self.get_trading_config()
        return Decimal(str(config.get('current_capital', 0)))

    async def set_trading_enabled(self, enabled: bool) -> None:
        """Enable or disable trading."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE trading_config SET is_trading_enabled = $1, last_updated = NOW() WHERE id = 1",
                enabled
            )
        logger.info(f"Trading {'enabled' if enabled else 'disabled'}")

    # Active Position Methods

    async def get_active_position(self) -> Optional[Dict[str, Any]]:
        """Get current active position if any (for backward compatibility)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM active_position LIMIT 1")
            return dict(row) if row else None

    async def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions (multi-position support)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM active_position ORDER BY entry_time DESC")
            return [dict(row) for row in rows] if rows else []

    async def create_active_position(self, position_data: Dict[str, Any]) -> int:
        """Create a new active position (supports multiple concurrent positions)."""
        async with self.pool.acquire() as conn:
            # Multi-position support: Don't delete existing positions
            # Just add the new one

            # Convert snapshot dict to JSON string for JSONB
            import json
            entry_snapshot_json = json.dumps(position_data.get('entry_snapshot')) if position_data.get('entry_snapshot') else None

            row = await conn.fetchrow("""
                INSERT INTO active_position (
                    symbol, side, leverage, entry_price, current_price, quantity,
                    position_value_usd, stop_loss_price, stop_loss_percent,
                    min_profit_target_usd, min_profit_price, liquidation_price,
                    exchange_order_id, stop_loss_order_id, ai_model_consensus, ai_confidence, ai_reasoning,
                    entry_snapshot, entry_slippage_percent, entry_fill_time_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                RETURNING id
            """,
                position_data['symbol'],
                position_data['side'],
                position_data['leverage'],
                position_data['entry_price'],
                position_data['current_price'],
                position_data['quantity'],
                position_data['position_value_usd'],
                position_data['stop_loss_price'],
                position_data['stop_loss_percent'],
                position_data['min_profit_target_usd'],
                position_data['min_profit_price'],
                position_data['liquidation_price'],
                position_data.get('exchange_order_id'),
                position_data.get('stop_loss_order_id'),
                position_data.get('ai_model_consensus'),
                position_data.get('ai_confidence'),
                position_data.get('ai_reasoning', ''),
                entry_snapshot_json,
                position_data.get('entry_slippage_percent'),
                position_data.get('entry_fill_time_ms')
            )

        position_id = row['id']
        logger.info(f"Active position created: {position_data['symbol']} {position_data['side']}")
        return position_id

    async def update_position_price(self, position_id: int, current_price: Decimal, unrealized_pnl: Decimal) -> None:
        """Update position with current price and P&L."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE active_position
                SET current_price = $1, unrealized_pnl_usd = $2, last_check_time = NOW()
                WHERE id = $3
            """, current_price, unrealized_pnl, position_id)

    async def remove_active_position(self, position_id: int) -> None:
        """Remove active position."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM active_position WHERE id = $1", position_id)
        logger.info(f"Active position {position_id} removed")

    # Trade History Methods

    async def record_trade(self, trade_data: Dict[str, Any]) -> int:
        """Record completed trade in history with ML learning snapshots."""
        async with self.pool.acquire() as conn:
            # Convert snapshot dicts to JSON strings for JSONB
            import json
            entry_snapshot_json = json.dumps(trade_data.get('entry_snapshot')) if trade_data.get('entry_snapshot') else None
            exit_snapshot_json = json.dumps(trade_data.get('exit_snapshot')) if trade_data.get('exit_snapshot') else None

            row = await conn.fetchrow("""
                INSERT INTO trade_history (
                    symbol, side, leverage, entry_price, exit_price, quantity,
                    position_value_usd, realized_pnl_usd, pnl_percent,
                    stop_loss_percent, close_reason, trade_duration_seconds,
                    ai_model_consensus, ai_confidence, ai_reasoning, entry_time, is_winner,
                    entry_snapshot, exit_snapshot,
                    entry_slippage_percent, exit_slippage_percent,
                    entry_fill_time_ms, exit_fill_time_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23)
                RETURNING id
            """,
                trade_data['symbol'],
                trade_data['side'],
                trade_data['leverage'],
                trade_data['entry_price'],
                trade_data['exit_price'],
                trade_data['quantity'],
                trade_data['position_value_usd'],
                trade_data['realized_pnl_usd'],
                trade_data['pnl_percent'],
                trade_data['stop_loss_percent'],
                trade_data['close_reason'],
                trade_data['trade_duration_seconds'],
                trade_data.get('ai_model_consensus'),
                trade_data.get('ai_confidence'),
                trade_data.get('ai_reasoning', ''),
                trade_data['entry_time'],
                trade_data['realized_pnl_usd'] > 0,
                entry_snapshot_json,
                exit_snapshot_json,
                trade_data.get('entry_slippage_percent'),
                trade_data.get('exit_slippage_percent'),
                trade_data.get('entry_fill_time_ms'),
                trade_data.get('exit_fill_time_ms')
            )

        trade_id = row['id']
        logger.info(f"Trade recorded: {trade_data['symbol']} P&L: ${trade_data['realized_pnl_usd']:.2f}")
        return trade_id

    async def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trade history."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM trade_history
                ORDER BY exit_time DESC
                LIMIT $1
            """, limit)
            return [dict(row) for row in rows]

    async def get_trade_history(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get trade history for ML learning.
        Alias for get_recent_trades with larger default limit.
        """
        return await self.get_recent_trades(limit=limit)

    async def get_consecutive_losses(self) -> int:
        """Count consecutive losing trades."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT is_winner FROM trade_history
                ORDER BY exit_time DESC
                LIMIT 10
            """)

            consecutive = 0
            for row in rows:
                if not row['is_winner']:
                    consecutive += 1
                else:
                    break

            return consecutive

    # Daily Performance Methods

    async def get_daily_performance(self, target_date: date) -> Optional[Dict[str, Any]]:
        """Get performance for a specific date."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM daily_performance WHERE date = $1
            """, target_date)
            return dict(row) if row else None

    async def update_daily_performance(self, target_date: date) -> None:
        """Calculate and update daily performance metrics."""
        async with self.pool.acquire() as conn:
            # Get trades for the day
            trades = await conn.fetch("""
                SELECT * FROM trade_history
                WHERE DATE(exit_time) = $1
            """, target_date)

            if not trades:
                return

            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['is_winner'])
            losing_trades = total_trades - winning_trades
            daily_pnl = sum(Decimal(str(t['realized_pnl_usd'])) for t in trades)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            largest_win = max((Decimal(str(t['realized_pnl_usd'])) for t in trades if t['is_winner']), default=Decimal("0"))
            largest_loss = min((Decimal(str(t['realized_pnl_usd'])) for t in trades if not t['is_winner']), default=Decimal("0"))

            # Get starting capital (ending capital from previous day or initial)
            prev_day = await conn.fetchrow("""
                SELECT ending_capital FROM daily_performance
                WHERE date < $1
                ORDER BY date DESC
                LIMIT 1
            """, target_date)

            if prev_day:
                starting_capital = Decimal(str(prev_day['ending_capital']))
            else:
                config = await conn.fetchrow("SELECT initial_capital FROM trading_config WHERE id = 1")
                starting_capital = Decimal(str(config['initial_capital']))

            ending_capital = starting_capital + daily_pnl

            # Upsert daily performance
            await conn.execute("""
                INSERT INTO daily_performance (
                    date, starting_capital, ending_capital, daily_pnl,
                    total_trades, winning_trades, losing_trades, win_rate,
                    largest_win, largest_loss
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (date) DO UPDATE SET
                    ending_capital = EXCLUDED.ending_capital,
                    daily_pnl = EXCLUDED.daily_pnl,
                    total_trades = EXCLUDED.total_trades,
                    winning_trades = EXCLUDED.winning_trades,
                    losing_trades = EXCLUDED.losing_trades,
                    win_rate = EXCLUDED.win_rate,
                    largest_win = EXCLUDED.largest_win,
                    largest_loss = EXCLUDED.largest_loss
            """,
                target_date, starting_capital, ending_capital, daily_pnl,
                total_trades, winning_trades, losing_trades, win_rate,
                largest_win, largest_loss
            )

        logger.info(f"Daily performance updated for {target_date}")

    async def get_daily_pnl(self, target_date: date = None) -> Decimal:
        """Get P&L for a specific date (defaults to today)."""
        if target_date is None:
            target_date = date.today()

        async with self.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COALESCE(SUM(realized_pnl_usd), 0)
                FROM trade_history
                WHERE DATE(exit_time) = $1
            """, target_date)

            return Decimal(str(result)) if result else Decimal("0")

    # System Logs Methods

    async def log_event(self, level: str, component: str, message: str, details: Dict[str, Any] = None) -> None:
        """Log system event to database."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO system_logs (log_level, component, message, details)
                VALUES ($1, $2, $3, $4)
            """, level, component, message, details)

    # Circuit Breaker Methods

    async def record_circuit_breaker(self, event_type: str, trigger_value: Decimal, threshold_value: Decimal, action_taken: str) -> int:
        """Record circuit breaker activation."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO circuit_breaker_events (event_type, trigger_value, threshold_value, action_taken)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, event_type, trigger_value, threshold_value, action_taken)

        logger.warning(f"Circuit breaker activated: {event_type}")
        return row['id']

    # AI Cache Methods

    async def get_ai_cache(self, symbol: str, ai_model: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get cached AI analysis if valid."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM ai_analysis_cache
                WHERE symbol = $1 AND ai_model = $2 AND timeframe = $3
                  AND expires_at > NOW()
                ORDER BY created_at DESC
                LIMIT 1
            """, symbol, ai_model, timeframe)

            return dict(row) if row else None

    async def set_ai_cache(self, symbol: str, ai_model: str, timeframe: str, analysis_json: Dict[str, Any], ttl_seconds: int) -> None:
        """Cache AI analysis result."""
        from datetime import timedelta
        import json

        async with self.pool.acquire() as conn:
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

            await conn.execute("""
                INSERT INTO ai_analysis_cache (symbol, ai_model, timeframe, analysis_json, confidence, action, expires_at)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
            """,
                symbol, ai_model, timeframe, json.dumps(analysis_json),
                analysis_json.get('confidence'), analysis_json.get('action'), expires_at
            )

    async def cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM ai_analysis_cache WHERE expires_at < NOW()
            """)
            logger.debug(f"Cleaned up expired AI cache entries")


# Singleton instance
_db_client: Optional[DatabaseClient] = None


async def get_db_client() -> DatabaseClient:
    """Get or create database client instance."""
    global _db_client
    if _db_client is None or _db_client.pool is None:
        _db_client = DatabaseClient()
        try:
            await _db_client.connect()
        except Exception as e:
            # Reset singleton on connection failure
            _db_client = None
            raise
    return _db_client
