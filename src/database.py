"""
Database client for the Autonomous Leverage Trading Bot.
Handles all database operations with connection pooling.
"""

import asyncpg
import asyncio
import json
import numpy as np
from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime, date
from src.config import get_settings
from src.utils import setup_logging

logger = setup_logging()


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize objects for JSON serialization.
    Converts NumPy types (bool_, int64, float64, etc.) to native Python types.

    This prevents "Object of type bool_ is not JSON serializable" errors
    when saving ML predictions and technical indicators to the database.
    """
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


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

    async def sync_config_from_env(self) -> None:
        """
        Sync trading configuration from environment settings to database.
        Used to update database when config.py values change.
        """
        from src.config import get_settings
        settings = get_settings()

        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE trading_config
                SET min_stop_loss_percent = $1,
                    max_stop_loss_percent = $2,
                    max_leverage = $3,
                    min_ai_confidence = $4,
                    position_size_percent = $5,
                    daily_loss_limit_percent = $6,
                    max_consecutive_losses = $7,
                    last_updated = NOW()
                WHERE id = 1
            """,
                settings.min_stop_loss_percent,
                settings.max_stop_loss_percent,
                settings.max_leverage,
                settings.min_ai_confidence,
                settings.position_size_percent,
                settings.daily_loss_limit_percent,
                settings.max_consecutive_losses
            )

        logger.info(
            f"✅ Config synced from environment: "
            f"leverage={settings.max_leverage}x, "
            f"stop-loss={float(settings.min_stop_loss_percent)}-{float(settings.max_stop_loss_percent)}%"
        )

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
        """
        Create a new active position (supports multiple concurrent positions).

        ENHANCED: Retry logic to prevent position tracking loss on DB failures.
        """
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                return await self._create_active_position_internal(position_data)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"⚠️ Database write failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed - CRITICAL!
                    logger.critical(
                        f"🚨 CRITICAL: Failed to save position after {max_retries} attempts!"
                    )
                    logger.critical(f"Position data: {position_data['symbol']} {position_data['side']}")

                    # Send emergency alert
                    from src.telegram_notifier import get_notifier
                    notifier = get_notifier()
                    await notifier.send_alert(
                        'critical',
                        f"🚨 <b>DATABASE WRITE FAILURE</b>\n\n"
                        f"Could NOT save position to database after {max_retries} attempts:\n\n"
                        f"<b>{position_data['symbol']}</b> {position_data['side']}\n"
                        f"Entry: ${float(position_data['entry_price']):.4f}\n\n"
                        f"⚠️ Position is OPEN on Binance but NOT tracked!\n"
                        f"Will be auto-recovered on next reconciliation cycle."
                    )

                    raise  # Re-raise to trigger position reconciliation

    async def _create_active_position_internal(self, position_data: Dict[str, Any]) -> int:
        """Internal method for actual database write."""
        async with self.pool.acquire() as conn:
            # Multi-position support: Don't delete existing positions
            # Just add the new one

            # Convert snapshot dict to JSON string for JSONB (sanitize NumPy types first)
            entry_snapshot = position_data.get('entry_snapshot')
            entry_snapshot_json = json.dumps(sanitize_for_json(entry_snapshot)) if entry_snapshot else None

            row = await conn.fetchrow("""
                INSERT INTO active_position (
                    symbol, side, leverage, entry_price, current_price, quantity,
                    position_value_usd, stop_loss_price, stop_loss_percent,
                    min_profit_target_usd, min_profit_price, liquidation_price,
                    exchange_order_id, stop_loss_order_id, ai_model_consensus, ai_confidence, ai_reasoning,
                    entry_snapshot, entry_slippage_percent, entry_fill_time_ms,
                    profit_target_1, profit_target_2, partial_exit_done, partial_exit_profit
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
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
                position_data.get('entry_fill_time_ms'),
                position_data.get('profit_target_1'),
                position_data.get('profit_target_2'),
                position_data.get('partial_exit_done', False),
                position_data.get('partial_exit_profit', Decimal("0"))
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
        """Record completed trade in history with ML learning snapshots and TIER 1 & 2 metrics."""
        async with self.pool.acquire() as conn:
            # Convert snapshot dicts to JSON strings for JSONB (sanitize NumPy types first)
            entry_snapshot = trade_data.get('entry_snapshot')
            exit_snapshot = trade_data.get('exit_snapshot')
            partial_exit_details = trade_data.get('partial_exit_details')

            entry_snapshot_json = json.dumps(sanitize_for_json(entry_snapshot)) if entry_snapshot else None
            exit_snapshot_json = json.dumps(sanitize_for_json(exit_snapshot)) if exit_snapshot else None
            partial_exit_details_json = json.dumps(sanitize_for_json(partial_exit_details)) if partial_exit_details else None

            row = await conn.fetchrow("""
                INSERT INTO trade_history (
                    symbol, side, leverage, entry_price, exit_price, quantity,
                    position_value_usd, realized_pnl_usd, pnl_percent,
                    stop_loss_percent, close_reason, trade_duration_seconds,
                    ai_model_consensus, ai_confidence, ai_reasoning, entry_time, is_winner,
                    entry_snapshot, exit_snapshot,
                    entry_slippage_percent, exit_slippage_percent,
                    entry_fill_time_ms, exit_fill_time_ms,
                    max_profit_percent_achieved, trailing_stop_triggered,
                    had_partial_exits, partial_exit_details,
                    entry_market_regime, exit_market_regime
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29)
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
                trade_data.get('exit_fill_time_ms'),
                # 🎯 TIER 1 & 2 METRICS
                trade_data.get('max_profit_percent_achieved', 0.0),
                trade_data.get('trailing_stop_triggered', False),
                trade_data.get('had_partial_exits', False),
                partial_exit_details_json,
                trade_data.get('entry_market_regime'),
                trade_data.get('exit_market_regime')
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

    async def get_last_closed_time(self, symbol: str) -> Optional[datetime]:
        """
        Get the most recent exit time for a given symbol.
        Used for cooldown period enforcement (prevent re-trading same symbol too soon).

        Returns:
            datetime: Most recent exit_time for this symbol, or None if never traded
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT exit_time
                FROM trade_history
                WHERE symbol = $1
                ORDER BY exit_time DESC
                LIMIT 1
            """, symbol)
            return row['exit_time'] if row else None

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
                symbol, ai_model, timeframe, json.dumps(sanitize_for_json(analysis_json)),
                analysis_json.get('confidence'), analysis_json.get('action'), expires_at
            )

    async def cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM ai_analysis_cache WHERE expires_at < NOW()
            """)
            logger.debug(f"Cleaned up expired AI cache entries")

    # 🚨 BUG FIX #1: Recent Trades Check (Prevent duplicate entries to same symbol)
    async def get_recent_trades_for_symbol(self, symbol: str, minutes: int = 30) -> List[Dict[str, Any]]:
        """
        Get trades for a symbol that closed within the last N minutes.

        Used to prevent re-entry into recently closed positions.
        This prevents the bot from:
        - Re-entering a losing trade immediately after exit
        - Trading the same symbol in quick succession
        - Over-concentration in a single asset

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            minutes: Lookback period in minutes (default: 30)

        Returns:
            List of recent trade dictionaries, sorted by exit_time DESC

        Example:
            recent = await db.get_recent_trades_for_symbol('SKL/USDT:USDT', 30)
            if recent:
                # SKL was traded in last 30 minutes - don't re-enter!
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    id, symbol, side, entry_price, exit_price,
                    quantity, realized_pnl_usd, entry_time, exit_time,
                    EXTRACT(EPOCH FROM (NOW() - exit_time)) / 60 as minutes_ago
                FROM trade_history
                WHERE symbol = $1
                  AND exit_time IS NOT NULL
                  AND exit_time >= NOW() - INTERVAL '1 minute' * $2
                ORDER BY exit_time DESC
            """, symbol, minutes)

            return [dict(row) for row in rows]


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
