"""
Database Controller - Handles all database queries for the GUI.

Production-ready features:
- Connection pooling for performance
- Model integration for type safety
- Retry logic for transient errors
- Comprehensive error handling
- Transaction management
- Health monitoring
"""

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
from datetime import date, datetime, timedelta
from decimal import Decimal
import os
import time
import logging
from functools import wraps
from dotenv import load_dotenv

# Import models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.trading_config import TradingConfig
from models.trade import Trade
from models.position import ActivePosition

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry database operations on transient failures."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except psycopg2.OperationalError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"DB operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"DB operation failed after {max_retries} attempts")
                except Exception as e:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise
            raise last_exception
        return wrapper
    return decorator


class DatabaseController:
    """
    Production-ready database controller.

    Features:
    - Connection pooling for high performance
    - Model integration for type safety
    - Automatic retry on transient failures
    - Transaction management
    - Health monitoring
    """

    def __init__(self, min_conn: int = 2, max_conn: int = 10):
        """
        Initialize database controller with connection pooling.

        Args:
            min_conn: Minimum connections in pool
            max_conn: Maximum connections in pool
        """
        self.db_url = os.getenv('DATABASE_URL', 'postgresql://trading_user:changeme123@localhost:5433/trading_bot')
        self.pool: Optional[pool.SimpleConnectionPool] = None
        self.min_conn = min_conn
        self.max_conn = max_conn
        self._connected = False

    def connect(self) -> bool:
        """Initialize connection pool."""
        try:
            self.pool = pool.SimpleConnectionPool(
                self.min_conn,
                self.max_conn,
                self.db_url
            )
            self._connected = True
            logger.info(f"Database pool created ({self.min_conn}-{self.max_conn} connections)")
            return True
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Close all connections in pool."""
        if self.pool:
            self.pool.closeall()
            self._connected = False
            logger.info("Database pool closed")

    def _get_connection(self):
        """Get connection from pool."""
        if not self.pool:
            raise Exception("Database not connected. Call connect() first.")
        return self.pool.getconn()

    def _return_connection(self, conn):
        """Return connection to pool."""
        if self.pool:
            self.pool.putconn(conn)

    def is_healthy(self) -> bool:
        """Check database health."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            self._return_connection(conn)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    # ==================== Model-based methods ====================

    @retry_on_failure(max_retries=3)
    def get_config(self) -> Optional[TradingConfig]:
        """
        Get trading configuration as model instance.

        Returns:
            TradingConfig model or None
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM trading_config WHERE id = 1")
                result = cur.fetchone()
                if result:
                    return TradingConfig.from_dict(dict(result))
                return None
        except Exception as e:
            logger.error(f"Error fetching config: {e}")
            return None
        finally:
            self._return_connection(conn)

    @retry_on_failure(max_retries=3)
    def get_active_position_model(self) -> Optional[ActivePosition]:
        """
        Get active position as model instance.

        Returns:
            ActivePosition model or None
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM active_position LIMIT 1")
                result = cur.fetchone()
                if result:
                    return ActivePosition.from_dict(dict(result))
                return None
        except Exception as e:
            logger.error(f"Error fetching active position: {e}")
            return None
        finally:
            self._return_connection(conn)

    @retry_on_failure(max_retries=3)
    def get_trades(self, limit: int = 100, winners_only: bool = False) -> List[Trade]:
        """
        Get trade history as model instances.

        Args:
            limit: Maximum number of trades to return
            winners_only: Only return winning trades

        Returns:
            List of Trade models
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                    SELECT * FROM trade_history
                    WHERE 1=1
                """
                if winners_only:
                    sql += " AND is_winner = true"

                sql += " ORDER BY exit_time DESC LIMIT %s"

                cur.execute(sql, (limit,))
                return [Trade.from_dict(dict(row)) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []
        finally:
            self._return_connection(conn)

    # ==================== Legacy dict-based methods (for backward compatibility) ====================

    def get_trading_config(self) -> Optional[Dict[str, Any]]:
        """Get current trading configuration (dict format)."""
        config = self.get_config()
        return config.to_dict() if config else None

    def get_current_capital(self) -> float:
        """Get current capital."""
        config = self.get_config()
        return float(config.current_capital) if config else 0.0

    def get_active_position(self) -> Optional[Dict[str, Any]]:
        """Get active position (dict format)."""
        position = self.get_active_position_model()
        return position.to_dict() if position else None

    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trade history (dict format)."""
        trades = self.get_trades(limit=limit)
        return [trade.to_dict() for trade in trades]

    # ==================== Performance and statistics ====================

    @retry_on_failure(max_retries=2)
    def get_daily_performance(self, target_date: date = None) -> Optional[Dict[str, Any]]:
        """Get performance for a specific date."""
        if target_date is None:
            target_date = date.today()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM daily_performance WHERE date = %s
                """, (target_date,))
                result = cur.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error fetching daily performance: {e}")
            return None
        finally:
            self._return_connection(conn)

    @retry_on_failure(max_retries=2)
    def get_daily_pnl(self, target_date: date = None) -> float:
        """Get P&L for a specific date."""
        if target_date is None:
            target_date = date.today()

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(SUM(realized_pnl_usd), 0)
                    FROM trade_history
                    WHERE DATE(exit_time) = %s
                """, (target_date,))
                result = cur.fetchone()
                return float(result[0]) if result else 0.0
        except Exception as e:
            logger.error(f"Error fetching daily P&L: {e}")
            return 0.0
        finally:
            self._return_connection(conn)

    @retry_on_failure(max_retries=2)
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Total trades
                cur.execute("SELECT COUNT(*) as total FROM trade_history")
                total_trades = cur.fetchone()['total']

                # Win rate
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE is_winner = true) as winners,
                        COUNT(*) FILTER (WHERE is_winner = false) as losers
                    FROM trade_history
                """)
                wins_losses = cur.fetchone()
                winners = wins_losses['winners']
                losers = wins_losses['losers']
                win_rate = (winners / total_trades * 100) if total_trades > 0 else 0.0

                # Total P&L
                cur.execute("SELECT COALESCE(SUM(realized_pnl_usd), 0) as total_pnl FROM trade_history")
                total_pnl = float(cur.fetchone()['total_pnl'])

                # Average trade
                cur.execute("""
                    SELECT
                        COALESCE(AVG(realized_pnl_usd) FILTER (WHERE is_winner = true), 0) as avg_win,
                        COALESCE(AVG(realized_pnl_usd) FILTER (WHERE is_winner = false), 0) as avg_loss
                    FROM trade_history
                """)
                avg_stats = cur.fetchone()
                avg_win = float(avg_stats['avg_win'])
                avg_loss = abs(float(avg_stats['avg_loss']))

                # Best/worst trades
                cur.execute("""
                    SELECT MAX(realized_pnl_usd) as best, MIN(realized_pnl_usd) as worst
                    FROM trade_history
                """)
                extremes = cur.fetchone()
                best_trade = float(extremes['best']) if extremes['best'] else 0.0
                worst_trade = float(extremes['worst']) if extremes['worst'] else 0.0

                # Profit factor
                total_wins = winners * avg_win if winners > 0 else 0
                total_losses = abs(losers * avg_loss) if losers > 0 else 0
                profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

                return {
                    'total_trades': total_trades,
                    'winners': winners,
                    'losers': losers,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'best_trade': best_trade,
                    'worst_trade': worst_trade,
                    'profit_factor': profit_factor
                }
        except Exception as e:
            logger.error(f"Error fetching statistics: {e}")
            return {
                'total_trades': 0,
                'winners': 0,
                'losers': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
                'profit_factor': 0.0
            }
        finally:
            self._return_connection(conn)

    @retry_on_failure(max_retries=2)
    def get_pnl_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get P&L history for the last N days."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT date, daily_pnl, ending_capital
                    FROM daily_performance
                    ORDER BY date DESC
                    LIMIT %s
                """, (days,))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching P&L history: {e}")
            return []
        finally:
            self._return_connection(conn)

    # ==================== Update methods ====================

    @retry_on_failure(max_retries=3)
    def update_trading_enabled(self, enabled: bool) -> bool:
        """Enable or disable trading."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE trading_config
                    SET is_trading_enabled = %s, last_updated = NOW()
                    WHERE id = 1
                """, (enabled,))
                conn.commit()
                logger.info(f"Trading {'enabled' if enabled else 'disabled'}")
                return True
        except Exception as e:
            logger.error(f"Error updating trading status: {e}")
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)

    @retry_on_failure(max_retries=3)
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """Update trading configuration with validation."""
        conn = self._get_connection()
        try:
            # Validate using TradingConfig model first
            current_config = self.get_config()
            if not current_config:
                logger.error("Cannot update config - no existing config found")
                return False

            # Merge updates
            config_dict = current_config.to_dict()
            config_dict.update(config_updates)

            # Validate merged config
            try:
                TradingConfig.from_dict(config_dict)
            except ValueError as e:
                logger.error(f"Config validation failed: {e}")
                return False

            # Build UPDATE statement dynamically
            fields = []
            values = []

            for key, value in config_updates.items():
                fields.append(f"{key} = %s")
                values.append(value)

            if not fields:
                return False

            fields.append("last_updated = NOW()")

            sql = f"UPDATE trading_config SET {', '.join(fields)} WHERE id = 1"

            with conn.cursor() as cur:
                cur.execute(sql, values)
                conn.commit()
                logger.info(f"Config updated: {list(config_updates.keys())}")
                return True

        except Exception as e:
            logger.error(f"Error updating config: {e}")
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)

    @retry_on_failure(max_retries=3)
    def save_config(self, config: TradingConfig) -> bool:
        """
        Save complete TradingConfig model to database.

        Args:
            config: TradingConfig model instance

        Returns:
            True if successful
        """
        conn = self._get_connection()
        try:
            # Validate first
            config.validate()

            config_dict = config.to_dict()
            config_dict.pop('id', None)  # Don't update ID

            # Build UPDATE statement
            fields = [f"{key} = %s" for key in config_dict.keys()]
            fields.append("last_updated = NOW()")
            values = list(config_dict.values())

            sql = f"UPDATE trading_config SET {', '.join(fields)} WHERE id = 1"

            with conn.cursor() as cur:
                cur.execute(sql, values)
                conn.commit()
                logger.info("Complete config saved successfully")
                return True

        except Exception as e:
            logger.error(f"Error saving config: {e}")
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)

    # ==================== Advanced features ====================

    def get_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """
        Get performance summary for recent days.

        Args:
            days: Number of days to analyze

        Returns:
            Performance metrics
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Calculate date range
                end_date = date.today()
                start_date = end_date - timedelta(days=days - 1)

                # Get trades in date range
                cur.execute("""
                    SELECT
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE is_winner = true) as winners,
                        COALESCE(SUM(realized_pnl_usd), 0) as total_pnl,
                        COALESCE(AVG(realized_pnl_usd), 0) as avg_pnl
                    FROM trade_history
                    WHERE DATE(exit_time) BETWEEN %s AND %s
                """, (start_date, end_date))

                result = cur.fetchone()

                total_trades = result['total_trades']
                win_rate = (result['winners'] / total_trades * 100) if total_trades > 0 else 0.0

                return {
                    'period_days': days,
                    'total_trades': total_trades,
                    'winners': result['winners'],
                    'win_rate': win_rate,
                    'total_pnl': float(result['total_pnl']),
                    'avg_pnl': float(result['avg_pnl'])
                }

        except Exception as e:
            logger.error(f"Error getting recent performance: {e}")
            return {
                'period_days': days,
                'total_trades': 0,
                'winners': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0
            }
        finally:
            self._return_connection(conn)

    def execute_transaction(self, operations: List[tuple]) -> bool:
        """
        Execute multiple operations in a single transaction.

        Args:
            operations: List of (sql, params) tuples

        Returns:
            True if all operations succeeded
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                for sql, params in operations:
                    cur.execute(sql, params)
                conn.commit()
                logger.info(f"Transaction completed: {len(operations)} operations")
                return True
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)
