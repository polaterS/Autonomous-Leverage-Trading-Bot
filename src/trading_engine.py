"""
Autonomous Trading Engine - Main orchestrator for the trading bot.
Manages the infinite loop, coordinates all components, handles errors.
"""

import asyncio
from datetime import datetime, date
from decimal import Decimal
from typing import Optional
from src.config import get_settings
from src.database import get_db_client
from src.exchange_client import get_exchange_client
from src.market_scanner import get_market_scanner
from src.position_monitor import get_position_monitor
from src.risk_manager import get_risk_manager
from src.telegram_notifier import get_notifier
from src.utils import setup_logging

logger = setup_logging()


class AutonomousTradingEngine:
    """Main trading engine that runs the autonomous bot."""

    def __init__(self):
        self.settings = get_settings()
        self.is_running = False
        self.error_count = 0
        self.max_errors = 20  # Increased from 10 - more tolerant to transient errors
        self.consecutive_errors = 0  # Track consecutive errors separately
        self.last_successful_cycle = None  # Track last successful operation

    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Autonomous Trading Engine...")

        try:
            # Connect to database
            db = await get_db_client()
            logger.info("✅ Database connected")

            # Initialize exchange
            exchange = await get_exchange_client()
            logger.info("✅ Exchange connected")

            # Get initial config
            config = await db.get_trading_config()
            logger.info(f"✅ Trading config loaded: ${config['current_capital']} capital")

            logger.info("🚀 All systems initialized successfully!")

        except Exception as e:
            error_msg = str(e)

            # Check if it's a Binance Futures account activation issue
            if '-2015' in error_msg or 'permissions for action' in error_msg:
                logger.error("=" * 70)
                logger.error("⚠️  BINANCE FUTURES ACCOUNT NOT ACTIVATED")
                logger.error("=" * 70)
                logger.error("")
                logger.error("Your API key has 'Enable Futures' permission, but your Binance")
                logger.error("account doesn't have Futures trading activated yet.")
                logger.error("")
                logger.error("📋 TO FIX THIS:")
                logger.error("")
                logger.error("1. Go to: https://www.binance.com/en/futures/BTCUSDT")
                logger.error("2. Click 'Open Now' to activate Futures trading")
                logger.error("3. Agree to the terms and conditions")
                logger.error("4. Complete any required verification (if prompted)")
                logger.error("5. Wait 1-2 minutes for activation to complete")
                logger.error("6. Restart this bot: docker-compose restart trading-bot")
                logger.error("")
                logger.error("=" * 70)
            else:
                logger.error(f"❌ Failed to initialize: {e}")
            raise

    async def run_forever(self):
        """
        Main infinite loop.
        Runs continuously, manages market scanning and position monitoring.
        """
        self.is_running = True
        notifier = get_notifier()

        # Send startup notification
        await notifier.send_startup_message()

        logger.info("=" * 60)
        logger.info("🤖 AUTONOMOUS TRADING ENGINE STARTED")
        logger.info("=" * 60)

        while self.is_running:
            try:
                # Check if trading is enabled
                if not await self.check_trading_enabled():
                    logger.info("⏸️  Trading disabled, waiting...")
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue

                # Get current position status
                db = await get_db_client()
                active_position = await db.get_active_position()

                if active_position:
                    # We have an open position - monitor it
                    logger.info(f"📊 Monitoring position: {active_position['symbol']} {active_position['side']}")

                    position_monitor = get_position_monitor()
                    await position_monitor.check_position(active_position)

                    # Check every 60 seconds when position is open
                    await asyncio.sleep(self.settings.position_check_seconds)

                else:
                    # No position - wait for manual /scan command
                    logger.info("💤 No active position, waiting for manual scan command...")

                    # Sleep for 60 seconds, then check again
                    # Bot will only scan when user runs /scan command via Telegram
                    await asyncio.sleep(60)

                # Reset error counts on successful cycle
                self.error_count = 0
                self.consecutive_errors = 0
                self.last_successful_cycle = datetime.now()

                # Cleanup expired AI cache periodically
                if datetime.now().minute % 10 == 0:  # Every 10 minutes
                    await db.cleanup_expired_cache()

            except KeyboardInterrupt:
                logger.info("⏸️  Keyboard interrupt received")
                break

            except asyncio.TimeoutError:
                # Transient timeout error - don't count as critical
                self.consecutive_errors += 1
                logger.warning(f"⏱️ Operation timeout (consecutive: {self.consecutive_errors})")

                if self.consecutive_errors >= 5:
                    logger.error(f"Too many consecutive timeouts: {self.consecutive_errors}")
                    await notifier.send_alert(
                        'warning',
                        f"⚠️ Network issues detected\n"
                        f"Consecutive timeouts: {self.consecutive_errors}\n\n"
                        f"Bot will retry with exponential backoff."
                    )

                # Exponential backoff for consecutive errors
                backoff_time = min(60 * (2 ** min(self.consecutive_errors - 1, 4)), 300)  # Max 5 min
                await asyncio.sleep(backoff_time)

            except (ConnectionError, OSError) as e:
                # Network/connection errors - transient, retry with backoff
                self.consecutive_errors += 1
                logger.warning(f"🌐 Network error (consecutive: {self.consecutive_errors}): {e}")

                if self.consecutive_errors >= 3:
                    await notifier.send_alert(
                        'warning',
                        f"🌐 Connectivity issues\n"
                        f"Consecutive errors: {self.consecutive_errors}\n"
                        f"Type: {type(e).__name__}\n\n"
                        f"Retrying with backoff..."
                    )

                backoff_time = min(30 * (2 ** min(self.consecutive_errors - 1, 3)), 180)  # Max 3 min
                await asyncio.sleep(backoff_time)

            except Exception as e:
                # General errors - count and handle
                self.error_count += 1
                self.consecutive_errors += 1

                error_type = type(e).__name__
                error_msg = str(e)

                logger.error(
                    f"❌ Error in main loop (total: {self.error_count}, "
                    f"consecutive: {self.consecutive_errors}): {error_type}: {error_msg}"
                )

                # Send error report for non-trivial errors
                if self.consecutive_errors >= 2:
                    await notifier.send_error_report('TradingEngine', f"{error_type}: {error_msg}")

                # Check if we should stop
                if self.error_count >= self.max_errors:
                    logger.critical(f"🚨 Too many total errors ({self.error_count}), stopping bot")
                    await notifier.send_alert(
                        'critical',
                        f"🚨 Bot stopped due to too many errors\n"
                        f"Total errors: {self.error_count}\n"
                        f"Consecutive: {self.consecutive_errors}\n"
                        f"Last error: {error_type}: {error_msg[:200]}\n\n"
                        f"Please investigate logs and restart manually."
                    )
                    break

                if self.consecutive_errors >= 10:
                    logger.critical(f"🚨 Too many consecutive errors ({self.consecutive_errors}), stopping bot")
                    await notifier.send_alert(
                        'critical',
                        f"🚨 Bot stopped due to repeated failures\n"
                        f"Consecutive errors: {self.consecutive_errors}\n"
                        f"Last error: {error_type}: {error_msg[:200]}\n\n"
                        f"System may be unstable. Manual intervention required."
                    )
                    break

                # Exponential backoff
                backoff_time = min(60 * (2 ** min(self.consecutive_errors - 1, 4)), 300)
                logger.info(f"Waiting {backoff_time}s before retry...")
                await asyncio.sleep(backoff_time)

        logger.info("🛑 Trading engine stopped")
        await self.cleanup()

    async def check_trading_enabled(self) -> bool:
        """
        Check circuit breakers and trading status.

        Returns:
            True if trading is allowed, False otherwise
        """
        risk_manager = get_risk_manager()
        notifier = get_notifier()

        limits_check = await risk_manager.check_daily_limits()

        if not limits_check['can_trade']:
            logger.warning(f"⚠️  Trading disabled: {limits_check['reason']}")
            # Note: Circuit breaker alert removed to prevent spam
            # User can check status anytime with /status command
            return False

        return True

    async def send_daily_summary(self):
        """Send end-of-day summary."""
        try:
            db = await get_db_client()
            notifier = get_notifier()

            today = date.today()

            # Update daily performance
            await db.update_daily_performance(today)

            # Get performance data
            performance = await db.get_daily_performance(today)

            if performance:
                await notifier.send_daily_summary(performance)
                logger.info("📊 Daily summary sent")

        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")

    async def cleanup(self):
        """Cleanup resources before shutdown."""
        logger.info("Cleaning up...")

        try:
            # Close exchange connection
            exchange = await get_exchange_client()
            await exchange.close()

            # Close database connection
            db = await get_db_client()
            await db.close()

            logger.info("✅ Cleanup complete")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def stop(self):
        """Stop the trading engine."""
        logger.info("Stopping trading engine...")
        self.is_running = False


# Singleton instance
_trading_engine: Optional[AutonomousTradingEngine] = None


def get_trading_engine() -> AutonomousTradingEngine:
    """Get or create trading engine instance."""
    global _trading_engine
    if _trading_engine is None:
        _trading_engine = AutonomousTradingEngine()
    return _trading_engine
