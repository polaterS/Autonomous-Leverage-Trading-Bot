"""
Autonomous Trading Engine - Main orchestrator for the trading bot.
Manages the infinite loop, coordinates all components, handles errors.
"""

import asyncio
from datetime import datetime, date
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
        self.max_errors = 10

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
                    # No position - scan for opportunities
                    logger.info("🔍 No active position, scanning markets...")

                    scanner = get_market_scanner()
                    await scanner.scan_and_execute()

                    # Check every 5 minutes when no position
                    await asyncio.sleep(self.settings.scan_interval_seconds)

                # Reset error count on successful cycle
                self.error_count = 0

                # Cleanup expired AI cache periodically
                if datetime.now().minute % 10 == 0:  # Every 10 minutes
                    await db.cleanup_expired_cache()

            except KeyboardInterrupt:
                logger.info("⏸️  Keyboard interrupt received")
                break

            except Exception as e:
                self.error_count += 1
                logger.error(f"❌ Error in main loop (#{self.error_count}): {e}")

                await notifier.send_error_report('TradingEngine', str(e))

                if self.error_count >= self.max_errors:
                    logger.critical(f"🚨 Too many errors ({self.error_count}), stopping bot")
                    await notifier.send_alert(
                        'critical',
                        f"🚨 Bot stopped due to too many errors ({self.error_count})\n"
                        f"Last error: {e}\n\n"
                        f"Please investigate and restart manually."
                    )
                    break

                # Wait before retry
                await asyncio.sleep(60)

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

            # Send alert if circuit breaker triggered
            if limits_check['type'] in ['daily_loss_limit', 'consecutive_losses']:
                await notifier.send_circuit_breaker_alert(
                    limits_check['type'],
                    Decimal("0"),  # Placeholder
                    Decimal("0")   # Placeholder
                )

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
