"""
Main entry point for the Autonomous Leverage Trading Bot.
Starts the trading engine and handles graceful shutdown.

🧠 VERSION: ML-ONLY-2.0 (Pure ML, AI Bypassed)
"""

import asyncio
import signal
import sys
import os
from src.trading_engine import get_trading_engine
from src.telegram_bot import get_telegram_bot
from src.utils import setup_logging
from src.config import get_settings

logger = setup_logging()

# Version marker for deployment verification
BOT_VERSION = "ML-ONLY-2.0"
logger.info(f"🚀 Starting Autonomous Trading Bot v{BOT_VERSION}")


async def main():
    """Main function."""
    # Get settings to trigger warnings
    settings = get_settings()

    # Auto-setup database tables if needed (for Railway deployment)
    try:
        from setup_database import setup_database
        logger.info("Checking database setup...")
        await setup_database()
    except Exception as e:
        # If tables already exist, setup_database will succeed anyway
        # Only fail if it's a connection error
        if "already exists" not in str(e).lower():
            logger.error(f"Database setup check failed: {e}")
            # Continue anyway - might be tables already exist

    # Database health check and auto-cleanup
    try:
        from src.db_maintenance import verify_database_health, cleanup_duplicate_configs
        logger.info("Running database health check...")
        health = await verify_database_health()

        # Auto-fix if duplicate configs detected
        if health['config_count'] > 1:
            logger.warning(f"Detected {health['config_count']} config rows, running auto-cleanup...")
            await cleanup_duplicate_configs()
        elif health['issues']:
            logger.warning(f"Database issues detected: {', '.join(health['issues'])}")
        else:
            logger.info("Database health check passed!")
    except Exception as e:
        logger.warning(f"Database health check failed (non-critical): {e}")

    # Show risk warning if not paper trading
    if not settings.use_paper_trading:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: LIVE TRADING MODE")
        print("=" * 60)
        print("Real money will be used for trading!")
        print("Ensure you understand the risks before proceeding.")
        print("=" * 60)

        # Wait for confirmation (skip in Docker/automated environments)
        auto_start = os.getenv('AUTO_START_LIVE_TRADING', 'false').lower() == 'true'

        if not auto_start:
            try:
                confirmation = input("\nType 'START' to begin trading: ")
                if confirmation.upper() != 'START':
                    print("Startup cancelled.")
                    return
            except (KeyboardInterrupt, EOFError):
                print("\nStartup cancelled or running in non-interactive mode.")
                print("Set AUTO_START_LIVE_TRADING=true in .env to enable automatic start.")
                return
        else:
            print("\n✅ AUTO_START_LIVE_TRADING enabled - Starting automatically...")
            print("⚠️  LIVE TRADING WILL BEGIN IN 3 SECONDS...")
            await asyncio.sleep(3)

    # Create and initialize trading engine
    engine = get_trading_engine()

    # Initialize telegram bot
    telegram_bot = None
    try:
        telegram_bot = await get_telegram_bot()
        logger.info("✅ Interactive Telegram bot initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Telegram bot: {e}")
        logger.warning("Continuing without Telegram bot...")

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Signal {sig} received, shutting down...")
        engine.stop()
        if telegram_bot:
            try:
                asyncio.create_task(telegram_bot.shutdown())
            except:
                pass

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize engine
    try:
        await engine.initialize()
    except Exception as e:
        error_msg = str(e)

        # Special handling for Binance Futures not activated
        if '-2015' in error_msg or 'permissions for action' in error_msg:
            print("\n" + "=" * 70)
            print("⚠️  BINANCE FUTURES ACCOUNT NOT ACTIVATED")
            print("=" * 70)
            print("")
            print("Your API key has 'Enable Futures' permission, but your Binance")
            print("account doesn't have Futures trading activated yet.")
            print("")
            print("📋 TO FIX THIS:")
            print("")
            print("1. Go to: https://www.binance.com/en/futures/BTCUSDT")
            print("2. Click 'Open Now' to activate Futures trading")
            print("3. Agree to the terms and conditions")
            print("4. Complete any required verification (if prompted)")
            print("5. Wait 1-2 minutes for activation to complete")
            print("6. Restart this bot: docker-compose restart trading-bot")
            print("")
            print("=" * 70)
        else:
            logger.critical(f"Failed to initialize engine: {e}")
        sys.exit(1)

    # Run both trading engine and telegram bot concurrently
    try:
        if telegram_bot:
            logger.info("🚀 Starting trading engine and Telegram bot...")
            # Run both in parallel
            await asyncio.gather(
                engine.run_forever(),
                telegram_bot.run()
            )
        else:
            logger.info("🚀 Starting trading engine only...")
            await engine.run_forever()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)

    logger.info("Bot shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete.")
        sys.exit(0)
