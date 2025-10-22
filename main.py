"""
Main entry point for the Autonomous Leverage Trading Bot.
Starts the trading engine and handles graceful shutdown.
"""

import asyncio
import signal
import sys
from src.trading_engine import get_trading_engine
from src.utils import setup_logging
from src.config import get_settings

logger = setup_logging()


async def main():
    """Main function."""
    # Get settings to trigger warnings
    settings = get_settings()

    # Show risk warning if not paper trading
    if not settings.use_paper_trading:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: LIVE TRADING MODE")
        print("=" * 60)
        print("Real money will be used for trading!")
        print("Ensure you understand the risks before proceeding.")
        print("=" * 60)

        # Wait for confirmation
        try:
            confirmation = input("\nType 'START' to begin trading: ")
            if confirmation.upper() != 'START':
                print("Startup cancelled.")
                return
        except KeyboardInterrupt:
            print("\nStartup cancelled.")
            return

    # Create and initialize trading engine
    engine = get_trading_engine()

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Signal {sig} received, shutting down...")
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize engine
    try:
        await engine.initialize()
    except Exception as e:
        logger.critical(f"Failed to initialize engine: {e}")
        sys.exit(1)

    # Run forever
    try:
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
