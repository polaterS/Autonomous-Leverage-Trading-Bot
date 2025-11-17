"""
Main entry point for the Autonomous Leverage Trading Bot.
Starts the trading engine and handles graceful shutdown.

🧠 VERSION: 6.2-PA-INTEGRATED (Classic + Price Action ML Integration)
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
BOT_VERSION = "6.2-PA-INTEGRATED"

# Log deployment version from VERSION file
try:
    with open('VERSION', 'r') as f:
        version_info = f.read().strip()
        deployment_version = version_info.split()[0]
        logger.info("="*70)
        logger.info(f"📦 DEPLOYMENT VERSION: {deployment_version}")
        logger.info("="*70)
except:
    logger.warning("⚠️ VERSION file not found")

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

    # 🔐 TIER 3: Security Validation (CRITICAL)
    try:
        from src.security_validator import get_security_validator
        logger.info("🔐 Running TIER 3 security validation...")

        validator = get_security_validator()
        security_results = await validator.run_startup_checks()

        # Critical: If security checks failed, stop bot
        if security_results['checks_failed'] > 0:
            logger.critical("🚨 CRITICAL SECURITY ISSUES DETECTED - BOT WILL NOT START")
            logger.critical("Review security validation report above")
            sys.exit(1)

        logger.info(f"✅ Security validation passed ({security_results['checks_passed']} checks)")

    except Exception as e:
        logger.error(f"Security validation failed: {e}")
        logger.critical("🚨 Cannot start bot without security validation")
        sys.exit(1)

    # 🔧 CRITICAL: Run profit target migration (if needed)
    try:
        logger.info("🔧 Checking for required database migrations...")
        from migrations.add_profit_targets import migrate
        await migrate()
        logger.info("✅ Database migrations completed successfully!")
    except Exception as e:
        # If columns already exist, that's fine
        if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
            logger.info("✅ Migration columns already exist")
        else:
            logger.error(f"⚠️ Migration check failed (non-critical): {e}")

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

    # 🤖 AUTO-TRAIN ML MODEL ON STARTUP (if historical data exists)
    try:
        from src.ml_predictor import get_ml_predictor
        from src.database import get_db_client

        logger.info("=" * 60)
        logger.info("🤖 ML MODEL STARTUP CHECK")
        logger.info("=" * 60)

        ml_predictor = get_ml_predictor()
        stats = ml_predictor.get_model_stats()

        # Check if model already trained
        if stats['is_trained']:
            logger.info(f"✅ ML model already trained ({stats['training_samples']} samples)")
            logger.info(f"   Last trained: {stats['last_training_time']}")
        else:
            # Check if we have enough historical data
            db = await get_db_client()
            trade_count = await db.pool.fetchval(
                "SELECT COUNT(*) FROM trade_history WHERE entry_snapshot IS NOT NULL"
            )

            logger.info(f"📊 Found {trade_count} trades with snapshots in database")

            if trade_count >= 30:
                logger.info(f"🎓 Training ML model on {trade_count} historical trades...")
                training_success = await ml_predictor.train(force_retrain=True)

                if training_success:
                    new_stats = ml_predictor.get_model_stats()
                    logger.info("=" * 60)
                    logger.info("✅ ML MODEL TRAINING SUCCESSFUL!")
                    logger.info(f"   Training Samples: {new_stats['training_samples']}")
                    logger.info(f"   Model Path: {new_stats['model_path']}")
                    logger.info("   Confidence will be 40-70% (vs 0-25% with rules)")
                    logger.info("=" * 60)
                else:
                    logger.warning("⚠️ ML training failed, will use rule-based fallback")
                    logger.warning("   Model will auto-train after 50 new trades")
            else:
                logger.info(f"ℹ️ Not enough data for training (need 30+, have {trade_count})")
                logger.info("   Using rule-based fallback until more trades collected")
                logger.info("   Model will auto-train after 30 trades")

        logger.info("=" * 60)

    except Exception as e:
        logger.warning(f"ML model startup check failed (non-critical): {e}")
        logger.info("   Bot will use rule-based fallback")

    # 🔐 TIER 3: Initialize API Key Manager
    try:
        from src.api_key_manager import get_api_key_manager

        logger.info("🔑 Initializing API Key Manager...")
        key_manager = get_api_key_manager()

        # Check key expiration
        expiration_status = await key_manager.check_key_expiration()

        if expiration_status['expired']:
            logger.critical(f"🚨 {len(expiration_status['expired'])} API key(s) EXPIRED!")
            for service, info in expiration_status['expired'].items():
                logger.critical(f"   - {service}: {info['days_overdue']} days overdue")

        if expiration_status['expiring_soon']:
            logger.warning(f"⚠️ {len(expiration_status['expiring_soon'])} API key(s) expiring soon")
            for service, info in expiration_status['expiring_soon'].items():
                logger.warning(f"   - {service}: {info['days_remaining']} days remaining")

        # Get all key statuses
        all_keys_status = key_manager.get_all_keys_status()
        logger.info(f"✅ API Key Manager initialized - {len(all_keys_status)} keys tracked")

    except ImportError:
        logger.warning("⚠️ cryptography not installed - API key rotation disabled")
        logger.info("   Install with: pip install cryptography")
    except Exception as e:
        logger.warning(f"API Key Manager initialization failed (non-critical): {e}")

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

    # 🌐 Initialize WebSocket price feed (eliminates Binance 429 rate limit errors)
    price_manager = None
    try:
        from src.price_manager import get_price_manager
        price_manager = get_price_manager()
        logger.info("=" * 60)
        logger.info("🌐 WEBSOCKET PRICE FEED INITIALIZATION")
        logger.info("=" * 60)
        logger.info("✅ Price manager initialized")
        logger.info("   Strategy: WebSocket (real-time) + REST cache (fallback)")
        logger.info("   Expected: -85% API calls, zero 429 rate limit errors")
        logger.info("=" * 60)
    except Exception as e:
        logger.warning(f"WebSocket price feed initialization failed (non-critical): {e}")
        logger.info("   Will use REST API with caching")

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
    finally:
        # Cleanup WebSocket connections
        if price_manager:
            try:
                logger.info("Cleaning up WebSocket price feed...")
                await price_manager.cleanup()
                logger.info("✅ WebSocket cleanup complete")
            except Exception as cleanup_error:
                logger.warning(f"WebSocket cleanup error: {cleanup_error}")

    logger.info("Bot shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete.")
        sys.exit(0)
