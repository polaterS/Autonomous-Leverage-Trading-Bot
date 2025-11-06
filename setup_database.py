"""
Database setup script for the Autonomous Leverage Trading Bot.
Creates all necessary tables and initializes configuration.
"""

import asyncio
import asyncpg
import sys
from src.config import get_settings
from pathlib import Path


async def setup_database():
    """Create database tables and initial configuration."""
    settings = get_settings()

    print("=" * 60)
    print("DATABASE SETUP - Autonomous Leverage Trading Bot")
    print("=" * 60)
    print(f"\nConnecting to database at: {settings.database_url.split('@')[-1] if '@' in settings.database_url else 'localhost'}")

    conn = None
    try:
        # Railway requires SSL for public connections
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Connect to database with timeout
        conn = await asyncio.wait_for(
            asyncpg.connect(settings.database_url, ssl=ssl_context),
            timeout=10.0
        )

        print("✅ Connected successfully!")
        print("\nCreating database tables...")

        # Read and execute schema
        schema_path = Path(__file__).parent / "schema.sql"

        if not schema_path.exists():
            print(f"❌ ERROR: schema.sql not found at {schema_path}")
            sys.exit(1)

        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()

        # Execute schema
        await conn.execute(schema_sql)

        print("✅ Tables created successfully!")

        # Run migration for snapshot columns (idempotent - safe to run multiple times)
        print("\n🔄 Running ML snapshot migration...")
        migration_path = Path(__file__).parent / "migrate_snapshots.sql"

        if migration_path.exists():
            with open(migration_path, 'r', encoding='utf-8') as f:
                migration_sql = f.read()

            try:
                await conn.execute(migration_sql)
                print("✅ ML snapshot migration complete!")
            except Exception as migration_error:
                print(f"⚠️ Migration warning (may be already applied): {migration_error}")
        else:
            print("⚠️ Migration file not found, skipping...")

        # Update initial configuration with environment values
        await conn.execute("""
            UPDATE trading_config
            SET initial_capital = $1,
                current_capital = $1,
                position_size_percent = $2,
                min_stop_loss_percent = $3,
                max_stop_loss_percent = $4,
                min_profit_usd = $5,
                max_leverage = $6,
                min_ai_confidence = $7,
                daily_loss_limit_percent = $8,
                max_consecutive_losses = $9,
                last_updated = NOW()
            WHERE id = 1
        """,
            settings.initial_capital,
            settings.position_size_percent,
            settings.min_stop_loss_percent,
            settings.max_stop_loss_percent,
            settings.min_profit_usd,
            settings.max_leverage,
            settings.min_ai_confidence,
            settings.daily_loss_limit_percent,
            settings.max_consecutive_losses
        )

        print("Configuration updated successfully!")

        # Verify setup
        config = await conn.fetchrow("SELECT * FROM trading_config WHERE id = 1")
        print("\nInitial Trading Configuration:")
        print(f"  Initial Capital: ${config['initial_capital']}")
        print(f"  Max Leverage: {config['max_leverage']}x")
        print(f"  Position Size: {float(config['position_size_percent'])*100}%")
        print(f"  Stop Loss Range: {float(config['min_stop_loss_percent'])*100}% - {float(config['max_stop_loss_percent'])*100}%")
        print(f"  Min Profit: ${config['min_profit_usd']}")
        print(f"  Min AI Confidence: {float(config['min_ai_confidence'])*100}%")
        print(f"  Daily Loss Limit: {float(config['daily_loss_limit_percent'])*100}%")
        print(f"  Max Consecutive Losses: {config['max_consecutive_losses']}")

        await conn.close()
        print("\n" + "=" * 60)
        print("✅ DATABASE SETUP COMPLETE!")
        print("=" * 60)
        print("\nYou can now start the trading bot with: python main.py")

    except asyncio.TimeoutError:
        print("❌ ERROR: Database connection timeout")
        print("Please ensure PostgreSQL is running and accessible.")
        sys.exit(1)
    except asyncpg.PostgresError as e:
        print(f"❌ ERROR: Database error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check DATABASE_URL in .env file")
        print("  2. Ensure PostgreSQL is running")
        print("  3. Verify database credentials")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")
        sys.exit(1)
    finally:
        if conn:
            await conn.close()


if __name__ == "__main__":
    try:
        asyncio.run(setup_database())
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user.")
        sys.exit(0)
