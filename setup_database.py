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

        # Execute schema (WITHOUT migration - it has syntax errors in single execute)
        # Remove the DO blocks from schema before executing
        schema_lines = schema_sql.split('\n')
        clean_schema = []
        skip_until_end = False

        for line in schema_lines:
            if line.strip().startswith('DO $$'):
                skip_until_end = True
                continue
            if skip_until_end and 'END $$;' in line:
                skip_until_end = False
                continue
            if not skip_until_end:
                clean_schema.append(line)

        clean_schema_sql = '\n'.join(clean_schema)
        await conn.execute(clean_schema_sql)

        print("✅ Tables created successfully!")

        # Run migration manually with Python (not SQL DO blocks)
        print("\n🔄 Running ML snapshot migration...")

        # Define columns to add
        migrations = [
            # active_position table
            ("active_position", "ai_reasoning", "TEXT"),
            ("active_position", "entry_snapshot", "JSONB"),
            ("active_position", "entry_slippage_percent", "DECIMAL(10,6)"),
            ("active_position", "entry_fill_time_ms", "INTEGER"),

            # trade_history table
            ("trade_history", "ai_reasoning", "TEXT"),
            ("trade_history", "entry_snapshot", "JSONB"),
            ("trade_history", "exit_snapshot", "JSONB"),
            ("trade_history", "entry_slippage_percent", "DECIMAL(10,6)"),
            ("trade_history", "exit_slippage_percent", "DECIMAL(10,6)"),
            ("trade_history", "entry_fill_time_ms", "INTEGER"),
            ("trade_history", "exit_fill_time_ms", "INTEGER"),
        ]

        for table, column, col_type in migrations:
            try:
                # Check if column exists
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name=$1 AND column_name=$2
                    )
                """, table, column)

                if not exists:
                    await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                    print(f"  ✅ Added {table}.{column}")
            except Exception as e:
                print(f"  ⚠️ {table}.{column}: {e}")

        # Add indexes
        try:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_active_entry_snapshot
                ON active_position USING GIN (entry_snapshot)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entry_snapshot_indicators
                ON trade_history USING GIN (entry_snapshot)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exit_snapshot_indicators
                ON trade_history USING GIN (exit_snapshot)
            """)
            print("  ✅ Added GIN indexes")
        except Exception as e:
            print(f"  ⚠️ Index creation: {e}")

        print("✅ ML snapshot migration complete!")

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
