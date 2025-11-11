"""
🚀 TIER 1 & 2 Database Migration Runner
Automatically runs migration on Railway production database
"""

import asyncio
import asyncpg
import ssl
import sys
from pathlib import Path
from src.config import get_settings


async def run_migration():
    """Run TIER 1 & 2 database migration."""
    settings = get_settings()

    print("=" * 60)
    print("🚀 TIER 1 & 2 Database Migration")
    print("=" * 60)
    print()

    conn = None
    try:
        # Connect to database
        print("🔄 Connecting to Railway database...")
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        conn = await asyncpg.connect(settings.database_url, ssl=ssl_context)
        print("✅ Connected successfully!")

        print("\n📋 Running TIER 1 & 2 migration...")
        print("=" * 60)

        # Migration script
        migrations = [
            # Active Position columns
            ("max_profit_percent", "active_position",
             "ALTER TABLE active_position ADD COLUMN max_profit_percent DECIMAL(10, 4) DEFAULT 0.0"),

            ("trailing_stop_activated", "active_position",
             "ALTER TABLE active_position ADD COLUMN trailing_stop_activated BOOLEAN DEFAULT FALSE"),

            ("partial_exits_completed", "active_position",
             "ALTER TABLE active_position ADD COLUMN partial_exits_completed TEXT[] DEFAULT '{}'"),

            ("original_quantity", "active_position",
             "ALTER TABLE active_position ADD COLUMN original_quantity DECIMAL(20, 8)"),

            ("entry_market_regime", "active_position",
             "ALTER TABLE active_position ADD COLUMN entry_market_regime VARCHAR(50)"),

            ("exit_market_regime", "active_position",
             "ALTER TABLE active_position ADD COLUMN exit_market_regime VARCHAR(50)"),

            # Trade History columns
            ("max_profit_percent_achieved", "trade_history",
             "ALTER TABLE trade_history ADD COLUMN max_profit_percent_achieved DECIMAL(10, 4)"),

            ("trailing_stop_triggered", "trade_history",
             "ALTER TABLE trade_history ADD COLUMN trailing_stop_triggered BOOLEAN DEFAULT FALSE"),

            ("had_partial_exits", "trade_history",
             "ALTER TABLE trade_history ADD COLUMN had_partial_exits BOOLEAN DEFAULT FALSE"),

            ("partial_exit_details", "trade_history",
             "ALTER TABLE trade_history ADD COLUMN partial_exit_details JSONB"),

            ("entry_market_regime", "trade_history",
             "ALTER TABLE trade_history ADD COLUMN entry_market_regime VARCHAR(50)"),

            ("exit_market_regime", "trade_history",
             "ALTER TABLE trade_history ADD COLUMN exit_market_regime VARCHAR(50)"),
        ]

        added_count = 0
        skipped_count = 0

        for column_name, table_name, alter_sql in migrations:
            # Check if column exists
            check_query = """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = $1 AND column_name = $2
                )
            """

            exists = await conn.fetchval(check_query, table_name, column_name)

            if exists:
                print(f"⏭️  {table_name}.{column_name} already exists")
                skipped_count += 1
            else:
                # Add column
                await conn.execute(alter_sql)
                print(f"✅ Added {table_name}.{column_name}")
                added_count += 1

        # Create indexes
        print("\n📊 Creating indexes...")

        indexes = [
            ("idx_trade_history_regime",
             "CREATE INDEX IF NOT EXISTS idx_trade_history_regime ON trade_history(entry_market_regime)"),

            ("idx_trade_history_partial_exits",
             "CREATE INDEX IF NOT EXISTS idx_trade_history_partial_exits ON trade_history(had_partial_exits)"),

            ("idx_trade_history_trailing_stop",
             "CREATE INDEX IF NOT EXISTS idx_trade_history_trailing_stop ON trade_history(trailing_stop_triggered)"),
        ]

        for index_name, create_sql in indexes:
            await conn.execute(create_sql)
            print(f"✅ Created index: {index_name}")

        print("\n" + "=" * 60)
        print(f"🎉 MIGRATION COMPLETE!")
        print(f"   Added: {added_count} columns")
        print(f"   Skipped: {skipped_count} (already existed)")
        print(f"   Indexes: {len(indexes)} created")
        print("\n✅ TIER 1 & 2 features are now ACTIVE!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if conn:
            await conn.close()


if __name__ == "__main__":
    print("\n🎯 Starting TIER 1 & 2 Migration...")
    print("This will add all necessary database columns.\n")

    success = asyncio.run(run_migration())

    if success:
        print("\n🎯 Next steps:")
        print("   ✅ Bot will automatically use new features")
        print("   ✅ Trailing stops will activate at 3% profit")
        print("   ✅ Partial exits at 2%, 4%, 6%")
        print("   ✅ Market regime detection enabled")
        sys.exit(0)
    else:
        print("\n❌ Migration failed - please check errors above")
        sys.exit(1)
