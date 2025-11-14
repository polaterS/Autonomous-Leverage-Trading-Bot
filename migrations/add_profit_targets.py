"""
Database migration: Add profit target columns for partial exit strategy.

This migration adds the following columns to active_position table:
- profit_target_1: First profit target (close 50% of position)
- profit_target_2: Second profit target (close remaining 50%)
- partial_exit_done: Boolean flag tracking if partial exit was executed
- partial_exit_profit: Profit from partial exit in USD
"""

import asyncio
import sys
import os

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import get_db_client


async def migrate():
    """Run the migration."""
    # Use existing database client
    db = await get_db_client()
    conn = await db.pool.acquire()

    try:
        print("🔄 Starting migration: Add profit target columns...")

        # Check if columns already exist
        check_query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'active_position'
            AND column_name IN ('profit_target_1', 'profit_target_2', 'partial_exit_done', 'partial_exit_profit')
        """
        existing_columns = await conn.fetch(check_query)
        existing_column_names = [row['column_name'] for row in existing_columns]

        if len(existing_column_names) == 4:
            print("✅ All columns already exist. Migration not needed.")
            return

        print(f"📊 Found {len(existing_column_names)} existing columns: {existing_column_names}")

        # Add columns that don't exist
        migrations = []

        if 'profit_target_1' not in existing_column_names:
            migrations.append("""
                ALTER TABLE active_position
                ADD COLUMN IF NOT EXISTS profit_target_1 DECIMAL(20, 8)
            """)

        if 'profit_target_2' not in existing_column_names:
            migrations.append("""
                ALTER TABLE active_position
                ADD COLUMN IF NOT EXISTS profit_target_2 DECIMAL(20, 8)
            """)

        if 'partial_exit_done' not in existing_column_names:
            migrations.append("""
                ALTER TABLE active_position
                ADD COLUMN IF NOT EXISTS partial_exit_done BOOLEAN DEFAULT FALSE
            """)

        if 'partial_exit_profit' not in existing_column_names:
            migrations.append("""
                ALTER TABLE active_position
                ADD COLUMN IF NOT EXISTS partial_exit_profit DECIMAL(20, 2) DEFAULT 0
            """)

        # Execute migrations
        for i, migration in enumerate(migrations, 1):
            print(f"⏳ Executing migration {i}/{len(migrations)}...")
            await conn.execute(migration)
            print(f"✅ Migration {i} completed")

        print("✅ All migrations completed successfully!")

        # Verify columns were added
        verify_query = """
            SELECT column_name, data_type, column_default
            FROM information_schema.columns
            WHERE table_name = 'active_position'
            AND column_name IN ('profit_target_1', 'profit_target_2', 'partial_exit_done', 'partial_exit_profit')
            ORDER BY column_name
        """
        columns = await conn.fetch(verify_query)

        print("\n📋 Verification - New columns:")
        for col in columns:
            print(f"  - {col['column_name']}: {col['data_type']} (default: {col['column_default']})")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        await db.pool.release(conn)
        await db.close()


if __name__ == "__main__":
    asyncio.run(migrate())
