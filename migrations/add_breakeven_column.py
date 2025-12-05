"""
Database migration: Add breakeven_active column for Level-Based TP system.

This migration adds the following column to active_position table:
- breakeven_active: Boolean flag tracking if stop was moved to breakeven after TP1 hit

v5.0.2 Feature: Level-Based Partial TP + Breakeven System
- When TP1 is hit, close 50% and move stop to breakeven (entry price)
- Trade becomes risk-free, remaining 50% runs to TP2
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
        print("🔄 Starting migration: Add breakeven_active column...")

        # Check if column already exists
        check_query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'active_position'
            AND column_name = 'breakeven_active'
        """
        existing_columns = await conn.fetch(check_query)

        if len(existing_columns) > 0:
            print("✅ Column breakeven_active already exists. Migration not needed.")
            return

        print("📊 Adding breakeven_active column...")

        # Add the column
        await conn.execute("""
            ALTER TABLE active_position
            ADD COLUMN IF NOT EXISTS breakeven_active BOOLEAN DEFAULT FALSE
        """)

        print("✅ Migration completed successfully!")

        # Verify column was added
        verify_query = """
            SELECT column_name, data_type, column_default
            FROM information_schema.columns
            WHERE table_name = 'active_position'
            AND column_name = 'breakeven_active'
        """
        columns = await conn.fetch(verify_query)

        print("\n📋 Verification - New column:")
        for col in columns:
            print(f"  - {col['column_name']}: {col['data_type']} (default: {col['column_default']})")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        await db.pool.release(conn)
        # DON'T close the pool - main bot needs it!


if __name__ == "__main__":
    asyncio.run(migrate())
