"""
Database migration: Add enhanced trading system columns.

This migration adds the following columns to trades table:
- confluence_score: Quality score (0-100) from confluence scoring engine
- quality: Quality classification (EXCELLENT, STRONG, GOOD, MEDIOCRE, POOR)
- risk_percentage: Percentage of account risked on this trade
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
        print("=" * 70)
        print("🔄 Starting migration: Add Enhanced Trading System columns...")
        print("=" * 70)

        # Check if trades table exists first (using direct PostgreSQL query)
        table_check_query = """
            SELECT EXISTS (
                SELECT FROM pg_tables
                WHERE schemaname = 'public'
                AND tablename = 'trades'
            )
        """
        table_exists = await conn.fetchval(table_check_query)

        print(f"   Table existence check: {table_exists}")

        if not table_exists:
            print("⚠️  'trades' table doesn't exist yet. Skipping migration.")
            print("   Migration will run automatically on next bot restart.")
            print("=" * 70)
            return

        print("✅ 'trades' table found! Proceeding with migration...")

        # Check if columns already exist
        check_query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name = 'trades'
            AND column_name IN ('confluence_score', 'quality', 'risk_percentage')
        """
        existing_columns = await conn.fetch(check_query)
        existing_column_names = [row['column_name'] for row in existing_columns]

        print(f"   Found {len(existing_column_names)} existing enhanced columns: {existing_column_names}")

        if len(existing_column_names) == 3:
            print("✅ All columns already exist. Migration not needed.")
            print("=" * 70)
            return

        print(f"📊 Found {len(existing_column_names)} existing columns: {existing_column_names}")
        print("")

        # Add columns that don't exist
        migrations = []

        if 'confluence_score' not in existing_column_names:
            migrations.append("""
                ALTER TABLE trades
                ADD COLUMN IF NOT EXISTS confluence_score FLOAT DEFAULT NULL
            """)

        if 'quality' not in existing_column_names:
            migrations.append("""
                ALTER TABLE trades
                ADD COLUMN IF NOT EXISTS quality VARCHAR(20) DEFAULT NULL
            """)

        if 'risk_percentage' not in existing_column_names:
            migrations.append("""
                ALTER TABLE trades
                ADD COLUMN IF NOT EXISTS risk_percentage FLOAT DEFAULT NULL
            """)

        # Execute migrations
        for i, migration in enumerate(migrations, 1):
            print(f"⏳ Executing migration {i}/{len(migrations)}...")
            await conn.execute(migration)
            print(f"✅ Migration {i} completed")

        print("")
        print("✅ All migrations completed successfully!")
        print("")

        # Verify columns were added
        verify_query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name = 'trades'
            AND column_name IN ('confluence_score', 'quality', 'risk_percentage')
            ORDER BY column_name
        """
        columns = await conn.fetch(verify_query)

        print("📋 Verification - New columns:")
        for col in columns:
            print(f"  ✓ {col['column_name']:<20} {col['data_type']:<25} (nullable: {col['is_nullable']})")

        print("=" * 70)

    except Exception as e:
        print("")
        print("=" * 70)
        print(f"❌ Migration failed: {e}")
        print("=" * 70)
        raise
    finally:
        await db.pool.release(conn)
        # DON'T close the pool - main bot needs it!
        # await db.close()


if __name__ == "__main__":
    asyncio.run(migrate())
