"""
Run database migration to add ML snapshot columns.
This script runs ONCE automatically on startup if needed.
"""

import asyncio
import asyncpg
import ssl
from pathlib import Path
from src.config import get_settings


async def run_migration():
    """Run database migration to add snapshot columns."""
    settings = get_settings()

    print("=" * 60)
    print("🔄 DATABASE MIGRATION - ML Snapshot Columns")
    print("=" * 60)

    conn = None
    try:
        # Connect to database
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        conn = await asyncpg.connect(settings.database_url, ssl=ssl_context)
        print("✅ Connected to database")

        # Read migration SQL
        migration_path = Path(__file__).parent / "migrate_snapshots.sql"

        if not migration_path.exists():
            print(f"❌ Migration file not found: {migration_path}")
            return False

        with open(migration_path, 'r', encoding='utf-8') as f:
            migration_sql = f.read()

        print("📜 Running migration script...")

        # Execute migration
        await conn.execute(migration_sql)

        print("=" * 60)
        print("✅ MIGRATION COMPLETE!")
        print("=" * 60)
        print("Added columns:")
        print("  - active_position: ai_reasoning, entry_snapshot, slippage, fill_time")
        print("  - trade_history: ai_reasoning, entry/exit snapshots, slippage, fill_time")
        print("Added indexes:")
        print("  - GIN indexes on JSONB columns")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Migration error: {e}")
        return False

    finally:
        if conn:
            await conn.close()


if __name__ == "__main__":
    success = asyncio.run(run_migration())
    exit(0 if success else 1)
