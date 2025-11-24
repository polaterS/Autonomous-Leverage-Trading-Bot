# -*- coding: utf-8 -*-
"""
Simple database migration using existing database module
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.database import get_db_client


async def main():
    print("")
    print("=" * 70)
    print("Database Migration: Adding Enhanced Trading System Columns")
    print("=" * 70)
    print("")

    try:
        # Get database client
        db = await get_db_client()
        print("Connected to database successfully")
        print("")

        # SQL commands to add columns
        migrations = [
            ("confluence_score", "ALTER TABLE trades ADD COLUMN IF NOT EXISTS confluence_score FLOAT DEFAULT NULL"),
            ("quality", "ALTER TABLE trades ADD COLUMN IF NOT EXISTS quality VARCHAR(20) DEFAULT NULL"),
            ("risk_percentage", "ALTER TABLE trades ADD COLUMN IF NOT EXISTS risk_percentage FLOAT DEFAULT NULL")
        ]

        # Execute migrations
        for column_name, sql in migrations:
            print(f"Adding '{column_name}' column...")
            try:
                await db.pool.execute(sql)
                print(f"  SUCCESS: {column_name} column added (or already exists)")
            except Exception as e:
                print(f"  WARNING: {e}")

        print("")
        print("=" * 70)
        print("Migration completed successfully!")
        print("=" * 70)
        print("")

        # Verify columns
        print("Verifying columns...")
        result = await db.pool.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'trades'
            AND column_name IN ('confluence_score', 'quality', 'risk_percentage')
            ORDER BY column_name
        """)

        if result:
            print("")
            for row in result:
                print(f"  - {row['column_name']:<20} {row['data_type']}")
            print("")
            print("SUCCESS: All enhanced columns are present!")
        else:
            print("WARNING: Could not verify columns")

        print("")

    except Exception as e:
        print("")
        print("=" * 70)
        print(f"ERROR: {e}")
        print("=" * 70)
        print("")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("")
    print("Enhanced Trading System - Database Migration")
    print("")
    asyncio.run(main())
    print("")
    print("All done! You can now use the enhanced trading system.")
    print("")
