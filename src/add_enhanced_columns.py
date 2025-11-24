# -*- coding: utf-8 -*-
"""
Database Migration: Add Enhanced Trading System Columns
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')


async def add_enhanced_columns():
    """Add new columns to trades table."""
    print("=" * 70)
    print("Database Migration: Adding Enhanced Trading System Columns")
    print("=" * 70)
    print("")

    if not DATABASE_URL:
        print("ERROR: DATABASE_URL not found")
        return

    print(f"Connecting to database...")
    print("")

    try:
        conn = await asyncpg.connect(DATABASE_URL)
        print("Connected successfully")
        print("")

        print("Checking existing columns...")
        existing_columns = await conn.fetch("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'trades'
        """)
        existing_column_names = [row['column_name'] for row in existing_columns]
        print(f"Found {len(existing_column_names)} existing columns")
        print("")

        # Add confluence_score column
        if 'confluence_score' not in existing_column_names:
            print("Adding 'confluence_score' column...")
            await conn.execute("""
                ALTER TABLE trades
                ADD COLUMN confluence_score FLOAT DEFAULT NULL
            """)
            print("  SUCCESS: confluence_score column added")
        else:
            print("  INFO: confluence_score column already exists")

        # Add quality column
        if 'quality' not in existing_column_names:
            print("Adding 'quality' column...")
            await conn.execute("""
                ALTER TABLE trades
                ADD COLUMN quality VARCHAR(20) DEFAULT NULL
            """)
            print("  SUCCESS: quality column added")
        else:
            print("  INFO: quality column already exists")

        # Add risk_percentage column
        if 'risk_percentage' not in existing_column_names:
            print("Adding 'risk_percentage' column...")
            await conn.execute("""
                ALTER TABLE trades
                ADD COLUMN risk_percentage FLOAT DEFAULT NULL
            """)
            print("  SUCCESS: risk_percentage column added")
        else:
            print("  INFO: risk_percentage column already exists")

        print("")
        print("=" * 70)
        print("Migration completed successfully!")
        print("=" * 70)
        print("")

        await conn.close()
        print("Database connection closed")
        print("")

    except Exception as e:
        print("")
        print("=" * 70)
        print(f"ERROR during migration: {e}")
        print("=" * 70)
        print("")
        raise


if __name__ == "__main__":
    print("")
    print("Enhanced Trading System - Database Migration")
    print("")
    asyncio.run(add_enhanced_columns())
    print("All done!")
    print("")
