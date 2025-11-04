"""
Database cleanup script - Fix duplicate trading_config rows
Run this ONCE to clean up Railway database
"""

import asyncio
import asyncpg
from decimal import Decimal
from datetime import date

# Railway database URL (from environment or config)
DATABASE_URL = "postgresql://postgres:Dd18102002.@junction.proxy.rlwy.net:17619/railway"


async def cleanup_database():
    """Clean up duplicate trading_config rows and fix capital"""
    print("Connecting to Railway database...")

    try:
        conn = await asyncpg.connect(DATABASE_URL)
        print("Connected to database!\n")

        # Step 1: Show current state
        print(" STEP 1: Current trading_config state")
        print("-" * 60)
        rows = await conn.fetch("""
            SELECT id, current_capital, last_updated
            FROM trading_config
            ORDER BY id
        """)

        for row in rows:
            print(f"  ID: {row['id']:3d} | Capital: ${row['current_capital']:12.2f} | Updated: {row['last_updated']}")

        print(f"\n  Total rows: {len(rows)}")

        # Step 2: Delete duplicates (keep only id=1)
        print("\n  STEP 2: Deleting duplicate config rows...")
        print("-" * 60)

        delete_result = await conn.execute("DELETE FROM trading_config WHERE id != 1")
        deleted_count = int(delete_result.split()[-1])  # Extract count from "DELETE X"
        print(f"   Deleted {deleted_count} duplicate rows")

        # Step 3: Get latest capital from daily_performance
        print("\n STEP 3: Fetching current capital from daily_performance...")
        print("-" * 60)

        today = date.today()
        daily_perf = await conn.fetchrow("""
            SELECT ending_capital
            FROM daily_performance
            WHERE date = $1
            LIMIT 1
        """, today)

        if daily_perf:
            correct_capital = daily_perf['ending_capital']
            print(f"   Today's ending capital: ${correct_capital:.2f}")
        else:
            print("    No daily_performance for today, keeping current value")
            correct_capital = None

        # Step 4: Update capital in trading_config
        if correct_capital:
            print("\n STEP 4: Updating trading_config capital...")
            print("-" * 60)

            await conn.execute("""
                UPDATE trading_config
                SET current_capital = $1,
                    last_updated = NOW()
                WHERE id = 1
            """, correct_capital)

            print(f"   Updated trading_config id=1 to ${correct_capital:.2f}")

        # Step 5: Verify final state
        print("\n STEP 5: Final verification")
        print("-" * 60)

        final_config = await conn.fetchrow("SELECT * FROM trading_config WHERE id = 1")

        if final_config:
            print(f"  ID: {final_config['id']}")
            print(f"  Initial Capital: ${final_config['initial_capital']:.2f}")
            print(f"  Current Capital: ${final_config['current_capital']:.2f}")
            print(f"  Last Updated: {final_config['last_updated']}")
            print(f"  Trading Enabled: {final_config['is_trading_enabled']}")
        else:
            print("   ERROR: trading_config id=1 not found!")

        # Close connection
        await conn.close()

        print("\n" + "=" * 60)
        print(" DATABASE CLEANUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n Summary:")
        print(f"   Deleted {deleted_count} duplicate config rows")
        print(f"   Capital updated to: ${correct_capital:.2f}" if correct_capital else "   Capital unchanged")
        print(f"   Single config row (id=1) maintained")
        print("\n You can now restart your Railway bot!")

    except Exception as e:
        print(f"\n ERROR: {e}")
        print("\nIf you see connection error, check:")
        print("  1. DATABASE_URL is correct")
        print("  2. Railway database is accessible")
        print("  3. Network connection is working")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print(" RAILWAY DATABASE CLEANUP SCRIPT")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Show all trading_config rows")
    print("  2. Delete duplicates (keep only id=1)")
    print("  3. Update capital from daily_performance")
    print("  4. Verify final state")
    print("\n  WARNING: This will modify your Railway database!")
    print("=" * 60)

    # Run cleanup
    asyncio.run(cleanup_database())
