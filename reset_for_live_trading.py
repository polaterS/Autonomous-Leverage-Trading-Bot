"""
Reset database for live trading with $100 capital.
This script will:
1. Clear daily performance (reset to $100 starting capital)
2. Keep trade history for ML training
3. Keep active positions
"""

import os
import sys
import psycopg2
from decimal import Decimal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
INITIAL_CAPITAL = Decimal('100.00')

def reset_for_live_trading():
    """Reset database for live trading with $100 capital."""

    print("=" * 60)
    print("🔧 RESETTING DATABASE FOR LIVE TRADING ($100)")
    print("=" * 60)

    try:
        # Connect to database
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        print("\n1. Clearing daily performance...")
        cur.execute("DELETE FROM daily_performance")

        print("2. Resetting config to $100 capital...")
        cur.execute("""
            UPDATE trading_config
            SET initial_capital = %s,
                updated_at = NOW()
            WHERE id = 1
        """, (str(INITIAL_CAPITAL),))

        print("3. Inserting today's starting capital...")
        cur.execute("""
            INSERT INTO daily_performance (date, starting_capital, current_capital)
            VALUES (CURRENT_DATE, %s, %s)
            ON CONFLICT (date) DO UPDATE
            SET starting_capital = %s,
                current_capital = %s,
                total_trades = 0,
                winning_trades = 0,
                losing_trades = 0,
                total_profit = 0,
                total_loss = 0,
                net_profit = 0
        """, (str(INITIAL_CAPITAL), str(INITIAL_CAPITAL), str(INITIAL_CAPITAL), str(INITIAL_CAPITAL)))

        # Commit changes
        conn.commit()

        print("\n" + "=" * 60)
        print("✅ DATABASE RESET COMPLETE!")
        print("=" * 60)
        print(f"💰 Starting Capital: ${INITIAL_CAPITAL}")
        print(f"📊 Trade History: KEPT (for ML training)")
        print(f"🔄 Daily Performance: RESET")
        print("=" * 60)
        print("\n🚀 Ready for live trading!")
        print("   Railway will auto-restart and begin trading.\n")

        cur.close()
        conn.close()

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n⚠️  WARNING: This will reset daily performance to $100!")
    print("   Trade history will be KEPT for ML training.")

    response = input("\nContinue? (yes/no): ").strip().lower()

    if response == 'yes':
        if reset_for_live_trading():
            print("\n✅ Success! Redeploy Railway to start live trading.")
            sys.exit(0)
        else:
            print("\n❌ Failed!")
            sys.exit(1)
    else:
        print("\n❌ Cancelled.")
        sys.exit(1)
