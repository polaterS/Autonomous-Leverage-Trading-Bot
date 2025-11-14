"""
Fix starting capital in database to $100 (current live balance)
This will eliminate the false 89.6% drawdown warning
"""
import os
import psycopg2
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

print("=" * 90)
print("FIXING STARTING CAPITAL - STEP BY STEP")
print("=" * 90)

# Step 1: Check current state
print("\n1. CURRENT STATE:")
print("-" * 90)

cur.execute("SELECT id, initial_capital FROM trading_config WHERE id = 1")
config = cur.fetchone()
if config:
    print(f"   trading_config.initial_capital: ${float(config[1]):.2f}")
else:
    print("   No config found!")

cur.execute("SELECT date, starting_capital, ending_capital FROM daily_performance ORDER BY date DESC LIMIT 1")
daily = cur.fetchone()
if daily:
    print(f"   daily_performance.starting_capital: ${float(daily[1]):.2f}")
    print(f"   daily_performance.ending_capital: ${float(daily[2]):.2f}")
else:
    print("   No daily performance found!")

# Step 2: Get real account balance
print("\n2. REAL ACCOUNT BALANCE:")
print("-" * 90)
real_balance = Decimal('107.39')  # Current balance from Telegram
print(f"   Real balance (from Telegram): ${float(real_balance):.2f}")

# Step 3: Fix trading_config
print("\n3. UPDATING trading_config.initial_capital...")
print("-" * 90)
cur.execute("""
    UPDATE trading_config
    SET initial_capital = %s
    WHERE id = 1
""", (str(Decimal('100.00')),))
print(f"   ✅ Set to $100.00")

# Step 4: Fix daily_performance (today's starting capital)
print("\n4. UPDATING daily_performance.starting_capital...")
print("-" * 90)
cur.execute("""
    UPDATE daily_performance
    SET starting_capital = %s
    WHERE date = CURRENT_DATE
""", (str(Decimal('100.00')),))
rows_updated = cur.rowcount
print(f"   ✅ Updated {rows_updated} row(s) to $100.00")

# Commit changes
conn.commit()

# Step 5: Verify fixes
print("\n5. VERIFICATION:")
print("-" * 90)

cur.execute("SELECT id, initial_capital FROM trading_config WHERE id = 1")
config = cur.fetchone()
print(f"   trading_config.initial_capital: ${float(config[1]):.2f}")

cur.execute("SELECT date, starting_capital, ending_capital FROM daily_performance ORDER BY date DESC LIMIT 1")
daily = cur.fetchone()
print(f"   daily_performance.starting_capital: ${float(daily[1]):.2f}")
print(f"   daily_performance.ending_capital: ${float(daily[2]):.2f}")

# Calculate new drawdown
starting = float(daily[1])
ending = float(daily[2])
drawdown = ((starting - ending) / starting) * 100 if starting > 0 else 0

print(f"\n   📊 NEW DRAWDOWN: {drawdown:.1f}%")
if drawdown < 20:
    print(f"   ✅ Drawdown is now SAFE (< 20% threshold)")
else:
    print(f"   ⚠️  Drawdown still high (>= 20% threshold)")

cur.close()
conn.close()

print("\n" + "=" * 90)
print("✅ DATABASE FIXED!")
print("=" * 90)
print("\nNext step: Revert max_drawdown threshold from 95% back to 20%")
print("(This will be done in src/risk_manager.py)")
