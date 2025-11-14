import os
import psycopg2
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

print("=" * 90)
print("FIXING trading_config.starting_capital")
print("=" * 90)

# Check current
cur.execute('SELECT starting_capital, initial_capital, current_capital FROM trading_config WHERE id = 1')
row = cur.fetchone()
print(f"\nBEFORE:")
print(f"  starting_capital: ${float(row[0]):.2f}")
print(f"  initial_capital:  ${float(row[1]):.2f}")
print(f"  current_capital:  ${float(row[2]):.2f}")

# Fix starting_capital to $100
print(f"\nUPDATING starting_capital to $100.00...")
cur.execute("""
    UPDATE trading_config
    SET starting_capital = %s
    WHERE id = 1
""", (str(Decimal('100.00')),))

conn.commit()

# Verify
cur.execute('SELECT starting_capital, initial_capital, current_capital FROM trading_config WHERE id = 1')
row = cur.fetchone()
print(f"\nAFTER:")
print(f"  starting_capital: ${float(row[0]):.2f}")
print(f"  initial_capital:  ${float(row[1]):.2f}")
print(f"  current_capital:  ${float(row[2]):.2f}")

# Calculate new drawdown
starting = float(row[0])
current = float(row[2])
drawdown = ((starting - current) / starting) * 100 if starting > 0 else 0

print(f"\nNEW DRAWDOWN: {drawdown:.1f}%")
if drawdown < 20 and drawdown >= 0:
    print("OK - Drawdown is safe (< 20%)")
elif drawdown < 0:
    print(f"OK - In PROFIT (+{abs(drawdown):.1f}%)")
else:
    print("WARNING - Drawdown >= 20%")

cur.close()
conn.close()

print("\n" + "=" * 90)
print("DONE! Restart Railway to apply.")
print("=" * 90)
