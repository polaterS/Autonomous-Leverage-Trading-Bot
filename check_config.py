import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

print("=" * 90)
print("TRADING_CONFIG TABLE VALUES")
print("=" * 90)

cur.execute('SELECT initial_capital, current_capital, starting_capital FROM trading_config WHERE id = 1')
row = cur.fetchone()

if row:
    initial = float(row[0]) if row[0] else 0
    current = float(row[1]) if row[1] else 0
    starting = float(row[2]) if row[2] else None

    print(f"initial_capital:  ${initial:.2f}")
    print(f"current_capital:  ${current:.2f}")
    print(f"starting_capital: ${starting:.2f}" if starting else "starting_capital: NULL")

    if starting:
        drawdown = ((starting - current) / starting) * 100
        print(f"\nCalculated drawdown: {drawdown:.1f}%")
else:
    print("No config found!")

cur.close()
conn.close()
