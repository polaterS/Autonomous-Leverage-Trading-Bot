import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

print("=" * 100)
print("LEVERAGE CHECK - LAST 10 TRADES")
print("=" * 100)

cur.execute('''
    SELECT symbol, side, leverage, entry_price, exit_price,
           realized_pnl_usd, entry_time, exit_time
    FROM trade_history
    ORDER BY entry_time DESC
    LIMIT 10
''')

trades = cur.fetchall()
for t in trades:
    symbol, side, leverage, entry, exit_p, pnl, entry_time, exit_time = t
    print(f'{symbol:15} {side:5} | LEVERAGE: {leverage:2}x | '
          f'Entry: ${float(entry):8.4f} -> Exit: ${float(exit_p):8.4f} | '
          f'PnL: ${float(pnl):6.2f} | {str(entry_time)[5:19]}')

# Check active position table schema
print("\n" + "=" * 100)
print("ACTIVE POSITION TABLE SCHEMA")
print("=" * 100)

cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'active_position'
    ORDER BY ordinal_position
""")

cols = cur.fetchall()
for col in cols:
    print(f'  {col[0]}: {col[1]}')

# Check if there are any active positions
print("\n" + "=" * 100)
print("CHECKING ACTIVE POSITIONS")
print("=" * 100)

cur.execute('SELECT * FROM active_position')
active = cur.fetchall()

if active:
    print(f"Found {len(active)} active position(s)")
    # Print all column values
    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'active_position'
        ORDER BY ordinal_position
    """)
    column_names = [row[0] for row in cur.fetchall()]

    for pos in active:
        print("\nPosition details:")
        for i, col in enumerate(column_names):
            print(f"  {col}: {pos[i]}")
else:
    print("No active positions found")

cur.close()
conn.close()
