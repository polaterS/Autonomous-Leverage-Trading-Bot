import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# Get today's performance
print("=" * 90)
print("LIVE TRADING STATUS")
print("=" * 90)

cur.execute('''
    SELECT date, starting_capital, ending_capital, daily_pnl,
           total_trades, winning_trades, losing_trades, win_rate
    FROM daily_performance
    ORDER BY date DESC LIMIT 1
''')

result = cur.fetchone()
if result:
    date, start_cap, end_cap, pnl, trades, wins, losses, wr = result
    print(f'Date: {date}')
    print(f'Starting Capital: ${float(start_cap):.2f}')
    print(f'Current Capital: ${float(end_cap):.2f}')
    print(f'Daily P&L: ${float(pnl):.2f}')
    print(f'Total Trades: {trades} (Wins: {wins}, Losses: {losses})')
    print(f'Win Rate: {wr:.1f}%' if wr else 'Win Rate: N/A')

# Get open positions
cur.execute('SELECT COUNT(*) FROM active_position')
open_count = cur.fetchone()[0]
print(f'\nOpen Positions: {open_count}/2')

# Get recent trades
print("\n" + "=" * 90)
print("RECENT TRADES (Last 10)")
print("=" * 90)

cur.execute('''
    SELECT symbol, side, entry_price, exit_price, realized_pnl_usd,
           pnl_percent, is_winner, exit_time, close_reason
    FROM trade_history
    ORDER BY exit_time DESC
    LIMIT 10
''')

trades = cur.fetchall()
for t in trades:
    symbol, side, entry, exit_p, pnl, pnl_pct, winner, exit_time, reason = t
    status = "WIN " if winner else "LOSS"
    print(f'{symbol:12} {side:5} | ${float(entry):8.4f} -> ${float(exit_p):8.4f} | '
          f'${float(pnl):6.2f} ({float(pnl_pct):5.1f}%) | {status} | {str(exit_time)[5:16]}')

cur.close()
conn.close()
