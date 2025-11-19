"""CORRECT analysis with proper column mapping"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

print(f"\n{'='*120}")
print("CORRECT ANALYSIS - PROPER COLUMN MAPPING")
print(f"{'='*120}\n")

# Get trade_history with column names
cur.execute("""
    SELECT
        symbol, side, leverage, entry_price, exit_price,
        realized_pnl_usd, close_reason, entry_time, exit_time,
        ai_confidence, trade_duration_seconds
    FROM trade_history
    ORDER BY entry_time DESC
    LIMIT 50
""")

trades = cur.fetchall()

if not trades:
    print("No trades found")
    conn.close()
    exit()

print(f"LAST 50 TRADES FROM trade_history:")
print(f"{'-'*120}\n")

# Stats
long_count = sum(1 for t in trades if t[1] == 'LONG')
short_count = sum(1 for t in trades if t[1] == 'SHORT')
winners = sum(1 for t in trades if t[5] and float(t[5]) > 0)
losers = sum(1 for t in trades if t[5] and float(t[5]) < 0)
total_pnl = sum(float(t[5] or 0) for t in trades)

print(f"SUMMARY:")
print(f"  Total trades:      {len(trades)}")
print(f"  LONG:              {long_count} ({long_count/len(trades)*100:.1f}%)")
print(f"  SHORT:             {short_count} ({short_count/len(trades)*100:.1f}%)")
print(f"  Winners:           {winners} ({winners/len(trades)*100:.1f}%)")
print(f"  Losers:            {losers} ({losers/len(trades)*100:.1f}%)")
print(f"  Total PNL:         ${total_pnl:+.2f}")

# Bias check
print(f"\nBIAS CHECK:")
if long_count > short_count * 2:
    print(f"  LONG BIAS! {long_count} LONG vs {short_count} SHORT")
elif short_count > long_count * 2:
    print(f"  SHORT BIAS! {short_count} SHORT vs {long_count} LONG")
else:
    print(f"  Balanced: {long_count} LONG vs {short_count} SHORT")

# Close reasons
print(f"\nCLOSE REASONS (Top 10):")
reasons = {}
for t in trades:
    r = t[6] if t[6] else 'Unknown'
    reasons[r] = reasons.get(r, 0) + 1

for i, (reason, count) in enumerate(sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:10], 1):
    # Remove emojis for Windows console
    reason_clean = reason.encode('ascii', 'ignore').decode('ascii') if reason else 'Unknown'
    print(f"  {i:2}. {reason_clean[:70]:70} -> {count:3} times")

# Show last 15 trades
print(f"\n\nLAST 15 TRADES:")
print(f"{'='*120}\n")

for i, t in enumerate(trades[:15], 1):
    symbol, side, leverage, entry, exit_p, pnl, reason, entry_time, exit_time, confidence, duration = t

    print(f"{'-'*120}")
    print(f"#{i:2} | {symbol:12} | {side:5} | {leverage}x | {entry_time}")
    print(f"{'-'*120}")
    print(f"  Entry:  ${float(entry):.4f}")
    print(f"  Exit:   ${float(exit_p):.4f}")
    print(f"  PNL:    ${float(pnl):+.2f}")
    reason_clean = reason.encode('ascii', 'ignore').decode('ascii') if reason else 'Unknown'
    print(f"  Reason: {reason_clean[:70]}")
    if duration:
        if duration < 60:
            print(f"  Duration: {duration}s")
        elif duration < 3600:
            print(f"  Duration: {duration/60:.1f} minutes")
        else:
            print(f"  Duration: {duration/3600:.1f} hours")
    if confidence:
        print(f"  AI Confidence: {float(confidence)*100:.1f}%")
    print()

# Active positions
print(f"\n{'='*120}")
print("ACTIVE POSITIONS:")
print(f"{'='*120}\n")

cur.execute("""
    SELECT symbol, side, leverage, entry_price, current_price,
           unrealized_pnl_usd, stop_loss_price, entry_time
    FROM active_position
    ORDER BY entry_time DESC
""")

active = cur.fetchall()

if active:
    print(f"Found {len(active)} open positions:\n")
    for i, pos in enumerate(active, 1):
        symbol, side, lev, entry, current, upnl, sl, etime = pos
        print(f"{i}. {symbol:12} {side:5} {lev}x | Entry: ${float(entry):.4f} | Current: ${float(current):.4f} | PNL: ${float(upnl):+.2f}")
        print(f"   Stop Loss: ${float(sl):.4f} | Open since: {etime}")
        print()
else:
    print("No open positions")

print(f"{'='*120}\n")

conn.close()
