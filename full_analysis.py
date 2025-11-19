"""Complete position analysis - Active + History"""
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

print("\n" + "="*120)
print("COMPLETE POSITION ANALYSIS - RAILWAY DATABASE")
print("="*120 + "\n")

# 1. Check trade_history columns first
cur.execute("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'trade_history'
    ORDER BY ordinal_position
""")
history_cols = [col[0] for col in cur.fetchall()]
print(f"Trade history columns: {', '.join(history_cols[:10])}...")

#2. Get closed positions (trade_history)
cur.execute("""
    SELECT COUNT(*) FROM trade_history
""")
total_history = cur.fetchone()[0]
print(f"\nTotal trade history records: {total_history}")

# Get recent history
cur.execute("""
    SELECT * FROM trade_history
    ORDER BY entry_time DESC
    LIMIT 50
""")
history = cur.fetchall()

# 3. Get open positions
cur.execute("""
    SELECT * FROM active_position
    ORDER BY entry_time DESC
""")
active = cur.fetchall()

print(f"Active positions: {len(active)}")
print(f"Recent history: {len(history)}")

all_positions = []

# Process active positions
for pos in active:
    all_positions.append({
        'type': 'ACTIVE',
        'symbol': pos[1],
        'side': pos[2],
        'leverage': pos[3],
        'entry_price': pos[4],
        'current_price': pos[5],
        'quantity': pos[6],
        'position_value': pos[7],
        'stop_loss': pos[8],
        'take_profit': pos[10] if len(pos) > 10 else None,
        'unrealized_pnl': pos[13],
        'confidence': pos[17] if len(pos) > 17 else None,
        'entry_time': pos[18],
        'close_reason': None,
        'closed_at': None,
        'realized_pnl': None
    })

# Process history (need to map columns correctly)
# Common columns: symbol, side, leverage, entry_price, exit_price, quantity, pnl, entry_time, exit_time
for pos in history:
    all_positions.append({
        'type': 'CLOSED',
        'symbol': pos[1] if len(pos) > 1 else 'UNKNOWN',
        'side': pos[2] if len(pos) > 2 else 'UNKNOWN',
        'leverage': pos[3] if len(pos) > 3 else 0,
        'entry_price': pos[4] if len(pos) > 4 else 0,
        'exit_price': pos[5] if len(pos) > 5 else None,
        'quantity': pos[6] if len(pos) > 6 else 0,
        'position_value': pos[7] if len(pos) > 7 else 0,
        'stop_loss': pos[8] if len(pos) > 8 else None,
        'take_profit': None,
        'unrealized_pnl': None,
        'confidence': pos[16] if len(pos) > 16 else None,
        'entry_time': pos[17] if len(pos) > 17 else None,
        'close_reason': pos[21] if len(pos) > 21 else 'Unknown',
        'closed_at': pos[18] if len(pos) > 18 else None,
        'realized_pnl': pos[12] if len(pos) > 12 else 0
    })

# Sort by entry time (handle None values and timezone)
def get_sort_key(x):
    t = x['entry_time']
    if not t or not isinstance(t, datetime):
        return datetime.min
    # Remove timezone for comparison
    if t.tzinfo:
        return t.replace(tzinfo=None)
    return t

all_positions.sort(key=get_sort_key, reverse=True)

print(f"\n" + "="*120)
print(f"ANALYSIS OF {len(all_positions)} TOTAL POSITIONS")
print("="*120 + "\n")

# Stats
long_count = sum(1 for p in all_positions if p['side'] == 'LONG')
short_count = sum(1 for p in all_positions if p['side'] == 'SHORT')
open_count = sum(1 for p in all_positions if p['type'] == 'ACTIVE')
closed_count = sum(1 for p in all_positions if p['type'] == 'CLOSED')

print(f"SUMMARY STATISTICS:")
print(f"{'-'*120}")
print(f"Total Positions:     {len(all_positions)}")
print(f"  LONG:              {long_count} ({long_count/len(all_positions)*100 if all_positions else 0:.1f}%)")
print(f"  SHORT:             {short_count} ({short_count/len(all_positions)*100 if all_positions else 0:.1f}%)")
print(f"\nPosition Status:")
print(f"  Open:              {open_count}")
print(f"  Closed:            {closed_count}")

# BIAS CHECK
print(f"\nBIAS DETECTION:")
print(f"{'-'*120}")
if long_count > short_count * 2:
    print(f"  STRONG LONG BIAS DETECTED!")
    print(f"  {long_count} LONG vs {short_count} SHORT (ratio: {long_count/(short_count if short_count > 0 else 1):.1f}:1)")
    print(f"  WARNING: PA analysis may not be working - system only opening longs!")
elif short_count > long_count * 2:
    print(f"  STRONG SHORT BIAS DETECTED!")
    print(f"  {short_count} SHORT vs {long_count} LONG (ratio: {short_count/(long_count if long_count > 0 else 1):.1f}:1)")
    print(f"  WARNING: PA analysis may not be working - system only opening shorts!")
else:
    print(f"  Balanced: {long_count} LONG vs {short_count} SHORT - PA analysis working correctly")

# Close reasons
closed = [p for p in all_positions if p['type'] == 'CLOSED']
if closed:
    print(f"\nCLOSE REASONS (Top 10):")
    print(f"{'-'*120}")
    reasons = {}
    for p in closed:
        r = p['close_reason'] if p['close_reason'] else 'Unknown'
        reasons[r] = reasons.get(r, 0) + 1

    for i, (reason, count) in enumerate(sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:10], 1):
        pct = count/len(closed)*100
        print(f"  {i:2}. {reason:70} -> {count:3} times ({pct:.1f}%)")

    # PNL
    winners = [p for p in closed if p['realized_pnl'] and float(p['realized_pnl']) > 0]
    losers = [p for p in closed if p['realized_pnl'] and float(p['realized_pnl']) < 0]
    total_pnl = sum(float(p['realized_pnl'] or 0) for p in closed)

    print(f"\nPNL STATISTICS:")
    print(f"{'-'*120}")
    print(f"  Winners:           {len(winners)}/{len(closed)} ({len(winners)/len(closed)*100:.1f}% win rate)")
    print(f"  Losers:            {len(losers)}/{len(closed)} ({len(losers)/len(closed)*100:.1f}%)")
    print(f"  Total PNL:         ${total_pnl:+.2f}")

# Show last 20 positions
print(f"\n\nLAST 20 POSITIONS (DETAILED):")
print(f"{'='*120}\n")

for i, pos in enumerate(all_positions[:20], 1):
    status_emoji = "[OPEN]" if pos['type'] == 'ACTIVE' else "[CLOSED]"
    print(f"{'-'*120}")
    print(f"#{i:2} {status_emoji} {pos['symbol']:12} {pos['side']:5} | {pos['entry_time']}")
    print(f"{'-'*120}")

    print(f"  POSITION:")
    print(f"    Entry Price:      ${float(pos['entry_price']):.4f}")
    if pos['type'] == 'CLOSED' and pos['exit_price']:
        print(f"    Exit Price:       ${float(pos['exit_price']):.4f}")
    elif pos['type'] == 'ACTIVE':
        print(f"    Current Price:    ${float(pos['current_price']):.4f}")

    print(f"    Quantity:         {float(pos['quantity']):.6f}")
    print(f"    Position Value:   ${float(pos['position_value']):.2f}")
    print(f"    Leverage:         {pos['leverage']}x")

    # PA Analysis
    print(f"\n  PA ANALYSIS:")
    if pos['confidence']:
        print(f"    Confidence:       {float(pos['confidence'])*100:.1f}%")
    else:
        print(f"    Confidence:       [NO DATA - BUG!]")

    # Risk
    print(f"\n  RISK MANAGEMENT:")
    if pos['stop_loss']:
        sl_dist = abs(float(pos['entry_price']) - float(pos['stop_loss'])) / float(pos['entry_price']) * 100
        print(f"    Stop Loss:        ${float(pos['stop_loss']):.4f} ({sl_dist:.2f}% from entry)")
    else:
        print(f"    Stop Loss:        [NOT SET - BUG!]")

    # Results
    if pos['type'] == 'CLOSED':
        print(f"\n  RESULTS:")
        pnl = float(pos['realized_pnl'] or 0)
        print(f"    PNL:              ${pnl:+.2f}")
        print(f"    Close Reason:     {pos['close_reason'] or 'Unknown'}")

        if pos['closed_at'] and pos['entry_time']:
            duration = (pos['closed_at'] - pos['entry_time']).total_seconds()
            if duration < 60:
                print(f"    Duration:         {duration:.0f}s [WARNING: TOO FAST!]")
            elif duration < 3600:
                print(f"    Duration:         {duration/60:.1f} minutes")
            else:
                print(f"    Duration:         {duration/3600:.1f} hours")
    else:
        print(f"\n  CURRENT STATUS:")
        upnl = float(pos['unrealized_pnl'] or 0)
        print(f"    Unrealized PNL:   ${upnl:+.2f}")

    print()

# FINAL HEALTH CHECK
print(f"\n{'='*120}")
print(f"SYSTEM HEALTH CHECK:")
print(f"{'='*120}\n")

issues = []

if long_count > short_count * 2:
    issues.append(f"[CRITICAL] Strong LONG bias - PA analysis may be broken!")
elif short_count > long_count * 2:
    issues.append(f"[CRITICAL] Strong SHORT bias - PA analysis may be broken!")

no_conf = sum(1 for p in all_positions if not p['confidence'])
if no_conf > 0:
    issues.append(f"[WARNING] {no_conf} positions without confidence score")

no_sl = sum(1 for p in all_positions if not p['stop_loss'])
if no_sl > 0:
    issues.append(f"[CRITICAL] {no_sl} positions without stop loss!")

very_fast = [p for p in closed if p['closed_at'] and p['entry_time'] and (p['closed_at'] - p['entry_time']).total_seconds() < 60]
if very_fast:
    issues.append(f"[WARNING] {len(very_fast)} positions closed in <60 seconds")

if issues:
    print("ISSUES FOUND:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("No major issues - system healthy!")

print(f"\n{'='*120}\n")

conn.close()
