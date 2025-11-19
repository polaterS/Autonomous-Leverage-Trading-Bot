"""Database position analysis using psycopg2"""
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Connect to database
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# Get all positions
cur.execute("""
    SELECT
        id, symbol, side, leverage, entry_price, exit_price,
        quantity, position_value_usd, stop_loss_price, take_profit_price,
        unrealized_pnl, realized_pnl, close_reason, status,
        created_at, closed_at, confidence_score, confluence_factors
    FROM positions
    ORDER BY created_at DESC
    LIMIT 50
""")

positions = cur.fetchall()

if not positions:
    print("❌ No positions found")
    conn.close()
    exit()

print(f"\n{'='*120}")
print(f"📊 ANALYZING LAST {len(positions)} POSITIONS FROM RAILWAY DATABASE")
print(f"{'='*120}\n")

# Calculate stats
long_count = sum(1 for p in positions if p[2] == 'LONG')
short_count = sum(1 for p in positions if p[2] == 'SHORT')
open_count = sum(1 for p in positions if p[13] == 'open')
closed_count = sum(1 for p in positions if p[13] == 'closed')

print(f"📈 SUMMARY STATISTICS:")
print(f"{'─'*120}")
print(f"Total Positions:     {len(positions)}")
print(f"  ├─ LONG:           {long_count} ({long_count/len(positions)*100:.1f}%)")
print(f"  └─ SHORT:          {short_count} ({short_count/len(positions)*100:.1f}%)")
print(f"\nPosition Status:")
print(f"  ├─ Open:           {open_count}")
print(f"  └─ Closed:         {closed_count}")

# Bias detection
print(f"\n⚠️ BIAS DETECTION:")
print(f"{'─'*120}")
if long_count > short_count * 2:
    print(f"  🚨 STRONG LONG BIAS DETECTED!")
    print(f"  {long_count} LONG vs {short_count} SHORT")
    print(f"  ⚠️ PA analysis may be broken - system favoring longs!")
elif short_count > long_count * 2:
    print(f"  🚨 STRONG SHORT BIAS DETECTED!")
    print(f"  {short_count} SHORT vs {long_count} LONG")
    print(f"  ⚠️ PA analysis may be broken - system favoring shorts!")
else:
    print(f"  ✅ Balanced distribution: {long_count} LONG vs {short_count} SHORT")

# PA analysis check
no_pa_count = sum(1 for p in positions if not p[16])  # confidence_score
if no_pa_count > 0:
    print(f"\n🚨 CRITICAL PA ANALYSIS ISSUE:")
    print(f"{'─'*120}")
    print(f"  ⚠️ {no_pa_count}/{len(positions)} positions opened WITHOUT PA analysis!")
    print(f"  This is a BUG - every position should have confidence score!")

# Close reasons analysis
closed_positions = [p for p in positions if p[13] == 'closed']
if closed_positions:
    print(f"\n📋 CLOSE REASONS (Top 5):")
    print(f"{'─'*120}")
    close_reasons = {}
    for p in closed_positions:
        reason = p[12] if p[12] else 'Unknown'
        close_reasons[reason] = close_reasons.get(reason, 0) + 1

    for i, (reason, count) in enumerate(sorted(close_reasons.items(), key=lambda x: x[1], reverse=True)[:5], 1):
        print(f"  {i}. {reason:80} → {count:3} times")

    # PNL analysis
    winners = [p for p in closed_positions if p[11] and float(p[11]) > 0]
    losers = [p for p in closed_positions if p[11] and float(p[11]) < 0]
    total_pnl = sum(float(p[11] or 0) for p in closed_positions)

    print(f"\n💰 PNL STATISTICS:")
    print(f"{'─'*120}")
    print(f"  Winning trades:    {len(winners)}/{len(closed_positions)} ({len(winners)/len(closed_positions)*100:.1f}%)")
    print(f"  Losing trades:     {len(losers)}/{len(closed_positions)} ({len(losers)/len(closed_positions)*100:.1f}%)")
    print(f"  Total PNL:         ${total_pnl:+.2f}")

# Show last 15 positions in detail
print(f"\n\n📊 LAST 15 POSITIONS (DETAILED VIEW):")
print(f"{'='*120}\n")

for i, pos in enumerate(positions[:15], 1):
    print(f"{'─'*120}")
    print(f"#{i:2} | {pos[1]:12} | {pos[2]:5} | {pos[13].upper():6} | Created: {pos[14]}")
    print(f"{'─'*120}")

    # Position details
    print(f"  💼 POSITION:")
    print(f"     Entry Price:      ${float(pos[4]):.4f}")
    if pos[5]:
        print(f"     Exit Price:       ${float(pos[5]):.4f}")
    print(f"     Quantity:         {float(pos[6]):.6f}")
    print(f"     Position Value:   ${float(pos[7]):.2f}")
    print(f"     Leverage:         {pos[3]}x")

    # PA Analysis
    print(f"\n  🎯 PA ANALYSIS:")
    if pos[16]:  # confidence_score
        print(f"     Confidence:       {float(pos[16])*100:.1f}%")
        if pos[17]:  # confluence_factors
            factors = str(pos[17])[:100]
            print(f"     Confluence:       {factors}...")
    else:
        print(f"     ❌❌❌ NO PA ANALYSIS! POSITION OPENED WITHOUT ANALYSIS! ❌❌❌")

    # Risk management
    print(f"\n  📊 RISK MANAGEMENT:")
    if pos[8]:  # stop_loss_price
        sl_distance = abs(float(pos[4]) - float(pos[8])) / float(pos[4]) * 100
        print(f"     Stop Loss:        ${float(pos[8]):.4f} ({sl_distance:.2f}% from entry)")
    else:
        print(f"     Stop Loss:        ❌ NOT SET - CRITICAL BUG!")

    if pos[9]:  # take_profit_price
        tp_distance = abs(float(pos[9]) - float(pos[4])) / float(pos[4]) * 100
        print(f"     Take Profit:      ${float(pos[9]):.4f} ({tp_distance:.2f}% from entry)")
    else:
        print(f"     Take Profit:      (Not set)")

    # Results
    if pos[13] == 'closed':
        print(f"\n  💰 CLOSED POSITION:")
        pnl = float(pos[11] or 0)
        print(f"     Realized PNL:     ${pnl:+.2f}")
        print(f"     Close Reason:     {pos[12] or 'Unknown'}")

        if pos[15]:  # closed_at
            duration = (pos[15] - pos[14]).total_seconds()
            if duration < 60:
                print(f"     Duration:         {duration:.0f} seconds ⚠️⚠️ TOO FAST!")
            elif duration < 3600:
                print(f"     Duration:         {duration/60:.1f} minutes")
            else:
                print(f"     Duration:         {duration/3600:.1f} hours")
    else:
        print(f"\n  💰 OPEN POSITION:")
        upnl = float(pos[10] or 0)
        print(f"     Unrealized PNL:   ${upnl:+.2f}")

    print()

# Final health check
print(f"\n{'='*120}")
print(f"🔍 SYSTEM HEALTH CHECK:")
print(f"{'='*120}\n")

issues = []

# Check bias
if long_count > short_count * 2:
    issues.append("🚨 STRONG LONG BIAS - PA analysis may not be working correctly!")
elif short_count > long_count * 2:
    issues.append("🚨 STRONG SHORT BIAS - PA analysis may not be working correctly!")

# Check PA analysis
if no_pa_count > 0:
    issues.append(f"🚨 CRITICAL: {no_pa_count} positions without PA analysis!")

# Check stop loss
no_sl = sum(1 for p in positions if not p[8])
if no_sl > 0:
    issues.append(f"🚨 CRITICAL: {no_sl} positions without stop loss!")

# Check for very short durations
very_short = [p for p in closed_positions if p[15] and (p[15] - p[14]).total_seconds() < 60]
if very_short:
    issues.append(f"⚠️ {len(very_short)} positions closed in <60 seconds (premature exits!)")

if issues:
    print("⚠️ ISSUES FOUND:")
    for issue in issues:
        print(f"   • {issue}")
else:
    print("✅ No major issues detected - system appears healthy!")

print(f"\n{'='*120}\n")

conn.close()
