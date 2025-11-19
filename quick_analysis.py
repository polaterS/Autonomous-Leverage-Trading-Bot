"""
Quick database analysis - directly connect to Railway
"""
import asyncio
import asyncpg
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

async def main():
    # Connect directly to Railway database
    conn = await asyncpg.connect(os.getenv('DATABASE_URL'))

    try:
        # Get all positions, ordered by newest first
        positions = await conn.fetch("""
            SELECT
                id, symbol, side, leverage, entry_price, exit_price,
                quantity, position_value_usd, stop_loss_price, take_profit_price,
                unrealized_pnl, realized_pnl, close_reason, status,
                created_at, closed_at, confidence_score, confluence_factors
            FROM positions
            ORDER BY created_at DESC
            LIMIT 50
        """)

        if not positions:
            print("❌ No positions found")
            return

        print(f"\n{'='*100}")
        print(f"📊 ANALYZING LAST {len(positions)} POSITIONS")
        print(f"{'='*100}\n")

        # Stats
        long_count = sum(1 for p in positions if p['side'] == 'LONG')
        short_count = sum(1 for p in positions if p['side'] == 'SHORT')
        open_pos = sum(1 for p in positions if p['status'] == 'open')
        closed_pos = sum(1 for p in positions if p['status'] == 'closed')

        print(f"📈 SUMMARY:")
        print(f"  Total: {len(positions)}")
        print(f"  LONG: {long_count} ({long_count/len(positions)*100:.1f}%)")
        print(f"  SHORT: {short_count} ({short_count/len(positions)*100:.1f}%)")
        print(f"  Open: {open_pos} | Closed: {closed_pos}")

        # Check for bias
        print(f"\n⚠️ BIAS CHECK:")
        if long_count > short_count * 2:
            print(f"  🚨 LONG BIAS! {long_count} LONG vs {short_count} SHORT")
        elif short_count > long_count * 2:
            print(f"  🚨 SHORT BIAS! {short_count} SHORT vs {long_count} LONG")
        else:
            print(f"  ✅ Balanced")

        # Close reasons
        closed = [p for p in positions if p['status'] == 'closed']
        if closed:
            print(f"\n📋 CLOSE REASONS:")
            reasons = {}
            for p in closed:
                r = p['close_reason'] or 'Unknown'
                reasons[r] = reasons.get(r, 0) + 1
            for r, c in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {r}: {c}")

        # PA Analysis check
        no_pa = sum(1 for p in positions if not p['confidence_score'])
        if no_pa:
            print(f"\n🚨 {no_pa} POSITIONS WITHOUT PA ANALYSIS!")

        # Show last 10 positions
        print(f"\n\n📊 LAST 10 POSITIONS:")
        print(f"{'='*100}")
        for i, p in enumerate(positions[:10], 1):
            print(f"\n{i}. {p['symbol']} {p['side']} | {p['status']} | {p['created_at']}")
            print(f"   Entry: ${float(p['entry_price']):.4f} | Pos Value: ${float(p['position_value_usd']):.2f} | Leverage: {p['leverage']}x")

            if p['confidence_score']:
                print(f"   ✅ PA: {float(p['confidence_score'])*100:.1f}% confidence")
            else:
                print(f"   ❌ NO PA ANALYSIS!")

            if p['stop_loss_price']:
                print(f"   Stop Loss: ${float(p['stop_loss_price']):.4f}")
            else:
                print(f"   ❌ NO STOP LOSS!")

            if p['status'] == 'closed':
                pnl = float(p['realized_pnl'] or 0)
                print(f"   PNL: ${pnl:+.2f} | Reason: {p['close_reason']}")
                if p['closed_at']:
                    dur = (p['closed_at'] - p['created_at']).total_seconds()
                    if dur < 60:
                        print(f"   ⚠️ Duration: {dur:.0f}s (TOO FAST!)")
                    else:
                        print(f"   Duration: {dur/60:.1f} minutes")

        print(f"\n{'='*100}")

    finally:
        await conn.close()

if __name__ == '__main__':
    asyncio.run(main())
