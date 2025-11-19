"""
Analyze all positions opened after Nov 19, 2025, 11:42 AM
Check for:
1. LONG vs SHORT bias
2. PA analysis quality
3. Stop loss/take profit accuracy
4. Close reasons
"""

import asyncio
import sys
import os
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database import get_db_client

async def analyze_positions():
    """Analyze recent trading positions."""

    db = await get_db_client()

    try:
        # Query all positions after Nov 19, 2025, 11:42 AM UTC
        # Note: Nov 19, 2025 is in the future, so we'll check recent positions instead
        # Let's get positions from the last 24 hours

        query = """
            SELECT
                id,
                symbol,
                side,
                leverage,
                entry_price,
                exit_price,
                quantity,
                position_value_usd,
                stop_loss_price,
                take_profit_price,
                unrealized_pnl,
                realized_pnl,
                close_reason,
                status,
                created_at,
                closed_at,
                confidence_score,
                confluence_factors
            FROM positions
            ORDER BY created_at DESC
            LIMIT 50
        """

        positions = await db.pool.fetch(query)

        if not positions:
            print(f"❌ No positions found in database")
            return

        # Filter positions after Nov 19, 2025, 11:42 (or show all recent)
        cutoff_time = datetime(2025, 11, 19, 11, 42, 0, tzinfo=timezone.utc)

        # Check if we have positions after this time
        recent_positions = [p for p in positions if p['created_at'] >= cutoff_time]

        if not recent_positions:
            print(f"⚠️ No positions found after Nov 19, 2025, 11:42 AM UTC")
            print(f"Showing last 50 positions instead...\n")
            recent_positions = positions

        print(f"\n{'='*100}")
        print(f"📊 POSITION ANALYSIS: {len(recent_positions)} positions")
        print(f"{'='*100}\n")

        # Statistics
        long_count = sum(1 for p in recent_positions if p['side'] == 'LONG')
        short_count = sum(1 for p in recent_positions if p['side'] == 'SHORT')

        open_positions = [p for p in recent_positions if p['status'] == 'open']
        closed_positions = [p for p in recent_positions if p['status'] == 'closed']

        # Close reasons
        close_reasons = {}
        for p in closed_positions:
            reason = p['close_reason'] or 'Unknown'
            close_reasons[reason] = close_reasons.get(reason, 0) + 1

        # Calculate PNL statistics
        winning_trades = [p for p in closed_positions if p.get('realized_pnl', 0) and float(p['realized_pnl']) > 0]
        losing_trades = [p for p in closed_positions if p.get('realized_pnl', 0) and float(p['realized_pnl']) < 0]

        print(f"📈 SUMMARY STATISTICS:")
        print(f"{'─'*100}")
        print(f"Total Positions:        {len(recent_positions)}")
        print(f"  ├─ LONG:              {long_count} ({long_count/len(recent_positions)*100:.1f}%)")
        print(f"  └─ SHORT:             {short_count} ({short_count/len(recent_positions)*100:.1f}%)")
        print(f"\nPosition Status:")
        print(f"  ├─ Open:              {len(open_positions)}")
        print(f"  └─ Closed:            {len(closed_positions)}")

        if closed_positions:
            print(f"\nClosed Position Results:")
            print(f"  ├─ Winning:           {len(winning_trades)} ({len(winning_trades)/len(closed_positions)*100:.1f}% win rate)")
            print(f"  └─ Losing:            {len(losing_trades)} ({len(losing_trades)/len(closed_positions)*100:.1f}%)")

            total_pnl = sum(float(p.get('realized_pnl', 0) or 0) for p in closed_positions)
            print(f"\nTotal Realized PNL:   ${total_pnl:+.2f}")

        print(f"\n📋 CLOSE REASONS:")
        print(f"{'─'*100}")
        if close_reasons:
            for reason, count in sorted(close_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {reason:60} → {count:3} times")
        else:
            print(f"  (No closed positions yet)")

        # Check for bias
        print(f"\n⚠️ BIAS DETECTION:")
        print(f"{'─'*100}")
        if long_count > short_count * 2:
            print(f"  🚨 STRONG LONG BIAS DETECTED! {long_count} LONG vs {short_count} SHORT")
            print(f"  ⚠️ System may not be analyzing PA correctly - favoring longs!")
        elif short_count > long_count * 2:
            print(f"  🚨 STRONG SHORT BIAS DETECTED! {short_count} SHORT vs {long_count} LONG")
            print(f"  ⚠️ System may not be analyzing PA correctly - favoring shorts!")
        else:
            print(f"  ✅ Balanced: {long_count} LONG vs {short_count} SHORT (no significant bias)")

        # Check PA analysis quality
        positions_without_pa = [p for p in recent_positions if not p['confidence_score']]
        if positions_without_pa:
            print(f"\n🚨 PA ANALYSIS ISSUES:")
            print(f"  ⚠️ {len(positions_without_pa)} positions opened WITHOUT PA analysis!")
            print(f"  This is a critical bug - every position should have PA analysis!")

        # Detailed position list
        print(f"\n\n📊 DETAILED POSITION LIST (Last 20):")
        print(f"{'='*100}")

        for i, pos in enumerate(recent_positions[:20], 1):
            print(f"\n{'─'*100}")
            print(f"#{i} | {pos['symbol']:15} | {pos['side']:5} | {pos['status'].upper():6} | {pos['created_at']}")
            print(f"{'─'*100}")

            # Position details
            print(f"  Entry Price:        ${float(pos['entry_price']):.4f}")
            if pos['exit_price']:
                print(f"  Exit Price:         ${float(pos['exit_price']):.4f}")
            print(f"  Quantity:           {float(pos['quantity']):.4f}")
            print(f"  Position Value:     ${float(pos['position_value_usd']):.2f}")
            print(f"  Leverage:           {pos['leverage']}x")

            # PA Analysis
            if pos['confidence_score']:
                print(f"\n  🎯 PA ANALYSIS:")
                print(f"    Confidence:       {float(pos['confidence_score'])*100:.1f}%")
                if pos['confluence_factors']:
                    # Truncate if too long
                    factors = str(pos['confluence_factors'])
                    if len(factors) > 150:
                        factors = factors[:150] + "..."
                    print(f"    Confluence:       {factors}")
            else:
                print(f"\n  ⚠️⚠️⚠️ NO PA ANALYSIS DATA! POSITION OPENED WITHOUT ANALYSIS! ⚠️⚠️⚠️")

            # Risk management
            print(f"\n  📊 RISK MANAGEMENT:")
            if pos['stop_loss_price']:
                print(f"    Stop Loss:        ${float(pos['stop_loss_price']):.4f}")
            else:
                print(f"    Stop Loss:        ❌ NOT SET (BUG!)")

            if pos['take_profit_price']:
                print(f"    Take Profit:      ${float(pos['take_profit_price']):.4f}")
            else:
                print(f"    Take Profit:      ❌ NOT SET")

            # PNL
            if pos['status'] == 'closed':
                print(f"\n  💰 RESULTS:")
                pnl = float(pos['realized_pnl'] or 0)
                print(f"    Realized PNL:     ${pnl:+.2f}")
                print(f"    Close Reason:     {pos['close_reason'] or 'Unknown'}")
                if pos['closed_at']:
                    duration = pos['closed_at'] - pos['created_at']
                    total_seconds = duration.total_seconds()
                    if total_seconds < 60:
                        print(f"    Duration:         {total_seconds:.0f} seconds ⚠️ TOO SHORT!")
                    elif total_seconds < 3600:
                        print(f"    Duration:         {total_seconds/60:.1f} minutes")
                    else:
                        print(f"    Duration:         {total_seconds/3600:.1f} hours")
            else:
                print(f"\n  💰 CURRENT:")
                upnl = float(pos['unrealized_pnl'] or 0)
                print(f"    Unrealized PNL:   ${upnl:+.2f}")

        print(f"\n{'='*100}")
        print(f"✅ Analysis Complete!")
        print(f"{'='*100}\n")

        # Final warnings
        print(f"\n🔍 SYSTEM HEALTH CHECK:")
        print(f"{'─'*100}")

        issues = []

        # Check for bias
        if long_count > short_count * 2 or short_count > long_count * 2:
            issues.append("⚠️ Directional bias detected - PA analysis may not be working correctly")

        # Check for missing PA analysis
        if positions_without_pa:
            issues.append(f"🚨 CRITICAL: {len(positions_without_pa)} positions without PA analysis")

        # Check for very short durations
        very_short = [p for p in closed_positions if p['closed_at'] and (p['closed_at'] - p['created_at']).total_seconds() < 60]
        if very_short:
            issues.append(f"⚠️ {len(very_short)} positions closed in <60 seconds (too fast!)")

        # Check for missing stop loss
        no_sl = [p for p in recent_positions if not p['stop_loss_price']]
        if no_sl:
            issues.append(f"🚨 CRITICAL: {len(no_sl)} positions without stop loss")

        if issues:
            print("  Issues found:")
            for issue in issues:
                print(f"    • {issue}")
        else:
            print("  ✅ No major issues detected!")

    finally:
        await db.close()

if __name__ == '__main__':
    asyncio.run(analyze_positions())
