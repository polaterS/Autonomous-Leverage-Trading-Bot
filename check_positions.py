"""
Quick script to check active positions and their hold times.
"""
import asyncio
from src.database import get_db_client
from datetime import datetime

async def check_positions():
    db = await get_db_client()

    positions = await db.get_active_positions()

    print("=" * 60)
    print("ACTIVE POSITIONS CHECK")
    print("=" * 60)
    print(f"\nTotal active positions: {len(positions)}")

    if positions:
        print("\nPosition details:")
        for i, pos in enumerate(positions, 1):
            entry_time = pos['entry_time']
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)

            duration = datetime.now() - entry_time
            hours = duration.total_seconds() / 3600

            pnl = float(pos.get('unrealized_pnl_usd', 0))
            pnl_pct = (pnl / float(pos['position_value_usd'])) * 100 if float(pos['position_value_usd']) > 0 else 0

            print(f"\n#{i} {pos['symbol']} {pos['side']} {pos['leverage']}x")
            print(f"  Entry: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Duration: {hours:.1f} hours")
            print(f"  Entry Price: ${float(pos['entry_price']):.4f}")
            print(f"  Current Price: ${float(pos.get('current_price', pos['entry_price'])):.4f}")
            print(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            print(f"  Position ID: {pos['id']}")
    else:
        print("\n❌ No active positions found")

    print("\n" + "=" * 60)
    await db.close()

if __name__ == "__main__":
    asyncio.run(check_positions())
