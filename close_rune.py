"""
Emergency script to close RUNE position manually.
"""
import asyncio
from src.database import get_db_client
from src.trade_executor import get_trade_executor
from decimal import Decimal

async def close_rune():
    print("=" * 60)
    print("CLOSING RUNE POSITION")
    print("=" * 60)

    db = await get_db_client()
    executor = get_trade_executor()

    # Find RUNE position
    positions = await db.get_active_positions()

    rune_position = None
    for pos in positions:
        if 'RUNE' in pos['symbol']:
            rune_position = pos
            break

    if not rune_position:
        print("\n[X] RUNE position not found!")
        await db.close()
        return

    print(f"\nFound RUNE position:")
    print(f"  Symbol: {rune_position['symbol']}")
    print(f"  Side: {rune_position['side']}")
    print(f"  Leverage: {rune_position['leverage']}x")
    print(f"  Entry: ${float(rune_position['entry_price']):.4f}")
    print(f"  Current: ${float(rune_position.get('current_price', rune_position['entry_price'])):.4f}")
    print(f"  P&L: ${float(rune_position.get('unrealized_pnl_usd', 0)):+.2f}")

    # Close position
    print("\n>> Closing position...")

    current_price = Decimal(str(rune_position.get('current_price', rune_position['entry_price'])))

    success = await executor.close_position(
        position=rune_position,
        current_price=current_price,
        close_reason="Manual close - Aggressive strategy switch (max 8h hold time)"
    )

    if success:
        print("\n[OK] RUNE position closed successfully!")
        print(f"Realized P&L: ${float(rune_position.get('unrealized_pnl_usd', 0)):+.2f}")
    else:
        print("\n[X] Failed to close RUNE position")

    print("\n" + "=" * 60)
    await db.close()

if __name__ == "__main__":
    asyncio.run(close_rune())
