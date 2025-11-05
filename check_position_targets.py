"""
Check position targets and stop-loss levels.
"""
import asyncio
from src.database import get_db_client
from datetime import datetime
from decimal import Decimal

async def check_targets():
    db = await get_db_client()

    positions = await db.get_active_positions()

    print("=" * 80)
    print("POSITION TARGETS AND EXIT CONDITIONS")
    print("=" * 80)

    for pos in positions:
        entry_time = pos['entry_time']
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)

        duration = datetime.now() - entry_time
        hours = duration.total_seconds() / 3600

        entry_price = Decimal(str(pos['entry_price']))
        current_price = Decimal(str(pos.get('current_price', pos['entry_price'])))
        stop_loss_price = Decimal(str(pos['stop_loss_price']))
        min_profit_price = Decimal(str(pos['min_profit_price']))
        min_profit_target = Decimal(str(pos['min_profit_target_usd']))
        unrealized_pnl = Decimal(str(pos.get('unrealized_pnl_usd', 0)))

        print(f"\n{'=' * 80}")
        print(f"#{pos['id']} {pos['symbol']} {pos['side']} {pos['leverage']}x - Open {hours:.1f} hours")
        print(f"{'=' * 80}")

        print(f"\nPRICES:")
        print(f"  Entry:         ${entry_price:.4f}")
        print(f"  Current:       ${current_price:.4f}")
        print(f"  Stop-Loss:     ${stop_loss_price:.4f}")
        print(f"  Min Profit:    ${min_profit_price:.4f}")

        print(f"\nP&L:")
        print(f"  Current P&L:   ${unrealized_pnl:+.2f}")
        print(f"  Target P&L:    ${min_profit_target:.2f}")
        print(f"  Progress:      {(unrealized_pnl / min_profit_target * 100):+.1f}%")

        # Check exit conditions
        print(f"\nEXIT CONDITIONS:")

        if pos['side'] == 'LONG':
            # Stop-loss check
            if current_price <= stop_loss_price:
                print(f"  [X] STOP-LOSS HIT! (Price ${current_price:.4f} <= SL ${stop_loss_price:.4f})")
            else:
                sl_distance_pct = ((current_price - stop_loss_price) / entry_price) * 100
                print(f"  [OK] Stop-loss safe: {sl_distance_pct:.2f}% away")

            # Profit target check
            if current_price >= min_profit_price and unrealized_pnl >= min_profit_target:
                print(f"  [OK] PROFIT TARGET HIT! (Price ${current_price:.4f} >= Target ${min_profit_price:.4f})")
            else:
                if current_price >= min_profit_price:
                    print(f"  [!] Price reached target, but P&L ${unrealized_pnl:.2f} < ${min_profit_target:.2f}")
                else:
                    pt_distance_pct = ((min_profit_price - current_price) / entry_price) * 100
                    print(f"  [WAIT] Profit target: {pt_distance_pct:.2f}% away")

        print(f"\nRECOMMENDATION:")

        # Time-based recommendation
        if hours > 24:
            if unrealized_pnl > 0:
                print(f"  [WIN] Position is winning after {hours:.1f}h - Let it run or take profit")
            elif unrealized_pnl < -3:
                print(f"  [LOSS] Position losing ${abs(unrealized_pnl):.2f} after {hours:.1f}h - Consider manual close")
            else:
                print(f"  [FLAT] Position flat after {hours:.1f}h - Monitor closely")

    print("\n" + "=" * 80)
    await db.close()

if __name__ == "__main__":
    asyncio.run(check_targets())
