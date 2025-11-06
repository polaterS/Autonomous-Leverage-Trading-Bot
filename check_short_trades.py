"""
Check if bot has EVER executed any SHORT trades.
"""
import asyncio
from src.database import get_db_client
from datetime import datetime

async def check_shorts():
    print("=" * 60)
    print("SHORT TRADES ANALYSIS")
    print("=" * 60)

    db = await get_db_client()

    # Check active positions
    print("\nACTIVE POSITIONS:")
    active = await db.get_active_positions()
    long_count = sum(1 for p in active if p['side'] == 'long')
    short_count = sum(1 for p in active if p['side'] == 'short')

    print(f"  LONG positions: {long_count}")
    print(f"  SHORT positions: {short_count}")

    for pos in active:
        print(f"  - {pos['symbol']} {pos['side'].upper()} @ ${float(pos['entry_price']):.4f}")

    # Check ALL trade history
    print("\nTRADE HISTORY (ALL CLOSED TRADES):")
    query = """
        SELECT
            symbol,
            side,
            entry_price,
            exit_price,
            realized_pnl_usd,
            entry_time,
            exit_time
        FROM trade_history
        ORDER BY exit_time DESC
    """

    async with db.pool.acquire() as conn:
        trades = await conn.fetch(query)

    if not trades:
        print("  No closed trades found!")
    else:
        long_trades = [t for t in trades if t['side'] == 'long']
        short_trades = [t for t in trades if t['side'] == 'short']

        print(f"\n  Total trades: {len(trades)}")
        print(f"  LONG trades: {len(long_trades)}")
        print(f"  SHORT trades: {len(short_trades)}")

        if short_trades:
            print("\n  [OK] SHORT TRADES FOUND:")
            for trade in short_trades[:10]:  # Show first 10
                pnl = float(trade['realized_pnl_usd'])
                symbol = trade['symbol']
                entry = float(trade['entry_price'])
                exit_price = float(trade['exit_price'])
                print(f"    {symbol} SHORT: Entry ${entry:.4f} -> Exit ${exit_price:.4f} | P&L: ${pnl:+.2f}")
        else:
            print("\n  [X] NO SHORT TRADES FOUND!")
            print("\n  Last 10 LONG trades:")
            for trade in long_trades[:10]:
                pnl = float(trade['realized_pnl_usd'])
                symbol = trade['symbol']
                entry = float(trade['entry_price'])
                exit_price = float(trade['exit_price'])
                print(f"    {symbol} LONG: Entry ${entry:.4f} -> Exit ${exit_price:.4f} | P&L: ${pnl:+.2f}")

    # Check if AI is EVER suggesting SHORT
    print("\nAI SIGNAL ANALYSIS:")

    # Get recent market scan results from logs or check ML patterns
    ml_query = """
        SELECT
            action,
            side,
            COUNT(*) as count
        FROM ml_patterns
        GROUP BY action, side
        ORDER BY count DESC
    """

    try:
        async with db.pool.acquire() as conn:
            ml_stats = await conn.fetch(ml_query)

        if ml_stats:
            print("  ML Pattern predictions by action/side:")
            for stat in ml_stats:
                action = stat['action']
                side = stat['side']
                count = stat['count']
                print(f"    {action} {side if side else 'N/A'}: {count} predictions")
        else:
            print("  No ML patterns found")
    except Exception as e:
        print(f"  ML patterns table check failed: {e}")

    print("\n" + "=" * 60)
    await db.close()

if __name__ == "__main__":
    asyncio.run(check_shorts())
