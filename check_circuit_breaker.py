"""
Check circuit breaker status and recent trades.
"""
import asyncio
from src.database import get_db_client
from datetime import datetime

async def check_circuit():
    print("=" * 60)
    print("CIRCUIT BREAKER STATUS CHECK")
    print("=" * 60)

    db = await get_db_client()

    # Get circuit breaker status
    query_cb = """
        SELECT consecutive_losses, circuit_breaker_active, last_loss_time
        FROM trading_config
    """
    async with db.pool.acquire() as conn:
        config = await conn.fetchrow(query_cb)

    print(f"\nCircuit Breaker Status:")
    print(f"  Consecutive Losses: {config['consecutive_losses']}")
    print(f"  Circuit Breaker Active: {config['circuit_breaker_active']}")
    print(f"  Last Loss Time: {config['last_loss_time']}")

    # Get last 10 closed trades
    query_trades = """
        SELECT symbol, side, realized_pnl_usd, exit_time, close_reason
        FROM trade_history
        ORDER BY exit_time DESC
        LIMIT 10
    """
    async with db.pool.acquire() as conn:
        trades = await conn.fetch(query_trades)

    print(f"\nLast 10 Closed Trades:")
    for i, trade in enumerate(trades, 1):
        pnl = float(trade['realized_pnl_usd'])
        result = "WIN" if pnl > 0 else "LOSS"
        emoji = "[+]" if pnl > 0 else "[-]"
        print(f"  {i}. {emoji} {trade['symbol']} {trade['side'].upper()} ${pnl:+.2f} - {result}")
        if trade['close_reason']:
            print(f"      Reason: {trade['close_reason'][:50]}")

    # Count consecutive losses from end
    consecutive = 0
    for trade in trades:
        if float(trade['realized_pnl_usd']) <= 0:
            consecutive += 1
        else:
            break

    print(f"\nActual Consecutive Losses: {consecutive}")

    # Check active positions
    query_active = "SELECT COUNT(*) as count FROM active_position"
    async with db.pool.acquire() as conn:
        result = await conn.fetchrow(query_active)

    print(f"\nActive Positions: {result['count']}")

    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    if config['circuit_breaker_active']:
        print("  Circuit breaker is ACTIVE - Bot is paused")
        print("  Use /reset command in Telegram to restart trading")
        print("  Or run: python reset_circuit_breaker.py")
    else:
        print("  Circuit breaker is INACTIVE - Bot should be trading")

    print("=" * 60)

    await db.close()

if __name__ == "__main__":
    asyncio.run(check_circuit())
