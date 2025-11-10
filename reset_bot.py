"""
Reset Bot Script
Resets capital, clears drawdown protection, and prepares bot for fresh start.

Usage:
    python reset_bot.py
"""

import asyncio
import asyncpg
import os
from decimal import Decimal

async def reset_bot():
    """Reset bot to initial state."""
    print("=" * 60)
    print("🔄 RESETTING TRADING BOT")
    print("=" * 60)

    # Connect to database
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ ERROR: DATABASE_URL not found in environment")
        return

    conn = await asyncpg.connect(database_url)

    try:
        # 1. Reset capital to $1000
        print("\n1️⃣ Resetting capital to $1000...")
        await conn.execute("""
            UPDATE trading_config
            SET current_capital = 1000.00,
                starting_capital = 1000.00
        """)
        print("   ✅ Capital reset to $1000")

        # 2. Enable trading (clear drawdown protection)
        print("\n2️⃣ Clearing drawdown protection...")
        await conn.execute("""
            UPDATE trading_config
            SET is_trading_enabled = true
        """)
        print("   ✅ Trading re-enabled")

        # 3. Clear circuit breakers (if table exists)
        print("\n3️⃣ Clearing circuit breakers...")
        try:
            result = await conn.execute("DELETE FROM circuit_breaker")
            print(f"   ✅ Cleared circuit breakers")
        except:
            print(f"   ℹ️  Circuit breaker table not found (skipping)")

        # 4. Show current status
        print("\n" + "=" * 60)
        print("📊 CURRENT STATUS")
        print("=" * 60)

        config = await conn.fetchrow("SELECT * FROM trading_config")
        print(f"Capital: ${float(config['current_capital']):.2f}")
        print(f"Starting Capital: ${float(config['starting_capital']):.2f}")
        print(f"Trading Enabled: {config['is_trading_enabled']}")
        print(f"Max Drawdown: {float(config.get('max_drawdown_percent', 20)):.0f}%")

        # 5. Show active positions
        active_positions = await conn.fetch("SELECT * FROM active_position")
        print(f"\nActive Positions: {len(active_positions)}")

        if active_positions:
            print("\n⚠️  WARNING: You still have active positions!")
            print("   Close them manually with /closeall before starting")
            for pos in active_positions:
                print(f"   - {pos['symbol']} {pos['side']} ${float(pos['position_value_usd']):.2f}")

        # 6. Show recent trades
        recent_trades = await conn.fetch("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(realized_pnl_usd) as total_pnl
            FROM trade_history
        """)

        trade_stats = recent_trades[0]
        total = trade_stats['total']
        wins = trade_stats['wins'] or 0
        win_rate = (wins / total * 100) if total > 0 else 0
        total_pnl = float(trade_stats['total_pnl'] or 0)

        print("\n" + "=" * 60)
        print("📈 HISTORICAL PERFORMANCE")
        print("=" * 60)
        print(f"Total Trades: {total}")
        print(f"Win Rate: {win_rate:.1f}% ({wins}/{total})")
        print(f"Total P&L: ${total_pnl:.2f}")

        print("\n" + "=" * 60)
        print("✅ BOT RESET COMPLETE!")
        print("=" * 60)
        print("\n🚀 Next Steps:")
        print("   1. Close any remaining positions with /closeall")
        print("   2. Deploy to Railway: git push")
        print("   3. Restart bot on Railway")
        print("   4. Start trading with /startbot")
        print("\n💡 New Conservative Mode Active:")
        print("   - Max 5 positions")
        print("   - $50 per position (5% of capital)")
        print("   - 70%+ confidence minimum")
        print("   - 3-10x leverage (max 10x)")
        print("   - 12-20% stop-loss (wider stops)")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(reset_bot())
