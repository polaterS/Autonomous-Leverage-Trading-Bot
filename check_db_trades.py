"""
Check Trade History from Database
Analyzes completed trades to find issues
"""

import asyncio
import asyncpg
from datetime import datetime
from decimal import Decimal
from src.config import get_settings
from tabulate import tabulate

async def analyze_db_trades():
    """Fetch and analyze trades from PostgreSQL database."""

    print("=" * 80)
    print("DATABASE TRADE HISTORY ANALYZER")
    print("=" * 80)

    settings = get_settings()

    print(f"\n📊 Connecting to database...")

    try:
        # Connect to database
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        conn = await asyncpg.connect(settings.database_url, ssl=ssl_context)
        print("✅ Connected!")

        # Get trading config
        config = await conn.fetchrow("SELECT * FROM trading_config WHERE id = 1")
        print(f"\n💰 Current Capital: ${config['current_capital']}")
        print(f"📉 Initial Capital: ${config['initial_capital']}")
        total_pnl = float(config['current_capital']) - float(config['initial_capital'])
        print(f"📊 Total P&L: ${total_pnl:.2f}")

        # Get trade history (last 50 trades)
        trades = await conn.fetch("""
            SELECT
                symbol,
                side,
                leverage,
                entry_price,
                exit_price,
                quantity,
                position_value_usd,
                realized_pnl,
                pnl_percent,
                entry_time,
                exit_time,
                exit_reason,
                ai_reasoning
            FROM trade_history
            ORDER BY exit_time DESC
            LIMIT 50
        """)

        if not trades:
            print("\n⚠️ No completed trades found in database!")
            await conn.close()
            return

        print(f"\n📋 Found {len(trades)} completed trades:")
        print("=" * 80)

        # Prepare data for table
        trade_data = []
        total_profit = 0
        total_loss = 0
        wins = 0
        losses = 0

        for trade in trades:
            pnl = float(trade['realized_pnl'])
            total_profit += pnl if pnl > 0 else 0
            total_loss += pnl if pnl < 0 else 0
            wins += 1 if pnl > 0 else 0
            losses += 1 if pnl < 0 else 0

            # Calculate trade duration
            duration = trade['exit_time'] - trade['entry_time']
            duration_minutes = duration.total_seconds() / 60

            trade_data.append([
                trade['exit_time'].strftime('%m-%d %H:%M'),
                trade['symbol'],
                trade['side'],
                f"{trade['leverage']}x",
                f"${float(trade['entry_price']):.2f}",
                f"${float(trade['exit_price']):.2f}",
                f"${pnl:+.2f}",
                f"{float(trade['pnl_percent']):+.1f}%",
                f"{duration_minutes:.0f}m",
                trade['exit_reason'][:15] if trade['exit_reason'] else 'Unknown'
            ])

        # Print table
        print(tabulate(
            trade_data,
            headers=['Date', 'Symbol', 'Side', 'Lev', 'Entry', 'Exit', 'PNL', '%', 'Duration', 'Exit Reason'],
            tablefmt='grid'
        ))

        # Statistics
        print("\n" + "=" * 80)
        print("📊 STATISTICS:")
        print("=" * 80)
        print(f"  Total Trades: {len(trades)}")
        print(f"  ✅ Wins: {wins} ({wins/len(trades)*100:.1f}%)")
        print(f"  ❌ Losses: {losses} ({losses/len(trades)*100:.1f}%)")
        print(f"  💰 Total Profit: ${total_profit:.2f}")
        print(f"  💸 Total Loss: ${total_loss:.2f}")
        print(f"  📊 Net P&L: ${total_profit + total_loss:.2f}")

        avg_win = total_profit / wins if wins > 0 else 0
        avg_loss = abs(total_loss / losses) if losses > 0 else 0
        print(f"  📈 Avg Win: ${avg_win:.2f}")
        print(f"  📉 Avg Loss: ${avg_loss:.2f}")

        if avg_loss > 0:
            print(f"  ⚖️ Risk/Reward Ratio: {avg_win/avg_loss:.2f}")

        # Analyze rapid trades (< 5 minutes)
        rapid_trades = [t for t in trade_data if 'm' in t[8] and float(t[8].replace('m', '')) < 5]
        if rapid_trades:
            print(f"\n⚠️ RAPID TRADES (<5 min): {len(rapid_trades)}")
            print("These closed too fast - likely stop-loss hit immediately!")

            rapid_loss = sum(float(t[6].replace('$', '').replace('+', '')) for t in rapid_trades)
            print(f"💸 Loss from rapid trades: ${rapid_loss:.2f}")

        # Analyze by exit reason
        print("\n📋 EXIT REASONS:")
        exit_reasons = {}
        for trade in trades:
            reason = trade['exit_reason'] or 'Unknown'
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'pnl': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['pnl'] += float(trade['realized_pnl'])

        for reason, data in sorted(exit_reasons.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  {reason}: {data['count']} trades, ${data['pnl']:+.2f} total")

        # Get current active positions
        active = await conn.fetch("SELECT * FROM active_position ORDER BY entry_time DESC")
        if active:
            print(f"\n🔴 ACTIVE POSITIONS: {len(active)}")
            active_data = []
            for pos in active:
                unrealized = (float(pos['current_price']) - float(pos['entry_price'])) / float(pos['entry_price']) * 100
                if pos['side'] == 'SHORT':
                    unrealized = -unrealized
                unrealized_pnl = float(pos['position_value_usd']) * (unrealized / 100)

                active_data.append([
                    pos['symbol'],
                    pos['side'],
                    f"{pos['leverage']}x",
                    f"${float(pos['entry_price']):.2f}",
                    f"${float(pos['current_price']):.2f}",
                    f"${unrealized_pnl:+.2f}",
                    f"{unrealized:+.1f}%"
                ])

            print(tabulate(
                active_data,
                headers=['Symbol', 'Side', 'Lev', 'Entry', 'Current', 'PNL', '%'],
                tablefmt='grid'
            ))

        await conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_db_trades())
