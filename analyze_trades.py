"""
Analyze Recent Trades from Binance
Fetches realized PNL and trade history to diagnose issues
"""

import asyncio
from datetime import datetime, timedelta
from src.exchange_client import get_exchange_client
from src.database import get_db_client
from src.config import get_settings

async def main():
    print("=" * 100)
    print("TRADE HISTORY ANALYZER - Diagnosing $16 Loss Issue")
    print("=" * 100)

    try:
        settings = get_settings()
        exchange = await get_exchange_client()
        db = await get_db_client()

        # 1. Check current balance
        print("\n💰 BALANCE CHECK:")
        print("-" * 100)
        balance = await exchange.fetch_balance()
        print(f"Current Balance: ${float(balance):.2f}")
        print(f"Initial Capital: ${float(settings.initial_capital):.2f}")
        total_pnl = float(balance) - float(settings.initial_capital)
        print(f"Total P&L: ${total_pnl:+.2f}")

        # 2. Fetch realized PNL from Binance (last 100 trades)
        print("\n📊 FETCHING REALIZED PNL FROM BINANCE:")
        print("-" * 100)

        try:
            income_records = await exchange.exchange.fapiPrivateGetIncome({
                'incomeType': 'REALIZED_PNL',
                'limit': 100
            })

            if income_records:
                print(f"Found {len(income_records)} realized PNL records\n")

                # Analyze trades
                total_realized = 0
                wins = 0
                losses = 0
                rapid_trades = []

                print(f"{'Date':<20} {'Symbol':<15} {'PNL':<12} {'Cumulative':<12}")
                print("-" * 100)

                for i, record in enumerate(income_records[:50]):  # Show last 50
                    symbol = record.get('symbol', 'UNKNOWN')
                    pnl = float(record.get('income', 0))
                    timestamp = int(record.get('time', 0)) / 1000
                    date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

                    total_realized += pnl

                    if pnl > 0:
                        wins += 1
                    elif pnl < 0:
                        losses += 1

                    # Color code PNL
                    pnl_str = f"${pnl:+.4f}"
                    cum_str = f"${total_realized:+.4f}"

                    print(f"{date:<20} {symbol:<15} {pnl_str:<12} {cum_str:<12}")

                    # Check for rapid trades (< 5 minutes apart)
                    if i < len(income_records) - 1:
                        next_time = int(income_records[i+1].get('time', 0)) / 1000
                        time_diff = timestamp - next_time

                        if abs(time_diff) < 300:  # Less than 5 minutes
                            rapid_trades.append({
                                'symbol': symbol,
                                'time_diff': abs(time_diff),
                                'pnl': pnl,
                                'timestamp': timestamp
                            })

                # Statistics
                print("\n" + "=" * 100)
                print("📈 STATISTICS:")
                print("-" * 100)
                print(f"Total Realized PNL: ${total_realized:+.4f}")
                print(f"Winning Trades: {wins}")
                print(f"Losing Trades: {losses}")
                total_trades = wins + losses
                if total_trades > 0:
                    win_rate = wins / total_trades * 100
                    print(f"Win Rate: {win_rate:.1f}%")

                # Analyze rapid trades
                if rapid_trades:
                    print(f"\n⚠️ RAPID TRADES DETECTED: {len(rapid_trades)} trades")
                    print("-" * 100)
                    print("These positions closed within 5 minutes - likely stop-loss hit too fast!")
                    print(f"\n{'Symbol':<15} {'Time Diff':<15} {'PNL':<12}")
                    print("-" * 100)

                    rapid_loss = 0
                    for rt in rapid_trades[:20]:
                        print(f"{rt['symbol']:<15} {rt['time_diff']:.0f} seconds{'':<3} ${rt['pnl']:+.4f}")
                        rapid_loss += rt['pnl']

                    print("-" * 100)
                    print(f"Total loss from rapid trades: ${rapid_loss:+.4f}")

                    if rapid_loss < -5:
                        print("\n🚨 PROBLEM IDENTIFIED:")
                        print("   Positions are closing too fast (< 5 min) = Stop-loss hit immediately!")
                        print("   Possible causes:")
                        print("   1. Entry price is poor (entering during volatility)")
                        print("   2. Stop-loss too tight for 20x leverage")
                        print("   3. Market moving against position right after entry")
                        print("   4. Slippage causing entry at worse price than expected")

        except Exception as e:
            print(f"❌ Error fetching Binance income: {e}")

        # 3. Check database trade history
        print("\n\n📚 DATABASE TRADE HISTORY:")
        print("-" * 100)

        trades = await db.pool.fetch("""
            SELECT
                symbol, side, leverage, entry_price, exit_price,
                realized_pnl, pnl_percent, entry_time, exit_time,
                exit_reason, position_value_usd
            FROM trade_history
            ORDER BY exit_time DESC
            LIMIT 30
        """)

        if trades:
            print(f"Found {len(trades)} trades in database\n")
            print(f"{'Date':<20} {'Symbol':<12} {'Side':<6} {'Entry':<10} {'Exit':<10} {'PNL':<12} {'%':<8} {'Duration':<10} {'Reason':<20}")
            print("-" * 100)

            db_total_pnl = 0
            for trade in trades:
                duration = trade['exit_time'] - trade['entry_time']
                duration_str = f"{duration.total_seconds()/60:.0f}m"

                pnl = float(trade['realized_pnl'])
                db_total_pnl += pnl

                print(f"{trade['exit_time'].strftime('%Y-%m-%d %H:%M'):<20} "
                      f"{trade['symbol']:<12} "
                      f"{trade['side']:<6} "
                      f"${float(trade['entry_price']):<9.2f} "
                      f"${float(trade['exit_price']):<9.2f} "
                      f"${pnl:+11.4f} "
                      f"{float(trade['pnl_percent']):+7.2f}% "
                      f"{duration_str:<10} "
                      f"{(trade['exit_reason'] or 'Unknown')[:20]:<20}")

            print("-" * 100)
            print(f"Total PNL from database: ${db_total_pnl:+.4f}")
        else:
            print("No trades found in database")

        # 4. Check current active positions
        print("\n\n🔴 ACTIVE POSITIONS:")
        print("-" * 100)

        active_positions = await db.get_active_positions()
        if active_positions:
            print(f"{'Symbol':<12} {'Side':<6} {'Leverage':<8} {'Entry':<10} {'Current':<10} {'Unrealized PNL':<15}")
            print("-" * 100)

            total_unrealized = 0
            for pos in active_positions:
                entry = float(pos['entry_price'])
                current = float(pos['current_price'])
                pos_value = float(pos['position_value_usd'])

                # Calculate unrealized PNL
                pnl_pct = (current - entry) / entry * 100
                if pos['side'] == 'SHORT':
                    pnl_pct = -pnl_pct

                unrealized_pnl = pos_value * (pnl_pct / 100) * pos['leverage']
                total_unrealized += unrealized_pnl

                print(f"{pos['symbol']:<12} "
                      f"{pos['side']:<6} "
                      f"{pos['leverage']}x{'':<6} "
                      f"${entry:<9.4f} "
                      f"${current:<9.4f} "
                      f"${unrealized_pnl:+14.4f}")

            print("-" * 100)
            print(f"Total Unrealized PNL: ${total_unrealized:+.4f}")
        else:
            print("No active positions")

        print("\n" + "=" * 100)
        print("ANALYSIS COMPLETE")
        print("=" * 100)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
