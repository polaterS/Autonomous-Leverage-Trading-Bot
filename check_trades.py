"""
Check Recent Binance Trades
Analyzes trade history to find issues causing losses
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from src.exchange_client import get_exchange_client
from src.config import get_settings
from tabulate import tabulate

async def analyze_recent_trades():
    """Fetch and analyze recent trades from Binance."""

    print("=" * 80)
    print("BINANCE TRADE HISTORY ANALYZER")
    print("=" * 80)

    settings = get_settings()
    exchange = await get_exchange_client()

    print(f"\n📊 Fetching trades from Binance Futures...")

    # Get all symbols bot has traded
    # We'll check the database for symbols, or fetch all recent trades
    try:
        # Fetch balance first
        balance = await exchange.fetch_balance()
        print(f"\n💰 Current Balance: ${float(balance):.2f}")
        print(f"📉 Initial Capital: ${float(settings.initial_capital):.2f}")
        print(f"📊 P&L: ${float(balance - settings.initial_capital):.2f}")

        # Fetch recent trades (last 7 days)
        print(f"\n🔍 Fetching trades from last 7 days...")

        # Get all positions to find which symbols were traded
        all_positions = await exchange.exchange.fetch_positions()

        # Extract unique symbols
        symbols_traded = set()
        for pos in all_positions:
            symbol = pos.get('symbol')
            if symbol:
                symbols_traded.add(symbol)

        print(f"📋 Found {len(symbols_traded)} symbols with position history")

        # Fetch trades for each symbol
        all_trades = []
        since = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

        for symbol in list(symbols_traded)[:20]:  # Limit to 20 symbols to avoid rate limit
            try:
                trades = await exchange.exchange.fetch_my_trades(symbol, since=since)
                if trades:
                    all_trades.extend(trades)
                    print(f"  ✅ {symbol}: {len(trades)} trades")
            except Exception as e:
                print(f"  ⚠️ {symbol}: {e}")

        if not all_trades:
            print("\n⚠️ No trades found in last 7 days!")
            print("Trying to fetch closed PNL history instead...")

            # Alternative: Fetch income history (realized PNL)
            income = await exchange.exchange.fapiPrivateGetIncome({
                'incomeType': 'REALIZED_PNL',
                'limit': 100
            })

            if income:
                print(f"\n📊 Found {len(income)} realized PNL records:")
                print("=" * 80)

                pnl_data = []
                total_pnl = 0

                for record in income[:30]:  # Show last 30
                    symbol = record.get('symbol', 'UNKNOWN')
                    pnl = float(record.get('income', 0))
                    timestamp = int(record.get('time', 0)) / 1000
                    date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')

                    total_pnl += pnl

                    pnl_data.append([
                        date,
                        symbol,
                        f"${pnl:.4f}",
                        f"${total_pnl:.4f}"
                    ])

                print(tabulate(
                    pnl_data,
                    headers=['Date', 'Symbol', 'PNL', 'Cumulative'],
                    tablefmt='grid'
                ))

                print(f"\n💰 Total Realized PNL: ${total_pnl:.4f}")

                # Analyze patterns
                print("\n🔍 ANALYSIS:")
                wins = sum(1 for r in income if float(r.get('income', 0)) > 0)
                losses = sum(1 for r in income if float(r.get('income', 0)) < 0)
                win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

                print(f"  ✅ Winning trades: {wins}")
                print(f"  ❌ Losing trades: {losses}")
                print(f"  📊 Win rate: {win_rate:.1f}%")

                # Check for rapid open/close pattern
                rapid_trades = []
                for i in range(len(income) - 1):
                    time1 = int(income[i].get('time', 0))
                    time2 = int(income[i+1].get('time', 0))
                    time_diff_seconds = abs(time1 - time2) / 1000

                    if time_diff_seconds < 300:  # Less than 5 minutes
                        rapid_trades.append({
                            'symbol': income[i].get('symbol'),
                            'time_diff': time_diff_seconds,
                            'pnl1': float(income[i].get('income', 0)),
                            'pnl2': float(income[i+1].get('income', 0))
                        })

                if rapid_trades:
                    print(f"\n⚠️ RAPID TRADES DETECTED ({len(rapid_trades)} instances):")
                    print("These positions closed within 5 minutes - likely hitting stop-loss too fast!")

                    for rt in rapid_trades[:10]:
                        print(f"  🚨 {rt['symbol']}: Closed in {rt['time_diff']:.0f}s, PNL: ${rt['pnl1']:.4f} + ${rt['pnl2']:.4f}")

                    # Calculate loss from rapid trades
                    rapid_loss = sum(rt['pnl1'] + rt['pnl2'] for rt in rapid_trades)
                    print(f"\n💸 Total loss from rapid trades: ${rapid_loss:.4f}")
                    print(f"   This is likely from:")
                    print(f"   1. Stop-loss hit too fast (market volatility)")
                    print(f"   2. Position opened at bad entry point")
                    print(f"   3. Stop-loss set too tight for 20x leverage")

        else:
            print(f"\n📊 Found {len(all_trades)} total trades")
            print("=" * 80)

            # Analyze trades
            trade_data = []
            for trade in sorted(all_trades, key=lambda x: x['timestamp'], reverse=True)[:30]:
                date = datetime.fromtimestamp(trade['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')
                symbol = trade['symbol']
                side = trade['side']
                price = trade['price']
                amount = trade['amount']
                cost = trade['cost']
                fee = trade.get('fee', {}).get('cost', 0)

                trade_data.append([
                    date,
                    symbol,
                    side.upper(),
                    f"${price:.4f}",
                    f"{amount:.4f}",
                    f"${cost:.2f}",
                    f"${fee:.4f}"
                ])

            print(tabulate(
                trade_data,
                headers=['Date', 'Symbol', 'Side', 'Price', 'Amount', 'Cost', 'Fee'],
                tablefmt='grid'
            ))

    except Exception as e:
        print(f"❌ Error fetching trades: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_recent_trades())
