"""
Check live Binance positions and leverage settings
"""
import os
import asyncio
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *

load_dotenv()

# Initialize Binance client
client = Client(
    api_key=os.getenv('BINANCE_API_KEY'),
    api_secret=os.getenv('BINANCE_SECRET_KEY')
)

print("=" * 90)
print("BINANCE FUTURES ACCOUNT CHECK")
print("=" * 90)

# Get account info
try:
    account = client.futures_account()
    balance = float(account['totalWalletBalance'])
    unrealized_pnl = float(account['totalUnrealizedProfit'])

    print(f"\n💰 ACCOUNT BALANCE:")
    print(f"   Total Wallet Balance: ${balance:.2f}")
    print(f"   Unrealized P&L: ${unrealized_pnl:.2f}")

    # Get open positions
    positions = client.futures_position_information()
    open_positions = [p for p in positions if float(p['positionAmt']) != 0]

    print(f"\n📊 OPEN POSITIONS: {len(open_positions)}")
    print("=" * 90)

    if open_positions:
        for pos in open_positions:
            symbol = pos['symbol']
            side = "LONG" if float(pos['positionAmt']) > 0 else "SHORT"
            entry_price = float(pos['entryPrice'])
            mark_price = float(pos['markPrice'])
            quantity = abs(float(pos['positionAmt']))
            leverage = int(pos['leverage'])
            unrealized = float(pos['unRealizedProfit'])

            print(f"\n🔴 {symbol}")
            print(f"   Side: {side}")
            print(f"   Entry Price: ${entry_price:.4f}")
            print(f"   Mark Price: ${mark_price:.4f}")
            print(f"   Quantity: {quantity}")
            print(f"   ⚠️  LEVERAGE: {leverage}x")
            print(f"   Unrealized P&L: ${unrealized:.2f}")
            print(f"   Position Value: ${quantity * mark_price:.2f}")
    else:
        print("✅ No open positions")

    # Get recent trades
    print("\n" + "=" * 90)
    print("RECENT TRADES (Last 5)")
    print("=" * 90)

    recent_trades = client.futures_account_trades(limit=5)
    for trade in recent_trades:
        print(f"{trade['symbol']:15} {trade['side']:4} | "
              f"Price: ${float(trade['price']):10.4f} | "
              f"Qty: {float(trade['qty']):8.4f} | "
              f"Time: {trade['time']}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
