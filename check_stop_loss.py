"""Quick script to check stop-loss orders on Binance"""
import asyncio
from src.exchange_client import get_exchange_client

async def check_stop_losses():
    print("🔍 Checking Binance for stop-loss orders...\n")

    exchange = await get_exchange_client()

    # Symbols to check
    symbols = ['ADA/USDT:USDT', 'DOGE/USDT:USDT', 'ATOM/USDT:USDT', 'AVAX/USDT:USDT', 'ICP/USDT:USDT']

    for symbol in symbols:
        try:
            print(f"\n{'='*60}")
            print(f"📊 {symbol}")
            print(f"{'='*60}")

            # Get open orders for this symbol
            orders = await exchange.fetch_open_orders(symbol)

            if not orders:
                print("❌ No open orders (NO STOP-LOSS!)")
                continue

            # Check for stop-loss orders
            stop_loss_orders = [
                o for o in orders
                if o.get('type') in ['stop_market', 'stop', 'stop_loss', 'STOP_MARKET', 'STOP']
            ]

            if stop_loss_orders:
                print(f"✅ Found {len(stop_loss_orders)} stop-loss order(s):")
                for order in stop_loss_orders:
                    print(f"   - ID: {order['id']}")
                    print(f"   - Type: {order['type']}")
                    print(f"   - Side: {order['side']}")
                    print(f"   - Stop Price: ${float(order.get('stopPrice', order.get('triggerPrice', 0))):.4f}")
                    print(f"   - Amount: {order['amount']}")
            else:
                print(f"⚠️ Open orders exist but NO STOP-LOSS orders!")
                print(f"   Found {len(orders)} other orders:")
                for order in orders:
                    print(f"   - {order['type']} {order['side']} @ ${float(order.get('price', 0)):.4f}")

        except Exception as e:
            print(f"❌ Error checking {symbol}: {e}")

    print("\n" + "="*60)
    print("✅ Check complete!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(check_stop_losses())
