"""Quick test script for Binance Futures API"""
import asyncio
import ccxt.async_support as ccxt


async def test_binance():
    try:
        exchange = ccxt.binance({
            'apiKey': 'YOUR_BINANCE_API_KEY',
            'secret': 'YOUR_BINANCE_SECRET_KEY',
            'enableRateLimit': True,
            'timeout': 60000,  # 60 seconds timeout
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True,
            }
        })

        print("1. Testing connection...")
        await exchange.load_markets()
        print("✅ Markets loaded successfully!")

        print("\n2. Testing balance fetch...")
        balance = await exchange.fetch_balance()
        print(f"✅ Balance fetched: {balance.get('USDT', {}).get('free', 0)} USDT")

        print("\n3. Testing ticker fetch...")
        ticker = await exchange.fetch_ticker('BTC/USDT:USDT')
        print(f"✅ BTC/USDT price: ${ticker['last']}")

        await exchange.close()
        print("\n🎉 ALL TESTS PASSED!")

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}")
        print(f"Message: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(test_binance())
