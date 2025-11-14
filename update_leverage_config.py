"""
Update database configuration for 10-20x aggressive leverage mode.
Updates stop-loss range to 5-8% and max leverage to 20x.
"""

import asyncio
import asyncpg
import ssl
from decimal import Decimal
from src.config import get_settings


async def update_leverage_config():
    """Update trading config for aggressive 10-20x leverage mode."""
    settings = get_settings()

    print("=" * 60)
    print("🔥 UPDATING TO AGGRESSIVE LEVERAGE MODE (10-20x)")
    print("=" * 60)

    # Create SSL context for Railway
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    conn = None
    try:
        # Connect to database
        print(f"\nConnecting to database...")
        conn = await asyncio.wait_for(
            asyncpg.connect(settings.database_url, ssl=ssl_context),
            timeout=10.0
        )
        print("✅ Connected!")

        # Get current config
        old_config = await conn.fetchrow("SELECT * FROM trading_config WHERE id = 1")
        print("\n📊 CURRENT Configuration:")
        print(f"  Max Leverage: {old_config['max_leverage']}x")
        print(f"  Stop Loss Range: {float(old_config['min_stop_loss_percent'])*100}% - {float(old_config['max_stop_loss_percent'])*100}%")

        # Update configuration with new aggressive settings
        print("\n🔄 Updating configuration...")
        await conn.execute("""
            UPDATE trading_config
            SET min_stop_loss_percent = $1,
                max_stop_loss_percent = $2,
                max_leverage = $3,
                last_updated = NOW()
            WHERE id = 1
        """,
            Decimal("0.05"),  # 5% min stop-loss
            Decimal("0.08"),  # 8% max stop-loss
            20                # 20x max leverage
        )

        # Verify update
        new_config = await conn.fetchrow("SELECT * FROM trading_config WHERE id = 1")
        print("\n✅ NEW Configuration:")
        print(f"  Max Leverage: {new_config['max_leverage']}x (was {old_config['max_leverage']}x)")
        print(f"  Stop Loss Range: {float(new_config['min_stop_loss_percent'])*100}% - {float(new_config['max_stop_loss_percent'])*100}%")
        print(f"  (was {float(old_config['min_stop_loss_percent'])*100}% - {float(old_config['max_stop_loss_percent'])*100}%)")

        print("\n" + "=" * 60)
        print("✅ CONFIGURATION UPDATE COMPLETE!")
        print("=" * 60)
        print("\n🔥 Bot is now ready for 10-20x aggressive trading!")
        print("   • Minimum leverage: 10x (low confidence)")
        print("   • Maximum leverage: 20x (high confidence)")
        print("   • Stop-loss range: 5-8% (tight stops)")
        print("   • Max risk per trade: $5-6 on $10 position")

    except asyncio.TimeoutError:
        print("❌ ERROR: Database connection timeout")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        if conn:
            await conn.close()

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(update_leverage_config())
        if not success:
            exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Update cancelled by user.")
        exit(0)
