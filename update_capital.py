"""
Update current_capital to $1000 in database.
Quick script to boost capital for more trading opportunities.
"""

import asyncio
from decimal import Decimal
from src.database import get_db_client
from src.utils import setup_logging

logger = setup_logging()


async def update_capital():
    """Update current_capital to $1000."""

    db = await get_db_client()

    print("=" * 60)
    print("💰 CAPITAL UPDATE TOOL")
    print("=" * 60)

    # Get current capital
    old_capital = await db.get_current_capital()
    print(f"\n📊 Current capital: ${float(old_capital):.2f}")

    # Update to $1000
    new_capital = Decimal("1000.00")

    async with db.pool.acquire() as conn:
        await conn.execute(
            "UPDATE trading_config SET current_capital = $1",
            new_capital
        )

    # Verify
    updated_capital = await db.get_current_capital()
    print(f"✅ Updated capital: ${float(updated_capital):.2f}")

    print(f"\n💡 Capital increased by: ${float(updated_capital - old_capital):+.2f}")
    print(f"\n🚀 Bot can now open up to 10 concurrent positions!")
    print(f"📊 Position size: ${float(updated_capital * Decimal('0.80')):.2f} each")

    await db.close()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(update_capital())
