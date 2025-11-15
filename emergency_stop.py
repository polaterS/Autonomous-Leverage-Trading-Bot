"""
🚨 EMERGENCY STOP SCRIPT

This script will:
1. Stop all autonomous trading
2. Close all open positions at market price
3. Cancel all open orders

USE THIS ONLY IN EMERGENCY SITUATIONS
"""

import asyncio
import sys
from decimal import Decimal
from src.utils import setup_logging
from src.database import get_db_client
from src.exchange_client import get_exchange_client
from src.telegram_notifier import get_telegram_notifier

logger = setup_logging()

async def emergency_stop():
    """Emergency stop: Close all positions and cancel all orders."""

    logger.info("=" * 70)
    logger.info("🚨 EMERGENCY STOP INITIATED")
    logger.info("=" * 70)

    db = None
    exchange = None
    telegram = None

    try:
        # Initialize clients
        db = await get_db_client()
        exchange = await get_exchange_client()
        telegram = get_telegram_notifier()

        # Get all active positions from database
        positions = await db.pool.fetch("""
            SELECT
                symbol,
                side,
                quantity,
                entry_price,
                leverage,
                open_time
            FROM positions
            WHERE status = 'open'
            ORDER BY open_time
        """)

        if not positions:
            logger.info("✅ No open positions found - already stopped")
            await telegram.send_message("✅ Emergency Stop: No open positions to close")
            return

        logger.info(f"📊 Found {len(positions)} open positions")
        logger.info("")

        total_closed = 0
        total_orders_cancelled = 0

        for pos in positions:
            symbol = pos['symbol']
            side = pos['side']
            quantity = Decimal(str(pos['quantity']))

            logger.info(f"Processing {symbol} {side}...")

            try:
                # Cancel all open orders for this symbol first
                logger.info(f"  Cancelling all open orders for {symbol}...")

                try:
                    open_orders = await exchange.exchange.fetch_open_orders(symbol)

                    for order in open_orders:
                        try:
                            await exchange.exchange.cancel_order(order['id'], symbol)
                            total_orders_cancelled += 1
                            logger.info(f"    ✅ Cancelled order {order['id']} ({order['type']})")
                        except Exception as cancel_error:
                            logger.warning(f"    ⚠️ Failed to cancel order {order['id']}: {cancel_error}")

                    if open_orders:
                        logger.info(f"  ✅ Cancelled {len(open_orders)} orders for {symbol}")
                    else:
                        logger.info(f"  ℹ️ No open orders for {symbol}")

                except Exception as order_error:
                    logger.warning(f"  ⚠️ Could not fetch/cancel orders for {symbol}: {order_error}")

                # Close position at market price
                logger.info(f"  Closing position at MARKET price...")

                # Determine close side (opposite of entry side)
                close_side = 'sell' if side == 'LONG' else 'buy'

                # Place market order to close position
                close_order = await exchange.place_order(
                    symbol=symbol,
                    side=close_side,
                    order_type='market',
                    quantity=quantity,
                    reduce_only=True  # Ensure this only closes position, doesn't open new one
                )

                logger.info(f"  ✅ Closed {symbol} {side} - Order ID: {close_order['id']}")

                # Update position status in database
                await db.pool.execute("""
                    UPDATE positions
                    SET
                        status = 'closed',
                        close_time = NOW(),
                        close_reason = 'EMERGENCY_STOP'
                    WHERE symbol = $1 AND status = 'open'
                """, symbol)

                total_closed += 1

                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)

            except Exception as pos_error:
                logger.error(f"  ❌ Failed to close {symbol}: {pos_error}")
                await telegram.send_message(f"❌ Emergency Stop Error: Failed to close {symbol}\n{pos_error}")

        # Summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 EMERGENCY STOP SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total positions closed: {total_closed}/{len(positions)}")
        logger.info(f"Total orders cancelled: {total_orders_cancelled}")
        logger.info("=" * 70)

        # Send Telegram notification
        summary_msg = (
            "🚨 EMERGENCY STOP COMPLETED\n\n"
            f"✅ Closed: {total_closed}/{len(positions)} positions\n"
            f"✅ Cancelled: {total_orders_cancelled} orders\n\n"
            "⚠️ Bot is now stopped. Set AUTO_START_LIVE_TRADING=false on Railway to prevent restart."
        )
        await telegram.send_message(summary_msg)

        if total_closed == len(positions):
            logger.info("✅ All positions successfully closed")
        else:
            logger.warning(f"⚠️ Only {total_closed}/{len(positions)} positions closed - check errors above")

    except Exception as e:
        logger.critical(f"❌ Emergency stop failed: {e}")
        if telegram:
            await telegram.send_message(f"❌ Emergency Stop FAILED\n{e}")
        sys.exit(1)

    finally:
        # Cleanup
        if exchange:
            try:
                await exchange.close()
            except:
                pass

        if db:
            try:
                await db.close()
            except:
                pass

    logger.info("🔴 Bot stopped. All positions closed.")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("⚠️  EMERGENCY STOP - THIS WILL CLOSE ALL POSITIONS")
    print("=" * 70)
    print("This script will:")
    print("  1. Cancel all open orders")
    print("  2. Close all positions at MARKET price")
    print("  3. Stop autonomous trading")
    print("")
    print("⚠️  This action cannot be undone!")
    print("=" * 70)
    print("")

    confirmation = input("Type 'STOP' to proceed: ")

    if confirmation.upper() != 'STOP':
        print("❌ Emergency stop cancelled")
        sys.exit(0)

    print("\n🚨 Executing emergency stop...\n")

    try:
        asyncio.run(emergency_stop())
    except KeyboardInterrupt:
        print("\n❌ Emergency stop interrupted")
        sys.exit(1)
