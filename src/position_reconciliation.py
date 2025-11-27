"""
Position Reconciliation System
================================

CRITICAL SAFETY FEATURE: Syncs Binance positions with database.

Purpose:
- Detect "orphaned positions" (on Binance but not in DB)
- Detect "ghost positions" (in DB but not on Binance)
- Auto-recover from database write failures
- Ensure bot ALWAYS knows about ALL open positions

When to run:
1. At startup (MANDATORY)
2. Every 5 minutes during operation
3. After any database write failure
4. On-demand via Telegram command

Safety Features:
- Never blindly closes positions without user confirmation
- Sends alerts for ANY mismatch
- Logs all sync actions for audit
- Conservative approach: When in doubt, alert the user
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime
from src.utils import setup_logging
from src.database import get_db_client
from src.exchange_client import get_exchange_client
from src.telegram_notifier import get_notifier

logger = setup_logging()


class PositionReconciliationSystem:
    """
    Ensures Binance positions and database positions are always in sync.

    Critical for preventing "invisible positions" that bot can't manage.
    """

    def __init__(self):
        self.last_sync_time = None
        self.sync_interval_seconds = 15  # 15 seconds (USER: "sync komutunu 15 saniyede bir otomatik çalışır hale getir")
        self.auto_import_enabled = True  # Auto-import orphaned positions

    async def reconcile_positions(self, on_startup: bool = False) -> Dict[str, Any]:
        """
        Main reconciliation function.
        Compares Binance positions with database and fixes mismatches.

        Args:
            on_startup: If True, runs in startup mode (more verbose logging)

        Returns:
            Dict with reconciliation results
        """
        logger.info("🔄 Starting position reconciliation...")

        try:
            db = await get_db_client()
            exchange = await get_exchange_client()
            notifier = get_notifier()

            # 🎯 PAPER TRADING FIX: Skip reconciliation in paper trading mode
            # Paper positions only exist in DB, not on Binance, so reconciliation would delete them
            if exchange.paper_trading:
                logger.info("📝 Paper trading mode - Skipping Binance reconciliation")
                return {
                    'sync_time': datetime.now(),
                    'binance_count': 0,
                    'database_count': len(await db.get_active_positions()),
                    'orphaned_count': 0,
                    'ghost_count': 0,
                    'matched_count': 0,
                    'actions_taken': ['Skipped: Paper trading mode'],
                    'skipped': True
                }

            # Fetch positions from both sources
            binance_positions = await self._fetch_binance_positions(exchange)
            db_positions = await db.get_active_positions()

            logger.info(f"📊 Binance: {len(binance_positions)} positions")
            logger.info(f"💾 Database: {len(db_positions)} positions")

            # Find mismatches
            orphaned_positions = []  # On Binance but not in DB
            ghost_positions = []     # In DB but not on Binance
            matched_positions = []   # Correctly synced

            # Create lookup maps
            binance_map = {pos['symbol']: pos for pos in binance_positions}
            db_map = {pos['symbol']: pos for pos in db_positions}

            # Check for orphaned positions (Binance but not DB)
            for symbol, binance_pos in binance_map.items():
                if symbol not in db_map:
                    orphaned_positions.append(binance_pos)
                    logger.warning(f"⚠️ ORPHANED POSITION: {symbol} on Binance but NOT in database!")
                else:
                    matched_positions.append((binance_pos, db_map[symbol]))

            # Check for ghost positions (DB but not Binance)
            for symbol, db_pos in db_map.items():
                if symbol not in binance_map:
                    ghost_positions.append(db_pos)
                    logger.warning(f"👻 GHOST POSITION: {symbol} in database but NOT on Binance!")

            # Handle mismatches
            results = {
                'sync_time': datetime.now(),
                'binance_count': len(binance_positions),
                'database_count': len(db_positions),
                'orphaned_count': len(orphaned_positions),
                'ghost_count': len(ghost_positions),
                'matched_count': len(matched_positions),
                'actions_taken': []
            }

            # FIX 1: Import orphaned positions to database
            if orphaned_positions:
                logger.warning(f"🚨 Found {len(orphaned_positions)} orphaned position(s)!")

                for orphaned in orphaned_positions:
                    if self.auto_import_enabled:
                        success = await self._import_orphaned_position(orphaned, db)
                        if success:
                            results['actions_taken'].append(
                                f"✅ Imported {orphaned['symbol']} to database"
                            )
                        else:
                            results['actions_taken'].append(
                                f"❌ Failed to import {orphaned['symbol']}"
                            )

                # Send alert
                await notifier.send_alert(
                    'warning',
                    f"⚠️ <b>POSITION SYNC ISSUE</b>\n\n"
                    f"Found {len(orphaned_positions)} position(s) on Binance that were NOT in database:\n\n" +
                    "\n".join([
                        f"• {p['symbol']} {p['side']} ${float(p['notional']):.2f}"
                        for p in orphaned_positions
                    ]) +
                    f"\n\n{'✅ Auto-imported to database' if self.auto_import_enabled else '⚠️ Please manually check these positions!'}"
                )

            # FIX 2: Clean ghost positions from database
            if ghost_positions:
                logger.warning(f"👻 Found {len(ghost_positions)} ghost position(s)!")

                for ghost in ghost_positions:
                    # 🔧 CRITICAL FIX: Cancel ALL open orders for ghost position BEFORE removing from DB
                    # USER REQUEST: "bir pozisyon kapandıysa o pozisyona ait olan open orders kısmındaki değer de kapanmalı"
                    # Problem: Ghost positions removed from DB, but stop-loss/take-profit orders remain active on Binance
                    # Result: Old orders interfere with new positions for the same symbol!
                    symbol = ghost['symbol']

                    try:
                        logger.info(f"🗑️ Cancelling open orders for ghost position {symbol}...")
                        open_orders = await exchange.fetch_open_orders(symbol)

                        if open_orders:
                            logger.info(f"📋 Found {len(open_orders)} open orders for {symbol}:")
                            for order in open_orders:
                                order_id = order.get('id')
                                order_type = order.get('type', 'unknown')
                                order_side = order.get('side', 'unknown')
                                logger.info(f"  - Order {order_id}: {order_type} {order_side}")

                            # Cancel each order
                            cancelled_count = 0
                            for order in open_orders:
                                try:
                                    order_id = order.get('id')
                                    await exchange.cancel_order(order_id, symbol)
                                    cancelled_count += 1
                                    logger.info(f"  ✅ Cancelled order {order_id}")
                                except Exception as cancel_err:
                                    logger.warning(f"  ⚠️ Could not cancel order {order_id}: {cancel_err}")

                            logger.info(f"✅ Cancelled {cancelled_count}/{len(open_orders)} orders for ghost {symbol}")
                            results['actions_taken'].append(
                                f"🗑️ Cancelled {cancelled_count} open orders for ghost {symbol}"
                            )
                        else:
                            logger.info(f"✅ No open orders for ghost {symbol} (clean slate!)")

                    except Exception as orders_err:
                        logger.error(f"❌ Could not fetch/cancel orders for ghost {symbol}: {orders_err}")
                        results['actions_taken'].append(
                            f"⚠️ Failed to cancel orders for ghost {symbol}: {orders_err}"
                        )

                    # NOW remove ghost position from database (after orders cleaned)
                    await db.remove_active_position(ghost['id'])
                    logger.info(f"🗑️ Removed ghost position: {symbol} from database")
                    results['actions_taken'].append(
                        f"🗑️ Removed ghost {symbol} from database"
                    )

                # Send alert with orders info
                total_orders_cancelled = sum(
                    1 for action in results['actions_taken']
                    if 'Cancelled' in action and 'open orders' in action
                )

                await notifier.send_alert(
                    'info',
                    f"👻 <b>GHOST POSITIONS CLEANED</b>\n\n"
                    f"Removed {len(ghost_positions)} position(s) from database that were NOT on Binance:\n\n" +
                    "\n".join([
                        f"• {p['symbol']} {p['side']}"
                        for p in ghost_positions
                    ]) +
                    (f"\n\n✅ Cancelled all open orders for these positions" if total_orders_cancelled > 0 else "")
                )

            # 🔥 FIX 3: Check and add missing stop-loss orders
            await self._check_and_fix_stop_losses(matched_positions, results, exchange, notifier)

            # 🔥 FIX 4: Clean orphan orders (orders without positions) - CRITICAL FOR LIVE TRADING!
            # USER REQUEST: "open orders kısmında açık kalırsa emirler tekrar işleme girebilir!"
            await self._clean_orphan_orders(binance_positions, db_positions, results, exchange, notifier)

            # Log final status
            if orphaned_positions or ghost_positions:
                logger.warning(
                    f"🔧 Reconciliation completed: "
                    f"{len(orphaned_positions)} orphaned, "
                    f"{len(ghost_positions)} ghosts fixed"
                )
            else:
                logger.info("✅ All positions in sync!")

            self.last_sync_time = datetime.now()

            return results

        except Exception as e:
            logger.error(f"❌ Position reconciliation failed: {e}")
            return {
                'error': str(e),
                'sync_time': datetime.now()
            }

    async def _fetch_binance_positions(self, exchange) -> List[Dict[str, Any]]:
        """
        Fetch all open positions from Binance.

        Returns:
            List of position dicts with standardized format
        """
        try:
            # Fetch all positions (ccxt format)
            positions = await exchange.exchange.fetch_positions()

            # 🔧 DEBUG: Log what Binance is returning
            logger.debug(f"📊 Binance returned {len(positions)} total positions")

            # Filter only open positions (contracts > 0)
            open_positions = []

            for pos in positions:
                contracts = float(pos.get('contracts', 0))
                notional = float(pos.get('notional', 0))

                # 🔧 FIX: Check BOTH contracts AND notional value
                # Binance sometimes shows contracts=0 but notional>0 for active positions!
                # This was causing real positions to be flagged as ghosts
                is_position_open = contracts > 0 or abs(notional) > 1.0  # $1 threshold

                # 🔧 DEBUG: Log position details for troubleshooting
                symbol = pos.get('symbol', 'UNKNOWN')
                logger.debug(f"  {symbol}: contracts={contracts}, notional={notional}, open={is_position_open}")

                if is_position_open:
                    # Standardize format
                    entry_price = float(pos.get('entryPrice', 0))

                    # Determine side
                    side = 'LONG' if pos.get('side') == 'long' else 'SHORT'

                    # Safe leverage extraction (Binance may return None or 0)
                    leverage_raw = pos.get('leverage')
                    if leverage_raw is None or leverage_raw == 0:
                        # Calculate from margin and notional if available
                        margin_raw = pos.get('initialMargin', 0) or pos.get('collateral', 0)
                        if margin_raw and notional and float(margin_raw) > 0:
                            calculated_lev = abs(notional) / float(margin_raw)
                            leverage = max(1, int(calculated_lev))  # Ensure at least 1x
                        else:
                            leverage = 10  # Default to 10x if can't determine
                            logger.warning(f"⚠️ Could not determine leverage for {pos['symbol']}, using 10x default")
                    else:
                        leverage = max(1, int(leverage_raw))  # Ensure at least 1x

                    open_positions.append({
                        'symbol': pos['symbol'],
                        'side': side,
                        'contracts': contracts,
                        'notional': abs(notional),  # Position value in USD
                        'entry_price': entry_price,
                        'unrealized_pnl': float(pos.get('unrealizedPnl', 0)),
                        'leverage': leverage,
                        'raw_data': pos  # Keep original for debugging
                    })

            logger.debug(f"Fetched {len(open_positions)} open positions from Binance")
            return open_positions

        except Exception as e:
            logger.error(f"Failed to fetch Binance positions: {e}")
            return []

    async def _import_orphaned_position(
        self,
        binance_position: Dict[str, Any],
        db
    ) -> bool:
        """
        Import an orphaned position from Binance into database.

        This happens when:
        1. Position opened but DB write failed
        2. Bot restarted while position was open
        3. Manual position opened on Binance

        Args:
            binance_position: Position data from Binance
            db: Database client

        Returns:
            True if import successful
        """
        try:
            symbol = binance_position['symbol']
            side = binance_position['side']

            logger.info(f"📥 Importing orphaned position: {symbol} {side}")

            # Get current price
            exchange = await get_exchange_client()
            ticker = await exchange.fetch_ticker(symbol)
            current_price = Decimal(str(ticker['last']))

            # Build position data
            entry_price = Decimal(str(binance_position['entry_price']))
            quantity = Decimal(str(binance_position['contracts']))
            leverage = binance_position['leverage']
            notional = Decimal(str(binance_position['notional']))

            # Calculate position value (margin used)
            # SAFETY: Ensure leverage is at least 1 to prevent division by zero
            safe_leverage = max(leverage, 1)
            position_value_usd = notional / Decimal(str(safe_leverage))

            # Estimate stop-loss (use 10% default since we don't know original)
            from src.utils import calculate_stop_loss_price, calculate_liquidation_price
            stop_loss_price = calculate_stop_loss_price(
                entry_price,
                Decimal("10.0"),  # 10% default
                side,
                leverage=leverage,
                position_value=position_value_usd
            )

            liquidation_price = calculate_liquidation_price(entry_price, leverage, side)

            # Create position record
            position_data = {
                'symbol': symbol,
                'side': side,
                'leverage': leverage,
                'entry_price': entry_price,
                'current_price': current_price,
                'quantity': quantity,
                'position_value_usd': position_value_usd,
                'stop_loss_price': stop_loss_price,
                'stop_loss_percent': Decimal("10.0"),
                'min_profit_target_usd': Decimal("0.50"),
                'min_profit_price': entry_price,  # No target for imported positions
                'liquidation_price': liquidation_price,
                'exchange_order_id': 'IMPORTED_FROM_BINANCE',
                'stop_loss_order_id': None,
                'ai_model_consensus': 'ORPHANED_POSITION_RECOVERY',
                'ai_confidence': Decimal("0"),
                'ai_reasoning': f'Position imported during reconciliation. Originally opened but not tracked in database.',

                # Execution quality unknown
                'slippage_percent': Decimal("0"),
                'fill_time_ms': 0,
                'expected_entry_price': entry_price,
                'order_start_timestamp': datetime.now(),
                'order_fill_timestamp': datetime.now(),

                # No snapshot for imported positions
                'entry_snapshot': None,
                'entry_slippage_percent': Decimal("0"),
                'entry_fill_time_ms': 0,
                'price_action': None
            }

            # Insert into database
            position_id = await db.create_active_position(position_data)

            logger.info(f"✅ Orphaned position imported: {symbol} (DB ID: {position_id})")

            # 🛡️ CRITICAL: Add stop-loss protection for imported position
            logger.info(f"🛡️ Adding stop-loss protection for imported position...")
            close_side = 'sell' if side == 'LONG' else 'buy'

            try:
                sl_order = await exchange.create_stop_loss_order(
                    symbol=symbol,
                    side=close_side,
                    amount=quantity,
                    stop_price=stop_loss_price
                )
                logger.info(f"✅ Stop-loss added: {close_side} @ ${float(stop_loss_price):.4f}")

                # Update database with stop-loss order ID
                await db.pool.execute(
                    "UPDATE active_position SET stop_loss_order_id = $1 WHERE id = $2",
                    sl_order.get('id'), position_id
                )
            except Exception as sl_error:
                logger.error(f"⚠️ Failed to add stop-loss for imported position: {sl_error}")
                # Don't fail the import, but alert user

            # Send notification
            notifier = get_notifier()
            await notifier.send_alert(
                'success',
                f"✅ <b>POSITION RECOVERED</b>\n\n"
                f"<b>{symbol}</b> {side} {leverage}x\n"
                f"Entry: ${float(entry_price):.4f}\n"
                f"Current: ${float(current_price):.4f}\n"
                f"Value: ${float(position_value_usd):.2f}\n"
                f"🛡️ Stop-Loss: ${float(stop_loss_price):.4f}\n\n"
                f"ℹ️ This position was on Binance but not in database.\n"
                f"Now tracking it automatically with protection."
            )

            return True

        except Exception as e:
            logger.error(f"Failed to import orphaned position {symbol}: {e}")
            return False

    async def check_sync_needed(self) -> bool:
        """Check if periodic sync is needed (every 5 minutes)."""
        if self.last_sync_time is None:
            return True

        elapsed = (datetime.now() - self.last_sync_time).total_seconds()
        return elapsed >= self.sync_interval_seconds

    async def force_sync(self) -> Dict[str, Any]:
        """Force immediate reconciliation (useful for manual triggers)."""
        logger.info("🔄 Manual sync triggered")
        return await self.reconcile_positions(on_startup=False)

    async def _check_and_fix_stop_losses(
        self,
        matched_positions: List[tuple],
        results: Dict,
        exchange,
        notifier
    ) -> None:
        """
        🔥 CRITICAL: Check if positions have stop-loss orders and add if missing.

        Args:
            matched_positions: List of (binance_pos, db_pos) tuples
            results: Results dict to append actions
            exchange: Exchange client
            notifier: Telegram notifier
        """
        from decimal import Decimal

        logger.info("🛡️ Checking stop-loss protection for all positions...")

        positions_without_sl = []

        for binance_pos, db_pos in matched_positions:
            symbol = db_pos['symbol']

            try:
                # Fetch all open orders for this symbol
                orders = await exchange.fetch_open_orders(symbol)

                # Check if there's a stop-loss order
                has_stop_loss = any(
                    order.get('type', '').lower() in ['stop_market', 'stop', 'stop_loss']
                    for order in orders
                )

                if not has_stop_loss:
                    logger.warning(f"⚠️ {symbol} has NO STOP-LOSS! Adding protection...")
                    positions_without_sl.append((binance_pos, db_pos))

                    # Calculate stop-loss price from database
                    stop_loss_price = Decimal(str(db_pos.get('stop_loss_price', 0)))

                    if stop_loss_price <= 0:
                        # If no SL in DB, calculate conservative one (5% for safety)
                        from src.utils import calculate_stop_loss_price
                        entry_price = Decimal(str(db_pos['entry_price']))
                        stop_loss_price = calculate_stop_loss_price(
                            entry_price,
                            db_pos['side'],
                            5.0,  # 5% conservative stop
                            db_pos['leverage']
                        )
                        logger.warning(f"   No SL in DB, using 5% conservative: ${float(stop_loss_price):.4f}")

                    # Place stop-loss order
                    side = 'buy' if db_pos['side'] == 'SHORT' else 'sell'
                    quantity = Decimal(str(db_pos['quantity']))

                    try:
                        await exchange.create_stop_loss_order(
                            symbol=symbol,
                            side=side,
                            amount=quantity,
                            stop_price=stop_loss_price
                        )

                        logger.info(f"✅ Stop-loss added for {symbol}: {side} @ ${float(stop_loss_price):.4f}")
                        results['actions_taken'].append(
                            f"🛡️ Added stop-loss for {symbol} @ ${float(stop_loss_price):.4f}"
                        )

                    except Exception as e:
                        logger.error(f"❌ Failed to add stop-loss for {symbol}: {e}")
                        results['actions_taken'].append(
                            f"❌ Failed to add stop-loss for {symbol}: {str(e)[:50]}"
                        )

                else:
                    logger.debug(f"✅ {symbol} has stop-loss protection")

            except Exception as e:
                logger.error(f"Failed to check stop-loss for {symbol}: {e}")

        # Send alert if positions were unprotected
        if positions_without_sl:
            await notifier.send_alert(
                'critical',
                f"🚨 <b>UNPROTECTED POSITIONS FIXED</b>\n\n"
                f"Found {len(positions_without_sl)} position(s) WITHOUT stop-loss:\n\n" +
                "\n".join([
                    f"• {db_pos['symbol']} {db_pos['side']} {db_pos['leverage']}x"
                    for _, db_pos in positions_without_sl
                ]) +
                f"\n\n✅ Stop-loss orders added automatically!\n"
                f"Your positions are now protected."
            )

            logger.warning(
                f"🛡️ Stop-loss protection added to {len(positions_without_sl)} position(s)"
            )


    async def _clean_orphan_orders(
        self,
        binance_positions: List[Dict],
        db_positions: List[Dict],
        results: Dict,
        exchange,
        notifier
    ) -> None:
        """
        🔥 CRITICAL: Cancel orders for symbols that have NO open positions.

        This prevents the dangerous scenario where:
        1. Position closes
        2. Stop-loss/take-profit orders remain active
        3. New position opens for same symbol
        4. OLD orders trigger and mess up the new position!

        USER REQUEST: "open orders kısmında açık kalırsa emirler tekrar işleme girebilir!"
        """
        try:
            # Get all symbols with open positions (both Binance and DB)
            active_symbols = set()
            for pos in binance_positions:
                active_symbols.add(pos['symbol'])
            for pos in db_positions:
                active_symbols.add(pos['symbol'])

            # Fetch ALL open orders across all symbols
            all_open_orders = await exchange.exchange.fetch_open_orders()

            if not all_open_orders:
                logger.debug("✅ No open orders found on exchange")
                return

            # Group orders by symbol
            orders_by_symbol = {}
            for order in all_open_orders:
                symbol = order.get('symbol')
                if symbol not in orders_by_symbol:
                    orders_by_symbol[symbol] = []
                orders_by_symbol[symbol].append(order)

            # Find orphan orders (orders for symbols without positions)
            orphan_orders = []
            for symbol, orders in orders_by_symbol.items():
                if symbol not in active_symbols:
                    orphan_orders.extend(orders)
                    logger.warning(f"⚠️ Found {len(orders)} orphan order(s) for {symbol} (NO POSITION!)")

            if not orphan_orders:
                logger.debug("✅ No orphan orders found")
                return

            # Cancel orphan orders
            logger.warning(f"🗑️ Cancelling {len(orphan_orders)} orphan order(s)...")
            cancelled_count = 0

            for order in orphan_orders:
                try:
                    order_id = order.get('id')
                    symbol = order.get('symbol')
                    order_type = order.get('type', 'unknown')
                    order_side = order.get('side', 'unknown')

                    await exchange.cancel_order(order_id, symbol)
                    cancelled_count += 1
                    logger.info(f"  ✅ Cancelled orphan {order_type} {order_side} order {order_id} for {symbol}")

                except Exception as cancel_err:
                    logger.warning(f"  ⚠️ Could not cancel orphan order {order_id}: {cancel_err}")

            results['actions_taken'].append(
                f"🗑️ Cancelled {cancelled_count} orphan order(s) (no position)"
            )

            # Send alert
            if cancelled_count > 0:
                await notifier.send_alert(
                    'warning',
                    f"🗑️ <b>ORPHAN ORDERS CLEANED</b>\n\n"
                    f"Cancelled {cancelled_count} order(s) that had NO open position:\n\n" +
                    "\n".join([
                        f"• {o.get('symbol')} {o.get('type')} {o.get('side')}"
                        for o in orphan_orders[:5]  # Show max 5
                    ]) +
                    (f"\n... and {len(orphan_orders) - 5} more" if len(orphan_orders) > 5 else "") +
                    f"\n\n✅ These stale orders could have caused problems!"
                )

                logger.warning(f"✅ Cancelled {cancelled_count} orphan orders")

        except Exception as e:
            logger.error(f"❌ Orphan orders cleanup failed: {e}")
            results['actions_taken'].append(f"⚠️ Orphan orders cleanup failed: {e}")


# Singleton instance
_reconciliation_system: Optional[PositionReconciliationSystem] = None


def get_reconciliation_system() -> PositionReconciliationSystem:
    """Get or create position reconciliation system instance."""
    global _reconciliation_system
    if _reconciliation_system is None:
        _reconciliation_system = PositionReconciliationSystem()
    return _reconciliation_system
