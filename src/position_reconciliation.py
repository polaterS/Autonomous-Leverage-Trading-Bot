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
        self.sync_interval_seconds = 300  # 5 minutes
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
                    # Ghost positions should be removed (Binance is source of truth)
                    await db.remove_active_position(ghost['id'])
                    logger.info(f"🗑️ Removed ghost position: {ghost['symbol']}")
                    results['actions_taken'].append(
                        f"🗑️ Removed ghost {ghost['symbol']} from database"
                    )

                # Send alert
                await notifier.send_alert(
                    'info',
                    f"👻 <b>GHOST POSITIONS CLEANED</b>\n\n"
                    f"Removed {len(ghost_positions)} position(s) from database that were NOT on Binance:\n\n" +
                    "\n".join([
                        f"• {p['symbol']} {p['side']}"
                        for p in ghost_positions
                    ])
                )

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

            # Filter only open positions (contracts > 0)
            open_positions = []

            for pos in positions:
                contracts = float(pos.get('contracts', 0))
                if contracts > 0:
                    # Standardize format
                    notional = float(pos.get('notional', 0))
                    entry_price = float(pos.get('entryPrice', 0))

                    # Determine side
                    side = 'LONG' if pos.get('side') == 'long' else 'SHORT'

                    open_positions.append({
                        'symbol': pos['symbol'],
                        'side': side,
                        'contracts': contracts,
                        'notional': abs(notional),  # Position value in USD
                        'entry_price': entry_price,
                        'unrealized_pnl': float(pos.get('unrealizedPnl', 0)),
                        'leverage': int(pos.get('leverage', 1)),
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
            position_value_usd = notional / Decimal(str(leverage))

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

            # Send notification
            notifier = get_notifier()
            await notifier.send_alert(
                'success',
                f"✅ <b>POSITION RECOVERED</b>\n\n"
                f"<b>{symbol}</b> {side} {leverage}x\n"
                f"Entry: ${float(entry_price):.4f}\n"
                f"Current: ${float(current_price):.4f}\n"
                f"Value: ${float(position_value_usd):.2f}\n\n"
                f"ℹ️ This position was on Binance but not in database.\n"
                f"Now tracking it automatically."
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


# Singleton instance
_reconciliation_system: Optional[PositionReconciliationSystem] = None


def get_reconciliation_system() -> PositionReconciliationSystem:
    """Get or create position reconciliation system instance."""
    global _reconciliation_system
    if _reconciliation_system is None:
        _reconciliation_system = PositionReconciliationSystem()
    return _reconciliation_system
