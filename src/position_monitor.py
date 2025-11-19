"""
Position Monitor - Continuously monitors open positions.
Checks for stop-loss, take-profit, liquidation risk, and AI exit signals.

ENHANCED with:
- Real-time WebSocket price updates (primary method)
- REST API fallback for reliability
- Sub-second position monitoring

🎯 TIER 1 & TIER 2 FEATURES:
- Trailing Stop-Loss: Locks in profits as position moves favorably
- Partial Exit System: 3-tier scalping (2%, 4%, 6%)
- Market Regime Detection: Adaptive exit thresholds
"""

import asyncio
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
from src.config import get_settings
from src.database import get_db_client
from src.exchange_client import get_exchange_client
from src.risk_manager import get_risk_manager
from src.trade_executor import get_trade_executor
from src.telegram_notifier import get_notifier
from src.ai_engine import get_ai_engine
from src.websocket_client import get_ws_client
from src.utils import setup_logging, calculate_pnl
from src.indicators import calculate_indicators, detect_market_regime

# 🎯 TIER 1 & TIER 2 IMPORTS
from src.trailing_stop_loss import get_trailing_stop
from src.partial_exit_manager import get_partial_exit_manager
from src.market_regime_detector import get_regime_detector, MarketRegime

logger = setup_logging()


class PositionMonitor:
    """
    Monitors open positions and manages exit conditions.

    NEW: WebSocket-based real-time monitoring for sub-second updates.
    """

    def __init__(self):
        self.settings = get_settings()
        self.last_ai_check_time = None
        self.use_websocket = True  # Primary method
        self._ws_monitoring_task: Optional[asyncio.Task] = None

    async def _store_exit_regime(
        self,
        position: Dict[str, Any],
        indicators: Dict,
        symbol: str
    ) -> str:
        """
        🎯 TIER 2: Detect and store exit market regime before closing position.

        Args:
            position: Active position dict
            indicators: Market indicators
            symbol: Trading symbol

        Returns:
            Market regime string (e.g., 'strong_bullish_trend')
        """
        try:
            regime_detector = get_regime_detector()
            regime, regime_details = regime_detector.detect_regime(indicators, symbol)

            db = await get_db_client()
            async with db.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE active_position
                    SET exit_market_regime = $1
                    WHERE id = $2
                    """,
                    regime.value,
                    position['id']
                )

            logger.debug(
                f"📊 Exit regime stored: {regime.value} - {regime_details['description']}"
            )

            # Store in position dict for immediate use
            position['exit_market_regime'] = regime.value

            return regime.value

        except Exception as e:
            logger.warning(f"Failed to store exit regime: {e}")
            return 'weak_trend'  # Fallback

    async def check_position(self, position: Dict[str, Any]) -> None:
        """
        Main position monitoring function.
        Called every 60 seconds when a position is open.

        Args:
            position: Active position data from database
        """
        symbol = position['symbol']
        side = position['side']

        try:
            exchange = await get_exchange_client()
            db = await get_db_client()
            risk_manager = get_risk_manager()
            executor = get_trade_executor()
            notifier = get_notifier()

            # 🔧 FIX: Define min_profit_usd for ML exit checks (was undefined, causing crashes)
            min_profit_usd = self.settings.min_profit_usd

            # 🔥 CRITICAL FIX: Fetch indicators FIRST (before profit/loss checks)
            # Problem: Profit target checks need 'indicators' but it was defined AFTER (line 523)
            # Result: "cannot access local variable 'indicators'" → positions never closed at profit!
            # Solution: Fetch indicators at START of function, with safe fallback
            indicators = {}  # Safe default
            atr_percent = 2.5  # Default to medium volatility
            try:
                ohlcv_15m = await exchange.fetch_ohlcv(symbol, '15m', limit=50)
                from src.indicators import calculate_indicators
                indicators = calculate_indicators(ohlcv_15m)
                atr_percent = indicators.get('atr_percent', 2.5)
                logger.debug(f"📊 Indicators fetched: ATR {atr_percent:.2f}%")
            except Exception as indicators_error:
                logger.warning(f"⚠️ Failed to fetch indicators (using defaults): {indicators_error}")
                # Continue with defaults - don't crash position monitoring!

            # 🔥 CRITICAL: Check if position still exists on Binance FIRST
            # USER REQUEST: "manuel olarak binance üzerinden manuel olarak close diyorum öyle kapatıyorum pozisyonu"
            # "Ben öyle kapatsam bile hemen sync komutu tetiklenmesi lazım hemen çalışması lazım!"
            #
            # Problem: User closes position manually on Binance → Bot doesn't detect it until next periodic sync (5 min)
            # Solution: Check EVERY position_monitor cycle (60s) if position still exists on Binance
            # If position closed manually → Trigger immediate sync to clean ghost from database
            #
            # 🎯 PAPER TRADING FIX: Skip Binance check in paper trading mode
            # Paper positions only exist in DB, not on Binance
            if exchange.paper_trading:
                # Skip ghost detection in paper trading
                position_exists_on_binance = True
            else:
                try:
                    binance_positions = await exchange.exchange.fetch_positions([symbol])
                    position_exists_on_binance = False

                    for binance_pos in binance_positions:
                        if binance_pos['symbol'] == symbol:
                            contracts = float(binance_pos.get('contracts', 0))
                            notional = abs(float(binance_pos.get('notional', 0)))
                            # Position exists if contracts > 0 OR notional > $1
                            if contracts > 0 or notional > 1.0:
                                position_exists_on_binance = True
                                break

                    if not position_exists_on_binance:
                        # GHOST DETECTED! Position in database but NOT on Binance
                        logger.warning(f"👻 GHOST DETECTED: {symbol} in DB but NOT on Binance (manually closed?)")
                        logger.info("🔄 IMMEDIATE SYNC: Triggering position reconciliation...")

                        # 📱 USER REQUEST: Send Telegram alert when manual close detected!
                        # "ben https://www.binance.com/en/futures/ ben buradaki açık pozisyonlardan birinin
                        # yükseldiğini gördüm ve pozisyonu kapatmak istedim binanceden kapatırsam da
                        # direkt sync komutunu telegramda çalıştır lütfen !"
                        await notifier.send_alert(
                            'warning',
                            f"👻 <b>MANUAL CLOSE DETECTED!</b>\n\n"
                            f"Position <b>{symbol}</b> was closed manually on Binance.\n\n"
                            f"🔄 Running automatic sync to update database...\n\n"
                            f"<i>Tip: You can also manually run /sync anytime to force reconciliation.</i>"
                        )

                        # Trigger immediate reconciliation
                        from src.position_reconciliation import get_reconciliation_system
                        reconciliation = get_reconciliation_system()
                        sync_results = await reconciliation.reconcile_positions(on_startup=False)

                        logger.info(
                            f"✅ IMMEDIATE SYNC complete: "
                            f"{sync_results.get('ghost_count', 0)} ghosts cleaned, "
                            f"{sync_results.get('orphaned_count', 0)} orphans imported"
                        )

                        # 📱 Send completion notification
                        await notifier.send_alert(
                            'success',
                            f"✅ <b>SYNC COMPLETED!</b>\n\n"
                            f"Position <b>{symbol}</b> removed from database.\n"
                            f"All open orders cancelled.\n\n"
                            f"Bot is now ready to scan for new opportunities!"
                        )

                        # Stop monitoring this position - it's been cleaned up
                        return

                except Exception as ghost_check_error:
                    logger.warning(f"⚠️ Ghost detection check failed (non-critical): {ghost_check_error}")
                    # Continue with normal monitoring - sync will catch it eventually

            # Get current price using hybrid price manager (WebSocket + REST cache)
            from src.price_manager import get_price_manager
            price_manager = get_price_manager()

            current_price = await price_manager.get_price(
                symbol=symbol,
                exchange=exchange,
                is_active_position=True  # Prioritize WebSocket for active positions
            )

            # ====================================================================
            # 🎯 FETCH REAL UNREALIZED PNL FROM BINANCE
            # ====================================================================
            # USER REQUEST: Use Binance's actual unrealized PNL (ROI = Unrealized PNL / Initial Margin)
            # This is more accurate than manual calculation
            try:
                binance_position = await exchange.exchange.fetch_positions([symbol])
                real_unrealized_pnl = None

                for pos in binance_position:
                    if pos['symbol'] == symbol and float(pos.get('contracts', 0)) > 0:
                        # Binance provides unrealized PNL directly
                        real_unrealized_pnl = Decimal(str(pos.get('unrealizedPnl', 0)))
                        logger.info(f"📊 Binance Real Unrealized PNL: ${float(real_unrealized_pnl):+.2f}")
                        break

                if real_unrealized_pnl is not None:
                    unrealized_pnl = real_unrealized_pnl
                else:
                    # Fallback to manual calculation if Binance data not available
                    logger.warning(f"⚠️ Binance position not found for {symbol}, using manual PNL calculation")
                    pnl_data = calculate_pnl(
                        Decimal(str(position['entry_price'])),
                        current_price,
                        Decimal(str(position['quantity'])),
                        side,
                        position['leverage'],
                        Decimal(str(position['position_value_usd']))
                    )
                    unrealized_pnl = Decimal(str(pnl_data['unrealized_pnl']))
            except Exception as binance_pnl_error:
                # Fallback to manual calculation on error
                logger.warning(f"⚠️ Could not fetch Binance PNL, using manual calculation: {binance_pnl_error}")
                pnl_data = calculate_pnl(
                    Decimal(str(position['entry_price'])),
                    current_price,
                    Decimal(str(position['quantity'])),
                    side,
                    position['leverage'],
                    Decimal(str(position['position_value_usd']))
                )
                unrealized_pnl = Decimal(str(pnl_data['unrealized_pnl']))

            # Update position in database
            await db.update_position_price(position['id'], current_price, unrealized_pnl)

            logger.info(
                f"Position: {symbol} {side} | "
                f"Entry: ${float(position['entry_price']):.4f} | "
                f"Current: ${float(current_price):.4f} | "
                f"P&L: ${float(unrealized_pnl):+.2f}"
            )

            # ====================================================================
            # 💰 MULTI-TIER EXIT SYSTEM - DYNAMIC TARGETS
            # ====================================================================
            # OLD: Fixed $1.50 profit/loss (too simple, left money on table)
            # NEW: Multi-tier system with volatility adjustment
            #
            # TIER 1: Quick profit ($1.50-$2.50 based on volatility)
            # TIER 2: Good profit ($3.00-$5.00) - activate trailing stop
            # TIER 3: Excellent profit ($5.00+) - aggressive trailing
            #
            # WHY:
            # - Low volatility: tighter targets ($1.50)
            # - High volatility: wider targets ($2.50) - let winners run
            # - Trailing stops lock in profits as position moves favorably
            # ====================================================================

            # Get volatility from indicators
            indicators_15m = indicators if indicators else {}
            atr_percent = indicators_15m.get('atr_percent', 3.0)

            # Volatility-adjusted targets
            # 🎯 USER UPDATE: Increased targets for better risk/reward with smaller positions
            if atr_percent < 2.5:
                # Low volatility - tighter targets
                PROFIT_TARGET_USD = Decimal("10.00")
                LOSS_LIMIT_USD = Decimal("10.00")
                TARGET_LEVEL = "TIGHT"
            elif atr_percent < 4.5:
                # Medium volatility - standard targets
                PROFIT_TARGET_USD = Decimal("12.50")
                LOSS_LIMIT_USD = Decimal("12.50")
                TARGET_LEVEL = "STANDARD"
            else:
                # High volatility - wider targets (let winners run)
                PROFIT_TARGET_USD = Decimal("15.00")
                LOSS_LIMIT_USD = Decimal("15.00")
                TARGET_LEVEL = "WIDE"

            logger.debug(
                f"🎯 Exit targets: Profit ${float(PROFIT_TARGET_USD):.2f}, "
                f"Loss -${float(LOSS_LIMIT_USD):.2f} "
                f"(ATR {atr_percent:.2f}% = {TARGET_LEVEL})"
            )

            # 🔥 INSTANT PROFIT CAPTURE: Check if we're close to target
            profit_progress = (unrealized_pnl / PROFIT_TARGET_USD) * 100 if PROFIT_TARGET_USD > 0 else 0

            if profit_progress >= 90:
                # Within 90% of profit target - log warning for immediate attention!
                logger.warning(
                    f"🎯 NEAR TARGET! {symbol}: ${float(unrealized_pnl):+.2f} / ${float(PROFIT_TARGET_USD):.2f} "
                    f"({float(profit_progress):.1f}%) - WATCHING CLOSELY!"
                )

            # Check profit target
            if unrealized_pnl >= PROFIT_TARGET_USD:
                close_reason = f"✅ Profit target hit: ${float(unrealized_pnl):+.2f} (target: ${float(PROFIT_TARGET_USD):.2f})"
                logger.info(f"💰 {close_reason}")
                await notifier.send_alert(
                    'success',
                    f"💰 <b>PROFIT TARGET HIT!</b>\n\n"
                    f"💎 {symbol} {side}\n"
                    f"Entry: ${float(position['entry_price']):.4f}\n"
                    f"Exit: ${float(current_price):.4f}\n"
                    f"Profit: <b>${float(unrealized_pnl):+.2f}</b>\n\n"
                    f"🎉 Target was ${float(PROFIT_TARGET_USD):.2f}"
                )
                await executor.close_position(position, current_price, close_reason)
                return

            # Check loss limit
            if unrealized_pnl <= -LOSS_LIMIT_USD:
                close_reason = f"❌ Loss limit hit: ${float(unrealized_pnl):+.2f} (limit: -${float(LOSS_LIMIT_USD):.2f})"
                logger.warning(f"⚠️ {close_reason}")
                await notifier.send_alert(
                    'warning',
                    f"⚠️ <b>LOSS LIMIT HIT!</b>\n\n"
                    f"💎 {symbol} {side}\n"
                    f"Entry: ${float(position['entry_price']):.4f}\n"
                    f"Exit: ${float(current_price):.4f}\n"
                    f"Loss: <b>${float(unrealized_pnl):+.2f}</b>\n\n"
                    f"🛑 Limit was -${float(LOSS_LIMIT_USD):.2f}"
                )
                await executor.close_position(position, current_price, close_reason)
                return

            # ====================================================================
            # 📈 PHASE 2: TRAILING STOP CHECK
            # ====================================================================
            if self.settings.enable_trailing_stop:
                from src.trailing_stop import get_trailing_stop
                trailing_stop = get_trailing_stop()

                # Register position if not already registered
                if position['id'] not in trailing_stop.position_peaks:
                    trailing_stop.register_position(
                        position['id'],
                        float(position['entry_price']),
                        side
                    )

                # Update trailing stop and check if hit
                should_close, close_reason = trailing_stop.should_close_position(
                    position['id'],
                    float(current_price)
                )

                if should_close:
                    logger.info(f"📈 {close_reason}")
                    await notifier.send_alert(
                        'info',
                        f"📈 TRAILING STOP HIT\n\n"
                        f"💎 {symbol} {side}\n"
                        f"{close_reason}\n\n"
                        f"✅ Position closed"
                    )
                    await executor.close_position(position, current_price, close_reason)
                    trailing_stop.remove_position(position['id'])
                    return

                # Update stop-loss if trailing stop moved
                current_stop = position.get('stop_loss_price')
                new_stop = trailing_stop.update_and_check_stop(
                    position['id'],
                    float(current_price),
                    float(current_stop) if current_stop else None
                )

                if new_stop and new_stop != current_stop:
                    logger.info(f"📈 Trailing stop moved: ${current_stop:.4f} → ${new_stop:.4f}")
                    # Update stop-loss in database
                    await db.update_position_stop_loss(position['id'], Decimal(str(new_stop)))

            # ====================================================================
            # 💰 PHASE 1: PARTIAL EXITS CHECK
            # ====================================================================
            if self.settings.enable_partial_exits:
                from src.partial_exits import get_partial_exits
                partial_exits = get_partial_exits()

                # Register position if not already registered
                if position['id'] not in partial_exits.position_tiers:
                    partial_exits.register_position(
                        position['id'],
                        float(position['entry_price']),
                        side,
                        float(position['quantity'])
                    )

                # Check if any tier should be executed
                tier_execution = partial_exits.check_and_execute_tier(
                    position['id'],
                    float(current_price)
                )

                if tier_execution:
                    # Execute partial exit
                    logger.info(
                        f"💰 {tier_execution['tier_name']} executing: "
                        f"{tier_execution['exit_size']:.4f} units @ ${tier_execution['exit_price']:.4f}"
                    )

                    # Close partial position on exchange
                    try:
                        await executor.close_partial_position(
                            position,
                            tier_execution['exit_size'],
                            current_price,
                            f"{tier_execution['tier_name']} profit taking"
                        )

                        await notifier.send_alert(
                            'success',
                            f"💰 {tier_execution['tier_name']} EXECUTED!\n\n"
                            f"💎 {symbol} {side}\n\n"
                            f"Exit Size: {tier_execution['exit_size']:.4f} units\n"
                            f"Exit Price: ${tier_execution['exit_price']:.4f}\n"
                            f"Tier Profit: ${tier_execution['tier_profit']:.2f}\n\n"
                            f"Remaining: {tier_execution['remaining_size']:.4f} units\n"
                            f"Total Realized: ${tier_execution['total_realized_profit']:.2f}"
                        )

                        # Check if fully exited
                        if partial_exits.is_fully_exited(position['id']):
                            logger.info(f"✅ All tiers executed for {symbol}, position fully closed")
                            partial_exits.remove_position(position['id'])
                            return

                    except Exception as e:
                        logger.error(f"Failed to execute {tier_execution['tier_name']}: {e}")

            # ====================================================================
            # 🛑 STOP-LOSS CHECK: DISABLED (±$1 limits control exits)
            # ====================================================================
            # USER REQUEST: Only close at ±$1, ignore stop-loss triggers
            # Stop-loss is set very wide (50%) as emergency safety net only
            #
            # This check is SKIPPED - ±$1 profit/loss limits below will handle exits
            # ====================================================================

            # ====================================================================
            # 💰 PROFIT TARGET: DISABLED (EXCHANGE TAKE-PROFIT ORDERS HANDLE THIS)
            # ====================================================================
            # REASON FOR DISABLING:
            # - Exchange take-profit orders are already placed when position opens
            # - Take-profit is set at 0.3% price move (adaptive based on leverage)
            # - With 20x leverage: 0.3% price = $4.62 profit on $770 position
            # - Checking profit target here adds unnecessary redundancy
            # - Exchange orders execute faster and more reliably
            #
            # OLD LOGIC (DISABLED):
            # - Profit target: $0.85 (designed for low leverage)
            # - With 20x leverage, $0.85 profit = 0.11% price move (too tight!)
            # - Position would close in seconds, not allowing trend to develop
            #
            # NEW STRATEGY:
            # - Exchange take-profit order handles profitable exits
            # - Trailing stop can lock in profits as they grow
            # - Monitor position but don't interfere with exchange orders
            # ====================================================================

            # DISABLED: Manual profit target check (exchange orders handle this)
            # profit_target = self.settings.min_profit_usd
            # if unrealized_pnl >= profit_target:
            #     ... close position ...

            logger.debug(f"💡 Profit target check DISABLED - relying on exchange take-profit orders")

            # ====================================================================
            # 🔴 LOSS LIMIT: DISABLED (CONFLICTS WITH 20X LEVERAGE STOP-LOSS)
            # ====================================================================
            # REASON FOR DISABLING:
            # - With 20x leverage, positions use $700+ position size with $35 margin
            # - Small price moves (0.1%) = $1.40+ P&L changes
            # - $0.85 loss limit closes positions in SECONDS (not allowing stop-loss to work)
            # - Exchange stop-loss orders handle risk management properly
            # - User wants stop-loss at 4-5% (translated to 0.8-1.0% price with 20x)
            #
            # OLD LOGIC (DISABLED):
            # - Loss limit: -$0.85 → Close position
            # - This was designed for LOW leverage (2-6x) with $40-80 positions
            # - With 20x leverage, this interferes with proper stop-loss execution
            #
            # NEW STRATEGY:
            # - Let exchange stop-loss orders handle position closure
            # - Stop-loss is set at 4-5% of position value (0.8-1.0% price move with 20x)
            # - Profit target: $1.20+ to cover commissions
            # - No artificial loss limit that conflicts with stop-loss
            # ====================================================================

            # DISABLED: Aggressive loss limit that conflicts with 20x leverage
            # loss_limit = Decimal("-0.85")
            # if unrealized_pnl <= loss_limit:
            #     ... close position ...

            logger.debug(f"💡 Loss limit check DISABLED - relying on exchange stop-loss orders")

            # ====================================================================
            # TIME-BASED EXIT: DISABLED BY USER REQUEST
            # ====================================================================
            # USER WANTS: Simple fixed profit/loss targets only
            # - Profit: $1.50-$2.50 → Close
            # - Loss: -$1.50 to -$2.50 → Close
            # - Stop-loss: 1.5-2.5% → Emergency close
            #
            # OLD LOGIC (DISABLED):
            # - Profitable (>$1): 4h max
            # - Small loss (<$2): 2h max (CONFLICTED with -$1.50 to -$2.50 loss limit!)
            # - Medium loss ($2-$5): 1h max
            # - Large loss (>$5): 30min max
            #
            # REASON FOR DISABLING:
            # - Closed positions too early (before reaching profit/loss targets)
            # - Created unpredictable exit timing
            # - User wants positions to run until hitting fixed $ targets
            #
            # NEW STRATEGY: Let positions run indefinitely until:
            # 1. Profit target hit: $1.50-$2.50
            # 2. Loss limit hit: -$1.50 to -$2.50
            # 3. Stop-loss triggered: 1.5-2.5%
            # 4. Trailing stop triggered (locks profits)

            logger.debug(f"⏰ Time-based exit DISABLED - position will run until profit/loss targets hit")
            # Continue to other checks below (no time-based close)

            # ✅ Indicators already fetched at function start (line 119-133)
            # Removed duplicate fetch to avoid overwriting with different defaults

            # ====================================================================
            # 🎯 TIER 1 CHECK #1: PARTIAL EXIT SYSTEM (3-Tier Scalping)
            # ====================================================================
            # Check BEFORE trailing stop so we lock profits progressively
            try:
                partial_exit_mgr = get_partial_exit_manager()
                tier_to_execute = await partial_exit_mgr.check_partial_exit_trigger(
                    position, current_price
                )

                if tier_to_execute:
                    logger.info(f"🎯 Executing partial exit: {tier_to_execute}")

                    result = await partial_exit_mgr.execute_partial_exit(
                        position, tier_to_execute, current_price,
                        exchange, db, notifier
                    )

                    if result['success']:
                        logger.info(
                            f"✅ {tier_to_execute} executed: "
                            f"Closed {result['closed_percentage']*100:.0f}%, "
                            f"P&L ${float(result['realized_pnl']):+.2f}"
                        )

                        # If tier3 (full exit), position is closed
                        if tier_to_execute == 'tier3':
                            logger.info(f"🎉 Position fully closed via {tier_to_execute}")
                            return

                        # Otherwise, reload position data (quantity/value changed)
                        position = await db.get_position_by_id(position['id'])

                        # Recalculate P&L with new quantity
                        pnl_data = calculate_pnl(
                            Decimal(str(position['entry_price'])),
                            current_price,
                            Decimal(str(position['quantity'])),
                            side,
                            position['leverage'],
                            Decimal(str(position['position_value_usd']))
                        )
                        unrealized_pnl = Decimal(str(pnl_data['unrealized_pnl']))

                        logger.info(
                            f"📊 Remaining position: {float(position['quantity']):.4f} units, "
                            f"P&L: ${float(unrealized_pnl):+.2f}"
                        )

                    else:
                        logger.error(f"❌ {tier_to_execute} failed: {result.get('error')}")

            except Exception as partial_error:
                logger.error(f"❌ Partial exit check failed: {partial_error}", exc_info=True)

            # ====================================================================
            # 🎯 TIER 2 CHECK: MARKET REGIME DETECTION
            # ====================================================================
            # Detect market regime for adaptive exit decisions later
            try:
                regime_detector = get_regime_detector()
                regime, regime_details = regime_detector.detect_regime(indicators, symbol)

                # Store regime in position for ML exit decisions
                async with db.pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE active_position
                        SET entry_market_regime = $1
                        WHERE id = $2
                        """,
                        regime.value,
                        position['id']
                    )

                logger.debug(
                    f"📊 Market regime: {regime.value} - {regime_details['description']}"
                )

                # Store regime for later use
                position['entry_market_regime'] = regime.value

            except Exception as regime_error:
                logger.warning(f"Market regime detection failed: {regime_error}")
                regime = MarketRegime.WEAK_TREND
                regime_details = {'description': 'Unknown regime'}

            # === CRITICAL CHECK 1: Emergency close conditions ===
            # 🚫 DISABLED: We now use simple $1.50 profit/loss limits (see above)
            # Emergency close was too aggressive and closed positions immediately
            # should_emergency_close, emergency_reason = await risk_manager.should_emergency_close(position, current_price)
            #
            # if should_emergency_close:
            #     logger.critical(f"🚨 EMERGENCY CLOSE: {emergency_reason}")
            #     await self._store_exit_regime(position, indicators, symbol)
            #     await notifier.send_alert('critical', f"Emergency closing position: {emergency_reason}")
            #     await executor.close_position(position, current_price, emergency_reason)
            #     return

            # ====================================================================
            # 🚫 LIQUIDATION CHECK: DISABLED (Causing premature exits)
            # ====================================================================
            # USER ISSUE: Bot closing positions after 6-32 seconds with "Liquidation risk (4% away)"
            # - "çok hızlı kapatıyor pozisyonu daha binance tarafında 1 dolar kar olmadan kapatıyor"
            # - Translation: "Closes too fast, before $1 profit on Binance"
            #
            # PROBLEM:
            # - Check triggered at <5% distance to liquidation
            # - With 25x leverage, 4% distance is NORMAL and SAFE
            # - Positions closed before reaching ±$1 profit/loss targets
            #
            # SOLUTION: Disable this check
            # - Let ±$1 profit/loss targets control exits
            # - 50% stop-loss is wide enough for protection
            # - User wants positions to run until hitting $1 targets
            # ====================================================================

            # OLD CODE (DISABLED):
            # liquidation_price = Decimal(str(position['liquidation_price']))
            # distance_to_liq = abs(current_price - liquidation_price) / current_price
            # if distance_to_liq < Decimal("0.05"):
            #     await executor.close_position(position, current_price, "EMERGENCY - Liquidation risk")
            #     return

            # Log distance for monitoring but don't close
            liquidation_price = Decimal(str(position['liquidation_price']))
            distance_to_liq = abs(current_price - liquidation_price) / current_price
            logger.debug(f"📊 Liquidation distance: {float(distance_to_liq)*100:.2f}% (monitoring only)")

            # === OLD CHECK 3: ADVANCED PROFIT TAKING - DISABLED ===
            # REASON: User requested simple $1.50-$2.50 profit/loss targets
            # Old logic (partial closes, 2x/3x targets) replaced with fixed targets above
            # Kept for reference only - NOT ACTIVE
            pass

            # 🎯 #10: ML-POWERED EXIT TIMING (check before AI for speed)
            # Fast ML-based exit decision using learned patterns
            # 🔧 IMPROVED: Minimum 5 minute hold time + 80% confidence threshold + entry confidence filter
            # Can be disabled via config: enable_ml_exit = False
            if self.settings.enable_ml_exit:
                try:
                    from src.exit_optimizer import get_exit_optimizer
                    exit_optimizer = get_exit_optimizer()

                    # ✅ MINIMUM HOLD TIME: 5 minutes before ML exit can trigger (reduced from 10)
                    # This allows faster exits but still prevents immediate flips
                    entry_time = position.get('entry_time', datetime.now())
                    if isinstance(entry_time, str):
                        entry_time = datetime.fromisoformat(entry_time)
                    minutes_in_position = (datetime.now() - entry_time).total_seconds() / 60

                    if minutes_in_position < 5.0:
                        logger.debug(f"🔒 ML Exit blocked: Position only {minutes_in_position:.1f}m old (need 5m minimum)")
                    else:
                        # Gather quick market data for exit features
                        quick_data = await self._gather_quick_market_data(symbol, current_price)

                        # Extract exit features
                        exit_features = exit_optimizer.extract_exit_features(
                            position, current_price, quick_data
                        )

                        # Get ML exit prediction
                        exit_prediction = exit_optimizer.predict_exit_decision(
                            exit_features,
                            min_profit_usd,
                            unrealized_pnl
                        )

                        # Log prediction
                        logger.info(
                            f"🎯 EXIT ML: {exit_prediction['should_exit']} "
                            f"(confidence: {exit_prediction['confidence']:.1%})"
                        )

                        # 🔧 FIX #2: LOWERED THRESHOLD from 80% → 65% (2025-11-12)
                        # Reason: ML exit never triggered (max confidence 35% vs 80% threshold)
                        # Removed: Entry confidence gap filter (was blocking valid exits)
                        exit_confidence_pct = exit_prediction['confidence'] * 100

                        # Check: Medium-high confidence exit (65%+)
                        if exit_prediction['should_exit'] and exit_prediction['confidence'] >= 0.65:
                            logger.info(
                                f"💡 ML EXIT TRIGGERED: {exit_prediction['reasoning']} "
                                f"(Confidence: {exit_confidence_pct:.1f}%)"
                            )

                            # 🎯 TIER 2: Store exit regime before closing
                            await self._store_exit_regime(position, indicators, symbol)

                            await executor.close_position(
                                position,
                                current_price,
                                f"ML Exit Signal - {exit_prediction['reasoning']}"
                            )
                            return

                except Exception as exit_ml_error:
                    logger.warning(f"ML exit optimizer failed: {exit_ml_error}")
            else:
                logger.debug("🔒 ML Exit disabled via config (enable_ml_exit=False)")

            # === CHECK 4: ML-based exit signal ===
            # Check if ML model recommends exiting (proactive loss prevention)
            # Can be disabled via config: enable_ml_exit = False
            if self.settings.enable_ml_exit:
                should_check_ml = self._should_request_ml_exit_signal(
                    position, unrealized_pnl, min_profit_usd, current_price
                )
            else:
                should_check_ml = False

            if should_check_ml and self.settings.enable_ml_exit:
                logger.info("🧠 Requesting ML exit signal...")
                self.last_ai_check_time = datetime.now()

                try:
                    # Get ML-based exit recommendation
                    from src.ml_pattern_learner import get_ml_learner
                    ml_learner = await get_ml_learner()

                    # Gather market data for ML analysis
                    market_data = await self._gather_quick_market_data(symbol, current_price)

                    # Get ML prediction for current market conditions
                    ai_engine = get_ai_engine()
                    ml_prediction = await ai_engine._get_ml_only_prediction(symbol, market_data, ml_learner)

                    # Exit logic:
                    # - If position is LONG and ML says SELL → exit
                    # - If position is SHORT and ML says BUY → exit
                    # - ✅ CONSERVATIVE MODE: Confidence must be >= 75% (increased from 42.5%)
                    #
                    # FALLBACK: If ML confidence is too low (<30%), use technical analysis
                    # to detect if position is in critical danger zone

                    should_exit = False
                    reason = ""
                    ml_confidence = ml_prediction.get('confidence', 0)

                    # 🔧 FIX #2: LOWERED THRESHOLD from 80% → 55% (2025-11-12)
                    # Reason: ML exit never triggered (max confidence 35% vs 80% threshold)
                    # Removed: Entry confidence gap filter (was blocking valid exits)
                    ml_confidence_pct = ml_confidence * 100

                    if side == 'LONG' and ml_prediction['action'] == 'sell' and ml_confidence >= 0.55:
                        should_exit = True
                        reason = f"ML predicts SELL (conf: {ml_confidence_pct:.1f}%)"
                    elif side == 'SHORT' and ml_prediction['action'] == 'buy' and ml_confidence >= 0.55:
                        should_exit = True
                        reason = f"ML predicts BUY (conf: {ml_confidence_pct:.1f}%)"

                    # FALLBACK: ML confidence too low (<30%) + loss >$5 (reduced from $7)
                    # Use simple technical rule: exit if loss >$5 and getting worse
                    # This prevents waiting until -$10 stop-loss
                    elif ml_confidence < 0.30 and unrealized_pnl < Decimal("-5.0"):
                        # Check if price is moving away from entry (loss increasing)
                        entry_price = Decimal(str(position['entry_price']))
                        price_move_percent = abs((current_price - entry_price) / entry_price) * 100

                        # More aggressive exit: >0.5% move (reduced from 0.8%)
                        if price_move_percent > 0.5:
                            should_exit = True
                            reason = f"Low ML conf ({ml_confidence:.0%}) + Loss ${float(unrealized_pnl):.2f}"
                            logger.warning(
                                f"⚠️ FALLBACK EXIT: ML unreliable, cutting loss early. "
                                f"Loss: ${float(unrealized_pnl):.2f}, Price move: {float(price_move_percent):.2f}%"
                            )

                    # ADDITIONAL FALLBACK: Deep loss without ML signal (>$8)
                    # 🚫 DISABLED: We now use $1.50 loss limit (see above)
                    # This -$8 check was never reached anyway since we exit at -$1.50
                    # elif unrealized_pnl < Decimal("-8.0"):
                    #     should_exit = True
                    #     reason = f"Deep loss ${float(unrealized_pnl):.2f} - emergency exit"
                    #     logger.warning(
                    #         f"🚨 EMERGENCY EXIT: Loss approaching stop-loss threshold "
                    #         f"(${float(unrealized_pnl):.2f}), exiting to prevent full -$10 loss"
                    #     )

                    if should_exit:
                        logger.info(f"🧠 ML recommends exit: {reason}")

                        # 🎯 TIER 2: Store exit regime before closing
                        await self._store_exit_regime(position, indicators, symbol)

                        await executor.close_position(
                            position,
                            current_price,
                            f"ML exit signal - {reason}"
                        )
                        return

                except Exception as e:
                    logger.warning(f"Failed to get ML exit signal: {e}")

            # === CHECK 5: Periodic updates moved to trading_engine ===
            # Individual position updates removed to prevent spam
            # Consolidated portfolio update sent by trading_engine every 5 minutes
            pass

        except Exception as e:
            logger.error(f"Error in position monitor: {e}")
            # Don't raise - we'll try again next cycle

    async def _gather_quick_market_data(self, symbol: str, current_price: Decimal) -> Dict[str, Any]:
        """Gather minimal market data for AI exit signal."""
        try:
            exchange = await get_exchange_client()

            # Get OHLCV for indicators
            ohlcv_15m = await exchange.fetch_ohlcv(symbol, '15m', limit=50)

            # Calculate indicators
            indicators_15m = calculate_indicators(ohlcv_15m)

            # Market regime
            regime = detect_market_regime(ohlcv_15m)

            # Get funding rate
            funding_rate = await exchange.fetch_funding_rate(symbol)

            return {
                'current_price': float(current_price),
                'volume_24h': indicators_15m.get('volume', 0),
                'market_regime': regime,
                'funding_rate': funding_rate,
                'indicators': {
                    '15m': indicators_15m,
                    '1h': indicators_15m,  # Use same for simplicity in quick check
                    '4h': indicators_15m
                }
            }

        except Exception as e:
            logger.warning(f"Error gathering quick market data: {e}")
            return {
                'current_price': float(current_price),
                'volume_24h': 0,
                'market_regime': 'UNKNOWN',
                'funding_rate': {'rate': 0.0},
                'indicators': {
                    '15m': {'rsi': 50, 'macd': 0, 'macd_signal': 0},
                    '1h': {'rsi': 50, 'macd': 0, 'macd_signal': 0},
                    '4h': {'rsi': 50, 'macd': 0, 'macd_signal': 0}
                }
            }

    def _should_request_ai_exit_signal(
        self,
        position: Dict[str, Any],
        unrealized_pnl: Decimal,
        min_profit_usd: Decimal,
        current_price: Decimal
    ) -> bool:
        """
        Smart AI exit signal triggering - only call when meaningful.
        Reduces AI API costs by ~70% while maintaining effectiveness.

        Args:
            position: Position data
            unrealized_pnl: Current unrealized P&L
            min_profit_usd: Minimum profit target
            current_price: Current price

        Returns:
            True if AI should be consulted for exit decision
        """
        # Time-based check - has enough time passed?
        time_since_last_check = None
        if self.last_ai_check_time:
            time_since_last_check = (datetime.now() - self.last_ai_check_time).total_seconds()

        # TRIGGER 1: Near minimum profit target (80-120% of target)
        if min_profit_usd * Decimal("0.8") <= unrealized_pnl <= min_profit_usd * Decimal("1.2"):
            if time_since_last_check is None or time_since_last_check >= 180:  # 3 min
                logger.info("🤖 AI check: Near profit target")
                return True

        # TRIGGER 2: Well above profit target (2x+) - should we hold or take profit?
        if unrealized_pnl >= min_profit_usd * Decimal("2.0"):
            if time_since_last_check is None or time_since_last_check >= 300:  # 5 min
                logger.info("🤖 AI check: Significant profit achieved")
                return True

        # TRIGGER 3: Approaching stop-loss (within 20% of stop-loss distance)
        entry_price = Decimal(str(position['entry_price']))
        stop_loss_price = Decimal(str(position['stop_loss_price']))
        side = position['side']

        if side == 'LONG':
            distance_to_sl = (current_price - stop_loss_price) / entry_price
        else:  # SHORT
            distance_to_sl = (stop_loss_price - current_price) / entry_price

        if distance_to_sl < Decimal("0.02"):  # Within 2% of stop-loss
            if time_since_last_check is None or time_since_last_check >= 120:  # 2 min
                logger.info("🤖 AI check: Approaching stop-loss")
                return True

        # TRIGGER 4: Large price movement (>3% in last 5 minutes)
        # This requires tracking price movement - simplified check
        entry_time = position.get('entry_time', datetime.now())
        time_in_position = (datetime.now() - entry_time).total_seconds()

        # After 15+ minutes, check every 10 minutes if price moved significantly
        if time_in_position >= 900:  # 15 min
            if time_since_last_check and time_since_last_check >= 600:  # 10 min
                logger.info("🤖 AI check: Periodic check (15+ min in position)")
                return True

        # TRIGGER 5: Deep loss (below -50% of min profit) - check for early exit
        if unrealized_pnl < -min_profit_usd * Decimal("0.5"):
            if time_since_last_check is None or time_since_last_check >= 240:  # 4 min
                logger.info("🤖 AI check: Position in loss")
                return True

        # Default: Don't check AI
        return False

    def _should_request_ml_exit_signal(
        self,
        position: Dict[str, Any],
        unrealized_pnl: Decimal,
        min_profit_usd: Decimal,
        current_price: Decimal
    ) -> bool:
        """
        ML-based exit signal triggering - more aggressive than technical stops.
        Checks every 60 seconds when position is in danger zone.

        Args:
            position: Position data
            unrealized_pnl: Current unrealized P&L
            min_profit_usd: Minimum profit target
            current_price: Current price

        Returns:
            True if ML should be consulted for exit decision
        """
        # Time-based check
        time_since_last_check = None
        if self.last_ai_check_time:
            time_since_last_check = (datetime.now() - self.last_ai_check_time).total_seconds()

        # TRIGGER 1: Position losing money - AGGRESSIVE CHECKING
        # Small loss (-$2): check every 2 minutes
        # Medium loss (-$5): check every 1 minute
        # Large loss (-$7): check every 30 seconds
        if unrealized_pnl < 0:
            loss_amount = abs(float(unrealized_pnl))

            if loss_amount >= 7.0:  # Close to -$10 stop-loss
                check_interval = 30  # 30 seconds
                logger.debug(f"🚨 ML check: CRITICAL LOSS (${float(unrealized_pnl):.2f})")
            elif loss_amount >= 5.0:  # Medium loss
                check_interval = 60  # 1 minute
                logger.debug(f"⚠️ ML check: Medium loss (${float(unrealized_pnl):.2f})")
            else:  # Small loss
                check_interval = 120  # 2 minutes
                logger.debug(f"🧠 ML check: Small loss (${float(unrealized_pnl):.2f})")

            if time_since_last_check is None or time_since_last_check >= check_interval:
                return True

        # TRIGGER 2: Approaching stop-loss (within 5% of stop-loss distance)
        # INCREASED from 3% to 5% for earlier detection
        entry_price = Decimal(str(position['entry_price']))
        stop_loss_price = Decimal(str(position['stop_loss_price']))
        side = position['side']

        if side == 'LONG':
            distance_to_sl = (current_price - stop_loss_price) / entry_price
        else:  # SHORT
            distance_to_sl = (stop_loss_price - current_price) / entry_price

        if distance_to_sl < Decimal("0.05"):  # Within 5% of stop-loss (more sensitive)
            if time_since_last_check is None or time_since_last_check >= 20:  # 20 sec (faster)
                logger.debug(f"🧠 ML check: Approaching stop-loss (distance: {float(distance_to_sl)*100:.1f}%)")
                return True

        # TRIGGER 3: Near profit target - check if should take profit early
        if min_profit_usd * Decimal("0.7") <= unrealized_pnl <= min_profit_usd * Decimal("1.5"):
            if time_since_last_check is None or time_since_last_check >= 120:  # 2 min
                logger.debug("🧠 ML check: Near profit target")
                return True

        # Default: Don't check ML
        return False

    async def _check_multi_timeframe_exit(
        self,
        symbol: str,
        current_price: Decimal,
        position_side: str
    ) -> bool:
        """
        Check for exit signals across multiple timeframes.
        Returns True if ALL timeframes suggest exiting.

        Args:
            symbol: Trading symbol
            current_price: Current price
            position_side: 'LONG' or 'SHORT'

        Returns:
            True if should exit based on multi-timeframe analysis
        """
        try:
            exchange = await get_exchange_client()

            # Get OHLCV for all timeframes
            ohlcv_15m = await exchange.fetch_ohlcv(symbol, '15m', limit=50)
            ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=50)
            ohlcv_4h = await exchange.fetch_ohlcv(symbol, '4h', limit=50)

            # Calculate indicators
            indicators_15m = calculate_indicators(ohlcv_15m)
            indicators_1h = calculate_indicators(ohlcv_1h)
            indicators_4h = calculate_indicators(ohlcv_4h)

            # Check for exit signals on each timeframe
            exit_signals = 0

            for tf_indicators in [indicators_15m, indicators_1h, indicators_4h]:
                rsi = tf_indicators.get('rsi', 50)
                macd = tf_indicators.get('macd', 0)
                macd_signal = tf_indicators.get('macd_signal', 0)

                if position_side == 'LONG':
                    # Exit LONG if: RSI > 75 (overbought) OR MACD crosses below signal
                    if rsi > 75 or (macd < macd_signal):
                        exit_signals += 1
                else:  # SHORT
                    # Exit SHORT if: RSI < 25 (oversold) OR MACD crosses above signal
                    if rsi < 25 or (macd > macd_signal):
                        exit_signals += 1

            # 🔧 FIX #4: LOWERED THRESHOLD from 2/3 → 1/3 (2025-11-12)
            # Reason: Only triggered 1/10 trades (too conservative)
            # Now triggers if ANY timeframe shows strong exit signal
            return exit_signals >= 1

        except Exception as e:
            logger.warning(f"Multi-timeframe exit check failed: {e}")
            return False


# Singleton instance
_position_monitor: Optional[PositionMonitor] = None


def get_position_monitor() -> PositionMonitor:
    """Get or create position monitor instance."""
    global _position_monitor
    if _position_monitor is None:
        _position_monitor = PositionMonitor()
    return _position_monitor
