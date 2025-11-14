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

            # Get current price using hybrid price manager (WebSocket + REST cache)
            from src.price_manager import get_price_manager
            price_manager = get_price_manager()

            current_price = await price_manager.get_price(
                symbol=symbol,
                exchange=exchange,
                is_active_position=True  # Prioritize WebSocket for active positions
            )

            # Calculate current P&L
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
            # 🛑 CRITICAL CHECK 0: MANUAL STOP-LOSS TRIGGER
            # ====================================================================
            # In paper trading mode, stop-loss orders are simulated and do NOT
            # auto-execute. We MUST manually check if price hit stop-loss.
            #
            # This is the #1 priority check - prevents catastrophic losses.
            # ADDED: 2025-11-12 after discovering 0/10 trades hit stop-loss
            stop_loss_price = Decimal(str(position['stop_loss_price']))

            sl_triggered = False
            if side == 'LONG':
                # LONG: Close if price drops to or below stop-loss
                if current_price <= stop_loss_price:
                    sl_triggered = True
            else:  # SHORT
                # SHORT: Close if price rises to or above stop-loss
                if current_price >= stop_loss_price:
                    sl_triggered = True

            if sl_triggered:
                logger.warning(
                    f"🛑 STOP-LOSS TRIGGERED: {symbol} {side} | "
                    f"Entry: ${float(position['entry_price']):.4f} | "
                    f"Current: ${float(current_price):.4f} | "
                    f"Stop: ${float(stop_loss_price):.4f} | "
                    f"Loss: ${float(unrealized_pnl):+.2f}"
                )

                await notifier.send_alert(
                    'warning',
                    f"🛑 STOP-LOSS HIT\n\n"
                    f"💎 {symbol}\n"
                    f"📊 {side} {position['leverage']}x\n\n"
                    f"📍 Entry: ${float(position['entry_price']):.4f}\n"
                    f"💥 Exit: ${float(current_price):.4f}\n"
                    f"🛑 Stop: ${float(stop_loss_price):.4f}\n\n"
                    f"💰 P&L: ${float(unrealized_pnl):+.2f}\n\n"
                    f"✅ Position closed automatically"
                )

                await executor.close_position(
                    position,
                    current_price,
                    "Stop-loss triggered"
                )
                return

            # ====================================================================
            # 🚀 QUICK PROFIT TAKE: Close at $1-2 profit (10-20x leverage mode)
            # ====================================================================
            # With high leverage (10-20x), small price movements = big profits
            # Take quick $1-2 profits to compound capital faster
            # This prevents giving back profits when market reverses

            if unrealized_pnl >= Decimal("1.0"):  # $1+ profit
                logger.info(
                    f"💰 QUICK PROFIT TAKE: {symbol} {side} | "
                    f"Entry: ${float(position['entry_price']):.4f} | "
                    f"Current: ${float(current_price):.4f} | "
                    f"Profit: ${float(unrealized_pnl):+.2f}"
                )

                await notifier.send_alert(
                    'success',
                    f"💰 QUICK PROFIT TAKE\n\n"
                    f"💎 {symbol}\n"
                    f"📊 {side} {position['leverage']}x\n\n"
                    f"📍 Entry: ${float(position['entry_price']):.4f}\n"
                    f"💰 Exit: ${float(current_price):.4f}\n\n"
                    f"✅ Profit: ${float(unrealized_pnl):+.2f}\n\n"
                    f"🚀 Fast profit locked in!"
                )

                await executor.close_position(
                    position,
                    current_price,
                    f"Quick profit take: ${float(unrealized_pnl):+.2f}"
                )
                return

            # ====================================================================
            # 🔧 FIX #5: TIME-BASED EXIT (GRADUATED APPROACH)
            # ====================================================================
            # OLD: 8 hours max for all positions
            # NEW: Dynamic based on P&L to cut losses faster
            # - Profitable (>$1): 4h max
            # - Small loss (<$2): 2h max
            # - Medium loss ($2-$5): 1h max
            # - Large loss (>$5): 30min max
            from datetime import datetime, timezone
            entry_time = position['entry_time']
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            elif not entry_time.tzinfo:
                entry_time = entry_time.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            hours_open = (now - entry_time).total_seconds() / 3600

            # 🔧 DYNAMIC MAX HOURS based on P&L
            if unrealized_pnl > Decimal("1.0"):
                max_hours = 4.0  # Profitable: let it run longer
            elif unrealized_pnl >= Decimal("-2.0"):
                max_hours = 2.0  # Small loss: cut within 2h
            elif unrealized_pnl >= Decimal("-5.0"):
                max_hours = 1.0  # Medium loss: cut within 1h
            else:
                max_hours = 0.5  # Large loss: cut within 30min

            if hours_open >= max_hours:
                close_reason = f"Time-based exit: {hours_open:.1f}h >= {max_hours}h max"

                # Close with profit or cut loss
                if unrealized_pnl > 0:
                    logger.info(f"⏰ {close_reason} | Taking profit: ${float(unrealized_pnl):+.2f}")
                    await notifier.send_alert(
                        'success',
                        f"⏰ TIME EXIT (PROFIT)\n"
                        f"{symbol} {side}\n"
                        f"Open {hours_open:.1f}h (max {max_hours}h)\n"
                        f"P&L: ${float(unrealized_pnl):+.2f}"
                    )
                else:
                    logger.warning(f"⏰ {close_reason} | Cutting loss: ${float(unrealized_pnl):+.2f}")
                    await notifier.send_alert(
                        'warning',
                        f"⏰ TIME EXIT (LOSS)\n"
                        f"{symbol} {side}\n"
                        f"Open {hours_open:.1f}h (max {max_hours}h)\n"
                        f"P&L: ${float(unrealized_pnl):+.2f}\n"
                        f"Cutting loss and moving on..."
                    )

                await executor.close_position(position, current_price, close_reason)
                return

            # Get market indicators (needed for multiple checks below)
            try:
                ohlcv_15m = await exchange.fetch_ohlcv(symbol, '15m', limit=50)
                indicators = calculate_indicators(ohlcv_15m)
                atr_percent = indicators.get('atr_percent', 2.0)
            except Exception as e:
                logger.warning(f"Failed to fetch indicators: {e}")
                indicators = {}
                atr_percent = 2.0

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
            # 🎯 TIER 1 CHECK #2: TRAILING STOP-LOSS
            # ====================================================================
            # Protects profits by locking in gains as position moves favorably
            try:
                trailing_stop = get_trailing_stop()
                should_exit, new_stop, reason = await trailing_stop.update_trailing_stop(
                    position, current_price, db
                )

                if should_exit:
                    logger.info(f"🎯 Trailing stop TRIGGERED: {reason}")

                    # 🎯 TIER 2: Store exit regime before closing
                    await self._store_exit_regime(position, indicators, symbol)

                    await notifier.send_alert(
                        'success',
                        f"🎯 TRAILING STOP EXIT\n"
                        f"{symbol} {side}\n"
                        f"{reason}\n"
                        f"Profit locked: ${float(unrealized_pnl):+.2f}"
                    )
                    await executor.close_position(position, current_price, reason)
                    return

                if new_stop:
                    logger.info(f"✅ Trailing stop moved to ${float(new_stop):.4f}")
                    # Position already updated in database by trailing_stop module
                    position['stop_loss_price'] = new_stop

            except Exception as trailing_error:
                logger.error(f"❌ Trailing stop check failed: {trailing_error}", exc_info=True)

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

            # 🎯 #9: DYNAMIC TRAILING STOP-LOSS (OLD IMPLEMENTATION - KEPT FOR COMPATIBILITY)
            # Note: TIER 1 Trailing Stop above is the new implementation
            # This is kept as backup/fallback
            try:
                # Check if trailing stop should be updated (old risk_manager method)
                new_stop_price = await risk_manager.update_trailing_stop(
                    position, current_price, atr_percent
                )

                if new_stop_price is not None and not new_stop:
                    # Only use this if TIER 1 trailing stop didn't trigger
                    async with db.pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE active_position SET stop_loss_price = $1 WHERE id = $2",
                            float(new_stop_price), position['id']
                        )
                    logger.debug(f"✅ Fallback trailing stop updated to ${float(new_stop_price):.4f}")
                    position['stop_loss_price'] = new_stop_price

            except Exception as trail_error:
                logger.debug(f"Fallback trailing stop check: {trail_error}")

            # === CRITICAL CHECK 1: Emergency close conditions ===
            should_emergency_close, emergency_reason = await risk_manager.should_emergency_close(position, current_price)

            if should_emergency_close:
                logger.critical(f"🚨 EMERGENCY CLOSE: {emergency_reason}")

                # 🎯 TIER 2: Store exit regime before closing
                await self._store_exit_regime(position, indicators, symbol)

                await notifier.send_alert('critical', f"Emergency closing position: {emergency_reason}")
                await executor.close_position(position, current_price, emergency_reason)
                return

            # === CRITICAL CHECK 2: Liquidation distance ===
            liquidation_price = Decimal(str(position['liquidation_price']))
            distance_to_liq = abs(current_price - liquidation_price) / current_price

            if distance_to_liq < Decimal("0.05"):  # Less than 5%
                logger.critical(f"🚨 LIQUIDATION RISK! Distance: {float(distance_to_liq)*100:.2f}%")

                # 🎯 TIER 2: Store exit regime before closing
                await self._store_exit_regime(position, indicators, symbol)

                await notifier.send_alert(
                    'critical',
                    f"🚨 LIQUIDATION RISK!\n"
                    f"{symbol} {side}\n"
                    f"Distance to liquidation: {float(distance_to_liq)*100:.2f}%\n"
                    f"Closing position immediately!"
                )
                await executor.close_position(position, current_price, "EMERGENCY - Liquidation risk")
                return

            # === CHECK 3: ADVANCED PROFIT TAKING with PARTIAL CLOSES ===
            min_profit_usd = Decimal(str(position['min_profit_target_usd']))

            # Check if we've already taken partial profit
            has_partial_close = position.get('partial_close_executed', False)

            if unrealized_pnl >= min_profit_usd:
                logger.info(f"✅ Minimum profit target reached: ${float(unrealized_pnl):.2f}")

                # STRATEGY 1: Partial close at first profit target (50% off)
                if unrealized_pnl >= min_profit_usd and not has_partial_close:
                    logger.info(f"💰 Taking partial profit: Closing 50% at target")

                    success = await executor.close_position_partial(
                        position,
                        current_price,
                        0.5,  # Close 50%
                        "Partial profit taking - Min target reached"
                    )

                    if success:
                        # Mark that we've done partial close
                        async with db.pool.acquire() as conn:
                            await conn.execute(
                                "UPDATE active_position SET partial_close_executed = TRUE WHERE id = $1",
                                position['id']
                            )
                        # TODO: Move stop-loss to breakeven for remaining 50%
                        # Feature temporarily disabled (method not implemented)
                        # await executor._move_stop_to_breakeven(position)
                        logger.info("🔒 Partial close complete - remaining 50% continues with original stop-loss")

                    # Continue monitoring the remaining 50%
                    return

                # STRATEGY 2: Close remaining 50% if profit is 3x minimum target
                if unrealized_pnl >= (min_profit_usd * 3) and has_partial_close:
                    logger.info(f"🎉 Excellent profit: ${float(unrealized_pnl):.2f} (3x target) - Closing remaining 50%")

                    # 🎯 TIER 2: Store exit regime before closing
                    await self._store_exit_regime(position, indicators, symbol)

                    await executor.close_position(
                        position,
                        current_price,
                        "Take profit - 3x target achieved (remaining 50%)"
                    )
                    return

                # STRATEGY 3: Full close if profit is 2x minimum (and no partial taken yet)
                if unrealized_pnl >= (min_profit_usd * 2) and not has_partial_close:
                    logger.info(f"🎉 Excellent profit: ${float(unrealized_pnl):.2f} (2x target)")

                    # 🎯 TIER 2: Store exit regime before closing
                    await self._store_exit_regime(position, indicators, symbol)

                    await executor.close_position(
                        position,
                        current_price,
                        "Take profit - 2x minimum target achieved"
                    )
                    return

                # STRATEGY 4: Multi-timeframe exit signal (if strong reversal on all TFs)
                tf_exit_signal = await self._check_multi_timeframe_exit(
                    symbol, current_price, side
                )

                if tf_exit_signal:
                    logger.info("⚠️ Multi-timeframe exit signal detected")

                    # 🎯 TIER 2: Store exit regime before closing
                    await self._store_exit_regime(position, indicators, symbol)

                    await executor.close_position(
                        position,
                        current_price,
                        "Multi-timeframe exit signal"
                    )
                    return

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
                    # Emergency exit before hitting -$10 stop-loss
                    elif unrealized_pnl < Decimal("-8.0"):
                        should_exit = True
                        reason = f"Deep loss ${float(unrealized_pnl):.2f} - emergency exit"
                        logger.warning(
                            f"🚨 EMERGENCY EXIT: Loss approaching stop-loss threshold "
                            f"(${float(unrealized_pnl):.2f}), exiting to prevent full -$10 loss"
                        )

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
