"""
Trade Executor - Handles actual trade execution.
Opens positions with proper risk management.

ENHANCED with:
- Retry logic for slippage recovery (5 attempts with exponential backoff)
- Critical failure notifications with manual intervention alerts
- Comprehensive error logging to database
"""

import asyncio
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
from src.config import get_settings
from src.database import get_db_client
from src.exchange_client import get_exchange_client
from src.risk_manager import get_risk_manager
from src.adaptive_risk import get_adaptive_risk_manager
from src.telegram_notifier import get_notifier
from src.utils import (
    setup_logging,
    calculate_liquidation_price,
    calculate_position_size,
    calculate_stop_loss_price,
    calculate_min_profit_price
)

logger = setup_logging()


class TradeExecutor:
    """Handles trade execution with comprehensive safety checks."""

    def __init__(self):
        self.settings = get_settings()

    async def open_position(
        self,
        trade_params: Dict[str, Any],
        ai_analysis: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> bool:
        """
        Open a new leveraged position.

        Args:
            trade_params: Validated trade parameters
            ai_analysis: AI analysis results
            market_data: Current market data

        Returns:
            True if position opened successfully, False otherwise
        """
        symbol = trade_params['symbol']
        side = trade_params['side']
        leverage = trade_params['leverage']
        stop_loss_percent = Decimal(str(trade_params['stop_loss_percent']))
        entry_price = Decimal(str(trade_params['current_price']))

        logger.info(f"Opening position: {symbol} {side} {leverage}x")

        # ========================================================================
        # 🎯 TECHNICAL VALIDATION: DISABLED BY USER REQUEST
        # ========================================================================
        # Technical validation disabled - was blocking all trades due to missing data:
        # - Order flow data consistently 0%
        # - Multi-timeframe data missing (0/0 timeframes)
        # - Volume too low (0.2x-0.3x average)
        # Result: All trades failing at 25% pass rate (need 50%)
        #
        # User decision: Trust AI/ML confidence + market sentiment filter only
        # Protection now relies on:
        # - AI confidence ≥50%
        # - Market sentiment filter (60%+ opposite blocked)
        # - Risk management (stop-loss, leverage limits, position sizing)
        logger.debug(f"⚠️ Technical validation DISABLED - trusting AI/ML confidence only")

        # ========================================================================
        # 🎯 PRIORITY 4: AGGRESSIVE MARKET SENTIMENT FILTERING
        # ========================================================================
        # Block trades with 60%+ opposite market sentiment (unless strong MTF confirmation)
        market_sentiment = market_data.get('market_sentiment', 'NEUTRAL') if market_data else 'NEUTRAL'

        # Extract MTF data for override check
        multi_tf = market_data.get('multi_timeframe', {}) if market_data else {}
        timeframe_agreement = multi_tf.get('timeframe_agreement', {})

        # Count agreeing timeframes
        agreeing_tfs = sum(
            1 for tf_data in timeframe_agreement.values()
            if tf_data.get('trend_direction') == side
        )
        total_tfs = len(timeframe_agreement)
        mtf_agreement_ratio = agreeing_tfs / total_tfs if total_tfs > 0 else 0

        # Check for sentiment conflicts
        sentiment_blocks_trade = False
        block_reason = ""

        if side == 'LONG':
            # Block LONG if market is BEARISH_STRONG (60%+ bearish)
            if market_sentiment in ['BEARISH_STRONG', 'CONTRARIAN_BEARISH']:
                # Allow override only if 75%+ MTF confirmation
                if mtf_agreement_ratio >= 0.75:
                    logger.warning(
                        f"⚠️ SENTIMENT OVERRIDE: {symbol} LONG in {market_sentiment} market, "
                        f"but {agreeing_tfs}/{total_tfs} timeframes agree ({mtf_agreement_ratio:.0%}) - ALLOWING"
                    )
                else:
                    sentiment_blocks_trade = True
                    block_reason = (
                        f"Market sentiment {market_sentiment} conflicts with LONG trade "
                        f"(only {mtf_agreement_ratio:.0%} MTF agreement, need ≥75%)"
                    )

        elif side == 'SHORT':
            # Block SHORT if market is BULLISH_STRONG (60%+ bullish)
            if market_sentiment in ['BULLISH_STRONG', 'CONTRARIAN_BULLISH']:
                # Allow override only if 75%+ MTF confirmation
                if mtf_agreement_ratio >= 0.75:
                    logger.warning(
                        f"⚠️ SENTIMENT OVERRIDE: {symbol} SHORT in {market_sentiment} market, "
                        f"but {agreeing_tfs}/{total_tfs} timeframes agree ({mtf_agreement_ratio:.0%}) - ALLOWING"
                    )
                else:
                    sentiment_blocks_trade = True
                    block_reason = (
                        f"Market sentiment {market_sentiment} conflicts with SHORT trade "
                        f"(only {mtf_agreement_ratio:.0%} MTF agreement, need ≥75%)"
                    )

        if sentiment_blocks_trade:
            logger.error(
                f"❌ MARKET SENTIMENT FILTER BLOCKED: {symbol} {side}\n"
                f"   Market: {market_sentiment}\n"
                f"   MTF Agreement: {agreeing_tfs}/{total_tfs} = {mtf_agreement_ratio:.0%}\n"
                f"   Reason: {block_reason}"
            )
            return False

        logger.info(
            f"✅ Market sentiment check PASSED: {market_sentiment} compatible with {side} "
            f"(MTF: {mtf_agreement_ratio:.0%})"
        )

        # 🔍 DEBUG: Check if price_action data is present
        pa_data = ai_analysis.get('price_action')
        if pa_data:
            logger.info(f"✅ Price action data present: validated={pa_data.get('validated')}, reason={pa_data.get('reason')}")
        else:
            logger.info("⚠️ No price action data in ai_analysis")

        try:
            db = await get_db_client()
            exchange = await get_exchange_client()
            notifier = get_notifier()
            adaptive_risk = get_adaptive_risk_manager()

            # Get current capital
            current_capital = await db.get_current_capital()

            # 🎯 ADAPTIVE STOP-LOSS: Adjust based on recent performance
            adaptive_sl_percent = await adaptive_risk.get_adaptive_stop_loss_percent(
                symbol, side, base_stop_loss=float(stop_loss_percent)
            )
            stop_loss_percent = Decimal(str(adaptive_sl_percent))
            logger.info(f"📊 Using adaptive SL: {adaptive_sl_percent:.1f}%")

            # 🎯 TIME-BASED RISK: Adjust position size based on hour performance
            time_multiplier = await adaptive_risk.get_time_based_risk_multiplier()

            # DYNAMIC position size based on capital (USER REQUEST: $100 margin per position @ 25x leverage)
            # 🔧 Auto-adjust to capital: position = (capital / 3 positions) × 25x leverage
            #
            # Example with $300 capital:
            # - $300 ÷ 3 positions = $100 margin per position
            # - $100 margin × 25x leverage = $2,500 position size
            #
            # Example with $105 capital (current):
            # - $105 ÷ 3 positions = $35 margin per position
            # - $35 margin × 25x leverage = $875 position size
            #
            # Target: ±$1 profit/loss with 25-30x leverage
            max_positions = 3  # User configured
            margin_per_position = current_capital / Decimal(str(max_positions))
            FIXED_POSITION_SIZE_USD = margin_per_position * Decimal(str(leverage)) * Decimal(str(time_multiplier))

            max_positions_allowed = max_positions
            logger.info(
                f"💰 Capital: ${current_capital:.2f} → {max_positions} positions × ${margin_per_position:.2f} margin "
                f"= ${FIXED_POSITION_SIZE_USD:.2f} position size (leverage: {leverage}x, time mult: {time_multiplier}x)"
            )

            quantity, position_value = calculate_position_size(
                current_capital,
                self.settings.position_size_percent,
                entry_price,
                leverage,
                fixed_position_usd=FIXED_POSITION_SIZE_USD
            )

            # Calculate risk management prices (WITH LEVERAGE ADJUSTMENT!)
            stop_loss_price = calculate_stop_loss_price(
                entry_price,
                stop_loss_percent,
                side,
                leverage=leverage,
                position_value=position_value
            )
            liquidation_price = calculate_liquidation_price(entry_price, leverage, side)
            min_profit_price = calculate_min_profit_price(
                entry_price,
                self.settings.min_profit_usd,
                position_value,
                leverage,
                side
            )

            # 🎯 PROFIT TARGETS: Calculate scaled exit targets for partial exit strategy
            from src.utils import calculate_profit_targets
            profit_targets = calculate_profit_targets(
                entry_price=entry_price,
                side=side,
                position_value=position_value,
                leverage=leverage,
                market_data=market_data
            )

            profit_target_1 = profit_targets['profit_target_1']
            profit_target_2 = profit_targets['profit_target_2']
            target_1_profit = profit_targets['target_1_profit_usd']
            target_2_profit = profit_targets['target_2_profit_usd']

            logger.info(f"Position details:")
            logger.info(f"  Quantity: {quantity:.6f}")
            logger.info(f"  Position Value: ${position_value:.2f}")
            logger.info(f"  Entry Price: ${entry_price:.4f}")
            logger.info(f"  Stop-Loss: ${stop_loss_price:.4f}")
            logger.info(f"  Liquidation: ${liquidation_price:.4f}")
            logger.info(f"  Min Profit Price: ${min_profit_price:.4f}")
            logger.info(f"🎯 Profit Target 1 (50%): ${float(profit_target_1):.4f} (Est. profit: ${float(target_1_profit):.2f})")
            logger.info(f"🎯 Profit Target 2 (50%): ${float(profit_target_2):.4f} (Est. total: ${float(target_2_profit):.2f})")

            # Set leverage and margin mode on exchange
            await exchange.set_leverage(symbol, leverage)
            await exchange.set_margin_mode(symbol, 'isolated')

            # 📊 EXECUTION QUALITY MONITORING: Track fill time and slippage
            order_start_time = datetime.now()
            order_side = 'buy' if side == 'LONG' else 'sell'

            # Execute entry order
            entry_order = await exchange.create_market_order(symbol, order_side, quantity)

            # Calculate fill time
            order_fill_time = datetime.now()
            fill_time_ms = int((order_fill_time - order_start_time).total_seconds() * 1000)

            actual_entry_price = Decimal(str(entry_order.get('average', entry_price)))
            exchange_order_id = entry_order.get('id')

            # CRITICAL: Check slippage
            slippage_percent = abs((actual_entry_price - entry_price) / entry_price) * 100
            max_slippage = Decimal("0.5")  # 0.5% maximum acceptable slippage

            # Log execution quality metrics
            logger.info(f"📊 Execution Quality: Fill time: {fill_time_ms}ms, Slippage: {float(slippage_percent):.3f}%")

            if slippage_percent > max_slippage:
                logger.error(
                    f"❌ EXCESSIVE SLIPPAGE: {float(slippage_percent):.3f}% "
                    f"(max: {float(max_slippage):.3f}%)"
                )

                # ENHANCED: Retry-based emergency close with exponential backoff
                close_side = 'sell' if side == 'LONG' else 'buy'
                max_close_attempts = 5
                close_success = False

                for attempt in range(max_close_attempts):
                    try:
                        logger.warning(f"Emergency close attempt {attempt + 1}/{max_close_attempts}")
                        await exchange.create_market_order(symbol, close_side, quantity)
                        logger.info(f"✅ Position closed successfully (attempt {attempt + 1})")
                        close_success = True
                        break
                    except Exception as close_err:
                        logger.error(f"Close attempt {attempt + 1} failed: {close_err}")
                        if attempt < max_close_attempts - 1:
                            # Exponential backoff
                            backoff_time = 2 ** attempt  # 1s, 2s, 4s, 8s
                            logger.info(f"Retrying in {backoff_time}s...")
                            await asyncio.sleep(backoff_time)

                if close_success:
                    await notifier.send_alert(
                        'warning',
                        f"⚠️ Trade cancelled due to excessive slippage:\n"
                        f"{symbol} {side}\n"
                        f"Expected: ${float(entry_price):.4f}\n"
                        f"Executed: ${float(actual_entry_price):.4f}\n"
                        f"Slippage: {float(slippage_percent):.3f}%\n\n"
                        f"✅ Position closed successfully after slippage detection."
                    )
                else:
                    # CRITICAL: All close attempts failed!
                    logger.critical(
                        f"🚨🚨🚨 CRITICAL: Failed to close position after {max_close_attempts} attempts!"
                    )
                    await notifier.send_alert(
                        'critical',
                        f"🚨🚨🚨 EMERGENCY - MANUAL INTERVENTION REQUIRED!\n\n"
                        f"Position {symbol} {side} could NOT be closed after excessive slippage.\n"
                        f"Excessive slippage: {float(slippage_percent):.3f}%\n"
                        f"Entry price: ${float(actual_entry_price):.4f}\n\n"
                        f"⚠️ CLOSE THIS POSITION MANUALLY IMMEDIATELY!\n"
                        f"Quantity: {float(quantity):.6f}\n"
                        f"Side: {close_side.upper()}"
                    )
                    # Mark in database for manual intervention
                    await db.log_event(
                        'CRITICAL',
                        'TradeExecutor',
                        'Emergency close failed after slippage',
                        {
                            'symbol': symbol,
                            'side': side,
                            'quantity': float(quantity),
                            'slippage_percent': float(slippage_percent),
                            'requires_manual_close': True
                        }
                    )
                return False

            logger.info(
                f"✅ Entry order executed: {exchange_order_id} @ ${float(actual_entry_price):.4f} "
                f"(slippage: {float(slippage_percent):.3f}%)"
            )

            # CRITICAL: Place stop-loss order (MANDATORY)
            stop_loss_order_id = None
            sl_order_side = 'sell' if side == 'LONG' else 'buy'

            try:
                sl_order = await exchange.create_stop_loss_order(
                    symbol,
                    sl_order_side,
                    quantity,
                    stop_loss_price
                )
                stop_loss_order_id = sl_order.get('id')
                logger.info(f"✅ Stop-loss order placed: {stop_loss_order_id}")

                # 🎯 ADAPTIVE TAKE-PROFIT: Place TP order for $2.0 profit target
                take_profit_price = await adaptive_risk.get_adaptive_take_profit(
                    symbol, side, entry_price, leverage, position_value
                )

                if take_profit_price:
                    tp_order_side = 'sell' if side == 'LONG' else 'buy'
                    try:
                        tp_order = await exchange.create_take_profit_order(
                            symbol,
                            tp_order_side,
                            quantity,
                            take_profit_price
                        )
                        logger.info(f"🎯 Take-profit order placed: ${float(take_profit_price):.4f}")
                    except Exception as tp_error:
                        # TP placement failed - not critical, position still protected by SL
                        logger.warning(f"Failed to place take-profit order: {tp_error}")

            except Exception as e:
                # CRITICAL: Stop-loss placement failed - close position immediately
                logger.critical(f"🚨 STOP-LOSS PLACEMENT FAILED: {e}")

                try:
                    # Close the position immediately
                    close_order = await exchange.create_market_order(symbol, sl_order_side, quantity)
                    logger.warning("Position closed immediately due to stop-loss placement failure")

                    await notifier.send_alert(
                        'critical',
                        f"🚨 CRITICAL: Position closed immediately\n\n"
                        f"Reason: Stop-loss order placement failed\n"
                        f"{symbol} {side} - Entry: ${float(actual_entry_price):.4f}\n\n"
                        f"Error: {str(e)[:200]}\n\n"
                        f"Position was closed at market to prevent unprotected exposure."
                    )

                except Exception as close_err:
                    logger.critical(f"FAILED TO CLOSE POSITION: {close_err}")
                    await notifier.send_alert(
                        'critical',
                        f"🚨🚨🚨 EMERGENCY: MANUAL INTERVENTION REQUIRED\n\n"
                        f"{symbol} {side} position is OPEN WITHOUT STOP-LOSS\n"
                        f"Entry: ${float(actual_entry_price):.4f}\n\n"
                        f"Please close this position manually IMMEDIATELY!"
                    )

                return False

            # 📸 CAPTURE ENTRY SNAPSHOT FOR ML LEARNING
            entry_snapshot = None
            try:
                from src.snapshot_capture import capture_market_snapshot

                # Extract indicators from multi-timeframe format
                # market_data['indicators'] = {'15m': {...}, '1h': {...}, '4h': {...}}
                # We use 1h timeframe as primary for snapshot
                indicators_dict = market_data.get('indicators', {})
                if isinstance(indicators_dict, dict) and '1h' in indicators_dict:
                    # Multi-timeframe format - use 1h
                    flat_indicators = indicators_dict['1h']
                elif isinstance(indicators_dict, dict):
                    # Already flat format
                    flat_indicators = indicators_dict
                else:
                    flat_indicators = {}

                # Add additional market data
                flat_indicators['volume'] = market_data.get('volume_24h', 0)
                flat_indicators['volume_24h_avg'] = market_data.get('volume_24h', 0) / 24 if market_data.get('volume_24h', 0) > 0 else 0
                flat_indicators['support_levels'] = [s['price'] for s in market_data.get('support_resistance', {}).get('support', [])]
                flat_indicators['resistance_levels'] = [r['price'] for r in market_data.get('support_resistance', {}).get('resistance', [])]

                entry_snapshot = await capture_market_snapshot(
                    exchange_client=exchange,
                    symbol=symbol,
                    current_price=actual_entry_price,
                    indicators=flat_indicators,
                    side=side,
                    snapshot_type="entry"
                )

                # 🎯 ADD PRICE ACTION DATA TO SNAPSHOT (for ML learning)
                price_action_data = ai_analysis.get('price_action')
                if price_action_data:
                    entry_snapshot['price_action'] = price_action_data
                    logger.info(f"📸 Entry snapshot captured for ML learning with PA data")
                else:
                    logger.info(f"📸 Entry snapshot captured for ML learning (ML-only)")
            except Exception as snapshot_error:
                logger.warning(f"⚠️ Entry snapshot capture failed (non-critical): {snapshot_error}")
                entry_snapshot = None

            # Record position in database
            position_data = {
                'symbol': symbol,
                'side': side,
                'leverage': leverage,
                'entry_price': actual_entry_price,
                'current_price': actual_entry_price,
                'quantity': quantity,
                'position_value_usd': position_value,
                'stop_loss_price': stop_loss_price,
                'stop_loss_percent': Decimal(str(stop_loss_percent)),
                'min_profit_target_usd': self.settings.min_profit_usd,
                'min_profit_price': min_profit_price,
                'liquidation_price': liquidation_price,
                'exchange_order_id': exchange_order_id,
                'stop_loss_order_id': stop_loss_order_id,
                'ai_model_consensus': '+'.join(ai_analysis.get('models_used', [])),
                'ai_confidence': Decimal(str(ai_analysis.get('confidence', 0))),
                'ai_reasoning': ai_analysis.get('reasoning', ''),

                # 📊 EXECUTION QUALITY METRICS (NEW!)
                'slippage_percent': slippage_percent,
                'fill_time_ms': fill_time_ms,
                'expected_entry_price': entry_price,
                'order_start_timestamp': order_start_time,
                'order_fill_timestamp': order_fill_time,

                # 📸 ML LEARNING SNAPSHOT
                'entry_snapshot': entry_snapshot,
                'entry_slippage_percent': slippage_percent,
                'entry_fill_time_ms': fill_time_ms,

                # 🎯 PRICE ACTION ANALYSIS (NEW!)
                'price_action': ai_analysis.get('price_action'),

                # 🎯 PARTIAL EXIT TARGETS (NEW!)
                'profit_target_1': profit_target_1,
                'profit_target_2': profit_target_2,
                'partial_exit_done': False,
                'partial_exit_profit': Decimal("0")
            }

            position_id = await db.create_active_position(position_data)
            logger.info(f"✅ Position recorded in database: ID {position_id}")

            # Send Telegram notification
            await notifier.send_position_opened(position_data)

            logger.info(f"🎉 Position opened successfully: {symbol} {side} {leverage}x")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to open position: {e}")
            await notifier.send_error_report('TradeExecutor', str(e))
            return False

    async def close_position_partial(
        self,
        position: Dict[str, Any],
        current_price: Decimal,
        close_percent: float,
        close_reason: str
    ) -> bool:
        """
        Close a portion of the position (PARTIAL PROFIT TAKING).

        Args:
            position: Active position data
            current_price: Current market price
            close_percent: Percentage to close (0.5 = 50%)
            close_reason: Reason for partial close

        Returns:
            True if closed successfully, False otherwise
        """
        symbol = position['symbol']
        side = position['side']
        full_quantity = Decimal(str(position['quantity']))
        close_quantity = full_quantity * Decimal(str(close_percent))

        logger.info(
            f"Partial close ({close_percent*100:.0f}%): {symbol} {side} - "
            f"Closing {close_quantity:.6f}/{full_quantity:.6f} - Reason: {close_reason}"
        )

        try:
            db = await get_db_client()
            exchange = await get_exchange_client()
            notifier = get_notifier()

            # Execute partial close order
            close_order = await exchange.close_position(symbol, side, close_quantity)
            exit_price = Decimal(str(close_order.get('average', current_price)))

            logger.info(f"✅ Partial position closed at ${exit_price:.4f}")

            # Calculate P&L for closed portion WITH FEES
            entry_price = Decimal(str(position['entry_price']))
            leverage = position['leverage']
            position_value = Decimal(str(position['position_value_usd'])) * Decimal(str(close_percent))

            # Gross PnL (before fees)
            if side == 'LONG':
                price_change_pct = (exit_price - entry_price) / entry_price
            else:  # SHORT
                price_change_pct = (entry_price - exit_price) / entry_price

            gross_pnl = position_value * price_change_pct * leverage

            # Calculate fees for this partial close
            taker_fee_rate = Decimal("0.0005")  # 0.05%

            # Entry fee (proportional to closed amount)
            entry_notional = close_quantity * entry_price
            entry_fee = entry_notional * taker_fee_rate

            # Exit fee
            exit_notional = close_quantity * exit_price
            exit_fee = exit_notional * taker_fee_rate

            total_fees = entry_fee + exit_fee

            # NET PnL (after fees)
            partial_pnl = gross_pnl - total_fees

            # Update capital
            new_capital = (await db.get_current_capital()) + partial_pnl
            await db.update_capital(new_capital)

            # Update paper trading balance if applicable
            if exchange.paper_trading:
                exchange.update_paper_balance(partial_pnl)

            # Update position in database (reduce quantity)
            remaining_quantity = full_quantity - close_quantity
            async with db.pool.acquire() as conn:
                await conn.execute(
                    "UPDATE active_position SET quantity = $1 WHERE id = $2",
                    remaining_quantity, position['id']
                )

            # Send notification with fee breakdown
            await notifier.send_alert(
                'success',
                f"✅ <b>SUCCESS</b>\n\n"
                f"💰 <b>Partial Close ({close_percent*100:.0f}%)</b>\n\n"
                f"<b>{symbol}</b> {side} {leverage}x\n"
                f"Exit: <b>${float(exit_price):.4f}</b>\n\n"
                f"<b>P&L Breakdown:</b>\n"
                f"├ Gross: ${float(gross_pnl):+.2f}\n"
                f"├ Fees: -${float(total_fees):.2f}\n"
                f"└ <b>Net: ${float(partial_pnl):+.2f}</b>\n\n"
                f"Reason: {close_reason}\n"
                f"Remaining: {(1-close_percent)*100:.0f}% still open"
            )

            logger.info(f"💰 Partial close: P&L ${partial_pnl:+.2f}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to partially close position: {e}")
            await notifier.send_error_report('TradeExecutor', f"Failed to partial close: {e}")
            return False

    async def close_position(
        self,
        position: Dict[str, Any],
        current_price: Decimal,
        close_reason: str
    ) -> bool:
        """
        Close an open position.

        Args:
            position: Active position data
            current_price: Current market price
            close_reason: Reason for closing

        Returns:
            True if closed successfully, False otherwise
        """
        symbol = position['symbol']
        side = position['side']
        quantity = Decimal(str(position['quantity']))

        logger.info(f"Closing position: {symbol} {side} - Reason: {close_reason}")

        try:
            db = await get_db_client()
            exchange = await get_exchange_client()
            notifier = get_notifier()

            # 🔧 CRITICAL FIX: Cancel ALL open orders for this symbol
            # Previously only cancelled stop-loss, leaving limit/TP orders open
            # This caused old orders to immediately close new positions!
            try:
                # Access CCXT exchange object directly (exchange.exchange)
                open_orders = await exchange.exchange.fetch_open_orders(symbol)
                if open_orders:
                    logger.info(f"🗑️ Cancelling {len(open_orders)} open orders for {symbol}")
                    for order in open_orders:
                        try:
                            await exchange.cancel_order(order['id'], symbol)
                            logger.info(f"  ✅ Cancelled order {order['id']} ({order['type']} {order['side']})")
                        except Exception as cancel_err:
                            logger.warning(f"  ⚠️ Could not cancel order {order['id']}: {cancel_err}")
                else:
                    logger.info(f"No open orders to cancel for {symbol}")
            except Exception as e:
                logger.warning(f"Could not fetch/cancel open orders: {e}")

                # Fallback: Try to cancel known order IDs from position record
                if position.get('stop_loss_order_id'):
                    try:
                        await exchange.cancel_order(position['stop_loss_order_id'], symbol)
                        logger.info("Stop-loss order cancelled (fallback)")
                    except Exception as e2:
                        logger.warning(f"Fallback SL cancel failed: {e2}")

            # 📊 EXECUTION QUALITY MONITORING: Track exit fill time and slippage
            exit_start_time = datetime.now()

            # Execute close order
            close_order = await exchange.close_position(symbol, side, quantity)

            # Calculate exit fill time
            exit_fill_time = datetime.now()
            exit_fill_time_ms = int((exit_fill_time - exit_start_time).total_seconds() * 1000)

            exit_price = Decimal(str(close_order.get('average', current_price)))

            # Calculate exit slippage
            exit_slippage_percent = abs((exit_price - current_price) / current_price) * 100

            logger.info(f"✅ Position closed at ${exit_price:.4f}")
            logger.info(f"📊 Exit Execution Quality: Fill time: {exit_fill_time_ms}ms, Slippage: {float(exit_slippage_percent):.3f}%")

            # Calculate final P&L
            entry_price = Decimal(str(position['entry_price']))
            leverage = position['leverage']
            position_value = Decimal(str(position['position_value_usd']))

            if side == 'LONG':
                price_change_pct = (exit_price - entry_price) / entry_price
            else:  # SHORT
                price_change_pct = (entry_price - exit_price) / entry_price

            realized_pnl = position_value * price_change_pct * leverage
            pnl_percent = float(price_change_pct * leverage * 100)

            # Update capital
            new_capital = (await db.get_current_capital()) + realized_pnl
            await db.update_capital(new_capital)

            # Update paper trading balance if applicable
            if exchange.paper_trading:
                exchange.update_paper_balance(realized_pnl)

            # 📸 CAPTURE EXIT SNAPSHOT FOR ML LEARNING
            exit_snapshot = None
            try:
                from src.snapshot_capture import capture_market_snapshot
                # Fetch current market data for exit snapshot
                exit_market_data = await exchange.fetch_ticker(symbol)
                exit_indicators = {}  # Will be populated by market analysis if available

                # Try to get fresh indicators from market scanner
                try:
                    from src.market_scanner import MarketScanner
                    scanner = MarketScanner()
                    analysis = await scanner._analyze_symbol(symbol)
                    exit_indicators = analysis.get('indicators', {}) if analysis else {}
                except:
                    # Fallback: use basic ticker data
                    exit_indicators = {
                        'close': float(exit_price),
                        'volume': exit_market_data.get('quoteVolume', 0),
                        'price_change_24h_percent': exit_market_data.get('percentage', 0)
                    }

                exit_snapshot = await capture_market_snapshot(
                    exchange_client=exchange,
                    symbol=symbol,
                    current_price=exit_price,
                    indicators=exit_indicators,
                    side=side,
                    snapshot_type="exit"
                )
                logger.info(f"📸 Exit snapshot captured for ML learning")
            except Exception as snapshot_error:
                logger.warning(f"⚠️ Exit snapshot capture failed (non-critical): {snapshot_error}")
                exit_snapshot = None

            # Record trade in history
            trade_duration = (datetime.now() - position.get('entry_time', datetime.now())).total_seconds()

            # 🎯 TIER 1 & 2: Extract metadata for performance tracking
            max_profit_achieved = position.get('max_profit_percent', 0.0)
            trailing_stop_triggered = 'trailing stop' in close_reason.lower()
            partial_exits_completed = position.get('partial_exits_completed', [])
            had_partial_exits = len(partial_exits_completed) > 0 if partial_exits_completed else False

            # Build partial exit details JSON
            partial_exit_details = None
            if had_partial_exits:
                partial_exit_details = {
                    'tiers_executed': partial_exits_completed,
                    'max_profit_before_exit': max_profit_achieved
                }

            trade_data = {
                'symbol': symbol,
                'side': side,
                'leverage': leverage,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'position_value_usd': position_value,
                'realized_pnl_usd': realized_pnl,
                'pnl_percent': Decimal(str(pnl_percent)),
                'stop_loss_percent': position.get('stop_loss_percent'),
                'close_reason': close_reason,
                'trade_duration_seconds': int(trade_duration),
                'ai_model_consensus': position.get('ai_model_consensus'),
                'ai_confidence': position.get('ai_confidence'),
                'ai_reasoning': position.get('ai_reasoning', ''),
                'entry_time': position.get('entry_time', datetime.now()),

                # 📊 EXECUTION QUALITY METRICS
                'entry_slippage_percent': position.get('entry_slippage_percent', 0),
                'entry_fill_time_ms': position.get('entry_fill_time_ms', 0),
                'exit_slippage_percent': exit_slippage_percent,
                'exit_fill_time_ms': exit_fill_time_ms,
                'avg_slippage_percent': (position.get('entry_slippage_percent', 0) + exit_slippage_percent) / 2,
                'total_fill_time_ms': position.get('entry_fill_time_ms', 0) + exit_fill_time_ms,

                # 📸 ML LEARNING SNAPSHOTS
                'entry_snapshot': position.get('entry_snapshot'),
                'exit_snapshot': exit_snapshot,

                # 🎯 TIER 1: Trailing Stop-Loss Metrics
                'max_profit_percent_achieved': max_profit_achieved,
                'trailing_stop_triggered': trailing_stop_triggered,

                # 🎯 TIER 1: Partial Exit Tracking
                'had_partial_exits': had_partial_exits,
                'partial_exit_details': partial_exit_details,

                # 🎯 TIER 2: Market Regime Tracking
                'entry_market_regime': position.get('entry_market_regime'),
                'exit_market_regime': position.get('exit_market_regime')  # Set by position_monitor
            }

            await db.record_trade(trade_data)

            # 🧠 ML LEARNING: Teach ML system from completed trade
            try:
                from src.ml_pattern_learner import get_ml_learner
                ml_learner = await get_ml_learner()
                await ml_learner.on_trade_completed(trade_data)
                logger.info(f"🎓 ML learned from trade: {symbol} ({trade_data['close_reason']})")
            except Exception as ml_error:
                logger.warning(f"ML learning failed: {ml_error} (trade still recorded)")

            # 🎯 EXIT OPTIMIZER LEARNING: Train exit optimizer from this trade
            try:
                from src.exit_optimizer import get_exit_optimizer
                exit_optimizer = get_exit_optimizer()

                # Extract exit features at close time
                exit_features = exit_optimizer.extract_exit_features(
                    position, exit_price, {'indicators': {}}  # Empty market data for now
                )

                # Calculate exit quality
                # Use trade_duration that was already calculated above
                time_in_position_minutes = trade_duration / 60  # Convert seconds to minutes
                exit_quality = exit_optimizer.calculate_exit_quality(
                    position,
                    realized_pnl,
                    self.settings.min_profit_usd,
                    int(time_in_position_minutes)
                )

                # Determine if exit was optimal (profit > 0 or loss < -$3)
                was_optimal = realized_pnl > 0 or (realized_pnl < 0 and realized_pnl > Decimal("-3.0"))

                # Train exit optimizer
                exit_optimizer.learn_from_exit(exit_features, exit_quality, was_optimal)
                logger.info(f"🎯 Exit Optimizer learned: Quality={exit_quality:.2f}, Optimal={was_optimal}")
            except Exception as exit_error:
                logger.warning(f"Exit optimizer learning failed: {exit_error}")

            # 🎯 TIER 2: ML CONTINUOUS LEARNING - Auto-retrain every 50 trades
            try:
                from src.ml_continuous_learner import get_continuous_learner
                continuous_learner = get_continuous_learner()

                # Check if retraining is needed and perform if so
                await continuous_learner.on_trade_completed(trade_data, db)

                logger.debug(f"🎓 Continuous learning check completed")
            except Exception as ml_train_error:
                logger.warning(f"ML continuous learning failed: {ml_train_error}")

            # 🎯 TRADE QUALITY MANAGER: Record trade outcome
            try:
                from src.trade_quality_manager import get_trade_quality_manager
                quality_mgr = get_trade_quality_manager()

                # Extract ML exit confidence if present in close reason
                exit_confidence = 0.0
                if 'ML' in close_reason or 'conf' in close_reason:
                    # Try to extract confidence percentage from reason
                    import re
                    conf_match = re.search(r'(\d+)%', close_reason)
                    if conf_match:
                        exit_confidence = float(conf_match.group(1))

                was_profitable = realized_pnl > 0

                quality_mgr.record_trade_closed(
                    symbol=symbol,
                    was_profitable=was_profitable,
                    exit_confidence=exit_confidence,
                    pnl_usd=float(realized_pnl),
                    exit_reason=close_reason
                )

                logger.info(f"📊 Trade outcome recorded in quality manager")
            except Exception as quality_error:
                logger.warning(f"Quality manager update failed: {quality_error}")

            # Remove active position
            await db.remove_active_position(position['id'])

            # Update daily performance
            from datetime import date
            await db.update_daily_performance(date.today())

            # Send Telegram notification
            await notifier.send_position_closed(position, exit_price, realized_pnl, close_reason)

            # Send updated portfolio
            await notifier.send_portfolio_update(new_capital, await db.get_daily_pnl())

            pnl_emoji = "🎉" if realized_pnl > 0 else "😔"
            logger.info(f"{pnl_emoji} Position closed: P&L ${realized_pnl:+.2f} ({pnl_percent:+.2f}%)")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to close position: {e}")
            await notifier.send_error_report('TradeExecutor', f"Failed to close position: {e}")
            return False


# Singleton instance
_trade_executor: Optional[TradeExecutor] = None


def get_trade_executor() -> TradeExecutor:
    """Get or create trade executor instance."""
    global _trade_executor
    if _trade_executor is None:
        _trade_executor = TradeExecutor()
    return _trade_executor
