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

        try:
            db = await get_db_client()
            exchange = await get_exchange_client()
            notifier = get_notifier()

            # Get current capital
            current_capital = await db.get_current_capital()

            # Calculate position size
            quantity, position_value = calculate_position_size(
                current_capital,
                self.settings.position_size_percent,
                entry_price,
                leverage
            )

            # Calculate risk management prices
            stop_loss_price = calculate_stop_loss_price(entry_price, stop_loss_percent, side)
            liquidation_price = calculate_liquidation_price(entry_price, leverage, side)
            min_profit_price = calculate_min_profit_price(
                entry_price,
                self.settings.min_profit_usd,
                position_value,
                leverage,
                side
            )

            logger.info(f"Position details:")
            logger.info(f"  Quantity: {quantity:.6f}")
            logger.info(f"  Position Value: ${position_value:.2f}")
            logger.info(f"  Entry Price: ${entry_price:.4f}")
            logger.info(f"  Stop-Loss: ${stop_loss_price:.4f}")
            logger.info(f"  Liquidation: ${liquidation_price:.4f}")
            logger.info(f"  Min Profit Price: ${min_profit_price:.4f}")

            # Set leverage and margin mode on exchange
            await exchange.set_leverage(symbol, leverage)
            await exchange.set_margin_mode(symbol, 'isolated')

            # Execute entry order
            order_side = 'buy' if side == 'LONG' else 'sell'
            entry_order = await exchange.create_market_order(symbol, order_side, quantity)

            actual_entry_price = Decimal(str(entry_order.get('average', entry_price)))
            exchange_order_id = entry_order.get('id')

            # CRITICAL: Check slippage
            slippage_percent = abs((actual_entry_price - entry_price) / entry_price) * 100
            max_slippage = Decimal("0.5")  # 0.5% maximum acceptable slippage

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
                'ai_confidence': Decimal(str(ai_analysis.get('confidence', 0)))
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

            # Calculate P&L for closed portion
            entry_price = Decimal(str(position['entry_price']))
            leverage = position['leverage']
            position_value = Decimal(str(position['position_value_usd'])) * Decimal(str(close_percent))

            if side == 'LONG':
                price_change_pct = (exit_price - entry_price) / entry_price
            else:  # SHORT
                price_change_pct = (entry_price - exit_price) / entry_price

            partial_pnl = position_value * price_change_pct * leverage

            # Update capital
            new_capital = (await db.get_current_capital()) + partial_pnl
            await db.update_capital(new_capital)

            # Update paper trading balance if applicable
            if exchange.paper_trading:
                exchange.update_paper_balance(partial_pnl)

            # Update position in database (reduce quantity)
            remaining_quantity = full_quantity - close_quantity
            await db.execute(
                "UPDATE active_position SET quantity = $1 WHERE id = $2",
                remaining_quantity, position['id']
            )

            # Send notification
            await notifier.send_alert(
                'success',
                f"💰 Partial Close ({close_percent*100:.0f}%)\n\n"
                f"{symbol} {side} {leverage}x\n"
                f"Exit: ${float(exit_price):.4f}\n"
                f"P&L: ${float(partial_pnl):+.2f}\n\n"
                f"Reason: {close_reason}\n"
                f"Remaining: {close_percent*100:.0f}% still open"
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

            # Cancel stop-loss order if exists
            if position.get('stop_loss_order_id'):
                try:
                    await exchange.cancel_order(position['stop_loss_order_id'], symbol)
                    logger.info("Stop-loss order cancelled")
                except Exception as e:
                    logger.warning(f"Could not cancel stop-loss order: {e}")

            # Execute close order
            close_order = await exchange.close_position(symbol, side, quantity)
            exit_price = Decimal(str(close_order.get('average', current_price)))

            logger.info(f"✅ Position closed at ${exit_price:.4f}")

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

            # Record trade in history
            trade_duration = (datetime.now() - position.get('entry_time', datetime.now())).total_seconds()

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
                'entry_time': position.get('entry_time', datetime.now())
            }

            await db.record_trade(trade_data)

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
