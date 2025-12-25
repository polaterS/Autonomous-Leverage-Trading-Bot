"""
Telegram notification system for the trading bot.
Sends all real-time updates to the user via Telegram.
"""

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import html
from src.config import get_settings
from src.utils import setup_logging, format_duration

logger = setup_logging()

# Turkey timezone (UTC+3)
TURKEY_TZ = timezone(timedelta(hours=3))


def get_turkey_time() -> datetime:
    """Get current time in Turkey timezone (UTC+3)."""
    return datetime.now(TURKEY_TZ)


class TelegramNotifier:
    """Telegram bot for sending trading notifications."""

    def __init__(self):
        self.settings = get_settings()
        self.bot = Bot(token=self.settings.telegram_bot_token)
        self.chat_id = self.settings.telegram_chat_id

    async def send_message(self, text: str, parse_mode: str = ParseMode.HTML) -> bool:
        """
        Send a message to Telegram.

        Returns:
            True if successful, False otherwise
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode
            )
            return True
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def send_startup_message(self) -> None:
        """Bot started notification."""
        mode = "PAPER TRADING" if self.settings.use_paper_trading else "LIVE TRADING"

        message = f"""
🤖 <b>AUTONOMOUS TRADING BOT STARTED</b>

<b>Mode:</b> {mode}

The bot is now running in fully autonomous mode.
You will receive notifications for all trading activity.

✅ Auto-scanning for opportunities
✅ AI-powered trade decisions
✅ Automatic position management
✅ Strict risk management enabled

<b>Configuration:</b>
💰 Initial Capital: ${self.settings.initial_capital}
⚡ Max Leverage: {self.settings.max_leverage}x
🛑 Stop-Loss Range: {float(self.settings.min_stop_loss_percent)}% - {float(self.settings.max_stop_loss_percent)}%
💎 Min Profit: ${self.settings.min_profit_usd}
🎯 Min AI Confidence: {float(self.settings.min_ai_confidence)*100}%

Sit back and monitor your portfolio! 💰

⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await self.send_message(message)

    async def send_position_opened(self, position: Dict[str, Any]) -> None:
        """New position opened notification with accurate stop-loss and fees."""
        from decimal import Decimal

        emoji = "🟢" if position['side'] == 'LONG' else "🔴"

        # Calculate ACCURATE stop-loss percentage (leverage-adjusted)
        entry_price = Decimal(str(position['entry_price']))
        stop_loss_price = Decimal(str(position['stop_loss_price']))
        leverage = int(position['leverage'])
        position_value = Decimal(str(position['position_value_usd']))

        # Calculate price movement percentage
        if position['side'] == 'LONG':
            price_move_pct = (entry_price - stop_loss_price) / entry_price * 100
        else:  # SHORT
            price_move_pct = (stop_loss_price - entry_price) / entry_price * 100

        # Calculate USD loss at stop-loss
        # 🔥 FIX: position_value ALREADY includes leverage, don't multiply again!
        usd_loss_at_sl = position_value * (price_move_pct / 100)

        # Calculate entry fee (Binance futures taker: 0.05%)
        taker_fee_rate = Decimal("0.0005")  # 0.05%
        notional_value = Decimal(str(position['quantity'])) * entry_price
        entry_fee = notional_value * taker_fee_rate

        # Calculate approximate exit fee (same rate)
        exit_fee_estimate = notional_value * taker_fee_rate
        total_fees = entry_fee + exit_fee_estimate

        # Net position value after entry fee
        net_position_value = position_value - entry_fee

        # 🔥 FIX: Use database profit target (user's configured value)
        # PROBLEM: Was using volatility-based targets ($1.50-2.50) in Telegram
        # - position_monitor.py uses database value ✓
        # - But telegram_notifier.py used old volatility logic ✗
        # - User sees $1.50 in Telegram but position closes at $2.50! ✗
        #
        # SOLUTION: Match position_monitor.py logic exactly
        # - Use min_profit_target_usd from position data (from database)
        # - Consistent messaging across Telegram and actual exits
        PROFIT_TARGET_USD = Decimal(str(position.get('min_profit_target_usd', self.settings.min_profit_usd)))

        # Get volatility for display level only (not for calculation)
        atr_percent = 3.0  # Default
        entry_snapshot = position.get('entry_snapshot', {})
        if entry_snapshot and isinstance(entry_snapshot, dict):
            indicators = entry_snapshot.get('indicators', {})
            if indicators:
                atr_percent = indicators.get('atr_percent', 3.0)

        # Display volatility level (informational only)
        TARGET_LEVEL = "Low Vol" if atr_percent < 2.5 else "Med Vol" if atr_percent < 4.5 else "High Vol"

        # Build base message
        message_parts = [
            f"{emoji} <b>NEW POSITION OPENED</b>\n",
            f"💎 <b>{position['symbol']}</b>",
            f"📊 Direction: <b>{position['side']} {position['leverage']}x</b>\n",
            f"💰 Position Size: <b>${float(position_value):.2f}</b>",
            f"💵 Entry Price: <b>${float(entry_price):.4f}</b>",
            f"📏 Quantity: <b>{float(position['quantity']):.6f}</b>\n",
            f"🛑 Stop-Loss: <b>${float(stop_loss_price):.4f}</b>",
            f"   ├ Price Move: <b>{float(price_move_pct):.2f}%</b>",
            f"   └ Max Loss: <b>${float(usd_loss_at_sl):.2f}</b> (with {leverage}x leverage)\n",
        ]
        
        # 🎯 v6.5: Show DYNAMIC profit targets
        profit_target_1 = position.get('profit_target_1')
        profit_target_2 = position.get('profit_target_2')
        target_source = position.get('target_source', 'R/R-based')
        
        message_parts.append(f"🎯 <b>DYNAMIC PROFIT TARGETS</b> ({target_source})")
        message_parts.append(f"   💎 Min Profit: <b>${float(PROFIT_TARGET_USD):.2f}</b> ({TARGET_LEVEL})")
        
        if profit_target_1:
            t1_dist = abs(float(profit_target_1) - float(entry_price)) / float(entry_price) * 100
            message_parts.append(f"   🎯 TP1 (50%): <b>${float(profit_target_1):.4f}</b> (+{t1_dist:.2f}%)")
        
        if profit_target_2:
            t2_dist = abs(float(profit_target_2) - float(entry_price)) / float(entry_price) * 100
            message_parts.append(f"   🎯 TP2 (50%): <b>${float(profit_target_2):.4f}</b> (+{t2_dist:.2f}%)")
        
        message_parts.append(f"⚠️ Liquidation: <b>${float(position['liquidation_price']):.4f}</b>\n")

        # 🎯 ALWAYS SHOW: Price Action analysis (even when no signals found)
        message_parts.append("\n📈 <b>PRICE ACTION ANALYSIS</b>")

        pa_data = position.get('price_action')
        if pa_data and isinstance(pa_data, dict):
            # 🔥 FIX: Support BOTH PA data formats:
            # 1. OLD: snapshot_capture format (nearest_support, nearest_resistance)
            # 2. NEW: price_action_analyzer format (support_levels, resistance_levels, validated, reason)

            # Check which format we have
            has_old_format = pa_data.get('nearest_support') and pa_data.get('nearest_resistance')
            has_new_format = pa_data.get('validated') and pa_data.get('reason')

            # Show if we have real PA data (either format)
            if has_old_format or has_new_format:
                # ━━━ NEW FORMAT (price_action_analyzer.py) ━━━
                if has_new_format:
                    # Show PA validation reason
                    message_parts.append(f"   ✅ <b>{pa_data.get('reason', 'PA setup confirmed')}</b>")

                    # Support & Resistance
                    support_levels = pa_data.get('support_levels', [])
                    resistance_levels = pa_data.get('resistance_levels', [])

                    # 🎯 ENHANCED: Show multiple S/R levels (2-4) for better context
                    # USER REQUEST: Show nearby candles' support/resistance levels
                    # - Traders need to see multiple levels for better decision making
                    # - Shows confluences and zones, not just single price
                    if support_levels:
                        # Get current price for distance calculation
                        current_price = Decimal(str(position['entry_price']))

                        # Show up to 3 nearest support levels
                        num_supports = min(3, len(support_levels))
                        if num_supports == 1:
                            message_parts.append(f"   🟢 Support: <b>${float(support_levels[0]):.4f}</b>")
                        else:
                            support_lines = []
                            for i, level in enumerate(support_levels[:num_supports]):
                                dist_pct = abs((current_price - Decimal(str(level))) / current_price * 100)
                                support_lines.append(f"${float(level):.4f} ({float(dist_pct):.1f}%)")
                            message_parts.append(f"   🟢 Supports: <b>{' → '.join(support_lines)}</b>")

                    if resistance_levels:
                        # Get current price for distance calculation
                        current_price = Decimal(str(position['entry_price']))

                        # Show up to 3 nearest resistance levels
                        num_resistances = min(3, len(resistance_levels))
                        if num_resistances == 1:
                            message_parts.append(f"   🔴 Resistance: <b>${float(resistance_levels[0]):.4f}</b>")
                        else:
                            resistance_lines = []
                            for i, level in enumerate(resistance_levels[:num_resistances]):
                                dist_pct = abs((Decimal(str(level)) - current_price) / current_price * 100)
                                resistance_lines.append(f"${float(level):.4f} ({float(dist_pct):.1f}%)")
                            message_parts.append(f"   🔴 Resistances: <b>{' → '.join(resistance_lines)}</b>")

                    # Risk/Reward from PA analyzer
                    rr_ratio = pa_data.get('rr_ratio', 0)
                    if rr_ratio > 0:
                        message_parts.append(f"   ⚖️ Risk/Reward: <b>{float(rr_ratio):.2f}:1</b>")

                    # Trend from PA analyzer
                    trend = pa_data.get('trend', 'N/A')
                    trend_strength = pa_data.get('trend_strength', 'N/A')
                    adx = pa_data.get('adx', 0)

                    trend_emoji = '📈' if 'UP' in str(trend).upper() else '📉' if 'DOWN' in str(trend).upper() else '➡️'
                    message_parts.append(
                        f"   {trend_emoji} Trend: <b>{trend}</b> ({trend_strength}, ADX: {float(adx):.1f})"
                    )

                    # Volume from PA analyzer
                    volume_surge = pa_data.get('volume_surge', False)
                    volume_ratio = pa_data.get('volume_ratio', 1.0)

                    surge_emoji = '🔥' if volume_surge else '📊'
                    message_parts.append(
                        f"   {surge_emoji} Volume: <b>{float(volume_ratio):.1f}x avg</b>" +
                        (" (SURGE!)" if volume_surge else "")
                    )

                # ━━━ OLD FORMAT (snapshot_capture) ━━━
                else:
                    nearest_support = pa_data.get('nearest_support')
                    nearest_resistance = pa_data.get('nearest_resistance')
                    support_dist = pa_data.get('support_dist', 0) * 100
                    resistance_dist = pa_data.get('resistance_dist', 0) * 100

                    message_parts.append(
                        f"   🟢 Support: <b>${float(nearest_support):.4f}</b> ({support_dist:.1f}% away)"
                    )
                    message_parts.append(
                        f"   🔴 Resistance: <b>${float(nearest_resistance):.4f}</b> ({resistance_dist:.1f}% away)"
                    )

                    # S/R Strength
                    sr_strength = pa_data.get('sr_strength_score', 0)
                    swing_lows = pa_data.get('swing_lows_count', 0)
                    swing_highs = pa_data.get('swing_highs_count', 0)
                    message_parts.append(
                        f"   💪 S/R Strength: <b>{float(sr_strength):.0%}</b> ({swing_lows} lows, {swing_highs} highs)"
                    )

                    # Risk/Reward
                    rr_long = pa_data.get('rr_long', 0)
                    rr_short = pa_data.get('rr_short', 0)
                    if position['side'] == 'LONG':
                        message_parts.append(f"   ⚖️ Risk/Reward: <b>{float(rr_long):.2f}:1</b> (LONG)")
                    else:
                        message_parts.append(f"   ⚖️ Risk/Reward: <b>{float(rr_short):.2f}:1</b> (SHORT)")

                    # Trend Analysis
                    trend_direction = pa_data.get('trend_direction', 'SIDEWAYS')
                    trend_adx = pa_data.get('trend_adx', 0)
                    trend_quality = pa_data.get('trend_quality', 0)

                    trend_emoji = '📈' if trend_direction == 'UPTREND' else '📉' if trend_direction == 'DOWNTREND' else '➡️'
                    message_parts.append(
                        f"   {trend_emoji} Trend: <b>{trend_direction}</b> | Quality: <b>{float(trend_quality):.0%}</b> (ADX: {float(trend_adx):.1f})"
                    )

                    # Volume Analysis
                    volume_surge = pa_data.get('volume_surge', False)
                    volume_ratio = pa_data.get('volume_surge_ratio', 1.0)

                    surge_emoji = '🔥' if volume_surge else '📊'
                    message_parts.append(
                        f"   {surge_emoji} Volume: <b>{float(volume_ratio):.1f}x avg</b>" +
                        (" (SURGE!)" if volume_surge else "")
                    )
            else:
                # No PA signals detected - show this to user so they know PA is running
                message_parts.append(
                    f"   ❌ <b>No signals detected</b> (PA scanner found no clear patterns)"
                )
        else:
            # No PA data at all - show this to user
            message_parts.append(
                f"   ❌ <b>No PA data available</b> (PA analysis not run for this position)"
            )

        message_parts.append("")  # Empty line after PA section

        # Add fees and AI info
        message_parts.extend([
            f"💸 <b>Trading Fees (Binance):</b>",
            f"   ├ Entry Fee: <b>${float(entry_fee):.2f}</b> (0.05% taker)",
            f"   ├ Est. Exit Fee: <b>${float(exit_fee_estimate):.2f}</b> (0.05% taker)",
            f"   └ Total Fees: <b>${float(total_fees):.2f}</b>\n",
            f"📊 <b>Net Position:</b>",
            f"   └ After Entry Fee: <b>${float(net_position_value):.2f}</b>\n",
            f"🤖 AI Confidence: <b>{float(position.get('ai_confidence', 0))*100:.0f}%</b>",
            f"🤝 Consensus: <b>{position.get('ai_model_consensus', 'N/A')}</b>\n",
            f"⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}"
        ])

        message = "\n".join(message_parts)
        await self.send_message(message)

    async def send_position_update(self, position: Dict[str, Any], pnl: Decimal) -> None:
        """Periodic position update (every 5 min)."""
        emoji = "💰" if pnl > 0 else "📉"
        pnl_emoji = "🟢" if pnl > 0 else "🔴"

        duration = (datetime.now() - position.get('entry_time', datetime.now())).total_seconds()
        duration_str = format_duration(int(duration))

        message = f"""
{emoji} <b>POSITION UPDATE</b>

💎 {position['symbol']} {position['side']} {position['leverage']}x

💵 Entry: ${float(position['entry_price']):.4f}
💵 Current: ${float(position.get('current_price', 0)):.4f}
{pnl_emoji} Unrealized P&L: <b>${float(pnl):+.2f}</b>

⏱️ Duration: {duration_str}
⏰ {get_turkey_time().strftime('%H:%M:%S')}
"""
        await self.send_message(message)

    async def send_position_closed(
        self,
        position: Dict[str, Any],
        exit_price: Decimal,
        pnl: Decimal,
        reason: str
    ) -> None:
        """Position closed notification with detailed fee breakdown."""
        from decimal import Decimal

        emoji = "✅" if pnl > 0 else "❌"
        position_value = Decimal(str(position['position_value_usd']))
        pnl_percent = (float(pnl) / float(position_value)) * 100

        # Calculate trading fees
        taker_fee_rate = Decimal("0.0005")  # 0.05%
        entry_price = Decimal(str(position['entry_price']))
        quantity = Decimal(str(position['quantity']))

        # Entry fee
        entry_notional = quantity * entry_price
        entry_fee = entry_notional * taker_fee_rate

        # Exit fee
        exit_notional = quantity * Decimal(str(exit_price))
        exit_fee = exit_notional * taker_fee_rate

        total_fees = entry_fee + exit_fee

        # Gross PnL (before fees) - calculate from price difference
        # 🔥 CRITICAL FIX: position_value ALREADY includes leverage!
        # position_value = margin × leverage (e.g., $40 × 25 = $1000)
        # So we should NOT multiply by leverage again!
        leverage = int(position['leverage'])
        if position['side'] == 'LONG':
            price_change_pct = (Decimal(str(exit_price)) - entry_price) / entry_price
        else:  # SHORT
            price_change_pct = (entry_price - Decimal(str(exit_price))) / entry_price

        # 🔥 FIX: Don't multiply by leverage - position_value already includes it!
        gross_pnl = position_value * price_change_pct

        # Net PnL (after fees)
        net_pnl = gross_pnl - total_fees

        duration = (datetime.now() - position.get('entry_time', datetime.now())).total_seconds()
        duration_str = format_duration(int(duration))

        message = f"""
{emoji} <b>POSITION CLOSED</b>

💎 <b>{position['symbol']}</b> {position['side']} {position['leverage']}x

💵 Entry: ${float(entry_price):.4f}
💵 Exit: ${float(exit_price):.4f}

💰 <b>P&L Breakdown:</b>
   ├ Gross P&L: <b>${float(gross_pnl):+.2f}</b>
   ├ Trading Fees: <b>-${float(total_fees):.2f}</b>
   │  ├ Entry Fee: ${float(entry_fee):.2f}
   │  └ Exit Fee: ${float(exit_fee):.2f}
   └ <b>Net P&L: ${float(net_pnl):+.2f} ({float(net_pnl/position_value*100):+.1f}%)</b>

📝 Reason: {reason}
⏱️ Duration: {duration_str}

⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await self.send_message(message)

    async def send_multi_position_update(self, positions: list) -> None:
        """
        Send a consolidated update for multiple positions with NET P&L (after fees).
        Reduces Telegram spam by showing all positions in one message.

        Args:
            positions: List of active positions with updated P&L
        """
        if not positions:
            return

        # Calculate total NET P&L (already includes fees from calculate_pnl)
        total_pnl = sum(Decimal(str(pos.get('unrealized_pnl_usd', 0))) for pos in positions)

        # Count winning/losing positions
        winners = sum(1 for pos in positions if Decimal(str(pos.get('unrealized_pnl_usd', 0))) > 0)
        losers = sum(1 for pos in positions if Decimal(str(pos.get('unrealized_pnl_usd', 0))) < 0)

        # Header
        emoji = "📈" if total_pnl > 0 else "📉" if total_pnl < 0 else "📊"
        message = f"""
{emoji} <b>PORTFOLIO UPDATE</b> ({len(positions)} position{'s' if len(positions) > 1 else ''})

"""

        # Position lines (group by 2 for compact display)
        for i, pos in enumerate(positions):
            pnl = Decimal(str(pos.get('unrealized_pnl_usd', 0)))
            pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            symbol_short = pos['symbol'].replace('/USDT:USDT', '').replace('/USDT', '')

            message += f"{pnl_emoji} <b>{symbol_short}</b>: ${float(pnl):+.2f}"

            # Add line break after every 2 positions
            if (i + 1) % 2 == 0:
                message += "\n"
            else:
                message += " | "

        # Remove trailing separator if odd number of positions
        if len(positions) % 2 != 0:
            message = message.rstrip(" | ")

        # Summary
        total_emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
        message += f"""

━━━━━━━━━━━━━━━━━━━━
{total_emoji} <b>Net P&L: ${float(total_pnl):+.2f}</b> (after est. fees)
📊 W/L: {winners}/{losers}

⏰ {get_turkey_time().strftime('%H:%M:%S')}
"""
        await self.send_message(message)

    async def send_portfolio_update(
        self,
        capital: Decimal,
        daily_pnl: Decimal,
        position: Optional[Dict[str, Any]] = None
    ) -> None:
        """Full portfolio status."""
        daily_emoji = "📈" if daily_pnl > 0 else "📉"

        message = f"""
💼 <b>PORTFOLIO STATUS</b>

💰 Total Capital: <b>${float(capital):.2f}</b>
{daily_emoji} Today's P&L: <b>${float(daily_pnl):+.2f}</b>

"""
        if position:
            pnl = position.get('unrealized_pnl_usd', 0)
            emoji = "💰" if float(pnl) > 0 else "📉"
            message += f"""
📍 <b>OPEN POSITION:</b>
💎 {position['symbol']} {position['side']} {position['leverage']}x
💵 Entry: ${float(position['entry_price']):.4f}
💵 Current: ${float(position.get('current_price', 0)):.4f}
{emoji} Unrealized: ${float(pnl):+.2f}
"""
        else:
            message += """
📍 <b>No Open Position</b>
🔍 Scanning for opportunities...
"""

        message += f"\n⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}"
        await self.send_message(message)

    async def send_scan_result(self, symbol: str, confidence: float, action: str) -> None:
        """Market scan result."""
        action_emoji = "📈" if action == 'buy' else "📉" if action == 'sell' else "⏸️"

        message = f"""
🔍 <b>SCAN COMPLETE</b>

💎 Best Opportunity: <b>{symbol}</b>
{action_emoji} Signal: <b>{action.upper()}</b>
🤖 AI Confidence: <b>{confidence*100:.0f}%</b>

{'📈 Initiating trade...' if confidence >= 0.80 else '⏳ Waiting for stronger signal...'}

⏰ {get_turkey_time().strftime('%H:%M:%S')}
"""
        await self.send_message(message)

    async def send_alert(self, alert_type: str, message_text: str) -> None:
        """General alerts."""
        emoji_map = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨',
            'success': '✅'
        }
        emoji = emoji_map.get(alert_type, 'ℹ️')

        # 🔥 FIX: Escape HTML entities in message_text to prevent parsing errors
        # PROBLEM: If message_text contains < or >, Telegram tries to parse as HTML tag
        # EXAMPLE: "BTC/USDT<something>" → "unsupported start tag" error
        # SOLUTION: Escape HTML entities before constructing message
        escaped_text = html.escape(message_text)

        message = f"{emoji} <b>{alert_type.upper()}</b>\n\n{escaped_text}\n\n⏰ {get_turkey_time().strftime('%H:%M:%S')}"
        await self.send_message(message)

    async def send_daily_summary(self, summary_data: Dict[str, Any]) -> None:
        """End of day summary."""
        pnl_emoji = "💹" if float(summary_data.get('daily_pnl', 0)) > 0 else "📉"

        message = f"""
📊 <b>DAILY SUMMARY</b>
{summary_data.get('date', datetime.now().strftime('%Y-%m-%d'))}

💰 Starting Capital: ${float(summary_data.get('starting_capital', 0)):.2f}
💰 Ending Capital: ${float(summary_data.get('ending_capital', 0)):.2f}
{pnl_emoji} <b>Daily P&L: ${float(summary_data.get('daily_pnl', 0)):+.2f}</b>

📈 Total Trades: {summary_data.get('total_trades', 0)}
✅ Winners: {summary_data.get('winning_trades', 0)}
❌ Losers: {summary_data.get('losing_trades', 0)}
📊 Win Rate: {float(summary_data.get('win_rate', 0)):.1f}%

💎 Best Trade: ${float(summary_data.get('largest_win', 0)):.2f}
📉 Worst Trade: ${float(summary_data.get('largest_loss', 0)):.2f}

See you tomorrow! 🌙
"""
        await self.send_message(message)

    async def send_circuit_breaker_alert(
        self,
        event_type: str,
        trigger_value: Decimal,
        threshold_value: Decimal
    ) -> None:
        """Circuit breaker activation alert."""
        message = f"""
🚨 <b>CIRCUIT BREAKER ACTIVATED</b>

Type: <b>{event_type}</b>
Trigger Value: ${float(trigger_value):.2f}
Threshold: ${float(threshold_value):.2f}

<b>⚠️ TRADING PAUSED ⚠️</b>

The bot has automatically stopped trading to protect your capital.

Please review your strategy and performance before resuming.

⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await self.send_message(message)

    async def send_error_report(self, component: str, error_message: str) -> None:
        """Send error report."""
        message = f"""
❌ <b>ERROR REPORTED</b>

Component: <b>{component}</b>

Error: {error_message[:500]}

The bot is attempting to recover automatically.
If this persists, manual intervention may be required.

⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await self.send_message(message)

    async def send_risk_warning(self, warning_message: str) -> None:
        """Send risk-related warning."""
        message = f"""
⚠️ <b>RISK WARNING</b>

{warning_message}

⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await self.send_message(message)


# Singleton instance
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get or create notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
