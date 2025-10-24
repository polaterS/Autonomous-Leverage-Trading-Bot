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
🛑 Stop-Loss Range: {float(self.settings.min_stop_loss_percent)*100}% - {float(self.settings.max_stop_loss_percent)*100}%
💎 Min Profit: ${self.settings.min_profit_usd}
🎯 Min AI Confidence: {float(self.settings.min_ai_confidence)*100}%

Sit back and monitor your portfolio! 💰

⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await self.send_message(message)

    async def send_position_opened(self, position: Dict[str, Any]) -> None:
        """New position opened notification."""
        emoji = "🟢" if position['side'] == 'LONG' else "🔴"

        message = f"""
{emoji} <b>NEW POSITION OPENED</b>

💎 <b>{position['symbol']}</b>
📊 Direction: <b>{position['side']} {position['leverage']}x</b>

💰 Position Size: <b>${float(position['position_value_usd']):.2f}</b>
💵 Entry Price: <b>${float(position['entry_price']):.4f}</b>
📏 Quantity: <b>{float(position['quantity']):.6f}</b>

🛑 Stop-Loss: <b>${float(position['stop_loss_price']):.4f}</b> (-{float(position['stop_loss_percent']):.1f}%)
💎 Min Profit Target: <b>${float(position['min_profit_target_usd']):.2f}</b>
⚠️ Liquidation: <b>${float(position['liquidation_price']):.4f}</b>

🤖 AI Confidence: <b>{float(position.get('ai_confidence', 0))*100:.0f}%</b>
🤝 Consensus: <b>{position.get('ai_model_consensus', 'N/A')}</b>

⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}
"""
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
        """Position closed notification."""
        emoji = "✅" if pnl > 0 else "❌"
        pnl_percent = (float(pnl) / float(position['position_value_usd'])) * 100

        duration = (datetime.now() - position.get('entry_time', datetime.now())).total_seconds()
        duration_str = format_duration(int(duration))

        message = f"""
{emoji} <b>POSITION CLOSED</b>

💎 <b>{position['symbol']}</b> {position['side']} {position['leverage']}x

💵 Entry: ${float(position['entry_price']):.4f}
💵 Exit: ${float(exit_price):.4f}

{emoji} <b>Profit/Loss: ${float(pnl):+.2f} ({pnl_percent:+.1f}%)</b>

📝 Reason: {reason}
⏱️ Duration: {duration_str}

⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}
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

        message = f"{emoji} <b>{alert_type.upper()}</b>\n\n{message_text}\n\n⏰ {get_turkey_time().strftime('%H:%M:%S')}"
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
