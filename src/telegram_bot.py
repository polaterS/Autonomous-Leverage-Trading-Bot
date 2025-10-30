"""
Interactive Telegram Bot for trade management and control.
Provides commands, buttons, and real-time interaction.
"""

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)
from telegram.constants import ParseMode
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import asyncio
import logging

from src.config import get_settings
from src.database import DatabaseClient
from src.utils import format_duration
from src.risk_manager import get_risk_manager
from src.trade_executor import get_trade_executor

logger = logging.getLogger('trading_bot')

# Turkey timezone (UTC+3)
TURKEY_TZ = timezone(timedelta(hours=3))


def get_turkey_time() -> datetime:
    """Get current time in Turkey timezone (UTC+3)."""
    return datetime.now(TURKEY_TZ)


class TradingTelegramBot:
    """Interactive Telegram bot for trade management."""

    def __init__(self, db_client: DatabaseClient):
        self.settings = get_settings()
        self.db = db_client
        self.application = None
        self.bot_running = True
        self.pending_trade = None  # Store pending trade for user confirmation

    async def initialize(self):
        """Initialize the Telegram bot application."""
        self.application = (
            Application.builder()
            .token(self.settings.telegram_bot_token)
            .build()
        )

        # Register command handlers
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("positions", self.cmd_positions))
        self.application.add_handler(CommandHandler("history", self.cmd_history))
        self.application.add_handler(CommandHandler("scan", self.cmd_scan))
        self.application.add_handler(CommandHandler("stopbot", self.cmd_stop_bot))
        self.application.add_handler(CommandHandler("startbot", self.cmd_start_bot))

        # Register callback query handler for buttons
        self.application.add_handler(CallbackQueryHandler(self.button_callback))

        # Initialize bot
        await self.application.initialize()
        await self.application.start()

        logger.info("✅ Interactive Telegram bot initialized")

    async def run(self):
        """Run the Telegram bot (polling)."""
        if not self.application:
            await self.initialize()

        logger.info("🤖 Starting Telegram bot polling...")
        try:
            # Run until stopped
            await self.application.updater.start_polling(
                allowed_updates=["message", "callback_query"]
            )

            # Keep running
            while True:
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Telegram bot error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown the Telegram bot."""
        if self.application:
            try:
                await self.application.updater.stop()
            except:
                pass
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram bot shutdown")

    # ==================== Command Handlers ====================

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        keyboard = [
            [
                InlineKeyboardButton("📊 Bot Durumu", callback_data="status"),
                InlineKeyboardButton("💼 Pozisyonlar", callback_data="positions"),
            ],
            [
                InlineKeyboardButton("📜 Geçmiş", callback_data="history"),
                InlineKeyboardButton("🔍 Market Tara", callback_data="scan"),
            ],
            [
                InlineKeyboardButton("▶️ Bot Başlat", callback_data="start_bot"),
                InlineKeyboardButton("■ Bot Durdur", callback_data="stop_bot"),
            ],
            [
                InlineKeyboardButton("❓ Yardım", callback_data="help"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        message = """
🤖 <b>AUTONOMOUS TRADING BOT</b>

Hoş geldiniz! Bot komutları:

<b>📊 Durum Kontrol:</b>
/status - Bot durumu ve aktif pozisyon
/positions - Açık pozisyonlarım
/history - Kapalı pozisyonlar

<b>🎮 Bot Kontrol:</b>
/scan - Manuel market tarama
/startbot - Botu başlat
/stopbot - Botu durdur

<b>❓ Yardım:</b>
/help - Detaylı yardım

Aşağıdaki butonları kullanarak da kontrol edebilirsiniz:
"""
        await update.message.reply_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        message = """
<b>📖 YARDIM MENÜSÜ</b>

<b>Temel Komutlar:</b>
/start - Ana menüyü aç
/status - Bot durumu ve sermaye bilgisi
/positions - Aktif pozisyonları görüntüle
/history - Kapalı pozisyon geçmişi
/scan - Manuel market tarama yap
/startbot - Botu çalıştır
/stopbot - Botu durdur

<b>Nasıl Çalışır?</b>

1️⃣ <b>Otomatik Tarama:</b>
Bot her 5 dakikada bir 35 kripto parayı tarar.

2️⃣ <b>Fırsat Bulma:</b>
AI analizi ile en iyi fırsatı bulur.

3️⃣ <b>Leverage Seçimi:</b>
Sana 2x'den 50x'e kadar tüm seçenekleri gösterir.
Sen hangi leverage'ı istediğini seçersin.

4️⃣ <b>Otomatik Yönetim:</b>
Position açıldıktan sonra bot:
- Her dakika P&L kontrolü yapar
- Stop-loss takip eder
- Liquidation mesafesini izler
- Kar hedefine ulaşınca kapatır

5️⃣ <b>Telegram Bildirimleri:</b>
Her adımda bilgilendirilirsin:
- Fırsat bulundu
- Position açıldı
- P&L güncellemeleri
- Position kapandı

<b>⚠️ Risk Yönetimi:</b>
- Yüksek leverage = Yüksek risk
- 30x-50x çok riskli, dikkatli kullan
- Stop-loss her zaman aktif
- Paper trading ile önce test et

Sorularınız için: @your_support
"""
        await update.message.reply_text(message, parse_mode=ParseMode.HTML)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        try:
            # Get bot status
            status_emoji = "🟢" if self.bot_running else "🔴"
            status_text = "RUNNING" if self.bot_running else "STOPPED"

            # Get capital info
            config = await self.db.get_trading_config()
            capital = float(config['current_capital']) if config else 0.0

            # Get daily P&L
            daily_pnl = await self.db.get_daily_pnl()
            pnl_emoji = "📈" if daily_pnl >= 0 else "📉"

            # Get active position
            position = await self.db.get_active_position()

            message = f"""
<b>📊 BOT DURUMU</b>

{status_emoji} <b>Durum:</b> {status_text}
💰 <b>Sermaye:</b> ${capital:.2f}
{pnl_emoji} <b>Bugünkü P&L:</b> ${daily_pnl:+.2f}

<b>📍 Aktif Pozisyon:</b>
"""
            if position:
                entry_price = float(position['entry_price'])
                current_price = float(position.get('current_price', entry_price))
                pnl = float(position.get('unrealized_pnl_usd', 0))
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"

                message += f"""
{pnl_emoji} <b>{position['symbol']}</b> {position['side']} {position['leverage']}x

💵 Entry: ${entry_price:.4f}
💵 Current: ${current_price:.4f}
💰 P&L: ${pnl:+.2f}
🛑 Stop-Loss: ${float(position['stop_loss_price']):.4f}
⚠️ Liquidation: ${float(position['liquidation_price']):.4f}
"""
            else:
                message += "\n❌ Şu anda açık pozisyon yok"

            message += f"\n⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}"

            await update.message.reply_text(message, parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await update.message.reply_text(f"❌ Hata: {e}")

    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command."""
        try:
            position = await self.db.get_active_position()

            if not position:
                await update.message.reply_text(
                    "❌ Şu anda açık pozisyon bulunmuyor.",
                    parse_mode=ParseMode.HTML
                )
                return

            entry_price = float(position['entry_price'])
            current_price = float(position.get('current_price', entry_price))
            pnl = float(position.get('unrealized_pnl_usd', 0))
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"

            message = f"""
<b>💼 AKTİF POZİSYON</b>

{pnl_emoji} <b>{position['symbol']}</b>

<b>📊 Detaylar:</b>
• Yön: {position['side']} {position['leverage']}x
• Miktar: {float(position['quantity']):.6f}
• Pozisyon Değeri: ${float(position['position_value_usd']):.2f}

<b>💵 Fiyatlar:</b>
• Entry: ${entry_price:.4f}
• Current: ${current_price:.4f}
• Stop-Loss: ${float(position['stop_loss_price']):.4f} ({float(position['stop_loss_percent'])*100:.1f}%)
• Liquidation: ${float(position['liquidation_price']):.4f}

<b>💰 Kar/Zarar:</b>
• P&L: ${pnl:+.2f}
• Min Kar Hedefi: ${float(position['min_profit_target_usd']):.2f}

<b>🤖 AI:</b>
• Model: {position.get('ai_model_consensus', 'N/A')}
• Güven: {float(position.get('ai_confidence', 0))*100:.0f}%

<b>⏰ Süre:</b>
• Açılış: {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}
"""
            # Add close position button
            keyboard = [[InlineKeyboardButton("❌ Pozisyonu Kapat", callback_data="close_position")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(message, parse_mode=ParseMode.HTML, reply_markup=reply_markup)

        except Exception as e:
            logger.error(f"Error in positions command: {e}")
            await update.message.reply_text(f"❌ Hata: {e}")

    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history command."""
        try:
            trades = await self.db.get_recent_trades(limit=10)

            if not trades:
                await update.message.reply_text(
                    "❌ Henüz kapalı pozisyon yok.",
                    parse_mode=ParseMode.HTML
                )
                return

            message = "<b>📜 KAPALI POZİSYONLAR (Son 10)</b>\n\n"

            for trade in trades:
                pnl = float(trade['realized_pnl_usd'])
                emoji = "✅" if pnl > 0 else "❌"

                message += f"""
{emoji} <b>{trade['symbol']}</b> {trade['side']} {trade['leverage']}x
💰 P&L: ${pnl:+.2f} ({float(trade['pnl_percent']):+.2f}%)
📅 {trade['exit_time'].strftime('%d/%m %H:%M')}
━━━━━━━━━━━━━━━━
"""

            await update.message.reply_text(message, parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"Error in history command: {e}")
            await update.message.reply_text(f"❌ Hata: {e}")

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /scan command."""
        await update.message.reply_text(
            "🔍 Market taraması başlatılıyor...\n\nBu işlem 5-6 dakika sürebilir.",
            parse_mode=ParseMode.HTML
        )
        # Actual scan will be triggered by trading engine
        # This just sends a notification

    async def cmd_start_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /startbot command."""
        self.bot_running = True
        await update.message.reply_text(
            "✅ Bot başlatıldı! Market tarama devam ediyor...",
            parse_mode=ParseMode.HTML
        )

    async def cmd_stop_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stopbot command."""
        self.bot_running = False
        await update.message.reply_text(
            "⏸️ Bot durduruldu. Yeni pozisyon açılmayacak.\n\n"
            "Mevcut pozisyon varsa takip edilmeye devam edilecek.",
            parse_mode=ParseMode.HTML
        )

    # ==================== Button Callback Handler ====================

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks."""
        query = update.callback_query
        await query.answer()

        callback_data = query.data

        if callback_data == "status":
            await self.handle_status_button(query)
        elif callback_data == "positions":
            await self.handle_positions_button(query)
        elif callback_data == "history":
            await self.handle_history_button(query)
        elif callback_data == "scan":
            await self.handle_scan_button(query)
        elif callback_data == "start_bot":
            await self.handle_start_bot_button(query)
        elif callback_data == "stop_bot":
            await self.handle_stop_bot_button(query)
        elif callback_data == "help":
            await self.handle_help_button(query)
        elif callback_data == "close_position":
            await self.handle_close_position_button(query)
        elif callback_data == "cancel_trade":
            await self.handle_cancel_trade_button(query)
        elif callback_data.startswith("leverage_"):
            await self.handle_leverage_selection(query, callback_data)

    async def handle_status_button(self, query):
        """Handle status button."""
        # Reuse cmd_status logic
        status_emoji = "🟢" if self.bot_running else "🔴"
        status_text = "RUNNING" if self.bot_running else "STOPPED"

        config = await self.db.get_trading_config()
        capital = float(config['current_capital']) if config else 0.0
        daily_pnl = await self.db.get_daily_pnl()

        message = f"""
<b>📊 BOT DURUMU</b>

{status_emoji} <b>Durum:</b> {status_text}
💰 <b>Sermaye:</b> ${capital:.2f}
📈 <b>Bugünkü P&L:</b> ${daily_pnl:+.2f}

⏰ {get_turkey_time().strftime('%H:%M:%S')}
"""
        await query.edit_message_text(message, parse_mode=ParseMode.HTML)

    async def handle_positions_button(self, query):
        """Handle positions button."""
        position = await self.db.get_active_position()

        if not position:
            await query.edit_message_text(
                "❌ Şu anda açık pozisyon bulunmuyor.",
                parse_mode=ParseMode.HTML
            )
            return

        pnl = float(position.get('unrealized_pnl_usd', 0))
        emoji = "🟢" if pnl >= 0 else "🔴"

        message = f"""
<b>💼 AKTİF POZİSYON</b>

{emoji} <b>{position['symbol']}</b> {position['side']} {position['leverage']}x

💰 P&L: ${pnl:+.2f}
💵 Entry: ${float(position['entry_price']):.4f}
💵 Current: ${float(position.get('current_price', 0)):.4f}

⏰ {get_turkey_time().strftime('%H:%M:%S')}
"""
        await query.edit_message_text(message, parse_mode=ParseMode.HTML)

    async def handle_history_button(self, query):
        """Handle history button."""
        trades = await self.db.get_recent_trades(limit=5)

        if not trades:
            await query.edit_message_text("❌ Henüz kapalı pozisyon yok.")
            return

        message = "<b>📜 Son 5 Pozisyon</b>\n\n"
        for trade in trades:
            pnl = float(trade['realized_pnl_usd'])
            emoji = "✅" if pnl > 0 else "❌"
            message += f"{emoji} {trade['symbol']}: ${pnl:+.2f}\n"

        await query.edit_message_text(message, parse_mode=ParseMode.HTML)

    async def handle_scan_button(self, query):
        """Handle scan button."""
        await query.edit_message_text(
            "🔍 Market taraması başlatılıyor...\n\n"
            "Bu işlem 5-6 dakika sürebilir.",
            parse_mode=ParseMode.HTML
        )

    async def handle_start_bot_button(self, query):
        """Handle start bot button."""
        self.bot_running = True
        await query.edit_message_text(
            "✅ Bot başlatıldı!",
            parse_mode=ParseMode.HTML
        )

    async def handle_stop_bot_button(self, query):
        """Handle stop bot button."""
        self.bot_running = False
        await query.edit_message_text(
            "⏸️ Bot durduruldu.",
            parse_mode=ParseMode.HTML
        )

    async def handle_help_button(self, query):
        """Handle help button."""
        message = """
<b>❓ HIZLI YARDIM</b>

/status - Bot durumu
/positions - Aktif pozisyonlar
/history - Geçmiş
/scan - Market tara
/startbot - Başlat
/stopbot - Durdur

Detaylı bilgi için /help yazın.
"""
        await query.edit_message_text(message, parse_mode=ParseMode.HTML)

    async def handle_close_position_button(self, query):
        """Handle close position button."""
        try:
            position = await self.db.get_active_position()

            if not position:
                await query.edit_message_text(
                    "❌ Kapatılacak pozisyon bulunamadı.",
                    parse_mode=ParseMode.HTML
                )
                return

            # Close the position
            from src.trade_executor import get_trade_executor
            executor = get_trade_executor()

            result = await executor.close_position(
                position=position,
                reason="Manual close via Telegram",
                exit_price=float(position.get('current_price', position['entry_price']))
            )

            if result.get('success'):
                pnl = float(result.get('realized_pnl_usd', 0))
                emoji = "✅" if pnl > 0 else "❌"

                await query.edit_message_text(
                    f"{emoji} <b>Pozisyon Kapatıldı!</b>\n\n"
                    f"💎 {position['symbol']} {position['side']} {position['leverage']}x\n"
                    f"💰 Realized P&L: ${pnl:+.2f}\n\n"
                    f"Pozisyon manuel olarak kapatıldı.",
                    parse_mode=ParseMode.HTML
                )
            else:
                await query.edit_message_text(
                    f"❌ Pozisyon kapatılamadı:\n\n{result.get('error', 'Unknown error')}",
                    parse_mode=ParseMode.HTML
                )

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            await query.edit_message_text(
                f"❌ Hata: {str(e)}",
                parse_mode=ParseMode.HTML
            )

    async def handle_cancel_trade_button(self, query):
        """Handle cancel trade button."""
        self.pending_trade = None
        await query.edit_message_text(
            "❌ Trade iptal edildi.",
            parse_mode=ParseMode.HTML
        )

    async def handle_leverage_selection(self, query, callback_data):
        """Handle leverage selection from opportunity message."""
        # Extract leverage from callback_data (e.g., "leverage_10x")
        leverage_str = callback_data.split("_")[1].replace("x", "")
        leverage = int(leverage_str)

        await query.edit_message_text(
            f"✅ {leverage}x leverage seçildi!\n\n"
            f"Position açılıyor...",
            parse_mode=ParseMode.HTML
        )

        # Execute trade if pending trade exists
        if self.pending_trade:
            try:
                await self._execute_pending_trade(leverage, query)
            except Exception as e:
                logger.error(f"Failed to execute trade: {e}")
                await query.edit_message_text(
                    f"❌ Trade execution failed: {str(e)}",
                    parse_mode=ParseMode.HTML
                )
                self.pending_trade = None

    async def _execute_pending_trade(self, leverage: int, query):
        """Execute the pending trade with selected leverage."""
        if not self.pending_trade:
            return

        symbol = self.pending_trade['symbol']
        analysis = self.pending_trade['analysis']
        market_data = self.pending_trade['market_data']

        logger.info(f"📊 Executing trade: {symbol} {analysis['side']} with {leverage}x leverage")

        # Update leverage in analysis
        analysis['suggested_leverage'] = leverage

        # Adaptive stop-loss based on leverage
        if leverage <= 5:
            analysis['stop_loss_percent'] = 15.0
        elif leverage <= 10:
            analysis['stop_loss_percent'] = 10.0
        elif leverage <= 20:
            analysis['stop_loss_percent'] = 5.0
        else:
            analysis['stop_loss_percent'] = 3.0

        # Validate with risk manager
        risk_manager = get_risk_manager()

        trade_params = {
            'symbol': symbol,
            'side': analysis['side'],
            'leverage': leverage,
            'stop_loss_percent': analysis['stop_loss_percent'],
            'current_price': market_data['current_price']
        }

        validation = await risk_manager.validate_trade(trade_params)

        if not validation['approved']:
            logger.warning(f"❌ Trade rejected by risk manager: {validation['reason']}")
            await query.edit_message_text(
                f"❌ Trade reddedildi:\n\n{validation['reason']}",
                parse_mode=ParseMode.HTML
            )
            self.pending_trade = None
            return

        # Use adjusted parameters if provided
        if 'adjusted_leverage' in validation:
            leverage = validation['adjusted_leverage']
            analysis['suggested_leverage'] = leverage

        if 'adjusted_stop_loss_percent' in validation:
            analysis['stop_loss_percent'] = validation['adjusted_stop_loss_percent']

        # Execute the trade
        executor = get_trade_executor()
        result = await executor.execute_trade(
            symbol=symbol,
            side=analysis['side'],
            leverage=leverage,
            stop_loss_percent=analysis['stop_loss_percent'],
            entry_price=market_data['current_price'],
            ai_confidence=analysis.get('confidence', 0),
            reasoning=analysis.get('reasoning', '')
        )

        if result['success']:
            logger.info(f"✅ Trade executed successfully: {result}")
            await query.edit_message_text(
                f"✅ Position açıldı!\n\n"
                f"💎 {symbol} {analysis['side']} {leverage}x\n"
                f"💵 Entry: ${result.get('entry_price', 0):.4f}\n"
                f"🛑 Stop Loss: ${result.get('stop_loss_price', 0):.4f}",
                parse_mode=ParseMode.HTML
            )
        else:
            logger.error(f"❌ Trade execution failed: {result.get('error', 'Unknown error')}")
            await query.edit_message_text(
                f"❌ Position açılamadı:\n\n{result.get('error', 'Unknown error')}",
                parse_mode=ParseMode.HTML
            )

        # Clear pending trade
        self.pending_trade = None

    # ==================== Multi-Leverage Opportunity ====================

    async def send_multi_leverage_opportunity(
        self,
        symbol: str,
        side: str,
        current_price: float,
        ai_confidence: float,
        ai_models: List[str],
        capital: float,
        analysis: Dict[str, Any],
        market_data: Dict[str, Any]
    ):
        """Send opportunity with multiple leverage options."""

        # Store pending trade for execution when user selects leverage
        self.pending_trade = {
            'symbol': symbol,
            'analysis': analysis,
            'market_data': market_data,
            'timestamp': datetime.now()
        }

        leverages = [2, 3, 5, 10, 15, 20, 25, 30, 35, 50]

        message = f"""
🔍 <b>FIRSAT BULUNDU!</b>

💎 <b>Coin:</b> {symbol}
📈 <b>Yön:</b> {side}
🤖 <b>AI Güven:</b> {ai_confidence*100:.0f}%
💵 <b>Fiyat:</b> ${current_price:.4f}
🤝 <b>Modeller:</b> {', '.join(ai_models)}

━━━━━━━━━━━━━━━━━━━━━
📊 <b>LEVERAGE SEÇENEKLERİ:</b>

"""

        buttons = []

        for leverage in leverages:
            # Calculate for this leverage
            position_size = capital * 0.8  # 80% of capital
            position_value = position_size * leverage

            # Stop-loss calculation (adaptive based on leverage)
            if leverage <= 5:
                stop_loss_percent = 0.15  # 15% for low leverage
            elif leverage <= 10:
                stop_loss_percent = 0.10  # 10% for medium leverage
            elif leverage <= 20:
                stop_loss_percent = 0.05  # 5% for high leverage
            else:
                stop_loss_percent = 0.03  # 3% for extreme leverage

            if side == "LONG":
                stop_loss_price = current_price * (1 - stop_loss_percent)
                liquidation_price = current_price * (1 - 0.95/leverage)
            else:
                stop_loss_price = current_price * (1 + stop_loss_percent)
                liquidation_price = current_price * (1 + 0.95/leverage)

            # Risk assessment
            if leverage <= 5:
                risk = "✅ Düşük"
            elif leverage <= 10:
                risk = "⚠️ Orta"
            elif leverage <= 20:
                risk = "⚠️ Yüksek"
            elif leverage <= 30:
                risk = "🔴 Çok Yüksek"
            else:
                risk = "💀 EXTREMe"

            message += f"""
<b>[{leverage}x]</b> Stop: ${stop_loss_price:.4f} ({stop_loss_percent*100:.0f}%) | Liq: ${liquidation_price:.4f}
      Pozisyon: ${position_value:.0f} | {risk}

"""

            # Add button for this leverage
            buttons.append([InlineKeyboardButton(
                f"{leverage}x - {risk}",
                callback_data=f"leverage_{leverage}x_{symbol.replace('/', '_')}"
            )])

        message += f"""
━━━━━━━━━━━━━━━━━━━━━
⏰ {get_turkey_time().strftime('%H:%M:%S')}

Hangi leverage'ı seçmek istersin?
"""

        # Add cancel button
        buttons.append([InlineKeyboardButton("❌ İptal Et", callback_data="cancel_trade")])

        reply_markup = InlineKeyboardMarkup(buttons)

        # Send message
        chat_id = self.settings.telegram_chat_id
        await self.application.bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )

    async def send_message(self, text: str, parse_mode: str = ParseMode.HTML):
        """Send a simple message to the user."""
        if self.application:
            chat_id = self.settings.telegram_chat_id
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode
            )


# ==================== Singleton ====================

_telegram_bot_instance: Optional['TradingTelegramBot'] = None


async def get_telegram_bot() -> 'TradingTelegramBot':
    """Get or create TradingTelegramBot singleton instance."""
    global _telegram_bot_instance
    if _telegram_bot_instance is None:
        from src.database import get_db_client
        db = await get_db_client()
        _telegram_bot_instance = TradingTelegramBot(db)
        await _telegram_bot_instance.initialize()
    return _telegram_bot_instance
