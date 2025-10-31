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
from src.chart_generator import get_chart_generator

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
        self.application.add_handler(CommandHandler("chart", self.cmd_chart))
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
                InlineKeyboardButton("📈 Grafik Oluştur", callback_data="chart"),
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

<b>📈 Analiz Araçları:</b>
/chart - TradingView benzeri grafik oluştur
/scan - Manuel market tarama

<b>🎮 Bot Kontrol:</b>
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
            logger.info("📋 /positions command called")
            position = await self.db.get_active_position()

            if not position:
                logger.info("No active position found")
                await update.message.reply_text(
                    "❌ Şu anda açık pozisyon bulunmuyor.",
                    parse_mode=ParseMode.HTML
                )
                return

            logger.info(f"Active position found: {position['symbol']} {position['side']}")

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

            logger.info("📍 Sending position info with close button")
            await update.message.reply_text(message, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
            logger.info("✅ Position message sent successfully")

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

    async def cmd_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /chart command - show coin selection menu."""
        logger.info("📈 /chart command called")

        # Popular coins (top 20 from settings)
        popular_coins = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT',
            'XRP/USDT:USDT', 'DOGE/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT',
            'TON/USDT:USDT', 'TRX/USDT:USDT', 'LINK/USDT:USDT', 'UNI/USDT:USDT',
            'AAVE/USDT:USDT', 'MKR/USDT:USDT', 'GRT/USDT:USDT', 'INJ/USDT:USDT',
            'ATOM/USDT:USDT', 'DOT/USDT:USDT', 'POL/USDT:USDT', 'ARB/USDT:USDT'
        ]

        # Create coin selection buttons (4 coins per row)
        keyboard = []
        row = []
        for i, coin in enumerate(popular_coins):
            # Display name (remove :USDT suffix for cleaner look)
            display_name = coin.replace('/USDT:USDT', '')
            # Callback data (encode coin)
            callback_data = f"chart_{coin.replace('/', '_').replace(':', '_')}"

            row.append(InlineKeyboardButton(display_name, callback_data=callback_data))

            # Create new row every 4 coins
            if (i + 1) % 4 == 0:
                keyboard.append(row)
                row = []

        # Add remaining coins
        if row:
            keyboard.append(row)

        reply_markup = InlineKeyboardMarkup(keyboard)

        message = """
📈 <b>GRAFİK OLUŞTURUCU</b>

Ultra profesyonel TradingView benzeri grafik:
• 📊 Candlestick chart (15m timeframe)
• 📍 Destek/Direnç seviyeleri
• 📈 Trend çizgileri (otomatik tespit)
• 📉 EMA 12, 26, 50
• 📊 RSI & MACD indikatörleri
• 💹 Volume analizi

Coin seçin:
"""

        await update.message.reply_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /scan command - triggers immediate market scan."""
        logger.info("🔍 /scan command called - triggering market scan")
        await update.message.reply_text(
            "🔍 Market taraması başlatılıyor...\n\nBu işlem 5-6 dakika sürebilir.",
            parse_mode=ParseMode.HTML
        )

        # Actually trigger the market scan
        try:
            from src.market_scanner import get_market_scanner
            scanner = get_market_scanner()
            await scanner.scan_and_execute()
            logger.info("✅ Market scan completed successfully")
        except Exception as e:
            logger.error(f"Error during market scan: {e}")
            await update.message.reply_text(
                f"❌ Market taraması sırasında hata oluştu: {str(e)[:100]}",
                parse_mode=ParseMode.HTML
            )

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
        elif callback_data == "chart":
            await self.handle_chart_button(query)
        elif callback_data.startswith("chart_"):
            await self.handle_chart_generation(query, callback_data)
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
/chart - TradingView grafik
/scan - Market tara
/startbot - Başlat
/stopbot - Durdur

Detaylı bilgi için /help yazın.
"""
        await query.edit_message_text(message, parse_mode=ParseMode.HTML)

    async def handle_chart_button(self, query):
        """Handle chart button - show coin selection."""
        # Reuse cmd_chart logic
        popular_coins = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT',
            'XRP/USDT:USDT', 'DOGE/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT',
            'TON/USDT:USDT', 'TRX/USDT:USDT', 'LINK/USDT:USDT', 'UNI/USDT:USDT',
            'AAVE/USDT:USDT', 'MKR/USDT:USDT', 'GRT/USDT:USDT', 'INJ/USDT:USDT',
            'ATOM/USDT:USDT', 'DOT/USDT:USDT', 'POL/USDT:USDT', 'ARB/USDT:USDT'
        ]

        keyboard = []
        row = []
        for i, coin in enumerate(popular_coins):
            display_name = coin.replace('/USDT:USDT', '')
            callback_data = f"chart_{coin.replace('/', '_').replace(':', '_')}"
            row.append(InlineKeyboardButton(display_name, callback_data=callback_data))
            if (i + 1) % 4 == 0:
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)

        reply_markup = InlineKeyboardMarkup(keyboard)

        message = """
📈 <b>GRAFİK OLUŞTURUCU</b>

Ultra profesyonel TradingView benzeri grafik:
• 📊 Candlestick chart (15m timeframe)
• 📍 Destek/Direnç seviyeleri
• 📈 Trend çizgileri (otomatik tespit)
• 📉 EMA 12, 26, 50
• 📊 RSI & MACD indikatörleri
• 💹 Volume analizi

Coin seçin:
"""
        await query.edit_message_text(message, parse_mode=ParseMode.HTML, reply_markup=reply_markup)

    async def handle_chart_generation(self, query, callback_data: str):
        """Handle chart generation for selected coin."""
        try:
            # Extract symbol from callback data
            # Format: chart_BTC_USDT_USDT -> BTC/USDT:USDT
            parts = callback_data.replace('chart_', '').split('_')
            if len(parts) == 3:
                symbol = f"{parts[0]}/{parts[1]}:{parts[2]}"
            else:
                await query.edit_message_text("❌ Geçersiz coin formatı")
                return

            logger.info(f"📈 Generating chart for {symbol}")

            # Show loading message
            await query.edit_message_text(
                f"📊 <b>{symbol}</b> için grafik oluşturuluyor...\n\n"
                f"⏳ Bu işlem 10-15 saniye sürebilir...",
                parse_mode=ParseMode.HTML
            )

            # Fetch OHLCV data from exchange
            from src.market_scanner import get_exchange
            exchange = get_exchange()
            ohlcv_data = await exchange.fetch_ohlcv(symbol, '15m', limit=100)

            if not ohlcv_data or len(ohlcv_data) < 50:
                await query.edit_message_text(
                    f"❌ {symbol} için yeterli veri bulunamadı",
                    parse_mode=ParseMode.HTML
                )
                return

            # Generate chart
            chart_generator = get_chart_generator()
            chart_bytes = await chart_generator.generate_chart(
                symbol=symbol,
                ohlcv_data=ohlcv_data,
                timeframe='15m',
                show_indicators=True,
                width=16,
                height=12
            )

            # Send chart as photo
            current_price = ohlcv_data[-1][4]  # Close price
            price_change = ((ohlcv_data[-1][4] - ohlcv_data[0][1]) / ohlcv_data[0][1]) * 100
            emoji = "📈" if price_change >= 0 else "📉"

            caption = f"""
{emoji} <b>{symbol}</b>
💵 Fiyat: ${current_price:.4f} ({price_change:+.2f}%)
📊 Timeframe: 15 dakika (100 mum)
⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}

🎨 TradingView benzeri ultra profesyonel grafik
"""

            # Delete loading message
            await query.message.delete()

            # Send photo
            await self.application.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=chart_bytes,
                caption=caption,
                parse_mode=ParseMode.HTML
            )

            logger.info(f"✅ Chart sent successfully for {symbol}")

        except Exception as e:
            logger.error(f"❌ Error generating chart: {e}")
            await query.edit_message_text(
                f"❌ Grafik oluşturulurken hata oluştu:\n\n{str(e)[:200]}",
                parse_mode=ParseMode.HTML
            )

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
            from decimal import Decimal
            executor = get_trade_executor()

            current_price = Decimal(str(position.get('current_price', position['entry_price'])))
            success = await executor.close_position(
                position=position,
                current_price=current_price,
                close_reason="Manual close via Telegram"
            )

            if success:
                pnl = float(position.get('unrealized_pnl_usd', 0))
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
                    "❌ Pozisyon kapatılamadı. Lütfen tekrar deneyin.",
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
                await query.answer("❌ Hata")
                await query.message.reply_text(
                    f"❌ Trade execution failed: {str(e)}",
                    parse_mode=ParseMode.HTML
                )
                self.pending_trade = None

    async def _execute_pending_trade(self, leverage: int, query):
        """Execute the pending trade with selected leverage."""
        from decimal import Decimal

        if not self.pending_trade:
            return

        symbol = self.pending_trade['symbol']
        analysis = self.pending_trade['analysis']
        market_data = self.pending_trade['market_data']

        logger.info(f"📊 Executing trade: {symbol} {analysis['side']} with {leverage}x leverage")

        # Update leverage in analysis
        analysis['suggested_leverage'] = leverage

        # Get current capital from database
        from src.database import get_db_client
        db = await get_db_client()
        config = await db.get_trading_config()
        capital = float(config.get('current_capital', 100))

        # BINANCE FUTURES CONSTANTS (SAME as in send_multi_leverage_opportunity)
        TRADING_FEE_RATE = 0.0004  # 0.04% taker fee
        MAX_LOSS_USD = 10.0  # Maximum loss per trade in USD

        # Calculate stop-loss based on MAX $10 LOSS LIMIT
        position_size = capital * 0.8  # 80% of capital (initial margin)
        position_value = position_size * leverage  # Notional value
        total_fees = position_value * TRADING_FEE_RATE * 2  # Entry + exit fees

        max_loss_after_fees = MAX_LOSS_USD - total_fees
        if max_loss_after_fees <= 0:
            max_loss_after_fees = 1.0  # Fallback to $1 minimum

        # Calculate max price movement percentage
        max_price_movement_pct = max_loss_after_fees / (leverage * position_size)

        # Cap between 5-10% for risk manager compatibility
        stop_loss_percent = min(10.0, max(5.0, max_price_movement_pct * 100))
        analysis['stop_loss_percent'] = stop_loss_percent

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
            await query.answer("❌ Trade reddedildi")
            await query.message.reply_text(
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

        # Execute the trade using open_position
        executor = get_trade_executor()

        # Prepare trade_params for open_position
        trade_params = {
            'symbol': symbol,
            'side': analysis['side'],
            'leverage': leverage,
            'stop_loss_percent': analysis['stop_loss_percent'],
            'current_price': market_data['current_price']
        }

        success = await executor.open_position(trade_params, analysis, market_data)

        if success:
            logger.info(f"✅ Trade executed successfully")

            # Calculate stop loss price for display
            entry_price = Decimal(str(market_data['current_price']))
            stop_loss_pct = Decimal(str(analysis['stop_loss_percent'])) / 100

            if analysis['side'] == 'LONG':
                stop_loss_price = entry_price * (1 - stop_loss_pct)
            else:
                stop_loss_price = entry_price * (1 + stop_loss_pct)

            await query.answer("✅ Position açıldı!")
            await query.message.reply_text(
                f"✅ Position açıldı!\n\n"
                f"💎 {symbol} {analysis['side']} {leverage}x\n"
                f"💵 Entry: ${float(entry_price):.4f}\n"
                f"🛑 Stop Loss: ${float(stop_loss_price):.4f}",
                parse_mode=ParseMode.HTML
            )
        else:
            logger.error(f"❌ Trade execution failed or position closed immediately")
            await query.answer("❌ Trade başarısız")
            await query.message.reply_text(
                f"❌ Trade başarısız oldu veya pozisyon hemen kapatıldı.\n\n"
                f"Yukarıdaki bildirimleri kontrol edin (slippage, risk limitleri vb.).",
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

        # BINANCE FUTURES CONSTANTS
        TRADING_FEE_RATE = 0.0004  # 0.04% taker fee
        MAINTENANCE_MARGIN_RATE = 0.004  # 0.4% for positions < 50k USDT
        MAX_LOSS_USD = 10.0  # Maximum loss per trade in USD

        for leverage in leverages:
            # Calculate for this leverage
            position_size = capital * 0.8  # 80% of capital (initial margin)
            position_value = position_size * leverage  # Notional value

            # Maintenance Margin (minimum to avoid liquidation)
            maintenance_margin = position_value * MAINTENANCE_MARGIN_RATE

            # Trading Fees (entry + exit)
            total_fees = position_value * TRADING_FEE_RATE * 2

            # Calculate stop-loss based on MAX $10 LOSS LIMIT
            # Real Loss = (Price Movement % × Leverage × Position Size) + Fees
            # We want: Real Loss <= $10
            # So: Price Movement % <= (10 - Fees) / (Leverage × Position Size)

            max_loss_after_fees = MAX_LOSS_USD - total_fees
            if max_loss_after_fees <= 0:
                # Fees alone exceed $10, skip this leverage
                continue

            # Calculate max price movement percentage
            max_price_movement_pct = max_loss_after_fees / (leverage * position_size)

            # Cap between 5-10% for risk manager compatibility
            stop_loss_percent = min(0.10, max(0.05, max_price_movement_pct))

            # Calculate prices
            if side == "LONG":
                stop_loss_price = current_price * (1 - stop_loss_percent)
                # Real Binance liquidation formula
                liquidation_price = current_price * (1 - (position_size - maintenance_margin) / position_value)
                take_profit_price = current_price * (1 + stop_loss_percent * 2)
            else:
                stop_loss_price = current_price * (1 + stop_loss_percent)
                # Real Binance liquidation formula for SHORT
                liquidation_price = current_price * (1 + (position_size - maintenance_margin) / position_value)
                take_profit_price = current_price * (1 - stop_loss_percent * 2)

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
                risk = "💀 EXTREME"

            # Calculate REAL losses/profits (including all costs)
            price_loss = stop_loss_percent * leverage * position_size
            real_max_loss = price_loss + total_fees  # Total loss including fees

            price_profit = stop_loss_percent * 2 * leverage * position_size
            real_max_profit = price_profit - total_fees  # Profit after fees

            message += f"""
<b>[{leverage}x Kaldıraç]</b> {risk}
├ 📍 Giriş: ${current_price:.4f}
├ 🛑 Stop-Loss: ${stop_loss_price:.4f} ({stop_loss_percent*100:.2f}%)
├ 🎯 Take-Profit: ${take_profit_price:.4f} ({stop_loss_percent*2*100:.2f}%)
├ ⚠️  Liquidation: ${liquidation_price:.4f}
├ 💰 Pozisyon: ${position_value:.0f} USDT ({leverage}x)
├ 💵 Teminat: ${position_size:.2f} USDT
├ 🏦 Komisyon: ${total_fees:.2f} USDT (giriş+çıkış)
├ 📉 GERÇEK Max Kayıp: ${real_max_loss:.2f} USDT
└ 📈 GERÇEK Hedef Kar: ${real_max_profit:.2f} USDT

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
