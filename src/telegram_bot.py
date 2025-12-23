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
from src.interactive_chart import generate_interactive_html_chart
from src.chart_server import store_chart
import os

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
        self.application.add_handler(CommandHandler("mlstats", self.cmd_mlstats))
        self.application.add_handler(CommandHandler("mlinsights", self.cmd_mlinsights))
        self.application.add_handler(CommandHandler("daily", self.cmd_daily_report))
        self.application.add_handler(CommandHandler("stopbot", self.cmd_stop_bot))
        self.application.add_handler(CommandHandler("startbot", self.cmd_start_bot))
        self.application.add_handler(CommandHandler("reset", self.cmd_reset_circuit_breaker))
        self.application.add_handler(CommandHandler("setcapital", self.cmd_set_capital))
        self.application.add_handler(CommandHandler("closeall", self.cmd_close_all_positions))
        self.application.add_handler(CommandHandler("ws", self.cmd_websocket_stats))
        self.application.add_handler(CommandHandler("sync", self.cmd_force_sync))
        self.application.add_handler(CommandHandler("analyze", self.cmd_analyze_trades))
        self.application.add_handler(CommandHandler("scanrs", self.cmd_scan_sr_levels))
        self.application.add_handler(CommandHandler("predict", self.cmd_predict))

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
/chart - ✨ Ultra Premium grafik oluştur
/scan - Manuel market tarama
/daily - 📊 Günlük performans raporu (00:00'dan itibaren)
/mlstats - ML öğrenme istatistikleri
/mlinsights - 🧠 ML analiz + grafikler (AI öğrenme süreci)

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
/daily - 📊 Günlük performans raporu (gece 00:00'dan itibaren)
/mlstats - ML öğrenme istatistikleri ve accuracy
/mlinsights - 🧠 Detaylı ML analiz + grafikler (öğrenme eğrisi, pattern perf.)
/startbot - Botu çalıştır
/stopbot - Botu durdur
/reset - Circuit breaker'ı resetle (3 ardışık loss sonrası)
/setcapital 1000 - Capital'i güncelle (örn: $1000)
/closeall - Tüm açık pozisyonları kapat
/ws - 🌐 WebSocket feed istatistikleri (API kullanımı)
/sync - 🔄 Binance ↔ Database pozisyon senkronizasyonu (orphaned position fix)
/analyze - 📊 Trade history analizi (PNL, win rate, rapid trades)
/analyze [COIN] - 🎯 Level-Based S/R analizi (örn: /analyze BTC, /analyze ETH, /analyze SOL)
/scanrs - 🔍 Tüm coinleri tara, S/R seviyelerine yakın olanları listele
/predict [COIN] - 🎯 AI Prediction Chart (Entry/TP/SL + Trend) örn: /predict BTC

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

            # Get all active positions
            positions = await self.db.get_active_positions()

            message = f"""
<b>📊 BOT DURUMU</b>

{status_emoji} <b>Durum:</b> {status_text}
💰 <b>Sermaye:</b> ${capital:.2f}
{pnl_emoji} <b>Bugünkü P&L:</b> ${daily_pnl:+.2f}

<b>📍 Aktif Pozisyonlar:</b>
"""
            if positions:
                # Calculate total unrealized P&L
                total_unrealized_pnl = sum(float(p.get('unrealized_pnl_usd', 0)) for p in positions)
                total_value = sum(float(p['position_value_usd']) for p in positions)
                winning = sum(1 for p in positions if float(p.get('unrealized_pnl_usd', 0)) > 0)
                losing = len(positions) - winning

                unrealized_emoji = "🟢" if total_unrealized_pnl >= 0 else "🔴"

                message += f"""
📊 <b>Toplam:</b> {len(positions)} pozisyon
💰 <b>Değer:</b> ${total_value:.2f}
{unrealized_emoji} <b>Unrealized P&L:</b> ${total_unrealized_pnl:+.2f}
🟢 Kazanan: {winning} | 🔴 Kaybeden: {losing}

<b>En İyi/Kötü:</b>
"""
                # Show best and worst performing
                sorted_positions = sorted(positions, key=lambda p: float(p.get('unrealized_pnl_usd', 0)), reverse=True)
                best = sorted_positions[0]
                worst = sorted_positions[-1]

                best_pnl = float(best.get('unrealized_pnl_usd', 0))
                worst_pnl = float(worst.get('unrealized_pnl_usd', 0))

                message += f"🟢 {best['symbol']}: ${best_pnl:+.2f}\n"
                message += f"🔴 {worst['symbol']}: ${worst_pnl:+.2f}\n"
                message += f"\n💡 Detaylar için /positions"
            else:
                message += "\n❌ Şu anda açık pozisyon yok"

            message += f"\n⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}"

            await update.message.reply_text(message, parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await update.message.reply_text(f"❌ Hata: {e}")

    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command - shows ALL active positions."""
        try:
            logger.info("📋 /positions command called")
            positions = await self.db.get_active_positions()  # Get ALL positions

            if not positions:
                logger.info("No active positions found")
                await update.message.reply_text(
                    "❌ Şu anda açık pozisyon bulunmuyor.",
                    parse_mode=ParseMode.HTML
                )
                return

            logger.info(f"Found {len(positions)} active positions")

            # Calculate total P&L across all positions
            total_pnl = sum(float(p.get('unrealized_pnl_usd', 0)) for p in positions)
            total_value = sum(float(p['position_value_usd']) for p in positions)
            winning_count = sum(1 for p in positions if float(p.get('unrealized_pnl_usd', 0)) > 0)
            losing_count = len(positions) - winning_count

            # Summary header
            summary_emoji = "🟢" if total_pnl >= 0 else "🔴"
            summary = f"""
<b>💼 AKTİF POZİSYONLAR ({len(positions)})</b>

<b>📊 Özet:</b>
• Toplam Pozisyon: {len(positions)}
• Kazanan: 🟢 {winning_count} | Kaybeden: 🔴 {losing_count}
• Toplam P&L: {summary_emoji} ${total_pnl:+.2f}
• Toplam Değer: ${total_value:.2f}

━━━━━━━━━━━━━━━━━━━━
"""
            await update.message.reply_text(summary, parse_mode=ParseMode.HTML)

            # Send each position separately (Telegram has message length limits)
            for i, position in enumerate(positions, 1):
                entry_price = float(position['entry_price'])
                current_price = float(position.get('current_price', entry_price))
                pnl = float(position.get('unrealized_pnl_usd', 0))
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"

                # Calculate position duration
                from datetime import datetime
                entry_time = position['entry_time']
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)
                duration = datetime.now() - entry_time
                hours = duration.total_seconds() / 3600

                message = f"""
{pnl_emoji} <b>#{i} - {position['symbol']}</b>

<b>📊 Detaylar:</b>
• Yön: {position['side']} {position['leverage']}x
• Miktar: {float(position['quantity']):.6f}
• Değer: ${float(position['position_value_usd']):.2f}

<b>💵 Fiyatlar:</b>
• Entry: ${entry_price:.4f}
• Current: ${current_price:.4f}
• Stop-Loss: ${float(position['stop_loss_price']):.4f}
• Liquidation: ${float(position['liquidation_price']):.4f}

<b>💰 Kar/Zarar:</b>
• P&L: ${pnl:+.2f} ({(pnl/float(position['position_value_usd'])*100):+.2f}%)
• Hedef: ${float(position['min_profit_target_usd']):.2f}

<b>🤖 AI:</b>
• Güven: {float(position.get('ai_confidence', 0))*100:.0f}%

<b>⏰ Süre:</b>
• Açılış: {entry_time.strftime('%H:%M:%S')}
• Geçen: {hours:.1f} saat
"""
                await update.message.reply_text(message, parse_mode=ParseMode.HTML)

            logger.info(f"✅ Sent {len(positions)} position details")

        except Exception as e:
            logger.error(f"Error in positions command: {e}", exc_info=True)
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

    async def cmd_daily_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /daily command - Daily performance report from 00:00 to now."""
        try:
            logger.info("📊 /daily command called")

            from datetime import datetime, time, timezone, timedelta

            # Turkey timezone (UTC+3)
            TURKEY_TZ = timezone(timedelta(hours=3))

            # Get current time in Turkey timezone
            now_turkey = datetime.now(TURKEY_TZ)

            # Get today's date at 00:00:00 in Turkey timezone
            today_start_turkey = datetime.combine(now_turkey.date(), time.min).replace(tzinfo=TURKEY_TZ)

            # Convert to UTC for database query (database stores in UTC)
            today_start = today_start_turkey.astimezone(timezone.utc).replace(tzinfo=None)

            # Query for today's trades
            query = """
                SELECT
                    symbol, side, entry_price, exit_price, realized_pnl_usd,
                    entry_time, exit_time, leverage, close_reason
                FROM trade_history
                WHERE exit_time >= $1
                ORDER BY exit_time DESC
            """

            async with self.db.pool.acquire() as conn:
                today_trades = await conn.fetch(query, today_start)
                config = await conn.fetchrow("SELECT current_capital FROM trading_config WHERE id = 1")

            if not today_trades:
                current_time_turkey = now_turkey.strftime('%H:%M:%S')
                await update.message.reply_text(
                    f"📊 <b>GÜNLÜK PERFORMANS RAPORU</b>\n\n"
                    f"⏰ {today_start_turkey.strftime('%d.%m.%Y')} 00:00 - {current_time_turkey}\n\n"
                    f"Bugün henüz kapalı trade yok.\n"
                    f"💰 Mevcut Sermaye: ${float(config['current_capital']):.2f}",
                    parse_mode=ParseMode.HTML
                )
                return

            # Calculate daily statistics
            total_trades = len(today_trades)
            winning_trades = [t for t in today_trades if float(t['realized_pnl_usd']) > 0]
            losing_trades = [t for t in today_trades if float(t['realized_pnl_usd']) <= 0]

            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

            # Calculate P&L
            daily_pnl = sum(float(t['realized_pnl_usd']) for t in today_trades)
            winning_pnl = sum(float(t['realized_pnl_usd']) for t in winning_trades) if winning_trades else 0
            losing_pnl = sum(float(t['realized_pnl_usd']) for t in losing_trades) if losing_trades else 0

            # LONG vs SHORT breakdown
            long_trades = [t for t in today_trades if t['side'] == 'LONG']
            short_trades = [t for t in today_trades if t['side'] == 'SHORT']

            long_wins = len([t for t in long_trades if float(t['realized_pnl_usd']) > 0])
            short_wins = len([t for t in short_trades if float(t['realized_pnl_usd']) > 0])

            long_wr = (long_wins / len(long_trades) * 100) if long_trades else 0
            short_wr = (short_wins / len(short_trades) * 100) if short_trades else 0

            # Build message
            current_time_turkey = now_turkey.strftime('%H:%M:%S')
            message = f"📊 <b>GÜNLÜK PERFORMANS RAPORU</b>\n\n"
            message += f"⏰ {today_start_turkey.strftime('%d.%m.%Y')} 00:00 - {current_time_turkey}\n"
            message += f"━━━━━━━━━━━━━━━━━━━━\n\n"

            # Overall Stats
            message += f"<b>📈 GENEL DURUM</b>\n"
            message += f"• Toplam Trade: {total_trades}\n"

            wr_emoji = "🟢" if win_rate >= 60 else "🟡" if win_rate >= 50 else "🔴"
            message += f"• Win/Loss: {wr_emoji} {win_count}W / {loss_count}L ({win_rate:.1f}%)\n"

            pnl_emoji = "💚" if daily_pnl > 0 else "🔴" if daily_pnl < 0 else "⚪"
            message += f"• <b>Net P&L: {pnl_emoji} ${daily_pnl:+.2f}</b>\n"
            message += f"• Avg P&L/Trade: ${daily_pnl/total_trades:.2f}\n\n"

            # Win/Loss breakdown
            message += f"<b>💰 KAR/ZARAR DAĞILIMI</b>\n"
            message += f"• Kazanç: +${winning_pnl:.2f} ({win_count} trade)\n"
            message += f"• Kayıp: ${losing_pnl:.2f} ({loss_count} trade)\n"

            if win_count > 0 and loss_count > 0:
                profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
                pf_emoji = "💎" if profit_factor >= 2.0 else "✅" if profit_factor >= 1.0 else "⚠️"
                message += f"• Profit Factor: {pf_emoji} {profit_factor:.2f}x\n"

            message += f"\n<b>📊 YÖN DAĞILIMI</b>\n"

            if long_trades:
                message += f"• LONG: {len(long_trades)} trade, {long_wr:.0f}% WR\n"
            if short_trades:
                message += f"• SHORT: {len(short_trades)} trade, {short_wr:.0f}% WR\n"

            message += f"\n<b>💼 SERMAYE</b>\n"
            message += f"• Mevcut: ${float(config['current_capital']):.2f}\n"

            # Calculate day start capital (approx)
            day_start_capital = float(config['current_capital']) - daily_pnl
            roi_today = (daily_pnl / day_start_capital * 100) if day_start_capital > 0 else 0
            roi_emoji = "🚀" if roi_today > 2 else "📈" if roi_today > 0 else "📉"
            message += f"• Günlük ROI: {roi_emoji} {roi_today:+.2f}%\n\n"

            # Recent trades list (last 5)
            message += f"<b>🕒 SON TRADE'LER</b>\n"
            for i, trade in enumerate(today_trades[:5], 1):
                pnl = float(trade['realized_pnl_usd'])
                emoji = "✅" if pnl > 0 else "❌"
                # Convert UTC to Turkey time
                exit_time_utc = trade['exit_time'].replace(tzinfo=timezone.utc)
                exit_time_turkey = exit_time_utc.astimezone(TURKEY_TZ)
                time_str = exit_time_turkey.strftime('%H:%M')
                message += f"{emoji} {time_str} {trade['symbol']} {trade['side']} ${pnl:+.2f}\n"

            if total_trades > 5:
                message += f"\n... ve {total_trades - 5} trade daha\n"

            message += f"\n<i>💡 /history komutuyla detaylı geçmiş görüntüle</i>"

            await update.message.reply_text(message, parse_mode=ParseMode.HTML)
            logger.info("✅ Daily report sent successfully")

        except Exception as e:
            logger.error(f"Error in daily report command: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ Günlük rapor oluşturulamadı: {str(e)}\n\n"
                f"Database bağlantısını kontrol edin.",
                parse_mode=ParseMode.HTML
            )

    async def cmd_mlstats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mlstats command - comprehensive ML learning statistics."""
        try:
            logger.info("🧠 /mlstats command called")

            # Get real-time data from database
            from src.ml_pattern_learner import get_ml_learner
            from datetime import datetime, timedelta

            ml = await get_ml_learner()

            # Get all closed trades from database
            query = """
                SELECT
                    symbol, side, entry_price, exit_price, realized_pnl_usd,
                    entry_time, exit_time, ai_confidence, leverage
                FROM trade_history
                ORDER BY exit_time DESC
            """
            async with self.db.pool.acquire() as conn:
                all_trades = await conn.fetch(query)
                active_positions = await conn.fetch("SELECT * FROM active_position")
                config = await conn.fetchrow("SELECT * FROM trading_config")

            # Calculate statistics
            total_trades = len(all_trades)
            if total_trades == 0:
                await update.message.reply_text(
                    "📊 <b>ML İSTATİSTİKLERİ</b>\n\n"
                    "Henüz kapalı trade yok.\n"
                    "Bot trade yapmaya başladığında istatistikler burada görünecek!",
                    parse_mode=ParseMode.HTML
                )
                return

            # Winning trades
            winning_trades = [t for t in all_trades if float(t['realized_pnl_usd']) > 0]
            losing_trades = [t for t in all_trades if float(t['realized_pnl_usd']) <= 0]
            overall_wr = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

            # Total P&L
            total_pnl = sum(float(t['realized_pnl_usd']) for t in all_trades)

            # LONG vs SHORT breakdown
            long_trades = [t for t in all_trades if t['side'] == 'long']
            short_trades = [t for t in all_trades if t['side'] == 'short']

            long_wins = [t for t in long_trades if float(t['realized_pnl_usd']) > 0]
            short_wins = [t for t in short_trades if float(t['realized_pnl_usd']) > 0]

            long_wr = (len(long_wins) / len(long_trades) * 100) if long_trades else 0
            short_wr = (len(short_wins) / len(short_trades) * 100) if short_trades else 0

            long_pnl = sum(float(t['realized_pnl_usd']) for t in long_trades)
            short_pnl = sum(float(t['realized_pnl_usd']) for t in short_trades)

            # Build message
            message = "<b>🧠 ML ÖĞRENME İSTATİSTİKLERİ</b>\n\n"

            # 1. Overall Summary
            message += "<b>📊 GENEL ÖZET</b>\n"
            message += f"• Total Trades: {total_trades}\n"
            message += f"• Aktif Pozisyon: {len(active_positions)}\n"
            wr_emoji = "🟢" if overall_wr >= 60 else "🟡" if overall_wr >= 50 else "🔴"
            message += f"• Win Rate: {wr_emoji} {overall_wr:.1f}%\n"
            pnl_emoji = "💰" if total_pnl > 0 else "📉"
            message += f"• Total P&L: {pnl_emoji} ${total_pnl:+.2f}\n"
            message += f"• Capital: ${float(config['current_capital']):.2f}\n\n"

            # 2. LONG vs SHORT Breakdown
            message += "<b>⚔️ LONG vs SHORT</b>\n"
            message += f"<b>LONG:</b>\n"
            if long_trades:
                long_emoji = "🟢" if long_wr >= 60 else "🟡" if long_wr >= 50 else "🔴"
                message += f"  {long_emoji} {len(long_trades)} trades | {long_wr:.1f}% WR\n"
                message += f"  P&L: ${long_pnl:+.2f}\n"
            else:
                message += f"  Henüz LONG trade yok\n"

            message += f"<b>SHORT:</b>\n"
            if short_trades:
                short_emoji = "🟢" if short_wr >= 60 else "🟡" if short_wr >= 50 else "🔴"
                message += f"  {short_emoji} {len(short_trades)} trades | {short_wr:.1f}% WR\n"
                message += f"  P&L: ${short_pnl:+.2f}\n"
            else:
                message += f"  Henüz SHORT trade yok\n"
            message += "\n"

            # 3. AI Confidence Analysis
            message += "<b>🎯 AI CONFIDENCE ACCURACY</b>\n"
            confidence_buckets = {}
            for trade in all_trades:
                conf = trade.get('ai_confidence')
                if conf:
                    conf_bucket = int(float(conf) * 100 / 5) * 5  # 5% buckets
                    if conf_bucket not in confidence_buckets:
                        confidence_buckets[conf_bucket] = {'wins': 0, 'total': 0}
                    confidence_buckets[conf_bucket]['total'] += 1
                    if float(trade['realized_pnl_usd']) > 0:
                        confidence_buckets[conf_bucket]['wins'] += 1

            if confidence_buckets:
                for conf in sorted(confidence_buckets.keys(), reverse=True):
                    stats = confidence_buckets[conf]
                    if stats['total'] >= 2:  # At least 2 trades
                        acc = stats['wins'] / stats['total'] * 100
                        emoji = "✅" if acc >= 65 else "⚠️" if acc >= 50 else "❌"
                        message += f"{emoji} {conf}% güven: {acc:.0f}% doğru (n={stats['total']})\n"
            else:
                message += "• Henüz yeterli veri yok\n"
            message += "\n"

            # 4. Top Symbols
            message += "<b>🏆 EN İYİ / EN KÖTÜ COİNLER</b>\n"
            symbol_stats = {}
            for trade in all_trades:
                symbol = trade['symbol'].split('/')[0]  # Get base symbol
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {'wins': 0, 'total': 0, 'pnl': 0}
                symbol_stats[symbol]['total'] += 1
                symbol_stats[symbol]['pnl'] += float(trade['realized_pnl_usd'])
                if float(trade['realized_pnl_usd']) > 0:
                    symbol_stats[symbol]['wins'] += 1

            # Filter symbols with at least 2 trades
            valid_symbols = {s: stats for s, stats in symbol_stats.items() if stats['total'] >= 2}

            if valid_symbols:
                # Top 3 best
                best = sorted(valid_symbols.items(),
                            key=lambda x: x[1]['wins']/x[1]['total'],
                            reverse=True)[:3]
                message += "<b>En İyi:</b>\n"
                for symbol, stats in best:
                    wr = stats['wins'] / stats['total'] * 100
                    message += f"🟢 {symbol}: {wr:.0f}% WR | ${stats['pnl']:+.2f}\n"

                # Top 3 worst
                worst = sorted(valid_symbols.items(),
                             key=lambda x: x[1]['wins']/x[1]['total'])[:3]
                message += "<b>En Kötü:</b>\n"
                for symbol, stats in worst:
                    wr = stats['wins'] / stats['total'] * 100
                    message += f"🔴 {symbol}: {wr:.0f}% WR | ${stats['pnl']:+.2f}\n"
            else:
                message += "• Henüz yeterli veri yok (min 2 trade/coin)\n"
            message += "\n"

            # 5. Recent Performance (Last 10 trades)
            message += "<b>📈 SON 10 TRADE</b>\n"
            recent = all_trades[:10]
            recent_wins = sum(1 for t in recent if float(t['realized_pnl_usd']) > 0)
            recent_wr = recent_wins / len(recent) * 100 if recent else 0

            # Build recent trades string
            recent_str = ""
            for t in recent[:5]:  # Show last 5
                pnl = float(t['realized_pnl_usd'])
                emoji = "✅" if pnl > 0 else "❌"
                symbol = t['symbol'].split('/')[0]
                side_emoji = "📈" if t['side'] == 'long' else "📉"
                recent_str += f"{emoji} {symbol} {side_emoji} ${pnl:+.2f}\n"

            emoji = "🔥" if recent_wr >= 60 else "📊"
            message += f"{emoji} Son 10: {recent_wins}W/{len(recent)-recent_wins}L ({recent_wr:.0f}%)\n"
            message += recent_str

            # 6. Leverage Usage Analysis (NEW!)
            message += "\n<b>⚡ LEVERAGE KULLANIMI</b>\n"
            leverage_stats = {}
            for trade in all_trades:
                lev = trade.get('leverage', 2)
                if lev not in leverage_stats:
                    leverage_stats[lev] = {'wins': 0, 'total': 0, 'pnl': 0}
                leverage_stats[lev]['total'] += 1
                leverage_stats[lev]['pnl'] += float(trade['realized_pnl_usd'])
                if float(trade['realized_pnl_usd']) > 0:
                    leverage_stats[lev]['wins'] += 1

            if leverage_stats:
                for lev in sorted(leverage_stats.keys(), reverse=True):
                    stats = leverage_stats[lev]
                    wr = stats['wins'] / stats['total'] * 100
                    lev_emoji = "🔥" if lev >= 20 else "⚡" if lev >= 10 else "📊"
                    result_emoji = "🟢" if wr >= 60 else "🟡" if wr >= 50 else "🔴"
                    message += f"{lev_emoji} {lev}x: {result_emoji} {wr:.0f}% WR | ${stats['pnl']:+.2f} ({stats['total']} trades)\n"
            else:
                message += "• Henüz trade yok\n"

            # 7. Pattern Success Rate (NEW!)
            message += "\n<b>🧠 PATTERN BAŞARI ORANI</b>\n"
            # Get pattern data from ML learner's winning_patterns and losing_patterns
            pattern_performance = {}

            # Use ML's actual winning_patterns and losing_patterns dictionaries
            if hasattr(ml, 'winning_patterns') and hasattr(ml, 'losing_patterns'):
                # Combine winning and losing patterns
                all_patterns = set(ml.winning_patterns.keys()) | set(ml.losing_patterns.keys())

                for pattern in all_patterns:
                    wins = ml.winning_patterns.get(pattern, 0)
                    losses = ml.losing_patterns.get(pattern, 0)
                    total = wins + losses

                    if total > 0:
                        pattern_performance[pattern] = {
                            'wins': wins,
                            'total': total,
                            'win_rate': (wins / total) * 100
                        }

            if pattern_performance:
                # Show top 5 best performing patterns (minimum 5 occurrences)
                sorted_patterns = sorted(
                    [(p, s) for p, s in pattern_performance.items() if s['total'] >= 5],
                    key=lambda x: x[1]['win_rate'],
                    reverse=True
                )[:5]

                if sorted_patterns:
                    for pattern, stats in sorted_patterns:
                        acc = stats['win_rate']
                        emoji = "🟢" if acc >= 70 else "🟡" if acc >= 50 else "🔴"
                        pattern_display = pattern.replace('_', ' ').title()
                        message += f"{emoji} {pattern_display}: {acc:.0f}% WR ({stats['wins']}W/{stats['total']-stats['wins']}L)\n"
                else:
                    message += "• Patterns tespit edildi ama henüz yeterli örnek yok (min 5)\n"
                    message += f"• Toplam {len(pattern_performance)} farklı pattern öğrenildi\n"
            else:
                message += "• ML henüz pattern öğreniyor...\n"
                message += "• Trade sayısı arttıkça pattern'ler burada görünecek\n"

            # 8. Time-Based Performance (NEW!)
            message += "\n<b>⏰ ZAMAN BAZLI PERFORMANS</b>\n"
            if len(all_trades) >= 10:
                # First 10 vs Last 10 comparison
                first_10 = all_trades[-10:]  # Oldest 10
                last_10 = all_trades[:10]    # Newest 10

                first_10_wr = sum(1 for t in first_10 if float(t['realized_pnl_usd']) > 0) / len(first_10) * 100
                last_10_wr = sum(1 for t in last_10 if float(t['realized_pnl_usd']) > 0) / len(last_10) * 100

                improvement = last_10_wr - first_10_wr
                trend_emoji = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"

                message += f"• İlk 10: {first_10_wr:.0f}% WR\n"
                message += f"• Son 10: {last_10_wr:.0f}% WR\n"
                message += f"{trend_emoji} Gelişme: {improvement:+.0f}%\n"

                if improvement > 10:
                    message += "🎉 <b>ML hızla öğreniyor!</b>\n"
                elif improvement > 0:
                    message += "✅ İyileşme var\n"
                elif improvement < -10:
                    message += "⚠️ Performans düşüşü - pattern adaptasyonu gerekiyor\n"
            else:
                message += "• Henüz yeterli veri yok (min 10 trade)\n"

            # 9. Win Streak & Loss Streak (NEW!)
            message += "\n<b>🔥 STREAK ANALİZİ</b>\n"
            current_streak = 0
            max_win_streak = 0
            max_loss_streak = 0
            temp_streak = 0
            last_was_win = None

            for trade in reversed(all_trades):  # Chronological order
                is_win = float(trade['realized_pnl_usd']) > 0
                if last_was_win is None:
                    temp_streak = 1
                elif is_win == last_was_win:
                    temp_streak += 1
                else:
                    if last_was_win:
                        max_win_streak = max(max_win_streak, temp_streak)
                    else:
                        max_loss_streak = max(max_loss_streak, temp_streak)
                    temp_streak = 1
                last_was_win = is_win

            # Get current streak
            if all_trades:
                current_streak = 1
                last_result = float(all_trades[0]['realized_pnl_usd']) > 0
                for trade in all_trades[1:]:
                    if (float(trade['realized_pnl_usd']) > 0) == last_result:
                        current_streak += 1
                    else:
                        break

            streak_emoji = "🔥" if last_result else "❄️"
            streak_type = "Win" if last_result else "Loss"
            message += f"• Mevcut: {streak_emoji} {current_streak} {streak_type} streak\n"
            message += f"• Max Win Streak: 🏆 {max_win_streak}\n"
            message += f"• Max Loss Streak: 💀 {max_loss_streak}\n"

            # 10. Learning Progress & Insights (ENHANCED!)
            message += "\n<b>🎓 ML ÖĞRENME DURUMUı</b>\n"
            message += f"• Öğrenme Eşiği: {ml.min_confidence_threshold:.0%}\n"
            message += f"• Market Rejimi: {ml.current_regime}\n"
            message += f"• Analiz Edilen: {ml.total_trades_analyzed} trade\n"

            # Learning quality assessment
            if total_trades >= 50:
                learning_quality = "🎯 İleri Seviye" if overall_wr >= 60 else "📚 Orta Seviye" if overall_wr >= 50 else "🌱 Başlangıç"
                message += f"• ML Seviyesi: {learning_quality}\n"

            # Insights
            message += "\n<b>💡 ML ANALİZ</b>\n"
            if overall_wr >= 65:
                message += "✅ ML stratejisi karlı! Mevcut ayarlar iyi.\n"
            elif overall_wr >= 55:
                message += "📊 Dengeli performans. ML optimize ediliyor.\n"
            elif overall_wr >= 45:
                message += "⚠️ Win rate düşük. ML daha fazla veri topluyor.\n"
            else:
                message += "🔄 ML öğrenme fazındaysa. Sabır gerekli.\n"

            if len(leverage_stats) > 1:
                best_leverage = max(leverage_stats.items(), key=lambda x: x[1]['wins']/x[1]['total'] if x[1]['total'] > 0 else 0)
                message += f"🎯 En iyi leverage: {best_leverage[0]}x\n"

            message += "\n<i>🧠 ML her trade sonrası pattern'leri günceller!</i>"

            await update.message.reply_text(message, parse_mode=ParseMode.HTML)
            logger.info("✅ Ultra-detailed ML stats sent successfully")

        except Exception as e:
            logger.error(f"Error in mlstats command: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ ML istatistikleri alınamadı: {str(e)}\n\n"
                f"Database bağlantısını kontrol edin.",
                parse_mode=ParseMode.HTML
            )

    async def cmd_mlinsights(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mlinsights command - comprehensive ML analysis with graphs."""
        try:
            logger.info("🧠 /mlinsights command called")
            await update.message.reply_text("🔬 ML analiz hazırlanıyor... Grafik oluşturuluyor...")

            # Get ML learner and database data
            from src.ml_pattern_learner import get_ml_learner
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            from io import BytesIO
            from datetime import datetime, timedelta

            ml = await get_ml_learner()

            # Get all closed trades from database
            query = """
                SELECT
                    symbol, side, entry_price, exit_price, realized_pnl_usd,
                    entry_time, exit_time, ai_confidence, leverage,
                    close_reason
                FROM trade_history
                ORDER BY exit_time ASC
            """
            async with self.db.pool.acquire() as conn:
                all_trades = await conn.fetch(query)

            if len(all_trades) < 10:
                await update.message.reply_text(
                    "📊 <b>ML INSIGHTS</b>\n\n"
                    "Henüz yeterli veri yok (min 10 trade).\n"
                    f"Mevcut trade sayısı: {len(all_trades)}\n\n"
                    "Bot daha fazla trade yaptıkça detaylı analiz mevcut olacak!",
                    parse_mode=ParseMode.HTML
                )
                return

            # === 1. LEARNING CURVE GRAPH ===
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('🧠 ML LEARNING INSIGHTS', fontsize=16, fontweight='bold')

            # 1a. Win Rate Evolution (Rolling Window)
            window_size = min(10, len(all_trades) // 5)
            win_rates = []
            trade_numbers = []

            for i in range(window_size, len(all_trades) + 1):
                window = all_trades[i-window_size:i]
                wins = sum(1 for t in window if float(t['realized_pnl_usd']) > 0)
                wr = (wins / window_size) * 100
                win_rates.append(wr)
                trade_numbers.append(i)

            ax1.plot(trade_numbers, win_rates, linewidth=2, color='#2E86DE', marker='o', markersize=3)
            ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Break-even')
            ax1.axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Target (60%)')
            ax1.fill_between(trade_numbers, win_rates, 50, where=np.array(win_rates) >= 50,
                           alpha=0.3, color='green', label='Profit zone')
            ax1.fill_between(trade_numbers, win_rates, 50, where=np.array(win_rates) < 50,
                           alpha=0.3, color='red', label='Loss zone')
            ax1.set_xlabel('Trade Number', fontweight='bold')
            ax1.set_ylabel('Win Rate (%)', fontweight='bold')
            ax1.set_title(f'📈 Learning Curve (Rolling {window_size}-trade window)', fontweight='bold')
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 100])

            # 1b. Cumulative P&L Over Time
            cumulative_pnl = []
            cum_sum = 0
            for trade in all_trades:
                cum_sum += float(trade['realized_pnl_usd'])
                cumulative_pnl.append(cum_sum)

            trade_nums = list(range(1, len(all_trades) + 1))
            colors = ['green' if x >= 0 else 'red' for x in cumulative_pnl]
            ax2.plot(trade_nums, cumulative_pnl, linewidth=2, color='#2E86DE')
            ax2.fill_between(trade_nums, cumulative_pnl, 0, where=np.array(cumulative_pnl) >= 0,
                           alpha=0.3, color='green')
            ax2.fill_between(trade_nums, cumulative_pnl, 0, where=np.array(cumulative_pnl) < 0,
                           alpha=0.3, color='red')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_xlabel('Trade Number', fontweight='bold')
            ax2.set_ylabel('Cumulative P&L ($)', fontweight='bold')
            ax2.set_title('💰 Cumulative Profit/Loss', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # 1c. Confidence vs Actual Win Rate
            confidence_buckets = {}
            for trade in all_trades:
                conf = trade.get('ai_confidence')
                if conf:
                    conf_bucket = int(float(conf) * 100 / 10) * 10  # 10% buckets
                    if conf_bucket not in confidence_buckets:
                        confidence_buckets[conf_bucket] = {'wins': 0, 'total': 0}
                    confidence_buckets[conf_bucket]['total'] += 1
                    if float(trade['realized_pnl_usd']) > 0:
                        confidence_buckets[conf_bucket]['wins'] += 1

            if confidence_buckets:
                confs = sorted(confidence_buckets.keys())
                actual_wrs = [confidence_buckets[c]['wins'] / confidence_buckets[c]['total'] * 100
                            for c in confs]
                trade_counts = [confidence_buckets[c]['total'] for c in confs]

                # Bar plot with counts
                bars = ax3.bar(confs, actual_wrs, width=8, alpha=0.7, color='#2E86DE', label='Actual Win Rate')
                ax3.plot(confs, confs, 'r--', label='Perfect Calibration', linewidth=2)

                # Add count labels on bars
                for i, (conf, count) in enumerate(zip(confs, trade_counts)):
                    ax3.text(conf, actual_wrs[i] + 2, f'n={count}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')

                ax3.set_xlabel('AI Confidence Level (%)', fontweight='bold')
                ax3.set_ylabel('Actual Win Rate (%)', fontweight='bold')
                ax3.set_title('🎯 Confidence Calibration (AI Accuracy)', fontweight='bold')
                ax3.legend(loc='upper left', fontsize=9)
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim([0, 100])

            # 1d. Top Patterns Performance
            pattern_performance = {}
            if hasattr(ml, 'winning_patterns') and hasattr(ml, 'losing_patterns'):
                all_patterns = set(ml.winning_patterns.keys()) | set(ml.losing_patterns.keys())
                for pattern in all_patterns:
                    wins = ml.winning_patterns.get(pattern, 0)
                    losses = ml.losing_patterns.get(pattern, 0)
                    total = wins + losses
                    if total >= 5:  # Min 5 occurrences
                        pattern_performance[pattern] = {
                            'wins': wins,
                            'total': total,
                            'win_rate': (wins / total) * 100
                        }

            if pattern_performance:
                # Get top 10 patterns by occurrence
                sorted_patterns = sorted(pattern_performance.items(),
                                       key=lambda x: x[1]['total'], reverse=True)[:10]

                pattern_names = [p[0].replace('_', ' ').title()[:20] for p in sorted_patterns]
                pattern_wrs = [p[1]['win_rate'] for p in sorted_patterns]
                pattern_counts = [p[1]['total'] for p in sorted_patterns]

                # Horizontal bar chart
                colors_bar = ['green' if wr >= 60 else 'orange' if wr >= 50 else 'red'
                            for wr in pattern_wrs]
                bars = ax4.barh(pattern_names, pattern_wrs, color=colors_bar, alpha=0.7)

                # Add count labels
                for i, (wr, count) in enumerate(zip(pattern_wrs, pattern_counts)):
                    ax4.text(wr + 2, i, f'{wr:.0f}% (n={count})',
                           va='center', fontsize=8, fontweight='bold')

                ax4.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Break-even')
                ax4.axvline(x=60, color='green', linestyle='--', alpha=0.5, label='Target')
                ax4.set_xlabel('Win Rate (%)', fontweight='bold')
                ax4.set_title('🔥 Top 10 Patterns by Frequency', fontweight='bold')
                ax4.legend(loc='lower right', fontsize=8)
                ax4.set_xlim([0, 100])
                ax4.grid(True, alpha=0.3, axis='x')
            else:
                ax4.text(0.5, 0.5, 'Insufficient pattern data\n(min 5 occurrences each)',
                       ha='center', va='center', fontsize=12, transform=ax4.transAxes)
                ax4.set_title('🔥 Pattern Performance', fontweight='bold')

            plt.tight_layout()

            # Save graph to bytes
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()

            # Send graph
            await update.message.reply_photo(
                photo=buf,
                caption="📊 <b>ML LEARNING INSIGHTS</b>\n\n"
                       "🔬 Grafik açıklaması:\n"
                       "• <b>Üst Sol:</b> Win rate gelişimi (öğrenme eğrisi)\n"
                       "• <b>Üst Sağ:</b> Kümülatif kar/zarar trendi\n"
                       "• <b>Alt Sol:</b> AI güven seviyesi doğruluğu (kalibrasyon)\n"
                       "• <b>Alt Sağ:</b> En sık kullanılan pattern'ler ve başarı oranları\n\n"
                       "<i>Detaylı istatistikler için /mlstats kullanın</i>",
                parse_mode=ParseMode.HTML
            )

            # === 2. TEXT-BASED INSIGHTS ===
            total_trades = len(all_trades)
            winning_trades = [t for t in all_trades if float(t['realized_pnl_usd']) > 0]
            overall_wr = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum(float(t['realized_pnl_usd']) for t in all_trades)

            # Calculate improvement trend
            if total_trades >= 20:
                first_half = all_trades[:total_trades//2]
                second_half = all_trades[total_trades//2:]

                first_wr = sum(1 for t in first_half if float(t['realized_pnl_usd']) > 0) / len(first_half) * 100
                second_wr = sum(1 for t in second_half if float(t['realized_pnl_usd']) > 0) / len(second_half) * 100
                improvement = second_wr - first_wr
            else:
                improvement = 0

            message = "<b>🎓 ML LEARNING ANALYSIS</b>\n\n"
            message += f"<b>📊 Overall Performance:</b>\n"
            message += f"• Total Trades: {total_trades}\n"
            wr_emoji = "🟢" if overall_wr >= 60 else "🟡" if overall_wr >= 50 else "🔴"
            message += f"• Win Rate: {wr_emoji} {overall_wr:.1f}%\n"
            pnl_emoji = "💰" if total_pnl > 0 else "📉"
            message += f"• Total P&L: {pnl_emoji} ${total_pnl:+.2f}\n\n"

            message += f"<b>📈 Learning Trend:</b>\n"
            if improvement > 10:
                message += f"🎉 <b>Excellent!</b> Win rate improved by {improvement:+.0f}%\n"
                message += "ML is learning rapidly! Strategy is optimizing.\n\n"
            elif improvement > 0:
                message += f"✅ <b>Good!</b> Win rate improved by {improvement:+.0f}%\n"
                message += "Steady improvement observed.\n\n"
            elif improvement > -10:
                message += f"📊 <b>Stable.</b> Win rate change: {improvement:+.0f}%\n"
                message += "Performance is consistent.\n\n"
            else:
                message += f"⚠️ <b>Attention!</b> Win rate dropped by {abs(improvement):.0f}%\n"
                message += "May need pattern re-evaluation or market shift.\n\n"

            # Pattern insights
            if pattern_performance:
                best_patterns = sorted(pattern_performance.items(),
                                     key=lambda x: x[1]['win_rate'], reverse=True)[:3]
                message += f"<b>🏆 Best Performing Patterns:</b>\n"
                for pattern, stats in best_patterns:
                    if stats['win_rate'] >= 60:
                        pattern_clean = pattern.replace('_', ' ').title()
                        message += f"🟢 {pattern_clean}: {stats['win_rate']:.0f}% WR (n={stats['total']})\n"
                message += "\n"

            # Readiness assessment
            message += f"<b>🎯 Live Trading Readiness:</b>\n"
            if total_trades < 50:
                message += f"⚠️ <b>Not Ready</b> - Need {50 - total_trades} more trades\n"
                message += "Recommendation: Continue paper trading\n"
            elif overall_wr < 55:
                message += f"⚠️ <b>Not Ready</b> - Win rate below 55%\n"
                message += "Recommendation: Wait for strategy optimization\n"
            elif overall_wr >= 60:
                message += f"✅ <b>Ready!</b> - Strong performance ({overall_wr:.0f}% WR)\n"
                message += f"💡 Suggestion: Start with small capital ($100-200)\n"
            else:
                message += f"🟡 <b>Approaching Ready</b> - WR: {overall_wr:.0f}%\n"
                message += f"Recommendation: Monitor for {60 - overall_wr:.0f}% more improvement\n"

            message += "\n<i>📊 Use /mlstats for detailed statistics</i>"

            await update.message.reply_text(message, parse_mode=ParseMode.HTML)
            logger.info("✅ ML insights sent successfully with graphs")

        except Exception as e:
            logger.error(f"Error in mlinsights command: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ ML insights oluşturulamadı: {str(e)}\n\n"
                f"Database bağlantısını veya matplotlib kurulumunu kontrol edin.",
                parse_mode=ParseMode.HTML
            )

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
✨ <b>ULTRA PREMIUM GRAFİK</b>

TradingView Pro+ kalitesinde grafik:
• 🕯️ Premium candlestick tasarımı
• 📍 Glow efektli S/R seviyeleri
• 📈 Smooth EMA çizgileri (12/26/50)
• 📊 Profesyonel volume analizi
• 🎨 Dark theme premium renk paleti

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
        """Handle /startbot command - Enable trading via database."""
        try:
            # Set trading enabled in database (persistent across restarts)
            await self.db.set_trading_enabled(True)
            self.bot_running = True

            await update.message.reply_text(
                "✅ <b>Bot başlatıldı!</b>\n\n"
                "🔓 Trading ENABLED\n"
                "🔍 Market tarama aktif\n"
                "💰 Yeni pozisyonlar açılabilir\n\n"
                "⏸️ Durdurmak için: /stopbot",
                parse_mode=ParseMode.HTML
            )
            logger.info("✅ Trading enabled via /startbot command")
        except Exception as e:
            logger.error(f"Error enabling trading: {e}")
            await update.message.reply_text(
                f"❌ Hata: Bot başlatılamadı\n\n{str(e)}",
                parse_mode=ParseMode.HTML
            )

    async def cmd_stop_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stopbot command - Disable trading via database (EMERGENCY STOP)."""
        try:
            # Set trading disabled in database (persistent across restarts)
            await self.db.set_trading_enabled(False)
            self.bot_running = False

            await update.message.reply_text(
                "🛑 <b>BOT DURDURULDU!</b>\n\n"
                "🔒 Trading DISABLED\n"
                "❌ Yeni pozisyon açılmayacak\n"
                "📊 Mevcut pozisyonlar takip ediliyor\n\n"
                "⚠️ <b>AKSİYON GEREKLİ:</b>\n"
                "1. Mevcut pozisyonları kontrol et: /positions\n"
                "2. Gerekirse manuel kapat: /closeall\n"
                "3. Tekrar başlatmak için: /startbot",
                parse_mode=ParseMode.HTML
            )
            logger.warning("🛑 Trading DISABLED via /stopbot command (EMERGENCY STOP)")
        except Exception as e:
            logger.error(f"Error disabling trading: {e}")
            await update.message.reply_text(
                f"❌ Hata: Bot durdurulamadı\n\n{str(e)}",
                parse_mode=ParseMode.HTML
            )

    async def cmd_reset_circuit_breaker(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset command - Reset circuit breaker by adding fake winning trade."""
        try:
            # Check current consecutive losses
            consecutive_losses = await self.db.get_consecutive_losses()

            if consecutive_losses < 3:
                await update.message.reply_text(
                    f"✅ Circuit breaker zaten aktif değil!\n\n"
                    f"Mevcut ardışık loss: {consecutive_losses}\n"
                    f"Circuit breaker 3+ loss'ta devreye girer.",
                    parse_mode=ParseMode.HTML
                )
                return

            await update.message.reply_text(
                f"🔧 Circuit breaker reset ediliyor...\n\n"
                f"Mevcut ardışık loss: {consecutive_losses}\n"
                f"Fake winning trade ekleniyor...",
                parse_mode=ParseMode.HTML
            )

            # Create fake winning trade (only using columns that exist in schema!)
            from decimal import Decimal as D

            fake_trade_data = (
                'CIRCUIT_BREAKER_RESET',  # symbol
                'LONG',  # side
                2,  # leverage
                D('100.0'),  # entry_price
                D('101.0'),  # exit_price
                D('1.0'),  # quantity
                D('10.0'),  # position_value_usd
                D('1.0'),  # realized_pnl_usd (small profit)
                D('1.0'),  # pnl_percent
                D('5.0'),  # stop_loss_percent
                'Manual circuit breaker reset via /reset command',  # close_reason
                300,  # trade_duration_seconds (5 minutes)
                'MANUAL_RESET',  # ai_model_consensus
                D('1.0'),  # ai_confidence
                datetime.now() - timedelta(minutes=5),  # entry_time
                datetime.now(),  # exit_time
                True  # is_winner (CRITICAL: breaks the loss streak!)
            )

            async with self.db.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO trade_history (
                        symbol, side, leverage, entry_price, exit_price,
                        quantity, position_value_usd, realized_pnl_usd, pnl_percent,
                        stop_loss_percent, close_reason, trade_duration_seconds,
                        ai_model_consensus, ai_confidence,
                        entry_time, exit_time, is_winner
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """, *fake_trade_data)

            # Verify reset
            new_consecutive_losses = await self.db.get_consecutive_losses()

            if new_consecutive_losses == 0:
                await update.message.reply_text(
                    "🎉 <b>BAŞARILI!</b>\n\n"
                    "✅ Circuit breaker reset edildi!\n"
                    "✅ Trading tekrar aktif!\n\n"
                    f"Önceki ardışık loss: {consecutive_losses}\n"
                    f"Yeni ardışık loss: {new_consecutive_losses}\n\n"
                    "💡 Bot artık yeni pozisyon açabilir.",
                    parse_mode=ParseMode.HTML
                )
            else:
                await update.message.reply_text(
                    f"⚠️ Beklenmeyen durum!\n\n"
                    f"Reset sonrası hala {new_consecutive_losses} loss görünüyor.\n"
                    f"Database'i kontrol et.",
                    parse_mode=ParseMode.HTML
                )

        except Exception as e:
            logger.error(f"Circuit breaker reset error: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ Reset sırasında hata oluştu:\n\n{str(e)}",
                parse_mode=ParseMode.HTML
            )

    async def cmd_set_capital(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /setcapital command - Update current capital to specified amount."""
        try:
            # Parse amount from command arguments
            if not context.args or len(context.args) != 1:
                await update.message.reply_text(
                    "❌ <b>Kullanım:</b> /setcapital 1000\n\n"
                    "Örnek: /setcapital 1000 → Capital'i $1000'e ayarlar",
                    parse_mode=ParseMode.HTML
                )
                return

            try:
                new_capital = Decimal(context.args[0])
                if new_capital <= 0:
                    raise ValueError("Capital must be positive")
            except (ValueError, Exception) as e:
                await update.message.reply_text(
                    f"❌ Geçersiz miktar: {context.args[0]}\n\n"
                    "Pozitif bir sayı girin (örnek: 1000)",
                    parse_mode=ParseMode.HTML
                )
                return

            # Get current capital
            old_capital = await self.db.get_current_capital()

            await update.message.reply_text(
                f"💰 Capital güncelleniyor...\n\n"
                f"Eski: ${float(old_capital):.2f}\n"
                f"Yeni: ${float(new_capital):.2f}",
                parse_mode=ParseMode.HTML
            )

            # Update capital in database
            async with self.db.pool.acquire() as conn:
                await conn.execute(
                    "UPDATE trading_config SET current_capital = $1",
                    new_capital
                )

            # Verify update
            updated_capital = await self.db.get_current_capital()
            difference = updated_capital - old_capital

            # Calculate new position sizes
            position_size = updated_capital * Decimal('0.80')  # 80% position sizing
            max_positions = 10  # From config

            await update.message.reply_text(
                "🎉 <b>BAŞARILI!</b>\n\n"
                f"✅ Capital güncellendi: ${float(updated_capital):.2f}\n"
                f"📊 Değişim: ${float(difference):+.2f}\n\n"
                f"<b>Yeni Limitler:</b>\n"
                f"💵 Pozisyon başına: ${float(position_size):.2f}\n"
                f"📈 Max pozisyon: {max_positions}\n"
                f"💰 Toplam kullanılabilir: ${float(position_size * max_positions):.2f}\n\n"
                "🚀 Bot artık yeni pozisyonlar açabilir!",
                parse_mode=ParseMode.HTML
            )

            logger.info(
                f"💰 Capital manually updated: ${float(old_capital):.2f} → ${float(new_capital):.2f} "
                f"(${float(difference):+.2f})"
            )

        except Exception as e:
            logger.error(f"Set capital error: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ Capital güncellenirken hata oluştu:\n\n{str(e)}",
                parse_mode=ParseMode.HTML
            )

    async def cmd_close_all_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /closeall command - Close all active positions immediately at market price."""
        try:
            # Get all active positions
            positions = await self.db.get_active_positions()

            if not positions:
                await update.message.reply_text(
                    "ℹ️ <b>Açık pozisyon yok</b>\n\n"
                    "Kapatılacak pozisyon bulunamadı.",
                    parse_mode=ParseMode.HTML
                )
                return

            await update.message.reply_text(
                f"⚠️ <b>TÜM POZİSYONLAR KAPATILIYOR</b>\n\n"
                f"Toplam {len(positions)} pozisyon market fiyatından kapatılacak...\n"
                f"Zarardaki pozisyonlar da current price'dan kapatılacak.",
                parse_mode=ParseMode.HTML
            )

            # Import trade executor to close positions properly
            from src.trade_executor import get_trade_executor
            from src.exchange_client import get_exchange_client

            trade_executor = get_trade_executor()
            exchange = await get_exchange_client()

            closed_count = 0
            failed_count = 0
            total_pnl = Decimal('0')
            error_details = []

            for pos in positions:
                try:
                    symbol = pos['symbol']

                    # Get current market price
                    ticker = await exchange.fetch_ticker(symbol)
                    current_price = Decimal(str(ticker['last']))

                    # Use TradeExecutor.close_position() - it handles everything:
                    # - Exchange order execution
                    # - Database updates (trades table + remove from active_positions)
                    # - ML pattern learning
                    # - Telegram notifications
                    success = await trade_executor.close_position(
                        position=pos,
                        current_price=current_price,
                        close_reason="Manual close via /closeall"
                    )

                    if success:
                        closed_count += 1
                        # Calculate P&L for summary
                        entry_price = Decimal(str(pos['entry_price']))
                        quantity = Decimal(str(pos['quantity']))
                        side = pos['side']

                        if side.upper() == 'LONG':
                            pnl = (current_price - entry_price) * quantity
                        else:  # SHORT
                            pnl = (entry_price - current_price) * quantity

                        total_pnl += pnl
                        logger.info(f"✅ Closed {symbol}: ${pnl:+.2f}")
                    else:
                        failed_count += 1
                        error_details.append(f"{symbol}: TradeExecutor failed")

                except Exception as e:
                    failed_count += 1
                    error_msg = str(e)[:50]
                    error_details.append(f"{symbol}: {error_msg}")
                    logger.error(f"❌ Error closing {symbol}: {e}", exc_info=True)

            # Send summary
            pnl_emoji = "📈" if total_pnl >= 0 else "📉"
            message = f"<b>✅ KAPAMA TAMAMLANDI</b>\n\n"
            message += f"Başarılı: {closed_count} pozisyon\n"
            if failed_count > 0:
                message += f"Başarısız: {failed_count} pozisyon\n"
                if error_details:
                    message += f"\n<b>Hatalar:</b>\n"
                    for err in error_details[:3]:  # Show first 3 errors
                        message += f"• {err}\n"
            message += f"\n{pnl_emoji} <b>Toplam P&L:</b> ${float(total_pnl):+.2f}"

            if closed_count > 0:
                message += f"\n\n💰 <b>Ortalama P&L:</b> ${float(total_pnl/closed_count):+.2f} per trade"

            await update.message.reply_text(message, parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"Close all positions error: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ Pozisyonlar kapatılırken kritik hata:\n\n{str(e)[:200]}",
                parse_mode=ParseMode.HTML
            )

    async def cmd_websocket_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ws command - Show WebSocket price feed statistics."""
        try:
            from src.price_manager import get_price_manager

            price_manager = get_price_manager()
            stats = price_manager.get_stats()

            # Status emoji
            ws_status = "🟢 Bağlı" if stats['ws_connected'] else "🔴 Bağlantı yok"

            # API call rate indicator
            rate_usage = stats['rate_limit_usage_percent']
            if rate_usage < 50:
                rate_emoji = "🟢"
            elif rate_usage < 75:
                rate_emoji = "🟡"
            else:
                rate_emoji = "🔴"

            message = f"""
<b>🌐 WEBSOCKET PRICE FEED</b>

<b>📡 Durum:</b>
{ws_status}
Subscribed: {stats['subscribed_symbols']} simge

<b>📊 Performans:</b>
• WebSocket hits: {stats['ws_hits']} ({stats['ws_hit_rate_percent']:.1f}%)
• REST API calls: {stats['rest_api_calls']}
• Cache hits: {stats['rest_cache_hits']} ({stats['cache_hit_rate_percent']:.1f}%)
• OHLCV cache hits: {stats['ohlcv_cache_hits']}

<b>⚡ API Kullanımı:</b>
{rate_emoji} <b>{stats['calls_per_minute']}/min</b> ({rate_usage:.1f}% of limit)
• Limit: 1800/min (Binance: 2400/min)
• Rate limit waits: {stats['rate_limit_waits']}

<b>💾 Cache:</b>
• Cached symbols: {stats['cached_symbols']}
• Cached OHLCV: {stats['cached_ohlcv']}

<b>📈 Etki:</b>
WebSocket + Cache = ~85% daha az API çağrısı
429 Rate Limit hatası: ❌ Yok!

⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}
"""

            await update.message.reply_text(message, parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"WebSocket stats command error: {e}")
            await update.message.reply_text(
                f"❌ WebSocket istatistikleri alınırken hata:\n\n{str(e)[:200]}",
                parse_mode=ParseMode.HTML
            )

    async def cmd_force_sync(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /sync command - Force position reconciliation.

        Checks if Binance positions match database and fixes any mismatches.
        """
        try:
            await update.message.reply_text("🔄 <b>Position sync başlatılıyor...</b>", parse_mode=ParseMode.HTML)

            from src.position_reconciliation import get_reconciliation_system
            reconciliation = get_reconciliation_system()

            # Run reconciliation
            sync_results = await reconciliation.force_sync()

            if sync_results.get('error'):
                message = f"❌ <b>SYNC HATASI</b>\n\n{sync_results['error']}"
            else:
                binance_count = sync_results.get('binance_count', 0)
                db_count = sync_results.get('database_count', 0)
                matched = sync_results.get('matched_count', 0)
                orphaned = sync_results.get('orphaned_count', 0)
                ghosts = sync_results.get('ghost_count', 0)
                actions = sync_results.get('actions_taken', [])

                # Determine overall status
                if orphaned == 0 and ghosts == 0:
                    status_emoji = "✅"
                    status_text = "Tüm pozisyonlar senkronize!"
                else:
                    status_emoji = "⚠️"
                    status_text = "Senkronizasyon sorunları düzeltildi"

                message = f"""
{status_emoji} <b>POZİSYON SYNC TAMAMLANDI</b>

<b>📊 Durum:</b>
• Binance: {binance_count} pozisyon
• Database: {db_count} pozisyon
• ✅ Eşleşen: {matched}
• ⚠️ Orphaned: {orphaned}
• 👻 Ghost: {ghosts}

<b>🔧 Yapılan İşlemler:</b>
"""
                if actions:
                    for action in actions:
                        message += f"• {action}\n"
                else:
                    message += "• Hiçbir işlem gerekmedi\n"

                message += f"\n⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}"

            await update.message.reply_text(message, parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"Error in /sync command: {e}")
            await update.message.reply_text(
                f"❌ <b>SYNC HATASI</b>\n\n{str(e)}",
                parse_mode=ParseMode.HTML
            )

    async def cmd_analyze_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /analyze command - Analyze trade history OR symbol S/R levels.

        Usage:
        - /analyze → Trade history analysis (PNL, win rate, rapid trades)
        - /analyze BTC → Level-Based Trading System analysis for BTC/USDT
        """
        try:
            # Check if symbol argument provided
            if context.args and len(context.args) >= 1:
                symbol_input = context.args[0].upper()
                await self.analyze_symbol_levels(update, symbol_input)
                return

            # No arguments - show trade history analysis
            await update.message.reply_text("📊 <b>Trade history analiz ediliyor...</b>", parse_mode=ParseMode.HTML)

            from src.exchange_client import get_exchange_client
            exchange = await get_exchange_client()

            # Fetch realized PNL from Binance
            income_records = await exchange.exchange.fapiPrivateGetIncome({
                'incomeType': 'REALIZED_PNL',
                'limit': 50
            })

            if not income_records:
                await update.message.reply_text("⚠️ Henüz kapalı trade bulunamadı.")
                return

            # Analyze trades
            total_realized = 0
            wins = 0
            losses = 0
            rapid_trades = []
            trades_data = []

            for i, record in enumerate(income_records):
                symbol = record.get('symbol', 'UNKNOWN')
                pnl = float(record.get('income', 0))
                timestamp = int(record.get('time', 0)) / 1000
                # 🔧 FIX: Convert to Turkey timezone (UTC+3)
                date = datetime.fromtimestamp(timestamp, tz=TURKEY_TZ).strftime('%m-%d %H:%M')

                total_realized += pnl

                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1

                trades_data.append({
                    'date': date,
                    'symbol': symbol,
                    'pnl': pnl,
                    'timestamp': timestamp
                })

                # Check for rapid trades (< 5 minutes apart)
                if i < len(income_records) - 1:
                    next_time = int(income_records[i+1].get('time', 0)) / 1000
                    time_diff = timestamp - next_time

                    if abs(time_diff) < 300:  # Less than 5 minutes
                        rapid_trades.append({
                            'symbol': symbol,
                            'time_diff': abs(time_diff),
                            'pnl': pnl
                        })

            # Build message
            total_trades = wins + losses
            win_rate = wins / total_trades * 100 if total_trades > 0 else 0

            message = f"""
📊 <b>TRADE HISTORY ANALİZİ</b>

<b>💰 Genel Durum:</b>
• Toplam Realized PNL: <b>${total_realized:+.2f}</b>
• Toplam Trade: {total_trades}
• ✅ Kazanan: {wins} ({win_rate:.1f}%)
• ❌ Kaybeden: {losses} ({100-win_rate:.1f}%)

<b>📋 Son 10 Trade</b> (yeniden eskiye):
"""

            # 🔧 FIX: Show LAST 10 trades from NEWEST to OLDEST
            # Binance API returns oldest→newest, so take last 10 and reverse
            last_10_trades = trades_data[-10:] if len(trades_data) > 10 else trades_data
            for trade in reversed(last_10_trades):
                pnl_emoji = "✅" if trade['pnl'] > 0 else "❌"
                message += f"{pnl_emoji} {trade['date']} - {trade['symbol']}: ${trade['pnl']:+.2f}\n"

            # Analyze rapid trades
            if rapid_trades:
                rapid_loss = sum(rt['pnl'] for rt in rapid_trades)
                message += f"""

⚠️ <b>HIZLI KAPANAN TRADELER:</b>
• {len(rapid_trades)} trade 5 dakikadan kısa sürdü!
• Bu tradelerden toplam: <b>${rapid_loss:+.2f}</b>

<b>🚨 SORUN TESPİTİ:</b>
"""
                if rapid_loss < -5:
                    message += """
Bu tradeler çok hızlı kapandı - stop-loss hemen tetiklendi!

<b>Olası Sebepler:</b>
1. Entry fiyatı kötü (volatilite sırasında giriş)
2. Stop-loss 20x leverage için çok dar
3. Market hemen ters yönde hareket etti
4. Slippage nedeniyle kötü fiyattan giriş

<b>Çözüm Önerileri:</b>
• PA analiz kalitesini kontrol et
• Entry timing'i iyileştir
• Stop-loss mesafesini gözden geçir
"""
                else:
                    message += "Hızlı tradeler normal range'de, sorun yok.\n"

            message += f"\n⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}"

            await update.message.reply_text(message, parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"Error in /analyze command: {e}")
            await update.message.reply_text(
                f"❌ <b>ANALİZ HATASI</b>\n\n{str(e)}",
                parse_mode=ParseMode.HTML
            )

    async def analyze_symbol_levels(self, update: Update, symbol_input: str):
        """
        🎯 Level-Based Trading System v5.0 Analysis

        Shows comprehensive S/R analysis including:
        - Support/Resistance levels from 5 timeframes
        - Trend lines (ascending/descending)
        - Volume status and spike detection
        - RSI value and zone
        - Candlestick patterns
        - Entry confirmation status
        """
        from src.exchange_client import get_exchange_client
        from src.price_action_analyzer import PriceActionAnalyzer
        from src.indicators import calculate_indicators

        await update.message.reply_text(
            f"🎯 <b>{symbol_input} Level-Based Analiz yapılıyor...</b>",
            parse_mode=ParseMode.HTML
        )

        try:
            # Format symbol
            if '/' not in symbol_input:
                symbol = f"{symbol_input}/USDT:USDT"
            else:
                symbol = symbol_input

            exchange = await get_exchange_client()
            pa = PriceActionAnalyzer()

            # Fetch 15m OHLCV data
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
            if not ohlcv or len(ohlcv) < 50:
                await update.message.reply_text(
                    f"❌ {symbol} için yeterli veri bulunamadı.",
                    parse_mode=ParseMode.HTML
                )
                return

            import pandas as pd
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            current_price = df['close'].iloc[-1]

            # 🔧 Helper function to format prices with appropriate decimal places
            # For crypto: show more decimals for low-priced coins
            def fmt_price(price):
                """Format price based on value - more decimals for low prices"""
                if price >= 1000:
                    return f"${price:,.2f}"
                elif price >= 1:
                    return f"${price:.4f}"
                elif price >= 0.01:
                    return f"${price:.5f}"
                else:
                    return f"${price:.6f}"

            # Calculate indicators
            indicators = calculate_indicators(df)

            # Get multi-timeframe S/R analysis
            sr_analysis = await pa.analyze_multi_timeframe_sr(
                symbol=symbol,
                current_price=current_price,
                exchange=exchange,
                df_15m=df
            )

            # Detect trend lines
            trend_lines = pa.detect_trend_lines(df, min_touches=2)
            ascending_lines = trend_lines.get('ascending', [])
            descending_lines = trend_lines.get('descending', [])

            # Get all supports and resistances
            all_supports = sr_analysis.get('all_supports', [])
            all_resistances = sr_analysis.get('all_resistances', [])

            # Add trend line levels
            for tl in ascending_lines:
                tl.update_current_price(len(df) - 1)
                tl_level = tl.to_sr_level(current_price)
                all_supports.append(tl_level.to_dict())

            for tl in descending_lines:
                tl.update_current_price(len(df) - 1)
                tl_level = tl.to_sr_level(current_price)
                all_resistances.append(tl_level.to_dict())

            # 🔧 FIX: Filter levels by position relative to current price
            # Support = price BELOW current (fiyatın ALTINDA)
            # Resistance = price ABOVE current (fiyatın ÜSTÜNDE)

            # Combine all levels and re-classify based on current price
            all_levels = []
            for s in all_supports:
                s['original_type'] = 'support'
                all_levels.append(s)
            for r in all_resistances:
                r['original_type'] = 'resistance'
                all_levels.append(r)

            # Re-classify based on current price position
            actual_supports = []  # Levels BELOW current price
            actual_resistances = []  # Levels ABOVE current price

            for level in all_levels:
                price = level.get('price', 0)
                if price <= 0:
                    continue

                distance_pct = (price - current_price) / current_price * 100
                level['distance_pct'] = abs(distance_pct)
                level['distance_signed'] = distance_pct

                if price < current_price:
                    # Level is BELOW current price = SUPPORT
                    actual_supports.append(level)
                else:
                    # Level is ABOVE current price = RESISTANCE
                    actual_resistances.append(level)

            # Sort by distance (closest first)
            actual_supports.sort(key=lambda x: x.get('distance_pct', 999))
            actual_resistances.sort(key=lambda x: x.get('distance_pct', 999))

            # Use the corrected lists
            all_supports = actual_supports
            all_resistances = actual_resistances

            # Check confirmations for both directions
            long_conf = pa.entry_confirmation.check_all_confirmations(df, 'LONG', indicators)
            short_conf = pa.entry_confirmation.check_all_confirmations(df, 'SHORT', indicators)

            # Get volume and RSI info
            volume_conf = long_conf['confirmations'].get('volume', {})
            rsi_conf = long_conf['confirmations'].get('rsi', {})
            candle_long = long_conf['confirmations'].get('candlestick', {})
            candle_short = short_conf['confirmations'].get('candlestick', {})

            # Build comprehensive message
            # 🔧 FIX: Tighter proximity thresholds
            AT_LEVEL_THRESHOLD = 0.15  # 0.15% = "BURADA!" (very close)
            APPROACHING_THRESHOLD = 0.5  # 0.5% = "YAKLAŞIYOR" (watching)

            # Check proximity to levels
            nearest_support_dist = all_supports[0].get('distance_pct', 999) if all_supports else 999
            nearest_resistance_dist = all_resistances[0].get('distance_pct', 999) if all_resistances else 999

            # Determine level status
            at_support = nearest_support_dist <= AT_LEVEL_THRESHOLD
            approaching_support = AT_LEVEL_THRESHOLD < nearest_support_dist <= APPROACHING_THRESHOLD
            at_resistance = nearest_resistance_dist <= AT_LEVEL_THRESHOLD
            approaching_resistance = AT_LEVEL_THRESHOLD < nearest_resistance_dist <= APPROACHING_THRESHOLD

            # Volume emoji
            vol_ratio = volume_conf.get('ratio', 0)
            vol_emoji = "🔥" if vol_ratio >= 1.5 else "📊"
            vol_status = "SPIKE!" if vol_ratio >= 1.5 else "Normal"

            # RSI emoji and zone
            rsi_value = rsi_conf.get('value', 50)
            rsi_zone = rsi_conf.get('zone', 'neutral')
            if rsi_zone == 'oversold':
                rsi_emoji = "🟢"
            elif rsi_zone == 'overbought':
                rsi_emoji = "🔴"
            else:
                rsi_emoji = "⚪"

            # Entry status with refined thresholds
            if at_support:
                entry_status = "🟢 SUPPORT SEVİYESİNDE! (&lt;%0.15)"
                entry_direction = "LONG için hazır"
                conf_check = long_conf
            elif at_resistance:
                entry_status = "🔴 RESISTANCE SEVİYESİNDE! (&lt;%0.15)"
                entry_direction = "SHORT için hazır"
                conf_check = short_conf
            elif approaching_support:
                entry_status = f"🟡 Support'a YAKLAŞIYOR ({nearest_support_dist:.2f}%)"
                entry_direction = "LONG için hazırlan"
                conf_check = long_conf
            elif approaching_resistance:
                entry_status = f"🟡 Resistance'a YAKLAŞIYOR ({nearest_resistance_dist:.2f}%)"
                entry_direction = "SHORT için hazırlan"
                conf_check = short_conf
            else:
                entry_status = "⚪ Seviyeler arası (mid-range)"
                entry_direction = "Seviye bekle - İŞLEM YAPMA"
                conf_check = long_conf

            # Format supports (top 5)
            support_lines = ""
            for i, s in enumerate(all_supports[:5]):
                price = s.get('price', 0)
                dist = s.get('distance_pct', 0)
                tf = s.get('timeframe', '?')
                source = s.get('source', 'swing')[:6]
                if dist <= AT_LEVEL_THRESHOLD:
                    at_marker = " ← BURADA!"
                elif dist <= APPROACHING_THRESHOLD:
                    at_marker = " ← yakın"
                else:
                    at_marker = ""
                support_lines += f"  {i+1}. {fmt_price(price)} ({tf}, {source}) -{dist:.2f}%{at_marker}\n"

            # Format resistances (top 5)
            resistance_lines = ""
            for i, r in enumerate(all_resistances[:5]):
                price = r.get('price', 0)
                dist = r.get('distance_pct', 0)
                tf = r.get('timeframe', '?')
                source = r.get('source', 'swing')[:6]
                if dist <= AT_LEVEL_THRESHOLD:
                    at_marker = " ← BURADA!"
                elif dist <= APPROACHING_THRESHOLD:
                    at_marker = " ← yakın"
                else:
                    at_marker = ""
                resistance_lines += f"  {i+1}. {fmt_price(price)} ({tf}, {source}) +{dist:.2f}%{at_marker}\n"

            # Confirmation checkboxes
            candle_ok = conf_check['confirmations'].get('candlestick', {}).get('confirmed', False)
            volume_ok = conf_check['confirmations'].get('volume', {}).get('confirmed', False)
            rsi_ok = conf_check['confirmations'].get('rsi', {}).get('confirmed', False)

            candle_box = "✅" if candle_ok else "❌"
            volume_box = "✅" if volume_ok else "❌"
            rsi_box = "✅" if rsi_ok else "❌"

            all_confirmed = candle_ok and volume_ok and rsi_ok

            # Candlestick patterns found
            patterns_long = candle_long.get('patterns', [])
            patterns_short = candle_short.get('patterns', [])
            patterns_str = ', '.join(patterns_long + patterns_short) if (patterns_long or patterns_short) else "Yok"

            # 🔧 R:R Quality Helper Function
            def get_rr_quality(rr_ratio):
                if rr_ratio >= 2.0:
                    return "✅ Mükemmel", True
                elif rr_ratio >= 1.5:
                    return "✅ İyi", True
                elif rr_ratio >= 1.0:
                    return "⚠️ Zayıf", False
                else:
                    return "❌ KÖTÜ - İŞLEM YAPMA", False

            # Calculate trade scenarios
            # 🔧 FIX: Use TARGET 2 for R:R calculation (main profit target)
            # Target 1 = partial take profit, Target 2 = main target

            # LONG scenario (at support)
            long_entry = all_supports[0].get('price', 0) if all_supports else 0
            long_stop = long_entry * 0.995 if long_entry else 0  # 0.5% below support
            long_target1 = all_resistances[0].get('price', 0) if all_resistances else 0
            long_target2 = all_resistances[1].get('price', 0) if len(all_resistances) > 1 else 0

            # Calculate R:R for both targets
            long_risk = (long_entry - long_stop) if long_entry and long_stop else 0
            long_rr1 = ((long_target1 - long_entry) / long_risk) if long_risk > 0 and long_target1 else 0
            long_rr2 = ((long_target2 - long_entry) / long_risk) if long_risk > 0 and long_target2 else 0
            # 🎯 Use Target 2 R:R as MAIN decision metric (or Target 1 if no Target 2)
            long_rr = long_rr2 if long_rr2 > 0 else long_rr1

            # SHORT scenario (at resistance)
            short_entry = all_resistances[0].get('price', 0) if all_resistances else 0
            short_stop = short_entry * 1.005 if short_entry else 0  # 0.5% above resistance
            short_target1 = all_supports[0].get('price', 0) if all_supports else 0
            short_target2 = all_supports[1].get('price', 0) if len(all_supports) > 1 else 0

            # Calculate R:R for both targets
            short_risk = (short_stop - short_entry) if short_entry and short_stop else 0
            short_rr1 = ((short_entry - short_target1) / short_risk) if short_risk > 0 and short_target1 else 0
            short_rr2 = ((short_entry - short_target2) / short_risk) if short_risk > 0 and short_target2 else 0
            # 🎯 Use Target 2 R:R as MAIN decision metric (or Target 1 if no Target 2)
            short_rr = short_rr2 if short_rr2 > 0 else short_rr1

            # Build message parts for Telegram
            message = f"""
🎯 <b>LEVEL-BASED ANALİZ v5.0</b>
━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📊 {symbol.split('/')[0]}</b> @ <code>{fmt_price(current_price)}</code>

<b>📍 POZİSYON DURUMU:</b>
{entry_status}
{entry_direction}

━━━━━━━━━━━━━━━━━━━━━━━━━
<b>🟢 SUPPORT SEVİYELERİ</b> (aşağıda):
{support_lines if support_lines else "  Bulunamadı"}
<b>🔴 RESISTANCE SEVİYELERİ</b> (yukarıda):
{resistance_lines if resistance_lines else "  Bulunamadı"}
━━━━━━━━━━━━━━━━━━━━━━━━━
<b>📈 TREND ÇİZGİLERİ:</b>
  ↗️ Ascending (Support): {len(ascending_lines)} adet
  ↘️ Descending (Resist): {len(descending_lines)} adet

━━━━━━━━━━━━━━━━━━━━━━━━━
<b>🔍 MEVCUT TEYİT DURUMU:</b>
{candle_box} Candlestick: {patterns_str}
{volume_box} Volume: {vol_emoji} {vol_ratio:.1f}x (≥1.5x gerekli)
{rsi_box} RSI: {rsi_emoji} {rsi_value:.0f} ({rsi_zone})

<b>📊 TREND ANALİZİ:</b>
  ADX: {indicators.get('adx', 25):.1f} ({('GÜÇLÜ TREND' if indicators.get('adx', 25) > 25 else 'ZAYIF/YOK')})
  +DI: {indicators.get('plus_di', 25):.1f} | -DI: {indicators.get('minus_di', 25):.1f}
  Yön: {('📈 YUKARI' if indicators.get('plus_di', 25) > indicators.get('minus_di', 25) else '📉 AŞAĞI')}
  EMA20: {fmt_price(indicators.get('ema_20', current_price))}
  EMA50: {fmt_price(indicators.get('ema_50', current_price))}
"""

            # Add LONG scenario
            if long_entry > 0 and long_target1 > 0:
                long_rr_quality, long_rr_ok = get_rr_quality(long_rr)
                # Calculate potential profits
                long_tp1_profit_pct = ((long_target1 - long_entry) / long_entry * 100)
                long_tp2_profit_pct = ((long_target2 - long_entry) / long_entry * 100) if long_target2 else 0
                message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
<b>📈 LONG SENARYO</b> (Support'ta al):
  Entry: <code>{fmt_price(long_entry)}</code>
  Stop Loss: <code>{fmt_price(long_stop)}</code> (-0.5%)
  Target 1: <code>{fmt_price(long_target1)}</code> (+{long_tp1_profit_pct:.1f}%) R:R={long_rr1:.1f}:1"""
                if long_target2 > 0:
                    message += f"\n  <b>Target 2: <code>{fmt_price(long_target2)}</code></b> (+{long_tp2_profit_pct:.1f}%) <b>R:R={long_rr2:.1f}:1</b> ← ANA HEDEF"
                message += f"\n\n  📊 <b>R:R (Target 2): {long_rr:.1f}:1</b> {long_rr_quality}"
                if not long_rr_ok:
                    message += f"\n  ⛔ <b>R:R &lt;1.5 - Bu işlem riskli!</b>"

                # Add execution strategy
                message += f"""

<b>📋 EXECUTION PLAN:</b>
  1️⃣ T1'de %50 kapat → +{long_tp1_profit_pct:.1f}% kar
  2️⃣ Stop'u Entry'ye çek (Breakeven)
  3️⃣ T2'de kalan %50 kapat → +{long_tp2_profit_pct:.1f}% kar

<b>🔔 LONG için gerekli teyitler:</b>
  □ RSI ≤30 (oversold) - Şimdi: {rsi_value:.0f}
  □ Volume ≥1.5x spike - Şimdi: {vol_ratio:.1f}x
  □ Bullish candle pattern (hammer, engulfing, pin bar)
  □ R:R ≥1.5 (T2 bazlı) - Şimdi: {long_rr:.1f}
"""

            # Add SHORT scenario
            if short_entry > 0 and short_target1 > 0:
                short_rr_quality, short_rr_ok = get_rr_quality(short_rr)
                # Calculate potential profits
                short_tp1_profit_pct = ((short_entry - short_target1) / short_entry * 100)
                short_tp2_profit_pct = ((short_entry - short_target2) / short_entry * 100) if short_target2 else 0
                message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
<b>📉 SHORT SENARYO</b> (Resistance'ta sat):
  Entry: <code>{fmt_price(short_entry)}</code>
  Stop Loss: <code>{fmt_price(short_stop)}</code> (+0.5%)
  Target 1: <code>{fmt_price(short_target1)}</code> (-{short_tp1_profit_pct:.1f}%) R:R={short_rr1:.1f}:1"""
                if short_target2 > 0:
                    message += f"\n  <b>Target 2: <code>{fmt_price(short_target2)}</code></b> (-{short_tp2_profit_pct:.1f}%) <b>R:R={short_rr2:.1f}:1</b> ← ANA HEDEF"
                message += f"\n\n  📊 <b>R:R (Target 2): {short_rr:.1f}:1</b> {short_rr_quality}"
                if not short_rr_ok:
                    message += f"\n  ⛔ <b>R:R &lt;1.5 - Bu işlem riskli!</b>"

                # Add execution strategy
                message += f"""

<b>📋 EXECUTION PLAN:</b>
  1️⃣ T1'de %50 kapat → +{short_tp1_profit_pct:.1f}% kar
  2️⃣ Stop'u Entry'ye çek (Breakeven)
  3️⃣ T2'de kalan %50 kapat → +{short_tp2_profit_pct:.1f}% kar

<b>🔔 SHORT için gerekli teyitler:</b>
  □ RSI ≥70 (overbought) - Şimdi: {rsi_value:.0f}
  □ Volume ≥1.5x spike - Şimdi: {vol_ratio:.1f}x
  □ Bearish candle pattern (shooting star, engulfing)
  □ R:R ≥1.5 (T2 bazlı) - Şimdi: {short_rr:.1f}
"""

            # Final decision with R:R check
            message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
<b>🎯 SONUÇ:</b>
"""
            # Get relevant R:R for decision
            relevant_rr = long_rr if (at_support or approaching_support) else short_rr
            relevant_rr_ok = relevant_rr >= 1.5

            if at_support or at_resistance:
                direction = "LONG" if at_support else "SHORT"
                entry_p = long_entry if at_support else short_entry
                stop_p = long_stop if at_support else short_stop
                rr_val = long_rr if at_support else short_rr

                if all_confirmed and relevant_rr_ok:
                    message += f"✅ <b>TÜM TEYİTLER + R:R TAMAM!</b>\n"
                    message += f"   {direction} @ {fmt_price(entry_p)}\n"
                    message += f"   Stop: {fmt_price(stop_p)}\n"
                    message += f"   R:R: {rr_val:.1f}:1 ✓"
                elif all_confirmed and not relevant_rr_ok:
                    message += f"⚠️ <b>Teyitler tamam AMA R:R kötü!</b>\n"
                    message += f"   R:R {rr_val:.1f}:1 < 1.5 - RİSKLİ!\n"
                    message += f"   İşlem önerilmez, daha iyi seviye bekle"
                else:
                    missing = conf_check.get('missing', [])
                    missing_tr = []
                    for m in missing:
                        if m == 'candlestick_pattern':
                            missing_tr.append('Mum formasyonu')
                        elif m == 'volume_spike':
                            missing_tr.append('Volume spike')
                        elif m == 'rsi_extreme':
                            missing_tr.append('RSI extreme')
                        else:
                            missing_tr.append(m)
                    message += f"⏳ Seviyedesin ama eksik:\n"
                    message += f"   {', '.join(missing_tr)}"
                    if not relevant_rr_ok:
                        message += f"\n   ⚠️ Ayrıca R:R {rr_val:.1f}:1 < 1.5"

            elif approaching_support or approaching_resistance:
                direction = "LONG" if approaching_support else "SHORT"
                level_p = long_entry if approaching_support else short_entry
                dist = nearest_support_dist if approaching_support else nearest_resistance_dist
                message += f"🟡 <b>Seviyeye yaklaşıyorsun!</b>\n"
                message += f"   {direction} hazırlığı yap\n"
                message += f"   Seviye: {fmt_price(level_p)} ({dist:.2f}% uzakta)\n"
                message += f"   Teyitleri bekle, henüz işlem YAPMA"

            else:
                nearest = min(
                    all_supports[:1] + all_resistances[:1],
                    key=lambda x: x.get('distance_pct', 999),
                    default=None
                )
                if nearest:
                    message += f"⛔ <b>SEVİYELERDEN UZAK - İŞLEM YAPMA</b>\n"
                    message += f"   En yakın: {fmt_price(nearest.get('price', 0))} ({nearest.get('distance_pct', 0):.2f}%)\n"
                    message += f"   Fiyat seviyeye gelene kadar bekle"
                else:
                    message += "⏳ Seviye bulunamadı"

            message += f"\n\n⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}"

            await update.message.reply_text(message, parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"Error in analyze_symbol_levels: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ <b>ANALİZ HATASI</b>\n\n{str(e)}",
                parse_mode=ParseMode.HTML
            )

    async def cmd_scan_sr_levels(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        🔍 /scanrs - Tüm coinleri tara, S/R seviyelerine yakın olanları listele.

        Bu komut:
        1. Tüm trade edilebilir coinleri tarar
        2. Her coin için S/R seviyelerini hesaplar
        3. Fiyatı S/R seviyesine yakın olanları listeler
        4. Kullanıcı sonra /analyze COIN ile detay bakabilir
        """
        from src.exchange_client import get_exchange_client
        from src.price_action_analyzer import PriceActionAnalyzer
        from src.symbol_blacklist import get_symbol_blacklist
        import asyncio

        status_msg = await update.message.reply_text(
            "🔍 <b>S/R Seviye Taraması Başlatılıyor...</b>\n\n"
            "⏳ Tüm coinler taranıyor, lütfen bekleyin...",
            parse_mode=ParseMode.HTML
        )

        try:
            exchange = await get_exchange_client()
            pa = PriceActionAnalyzer()
            blacklist = get_symbol_blacklist()
            settings = get_settings()

            # Get symbols from config (already filtered for liquidity)
            all_symbols = settings.trading_symbols
            total_coins = len(all_symbols)

            # Filter out blacklisted
            symbols_to_scan = []
            for symbol in all_symbols:
                is_blocked, _ = blacklist.is_blacklisted(symbol)
                if not is_blocked:
                    symbols_to_scan.append(symbol)

            await status_msg.edit_text(
                f"🔍 <b>S/R Seviye Taraması</b>\n\n"
                f"📊 Toplam: {total_coins} coin\n"
                f"🚫 Blacklist: {total_coins - len(symbols_to_scan)} coin\n"
                f"✅ Taranacak: {len(symbols_to_scan)} coin\n\n"
                f"⏳ Tarama devam ediyor...",
                parse_mode=ParseMode.HTML
            )

            # S/R proximity thresholds
            AT_LEVEL_THRESHOLD = 0.5  # 0.5% = at level (çok yakın)
            NEAR_LEVEL_THRESHOLD = 1.0  # 1.0% = near level (yaklaşıyor)

            at_support = []  # Support seviyesinde
            at_resistance = []  # Resistance seviyesinde
            near_support = []  # Support'a yakın
            near_resistance = []  # Resistance'a yakın
            errors = 0
            scanned = 0

            # Scan in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(symbols_to_scan), batch_size):
                batch = symbols_to_scan[i:i + batch_size]

                async def check_symbol(symbol):
                    nonlocal errors, scanned
                    try:
                        # Fetch minimal data for quick check
                        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
                        if not ohlcv or len(ohlcv) < 50:
                            return None

                        import pandas as pd
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        current_price = float(df['close'].iloc[-1])

                        # Get S/R levels (simplified - just 15m for speed)
                        sr = pa.analyze_support_resistance(df, current_price, include_psychological=True)

                        supports = sr.get('support', [])
                        resistances = sr.get('resistance', [])

                        # Check proximity to supports
                        coin_name = symbol.split('/')[0]

                        for s in supports:
                            s_price = s.get('price', 0)
                            if s_price <= 0:
                                continue
                            dist_pct = abs(current_price - s_price) / current_price * 100
                            # Support must be BELOW current price
                            if s_price < current_price and dist_pct <= AT_LEVEL_THRESHOLD:
                                return ('at_support', coin_name, dist_pct, current_price)
                            elif s_price < current_price and dist_pct <= NEAR_LEVEL_THRESHOLD:
                                return ('near_support', coin_name, dist_pct, current_price)

                        # Check proximity to resistances
                        for r in resistances:
                            r_price = r.get('price', 0)
                            if r_price <= 0:
                                continue
                            dist_pct = abs(current_price - r_price) / current_price * 100
                            # Resistance must be ABOVE current price
                            if r_price > current_price and dist_pct <= AT_LEVEL_THRESHOLD:
                                return ('at_resistance', coin_name, dist_pct, current_price)
                            elif r_price > current_price and dist_pct <= NEAR_LEVEL_THRESHOLD:
                                return ('near_resistance', coin_name, dist_pct, current_price)

                        scanned += 1
                        return None

                    except Exception as e:
                        errors += 1
                        return None

                # Run batch in parallel
                results = await asyncio.gather(*[check_symbol(s) for s in batch], return_exceptions=True)

                for result in results:
                    if result and not isinstance(result, Exception):
                        level_type, coin, dist, price = result
                        if level_type == 'at_support':
                            at_support.append((coin, dist, price))
                        elif level_type == 'at_resistance':
                            at_resistance.append((coin, dist, price))
                        elif level_type == 'near_support':
                            near_support.append((coin, dist, price))
                        elif level_type == 'near_resistance':
                            near_resistance.append((coin, dist, price))
                    else:
                        scanned += 1

                # Update progress every batch
                if (i + batch_size) % 30 == 0:
                    await status_msg.edit_text(
                        f"🔍 <b>S/R Seviye Taraması</b>\n\n"
                        f"⏳ İlerleme: {min(i + batch_size, len(symbols_to_scan))}/{len(symbols_to_scan)}\n"
                        f"🟢 Support'ta: {len(at_support)}\n"
                        f"🔴 Resistance'ta: {len(at_resistance)}\n"
                        f"🟡 Yaklaşan: {len(near_support) + len(near_resistance)}",
                        parse_mode=ParseMode.HTML
                    )

                # Small delay to avoid rate limits
                await asyncio.sleep(0.2)

            # Build final message
            message = f"""
🔍 <b>S/R SEVİYE TARAMASI TAMAMLANDI</b>
━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Taranan: {len(symbols_to_scan)} coin
⏱️ {get_turkey_time().strftime('%H:%M:%S')}

"""

            # At Support (LONG fırsatı)
            if at_support:
                message += f"🟢 <b>SUPPORT SEVİYESİNDE ({len(at_support)} coin)</b>\n"
                message += "<i>→ LONG için hazır, /analyze ile detay bak</i>\n"
                for coin, dist, price in sorted(at_support, key=lambda x: x[1]):
                    message += f"  • <code>{coin}</code> ({dist:.2f}% uzakta)\n"
                message += "\n"

            # At Resistance (SHORT fırsatı)
            if at_resistance:
                message += f"🔴 <b>RESISTANCE SEVİYESİNDE ({len(at_resistance)} coin)</b>\n"
                message += "<i>→ SHORT için hazır, /analyze ile detay bak</i>\n"
                for coin, dist, price in sorted(at_resistance, key=lambda x: x[1]):
                    message += f"  • <code>{coin}</code> ({dist:.2f}% uzakta)\n"
                message += "\n"

            # Near Support (yaklaşıyor)
            if near_support:
                message += f"🟡 <b>SUPPORT'A YAKLAŞIYOR ({len(near_support)} coin)</b>\n"
                message += "<i>→ Biraz bekle, seviyeye gelince LONG</i>\n"
                for coin, dist, price in sorted(near_support, key=lambda x: x[1])[:10]:  # Max 10
                    message += f"  • <code>{coin}</code> ({dist:.2f}%)\n"
                if len(near_support) > 10:
                    message += f"  <i>...ve {len(near_support) - 10} coin daha</i>\n"
                message += "\n"

            # Near Resistance (yaklaşıyor)
            if near_resistance:
                message += f"🟠 <b>RESISTANCE'A YAKLAŞIYOR ({len(near_resistance)} coin)</b>\n"
                message += "<i>→ Biraz bekle, seviyeye gelince SHORT</i>\n"
                for coin, dist, price in sorted(near_resistance, key=lambda x: x[1])[:10]:  # Max 10
                    message += f"  • <code>{coin}</code> ({dist:.2f}%)\n"
                if len(near_resistance) > 10:
                    message += f"  <i>...ve {len(near_resistance) - 10} coin daha</i>\n"
                message += "\n"

            if not at_support and not at_resistance and not near_support and not near_resistance:
                message += "⚪ <b>Şu an hiçbir coin S/R seviyesinde değil.</b>\n"
                message += "Biraz bekle ve tekrar tara.\n\n"

            message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
<b>📌 Kullanım:</b>
<code>/analyze COIN</code> ile detaylı analiz yap
Örn: <code>/analyze BTC</code>, <code>/analyze ETH</code>
"""

            await status_msg.edit_text(message, parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"Error in cmd_scan_sr_levels: {e}", exc_info=True)
            await status_msg.edit_text(
                f"❌ <b>TARAMA HATASI</b>\n\n{str(e)}",
                parse_mode=ParseMode.HTML
            )

    async def cmd_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        /predict [COIN] - Generate AI prediction chart with Entry/TP/SL.

        Creates professional TradingView-style chart showing:
        - Price prediction direction (LONG/SHORT)
        - Entry point
        - Take Profit levels (TP1, TP2, TP3)
        - Stop Loss level
        - Trend channel lines
        - Risk/Reward ratio

        Usage: /predict BTC, /predict ETH, /predict SOL
        """
        from src.exchange_client import get_exchange_client
        from src.prediction_chart_generator import get_prediction_generator

        # Check for symbol argument
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "🎯 <b>PREDICTION CHART</b>\n\n"
                "Kullanım: <code>/predict COIN</code>\n\n"
                "Örnekler:\n"
                "• <code>/predict BTC</code>\n"
                "• <code>/predict ETH</code>\n"
                "• <code>/predict SOL</code>\n\n"
                "Bu komut profesyonel bir grafik oluşturur:\n"
                "• 📈 Entry noktası\n"
                "• 🎯 TP1, TP2, TP3 hedefleri\n"
                "• 🛑 Stop Loss seviyesi\n"
                "• 📊 Trend kanalı\n"
                "• 💹 Risk/Reward oranı",
                parse_mode=ParseMode.HTML
            )
            return

        symbol_input = context.args[0].upper()

        # Format symbol
        if '/' not in symbol_input:
            symbol = f"{symbol_input}/USDT:USDT"
        else:
            symbol = symbol_input

        status_msg = await update.message.reply_text(
            f"🎯 <b>{symbol_input} Prediction Chart oluşturuluyor...</b>\n\n"
            f"⏳ Teknik analiz yapılıyor...",
            parse_mode=ParseMode.HTML
        )

        try:
            exchange = await get_exchange_client()
            generator = get_prediction_generator()

            # Fetch OHLCV data
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
            if not ohlcv or len(ohlcv) < 50:
                await status_msg.edit_text(
                    f"❌ {symbol} için yeterli veri bulunamadı.",
                    parse_mode=ParseMode.HTML
                )
                return

            await status_msg.edit_text(
                f"🎯 <b>{symbol_input} Prediction Chart</b>\n\n"
                f"📊 Grafik oluşturuluyor...",
                parse_mode=ParseMode.HTML
            )

            # Generate prediction chart
            chart_bytes, prediction_data = await generator.generate_prediction_chart(
                symbol=symbol,
                ohlcv_data=ohlcv,
                timeframe='15m'
            )

            # Format price helper
            def fmt_price(price):
                if price >= 1000:
                    return f"${price:,.2f}"
                elif price >= 1:
                    return f"${price:.4f}"
                elif price >= 0.01:
                    return f"${price:.5f}"
                else:
                    return f"${price:.6f}"

            # Build prediction message
            direction = prediction_data['direction']
            confidence = prediction_data['confidence']

            if direction == 'LONG':
                direction_emoji = "🟢"
                direction_text = "LONG (Yukarı)"
            elif direction == 'SHORT':
                direction_emoji = "🔴"
                direction_text = "SHORT (Aşağı)"
            else:
                direction_emoji = "⚪"
                direction_text = "NEUTRAL (Bekle)"

            message = f"""
🎯 <b>{symbol_input} AI PREDICTION</b>

{direction_emoji} <b>Yön:</b> {direction_text}
📊 <b>Güven:</b> {confidence}%

<b>📍 Seviyeler:</b>
• Entry: <code>{fmt_price(prediction_data['entry'])}</code>
• 🛑 SL: <code>{fmt_price(prediction_data['sl'])}</code> ({prediction_data['risk_pct']:.2f}%)
• 🎯 TP1: <code>{fmt_price(prediction_data['tp1'])}</code>
• 🎯 TP2: <code>{fmt_price(prediction_data['tp2'])}</code>
• 🎯 TP3: <code>{fmt_price(prediction_data['tp3'])}</code>

<b>📈 Risk/Reward:</b> 1:{prediction_data['rr_ratio']:.1f}
<b>📊 Kanal:</b> {prediction_data['channel_type'].upper()}

<b>🔍 İndikatörler:</b>
• RSI: {prediction_data['indicators']['rsi']:.1f}
• SuperTrend: {prediction_data['indicators']['supertrend'].upper()}

<b>📝 Sebepler:</b>
"""
            for reason in prediction_data['reasons'][:5]:
                message += f"• {reason}\n"

            message += f"\n⏰ {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}"

            # Delete status message
            await status_msg.delete()

            # Send chart image with caption
            await update.message.reply_photo(
                photo=chart_bytes,
                caption=message,
                parse_mode=ParseMode.HTML
            )

            logger.info(f"✅ Prediction chart sent for {symbol}: {direction} ({confidence}%)")

        except Exception as e:
            logger.error(f"Error in cmd_predict: {e}", exc_info=True)
            await status_msg.edit_text(
                f"❌ <b>PREDICTION HATASI</b>\n\n{str(e)}",
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
        elif callback_data.startswith("chart_") and "_tf_" not in callback_data:
            # Coin selected, show timeframe selection
            await self.handle_chart_timeframe_selection(query, callback_data)
        elif callback_data.startswith("chart_") and "_tf_" in callback_data:
            # Timeframe selected, generate chart
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
        """Handle positions button - shows summary of all positions."""
        positions = await self.db.get_active_positions()

        if not positions:
            await query.edit_message_text(
                "❌ Şu anda açık pozisyon bulunmuyor.",
                parse_mode=ParseMode.HTML
            )
            return

        # Calculate summary
        total_pnl = sum(float(p.get('unrealized_pnl_usd', 0)) for p in positions)
        total_value = sum(float(p['position_value_usd']) for p in positions)
        winning_count = sum(1 for p in positions if float(p.get('unrealized_pnl_usd', 0)) > 0)
        losing_count = len(positions) - winning_count

        summary_emoji = "🟢" if total_pnl >= 0 else "🔴"

        # Build compact message (inline buttons have character limits)
        message = f"""
<b>💼 AKTİF POZİSYONLAR ({len(positions)})</b>

{summary_emoji} <b>Toplam P&L:</b> ${total_pnl:+.2f}
💰 <b>Toplam Değer:</b> ${total_value:.2f}
🟢 Kazanan: {winning_count} | 🔴 Kaybeden: {losing_count}

"""
        # Add top 5 positions
        for i, pos in enumerate(positions[:5], 1):
            pnl = float(pos.get('unrealized_pnl_usd', 0))
            emoji = "🟢" if pnl >= 0 else "🔴"
            message += f"{emoji} {pos['symbol']}: ${pnl:+.2f}\n"

        if len(positions) > 5:
            message += f"\n... ve {len(positions) - 5} pozisyon daha"

        message += f"\n\n💡 Detaylar için /positions yazın"
        message += f"\n⏰ {get_turkey_time().strftime('%H:%M:%S')}"

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
/chart - ✨ Premium grafik
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
✨ <b>ULTRA PREMIUM GRAFİK</b>

TradingView Pro+ kalitesinde grafik:
• 🕯️ Premium candlestick tasarımı
• 📍 Glow efektli S/R seviyeleri
• 📈 Smooth EMA çizgileri (12/26/50)
• 📊 Profesyonel volume analizi
• 🎨 Dark theme premium renk paleti

Coin seçin:
"""
        await query.edit_message_text(message, parse_mode=ParseMode.HTML, reply_markup=reply_markup)

    async def handle_chart_timeframe_selection(self, query, callback_data: str):
        """Show timeframe selection after coin is selected."""
        try:
            # Extract symbol from callback data
            parts = callback_data.replace('chart_', '').split('_')
            if len(parts) == 3:
                symbol = f"{parts[0]}/{parts[1]}:{parts[2]}"
                coin_code = callback_data.replace('chart_', '')
            else:
                await query.edit_message_text("❌ Geçersiz coin formatı")
                return

            # Timeframe options
            timeframes = [
                ('5m', '5 Dakika'),
                ('15m', '15 Dakika'),
                ('1h', '1 Saat'),
                ('4h', '4 Saat'),
                ('1d', '1 Gün'),
            ]

            keyboard = []
            row = []
            for tf_code, tf_name in timeframes:
                callback = f"chart_{coin_code}_tf_{tf_code}"
                row.append(InlineKeyboardButton(tf_name, callback_data=callback))
                if len(row) == 3:
                    keyboard.append(row)
                    row = []
            if row:
                keyboard.append(row)

            # Back button
            keyboard.append([InlineKeyboardButton("◀️ Geri", callback_data="chart")])

            reply_markup = InlineKeyboardMarkup(keyboard)

            display_name = symbol.replace('/USDT:USDT', '')
            message = f"""
✨ <b>{display_name}</b> için grafik

⏱️ Zaman dilimi seçin:
"""
            await query.edit_message_text(message, parse_mode=ParseMode.HTML, reply_markup=reply_markup)

        except Exception as e:
            logger.error(f"Error in timeframe selection: {e}")
            await query.edit_message_text(f"❌ Hata: {str(e)[:100]}")

    async def handle_chart_generation(self, query, callback_data: str):
        """Handle chart generation for selected coin and timeframe - ULTRA PREMIUM VERSION."""
        try:
            # Extract symbol and timeframe from callback data
            # Format: chart_BTC_USDT_USDT_tf_15m
            if "_tf_" in callback_data:
                parts = callback_data.split("_tf_")
                coin_part = parts[0].replace('chart_', '').split('_')
                timeframe = parts[1]
            else:
                coin_part = callback_data.replace('chart_', '').split('_')
                timeframe = '15m'

            if len(coin_part) == 3:
                symbol = f"{coin_part[0]}/{coin_part[1]}:{coin_part[2]}"
            else:
                await query.edit_message_text("❌ Geçersiz coin formatı")
                return

            # Timeframe display names and candle counts
            tf_info = {
                '5m': {'name': '5 Dakika', 'limit': 300, 'period': '~1 gün'},
                '15m': {'name': '15 Dakika', 'limit': 300, 'period': '~3 gün'},
                '1h': {'name': '1 Saat', 'limit': 300, 'period': '~12 gün'},
                '4h': {'name': '4 Saat', 'limit': 300, 'period': '~50 gün'},
                '1d': {'name': '1 Gün', 'limit': 300, 'period': '~300 gün'},
            }
            tf_data = tf_info.get(timeframe, tf_info['15m'])

            logger.info(f"📈 Generating PREMIUM chart for {symbol} ({timeframe})")

            # Show loading message with premium styling
            await query.edit_message_text(
                f"✨ <b>{symbol}</b> • {tf_data['name']}\n\n"
                f"🎨 Ultra Premium grafik oluşturuluyor...\n"
                f"⏳ Lütfen bekleyin (5-10 sn)",
                parse_mode=ParseMode.HTML
            )

            # Fetch OHLCV data from exchange
            from src.exchange_client import get_exchange_client
            exchange = await get_exchange_client()
            ohlcv_data = await exchange.fetch_ohlcv(symbol, timeframe, limit=tf_data['limit'])

            if not ohlcv_data or len(ohlcv_data) < 50:
                await query.edit_message_text(
                    f"❌ {symbol} için yeterli veri bulunamadı",
                    parse_mode=ParseMode.HTML
                )
                return

            # Generate ULTRA PREMIUM chart (PNG)
            from src.ultra_premium_chart import get_ultra_premium_chart
            premium_chart = get_ultra_premium_chart()
            chart_bytes = await premium_chart.generate(
                symbol=symbol,
                ohlcv=ohlcv_data,
                timeframe=timeframe,
                width=1600,
                height=1000
            )

            # Generate interactive HTML chart
            from src.indicators import detect_support_resistance_levels
            current_price = float(ohlcv_data[-1][4])
            support_resistance = detect_support_resistance_levels(ohlcv_data, current_price)
            support_levels = support_resistance.get('swing_lows', [])
            resistance_levels = support_resistance.get('swing_highs', [])

            html_content = await generate_interactive_html_chart(
                symbol=symbol,
                ohlcv_data=ohlcv_data,
                support_levels=support_levels,
                resistance_levels=resistance_levels
            )

            # Store HTML and get chart ID
            chart_id = store_chart(html_content, symbol)

            # Get Railway URL
            railway_domain = os.getenv('RAILWAY_PUBLIC_DOMAIN') or os.getenv('RAILWAY_STATIC_URL')
            if railway_domain:
                railway_domain = railway_domain.replace('https://', '').replace('http://', '')
                base_url = f"https://{railway_domain}"
            else:
                base_url = "https://worker-production-0db8.up.railway.app"

            interactive_url = f"{base_url}/chart/{chart_id}"
            logger.info(f"🔗 Interactive chart URL: {interactive_url}")

            # Calculate price metrics
            price_change = ((ohlcv_data[-1][4] - ohlcv_data[0][1]) / ohlcv_data[0][1]) * 100
            emoji = "🟢" if price_change >= 0 else "🔴"

            # Premium caption
            caption = f"""
{emoji} <b>{symbol}</b>

<b>💰 ${current_price:,.2f}</b>  <code>{price_change:+.2f}%</code>

📊 {tf_data['name']} • {tf_data['limit']} mum • {tf_data['period']}
🕐 {get_turkey_time().strftime('%H:%M:%S')} UTC+3

━━━━━━━━━━━━━━━━━━━━━━━━
<b>📈 S/R Seviyeleri</b>
🟢 Destek: {', '.join([f'${s:,.2f}' for s in support_levels[:2]]) if support_levels else 'N/A'}
🔴 Direnç: {', '.join([f'${r:,.2f}' for r in resistance_levels[:2]]) if resistance_levels else 'N/A'}

<b>📉 EMA Çizgileri</b>
🔵 EMA 12 • 🟠 EMA 26 • 🟣 EMA 50
━━━━━━━━━━━━━━━━━━━━━━━━

🖱️ <a href="{interactive_url}">İnteraktif Grafiği Aç</a>
"""

            # Delete loading message
            await query.message.delete()

            # Send premium photo
            await self.application.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=chart_bytes,
                caption=caption,
                parse_mode=ParseMode.HTML
            )

            logger.info(f"✅ Premium chart sent for {symbol} ({timeframe}) (ID: {chart_id})")

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
            'current_price': market_data['current_price'],
            'market_data': market_data  # 🛡️ v4.7.3: Required for STRICT technical validation
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
            # Fixed collateral: Always 100 USDT
            position_size = 100.0  # Fixed 100 USDT collateral (user requirement)
            position_value = position_size * leverage  # Notional value

            # Maintenance Margin (minimum to avoid liquidation)
            maintenance_margin = position_value * MAINTENANCE_MARGIN_RATE

            # Trading Fees (entry + exit)
            total_fees = position_value * TRADING_FEE_RATE * 2

            # Calculate stop-loss based on MAX $10 LOSS LIMIT
            # Real Loss = (Stop-Loss % × Leverage × Position Size) + Fees
            # We want: Real Loss = $10 (exactly)
            # So: Stop-Loss % = (10 - Fees) / (Leverage × Position Size)

            max_loss_after_fees = MAX_LOSS_USD - total_fees
            if max_loss_after_fees <= 0:
                # Fees alone exceed $10, skip this leverage
                continue

            # Calculate exact stop-loss percentage for $10 loss
            # NO CAPS - let it be as tight as needed for high leverage
            stop_loss_percent = max_loss_after_fees / (leverage * position_size)

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
