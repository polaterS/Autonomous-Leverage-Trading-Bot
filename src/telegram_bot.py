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
        self.application.add_handler(CommandHandler("stopbot", self.cmd_stop_bot))
        self.application.add_handler(CommandHandler("startbot", self.cmd_start_bot))
        self.application.add_handler(CommandHandler("reset", self.cmd_reset_circuit_breaker))
        self.application.add_handler(CommandHandler("setcapital", self.cmd_set_capital))
        self.application.add_handler(CommandHandler("closeall", self.cmd_close_all_positions))

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
/mlstats - ML öğrenme istatistikleri

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
/mlstats - ML öğrenme istatistikleri ve accuracy
/startbot - Botu çalıştır
/stopbot - Botu durdur
/reset - Circuit breaker'ı resetle (3 ardışık loss sonrası)
/setcapital 1000 - Capital'i güncelle (örn: $1000)

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
            # Get pattern data from ML learner
            pattern_performance = {}

            # Check if ML has symbol_patterns attribute (new ML implementation)
            if hasattr(ml, 'symbol_patterns') and ml.symbol_patterns:
                for symbol, patterns in ml.symbol_patterns.items():
                    for pattern_name, pattern_data in patterns.items():
                        if pattern_name not in pattern_performance:
                            pattern_performance[pattern_name] = {'wins': 0, 'total': 0}
                        pattern_performance[pattern_name]['wins'] += pattern_data.get('wins', 0)
                        pattern_performance[pattern_name]['total'] += pattern_data.get('total', 0)

            if pattern_performance:
                # Show top 5 best performing patterns
                sorted_patterns = sorted(
                    pattern_performance.items(),
                    key=lambda x: x[1]['wins'] / x[1]['total'] if x[1]['total'] > 0 else 0,
                    reverse=True
                )[:5]
                for pattern, stats in sorted_patterns:
                    if stats['total'] >= 2:  # At least 2 occurrences
                        acc = stats['wins'] / stats['total'] * 100
                        emoji = "🟢" if acc >= 70 else "🟡" if acc >= 50 else "🔴"
                        pattern_display = pattern.replace('_', ' ').title()
                        message += f"{emoji} {pattern_display}: {acc:.0f}% ({stats['total']})\n"
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

            # Import trade executor to close positions directly
            from src.trade_executor import get_trade_executor
            from src.exchange_client import get_exchange_client

            trade_executor = get_trade_executor()
            exchange = get_exchange_client()

            closed_count = 0
            failed_count = 0
            total_pnl = Decimal('0')
            error_details = []

            for pos in positions:
                try:
                    symbol = pos['symbol']
                    side = pos['side']
                    quantity = Decimal(str(pos['quantity']))
                    entry_price = Decimal(str(pos['entry_price']))

                    # Get current market price
                    ticker = await exchange.fetch_ticker(symbol)
                    current_price = Decimal(str(ticker['last']))

                    # Calculate P&L before closing
                    if side.upper() == 'LONG':
                        pnl = (current_price - entry_price) * quantity
                        # Close long = SELL
                        close_side = 'sell'
                    else:  # SHORT
                        pnl = (entry_price - current_price) * quantity
                        # Close short = BUY
                        close_side = 'buy'

                    logger.info(f"🔄 Closing {symbol} {side} @ ${current_price:.4f} (Entry: ${entry_price:.4f}, P&L: ${pnl:+.2f})")

                    # Close position at market price
                    order = await exchange.create_order(
                        symbol=symbol,
                        type='market',
                        side=close_side,
                        amount=float(quantity),
                        params={'reduceOnly': True}  # Important: close position only
                    )

                    if order:
                        # Record trade in database
                        await self.db.close_position(
                            symbol=symbol,
                            exit_price=current_price,
                            realized_pnl_usd=pnl,
                            exit_reason=f"Manual close via /closeall (P&L: ${pnl:+.2f})"
                        )

                        closed_count += 1
                        total_pnl += pnl
                        logger.info(f"✅ Closed {symbol}: ${pnl:+.2f}")
                    else:
                        failed_count += 1
                        error_details.append(f"{symbol}: Order failed")
                        logger.error(f"❌ Failed to close {symbol}: Order returned None")

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
            from src.exchange_client import get_exchange_client
            exchange = await get_exchange_client()
            ohlcv_data = await exchange.fetch_ohlcv(symbol, '15m', limit=500)

            if not ohlcv_data or len(ohlcv_data) < 50:
                await query.edit_message_text(
                    f"❌ {symbol} için yeterli veri bulunamadı",
                    parse_mode=ParseMode.HTML
                )
                return

            # Generate static chart (PNG)
            chart_generator = get_chart_generator()
            chart_bytes = await chart_generator.generate_chart(
                symbol=symbol,
                ohlcv_data=ohlcv_data,
                timeframe='15m',
                show_indicators=True,
                width=16,
                height=12
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

            # Get Railway URL from environment
            # Railway provides RAILWAY_PUBLIC_DOMAIN or we can construct from RAILWAY_STATIC_URL
            railway_domain = os.getenv('RAILWAY_PUBLIC_DOMAIN') or os.getenv('RAILWAY_STATIC_URL')

            if railway_domain:
                # Clean up domain (remove protocol if present)
                railway_domain = railway_domain.replace('https://', '').replace('http://', '')
                base_url = f"https://{railway_domain}"
            else:
                # Fallback: use current Railway domain
                base_url = "https://worker-production-0db8.up.railway.app"

            interactive_url = f"{base_url}/chart/{chart_id}"

            logger.info(f"🔗 Interactive chart URL: {interactive_url}")

            # Prepare caption
            price_change = ((ohlcv_data[-1][4] - ohlcv_data[0][1]) / ohlcv_data[0][1]) * 100
            emoji = "📈" if price_change >= 0 else "📉"

            caption = f"""
{emoji} <b>{symbol}</b>

💵 <b>Fiyat:</b> ${current_price:.2f} ({price_change:+.2f}%)
📊 <b>Timeframe:</b> 15 dakika (500 mum - ~5 gün geçmiş)
⏰ <b>Zaman:</b> {get_turkey_time().strftime('%Y-%m-%d %H:%M:%S')}

🎨 <b>TradingView benzeri ultra profesyonel grafik</b>
📈 <b>Destek/Direnç Seviyeleri:</b> Yeşil ve kırmızı kesikli çizgiler

━━━━━━━━━━━━━━━━━━━━━━━━
🖱️ <b>İNTERAKTİF GRAFİK:</b>
<a href="{interactive_url}">📊 Tıkla ve İnteraktif Grafiği Aç</a>

✨ Zoom, pan, hover tooltips ile detaylı analiz
✨ Geçmişe doğru kaydırarak 5 günlük veriyi incele
✨ 24 saat aktif kalacak
━━━━━━━━━━━━━━━━━━━━━━━━
"""

            # Delete loading message
            await query.message.delete()

            # Send photo with interactive link
            await self.application.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=chart_bytes,
                caption=caption,
                parse_mode=ParseMode.HTML
            )

            logger.info(f"✅ Chart sent successfully for {symbol} (ID: {chart_id})")

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
