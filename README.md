# 🤖 Autonomous Leverage Trading Bot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

![GitHub Stars](https://img.shields.io/github/stars/polaterS/Autonomous-Leverage-Trading-Bot?style=for-the-badge&logo=github)
![GitHub Forks](https://img.shields.io/github/forks/polaterS/Autonomous-Leverage-Trading-Bot?style=for-the-badge&logo=github)
![GitHub Issues](https://img.shields.io/github/issues/polaterS/Autonomous-Leverage-Trading-Bot?style=for-the-badge&logo=github)
![GitHub Last Commit](https://img.shields.io/github/last-commit/polaterS/Autonomous-Leverage-Trading-Bot?style=for-the-badge&logo=github)

**A production-ready, fully autonomous cryptocurrency leverage trading bot powered by multiple AI models**

[Features](#features) • [Quick Start](#quick-start) • [Documentation](./SETUP_GUIDE.md) • [Telegram Notifications](#telegram-commands)

</div>

---

A fully autonomous cryptocurrency leverage trading bot powered by multiple AI models (Claude, DeepSeek, Grok). The bot continuously analyzes markets, executes trades automatically, and sends real-time updates via Telegram.

## ⚠️ CRITICAL WARNINGS

**LEVERAGE TRADING IS EXTREMELY RISKY:**
- ❌ You can lose 100% of your capital in minutes
- ❌ 95% of leverage traders lose money
- ❌ AI cannot predict the future
- ❌ Markets are unpredictable and volatile

**BEFORE USING WITH REAL MONEY:**
1. ✅ Complete 2-4 weeks of paper trading
2. ✅ Verify bot is profitable in simulation
3. ✅ Start with money you can afford to lose
4. ✅ Begin with $50-100 maximum
5. ✅ Use 2x-3x leverage only initially
6. ✅ Monitor daily for first 2 weeks

**YOU ACCEPT FULL RESPONSIBILITY:**
- This is educational software
- No profit guarantees
- Past performance ≠ future results
- Developer is not liable for losses

---

## Features

### Core Functionality
- ✅ Fully autonomous trading (no manual intervention required)
- ✅ Multi-AI consensus system (Claude, DeepSeek, Grok)
- ✅ Automatic position management with strict risk controls
- ✅ Real-time Telegram notifications for all events
- ✅ Paper trading mode for safe testing
- ✅ Comprehensive risk management and circuit breakers

### Risk Management (MANDATORY)
- 🛑 Every trade has a 5-10% stop-loss
- 💰 Minimum $2.50 profit target before closing winners
- ⚡ Auto-close if liquidation distance falls below 5%
- 📊 Daily loss circuit breaker (10% max)
- 🔄 Consecutive loss limiter (3 losses → pause)
- 📈 Position size: Maximum 80% of capital per trade

### AI Analysis
- 🤖 Multi-model consensus (2 out of 3 AI models must agree)
- 📊 Multi-timeframe analysis (15m, 1h, 4h)
- 📈 Technical indicators: RSI, MACD, Bollinger Bands, Volume, SMA/EMA
- 🎯 Market regime detection: Trending, Ranging, or Volatile
- 💎 Minimum 75% confidence threshold for trades

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- PostgreSQL 15+ (or use Docker)
- Redis 7+ (or use Docker)
- Binance Futures account with API keys
- AI API keys (Claude, DeepSeek)
- Telegram bot token

### 2. Installation

```bash
# Clone or extract the project
cd "Autonomous Leverage Trading Bot"

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your actual credentials
nano .env
```

### 3. Configuration

Edit `.env` file:

```env
# Exchange API (REQUIRED)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# AI APIs (REQUIRED)
CLAUDE_API_KEY=sk-ant-your-claude-key
DEEPSEEK_API_KEY=your_deepseek_key
GROK_API_KEY=your_grok_key  # Optional

# Telegram (REQUIRED)
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789

# Database (adjust if not using Docker)
DATABASE_URL=postgresql://trading_user:changeme123@localhost:5432/trading_bot
REDIS_URL=redis://localhost:6379/0

# Trading Settings
INITIAL_CAPITAL=100.00
MAX_LEVERAGE=5
USE_PAPER_TRADING=true  # KEEP THIS TRUE FOR TESTING!
```

### 4. Get API Keys

#### Binance Futures API
1. Go to https://www.binance.com/en/my/settings/api-management
2. Create new API key with **trading permissions only**
3. **IMPORTANT**: Disable withdrawal permissions for security
4. Save API key and secret

#### Claude API
1. Go to https://console.anthropic.com/
2. Create account and get API key
3. Ensure you have credits

#### DeepSeek API
1. Go to https://platform.deepseek.com/
2. Create account and get API key

#### Telegram Bot
1. Talk to @BotFather on Telegram
2. Create new bot with `/newbot`
3. Save bot token
4. Get your chat ID:
   - Talk to @userinfobot
   - Save your user ID

### 5. Setup Database

#### Option A: Docker (Recommended)

```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Wait 10 seconds for database to start
# Then run setup script
python setup_database.py
```

#### Option B: Local Installation

```bash
# Install PostgreSQL and Redis locally
# Then run:
python setup_database.py
```

### 6. Run the Bot

#### Development Mode

```bash
# Run directly
python main.py
```

#### Production Mode with Docker

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f trading-bot

# Stop bot
docker-compose down
```

---

## Testing Strategy

### Phase 1: Paper Trading (MANDATORY - 2-4 weeks)

```bash
# Ensure paper trading is enabled in .env
USE_PAPER_TRADING=true

# Start bot
python main.py
```

**Success Criteria:**
- Bot runs continuously for 2+ weeks without crashes
- Win rate > 50%
- Average profit per trade > $1.00
- No critical bugs or logic errors
- Telegram notifications working correctly

⚠️ **If paper trading is NOT profitable, DO NOT proceed to real money!**

### Phase 2: Micro-Capital Test ($50-100, 2 weeks)

After successful paper trading:

```bash
# Disable paper trading in .env
USE_PAPER_TRADING=false

# Start with small capital
INITIAL_CAPITAL=50.00
MAX_LEVERAGE=3

# Start bot
python main.py
```

**Monitor:**
- All trades executed correctly
- Stop-losses trigger as expected
- Telegram notifications accurate
- No exchange API errors

### Phase 3: Full Operation ($200-500+)

After successful micro-capital test:

```bash
# Increase capital in .env
INITIAL_CAPITAL=200.00
MAX_LEVERAGE=5
```

---

## Project Structure

```
Autonomous Leverage Trading Bot/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration & settings
│   ├── utils.py               # Utility functions
│   ├── database.py            # Database client
│   ├── exchange_client.py     # CCXT wrapper for Binance
│   ├── indicators.py          # Technical indicators
│   ├── ai_engine.py           # AI consensus engine
│   ├── risk_manager.py        # Risk management system
│   ├── telegram_notifier.py   # Telegram notifications
│   ├── trade_executor.py      # Trade execution
│   ├── position_monitor.py    # Position monitoring
│   ├── market_scanner.py      # Market scanning
│   └── trading_engine.py      # Main trading engine
├── main.py                    # Entry point
├── setup_database.py          # Database setup script
├── health_server.py           # Health check endpoint
├── schema.sql                 # Database schema
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker image
├── docker-compose.yml         # Docker services
├── .env.example               # Environment template
└── README.md                  # This file
```

---

## How It Works

### Main Loop

```
┌─────────────────────────────────────┐
│  Check if trading is enabled        │
│  (circuit breakers, limits)         │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Do we have an active position?     │
└────────┬───────────────┬────────────┘
         │ YES           │ NO
         ▼               ▼
┌────────────────┐  ┌────────────────┐
│ Monitor        │  │ Scan markets   │
│ Position       │  │ for            │
│ (every 60s)    │  │ opportunities  │
│                │  │ (every 5 min)  │
└────────────────┘  └────────────────┘
         │               │
         └───────┬───────┘
                 ▼
         Continue loop...
```

### Position Monitoring

When a position is open, the bot checks every 60 seconds:

1. **Critical Checks:**
   - Liquidation distance < 5%? → Emergency close
   - Stop-loss hit? → Close position

2. **Profit Targets:**
   - P&L ≥ $2.50 minimum? → Consider taking profit
   - P&L ≥ 2x minimum ($5+)? → Close immediately

3. **AI Exit Signal** (every 5 minutes):
   - Get fresh AI analysis
   - If AI recommends exit → Close position

4. **Updates:**
   - Send Telegram update every 5 minutes

### Market Scanning

When no position is open, the bot scans every 5 minutes:

1. **Gather Data** for each symbol:
   - OHLCV data (15m, 1h, 4h)
   - Technical indicators
   - Market regime
   - Funding rate

2. **AI Analysis:**
   - Get consensus from 2-3 AI models
   - Calculate opportunity score (0-100)
   - Rank all opportunities

3. **Execute Best Trade:**
   - If score ≥ 80 → Execute trade
   - If score 75-80 → Wait for better setup
   - If score < 75 → Skip

---

## Configuration Reference

### Trading Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `INITIAL_CAPITAL` | 100.00 | > 10 | Starting capital in USD |
| `MAX_LEVERAGE` | 5 | 1-10 | Maximum leverage multiplier |
| `POSITION_SIZE_PERCENT` | 0.80 | 0.1-1.0 | % of capital per trade |
| `MIN_STOP_LOSS_PERCENT` | 0.05 | 0.01-0.20 | Minimum stop-loss (5%) |
| `MAX_STOP_LOSS_PERCENT` | 0.10 | 0.05-0.30 | Maximum stop-loss (10%) |
| `MIN_PROFIT_USD` | 2.50 | > 0 | Minimum profit to close |
| `MIN_AI_CONFIDENCE` | 0.75 | 0.5-1.0 | Minimum AI confidence (75%) |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DAILY_LOSS_LIMIT_PERCENT` | 0.10 | Max 10% daily loss |
| `MAX_CONSECUTIVE_LOSSES` | 3 | Pause after 3 losses |

### Timing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SCAN_INTERVAL_SECONDS` | 300 | Scan every 5 minutes |
| `POSITION_CHECK_SECONDS` | 60 | Monitor every 60 seconds |

---

## Telegram Commands

The bot sends these notifications automatically:

- 🤖 **Startup**: Bot started confirmation
- 🟢/🔴 **Position Opened**: Entry details with stop-loss and targets
- 💰/📉 **Position Update**: P&L updates every 5 minutes
- ✅/❌ **Position Closed**: Exit details with profit/loss
- 💼 **Portfolio Status**: Capital and daily P&L
- 🔍 **Scan Results**: Best opportunities found
- ⚠️ **Alerts**: Warnings, errors, circuit breakers
- 📊 **Daily Summary**: End-of-day performance report

---

## Monitoring & Maintenance

### View Logs

```bash
# Docker
docker-compose logs -f trading-bot

# Direct
# Logs are printed to console
```

### Check Health

```bash
curl http://localhost:8000/health
```

### Database Queries

```sql
-- Check active position
SELECT * FROM active_position;

-- Recent trades
SELECT * FROM trade_history ORDER BY exit_time DESC LIMIT 10;

-- Today's performance
SELECT * FROM daily_performance WHERE date = CURRENT_DATE;

-- Current capital
SELECT current_capital FROM trading_config WHERE id = 1;
```

### Emergency Stop

```bash
# Stop the bot
docker-compose down

# Or press Ctrl+C if running directly
```

### Disable Trading

```sql
-- Pause trading without stopping bot
UPDATE trading_config SET is_trading_enabled = false WHERE id = 1;

-- Resume trading
UPDATE trading_config SET is_trading_enabled = true WHERE id = 1;
```

---

## Troubleshooting

### Bot won't start

- Check all API keys are correct
- Ensure database is running
- Check logs for specific error

### No trades being executed

- Check `USE_PAPER_TRADING` setting
- Verify capital > $12.50
- Check if circuit breakers activated
- Review AI confidence levels in logs

### Exchange API errors

- Verify API keys have trading permissions
- Check Binance API rate limits
- Ensure futures account is enabled

### Telegram not working

- Verify bot token is correct
- Check chat ID is your user ID
- Send a message to bot first

---

## Cost Analysis

### Monthly Operating Costs

**For $100-300 Capital (Conservative Trading):**

| Service | Cost/Month |
|---------|------------|
| Claude API | $15-30 |
| DeepSeek API | $5-10 |
| Grok API (optional) | $10-15 |
| DigitalOcean Droplet | $12 |
| **Total** | **$42-67** |

**Cost Optimization:**
- Use DeepSeek for routine scans (80% of calls)
- Use Claude only for entry/exit decisions (20%)
- Enable AI caching (reduces costs by 60%)
- **Optimized Total: $25-35/month**

**Recommendation:**
- Minimum $200-300 capital to keep costs reasonable
- With $300 capital, operating costs = ~10% per month

---

## Support & Development

### Report Issues

If you encounter issues:
1. Check logs for error messages
2. Review Telegram alerts
3. Check database state
4. Verify all API keys are valid

### Customize

Want to modify the bot?

- **Add symbols**: Edit `trading_symbols` in `src/config.py`
- **Adjust risk**: Modify `.env` parameters
- **Change AI prompts**: Edit `src/config.py`
- **Custom indicators**: Add to `src/indicators.py`

---

## Disclaimer

This software is provided for educational purposes only. Trading cryptocurrency with leverage is extremely risky and you can lose all your capital. The developers are not responsible for any losses incurred while using this software.

**By using this bot, you acknowledge:**
- You understand the risks of leverage trading
- You have tested thoroughly in paper trading mode
- You are using only capital you can afford to lose
- You accept full responsibility for all trading decisions
- No profits are guaranteed

---

## License

This project is open source for educational purposes. Use at your own risk.

---

## Acknowledgments

Built with:
- CCXT for exchange integration
- Claude API by Anthropic
- DeepSeek API
- Telegram Bot API
- PostgreSQL & Redis
- Python asyncio

---

**Happy (and safe) trading! 🚀**

Remember: Start with paper trading, test extensively, and never invest more than you can afford to lose.
