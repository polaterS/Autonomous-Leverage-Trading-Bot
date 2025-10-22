# Quick Start Guide

Get your trading bot running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit with your credentials
nano .env  # or use any text editor
```

**Minimum required:**
- `BINANCE_API_KEY` and `BINANCE_SECRET_KEY`
- `CLAUDE_API_KEY`
- `DEEPSEEK_API_KEY`
- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`

**Important:** Keep `USE_PAPER_TRADING=true` for testing!

## Step 3: Setup Database

### Docker (Recommended):

```bash
# Start database services
docker-compose up -d postgres redis

# Wait 10 seconds, then:
python setup_database.py
```

### Without Docker:

Install PostgreSQL and Redis locally, then:

```bash
python setup_database.py
```

## Step 4: Run the Bot

```bash
# Windows
start.bat

# Linux/Mac
chmod +x start.sh
./start.sh

# Or directly
python main.py
```

## Step 5: Monitor

- Check Telegram for startup message
- Watch console logs
- Wait for first market scan (5 minutes)

## Next Steps

1. **Paper Trading (2-4 weeks)**
   - Let the bot run in paper trading mode
   - Monitor performance via Telegram
   - Ensure win rate > 50%

2. **Evaluate Performance**
   - Check database: `SELECT * FROM trade_history`
   - Review daily summaries on Telegram
   - Analyze winning vs losing trades

3. **Go Live (Only if paper trading is profitable!)**
   - Set `USE_PAPER_TRADING=false`
   - Start with small capital ($50-100)
   - Use low leverage (2x-3x)
   - Monitor closely for first week

## Troubleshooting

### "Failed to initialize"
- Check all API keys are correct
- Verify database is running
- Check internet connection

### "No good opportunities found"
- Normal! Bot scans every 5 minutes
- Market may not have strong signals
- Check AI confidence in logs

### "Trading disabled"
- Check circuit breakers in logs
- Verify `is_trading_enabled` in database
- Review daily loss limits

## Emergency Stop

Press `Ctrl+C` in console or run:

```bash
docker-compose down
```

## Get Help

1. Check `README.md` for detailed documentation
2. Review console logs for errors
3. Check Telegram alerts for warnings
4. Inspect database tables for state

---

**Remember:** ALWAYS start with paper trading mode! Never use real money until you've verified profitability.

Good luck! 🚀
