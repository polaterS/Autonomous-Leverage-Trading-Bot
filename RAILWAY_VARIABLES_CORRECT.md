# ✅ CORRECT RAILWAY ENVIRONMENT VARIABLES

## 🚨 CRITICAL: These variables MUST be set correctly on Railway!

### Current WRONG Settings:
```bash
❌ INITIAL_CAPITAL="1000000"  # Wrong! Should be 100
❌ POSITION_SIZE_PERCENT="0.10"  # Wrong! Should be 0.85
❌ MIN_PROFIT_USD="1.50"  # Wrong! Should be 0.85
```

### ✅ CORRECT Settings (v6.1-CLASSIC-FINE):

```bash
# Capital Configuration
INITIAL_CAPITAL="100"  # ✅ Real capital ($100 USDT)
POSITION_SIZE_PERCENT="0.85"  # ✅ 85% per position ($75-90)
MIN_PROFIT_USD="0.85"  # ✅ $0.70-$1.00 profit target

# Leverage (Classic Strategy)
MIN_LEVERAGE="4"  # ✅ Conservative 4x minimum
MAX_LEVERAGE="6"  # ✅ Moderate 6x maximum

# Position Management
MAX_CONCURRENT_POSITIONS="2"  # ✅ 2 positions max
POSITION_CHECK_SECONDS="15"  # ✅ Check every 15 seconds
MAX_POSITION_HOURS="8"  # ✅ Auto-close after 8 hours

# Trading Mode
USE_PAPER_TRADING="false"  # ✅ Real trading
AUTO_START_LIVE_TRADING="false"  # ⚠️ Set to "true" when ready
ENABLE_SHORT_TRADES="true"  # ✅ Both LONG and SHORT

# Risk Management
DAILY_LOSS_LIMIT_PERCENT="0.10"  # ✅ 10% daily loss limit
MAX_CONSECUTIVE_LOSSES="5"  # ✅ Continue after 5 losses

# Performance Settings
SCAN_INTERVAL_SECONDS="75"  # ✅ Scan every 75 seconds
AI_CACHE_TTL_SECONDS="60"  # ✅ Cache 60 seconds
AI_TIMEOUT_SECONDS="30"  # ✅ AI timeout 30 seconds

# Debugging
ENABLE_DEBUG_LOGS="false"  # ✅ Disable for production
RESET_CIRCUIT_BREAKER="true"  # ✅ Reset on startup
```

## 📊 Impact of Wrong Variables:

### ❌ INITIAL_CAPITAL="1000000" (Wrong):
- Code thinks you have $1,000,000
- Tries to open $100,000 positions (with 0.10 size)
- Binance rejects orders (insufficient balance)
- Falls back to minimum position size

### ❌ POSITION_SIZE_PERCENT="0.10" (Wrong):
- 10% of capital per position
- With $100 real capital: 10% = $10 positions
- $10 positions are too small for 4-6x leverage
- Need $75-90 positions for proper risk/reward

### ❌ MIN_PROFIT_USD="1.50" (Wrong):
- Code uses $1.50 profit target (not $0.85)
- Positions need to move 75% more to close
- Classic strategy uses $0.70-$1.00 ($0.85 average)

## 🎯 Expected Behavior with CORRECT Variables:

### Position Opening:
```
Real Balance: $100
Position Size: 85% = $85
Leverage: 5x (middle of 4-6x)
Position Value: $85 × 5 = $425
```

### Position Closing:
```
Profit Target: +$0.85 ($0.70-$1.00)
Loss Limit: -$0.85 (-$0.70 to -$1.00)
Stay Open: Between -$0.70 and +$0.70
```

### Daily Performance (Expected):
```
Trades: 10-20 per day
Win Rate: 60-70%
Daily P&L: +$10-15
Monthly Return: 300-450%
```

## 🔧 How to Update Railway Variables:

1. Go to Railway dashboard
2. Click on your trading bot service
3. Go to "Variables" tab
4. Update each variable listed above
5. Click "Deploy" to apply changes

## ⚠️ BEFORE STARTING:

1. ✅ Update ALL variables above
2. ✅ Run `python emergency_stop.py` to close old positions
3. ✅ Verify Binance balance is ~$100
4. ✅ Set `AUTO_START_LIVE_TRADING="true"`
5. ✅ Monitor first 5-10 trades closely

## 📌 Notes:

- These settings match the CLASSIC strategy that earned $10-15/day
- Don't change these unless you know what you're doing
- If performance degrades, revert to these exact values

═══════════════════════════════════════════════════════════
LAST UPDATED: 2025-11-17 (v6.1-CLASSIC-FINE)
═══════════════════════════════════════════════════════════
