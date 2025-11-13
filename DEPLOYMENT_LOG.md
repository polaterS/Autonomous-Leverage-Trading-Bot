# Deployment Log

## 2025-11-13 23:30 - Database Fix: starting_capital

**Problem:**
- Bot showed: "MAX DRAWDOWN: 90.0% ($1000 → $100)"
- `trading_config.starting_capital` was still $1000 (old paper trading value)

**Fix:**
- Updated `trading_config.starting_capital` from $1000 to $100
- Updated `trading_config.is_trading_enabled` to true

**Verification:**
```sql
SELECT starting_capital, initial_capital, current_capital
FROM trading_config WHERE id = 1;

Result:
- starting_capital: $100.00 ✅
- initial_capital: $100.00 ✅
- current_capital: $100.00 ✅
```

**Expected Result:**
- Drawdown: 0.0% (no false alarm)
- Trading enabled
- Bot will start normally

---

## Previous Deployments

### 2025-11-13 23:25 - Contrarian Mode Disabled
- Threshold: 70% → 999% (disabled)
- Reason: Win rate 49.5%, testing simple trend-following

### 2025-11-13 23:20 - Position Sizing Fix
- Fixed leverage calculation bug
- Position sizes now correct ($100 instead of $600)

### 2025-11-13 23:00 - Live Trading Started
- Initial capital: $100
- Max leverage: 10x
- Mode: LIVE (real money)
