# 🖥️ Windows GUI Dashboard

Professional monitoring and control interface for the Autonomous Leverage Trading Bot.

## ✨ Features

### 📊 Real-Time Dashboard
- **Live Metrics**: Current capital, daily P&L, total P&L
- **Win Rate Tracking**: Win percentage, total trades, average win/loss
- **Active Position Monitor**: Real-time position tracking with P&L
- **Bot Status**: CPU, Memory usage, PID monitoring
- **Bot Control**: Start, Stop, Restart buttons

### 📈 Trade History
- **Complete Trade Log**: All executed trades with timestamps
- **Detailed Metrics**: Entry/exit prices, P&L, duration
- **Color-Coded**: Green for profits, red for losses
- **Statistics Summary**: Winners, losers, win rate at a glance

### 📉 Performance Charts
- **Capital Over Time**: Interactive line chart showing portfolio growth
- **Last 30 Days**: Historical performance visualization
- **Real-Time Updates**: Auto-refresh every 30 seconds

### 📝 Live Logs Viewer
- **Real-Time Monitoring**: See bot logs as they happen
- **Level Filtering**: Filter by INFO, WARNING, ERROR, DEBUG
- **Search Function**: Find specific log entries quickly
- **Color-Coded Levels**: Easy visual identification
- **Auto-Scroll**: Optional auto-scroll to latest logs

### ⚙️ Configuration Panel
- **Trading Parameters**: Leverage, position size, stop-loss
- **Risk Management**: Daily loss limits, consecutive loss limits
- **AI Settings**: Minimum confidence threshold
- **Live Updates**: Save changes to database instantly

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Navigate to windows_app directory
cd windows_app

# Install GUI requirements
pip install -r requirements_gui.txt
```

**Dependencies installed:**
- PyQt6 (GUI framework)
- PyQt6-Charts (Charts)
- psycopg2-binary (PostgreSQL)
- pandas, numpy (Data processing)

### 2. Ensure Services are Running

```bash
# Start PostgreSQL and Redis (if using Docker)
docker-compose up -d postgres redis

# Or ensure they're running locally
```

### 3. Run the Application

```bash
# From project root
python windows_app/app.py

# Or from windows_app directory
python app.py
```

---

## 📐 Application Architecture

```
windows_app/
├── app.py                      # Main entry point
├── models/                     # 🆕 Data models (NEW)
│   ├── trading_config.py      # TradingConfig with validation
│   ├── trade.py               # Trade with analytics
│   └── position.py            # ActivePosition with calculations
├── controllers/                # 🚀 Enhanced business logic (PRODUCTION-READY)
│   ├── db_controller.py       # Database with connection pooling & retry logic
│   ├── bot_controller.py      # Bot control with health monitoring & crash recovery
│   └── log_controller.py      # Logs with advanced parsing & event detection
├── ui/                         # User interface
│   ├── main_window.py         # Main window with tabs
│   ├── dashboard_widget.py    # Dashboard tab
│   ├── trades_widget.py       # Trade history tab
│   ├── charts_widget.py       # Charts tab
│   ├── logs_widget.py         # Logs viewer tab
│   ├── config_widget.py       # Settings tab
│   └── styles.py              # Dark theme styles
├── resources/                  # 🎨 Static resources (NEW)
│   └── icons/                 # 25 professional icons
│       ├── generate_icons.py  # Icon generator script
│       ├── ICONS.md           # Icon specifications
│       └── [25 icons]         # Auto-generated assets
└── requirements_gui.txt        # GUI dependencies
```

### Design Patterns Used

1. **MVC Pattern**: Model (models/) - View (UI) - Controller (controllers/) separation
2. **Singleton Controllers**: Single database/bot/log controller instances
3. **Observer Pattern**: QTimer for real-time updates
4. **Component-Based**: Reusable widgets (MetricCard, etc.)
5. **🆕 Data Models**: Type-safe dataclasses for all entities
6. **🆕 Repository Pattern**: Centralized data access via db_controller
7. **🆕 Connection Pooling**: Efficient database resource management
8. **🆕 Event-Driven**: Callbacks for log streaming and bot status changes

---

## 🚀 Production-Ready Features (NEW)

### 🗂️ Data Models Layer

**NEW in v2.0**: Comprehensive type-safe data models with business logic.

#### TradingConfig Model
```python
from models.trading_config import TradingConfig

config = TradingConfig(
    initial_capital=Decimal("100.00"),
    max_leverage=5,
    min_ai_confidence=Decimal("0.60"),
    # ... more fields
)

# Automatic validation
config.validate()  # Raises ValueError if invalid

# Business logic methods
risk_level = config.get_risk_level()  # Returns 'LOW', 'MEDIUM', or 'HIGH'
max_loss = config.calculate_max_loss_per_trade()
is_safe, reason = config.is_safe_to_trade()  # Safety check
```

**Features:**
- Comprehensive validation (leverage 1-20, confidence 50-100%, etc.)
- Risk level calculation based on multiple factors
- Maximum loss per trade calculation
- Safety checks before trading

#### Trade Model
```python
from models.trade import Trade

trade = Trade(
    symbol="BTCUSDT",
    side="LONG",
    realized_pnl_usd=Decimal("5.50"),
    # ... more fields
)

# Analytics methods
roi = trade.get_roi()  # Return on investment
rr_ratio = trade.get_risk_reward_ratio()  # Risk/reward ratio
summary = trade.get_summary()  # "BTCUSDT LONG 5x: +$5.50 (+3.25%) in 2h 15m"
duration = trade.duration_str  # "2h 15m"
```

**Features:**
- Automatic winner/loser classification
- ROI calculations
- Risk/reward ratio analysis
- Human-readable duration formatting
- Color properties for UI (pnl_color, side_color)

#### ActivePosition Model
```python
from models.position import ActivePosition

position = ActivePosition(
    symbol="BTCUSDT",
    entry_price=Decimal("50000"),
    current_price=Decimal("51000"),
    # ... more fields
)

# Real-time calculations
unrealized_pnl = position.calculate_unrealized_pnl()
pnl_percent = position.get_pnl_percent()
distance_to_sl = position.get_distance_to_stop_loss()  # Percentage
distance_to_liq = position.get_distance_to_liquidation()

# Risk checks
if position.is_near_stop_loss(threshold_percent=2.0):
    print("⚠️ Dangerously close to stop-loss!")

if position.reached_min_profit():
    print("✓ Minimum profit target reached")

# Comprehensive status
status = position.get_status_summary()
# Returns dict with all key metrics and flags
```

**Features:**
- Real-time P&L calculations with leverage
- Distance to stop-loss and liquidation (%)
- Risk assessment (near SL/liquidation checks)
- Profit target tracking
- Comprehensive status summaries

### 🗄️ Enhanced Database Controller

**NEW in v2.0**: Production-ready database operations with connection pooling.

#### Connection Pooling
```python
from controllers.db_controller import DatabaseController

# Create pool with 2-10 connections
db = DatabaseController(min_conn=2, max_conn=10)
db.connect()

# Health check
if db.is_healthy():
    print("Database connection healthy")
```

**Features:**
- **Connection pooling**: 2-10 concurrent connections for high performance
- **Retry logic**: Automatic retry with exponential backoff (3 attempts)
- **Transaction management**: Execute multiple operations atomically
- **Health monitoring**: Connection health checks

#### Model Integration
```python
# Get data as models (type-safe)
config: TradingConfig = db.get_config()
position: ActivePosition = db.get_active_position_model()
trades: List[Trade] = db.get_trades(limit=50, winners_only=True)

# Save validated models
new_config = TradingConfig(...)
db.save_config(new_config)  # Validates before saving
```

**Features:**
- Type-safe model instances
- Automatic validation on save
- Backward compatibility with dict methods

#### Advanced Statistics
```python
# Comprehensive statistics
stats = db.get_statistics()
print(f"Win rate: {stats['win_rate']:.1f}%")
print(f"Profit factor: {stats['profit_factor']:.2f}")
print(f"Best trade: ${stats['best_trade']:.2f}")
print(f"Worst trade: ${stats['worst_trade']:.2f}")

# Recent performance
perf = db.get_recent_performance(days=7)
print(f"Last 7 days: {perf['total_trades']} trades, {perf['win_rate']:.1f}% win rate")
```

**NEW Methods:**
- `get_config()` → TradingConfig model
- `get_active_position_model()` → ActivePosition model
- `get_trades(limit, winners_only)` → List[Trade]
- `save_config(config)` → Validated save
- `get_recent_performance(days)` → Time-windowed stats
- `execute_transaction(operations)` → Atomic multi-op
- `is_healthy()` → Health check

### 🤖 Enhanced Bot Controller

**NEW in v2.0**: Comprehensive process management with health monitoring.

#### Health Monitoring
```python
from controllers.bot_controller import BotController

bot = BotController(auto_restart=True, max_restarts=5)

# Start with validation
bot.start_bot(validate_first=True)

# Get comprehensive status
status = bot.get_bot_status()
print(f"PID: {status['pid']}, Uptime: {status['uptime_str']}")
print(f"CPU: {status['cpu_percent']:.1f}%, Memory: {status['memory_mb']:.1f} MB")
print(f"Restarts: {status['restart_count']}, Crashes: {status['crash_count']}")

# Health assessment
health = bot.get_health_status()
if not health['healthy']:
    print(f"Issues: {health['issues']}")
    print(f"Warnings: {health['warnings']}")
```

**Features:**
- **Process metrics**: CPU, memory, uptime tracking
- **Crash detection**: Automatic detection and optional recovery
- **Lifecycle tracking**: Start time, stop time, restart/crash counts
- **Health assessment**: Evaluates CPU/memory thresholds

#### Auto-Restart & Crash Recovery
```python
# Enable auto-restart on crashes (max 5 attempts)
bot = BotController(auto_restart=True, max_restarts=5)
bot.start_bot()

# Manual crash check and recovery
if bot.check_for_crash():
    print("Crash detected and recovery attempted")

# Metrics history
metrics = bot.get_metrics_summary(last_n_minutes=60)
print(f"Avg CPU: {metrics['avg_cpu']:.1f}%, Max: {metrics['max_cpu']:.1f}%")
print(f"Avg Memory: {metrics['avg_memory_mb']:.1f} MB")
```

**Features:**
- **Auto-restart**: Configurable automatic restart on crashes
- **Metrics history**: Rolling window of 1000 metrics snapshots
- **Position safety**: Checks for active positions before stopping
- **Startup validation**: Validates environment before starting

#### Status Callbacks
```python
# Register callback for status changes
def on_status_change(status: str):
    print(f"Bot {status}")

bot.register_status_callback(on_status_change)
# Fires on: 'started', 'stopped', 'crashed', 'restarted'
```

**NEW Methods:**
- `start_bot(validate_first)` → Validated startup
- `stop_bot(force, check_position)` → Safe shutdown
- `get_bot_status()` → CPU, memory, uptime, crashes
- `get_health_status()` → Health with issues/warnings
- `check_for_crash()` → Auto-detect and recover
- `get_metrics_summary(last_n_minutes)` → Historical metrics
- `get_lifecycle_summary()` → Complete lifecycle data
- `register_status_callback(callback)` → Event notifications

### 📝 Enhanced Log Controller

**NEW in v2.0**: Advanced log parsing with event detection.

#### Real-Time Streaming
```python
from controllers.log_controller import LogController, LogLevel

logs = LogController()

# Register stream callback
def on_new_logs(log_lines: List[str]):
    print(f"Received {len(log_lines)} new logs")

logs.register_stream_callback(on_new_logs)

# Get new logs (triggers callbacks)
new_logs = logs.get_new_logs(parse=True, notify=True)
```

**Features:**
- **Callback system**: Register callbacks for new log streams
- **Event detection**: Automatic detection of trades, errors, warnings
- **Log rotation**: Automatic detection and handling
- **Structured parsing**: ParsedLog objects with extracted data

#### Advanced Parsing & Event Detection
```python
from controllers.log_controller import ParsedLog, LogLevel

# Parse with data extraction
parsed_logs = logs.get_recent_logs(lines=100, parse=True)
for log in parsed_logs:
    print(f"{log.timestamp}: [{log.level.value}] {log.message}")
    if log.symbol:
        print(f"  Symbol: {log.symbol}")
    if log.action:
        print(f"  Action: {log.action}")
    if log.price:
        print(f"  Price: ${log.price}")
    if log.pnl:
        print(f"  P&L: ${log.pnl:+.2f}")

# Register event callbacks
def on_trade_entry(parsed: ParsedLog):
    print(f"Trade entered: {parsed.symbol} at ${parsed.price}")

def on_error(parsed: ParsedLog):
    print(f"Error detected: {parsed.message}")

logs.register_event_callback('trade_entry', on_trade_entry)
logs.register_event_callback('trade_exit', lambda p: print(f"Trade exited"))
logs.register_event_callback('error', on_error)
logs.register_event_callback('warning', lambda p: print(f"Warning: {p.message}"))
```

**Features:**
- **Data extraction**: Automatically extracts symbols, prices, P&L, actions
- **Event callbacks**: Register handlers for specific events
- **Structured data**: ParsedLog with timestamp, level, component, message, + extracted data

#### Search & Filter
```python
# Regex search
errors = logs.search_logs(regex=r'ERROR.*timeout', max_results=50)
btc_trades = logs.search_logs(keyword='BTCUSDT', case_sensitive=False)

# Filter by level
error_logs = logs.filter_by_level(LogLevel.ERROR, lines=100)
warnings = logs.filter_by_level(LogLevel.WARNING, lines=50)

# Get errors and warnings from last 24 hours
issues = logs.get_errors_and_warnings(hours=24)
print(f"Errors: {len(issues['errors'])}")
print(f"Warnings: {len(issues['warnings'])}")
```

**Features:**
- **Regex support**: Pattern-based search with case sensitivity
- **Level filtering**: Filter by INFO, WARNING, ERROR, DEBUG
- **Time-windowed queries**: Get errors/warnings from recent hours

#### Statistics & Analytics
```python
# Log statistics
stats = logs.get_statistics()
print(f"Total lines: {stats['total_lines']}")
print(f"Error rate: {stats['error_rate']:.2f}%")
print(f"By level: {stats['by_level']}")
print(f"Trade entries: {stats['trade_entries']}")
print(f"Trade exits: {stats['trade_exits']}")
print(f"Last error: {stats['last_error']}")

# Archive logs
logs.archive_logs()  # Archives to logs/bot_YYYYMMDD_HHMMSS.log
```

**Features:**
- **Statistics**: Error rate, counts by level, trade events
- **Archiving**: Timestamp-based log archiving
- **Reset**: Clear statistics and log files

**NEW Methods:**
- `get_recent_logs(lines, parse)` → String list or ParsedLog list
- `get_new_logs(parse, notify)` → Real-time streaming
- `search_logs(keyword, regex, case_sensitive)` → Pattern search
- `filter_by_level(level, lines)` → Level-based filtering
- `get_errors_and_warnings(hours)` → Recent issues
- `parse_log_line_advanced(line)` → Structured ParsedLog
- `get_statistics()` → Comprehensive log stats
- `archive_logs(archive_path)` → Log archiving
- `register_stream_callback(callback)` → Real-time notifications
- `register_event_callback(event_type, callback)` → Event detection

### 🎨 Icon System

**NEW in v2.0**: 25 professional icons with consistent theme.

#### Generated Icons
- **Application Icons**: app.ico (16/32/48/256), splash_logo.png (200x200)
- **Tab Icons** (24x24): dashboard, trades, charts, logs, settings
- **Status Icons** (16x16): bot_running, bot_stopped, bot_warning
- **Trade Icons** (16x16): long, short, profit, loss
- **Action Icons** (20x20): start, stop, restart, refresh, save, reset, export
- **Log Icons** (14x14): log_info, log_warning, log_error, log_debug

#### Regenerate Icons
```bash
cd windows_app/resources/icons
python generate_icons.py
# Generates all 25 icons in ~1 second
```

#### Use in Code
```python
from PyQt6.QtGui import QIcon
from pathlib import Path

ICON_DIR = Path(__file__).parent / "resources" / "icons"

def get_icon(name: str) -> QIcon:
    icon_path = ICON_DIR / f"{name}.png"
    return QIcon(str(icon_path)) if icon_path.exists() else QIcon()

# Usage
dashboard_icon = get_icon("dashboard")
bot_status_icon = get_icon("bot_running")
```

**Color Palette:**
- Blue (#5E81AC): Dashboard, charts, actions
- Green (#A3BE8C): Profits, long positions
- Red (#BF616A): Losses, short positions
- Yellow (#EBCB8B): Warnings, logs
- Purple (#B48EAD): Charts
- Orange (#D08770): Reset actions

See `resources/icons/ICONS.md` for complete specifications.

---

## 🎨 User Interface Guide

### Dashboard Tab

**Top Section - Bot Control:**
- `● ONLINE/OFFLINE`: Bot status indicator
- `▶ Start Bot`: Launch the trading bot
- `■ Stop Bot`: Gracefully stop the bot
- `↻ Restart`: Restart the bot process

**Metrics Grid:**
- **Current Capital**: Total portfolio value
- **Today's P&L**: Profit/loss for today
- **Total P&L**: All-time profit/loss
- **Win Rate**: Percentage of winning trades
- **Total Trades**: Number of completed trades
- **Avg Win**: Average winning trade amount

**Active Position:**
- **Entry/Current Price**: Position entry and current market price
- **Quantity**: Position size
- **P&L**: Real-time unrealized profit/loss
- **Stop-Loss**: Automatic exit price
- **Liquidation**: Liquidation price

### Trades Tab

**Table Columns:**
- **Date/Time**: Trade execution timestamp
- **Symbol**: Trading pair (e.g., BTC/USDT)
- **Side**: LONG or SHORT
- **Leverage**: Leverage multiplier used
- **Entry/Exit**: Entry and exit prices
- **Quantity**: Position size
- **P&L $**: Profit/loss in USD
- **P&L %**: Profit/loss percentage
- **Duration**: How long the trade lasted

**Features:**
- Auto-refresh every 5 seconds
- Color-coded P&L (green = profit, red = loss)
- Last 100 trades shown
- Statistics summary at top

### Charts Tab

**Capital Over Time Chart:**
- Line chart showing portfolio value
- Last 30 days of data
- Interactive hover tooltips
- Auto-refresh every 30 seconds

### Logs Tab

**Log Viewer:**
- Real-time log streaming
- Color-coded by level:
  - INFO: White
  - WARNING: Yellow
  - ERROR: Red
  - DEBUG: Gray

**Controls:**
- **Level Filter**: Show only specific log levels
- **Search**: Find specific log entries
- **Auto-scroll**: Toggle automatic scrolling
- **Clear Logs**: Clear log file and display

### Settings Tab

**Trading Parameters:**
- **Max Leverage**: 1-10x
- **Position Size %**: Percentage of capital per trade
- **Min/Max Stop-Loss %**: Stop-loss range
- **Min Profit USD**: Minimum profit target
- **Min AI Confidence**: AI confidence threshold (50-100%)

**Risk Management:**
- **Daily Loss Limit %**: Maximum daily loss before circuit breaker
- **Max Consecutive Losses**: Losses before pause
- **Trading Enabled**: Master on/off switch

**Buttons:**
- **💾 Save Configuration**: Save changes to database
- **↻ Reset to Defaults**: Restore default values

---

## 🔧 Configuration

### Database Connection

The GUI reads `DATABASE_URL` from the `.env` file in the project root.

```env
DATABASE_URL=postgresql://trading_user:changeme123@localhost:5432/trading_bot
```

### Log File Location

By default, logs are read from `bot.log` in the project root. You can customize this in `LogController`.

---

## ⚡ Performance & Optimization

### Auto-Refresh Intervals

- **Dashboard**: 2 seconds (real-time)
- **Trades**: 5 seconds
- **Charts**: 30 seconds
- **Logs**: 1 second (new lines only)
- **Status Bar**: 2 seconds

### Memory Management

- Tables limited to 100 recent entries
- Logs limited to last 500 lines on load
- Charts limited to 30 days

### Database Queries

- Connection pooling via psycopg2
- Single connection reused across widgets
- Prepared statements for efficiency

---

## 🐛 Troubleshooting

### "Failed to connect to database"

**Solution:**
1. Ensure PostgreSQL is running:
   ```bash
   docker-compose ps postgres
   ```
2. Check DATABASE_URL in `.env`
3. Test connection:
   ```bash
   docker-compose exec postgres psql -U trading_user -d trading_bot
   ```

### "Bot won't start"

**Solution:**
1. Check if bot is already running elsewhere
2. Ensure `main.py` exists in project root
3. Check bot.log for errors
4. Verify Python environment has all dependencies

### Charts not showing

**Solution:**
1. Ensure PyQt6-Charts is installed:
   ```bash
   pip install PyQt6-Charts
   ```
2. Check database has daily_performance data:
   ```sql
   SELECT * FROM daily_performance;
   ```

### GUI is slow/laggy

**Solution:**
1. Increase refresh intervals in widget constructors
2. Reduce table row limits
3. Check CPU/memory usage of bot process
4. Close other applications

---

## 🎯 Best Practices

### Daily Workflow

1. **Morning**:
   - Launch GUI
   - Check Dashboard for overnight performance
   - Review active positions
   - Check logs for errors

2. **During Trading Hours**:
   - Monitor Dashboard for real-time P&L
   - Watch Logs for AI decisions
   - Check Trades tab for completed trades

3. **Evening**:
   - Review Charts for performance trends
   - Analyze trade history for patterns
   - Adjust Settings if needed
   - Check daily performance metrics

### Monitoring Tips

- Keep Dashboard tab open for real-time updates
- Use Logs tab to understand bot decisions
- Check Charts weekly for trend analysis
- Review Settings after significant changes

### Configuration Changes

- ⚠️ **IMPORTANT**: Stop bot before changing critical settings
- Test new configurations in paper trading first
- Save configurations regularly
- Document changes and their effects

---

## 🔐 Security Notes

### Database Access

- GUI has **read/write** access to trading_config
- Cannot directly modify trade_history (read-only)
- Cannot delete trades

### Bot Control

- GUI can start/stop bot process
- Uses process signals for graceful shutdown
- Cannot modify running bot's internal state

### Log Access

- Read-only access to log files
- Can clear logs but cannot modify entries
- Search function is client-side (no SQL injection risk)

---

## 📊 Screenshots

### Dashboard
```
┌──────────────────────────────────────────────┐
│ Bot: 🟢 RUNNING                              │
│ Current Capital: $150.00 | Today P&L: +$5.20│
│ ┌────────┐  ┌────────┐  ┌────────┐          │
│ │Capital │  │Today's │  │Total   │          │
│ │$150.00 │  │+$5.20  │  │+$50.00 │          │
│ └────────┘  └────────┘  └────────┘          │
│ Active Position: BTC/USDT LONG 3x            │
│ Entry: $50,000 | P&L: +$2.50                 │
└──────────────────────────────────────────────┘
```

### Trades Table
```
Date/Time       Symbol    Side  Entry     Exit      P&L
2024-01-15 10:30  BTC/USDT  LONG  $50,000  $50,500  +$5.20
2024-01-15 09:15  ETH/USDT  SHORT $3,000   $2,950   +$3.80
2024-01-15 08:00  BTC/USDT  LONG  $49,500  $49,400  -$2.10
```

---

## 🆘 Support

### Common Issues

| Issue | Solution |
|-------|----------|
| Black screen | Check GPU drivers, try software rendering |
| Frozen UI | Increase refresh intervals, reduce data limits |
| Database errors | Verify PostgreSQL connection, check schema |
| Bot won't stop | Use "Force Kill" or Task Manager |

### Logs Location

- **Application logs**: `bot.log`
- **Database logs**: Docker logs or PostgreSQL logs
- **GUI errors**: Console output when running `app.py`

### Getting Help

1. Check logs for error messages
2. Review this README
3. Check main project README.md
4. Open GitHub issue with:
   - Error message
   - Steps to reproduce
   - Screenshots
   - Log excerpts

---

## 🚀 Future Enhancements

Potential features for future versions:

- [ ] **Multi-bot support**: Monitor multiple bots
- [ ] **Trade notifications**: Desktop notifications for trades
- [ ] **Advanced charts**: Candlestick charts, indicators
- [ ] **Export functionality**: Export trades to CSV/Excel
- [ ] **Themes**: Light theme option
- [ ] **Custom dashboards**: Drag-and-drop layout
- [ ] **Alerts**: Price alerts, P&L alerts
- [ ] **Mobile companion**: Mobile app integration

---

## 📝 License

Same license as main project. See LICENSE file.

---

## 👏 Credits

Built with:
- **PyQt6**: Modern Python GUI framework
- **PostgreSQL**: Robust database
- **Python**: Elegant programming language

---

**Happy Trading! 📈💰**

Remember: Monitor your bot regularly, especially during the first few weeks!
