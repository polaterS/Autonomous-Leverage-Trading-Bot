-- Database Schema for Autonomous Leverage Trading Bot

-- Trading Configuration
CREATE TABLE IF NOT EXISTS trading_config (
    id SERIAL PRIMARY KEY,
    initial_capital DECIMAL(20, 8) NOT NULL,
    current_capital DECIMAL(20, 8) NOT NULL,
    position_size_percent DECIMAL(5, 2) DEFAULT 0.80,
    min_stop_loss_percent DECIMAL(5, 2) DEFAULT 0.05,
    max_stop_loss_percent DECIMAL(5, 2) DEFAULT 0.10,
    min_profit_usd DECIMAL(10, 2) DEFAULT 2.50,
    max_leverage INTEGER DEFAULT 5,
    min_ai_confidence DECIMAL(3, 2) DEFAULT 0.75,
    daily_loss_limit_percent DECIMAL(5, 2) DEFAULT 0.10,
    max_consecutive_losses INTEGER DEFAULT 3,
    is_trading_enabled BOOLEAN DEFAULT true,
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Active Positions (should only ever have 0 or 1 row)
CREATE TABLE IF NOT EXISTS active_position (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(30) NOT NULL,
    side VARCHAR(10) NOT NULL,
    leverage INTEGER NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    quantity DECIMAL(20, 8) NOT NULL,
    position_value_usd DECIMAL(20, 8) NOT NULL,
    stop_loss_price DECIMAL(20, 8) NOT NULL,
    stop_loss_percent DECIMAL(5, 2) NOT NULL,
    min_profit_target_usd DECIMAL(10, 2) NOT NULL,
    min_profit_price DECIMAL(20, 8) NOT NULL,
    liquidation_price DECIMAL(20, 8) NOT NULL,
    unrealized_pnl_usd DECIMAL(20, 8) DEFAULT 0,
    exchange_order_id VARCHAR(100),
    stop_loss_order_id VARCHAR(100),
    ai_model_consensus VARCHAR(100),
    ai_confidence DECIMAL(3, 2),
    entry_time TIMESTAMP DEFAULT NOW(),
    last_check_time TIMESTAMP DEFAULT NOW(),
    partial_close_executed BOOLEAN DEFAULT FALSE
);

-- Trade History
CREATE TABLE IF NOT EXISTS trade_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(30) NOT NULL,
    side VARCHAR(10) NOT NULL,
    leverage INTEGER NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    exit_price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    position_value_usd DECIMAL(20, 8) NOT NULL,
    realized_pnl_usd DECIMAL(20, 8) NOT NULL,
    pnl_percent DECIMAL(10, 4) NOT NULL,
    stop_loss_percent DECIMAL(5, 2),
    close_reason VARCHAR(100),
    trade_duration_seconds INTEGER,
    ai_model_consensus VARCHAR(100),
    ai_confidence DECIMAL(3, 2),
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP DEFAULT NOW(),
    is_winner BOOLEAN
);

-- Performance indexes for trade_history
CREATE INDEX IF NOT EXISTS idx_trade_history_time ON trade_history(exit_time DESC);
CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_history_winner ON trade_history(is_winner, exit_time DESC);
CREATE INDEX IF NOT EXISTS idx_trade_history_daily_pnl ON trade_history(DATE(exit_time), realized_pnl_usd);

-- AI Analysis Cache
CREATE TABLE IF NOT EXISTS ai_analysis_cache (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(30) NOT NULL,
    ai_model VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    analysis_json JSONB NOT NULL,
    confidence DECIMAL(3, 2),
    action VARCHAR(10),
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ai_cache_lookup ON ai_analysis_cache(symbol, ai_model, timeframe, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_cache_expiry ON ai_analysis_cache(expires_at);

-- Daily Performance Tracking
CREATE TABLE IF NOT EXISTS daily_performance (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    starting_capital DECIMAL(20, 8),
    ending_capital DECIMAL(20, 8),
    daily_pnl DECIMAL(20, 8),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2),
    largest_win DECIMAL(20, 8),
    largest_loss DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Performance index for daily_performance
CREATE INDEX IF NOT EXISTS idx_daily_perf_date ON daily_performance(date DESC);

-- System Logs & Alerts
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    log_level VARCHAR(20),
    component VARCHAR(50),
    message TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_logs_time ON system_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(log_level);

-- Circuit Breaker Events
CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50),
    trigger_value DECIMAL(20, 8),
    threshold_value DECIMAL(20, 8),
    action_taken VARCHAR(100),
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Performance index for circuit breaker lookups
CREATE INDEX IF NOT EXISTS idx_circuit_breaker_time ON circuit_breaker_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_circuit_breaker_type ON circuit_breaker_events(event_type, resolved_at);

-- Initial configuration insert (UPSERT to prevent duplicates)
INSERT INTO trading_config (
    id,
    initial_capital,
    current_capital,
    position_size_percent,
    min_stop_loss_percent,
    max_stop_loss_percent,
    min_profit_usd,
    max_leverage,
    min_ai_confidence,
    daily_loss_limit_percent,
    max_consecutive_losses,
    is_trading_enabled
) VALUES (
    1,  -- Fixed ID to prevent duplicates
    100.00,
    100.00,
    0.80,
    0.05,
    0.10,
    2.50,
    5,
    0.75,
    0.10,
    3,
    true
) ON CONFLICT (id) DO UPDATE SET
    -- Only update params if capital hasn't changed (fresh restart)
    position_size_percent = EXCLUDED.position_size_percent,
    min_stop_loss_percent = EXCLUDED.min_stop_loss_percent,
    max_stop_loss_percent = EXCLUDED.max_stop_loss_percent,
    min_profit_usd = EXCLUDED.min_profit_usd,
    max_leverage = EXCLUDED.max_leverage,
    min_ai_confidence = EXCLUDED.min_ai_confidence,
    daily_loss_limit_percent = EXCLUDED.daily_loss_limit_percent,
    max_consecutive_losses = EXCLUDED.max_consecutive_losses,
    is_trading_enabled = EXCLUDED.is_trading_enabled
    -- NOTE: Do NOT update current_capital to preserve trading progress!
