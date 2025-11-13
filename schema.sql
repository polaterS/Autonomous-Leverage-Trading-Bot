-- Database Schema for Autonomous Leverage Trading Bot
-- EMERGENCY FIX v3.2: Force rebuild to apply max_concurrent_positions=3

-- Trading Configuration
CREATE TABLE IF NOT EXISTS trading_config (
    id SERIAL PRIMARY KEY,
    initial_capital DECIMAL(20, 8) NOT NULL,
    current_capital DECIMAL(20, 8) NOT NULL,
    starting_capital DECIMAL(20, 8),  -- Capital at session/day start (for drawdown calc)
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

-- Add starting_capital column if it doesn't exist (migration)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'trading_config' AND column_name = 'starting_capital'
    ) THEN
        ALTER TABLE trading_config ADD COLUMN starting_capital DECIMAL(20, 8);
        -- Initialize starting_capital to current_capital for existing rows
        UPDATE trading_config SET starting_capital = current_capital WHERE starting_capital IS NULL;
    END IF;
END $$;

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
    ai_reasoning TEXT,
    entry_time TIMESTAMP DEFAULT NOW(),
    last_check_time TIMESTAMP DEFAULT NOW(),
    partial_close_executed BOOLEAN DEFAULT FALSE,

    -- 📸 ML LEARNING: Entry snapshot (captured when position opens)
    entry_snapshot JSONB,
    entry_slippage_percent DECIMAL(10, 6),
    entry_fill_time_ms INTEGER,

    -- 🎯 TIER 1: Trailing Stop-Loss
    max_profit_percent DECIMAL(10, 4) DEFAULT 0.0,
    trailing_stop_activated BOOLEAN DEFAULT FALSE,

    -- 🎯 TIER 1: Partial Exit System
    partial_exits_completed TEXT[] DEFAULT '{}',
    original_quantity DECIMAL(20, 8),

    -- 🎯 TIER 2: Market Regime Tracking
    entry_market_regime VARCHAR(50)
);

-- Trade History with ML Learning Snapshots
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
    ai_reasoning TEXT,
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP DEFAULT NOW(),
    is_winner BOOLEAN,

    -- 🎯 ML LEARNING SNAPSHOTS (JSONB for flexible schema)
    -- Captured at trade ENTRY for pattern learning
    entry_snapshot JSONB,
    -- Captured at trade EXIT for outcome correlation
    exit_snapshot JSONB,

    -- 📊 EXECUTION QUALITY METRICS (for slippage/timing learning)
    entry_slippage_percent DECIMAL(10, 6),
    exit_slippage_percent DECIMAL(10, 6),
    entry_fill_time_ms INTEGER,
    exit_fill_time_ms INTEGER,

    -- 🎯 TIER 1: Trailing Stop-Loss Metrics
    max_profit_percent_achieved DECIMAL(10, 4),
    trailing_stop_triggered BOOLEAN DEFAULT FALSE,

    -- 🎯 TIER 1: Partial Exit Tracking
    had_partial_exits BOOLEAN DEFAULT FALSE,
    partial_exit_details JSONB,

    -- 🎯 TIER 2: Market Regime at Entry/Exit
    entry_market_regime VARCHAR(50),
    exit_market_regime VARCHAR(50)
);

-- Performance indexes for trade_history
CREATE INDEX IF NOT EXISTS idx_trade_history_time ON trade_history(exit_time DESC);
CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_history_winner ON trade_history(is_winner, exit_time DESC);
CREATE INDEX IF NOT EXISTS idx_trade_history_daily_pnl ON trade_history(DATE(exit_time), realized_pnl_usd);

-- 🎯 ML SNAPSHOT INDEXES (GIN for JSONB queries)
CREATE INDEX IF NOT EXISTS idx_entry_snapshot_indicators ON trade_history USING GIN (entry_snapshot);
CREATE INDEX IF NOT EXISTS idx_exit_snapshot_indicators ON trade_history USING GIN (exit_snapshot);

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
-- Migration: Add ML snapshot columns to existing tables
-- Run this ONCE on Railway database to add new columns

-- ============================================================
-- ADD COLUMNS TO active_position TABLE
-- ============================================================

-- Add ai_reasoning column if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='ai_reasoning') THEN
        ALTER TABLE active_position ADD COLUMN ai_reasoning TEXT;
        RAISE NOTICE 'Added ai_reasoning to active_position';
    END IF;
END $$;

-- Add entry_snapshot column if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='entry_snapshot') THEN
        ALTER TABLE active_position ADD COLUMN entry_snapshot JSONB;
        RAISE NOTICE 'Added entry_snapshot to active_position';
    END IF;
END $$;

-- Add entry_slippage_percent column if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='entry_slippage_percent') THEN
        ALTER TABLE active_position ADD COLUMN entry_slippage_percent DECIMAL(10, 6);
        RAISE NOTICE 'Added entry_slippage_percent to active_position';
    END IF;
END $$;

-- Add entry_fill_time_ms column if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='entry_fill_time_ms') THEN
        ALTER TABLE active_position ADD COLUMN entry_fill_time_ms INTEGER;
        RAISE NOTICE 'Added entry_fill_time_ms to active_position';
    END IF;
END $$;

-- ============================================================
-- ADD COLUMNS TO trade_history TABLE
-- ============================================================

-- Add ai_reasoning column if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='ai_reasoning') THEN
        ALTER TABLE trade_history ADD COLUMN ai_reasoning TEXT;
        RAISE NOTICE 'Added ai_reasoning to trade_history';
    END IF;
END $$;

-- Add entry_snapshot column if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='entry_snapshot') THEN
        ALTER TABLE trade_history ADD COLUMN entry_snapshot JSONB;
        RAISE NOTICE 'Added entry_snapshot to trade_history';
    END IF;
END $$;

-- Add exit_snapshot column if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='exit_snapshot') THEN
        ALTER TABLE trade_history ADD COLUMN exit_snapshot JSONB;
        RAISE NOTICE 'Added exit_snapshot to trade_history';
    END IF;
END $$;

-- Add entry_slippage_percent column if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='entry_slippage_percent') THEN
        ALTER TABLE trade_history ADD COLUMN entry_slippage_percent DECIMAL(10, 6);
        RAISE NOTICE 'Added entry_slippage_percent to trade_history';
    END IF;
END $$;

-- Add exit_slippage_percent column if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='exit_slippage_percent') THEN
        ALTER TABLE trade_history ADD COLUMN exit_slippage_percent DECIMAL(10, 6);
        RAISE NOTICE 'Added exit_slippage_percent to trade_history';
    END IF;
END $$;

-- Add entry_fill_time_ms column if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='entry_fill_time_ms') THEN
        ALTER TABLE trade_history ADD COLUMN entry_fill_time_ms INTEGER;
        RAISE NOTICE 'Added entry_fill_time_ms to trade_history';
    END IF;
END $$;

-- Add exit_fill_time_ms column if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='exit_fill_time_ms') THEN
        ALTER TABLE trade_history ADD COLUMN exit_fill_time_ms INTEGER;
        RAISE NOTICE 'Added exit_fill_time_ms to trade_history';
    END IF;
END $$;

-- ============================================================
-- ADD INDEXES FOR JSONB COLUMNS
-- ============================================================

-- GIN index for active_position.entry_snapshot
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_active_entry_snapshot') THEN
        CREATE INDEX idx_active_entry_snapshot ON active_position USING GIN (entry_snapshot);
        RAISE NOTICE 'Created GIN index on active_position.entry_snapshot';
    END IF;
END $$;

-- GIN index for trade_history.entry_snapshot
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_entry_snapshot_indicators') THEN
        CREATE INDEX idx_entry_snapshot_indicators ON trade_history USING GIN (entry_snapshot);
        RAISE NOTICE 'Created GIN index on trade_history.entry_snapshot';
    END IF;
END $$;

-- GIN index for trade_history.exit_snapshot
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_exit_snapshot_indicators') THEN
        CREATE INDEX idx_exit_snapshot_indicators ON trade_history USING GIN (exit_snapshot);
        RAISE NOTICE 'Created GIN index on trade_history.exit_snapshot';
    END IF;
END $$;

-- ============================================================
-- MIGRATION COMPLETE
-- ============================================================

DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE '✅ ML SNAPSHOT MIGRATION COMPLETE!';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'Added columns:';
    RAISE NOTICE '  - active_position: ai_reasoning, entry_snapshot, entry_slippage_percent, entry_fill_time_ms';
    RAISE NOTICE '  - trade_history: ai_reasoning, entry_snapshot, exit_snapshot, *_slippage_percent, *_fill_time_ms';
    RAISE NOTICE 'Added indexes:';
    RAISE NOTICE '  - GIN indexes on JSONB columns for fast queries';
    RAISE NOTICE '============================================================';
END $$;

-- ============================================================
-- 🎯 TIER 1 & TIER 2 MIGRATIONS
-- ============================================================

-- Add max_profit_percent to active_position (Trailing Stop-Loss)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='max_profit_percent') THEN
        ALTER TABLE active_position ADD COLUMN max_profit_percent DECIMAL(10, 4) DEFAULT 0.0;
        RAISE NOTICE '✅ Added max_profit_percent to active_position';
    END IF;
END $$;

-- Add trailing_stop_activated to active_position
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='trailing_stop_activated') THEN
        ALTER TABLE active_position ADD COLUMN trailing_stop_activated BOOLEAN DEFAULT FALSE;
        RAISE NOTICE '✅ Added trailing_stop_activated to active_position';
    END IF;
END $$;

-- Add partial_exits_completed to active_position (Partial Exit System)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='partial_exits_completed') THEN
        ALTER TABLE active_position ADD COLUMN partial_exits_completed TEXT[] DEFAULT '{}';
        RAISE NOTICE '✅ Added partial_exits_completed to active_position';
    END IF;
END $$;

-- Add original_quantity to active_position
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='original_quantity') THEN
        ALTER TABLE active_position ADD COLUMN original_quantity DECIMAL(20, 8);
        RAISE NOTICE '✅ Added original_quantity to active_position';
    END IF;
END $$;

-- Add entry_market_regime to active_position (Market Regime Detection)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='entry_market_regime') THEN
        ALTER TABLE active_position ADD COLUMN entry_market_regime VARCHAR(50);
        RAISE NOTICE '✅ Added entry_market_regime to active_position';
    END IF;
END $$;

-- Add max_profit_percent_achieved to trade_history
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='max_profit_percent_achieved') THEN
        ALTER TABLE trade_history ADD COLUMN max_profit_percent_achieved DECIMAL(10, 4);
        RAISE NOTICE '✅ Added max_profit_percent_achieved to trade_history';
    END IF;
END $$;

-- Add trailing_stop_triggered to trade_history
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='trailing_stop_triggered') THEN
        ALTER TABLE trade_history ADD COLUMN trailing_stop_triggered BOOLEAN DEFAULT FALSE;
        RAISE NOTICE '✅ Added trailing_stop_triggered to trade_history';
    END IF;
END $$;

-- Add had_partial_exits to trade_history
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='had_partial_exits') THEN
        ALTER TABLE trade_history ADD COLUMN had_partial_exits BOOLEAN DEFAULT FALSE;
        RAISE NOTICE '✅ Added had_partial_exits to trade_history';
    END IF;
END $$;

-- Add partial_exit_details to trade_history
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='partial_exit_details') THEN
        ALTER TABLE trade_history ADD COLUMN partial_exit_details JSONB;
        RAISE NOTICE '✅ Added partial_exit_details to trade_history';
    END IF;
END $$;

-- Add entry_market_regime to trade_history
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='entry_market_regime') THEN
        ALTER TABLE trade_history ADD COLUMN entry_market_regime VARCHAR(50);
        RAISE NOTICE '✅ Added entry_market_regime to trade_history';
    END IF;
END $$;

-- Add exit_market_regime to trade_history
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='exit_market_regime') THEN
        ALTER TABLE trade_history ADD COLUMN exit_market_regime VARCHAR(50);
        RAISE NOTICE '✅ Added exit_market_regime to trade_history';
    END IF;
END $$;

-- Add index for trailing stop analysis
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_trailing_stop_triggered') THEN
        CREATE INDEX idx_trailing_stop_triggered ON trade_history(trailing_stop_triggered)
        WHERE trailing_stop_triggered = TRUE;
        RAISE NOTICE '✅ Created index on trailing_stop_triggered';
    END IF;
END $$;

-- Add index for partial exits analysis
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_partial_exits') THEN
        CREATE INDEX idx_partial_exits ON trade_history(had_partial_exits)
        WHERE had_partial_exits = TRUE;
        RAISE NOTICE '✅ Created index on had_partial_exits';
    END IF;
END $$;

-- Add index for market regime analysis
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_market_regime') THEN
        CREATE INDEX idx_market_regime ON trade_history(entry_market_regime, exit_market_regime);
        RAISE NOTICE '✅ Created index on market regimes';
    END IF;
END $$;

DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE '🎯 TIER 1 & TIER 2 MIGRATION COMPLETE!';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'TIER 1 Features:';
    RAISE NOTICE '  ✅ Trailing Stop-Loss columns added';
    RAISE NOTICE '  ✅ Partial Exit System columns added';
    RAISE NOTICE '  ✅ Correlation tracking ready (uses existing ml_learner)';
    RAISE NOTICE '';
    RAISE NOTICE 'TIER 2 Features:';
    RAISE NOTICE '  ✅ Market Regime Detection columns added';
    RAISE NOTICE '  ✅ ML Continuous Learning (uses existing columns)';
    RAISE NOTICE '';
    RAISE NOTICE '📊 Performance indexes created';
    RAISE NOTICE '============================================================';
END $$;
