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
