-- ============================================================================
-- TIER 1 & 2 Database Migration Script
-- Run this on Railway database to add new columns
-- Safe to run multiple times (uses IF NOT EXISTS)
-- ============================================================================

-- TIER 1: Trailing Stop-Loss columns
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='max_profit_percent') THEN
        ALTER TABLE active_position ADD COLUMN max_profit_percent DECIMAL(10, 4) DEFAULT 0.0;
        RAISE NOTICE '✅ Added max_profit_percent to active_position';
    ELSE
        RAISE NOTICE '⏭️  max_profit_percent already exists in active_position';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='trailing_stop_activated') THEN
        ALTER TABLE active_position ADD COLUMN trailing_stop_activated BOOLEAN DEFAULT FALSE;
        RAISE NOTICE '✅ Added trailing_stop_activated to active_position';
    ELSE
        RAISE NOTICE '⏭️  trailing_stop_activated already exists in active_position';
    END IF;
END $$;

-- TIER 1: Partial Exit columns
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='partial_exits_completed') THEN
        ALTER TABLE active_position ADD COLUMN partial_exits_completed TEXT[] DEFAULT '{}';
        RAISE NOTICE '✅ Added partial_exits_completed to active_position';
    ELSE
        RAISE NOTICE '⏭️  partial_exits_completed already exists in active_position';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='original_quantity') THEN
        ALTER TABLE active_position ADD COLUMN original_quantity DECIMAL(20, 8);
        RAISE NOTICE '✅ Added original_quantity to active_position';
    ELSE
        RAISE NOTICE '⏭️  original_quantity already exists in active_position';
    END IF;
END $$;

-- TIER 2: Market Regime columns
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='entry_market_regime') THEN
        ALTER TABLE active_position ADD COLUMN entry_market_regime VARCHAR(50);
        RAISE NOTICE '✅ Added entry_market_regime to active_position';
    ELSE
        RAISE NOTICE '⏭️  entry_market_regime already exists in active_position';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='active_position' AND column_name='exit_market_regime') THEN
        ALTER TABLE active_position ADD COLUMN exit_market_regime VARCHAR(50);
        RAISE NOTICE '✅ Added exit_market_regime to active_position';
    ELSE
        RAISE NOTICE '⏭️  exit_market_regime already exists in active_position';
    END IF;
END $$;

-- =======================================================================
-- TRADE HISTORY TABLE MIGRATIONS
-- =======================================================================

-- TIER 1: Trailing Stop-Loss tracking in trade_history
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='max_profit_percent_achieved') THEN
        ALTER TABLE trade_history ADD COLUMN max_profit_percent_achieved DECIMAL(10, 4);
        RAISE NOTICE '✅ Added max_profit_percent_achieved to trade_history';
    ELSE
        RAISE NOTICE '⏭️  max_profit_percent_achieved already exists in trade_history';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='trailing_stop_triggered') THEN
        ALTER TABLE trade_history ADD COLUMN trailing_stop_triggered BOOLEAN DEFAULT FALSE;
        RAISE NOTICE '✅ Added trailing_stop_triggered to trade_history';
    ELSE
        RAISE NOTICE '⏭️  trailing_stop_triggered already exists in trade_history';
    END IF;
END $$;

-- TIER 1: Partial Exit tracking in trade_history
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='had_partial_exits') THEN
        ALTER TABLE trade_history ADD COLUMN had_partial_exits BOOLEAN DEFAULT FALSE;
        RAISE NOTICE '✅ Added had_partial_exits to trade_history';
    ELSE
        RAISE NOTICE '⏭️  had_partial_exits already exists in trade_history';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='partial_exit_details') THEN
        ALTER TABLE trade_history ADD COLUMN partial_exit_details JSONB;
        RAISE NOTICE '✅ Added partial_exit_details to trade_history';
    ELSE
        RAISE NOTICE '⏭️  partial_exit_details already exists in trade_history';
    END IF;
END $$;

-- TIER 2: Market Regime tracking in trade_history
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='entry_market_regime') THEN
        ALTER TABLE trade_history ADD COLUMN entry_market_regime VARCHAR(50);
        RAISE NOTICE '✅ Added entry_market_regime to trade_history';
    ELSE
        RAISE NOTICE '⏭️  entry_market_regime already exists in trade_history';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='trade_history' AND column_name='exit_market_regime') THEN
        ALTER TABLE trade_history ADD COLUMN exit_market_regime VARCHAR(50);
        RAISE NOTICE '✅ Added exit_market_regime to trade_history';
    ELSE
        RAISE NOTICE '⏭️  exit_market_regime already exists in trade_history';
    END IF;
END $$;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_trade_history_regime ON trade_history(entry_market_regime);
CREATE INDEX IF NOT EXISTS idx_trade_history_partial_exits ON trade_history(had_partial_exits);
CREATE INDEX IF NOT EXISTS idx_trade_history_trailing_stop ON trade_history(trailing_stop_triggered);

-- Done!
SELECT 'Migration complete! All TIER 1 & 2 columns added successfully.' AS status;
