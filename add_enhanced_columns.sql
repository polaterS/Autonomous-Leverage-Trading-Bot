-- ================================================================
-- Enhanced Trading System - Database Migration
-- Adds columns to track professional trading metrics
-- ================================================================

-- Add confluence_score column (0-100 quality score)
ALTER TABLE trades
ADD COLUMN IF NOT EXISTS confluence_score FLOAT DEFAULT NULL;

-- Add quality column (EXCELLENT, STRONG, GOOD, etc.)
ALTER TABLE trades
ADD COLUMN IF NOT EXISTS quality VARCHAR(20) DEFAULT NULL;

-- Add risk_percentage column (% of account risked)
ALTER TABLE trades
ADD COLUMN IF NOT EXISTS risk_percentage FLOAT DEFAULT NULL;

-- Verify columns were added
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'trades'
AND column_name IN ('confluence_score', 'quality', 'risk_percentage')
ORDER BY column_name;
