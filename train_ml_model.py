"""
ML Model Training Script

Trains the GradientBoosting classifier on historical trade data.
Run this script:
1. On first deployment (cold start)
2. After major changes to feature engineering
3. Manually when you want to retrain

Usage:
    python train_ml_model.py
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ml_predictor import get_ml_predictor
from src.utils import setup_logging

logger = setup_logging()


async def main():
    """Train ML model on historical data."""
    logger.info("🚀 Starting ML Model Training...")
    logger.info("=" * 60)

    try:
        # Get ML predictor
        ml_predictor = get_ml_predictor()

        # Check current model stats
        stats = ml_predictor.get_model_stats()
        logger.info(f"📊 Current Model Stats:")
        logger.info(f"   - Trained: {stats['is_trained']}")
        logger.info(f"   - Training Samples: {stats['training_samples']}")
        logger.info(f"   - Last Training: {stats['last_training_time']}")
        logger.info(f"   - Predictions Made: {stats['prediction_count']}")
        logger.info(f"   - Avg Confidence: {stats['avg_confidence']:.1%}")

        # Train model
        logger.info("=" * 60)
        logger.info("🎓 Training ML model...")

        success = await ml_predictor.train(force_retrain=True)

        if success:
            # Get updated stats
            new_stats = ml_predictor.get_model_stats()

            logger.info("=" * 60)
            logger.info("✅ ML Model Training SUCCESSFUL!")
            logger.info(f"📈 New Model Stats:")
            logger.info(f"   - Training Samples: {new_stats['training_samples']}")
            logger.info(f"   - Model Saved: {new_stats['model_path']}")
            logger.info("=" * 60)

            logger.info("🎉 Ready for production!")
            logger.info("💡 The bot will now use trained ML model for predictions")
            logger.info("📊 Confidence should be 40-70% (vs 0-25% with rules)")

        else:
            logger.error("=" * 60)
            logger.error("❌ ML Model Training FAILED")
            logger.error("=" * 60)
            logger.error("📋 Possible reasons:")
            logger.error("   1. Not enough historical trades (minimum 30 required)")
            logger.error("   2. Missing entry_snapshot data in trade_history")
            logger.error("   3. Database connection issues")
            logger.error("")
            logger.error("💡 Solutions:")
            logger.error("   1. Let bot trade for a few days to collect data")
            logger.error("   2. Bot will use rule-based fallback until trained")
            logger.error("   3. Model will auto-train after 50 trades")

            return 1

        return 0

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"💥 Training script crashed: {e}")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
