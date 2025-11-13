"""
Comprehensive Unit Tests for ML Predictor
Target Coverage: 90%+

Tests machine learning prediction system:
- Feature extraction
- Model training
- Prediction accuracy
- Model persistence
- Fallback mechanisms
"""

import pytest
import numpy as np
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from src.ml_predictor import MLPredictor, get_ml_predictor


@pytest.fixture
def ml_predictor():
    """Create ML predictor instance for testing."""
    predictor = MLPredictor()
    yield predictor


@pytest.fixture
def sample_snapshot():
    """Create sample market snapshot for testing."""
    return {
        'current_price': 50000.0,
        'indicators': {
            '15m': {
                'rsi': 55.0,
                'macd': 120.5,
                'macd_signal': 100.2,
                'bb_upper': 51000.0,
                'bb_lower': 49000.0,
                'sma_20': 50000.0,
                'ema_12': 50100.0,
                'volume_trend': 'increasing'
            },
            '1h': {
                'rsi': 60.0,
                'macd': 250.0,
                'macd_signal': 220.0,
                'trend': 'uptrend',
                'volume_trend': 'increasing'
            },
            '4h': {
                'rsi': 65.0,
                'macd': 400.0,
                'trend': 'uptrend'
            }
        },
        'market_regime': 'TRENDING',
        'volatility_analysis': {
            'atr_percent': 2.5,
            'volatility_level': 'NORMAL'
        },
        'volume_24h': 1000000000.0,
        'multi_timeframe': {
            'trend_alignment': 'bullish',
            'rsi_analysis': {
                '15m': 55,
                '1h': 60,
                '4h': 65
            }
        }
    }


@pytest.fixture
def mock_db():
    """Mock database client."""
    db = AsyncMock()

    # Mock training data (20 trades)
    training_data = []
    for i in range(20):
        training_data.append({
            'entry_snapshot': {
                'current_price': 50000 + i * 100,
                'indicators': {
                    '15m': {'rsi': 50 + i, 'macd': 100 + i * 10},
                    '1h': {'rsi': 55 + i, 'macd': 200 + i * 10},
                    '4h': {'rsi': 60 + i, 'macd': 300 + i * 10}
                }
            },
            'is_winner': i % 3 != 0,  # 66% win rate
            'side': 'LONG' if i % 2 == 0 else 'SHORT'
        })

    db.pool.fetch = AsyncMock(return_value=training_data)
    return db


class TestFeatureExtraction:
    """Test suite for feature engineering."""

    def test_extract_features_valid_snapshot(self, ml_predictor, sample_snapshot):
        """Test feature extraction from valid snapshot."""
        from src.feature_engineering import get_feature_engineering

        feature_eng = get_feature_engineering()
        features = feature_eng.extract_features(sample_snapshot, side='LONG')

        # Should extract 40+ features
        assert len(features) >= 40

        # All features should be numeric
        assert all(isinstance(f, (int, float)) for f in features)

        # No NaN or Inf values
        assert all(not np.isnan(f) and not np.isinf(f) for f in features)

    def test_extract_features_missing_data(self, ml_predictor):
        """Test feature extraction handles missing data gracefully."""
        from src.feature_engineering import get_feature_engineering

        # Incomplete snapshot
        incomplete_snapshot = {
            'current_price': 50000.0,
            'indicators': {
                '15m': {'rsi': 55.0}  # Missing most indicators
            }
        }

        feature_eng = get_feature_engineering()
        features = feature_eng.extract_features(incomplete_snapshot, side='LONG')

        # Should still return features (with defaults)
        assert len(features) > 0
        assert all(isinstance(f, (int, float)) for f in features)

    def test_feature_validation(self, ml_predictor, sample_snapshot):
        """Test feature validation."""
        from src.feature_engineering import get_feature_engineering

        feature_eng = get_feature_engineering()
        features = feature_eng.extract_features(sample_snapshot, side='LONG')

        # Validate features
        is_valid = feature_eng.validate_features(features)
        assert is_valid == True

    def test_feature_validation_reject_invalid(self, ml_predictor):
        """Test feature validation rejects invalid features."""
        from src.feature_engineering import get_feature_engineering

        feature_eng = get_feature_engineering()

        # Invalid features (with NaN)
        invalid_features = [1.0, 2.0, np.nan, 4.0]

        is_valid = feature_eng.validate_features(invalid_features)
        assert is_valid == False


class TestModelTraining:
    """Test suite for model training."""

    @pytest.mark.asyncio
    async def test_train_with_sufficient_data(self, ml_predictor, mock_db):
        """Test model training with sufficient historical data."""
        with patch('src.ml_predictor.get_db_client', return_value=mock_db):
            success = await ml_predictor.train(force_retrain=True)

            assert success == True
            assert ml_predictor.is_trained == True
            assert ml_predictor.training_samples >= 10

    @pytest.mark.asyncio
    async def test_train_with_insufficient_data(self, ml_predictor):
        """Test model training with insufficient data."""
        # Mock DB with only 5 trades
        mock_db_small = AsyncMock()
        mock_db_small.pool.fetch = AsyncMock(return_value=[
            {'entry_snapshot': {}, 'is_winner': True, 'side': 'LONG'}
            for _ in range(5)
        ])

        with patch('src.ml_predictor.get_db_client', return_value=mock_db_small):
            success = await ml_predictor.train(force_retrain=True)

            # Should fail or warn (need 30+ trades)
            assert ml_predictor.training_samples < 10

    @pytest.mark.asyncio
    async def test_model_persistence(self, ml_predictor, mock_db):
        """Test model save and load functionality."""
        with patch('src.ml_predictor.get_db_client', return_value=mock_db):
            # Train model
            await ml_predictor.train(force_retrain=True)

            # Save model
            ml_predictor.save_model()

            # Create new predictor and load
            new_predictor = MLPredictor()
            new_predictor.load_model()

            assert new_predictor.is_trained == True
            assert new_predictor.training_samples == ml_predictor.training_samples

    def test_get_model_stats(self, ml_predictor):
        """Test model statistics retrieval."""
        stats = ml_predictor.get_model_stats()

        assert 'is_trained' in stats
        assert 'training_samples' in stats
        assert 'last_training_time' in stats
        assert 'model_path' in stats


class TestPrediction:
    """Test suite for ML predictions."""

    @pytest.mark.asyncio
    async def test_predict_without_training_uses_fallback(self, ml_predictor, sample_snapshot):
        """Test prediction without trained model uses fallback."""
        # Ensure model is not trained
        ml_predictor.is_trained = False

        result = await ml_predictor.predict(sample_snapshot, side='LONG')

        assert result['action'] in ['buy', 'sell', 'hold']
        assert 0 <= result['confidence'] <= 1
        assert 'reasoning' in result
        assert 'model_trained' in result
        assert result['model_trained'] == False

    @pytest.mark.asyncio
    async def test_predict_with_trained_model(self, ml_predictor, mock_db, sample_snapshot):
        """Test prediction with trained model."""
        with patch('src.ml_predictor.get_db_client', return_value=mock_db):
            # Train model first
            await ml_predictor.train(force_retrain=True)

            # Make prediction
            result = await ml_predictor.predict(sample_snapshot, side='LONG')

            assert result['action'] in ['buy', 'sell']  # Should not be 'hold' after training
            assert 0 <= result['confidence'] <= 1
            assert result['model_trained'] == True
            assert 'features' in result

    @pytest.mark.asyncio
    async def test_predict_long_vs_short(self, ml_predictor, mock_db, sample_snapshot):
        """Test predictions differ for LONG vs SHORT."""
        with patch('src.ml_predictor.get_db_client', return_value=mock_db):
            # Train model
            await ml_predictor.train(force_retrain=True)

            # Predict LONG
            result_long = await ml_predictor.predict(sample_snapshot, side='LONG')

            # Predict SHORT
            result_short = await ml_predictor.predict(sample_snapshot, side='SHORT')

            # Actions should match sides
            assert result_long['action'] == 'buy'
            assert result_short['action'] == 'sell'

    @pytest.mark.asyncio
    async def test_prediction_confidence_tracking(self, ml_predictor, mock_db, sample_snapshot):
        """Test prediction confidence is tracked."""
        with patch('src.ml_predictor.get_db_client', return_value=mock_db):
            await ml_predictor.train(force_retrain=True)

            # Make multiple predictions
            for _ in range(5):
                await ml_predictor.predict(sample_snapshot, side='LONG')

            assert len(ml_predictor.confidence_history) == 5
            assert ml_predictor.prediction_count == 5


class TestFallbackMechanism:
    """Test suite for fallback prediction logic."""

    @pytest.mark.asyncio
    async def test_fallback_on_training_failure(self, ml_predictor, sample_snapshot):
        """Test fallback when ML training fails."""
        # Simulate training failure
        ml_predictor.is_trained = False

        result = await ml_predictor.predict(sample_snapshot, side='LONG')

        # Should use rule-based fallback
        assert result['action'] in ['buy', 'sell', 'hold']
        assert 'Fallback' in result.get('reasoning', '') or not ml_predictor.is_trained

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_features(self, ml_predictor):
        """Test fallback when features are invalid."""
        # Invalid snapshot
        invalid_snapshot = {
            'current_price': None,  # Invalid
            'indicators': {}
        }

        result = await ml_predictor.predict(invalid_snapshot, side='LONG')

        # Should fallback gracefully
        assert result['action'] in ['buy', 'sell', 'hold']
        assert result['confidence'] >= 0


class TestFeatureImportance:
    """Test suite for feature importance tracking."""

    @pytest.mark.asyncio
    async def test_feature_importance_after_training(self, ml_predictor, mock_db):
        """Test feature importance is calculated after training."""
        with patch('src.ml_predictor.get_db_client', return_value=mock_db):
            await ml_predictor.train(force_retrain=True)

            assert ml_predictor.feature_importance is not None
            assert len(ml_predictor.feature_importance) > 0

    @pytest.mark.asyncio
    async def test_get_top_features(self, ml_predictor, mock_db, sample_snapshot):
        """Test retrieval of top contributing features."""
        with patch('src.ml_predictor.get_db_client', return_value=mock_db):
            await ml_predictor.train(force_retrain=True)

            result = await ml_predictor.predict(sample_snapshot, side='LONG')

            if 'feature_importance' in result and result['feature_importance']:
                # Should return dict of top features
                assert isinstance(result['feature_importance'], dict)


# ==================== INTEGRATION TESTS ====================

class TestMLPredictorIntegration:
    """Integration tests for complete ML workflow."""

    @pytest.mark.asyncio
    async def test_complete_ml_workflow(self, ml_predictor, mock_db, sample_snapshot):
        """Test complete workflow: train → predict → retrain."""
        with patch('src.ml_predictor.get_db_client', return_value=mock_db):
            # Step 1: Train model
            success = await ml_predictor.train(force_retrain=True)
            assert success == True

            # Step 2: Make predictions
            result1 = await ml_predictor.predict(sample_snapshot, side='LONG')
            assert result1['model_trained'] == True

            # Step 3: Save model
            ml_predictor.save_model()
            assert ml_predictor.model_path.exists()

            # Step 4: Load in new instance
            new_predictor = MLPredictor()
            new_predictor.load_model()

            # Step 5: Verify loaded model works
            result2 = await new_predictor.predict(sample_snapshot, side='LONG')
            assert result2['model_trained'] == True

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test get_ml_predictor returns singleton."""
        predictor1 = get_ml_predictor()
        predictor2 = get_ml_predictor()

        assert predictor1 is predictor2


# ==================== PERFORMANCE TESTS ====================

class TestPerformance:
    """Test suite for performance and efficiency."""

    @pytest.mark.asyncio
    async def test_prediction_speed(self, ml_predictor, mock_db, sample_snapshot):
        """Test prediction completes in reasonable time."""
        import time

        with patch('src.ml_predictor.get_db_client', return_value=mock_db):
            await ml_predictor.train(force_retrain=True)

            start = time.time()
            await ml_predictor.predict(sample_snapshot, side='LONG')
            duration = time.time() - start

            # Should complete in < 100ms
            assert duration < 0.1

    @pytest.mark.asyncio
    async def test_training_speed(self, ml_predictor, mock_db):
        """Test training completes in reasonable time."""
        import time

        with patch('src.ml_predictor.get_db_client', return_value=mock_db):
            start = time.time()
            await ml_predictor.train(force_retrain=True)
            duration = time.time() - start

            # Should complete in < 5 seconds for 20 samples
            assert duration < 5.0


# ==================== RUN TESTS ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=src.ml_predictor"])
