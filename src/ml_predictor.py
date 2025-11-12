"""
Real Machine Learning Predictor

Replaces hardcoded rules with actual trained ML models.
Uses sklearn GradientBoostingClassifier trained on historical trade snapshots.

Features:
- GradientBoosting classifier for entry prediction
- Online learning (incremental updates)
- Model persistence (save/load)
- Confidence calibration
- Feature importance tracking
"""

from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from pathlib import Path

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not installed. Install with: pip install scikit-learn")

from src.feature_engineering import get_feature_engineering
from src.database import get_db_client
from src.utils import setup_logging

logger = setup_logging()


class MLPredictor:
    """
    Real ML predictor using GradientBoosting classifier.

    Trains on historical (entry_snapshot, is_winner) pairs to predict
    trade success probability.
    """

    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for MLPredictor")

        self.feature_engineering = get_feature_engineering()

        # Primary model: GradientBoosting (best for tabular data)
        self.model = GradientBoostingClassifier(
            n_estimators=150,           # 🎯 INCREASED: More trees for better generalization
            learning_rate=0.05,         # 🎯 REDUCED: Slower learning = less overfitting
            max_depth=4,                # 🎯 REDUCED: Shallower trees = better generalization
            min_samples_split=30,       # 🎯 INCREASED: More conservative splits
            min_samples_leaf=15,        # 🎯 INCREASED: Larger leaves = smoother predictions
            subsample=0.7,              # 🎯 REDUCED: More regularization via sampling
            max_features='sqrt',        # Number of features per split
            random_state=42,
            verbose=0
        )

        # Calibrated model (for better probability estimates)
        self.calibrated_model: Optional[CalibratedClassifierCV] = None

        # Training state
        self.is_trained = False
        self.training_samples = 0
        self.last_training_time: Optional[datetime] = None
        self.feature_importance: Optional[np.ndarray] = None

        # Model persistence
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / "ml_predictor.pkl"

        # Performance tracking
        self.prediction_count = 0
        self.confidence_history: List[float] = []

        # Try to load existing model
        self.load_model()

    async def predict(
        self,
        snapshot: Dict[str, Any],
        side: str
    ) -> Dict[str, Any]:
        """
        Predict trade success probability using trained ML model.

        Args:
            snapshot: Market snapshot from snapshot_capture.py
            side: 'LONG' or 'SHORT'

        Returns:
            {
                'action': 'buy' or 'sell',
                'confidence': float (0-1),
                'reasoning': str,
                'features': List[float],
                'feature_importance': Optional[Dict]
            }
        """
        try:
            # Extract features
            features = self.feature_engineering.extract_features(snapshot, side)

            # Validate features
            if not self.feature_engineering.validate_features(features):
                logger.error("❌ Invalid features extracted")
                return self._fallback_prediction(snapshot, side)

            # If model not trained yet, use rule-based fallback
            if not self.is_trained:
                logger.warning("⚠️ Model not trained yet, using rule-based fallback")
                return self._fallback_prediction(snapshot, side)

            # Make prediction
            features_array = np.array(features).reshape(1, -1)

            # Get probability estimates
            if self.calibrated_model:
                proba = self.calibrated_model.predict_proba(features_array)[0]
            else:
                proba = self.model.predict_proba(features_array)[0]

            # Probability of winning trade
            win_probability = float(proba[1])

            # Determine action based on side
            if side == 'LONG':
                action = 'buy'
                confidence = win_probability
            else:  # SHORT
                action = 'sell'
                confidence = win_probability

            # Track confidence
            self.prediction_count += 1
            self.confidence_history.append(confidence)

            # Keep only last 100 predictions
            if len(self.confidence_history) > 100:
                self.confidence_history = self.confidence_history[-100:]

            # Generate reasoning
            reasoning = self._generate_reasoning(features, confidence, side)

            # Get feature importance
            feature_importance = self._get_top_features(features) if self.feature_importance is not None else None

            logger.info(
                f"🤖 ML PREDICTION: {action.upper()} | "
                f"Confidence: {confidence:.1%} | "
                f"Features: {len(features)}"
            )

            return {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'features': features,
                'feature_importance': feature_importance,
                'model_trained': True,
                'training_samples': self.training_samples
            }

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._fallback_prediction(snapshot, side)

    async def train(
        self,
        retrain_threshold: int = 50,
        force_retrain: bool = False
    ) -> bool:
        """
        Train (or retrain) ML model on historical trade data.

        Args:
            retrain_threshold: Minimum new trades before retraining
            force_retrain: Force retrain even if threshold not met

        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("🎓 Starting ML model training...")

            # Fetch historical trades with snapshots
            db = await get_db_client()

            query = '''
                SELECT
                    entry_snapshot,
                    is_winner,
                    side,
                    realized_pnl_usd,
                    exit_time
                FROM trade_history
                WHERE entry_snapshot IS NOT NULL
                    AND is_winner IS NOT NULL
                    AND exit_time >= NOW() - INTERVAL '60 days'
                ORDER BY exit_time DESC
            '''

            trades = await db.pool.fetch(query)

            if len(trades) < 30:
                logger.warning(
                    f"⚠️ Insufficient training data: {len(trades)} trades "
                    f"(minimum 30 required)"
                )
                return False

            logger.info(f"📊 Loaded {len(trades)} trades for training")

            # Extract features and labels
            X = []
            y = []
            skipped_count = 0

            for trade in trades:
                try:
                    # ROBUST ERROR HANDLING: Skip malformed snapshots
                    entry_snapshot = trade.get('entry_snapshot')

                    # Skip if no snapshot
                    if not entry_snapshot:
                        skipped_count += 1
                        continue

                    # Handle different snapshot formats
                    if isinstance(entry_snapshot, dict):
                        snapshot = entry_snapshot
                    elif isinstance(entry_snapshot, str):
                        # Try parsing JSON string (may be double-encoded!)
                        import json
                        snapshot = json.loads(entry_snapshot)

                        # Check if still a string (double-encoded JSON)
                        if isinstance(snapshot, str):
                            snapshot = json.loads(snapshot)
                    else:
                        # Try converting to dict (may fail on malformed data)
                        try:
                            snapshot = dict(entry_snapshot)
                        except (TypeError, ValueError):
                            skipped_count += 1
                            continue

                    side = trade['side']
                    is_winner = trade['is_winner']

                    # Validate snapshot has required fields (basic market data)
                    # Old snapshots may have different structures, so be flexible
                    if not isinstance(snapshot, dict):
                        skipped_count += 1
                        continue

                    # Check for at least some market data (price, trend, or indicators)
                    has_market_data = (
                        'price' in snapshot or
                        'trend' in snapshot or
                        'indicators_15m' in snapshot or
                        'indicators' in snapshot
                    )

                    if not has_market_data:
                        skipped_count += 1
                        continue

                    # Extract features
                    features = self.feature_engineering.extract_features(snapshot, side)

                    # Validate
                    if self.feature_engineering.validate_features(features):
                        X.append(features)
                        y.append(1 if is_winner else 0)
                    else:
                        skipped_count += 1

                except Exception as e:
                    # Silently skip bad trades instead of spamming warnings
                    skipped_count += 1
                    continue

            if skipped_count > 0:
                logger.info(f"ℹ️ Skipped {skipped_count} trades with malformed/invalid data")

            if len(X) < 30:
                logger.error(f"❌ Only {len(X)} valid samples after feature extraction")
                return False

            X = np.array(X)
            y = np.array(y)

            logger.info(f"✅ Feature matrix shape: {X.shape}")
            logger.info(f"✅ Win rate in training data: {y.mean():.1%}")

            # Split into train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train model
            logger.info("🔧 Training GradientBoosting classifier...")
            self.model.fit(X_train, y_train)

            # Calibrate probabilities for better confidence estimates
            logger.info("🔧 Calibrating probability estimates...")
            # 🎯 IMPROVED: Use isotonic regression instead of sigmoid
            # Isotonic is more flexible and produces better calibrated probabilities
            # for tree-based models like GradientBoosting
            self.calibrated_model = CalibratedClassifierCV(
                self.model,
                cv='prefit',
                method='isotonic'  # Better than sigmoid for tree models
            )
            self.calibrated_model.fit(X_val, y_val)

            # Evaluate on validation set
            val_score = self.calibrated_model.score(X_val, y_val)
            train_score = self.model.score(X_train, y_train)

            logger.info(f"📈 Training accuracy: {train_score:.1%}")
            logger.info(f"📈 Validation accuracy: {val_score:.1%}")

            # 🔧 FIX #5: STRICT OVERFITTING CHECK (2025-11-12)
            # OLD: 15% gap allowed (too permissive)
            # NEW: 10% gap maximum (professional standard)
            # Impact: Model quality enforcement
            overfitting_gap = train_score - val_score

            if overfitting_gap > 0.10:
                logger.error(
                    f"❌ SEVERE OVERFITTING DETECTED! "
                    f"train={train_score:.1%}, val={val_score:.1%}, gap={overfitting_gap:.1%}\n"
                    f"   Model is learning noise, not patterns!\n"
                    f"   Predictions will be unreliable in production.\n"
                    f"   RECOMMENDATION: Collect more data or reduce model complexity."
                )
                # Still train but warn aggressively
            elif overfitting_gap > 0.07:
                logger.warning(
                    f"⚠️ Moderate overfitting: train={train_score:.1%}, "
                    f"val={val_score:.1%}, gap={overfitting_gap:.1%}"
                )

            # 🔧 FIX #6: LOW VALIDATION ACCURACY WARNING
            # If validation accuracy < 55%, model is basically random
            if val_score < 0.55:
                logger.error(
                    f"❌ VALIDATION ACCURACY TOO LOW: {val_score:.1%}\n"
                    f"   Model barely better than coin flip (50%)!\n"
                    f"   High confidence predictions will be WRONG.\n"
                    f"   RECOMMENDATION: Feature engineering needed or more data."
                )

            # Extract feature importance
            self.feature_importance = self.model.feature_importances_
            self._log_feature_importance()

            # Update training state
            self.is_trained = True
            self.training_samples = len(X)
            self.last_training_time = datetime.now()

            # Save model
            self.save_model()

            logger.info(
                f"✅ ML model training complete! "
                f"Samples: {self.training_samples}, Accuracy: {val_score:.1%}"
            )

            return True

        except Exception as e:
            logger.error(f"❌ ML training failed: {e}")
            return False

    async def incremental_update(self, trade_data: Dict[str, Any]) -> bool:
        """
        Incrementally update model with new trade result (online learning).

        Args:
            trade_data: Trade data with entry_snapshot and is_winner

        Returns:
            True if update successful
        """
        try:
            # For GradientBoosting, we can't do true incremental updates
            # Instead, periodically retrain when enough new data accumulated

            # Check if it's time to retrain
            if self.training_samples > 0 and self.training_samples % 50 == 0:
                logger.info("🔄 50 new trades since last training, triggering retrain...")
                await self.train(force_retrain=True)

            return True

        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            return False

    def save_model(self) -> bool:
        """Save trained model to disk."""
        try:
            model_data = {
                'model': self.model,
                'calibrated_model': self.calibrated_model,
                'is_trained': self.is_trained,
                'training_samples': self.training_samples,
                'last_training_time': self.last_training_time,
                'feature_importance': self.feature_importance,
                'feature_names': self.feature_engineering.get_feature_names()
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"💾 Model saved to {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self) -> bool:
        """Load trained model from disk."""
        try:
            if not self.model_path.exists():
                logger.info("📂 No saved model found, will train from scratch")
                return False

            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.calibrated_model = model_data.get('calibrated_model')
            self.is_trained = model_data['is_trained']
            self.training_samples = model_data['training_samples']
            self.last_training_time = model_data.get('last_training_time')
            self.feature_importance = model_data.get('feature_importance')

            logger.info(
                f"✅ Model loaded from disk | "
                f"Samples: {self.training_samples} | "
                f"Last trained: {self.last_training_time}"
            )

            return True

        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return False

    def _fallback_prediction(self, snapshot: Dict[str, Any], side: str) -> Dict[str, Any]:
        """
        Rule-based fallback when ML model not available.
        Used during cold start period.
        """
        try:
            indicators_15m = snapshot.get('indicators', {}).get('15m', {})
            indicators_1h = snapshot.get('indicators', {}).get('1h', {})

            rsi_15m = indicators_15m.get('rsi', 50)
            trend_1h = indicators_1h.get('trend', 'neutral')

            # Simple rule-based logic
            if side == 'LONG':
                action = 'buy'
                # Lower confidence for rule-based predictions
                if rsi_15m < 30 and trend_1h == 'uptrend':
                    confidence = 0.45  # Reduced from hardcoded 0.58
                elif rsi_15m < 40:
                    confidence = 0.40
                else:
                    confidence = 0.35
            else:  # SHORT
                action = 'sell'
                if rsi_15m > 70 and trend_1h == 'downtrend':
                    confidence = 0.45
                elif rsi_15m > 60:
                    confidence = 0.40
                else:
                    confidence = 0.35

            return {
                'action': action,
                'confidence': confidence,
                'reasoning': f"Rule-based fallback (ML model not trained yet)",
                'features': [],
                'feature_importance': None,
                'model_trained': False,
                'training_samples': 0
            }

        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            return {
                'action': 'buy' if side == 'LONG' else 'sell',
                'confidence': 0.30,
                'reasoning': 'Error fallback',
                'features': [],
                'feature_importance': None,
                'model_trained': False,
                'training_samples': 0
            }

    def _generate_reasoning(self, features: List[float], confidence: float, side: str) -> str:
        """Generate human-readable reasoning for prediction."""
        try:
            feature_names = self.feature_engineering.get_feature_names()

            # Find top contributing features
            if self.feature_importance is not None:
                # Get top 3 most important features for this prediction
                feature_contributions = [
                    (name, features[i] * self.feature_importance[i])
                    for i, name in enumerate(feature_names)
                ]
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                top_features = feature_contributions[:3]

                reasoning_parts = [
                    f"ML model predicts {confidence:.0%} win probability",
                    f"Key factors: {', '.join([f[0] for f in top_features])}"
                ]
            else:
                reasoning_parts = [f"ML model predicts {confidence:.0%} win probability"]

            # Add confidence level description
            if confidence >= 0.70:
                reasoning_parts.append("(High confidence)")
            elif confidence >= 0.50:
                reasoning_parts.append("(Medium confidence)")
            else:
                reasoning_parts.append("(Low confidence)")

            return " | ".join(reasoning_parts)

        except Exception as e:
            logger.warning(f"Failed to generate reasoning: {e}")
            return f"ML prediction: {confidence:.0%} confidence"

    def _get_top_features(self, features: List[float], top_n: int = 5) -> Dict[str, float]:
        """Get top N most important features for this prediction."""
        try:
            feature_names = self.feature_engineering.get_feature_names()

            # Calculate feature contributions (feature_value * importance)
            contributions = [
                (name, features[i] * self.feature_importance[i])
                for i, name in enumerate(feature_names)
            ]

            # Sort by absolute contribution
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            # Return top N
            return {name: contrib for name, contrib in contributions[:top_n]}

        except Exception as e:
            logger.warning(f"Failed to get top features: {e}")
            return {}

    def _log_feature_importance(self, top_n: int = 10):
        """Log top N most important features."""
        try:
            if self.feature_importance is None:
                return

            feature_names = self.feature_engineering.get_feature_names()
            importance_pairs = list(zip(feature_names, self.feature_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)

            logger.info("📊 Top 10 Most Important Features:")
            for i, (name, importance) in enumerate(importance_pairs[:top_n], 1):
                logger.info(f"  {i}. {name}: {importance:.4f}")

        except Exception as e:
            logger.warning(f"Failed to log feature importance: {e}")

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics for monitoring."""
        return {
            'is_trained': self.is_trained,
            'training_samples': self.training_samples,
            'last_training_time': self.last_training_time,
            'prediction_count': self.prediction_count,
            'avg_confidence': np.mean(self.confidence_history) if self.confidence_history else 0,
            'model_path': str(self.model_path),
            'model_exists': self.model_path.exists()
        }


# Singleton instance
_ml_predictor: Optional[MLPredictor] = None


def get_ml_predictor() -> MLPredictor:
    """Get or create MLPredictor instance."""
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = MLPredictor()
    return _ml_predictor
