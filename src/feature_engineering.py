"""
Feature Engineering Pipeline for ML Trading System

Extracts 30+ numerical features from market snapshots for ML prediction.
Transforms raw market data into ML-ready feature vectors.

Features extracted:
- Price action (12 features)
- Momentum indicators (8 features)
- Volume analysis (6 features)
- Multi-timeframe alignment (4 features)
- Market structure (6 features)
- Sentiment & external factors (4 features)
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
import numpy as np
from src.utils import setup_logging

logger = setup_logging()


class FeatureEngineering:
    """
    Feature extraction pipeline for ML models.

    Converts market snapshots into numerical feature vectors suitable for
    GradientBoosting, RandomForest, and other sklearn models.
    """

    def __init__(self):
        self.feature_names = self._get_feature_names()
        self.feature_count = len(self.feature_names)
        logger.info(f"🔧 FeatureEngineering initialized with {self.feature_count} features")

    def extract_features(self, snapshot: Dict[str, Any], side: str = 'LONG') -> List[float]:
        """
        Extract numerical features from market snapshot.

        Args:
            snapshot: Market snapshot dictionary (from snapshot_capture.py)
            side: 'LONG' or 'SHORT' (for direction-specific features)

        Returns:
            List of 40+ numerical features ready for ML model
        """
        try:
            features = []

            # === PRICE ACTION FEATURES (12) ===
            features.extend(self._extract_price_features(snapshot, side))

            # === MOMENTUM INDICATORS (8) ===
            features.extend(self._extract_momentum_features(snapshot, side))

            # === VOLUME ANALYSIS (6) ===
            features.extend(self._extract_volume_features(snapshot))

            # === MULTI-TIMEFRAME ALIGNMENT (4) ===
            features.extend(self._extract_timeframe_features(snapshot, side))

            # === MARKET STRUCTURE (6) ===
            features.extend(self._extract_structure_features(snapshot, side))

            # === SENTIMENT & EXTERNAL (4) ===
            features.extend(self._extract_sentiment_features(snapshot))

            # Validate feature count
            if len(features) != self.feature_count:
                logger.warning(
                    f"⚠️ Feature count mismatch: expected {self.feature_count}, "
                    f"got {len(features)}"
                )

            # Replace NaN/Inf with 0
            features = [0.0 if (np.isnan(f) or np.isinf(f)) else f for f in features]

            return features

        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            # Return zero vector as fallback
            return [0.0] * self.feature_count

    def _extract_price_features(self, snapshot: Dict[str, Any], side: str) -> List[float]:
        """Extract 12 price action features."""
        try:
            price = float(snapshot.get('current_price', 0))
            indicators_15m = snapshot.get('indicators', {}).get('15m', {})
            indicators_1h = snapshot.get('indicators', {}).get('1h', {})

            # EMAs
            ema12_15m = float(indicators_15m.get('ema_12', price))
            ema50_15m = float(indicators_15m.get('ema_50', price))
            ema12_1h = float(indicators_1h.get('ema_12', price))
            ema50_1h = float(indicators_1h.get('ema_50', price))

            features = [
                # Distance from EMAs (normalized by price)
                (price - ema12_15m) / price if price > 0 else 0,  # F1
                (price - ema50_15m) / price if price > 0 else 0,  # F2
                (price - ema12_1h) / price if price > 0 else 0,   # F3
                (price - ema50_1h) / price if price > 0 else 0,   # F4

                # EMA alignment strength
                (ema12_15m - ema50_15m) / price if price > 0 else 0,  # F5
                (ema12_1h - ema50_1h) / price if price > 0 else 0,    # F6

                # Price momentum (rate of change)
                float(snapshot.get('price_momentum_15m', 0)),  # F7
                float(snapshot.get('price_momentum_1h', 0)),   # F8

                # ATR (volatility)
                float(indicators_15m.get('atr_percent', 2.0)),  # F9
                float(indicators_1h.get('atr_percent', 2.0)),   # F10

                # Bollinger Band position (0-1 scale)
                self._calculate_bb_position(price, indicators_15m),  # F11
                self._calculate_bb_position(price, indicators_1h),   # F12
            ]

            return features

        except Exception as e:
            logger.warning(f"Price feature extraction failed: {e}")
            return [0.0] * 12

    def _extract_momentum_features(self, snapshot: Dict[str, Any], side: str) -> List[float]:
        """Extract 8 momentum indicator features."""
        try:
            indicators_15m = snapshot.get('indicators', {}).get('15m', {})
            indicators_1h = snapshot.get('indicators', {}).get('1h', {})
            indicators_4h = snapshot.get('indicators', {}).get('4h', {})

            # RSI
            rsi_15m = float(indicators_15m.get('rsi', 50))
            rsi_1h = float(indicators_1h.get('rsi', 50))
            rsi_4h = float(indicators_4h.get('rsi', 50))

            # MACD
            macd_15m = float(indicators_15m.get('macd', 0))
            macd_signal_15m = float(indicators_15m.get('macd_signal', 0))
            macd_hist_15m = macd_15m - macd_signal_15m

            features = [
                # RSI (normalized to -1 to +1)
                (rsi_15m - 50) / 50,  # F13
                (rsi_1h - 50) / 50,   # F14
                (rsi_4h - 50) / 50,   # F15

                # RSI extremes (binary indicators)
                1.0 if (rsi_15m < 30 and side == 'LONG') or (rsi_15m > 70 and side == 'SHORT') else 0.0,  # F16

                # MACD
                macd_hist_15m,  # F17
                1.0 if macd_15m > macd_signal_15m else 0.0,  # F18 (bullish cross)

                # RSI slope (momentum of momentum)
                self._calculate_rsi_slope(snapshot),  # F19

                # Stochastic (if available)
                float(indicators_15m.get('stoch_k', 50)) / 100,  # F20
            ]

            return features

        except Exception as e:
            logger.warning(f"Momentum feature extraction failed: {e}")
            return [0.0] * 8

    def _extract_volume_features(self, snapshot: Dict[str, Any]) -> List[float]:
        """Extract 6 volume analysis features."""
        try:
            volume_24h = float(snapshot.get('volume_24h', 0))
            indicators_15m = snapshot.get('indicators', {}).get('15m', {})

            volume_current = float(indicators_15m.get('volume', 0))
            volume_sma = float(indicators_15m.get('volume_sma_20', volume_current))

            # Safe division checks
            volume_surge = (volume_current / volume_sma) if volume_sma > 1e-10 else 1.0
            volume_volatility = (float(indicators_15m.get('volume_std', 0)) / volume_sma) if volume_sma > 1e-10 else 0.0
            high_volume = 1.0 if (volume_sma > 1e-10 and (volume_current / volume_sma) > 1.5) else 0.0

            features = [
                # Volume surge
                volume_surge,  # F21

                # 24h volume (log scale for stability)
                np.log1p(volume_24h) / 20,  # F22 (normalized)

                # Volume trend (increasing/decreasing)
                self._calculate_volume_trend(snapshot),  # F23

                # Volume profile (buy vs sell pressure)
                float(snapshot.get('buy_sell_ratio', 1.0)),  # F24

                # Volume volatility
                volume_volatility,  # F25

                # High volume confirmation
                high_volume,  # F26
            ]

            return features

        except Exception as e:
            logger.warning(f"Volume feature extraction failed: {e}")
            return [0.0] * 6

    def _extract_timeframe_features(self, snapshot: Dict[str, Any], side: str) -> List[float]:
        """Extract 4 multi-timeframe alignment features."""
        try:
            indicators_15m = snapshot.get('indicators', {}).get('15m', {})
            indicators_1h = snapshot.get('indicators', {}).get('1h', {})
            indicators_4h = snapshot.get('indicators', {}).get('4h', {})

            # Trend alignment
            trend_15m = indicators_15m.get('trend', 'neutral')
            trend_1h = indicators_1h.get('trend', 'neutral')
            trend_4h = indicators_4h.get('trend', 'neutral')

            target_trend = 'uptrend' if side == 'LONG' else 'downtrend'

            features = [
                # Trend alignment score (0-3)
                sum([
                    1.0 if trend_15m == target_trend else 0.0,
                    1.0 if trend_1h == target_trend else 0.0,
                    1.0 if trend_4h == target_trend else 0.0,
                ]) / 3,  # F27

                # RSI alignment
                self._calculate_rsi_alignment(snapshot, side),  # F28

                # MACD alignment
                self._calculate_macd_alignment(snapshot, side),  # F29

                # Overall timeframe strength
                self._calculate_timeframe_strength(snapshot, side),  # F30
            ]

            return features

        except Exception as e:
            logger.warning(f"Timeframe feature extraction failed: {e}")
            return [0.0] * 4

    def _extract_structure_features(self, snapshot: Dict[str, Any], side: str) -> List[float]:
        """Extract 6 market structure features."""
        try:
            price = float(snapshot.get('current_price', 0))
            support = float(snapshot.get('nearest_support', price * 0.95))
            resistance = float(snapshot.get('nearest_resistance', price * 1.05))

            features = [
                # Distance to support/resistance (normalized)
                (price - support) / price if price > 0 else 0,      # F31
                (resistance - price) / price if price > 0 else 0,   # F32

                # Risk/Reward ratio
                ((resistance - price) / (price - support)) if (price - support) > 0 else 1.0,  # F33

                # Market regime
                self._encode_market_regime(snapshot),  # F34

                # Volatility regime
                self._encode_volatility_regime(snapshot),  # F35

                # Price position in range (0-1)
                ((price - support) / (resistance - support)) if (resistance - support) > 0 else 0.5,  # F36
            ]

            return features

        except Exception as e:
            logger.warning(f"Structure feature extraction failed: {e}")
            return [0.0] * 6

    def _extract_sentiment_features(self, snapshot: Dict[str, Any]) -> List[float]:
        """Extract 4 sentiment and external factor features."""
        try:
            features = [
                # Funding rate (perpetual futures)
                float(snapshot.get('funding_rate', {}).get('rate', 0)) * 100,  # F37

                # Market breadth sentiment
                float(snapshot.get('market_breadth', {}).get('bullish_percent', 50)) / 100,  # F38

                # Fear & Greed index (if available)
                float(snapshot.get('fear_greed_index', 50)) / 100,  # F39

                # Time of day factor (hour / 24)
                float(snapshot.get('hour_of_day', 12)) / 24,  # F40
            ]

            return features

        except Exception as e:
            logger.warning(f"Sentiment feature extraction failed: {e}")
            return [0.0] * 4

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def _calculate_bb_position(self, price: float, indicators: Dict) -> float:
        """Calculate position within Bollinger Bands (0-1)."""
        try:
            bb_upper = float(indicators.get('bb_upper', price * 1.02))
            bb_lower = float(indicators.get('bb_lower', price * 0.98))

            if bb_upper == bb_lower:
                return 0.5

            position = (price - bb_lower) / (bb_upper - bb_lower)
            return max(0, min(1, position))  # Clamp to 0-1

        except:
            return 0.5

    def _calculate_rsi_slope(self, snapshot: Dict) -> float:
        """Calculate RSI momentum (slope)."""
        try:
            # Approximate slope from 15m RSI
            rsi_current = float(snapshot.get('indicators', {}).get('15m', {}).get('rsi', 50))
            rsi_history = snapshot.get('rsi_history', [rsi_current])

            if len(rsi_history) < 2:
                return 0.0

            # Simple slope: (current - previous) / periods
            slope = (rsi_current - rsi_history[-2]) / 1
            return slope / 10  # Normalize

        except:
            return 0.0

    def _calculate_volume_trend(self, snapshot: Dict) -> float:
        """Calculate volume trend direction."""
        try:
            volume_history = snapshot.get('volume_history', [])

            if len(volume_history) < 3:
                return 0.0

            # Linear regression slope
            x = np.arange(len(volume_history))
            y = np.array(volume_history)

            # Check for zero or near-zero mean to prevent division by zero
            y_mean = np.mean(y)
            if y_mean < 1e-10:  # Extremely low volume
                return 0.0

            slope = np.polyfit(x, y, 1)[0]
            return np.tanh(slope / y_mean)  # Normalized slope

        except Exception as e:
            # Catch all errors including division by zero, polyfit errors
            return 0.0

    def _calculate_rsi_alignment(self, snapshot: Dict, side: str) -> float:
        """Calculate RSI alignment across timeframes."""
        try:
            rsi_15m = float(snapshot.get('indicators', {}).get('15m', {}).get('rsi', 50))
            rsi_1h = float(snapshot.get('indicators', {}).get('1h', {}).get('rsi', 50))
            rsi_4h = float(snapshot.get('indicators', {}).get('4h', {}).get('rsi', 50))

            if side == 'LONG':
                # Count oversold signals
                score = sum([
                    1.0 if rsi_15m < 40 else 0.0,
                    1.0 if rsi_1h < 40 else 0.0,
                    1.0 if rsi_4h < 40 else 0.0,
                ])
            else:  # SHORT
                # Count overbought signals
                score = sum([
                    1.0 if rsi_15m > 60 else 0.0,
                    1.0 if rsi_1h > 60 else 0.0,
                    1.0 if rsi_4h > 60 else 0.0,
                ])

            return score / 3

        except:
            return 0.0

    def _calculate_macd_alignment(self, snapshot: Dict, side: str) -> float:
        """Calculate MACD alignment across timeframes."""
        try:
            indicators_15m = snapshot.get('indicators', {}).get('15m', {})
            indicators_1h = snapshot.get('indicators', {}).get('1h', {})

            macd_15m = float(indicators_15m.get('macd', 0))
            macd_signal_15m = float(indicators_15m.get('macd_signal', 0))
            macd_1h = float(indicators_1h.get('macd', 0))
            macd_signal_1h = float(indicators_1h.get('macd_signal', 0))

            if side == 'LONG':
                score = sum([
                    1.0 if macd_15m > macd_signal_15m else 0.0,
                    1.0 if macd_1h > macd_signal_1h else 0.0,
                ])
            else:  # SHORT
                score = sum([
                    1.0 if macd_15m < macd_signal_15m else 0.0,
                    1.0 if macd_1h < macd_signal_1h else 0.0,
                ])

            return score / 2

        except:
            return 0.0

    def _calculate_timeframe_strength(self, snapshot: Dict, side: str) -> float:
        """Calculate overall multi-timeframe confirmation strength."""
        try:
            # Combine trend, RSI, and MACD alignment
            trend_score = self._extract_timeframe_features(snapshot, side)[0]
            rsi_score = self._calculate_rsi_alignment(snapshot, side)
            macd_score = self._calculate_macd_alignment(snapshot, side)

            # Weighted average
            strength = (trend_score * 0.4) + (rsi_score * 0.3) + (macd_score * 0.3)
            return strength

        except:
            return 0.0

    def _encode_market_regime(self, snapshot: Dict) -> float:
        """Encode market regime as numerical feature."""
        try:
            regime = snapshot.get('market_regime', 'RANGING')

            encoding = {
                'STRONG_BULLISH': 1.0,
                'BULLISH': 0.7,
                'RANGING': 0.0,
                'BEARISH': -0.7,
                'STRONG_BEARISH': -1.0,
                'CHOPPY': 0.0,
                'VOLATILE': 0.0,
            }

            return encoding.get(regime, 0.0)

        except:
            return 0.0

    def _encode_volatility_regime(self, snapshot: Dict) -> float:
        """Encode volatility regime (0=low, 1=high)."""
        try:
            atr = float(snapshot.get('indicators', {}).get('15m', {}).get('atr_percent', 2.0))

            # Normalize ATR to 0-1 scale
            # Low volatility: <1.5%, High volatility: >4%
            if atr < 1.5:
                return 0.0
            elif atr > 4.0:
                return 1.0
            else:
                return (atr - 1.5) / (4.0 - 1.5)

        except:
            return 0.5

    def _get_feature_names(self) -> List[str]:
        """Return list of all feature names for debugging/logging."""
        return [
            # Price Action (12)
            'price_dist_ema12_15m', 'price_dist_ema50_15m', 'price_dist_ema12_1h', 'price_dist_ema50_1h',
            'ema_alignment_15m', 'ema_alignment_1h', 'price_momentum_15m', 'price_momentum_1h',
            'atr_15m', 'atr_1h', 'bb_position_15m', 'bb_position_1h',

            # Momentum (8)
            'rsi_15m_norm', 'rsi_1h_norm', 'rsi_4h_norm', 'rsi_extreme',
            'macd_hist_15m', 'macd_cross_15m', 'rsi_slope', 'stoch_k',

            # Volume (6)
            'volume_surge', 'volume_24h_log', 'volume_trend', 'buy_sell_ratio',
            'volume_volatility', 'high_volume_flag',

            # Timeframe (4)
            'trend_alignment', 'rsi_alignment', 'macd_alignment', 'timeframe_strength',

            # Structure (6)
            'support_distance', 'resistance_distance', 'risk_reward', 'market_regime',
            'volatility_regime', 'price_range_position',

            # Sentiment (4)
            'funding_rate', 'market_breadth', 'fear_greed', 'hour_of_day',
        ]

    def get_feature_names(self) -> List[str]:
        """Public accessor for feature names."""
        return self.feature_names

    def validate_features(self, features: List[float]) -> bool:
        """Validate feature vector (check for NaN, Inf, correct length)."""
        if len(features) != self.feature_count:
            logger.error(f"❌ Invalid feature count: {len(features)} != {self.feature_count}")
            return False

        if any(np.isnan(f) or np.isinf(f) for f in features):
            logger.warning("⚠️ Features contain NaN or Inf values")
            return False

        return True


# Singleton instance
_feature_engineering: Optional[FeatureEngineering] = None


def get_feature_engineering() -> FeatureEngineering:
    """Get or create FeatureEngineering instance."""
    global _feature_engineering
    if _feature_engineering is None:
        _feature_engineering = FeatureEngineering()
    return _feature_engineering
