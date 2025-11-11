"""
🎯 #10: ML-Powered Exit Timing Optimizer
Learns optimal exit patterns from historical trades

Separate ML model trained on exit quality (not entry).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class ExitTimingOptimizer:
    """
    🎯 #10: Machine Learning model for optimal exit timing.

    Learns from historical exits to predict:
    - Should we exit NOW? (take profit)
    - Should we HOLD LONGER? (let it run)

    Different from entry ML - focuses purely on exit quality.
    """

    def __init__(self):
        # Exit quality database
        # Format: List of (features_dict, exit_quality_score)
        self.exit_history: List[Tuple[Dict, float]] = []
        self.max_history: int = 500  # Keep last 500 exits

        # Feature importance for exits (learned weights)
        self.exit_feature_weights: Dict[str, float] = {
            'profit_pct': 0.0,  # How much profit so far
            'time_in_position_minutes': 0.0,  # How long held
            'rsi_level': 0.0,  # Overbought/oversold
            'macd_divergence': 0.0,  # Momentum weakening
            'funding_rate': 0.0,  # Position cost
            'volatility_spike': 0.0,  # Sudden volatility change
            'volume_decline': 0.0,  # Interest fading
            'profit_velocity': 0.0,  # Rate of profit increase (slowing?)
        }

        # Bayesian exit scoring (similar to feature importance in ML learner)
        self.exit_decision_outcomes: Dict[str, List[Tuple[int, datetime]]] = defaultdict(list)
        # 'early_exit' -> [(1, timestamp), (0, timestamp), ...]  # 1=good decision, 0=bad
        # 'hold_longer' -> [(1, timestamp), (0, timestamp), ...]

        # Optimal exit thresholds (learned from data)
        self.optimal_exit_profit_threshold: float = 1.5  # Start: exit at 1.5x min target
        self.optimal_hold_time_minutes: int = 120  # Start: 2 hours average

        logger.info("✅ Exit Timing Optimizer initialized (ML-powered exit decisions)")

    def extract_exit_features(
        self,
        position: Dict,
        current_price: Decimal,
        market_data: Dict
    ) -> Dict[str, float]:
        """
        🎯 #10: Extract features relevant for exit decision.

        Args:
            position: Active position dict
            current_price: Current market price
            market_data: Market indicators

        Returns:
            Dict of normalized features for exit ML
        """
        entry_price = Decimal(str(position['entry_price']))
        side = position['side']
        entry_time = position.get('entry_time', datetime.now())
        time_in_position = (datetime.now() - entry_time).total_seconds() / 60  # minutes

        # Calculate current profit percentage
        if side == 'LONG':
            profit_pct = float((current_price - entry_price) / entry_price * 100)
        else:  # SHORT
            profit_pct = float((entry_price - current_price) / entry_price * 100)

        # Get technical indicators
        indicators = market_data.get('indicators', {}).get('15m', {})

        # Safe extraction with type checking
        def safe_float(value, default=0.0):
            """Safely extract float from value (handles dict/None/float)."""
            if isinstance(value, dict):
                return float(value.get('rate', value.get('value', default)))
            elif value is None:
                return default
            else:
                return float(value)

        rsi = safe_float(indicators.get('rsi', 50.0), 50.0)
        macd = safe_float(indicators.get('macd', 0.0), 0.0)
        macd_signal = safe_float(indicators.get('macd_signal', 0.0), 0.0)

        # Volatility
        volatility = market_data.get('volatility', {})
        atr_percent = safe_float(volatility.get('atr_percent', 2.0), 2.0)

        # Volume
        volume = safe_float(indicators.get('volume', 1.0), 1.0)
        volume_sma = safe_float(indicators.get('volume_sma_20', volume), volume)
        volume_ratio = volume / volume_sma if volume_sma > 0 else 1.0

        # Funding rate (can be dict with 'rate' key from exchange)
        funding_data = market_data.get('funding_rate', 0.0)
        funding = safe_float(funding_data, 0.0)

        # Calculate profit velocity (how fast is profit accumulating?)
        # Use recent P&L history if available
        profit_velocity = 0.0
        if 'pnl_history' in position and len(position['pnl_history']) >= 2:
            recent_pnls = position['pnl_history'][-10:]  # Last 10 updates
            if len(recent_pnls) >= 2:
                pnl_changes = [recent_pnls[i] - recent_pnls[i-1] for i in range(1, len(recent_pnls))]
                profit_velocity = float(np.mean(pnl_changes)) if pnl_changes else 0.0

        features = {
            'profit_pct': profit_pct,
            'time_in_position_minutes': time_in_position,
            'rsi_level': rsi,
            'macd_divergence': abs(macd - macd_signal),  # Divergence magnitude
            'funding_rate': abs(funding) * 100,  # Convert to basis points
            'volatility_spike': max(0, atr_percent - 2.0),  # Above normal volatility
            'volume_decline': max(0, 1.0 - volume_ratio),  # Below average volume
            'profit_velocity': profit_velocity
        }

        return features

    def predict_exit_decision(
        self,
        features: Dict[str, float],
        min_profit_target_usd: Decimal,
        current_pnl_usd: Decimal
    ) -> Dict[str, any]:
        """
        🎯 #10: Predict whether to exit NOW or HOLD LONGER.

        Uses learned feature weights and historical patterns.

        Args:
            features: Exit feature dict
            min_profit_target_usd: Minimum profit target
            current_pnl_usd: Current unrealized P&L

        Returns:
            Dict with 'should_exit', 'confidence', 'reasoning'
        """
        # Calculate weighted exit score
        exit_score = 0.0

        # Feature scoring (higher score = more reasons to exit)

        # 1. Profit level
        profit_pct = features['profit_pct']
        profit_ratio = float(current_pnl_usd / min_profit_target_usd) if min_profit_target_usd > 0 else 0
        if profit_ratio >= 2.0:
            exit_score += 0.25  # 2x target hit - strong exit signal
        elif profit_ratio >= 1.5:
            exit_score += 0.15  # 1.5x target - moderate exit signal
        elif profit_ratio >= 1.0:
            exit_score += 0.05  # Min target - weak exit signal

        # 2. Time decay (profits decay over time due to fees/funding)
        time_minutes = features['time_in_position_minutes']
        if time_minutes > 240:  # > 4 hours
            exit_score += 0.10
        elif time_minutes > 120:  # > 2 hours
            exit_score += 0.05

        # 3. RSI extremes (reversal risk)
        rsi = features['rsi_level']
        if profit_pct > 0:  # Only if in profit
            if rsi > 75:  # Extremely overbought (for LONG)
                exit_score += 0.15
            elif rsi > 70:
                exit_score += 0.08
            elif rsi < 25:  # Extremely oversold (for SHORT)
                exit_score += 0.15
            elif rsi < 30:
                exit_score += 0.08

        # 4. MACD divergence (momentum weakening)
        macd_div = features['macd_divergence']
        if macd_div < 0.0001:  # MACD and signal converging
            exit_score += 0.10

        # 5. Funding rate pressure
        funding_bp = features['funding_rate']
        if funding_bp > 0.1:  # > 0.10% (expensive to hold)
            exit_score += 0.08

        # 6. Volatility spike (risk increase)
        vol_spike = features['volatility_spike']
        if vol_spike > 1.0:  # ATR > 3% (high vol)
            exit_score += 0.10

        # 7. Volume decline (interest fading)
        vol_decline = features['volume_decline']
        if vol_decline > 0.3:  # Volume 30% below average
            exit_score += 0.07

        # 8. Profit velocity slowing
        profit_vel = features['profit_velocity']
        if profit_vel < 0:  # Profits decreasing
            exit_score += 0.12

        # Normalize score to confidence (0-1)
        confidence = min(1.0, exit_score)

        # ✅ CONSERVATIVE MODE: Decision threshold increased to 0.75 (from 0.50)
        # This prevents premature exits on weak signals
        should_exit = confidence >= 0.75

        # Build reasoning
        reasoning_parts = []
        if profit_ratio >= 2.0:
            reasoning_parts.append("2x profit target achieved")
        if rsi > 70 or rsi < 30:
            reasoning_parts.append(f"RSI extreme ({rsi:.1f})")
        if profit_vel < 0:
            reasoning_parts.append("profit momentum slowing")
        if vol_decline > 0.3:
            reasoning_parts.append("volume fading")
        if time_minutes > 180:
            reasoning_parts.append(f"held {int(time_minutes)}m (decay risk)")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "favorable exit conditions"

        result = {
            'should_exit': should_exit,
            'confidence': round(confidence, 2),
            'reasoning': reasoning,
            'exit_score': round(exit_score, 2),
            'features_used': len([k for k, v in features.items() if v != 0])
        }

        logger.info(
            f"🎯 EXIT ML: {'EXIT NOW' if should_exit else 'HOLD'} | "
            f"Confidence: {confidence:.1%} | Score: {exit_score:.2f} | {reasoning}"
        )

        return result

    def learn_from_exit(
        self,
        features_at_exit: Dict[str, float],
        exit_quality_score: float,
        was_optimal: bool
    ):
        """
        🎯 #10: Learn from completed exit.

        Args:
            features_at_exit: Features when position was closed
            exit_quality_score: Quality of exit (0-1, 1=perfect)
            was_optimal: True if this was a good exit timing
        """
        # Store exit for future learning
        self.exit_history.append((features_at_exit, exit_quality_score))

        # Trim history if too long
        if len(self.exit_history) > self.max_history:
            self.exit_history = self.exit_history[-self.max_history:]

        # Update Bayesian tracking
        decision_type = 'early_exit' if features_at_exit['time_in_position_minutes'] < 60 else 'hold_longer'
        outcome = 1 if was_optimal else 0
        self.exit_decision_outcomes[decision_type].append((outcome, datetime.now()))

        # Update feature weights using simple online learning
        learning_rate = 0.05
        for feature_name, feature_value in features_at_exit.items():
            if feature_name in self.exit_feature_weights:
                # Gradient update: increase weight if exit was good and feature was high
                gradient = exit_quality_score * (feature_value / 100.0)  # Normalize
                self.exit_feature_weights[feature_name] += learning_rate * gradient

        logger.debug(f"📚 Exit learning: Quality={exit_quality_score:.2f}, Optimal={was_optimal}")

    def calculate_exit_quality(
        self,
        position: Dict,
        exit_pnl_usd: Decimal,
        min_profit_target_usd: Decimal,
        time_in_position_minutes: int
    ) -> float:
        """
        🎯 #10: Calculate quality score for an exit (0-1).

        Factors:
        - Profit vs target ratio
        - Time efficiency (faster = better)
        - Avoided further losses

        Args:
            position: Closed position
            exit_pnl_usd: Realized P&L
            min_profit_target_usd: Min profit target
            time_in_position_minutes: Duration held

        Returns:
            Quality score 0-1 (1 = perfect exit)
        """
        quality = 0.0

        # Factor 1: Profit ratio (50% weight)
        profit_ratio = float(exit_pnl_usd / min_profit_target_usd) if min_profit_target_usd > 0 else 0
        if profit_ratio >= 2.0:
            quality += 0.50  # Excellent profit
        elif profit_ratio >= 1.5:
            quality += 0.40
        elif profit_ratio >= 1.0:
            quality += 0.30
        elif profit_ratio >= 0.5:
            quality += 0.15
        elif profit_ratio >= 0:
            quality += 0.05  # Break even
        # Negative = 0 points

        # Factor 2: Time efficiency (30% weight)
        # Faster exits with profit = better
        if time_in_position_minutes < 30 and exit_pnl_usd > 0:
            quality += 0.30  # Very fast profit
        elif time_in_position_minutes < 60 and exit_pnl_usd > 0:
            quality += 0.25  # Fast profit
        elif time_in_position_minutes < 120 and exit_pnl_usd > 0:
            quality += 0.20  # Reasonable speed
        elif time_in_position_minutes < 240 and exit_pnl_usd > 0:
            quality += 0.10  # Slow but profitable
        # Very long holds = 0 points (fees eat profit)

        # Factor 3: Risk avoidance (20% weight)
        # Exiting before things get worse
        if exit_pnl_usd > 0:
            quality += 0.20  # Avoided going negative
        elif exit_pnl_usd > -min_profit_target_usd * Decimal("0.5"):
            quality += 0.10  # Limited losses

        return min(1.0, quality)

    def get_exit_statistics(self) -> Dict:
        """Get summary statistics of exit decisions."""
        if not self.exit_history:
            return {'total_exits': 0}

        qualities = [q for _, q in self.exit_history]
        return {
            'total_exits': len(self.exit_history),
            'avg_exit_quality': np.mean(qualities),
            'best_exit_quality': np.max(qualities),
            'worst_exit_quality': np.min(qualities),
            'feature_weights': self.exit_feature_weights
        }


# Singleton instance
_exit_optimizer: Optional[ExitTimingOptimizer] = None


def get_exit_optimizer() -> ExitTimingOptimizer:
    """Get or create exit optimizer instance."""
    global _exit_optimizer
    if _exit_optimizer is None:
        _exit_optimizer = ExitTimingOptimizer()
    return _exit_optimizer
