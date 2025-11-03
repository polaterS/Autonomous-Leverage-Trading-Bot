"""
Machine Learning Pattern Learner
Analyzes historical trades to learn winning patterns and adapt strategy
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MLPatternLearner:
    """
    Machine Learning system that learns from past trades to improve future decisions.

    Features:
    - Pattern recognition from winning/losing trades
    - Adaptive confidence threshold optimization
    - Market regime detection
    - Feature importance analysis
    - AI prediction accuracy tracking
    - Symbol-specific performance profiling
    """

    def __init__(self, db_client):
        self.db = db_client

        # Performance tracking
        self.symbol_performance: Dict[str, Dict] = defaultdict(lambda: {
            'wins': 0,
            'losses': 0,
            'total_pnl': Decimal('0'),
            'avg_confidence': 0.0,
            'best_timeframes': [],
            'best_patterns': []
        })

        # AI accuracy tracking
        self.ai_accuracy_by_confidence: Dict[float, Dict] = defaultdict(lambda: {
            'predictions': 0,
            'correct': 0,
            'accuracy': 0.0
        })

        # Pattern tracking
        self.winning_patterns: Dict[str, int] = defaultdict(int)
        self.losing_patterns: Dict[str, int] = defaultdict(int)

        # Market regime detection
        self.market_regimes: deque = deque(maxlen=100)  # Last 100 trades
        self.current_regime: str = "UNKNOWN"  # BULL, BEAR, RANGE, VOLATILE

        # Adaptive thresholds
        self.min_confidence_threshold: float = 0.75  # Dynamic adjustment
        self.optimal_leverage_by_symbol: Dict[str, float] = {}

        # Feature importance (what matters most)
        self.feature_scores: Dict[str, float] = {
            'rsi_divergence': 0.0,
            'volume_surge': 0.0,
            'macd_cross': 0.0,
            'support_resistance': 0.0,
            'order_flow': 0.0,
            'btc_correlation': 0.0,
            'funding_rate': 0.0,
            'liquidation_clusters': 0.0,
            'multi_timeframe_align': 0.0,
            'smart_money_concepts': 0.0
        }

        # Learning statistics
        self.total_trades_analyzed: int = 0
        self.learning_rate: float = 0.1  # How fast to adapt

        logger.info("✅ ML Pattern Learner initialized")


    async def initialize(self):
        """Load historical data and build initial models"""
        logger.info("🧠 Initializing ML Pattern Learner with historical data...")

        try:
            # Load last 500 trades
            trades = await self.db.get_trade_history(limit=500)

            if not trades or len(trades) < 10:
                logger.warning("⚠️ Not enough historical trades (<10), starting fresh")
                return

            logger.info(f"📊 Analyzing {len(trades)} historical trades...")

            # Analyze each trade
            for trade in trades:
                await self._analyze_completed_trade(trade, initial_load=True)

            # Detect current market regime
            await self._detect_market_regime()

            # Calculate optimal thresholds
            self._optimize_confidence_threshold()

            logger.info(f"✅ ML initialization complete:")
            logger.info(f"  - Total trades analyzed: {self.total_trades_analyzed}")
            logger.info(f"  - Current market regime: {self.current_regime}")
            logger.info(f"  - Optimal confidence threshold: {self.min_confidence_threshold:.1%}")
            logger.info(f"  - Top performing symbols: {self._get_top_symbols(3)}")

        except Exception as e:
            logger.error(f"❌ ML initialization failed: {e}", exc_info=True)


    async def _analyze_completed_trade(self, trade: Dict, initial_load: bool = False):
        """Analyze a completed trade to extract patterns"""
        try:
            symbol = trade.get('symbol', 'UNKNOWN')
            is_winner = trade.get('is_winner', False)
            pnl = trade.get('realized_pnl_usd', Decimal('0'))
            ai_confidence = trade.get('ai_confidence', 0.0)
            leverage = trade.get('leverage', 1)
            patterns = trade.get('ai_reasoning', '')  # Extract patterns from AI reasoning

            # Update symbol performance
            perf = self.symbol_performance[symbol]
            if is_winner:
                perf['wins'] += 1
            else:
                perf['losses'] += 1
            perf['total_pnl'] += pnl

            # Track AI accuracy by confidence level
            confidence_bucket = round(ai_confidence * 20) / 20  # 0.75, 0.80, 0.85, etc.
            acc = self.ai_accuracy_by_confidence[confidence_bucket]
            acc['predictions'] += 1
            if is_winner:
                acc['correct'] += 1
            acc['accuracy'] = acc['correct'] / acc['predictions']

            # Extract and track patterns
            pattern_keywords = self._extract_pattern_keywords(patterns)
            for keyword in pattern_keywords:
                if is_winner:
                    self.winning_patterns[keyword] += 1
                else:
                    self.losing_patterns[keyword] += 1

            # Update feature scores based on trade outcome
            if 'divergence' in patterns.lower():
                self._update_feature_score('rsi_divergence', is_winner)
            if 'volume' in patterns.lower():
                self._update_feature_score('volume_surge', is_winner)
            if 'macd' in patterns.lower():
                self._update_feature_score('macd_cross', is_winner)
            if 'support' in patterns.lower() or 'resistance' in patterns.lower():
                self._update_feature_score('support_resistance', is_winner)
            if 'order flow' in patterns.lower():
                self._update_feature_score('order_flow', is_winner)
            if 'btc' in patterns.lower():
                self._update_feature_score('btc_correlation', is_winner)
            if 'funding' in patterns.lower():
                self._update_feature_score('funding_rate', is_winner)
            if 'liquidation' in patterns.lower():
                self._update_feature_score('liquidation_clusters', is_winner)
            if 'timeframe' in patterns.lower():
                self._update_feature_score('multi_timeframe_align', is_winner)
            if 'smart money' in patterns.lower() or 'order block' in patterns.lower():
                self._update_feature_score('smart_money_concepts', is_winner)

            # Track market regime at time of trade
            market_state = {
                'timestamp': trade.get('exit_time', datetime.now()),
                'symbol': symbol,
                'side': trade.get('side', 'LONG'),
                'is_winner': is_winner,
                'pnl_percent': float(pnl) / float(trade.get('entry_price', 1) * trade.get('quantity', 1)) * 100
            }
            self.market_regimes.append(market_state)

            self.total_trades_analyzed += 1

            if not initial_load:
                logger.info(f"📈 Learned from trade: {symbol} {'WIN' if is_winner else 'LOSS'} "
                           f"(Confidence: {ai_confidence:.1%}, Total analyzed: {self.total_trades_analyzed})")

        except Exception as e:
            logger.error(f"❌ Error analyzing trade: {e}")


    def _extract_pattern_keywords(self, reasoning: str) -> List[str]:
        """Extract key pattern indicators from AI reasoning"""
        keywords = []
        reasoning_lower = reasoning.lower()

        pattern_indicators = [
            'bullish divergence', 'bearish divergence',
            'volume breakout', 'volume surge',
            'macd crossover', 'golden cross', 'death cross',
            'support bounce', 'resistance break',
            'order flow imbalance',
            'funding rate extreme',
            'liquidation cluster',
            'smart money accumulation', 'smart money distribution',
            'order block', 'fair value gap',
            'trend continuation', 'trend reversal',
            'range breakout', 'consolidation',
            'oversold bounce', 'overbought rejection'
        ]

        for pattern in pattern_indicators:
            if pattern in reasoning_lower:
                keywords.append(pattern)

        return keywords


    def _update_feature_score(self, feature: str, is_winner: bool):
        """Update feature importance score using exponential moving average"""
        # Award +1 for win, -1 for loss
        reward = 1.0 if is_winner else -1.0

        # EMA update: new_score = (1 - lr) * old_score + lr * reward
        current_score = self.feature_scores.get(feature, 0.0)
        self.feature_scores[feature] = (1 - self.learning_rate) * current_score + self.learning_rate * reward


    async def _detect_market_regime(self):
        """Detect current market regime based on recent trades"""
        if len(self.market_regimes) < 20:
            self.current_regime = "INSUFFICIENT_DATA"
            return

        recent_trades = list(self.market_regimes)[-20:]  # Last 20 trades

        # Calculate metrics
        win_rate = sum(1 for t in recent_trades if t['is_winner']) / len(recent_trades)
        avg_pnl = np.mean([t['pnl_percent'] for t in recent_trades])
        pnl_volatility = np.std([t['pnl_percent'] for t in recent_trades])

        long_wins = sum(1 for t in recent_trades if t['side'] == 'LONG' and t['is_winner'])
        short_wins = sum(1 for t in recent_trades if t['side'] == 'SHORT' and t['is_winner'])

        # Regime detection logic
        if win_rate > 0.60 and avg_pnl > 2.0:
            if long_wins > short_wins * 1.5:
                self.current_regime = "BULL_TREND"
            elif short_wins > long_wins * 1.5:
                self.current_regime = "BEAR_TREND"
            else:
                self.current_regime = "FAVORABLE_VOLATILE"

        elif win_rate < 0.40 and pnl_volatility > 5.0:
            self.current_regime = "HIGH_VOLATILITY_UNFAVORABLE"

        elif 0.45 <= win_rate <= 0.55 and pnl_volatility < 3.0:
            self.current_regime = "RANGE_BOUND"

        else:
            self.current_regime = "MIXED_CONDITIONS"

        logger.info(f"📊 Market regime detected: {self.current_regime} "
                   f"(Win rate: {win_rate:.1%}, Avg PnL: {avg_pnl:.2f}%, Volatility: {pnl_volatility:.2f}%)")


    def _optimize_confidence_threshold(self):
        """Find optimal confidence threshold based on historical accuracy"""
        if not self.ai_accuracy_by_confidence:
            return

        # Find confidence level with highest accuracy and >20 samples
        best_threshold = 0.75
        best_accuracy = 0.0

        for confidence, stats in self.ai_accuracy_by_confidence.items():
            if stats['predictions'] >= 20 and stats['accuracy'] > best_accuracy:
                best_accuracy = stats['accuracy']
                best_threshold = confidence

        # Be conservative: only lower threshold if accuracy is very high
        if best_accuracy >= 0.65 and best_threshold < self.min_confidence_threshold:
            self.min_confidence_threshold = best_threshold

        logger.info(f"🎯 Optimal confidence threshold: {self.min_confidence_threshold:.1%} "
                   f"(Best observed accuracy: {best_accuracy:.1%})")


    def _get_top_symbols(self, limit: int = 5) -> List[str]:
        """Get top performing symbols by win rate and total PnL"""
        symbol_scores = []

        for symbol, perf in self.symbol_performance.items():
            total_trades = perf['wins'] + perf['losses']
            if total_trades < 3:  # Need at least 3 trades
                continue

            win_rate = perf['wins'] / total_trades
            total_pnl = float(perf['total_pnl'])

            # Combined score: 70% win rate + 30% total PnL
            score = 0.7 * win_rate + 0.3 * (total_pnl / 100)  # Normalize PnL
            symbol_scores.append((symbol, score, win_rate, total_pnl))

        # Sort by score descending
        symbol_scores.sort(key=lambda x: x[1], reverse=True)

        return [f"{sym} ({wr:.1%} WR, ${pnl:.2f})"
                for sym, score, wr, pnl in symbol_scores[:limit]]


    async def analyze_opportunity(self, symbol: str, ai_analysis: Dict, market_data: Dict) -> Dict:
        """
        Enhance AI analysis with ML insights

        Returns:
        {
            'should_trade': bool,
            'adjusted_confidence': float,
            'ml_score': float,
            'reasoning': str,
            'regime_warning': str (optional)
        }
        """
        try:
            base_confidence = ai_analysis.get('confidence', 0.0)

            # 1. Check symbol historical performance
            perf = self.symbol_performance.get(symbol)
            symbol_modifier = 0.0

            if perf and (perf['wins'] + perf['losses']) >= 5:
                win_rate = perf['wins'] / (perf['wins'] + perf['losses'])
                if win_rate >= 0.60:
                    symbol_modifier = +0.05  # Boost good performers
                elif win_rate <= 0.35:
                    symbol_modifier = -0.10  # Penalize bad performers

            # 2. Check AI accuracy at this confidence level
            confidence_bucket = round(base_confidence * 20) / 20
            acc_stats = self.ai_accuracy_by_confidence.get(confidence_bucket)
            accuracy_modifier = 0.0

            if acc_stats and acc_stats['predictions'] >= 10:
                if acc_stats['accuracy'] >= 0.65:
                    accuracy_modifier = +0.03
                elif acc_stats['accuracy'] <= 0.45:
                    accuracy_modifier = -0.07

            # 3. Feature importance boost
            reasoning = ai_analysis.get('reasoning', '')
            feature_boost = 0.0

            # Check if trade uses high-value features
            for feature, score in self.feature_scores.items():
                if score > 0.3:  # Positive feature
                    feature_key = feature.replace('_', ' ')
                    if feature_key in reasoning.lower():
                        feature_boost += 0.02  # Small boost per good feature

            feature_boost = min(feature_boost, 0.08)  # Cap at +8%

            # 4. Market regime adjustment
            regime_modifier = 0.0
            regime_warning = ""

            if self.current_regime == "HIGH_VOLATILITY_UNFAVORABLE":
                regime_modifier = -0.10
                regime_warning = "⚠️ High volatility regime detected - reduced confidence"
            elif self.current_regime in ["BULL_TREND", "BEAR_TREND"]:
                side = ai_analysis.get('side', 'LONG')
                if (self.current_regime == "BULL_TREND" and side == "LONG") or \
                   (self.current_regime == "BEAR_TREND" and side == "SHORT"):
                    regime_modifier = +0.05
                    regime_warning = f"✅ Trade aligned with {self.current_regime}"

            # 5. Pattern matching
            pattern_boost = 0.0
            detected_patterns = self._extract_pattern_keywords(reasoning)

            for pattern in detected_patterns:
                wins = self.winning_patterns.get(pattern, 0)
                losses = self.losing_patterns.get(pattern, 0)

                if wins + losses >= 3:  # Need at least 3 occurrences
                    pattern_win_rate = wins / (wins + losses)
                    if pattern_win_rate >= 0.65:
                        pattern_boost += 0.02
                    elif pattern_win_rate <= 0.35:
                        pattern_boost -= 0.03

            pattern_boost = max(-0.10, min(pattern_boost, 0.08))  # Cap at ±8-10%

            # 6. Calculate final ML-adjusted confidence
            ml_adjustment = (
                symbol_modifier +
                accuracy_modifier +
                feature_boost +
                regime_modifier +
                pattern_boost
            )

            adjusted_confidence = base_confidence + ml_adjustment
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))  # Clamp to [0, 1]

            # 7. Decision
            should_trade = adjusted_confidence >= self.min_confidence_threshold

            # 8. Calculate ML score (0-100)
            ml_score = adjusted_confidence * 100

            reasoning_parts = []
            if abs(symbol_modifier) > 0.01:
                reasoning_parts.append(f"Symbol history: {symbol_modifier:+.1%}")
            if abs(accuracy_modifier) > 0.01:
                reasoning_parts.append(f"AI accuracy: {accuracy_modifier:+.1%}")
            if feature_boost > 0.01:
                reasoning_parts.append(f"Feature boost: +{feature_boost:.1%}")
            if abs(regime_modifier) > 0.01:
                reasoning_parts.append(f"Regime: {regime_modifier:+.1%}")
            if abs(pattern_boost) > 0.01:
                reasoning_parts.append(f"Patterns: {pattern_boost:+.1%}")

            ml_reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No adjustments"

            logger.info(f"🧠 ML Analysis for {symbol}:")
            logger.info(f"   Base confidence: {base_confidence:.1%} → Adjusted: {adjusted_confidence:.1%}")
            logger.info(f"   ML adjustment: {ml_adjustment:+.1%} ({ml_reasoning})")
            logger.info(f"   Decision: {'✅ TRADE' if should_trade else '❌ SKIP'}")

            return {
                'should_trade': should_trade,
                'adjusted_confidence': adjusted_confidence,
                'ml_score': ml_score,
                'ml_adjustment': ml_adjustment,
                'reasoning': ml_reasoning,
                'regime_warning': regime_warning,
                'feature_scores': {k: v for k, v in self.feature_scores.items() if v > 0.2},
                'symbol_win_rate': perf['wins'] / (perf['wins'] + perf['losses']) if perf and (perf['wins'] + perf['losses']) > 0 else None
            }

        except Exception as e:
            logger.error(f"❌ ML analysis failed: {e}", exc_info=True)
            return {
                'should_trade': base_confidence >= 0.75,
                'adjusted_confidence': base_confidence,
                'ml_score': 0.0,
                'reasoning': f"ML error: {str(e)}"
            }


    async def on_trade_completed(self, trade: Dict):
        """Called when a trade is completed to update ML models"""
        logger.info(f"🎓 Learning from completed trade: {trade.get('symbol')}")
        await self._analyze_completed_trade(trade, initial_load=False)

        # Periodically re-detect regime and optimize thresholds
        if self.total_trades_analyzed % 10 == 0:
            await self._detect_market_regime()
            self._optimize_confidence_threshold()

            # Log updated insights
            logger.info(f"📊 ML Insights Update (after {self.total_trades_analyzed} trades):")
            logger.info(f"   Current regime: {self.current_regime}")
            logger.info(f"   Optimal threshold: {self.min_confidence_threshold:.1%}")
            logger.info(f"   Top symbols: {', '.join(self._get_top_symbols(3))}")
            logger.info(f"   Best features: {self._get_top_features(3)}")


    def _get_top_features(self, limit: int = 3) -> str:
        """Get top performing features"""
        sorted_features = sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)
        return ", ".join([f"{feat.replace('_', ' ')}: {score:.2f}"
                         for feat, score in sorted_features[:limit] if score > 0])


    def get_statistics(self) -> Dict:
        """Get comprehensive ML statistics"""
        return {
            'total_trades_analyzed': self.total_trades_analyzed,
            'current_regime': self.current_regime,
            'min_confidence_threshold': self.min_confidence_threshold,
            'top_symbols': self._get_top_symbols(5),
            'feature_scores': dict(sorted(self.feature_scores.items(),
                                         key=lambda x: x[1], reverse=True)),
            'ai_accuracy_by_confidence': dict(self.ai_accuracy_by_confidence),
            'winning_patterns': dict(sorted(self.winning_patterns.items(),
                                           key=lambda x: x[1], reverse=True)[:10]),
            'losing_patterns': dict(sorted(self.losing_patterns.items(),
                                          key=lambda x: x[1], reverse=True)[:10])
        }


# Singleton instance
_ml_learner_instance: Optional[MLPatternLearner] = None


async def get_ml_learner(db_client=None):
    """Get or create ML Pattern Learner singleton"""
    global _ml_learner_instance

    if _ml_learner_instance is None:
        if db_client is None:
            from src.database import get_db_client
            db_client = await get_db_client()

        _ml_learner_instance = MLPatternLearner(db_client)
        await _ml_learner_instance.initialize()

    return _ml_learner_instance
