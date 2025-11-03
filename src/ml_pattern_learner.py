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
from scipy import stats  # For statistical validation

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

        # 📊 STATISTICAL VALIDATION (NEW!)
        # Track win/loss outcomes per feature for t-test validation
        self.feature_outcomes: Dict[str, List[int]] = defaultdict(list)  # 1 = win, 0 = loss
        self.pattern_outcomes: Dict[str, List[int]] = defaultdict(list)
        self.min_sample_size: int = 30  # Minimum trades before trusting pattern

        logger.info("✅ ML Pattern Learner initialized with statistical validation")


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

                # 📊 Track outcome for statistical validation
                self.pattern_outcomes[keyword].append(1 if is_winner else 0)

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
        """Update feature importance score using exponential moving average + track for statistical validation"""
        # Award +1 for win, -1 for loss
        reward = 1.0 if is_winner else -1.0

        # EMA update: new_score = (1 - lr) * old_score + lr * reward
        current_score = self.feature_scores.get(feature, 0.0)
        self.feature_scores[feature] = (1 - self.learning_rate) * current_score + self.learning_rate * reward

        # 📊 Track outcome for statistical validation
        self.feature_outcomes[feature].append(1 if is_winner else 0)


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


    def _is_pattern_statistically_significant(self, pattern: str) -> Tuple[bool, float, str]:
        """
        Test if a pattern's win rate is statistically significant using t-test.

        Returns:
            Tuple of (is_significant, p_value, interpretation)
        """
        outcomes = self.pattern_outcomes.get(pattern, [])

        # Need minimum sample size
        if len(outcomes) < self.min_sample_size:
            return (False, 1.0, f"Insufficient data ({len(outcomes)}/{self.min_sample_size} trades)")

        # Calculate win rate
        win_rate = sum(outcomes) / len(outcomes)

        # H0: Win rate = 0.5 (random chance)
        # H1: Win rate ≠ 0.5 (pattern has predictive power)
        t_stat, p_value = stats.ttest_1samp(outcomes, 0.5)

        # Significance level: 0.05 (95% confidence)
        is_significant = p_value < 0.05

        if is_significant:
            if win_rate > 0.5:
                interpretation = f"✅ Reliable winning pattern ({win_rate:.1%} win rate, p={p_value:.4f})"
            else:
                interpretation = f"❌ Reliable losing pattern ({win_rate:.1%} win rate, p={p_value:.4f})"
        else:
            interpretation = f"⚠️ Not statistically significant ({win_rate:.1%} win rate, p={p_value:.4f} - likely random)"

        return (is_significant, p_value, interpretation)

    def _is_feature_reliable(self, feature: str) -> Tuple[bool, float, str]:
        """
        Test if a feature's impact is statistically significant.

        Returns:
            Tuple of (is_reliable, p_value, interpretation)
        """
        outcomes = self.feature_outcomes.get(feature, [])

        # Need minimum sample size
        if len(outcomes) < self.min_sample_size:
            return (False, 1.0, f"Insufficient data ({len(outcomes)}/{self.min_sample_size})")

        # Calculate win rate when this feature is present
        win_rate = sum(outcomes) / len(outcomes)

        # T-test: Is win rate significantly different from 50% (random)?
        t_stat, p_value = stats.ttest_1samp(outcomes, 0.5)

        is_reliable = p_value < 0.05 and win_rate > 0.55  # At least 55% win rate + significant

        if is_reliable:
            interpretation = f"✅ Reliable feature ({win_rate:.1%} win rate, p={p_value:.4f})"
        elif p_value < 0.05 and win_rate < 0.45:
            interpretation = f"❌ Negative feature ({win_rate:.1%} win rate, p={p_value:.4f} - AVOID)"
        else:
            interpretation = f"⚠️ Unreliable feature ({win_rate:.1%} win rate, p={p_value:.4f})"

        return (is_reliable, p_value, interpretation)

    def get_confidence_interval(self, pattern: str, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for pattern win rate.

        Returns:
            Tuple of (lower_bound, upper_bound) for win rate
        """
        outcomes = self.pattern_outcomes.get(pattern, [])

        if len(outcomes) < 10:
            return (0.0, 1.0)  # Too few samples, very wide interval

        win_rate = sum(outcomes) / len(outcomes)
        n = len(outcomes)

        # Standard error
        se = np.sqrt(win_rate * (1 - win_rate) / n)

        # Z-score for confidence level (95% → 1.96)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        # Confidence interval
        lower = max(0.0, win_rate - z_score * se)
        upper = min(1.0, win_rate + z_score * se)

        return (lower, upper)

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

            # 3. Feature importance boost (WITH STATISTICAL VALIDATION)
            reasoning = ai_analysis.get('reasoning', '')
            feature_boost = 0.0
            validated_features = []
            unreliable_features = []

            # Check if trade uses high-value features (statistically validated)
            for feature, score in self.feature_scores.items():
                if score > 0.3:  # Positive feature
                    feature_key = feature.replace('_', ' ')
                    if feature_key in reasoning.lower():
                        # 📊 STATISTICAL VALIDATION: Only trust validated features
                        is_reliable, p_value, interpretation = self._is_feature_reliable(feature)

                        if is_reliable:
                            feature_boost += 0.02  # Small boost per validated feature
                            validated_features.append(feature)
                            logger.debug(f"   ✅ Using validated feature '{feature}': {interpretation}")
                        else:
                            unreliable_features.append((feature, interpretation))
                            logger.warning(f"   ⚠️ Skipping unreliable feature '{feature}': {interpretation}")

            feature_boost = min(feature_boost, 0.08)  # Cap at +8%

            if unreliable_features:
                logger.info(f"   📊 Ignored {len(unreliable_features)} statistically unreliable features")

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

            # 5. Pattern matching (WITH STATISTICAL VALIDATION)
            pattern_boost = 0.0
            detected_patterns = self._extract_pattern_keywords(reasoning)
            validated_patterns = []
            weak_patterns = []
            negative_patterns = []

            for pattern in detected_patterns:
                wins = self.winning_patterns.get(pattern, 0)
                losses = self.losing_patterns.get(pattern, 0)
                total_occurrences = wins + losses

                # 📊 STATISTICAL VALIDATION: Test pattern significance
                is_significant, p_value, interpretation = self._is_pattern_statistically_significant(pattern)

                if not is_significant:
                    # Pattern not statistically validated - skip it
                    if total_occurrences < self.min_sample_size:
                        weak_patterns.append((pattern, f"{total_occurrences}/{self.min_sample_size} samples"))
                    else:
                        weak_patterns.append((pattern, f"p={p_value:.4f} (not significant)"))
                    continue

                # Pattern is statistically significant - use it!
                pattern_win_rate = wins / total_occurrences
                lower_ci, upper_ci = self.get_confidence_interval(pattern)

                if pattern_win_rate > 0.5:
                    # Winning pattern - boost confidence
                    # Stronger boost for higher confidence intervals
                    confidence_strength = min(lower_ci, 0.7)  # Cap at 70% lower bound
                    pattern_boost += 0.03 * (confidence_strength / 0.5)  # Scale boost
                    validated_patterns.append((pattern, pattern_win_rate, (lower_ci, upper_ci)))
                    logger.info(f"   ✅ Validated winning pattern '{pattern}': {pattern_win_rate:.1%} "
                               f"(95% CI: [{lower_ci:.1%}, {upper_ci:.1%}])")
                else:
                    # Losing pattern - reduce confidence
                    pattern_boost -= 0.04
                    negative_patterns.append((pattern, pattern_win_rate))
                    logger.warning(f"   ❌ Validated LOSING pattern '{pattern}': {pattern_win_rate:.1%} - REDUCING confidence")

            pattern_boost = max(-0.15, min(pattern_boost, 0.10))  # Cap at ±10-15%

            # Log validation summary
            if weak_patterns:
                logger.info(f"   ⚠️ Ignored {len(weak_patterns)} unvalidated patterns (insufficient data or p>0.05)")
            if validated_patterns:
                logger.info(f"   📊 Applied {len(validated_patterns)} statistically validated patterns")

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

            # 📊 Periodic statistical validation report (every 30 trades)
            if self.total_trades_analyzed % 30 == 0:
                self._log_statistical_validation_report()


    def _log_statistical_validation_report(self):
        """Log comprehensive statistical validation report"""
        logger.info("=" * 80)
        logger.info("📊 STATISTICAL VALIDATION REPORT")
        logger.info("=" * 80)

        # Feature validation
        logger.info("\n🔬 FEATURE VALIDATION:")
        validated_count = 0
        unreliable_count = 0

        for feature, score in sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True):
            outcomes = self.feature_outcomes.get(feature, [])
            if len(outcomes) >= self.min_sample_size:
                is_reliable, p_val, interp = self._is_feature_reliable(feature)
                win_rate = sum(outcomes) / len(outcomes)

                if is_reliable:
                    validated_count += 1
                    logger.info(f"   ✅ {feature.replace('_', ' ').title()}: {win_rate:.1%} win rate "
                               f"(n={len(outcomes)}, p={p_val:.4f})")
                else:
                    unreliable_count += 1
                    logger.warning(f"   ⚠️ {feature.replace('_', ' ').title()}: {win_rate:.1%} win rate "
                                 f"(n={len(outcomes)}, p={p_val:.4f}) - NOT RELIABLE")

        logger.info(f"\n   Summary: {validated_count} validated, {unreliable_count} unreliable features")

        # Pattern validation
        logger.info("\n🔬 PATTERN VALIDATION:")
        winning_validated = 0
        losing_validated = 0
        unvalidated = 0

        # Get top patterns by occurrence
        all_patterns = {**self.winning_patterns, **self.losing_patterns}
        top_patterns = sorted(all_patterns.items(), key=lambda x: x[1], reverse=True)[:15]

        for pattern, _ in top_patterns:
            wins = self.winning_patterns.get(pattern, 0)
            losses = self.losing_patterns.get(pattern, 0)
            total = wins + losses

            is_sig, p_val, interp = self._is_pattern_statistically_significant(pattern)

            if is_sig:
                win_rate = wins / total
                lower_ci, upper_ci = self.get_confidence_interval(pattern)

                if win_rate > 0.5:
                    winning_validated += 1
                    logger.info(f"   ✅ WIN: '{pattern}': {win_rate:.1%} "
                               f"(95% CI: [{lower_ci:.1%}, {upper_ci:.1%}], n={total}, p={p_val:.4f})")
                else:
                    losing_validated += 1
                    logger.warning(f"   ❌ LOSS: '{pattern}': {win_rate:.1%} "
                                 f"(95% CI: [{lower_ci:.1%}, {upper_ci:.1%}], n={total}, p={p_val:.4f})")
            else:
                unvalidated += 1
                win_rate = wins / total if total > 0 else 0
                status = "insufficient data" if total < self.min_sample_size else "not significant"
                logger.info(f"   ⚠️ UNVALIDATED: '{pattern}': {win_rate:.1%} "
                           f"(n={total}, p={p_val:.4f}) - {status}")

        logger.info(f"\n   Summary: {winning_validated} winning, {losing_validated} losing, "
                   f"{unvalidated} unvalidated patterns")

        logger.info("=" * 80 + "\n")

    def _get_top_features(self, limit: int = 3) -> str:
        """Get top performing features"""
        sorted_features = sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)
        return ", ".join([f"{feat.replace('_', ' ')}: {score:.2f}"
                         for feat, score in sorted_features[:limit] if score > 0])


    def get_statistics(self) -> Dict:
        """Get comprehensive ML statistics with statistical validation results"""
        # 📊 Statistical validation for patterns
        validated_winning_patterns = {}
        validated_losing_patterns = {}
        unvalidated_patterns = {}

        for pattern, wins in self.winning_patterns.items():
            losses = self.losing_patterns.get(pattern, 0)
            total = wins + losses
            is_sig, p_val, interp = self._is_pattern_statistically_significant(pattern)

            if is_sig and (wins / total) > 0.5:
                validated_winning_patterns[pattern] = {
                    'wins': wins,
                    'losses': losses,
                    'win_rate': wins / total,
                    'p_value': p_val,
                    'sample_size': total
                }
            elif is_sig and (wins / total) <= 0.5:
                validated_losing_patterns[pattern] = {
                    'wins': wins,
                    'losses': losses,
                    'win_rate': wins / total,
                    'p_value': p_val,
                    'sample_size': total
                }
            else:
                unvalidated_patterns[pattern] = {
                    'wins': wins,
                    'losses': losses,
                    'win_rate': wins / total if total > 0 else 0,
                    'p_value': p_val,
                    'sample_size': total,
                    'status': 'insufficient_data' if total < self.min_sample_size else 'not_significant'
                }

        # 📊 Statistical validation for features
        validated_features = {}
        unreliable_features = {}

        for feature, score in self.feature_scores.items():
            is_reliable, p_val, interp = self._is_feature_reliable(feature)
            outcomes = self.feature_outcomes.get(feature, [])

            if len(outcomes) >= self.min_sample_size:
                win_rate = sum(outcomes) / len(outcomes)
                if is_reliable:
                    validated_features[feature] = {
                        'score': score,
                        'win_rate': win_rate,
                        'p_value': p_val,
                        'sample_size': len(outcomes)
                    }
                else:
                    unreliable_features[feature] = {
                        'score': score,
                        'win_rate': win_rate,
                        'p_value': p_val,
                        'sample_size': len(outcomes)
                    }

        return {
            'total_trades_analyzed': self.total_trades_analyzed,
            'current_regime': self.current_regime,
            'min_confidence_threshold': self.min_confidence_threshold,
            'top_symbols': self._get_top_symbols(5),

            # 📊 STATISTICAL VALIDATION RESULTS
            'validated_winning_patterns': dict(sorted(validated_winning_patterns.items(),
                                                     key=lambda x: x[1]['win_rate'], reverse=True)[:10]),
            'validated_losing_patterns': dict(sorted(validated_losing_patterns.items(),
                                                    key=lambda x: x[1]['win_rate'])[:5]),
            'unvalidated_patterns': dict(sorted(unvalidated_patterns.items(),
                                               key=lambda x: x[1]['sample_size'], reverse=True)[:10]),

            'validated_features': dict(sorted(validated_features.items(),
                                             key=lambda x: x[1]['win_rate'], reverse=True)),
            'unreliable_features': dict(sorted(unreliable_features.items(),
                                              key=lambda x: x[1]['sample_size'], reverse=True)),

            # Legacy stats
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
