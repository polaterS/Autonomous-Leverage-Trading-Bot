"""
Machine Learning Pattern Learner
Analyzes historical trades to learn winning patterns and adapt strategy
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
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
        self.min_confidence_threshold: float = 0.60  # Global default (fallback)
        self.optimal_threshold_by_symbol: Dict[str, float] = {}  # 🎯 #4: Symbol-specific thresholds
        self.optimal_leverage_by_symbol: Dict[str, float] = {}

        # Feature importance (what matters most)
        # OLD: Simple EMA scores
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

        # 🎯 #1: BAYESIAN FEATURE IMPORTANCE (Beta Distribution)
        # Track wins/losses per feature using Bayesian inference
        self.bayesian_features: Dict[str, Dict[str, int]] = {
            feature: {'alpha': 1, 'beta': 1}  # Uniform prior (α=1, β=1)
            for feature in self.feature_scores.keys()
        }

        # Learning statistics
        self.total_trades_analyzed: int = 0
        self.learning_rate: float = 0.1  # How fast to adapt

        # 🎯 SNAPSHOT-BASED NUMERICAL LEARNING (NEW!)
        # Learn from actual indicator values, not just text patterns
        self.indicator_win_conditions: Dict[str, List[Tuple[Dict, bool, datetime]]] = defaultdict(list)
        # Format: {'RSI': [(value, won/lost, timestamp), ...]}

        self.learned_indicator_rules: Dict[str, Dict] = {
            'rsi': {'winning_ranges': [], 'losing_ranges': []},
            'volume_ratio': {'winning_ranges': [], 'losing_ranges': []},
            'support_distance': {'winning_ranges': [], 'losing_ranges': []},
            'resistance_distance': {'winning_ranges': [], 'losing_ranges': []},
            'macd_histogram': {'winning_ranges': [], 'losing_ranges': []},
            'bb_position': {'winning_categories': defaultdict(lambda: {'wins': 0, 'losses': 0}), 'losing_categories': defaultdict(lambda: {'wins': 0, 'losses': 0})},
            'order_book_imbalance': {'winning_ranges': [], 'losing_ranges': []},
            'funding_rate': {'winning_ranges': [], 'losing_ranges': []}
        }

        # 📊 STATISTICAL VALIDATION (NEW!)
        # Track win/loss outcomes per feature for t-test validation
        # 🎯 #8: Now with timestamps for time-decay weighting
        self.feature_outcomes: Dict[str, List[Tuple[int, datetime]]] = defaultdict(list)  # (outcome, timestamp)
        self.pattern_outcomes: Dict[str, List[Tuple[int, datetime]]] = defaultdict(list)  # (outcome, timestamp)
        self.min_sample_size: int = 30  # Minimum trades before trusting pattern
        self.time_decay_half_life_days: int = 30  # 🎯 #8: Pattern weight halves every 30 days

        # 🎯 #2: MULTI-MODEL AI ENSEMBLE TRACKING
        # Track each AI model's performance per symbol and market regime
        self.model_performance: Dict[str, Dict[str, Dict]] = {
            'qwen3-max': defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0}),
            'deepseek': defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0})
        }

        # Model performance by market regime (trending_up, trending_down, ranging, volatile)
        self.model_performance_by_regime: Dict[str, Dict[str, Dict]] = {
            'qwen3-max': defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0}),
            'deepseek': defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0})
        }

        # Real-time reliability scores (0-1) - updated after each trade
        self.model_reliability_scores: Dict[str, float] = {
            'qwen3-max': 0.50,  # Start neutral (50% = random)
            'deepseek': 0.50
        }

        # 🎯 #6: CROSS-SYMBOL CORRELATION MATRIX
        # Track rolling price history for correlation calculation
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))  # Last 100 prices
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}  # symbol1 -> symbol2 -> correlation
        self.last_correlation_update: Optional[datetime] = None
        self.correlation_update_interval_minutes: int = 60  # Update every hour
        self.max_correlation_threshold: float = 0.70  # Max allowed correlation for new positions

        logger.info("✅ ML Pattern Learner initialized with statistical validation + multi-model ensemble + correlation tracking")


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
            patterns = trade.get('ai_reasoning') or ''  # Handle None from old trades

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
            trade_timestamp = trade.get('exit_time', datetime.now())  # 🎯 #8: Get trade timestamp

            for keyword in pattern_keywords:
                if is_winner:
                    self.winning_patterns[keyword] += 1
                else:
                    self.losing_patterns[keyword] += 1

                # 📊 Track outcome for statistical validation + 🎯 #8: With timestamp
                self.pattern_outcomes[keyword].append((1 if is_winner else 0, trade_timestamp))

            # Update feature scores based on trade outcome (🎯 #8: Pass timestamp)
            if 'divergence' in patterns.lower():
                self._update_feature_score('rsi_divergence', is_winner, trade_timestamp)
            if 'volume' in patterns.lower():
                self._update_feature_score('volume_surge', is_winner, trade_timestamp)
            if 'macd' in patterns.lower():
                self._update_feature_score('macd_cross', is_winner, trade_timestamp)
            if 'support' in patterns.lower() or 'resistance' in patterns.lower():
                self._update_feature_score('support_resistance', is_winner, trade_timestamp)
            if 'order flow' in patterns.lower():
                self._update_feature_score('order_flow', is_winner, trade_timestamp)
            if 'btc' in patterns.lower():
                self._update_feature_score('btc_correlation', is_winner, trade_timestamp)
            if 'funding' in patterns.lower():
                self._update_feature_score('funding_rate', is_winner, trade_timestamp)
            if 'liquidation' in patterns.lower():
                self._update_feature_score('liquidation_clusters', is_winner, trade_timestamp)
            if 'timeframe' in patterns.lower():
                self._update_feature_score('multi_timeframe_align', is_winner, trade_timestamp)
            if 'smart money' in patterns.lower() or 'order block' in patterns.lower():
                self._update_feature_score('smart_money_concepts', is_winner, trade_timestamp)

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

            # 🎯 SNAPSHOT-BASED LEARNING: Learn from numerical indicators
            entry_snapshot = trade.get('entry_snapshot')
            exit_snapshot = trade.get('exit_snapshot')

            if entry_snapshot:
                await self._learn_from_snapshot(entry_snapshot, is_winner, trade_timestamp)

            if not initial_load:
                logger.info(f"📈 Learned from trade: {symbol} {'WIN' if is_winner else 'LOSS'} "
                           f"(Confidence: {ai_confidence:.1%}, Total analyzed: {self.total_trades_analyzed})")

        except Exception as e:
            logger.error(f"❌ Error analyzing trade: {e}")


    def _extract_pattern_keywords(self, reasoning: str) -> List[str]:
        """Extract key pattern indicators from AI reasoning AND ML-ONLY patterns"""
        keywords = []
        reasoning_lower = reasoning.lower()

        # Traditional AI pattern indicators
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

        # ML-ONLY mode patterns (from ai_engine.py _get_ml_only_prediction)
        ml_only_patterns = [
            'strong_bullish_trend', 'strong_bearish_trend',
            'bullish_trend', 'bearish_trend',
            'rsi_oversold', 'rsi_overbought',
            'rsi_weak', 'rsi_strong',
            'regime_low_vol', 'regime_normal_vol', 'regime_high_vol'
        ]

        # Extract both traditional and ML-ONLY patterns
        all_patterns = pattern_indicators + ml_only_patterns

        for pattern in all_patterns:
            if pattern in reasoning_lower:
                keywords.append(pattern)

        return keywords


    def _update_feature_score(self, feature: str, is_winner: bool, timestamp: datetime):
        """
        Update feature importance score using DUAL approach:
        1. OLD: EMA for backward compatibility
        2. 🎯 #1: BAYESIAN update (Beta distribution)
        3. 🎯 #8: Time-decay weighting

        Bayesian approach provides:
        - Probabilistic win rate with uncertainty quantification
        - Confidence intervals
        - Better handling of limited data
        """
        # OLD: EMA update (kept for compatibility)
        reward = 1.0 if is_winner else -1.0
        current_score = self.feature_scores.get(feature, 0.0)
        self.feature_scores[feature] = (1 - self.learning_rate) * current_score + self.learning_rate * reward

        # 🎯 #1: BAYESIAN UPDATE (Beta Distribution)
        if feature in self.bayesian_features:
            if is_winner:
                self.bayesian_features[feature]['alpha'] += 1  # Success
            else:
                self.bayesian_features[feature]['beta'] += 1   # Failure

        # 📊 Track outcome for statistical validation + 🎯 #8: With timestamp
        self.feature_outcomes[feature].append((1 if is_winner else 0, timestamp))


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


    def get_bayesian_win_rate(self, feature: str) -> float:
        """
        🎯 #1: Get Bayesian win rate estimate (posterior mean) for a feature.

        Mean of Beta(α, β) = α / (α + β)

        Returns:
            Win rate estimate between 0 and 1
        """
        if feature not in self.bayesian_features:
            return 0.5  # Unknown feature, assume 50/50

        alpha = self.bayesian_features[feature]['alpha']
        beta = self.bayesian_features[feature]['beta']

        return alpha / (alpha + beta)

    def get_bayesian_confidence_interval(self, feature: str, confidence: float = 0.95) -> Tuple[float, float]:
        """
        🎯 #1: Get Bayesian credible interval for feature win rate.

        Uses beta distribution quantiles for credible interval.

        Args:
            feature: Feature name
            confidence: Confidence level (default 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if feature not in self.bayesian_features:
            return (0.0, 1.0)

        alpha = self.bayesian_features[feature]['alpha']
        beta = self.bayesian_features[feature]['beta']

        # Beta distribution credible interval
        lower = stats.beta.ppf((1 - confidence) / 2, alpha, beta)
        upper = stats.beta.ppf((1 + confidence) / 2, alpha, beta)

        return (lower, upper)

    def get_bayesian_weight(self, feature: str) -> float:
        """
        🎯 #1: Get dynamic weight for feature using Thompson Sampling.

        Thompson Sampling: Sample from posterior Beta(α, β) to balance
        exploration (uncertain features) vs exploitation (proven features).

        Returns:
            Weight between 0 and 1 (sampled from posterior)
        """
        if feature not in self.bayesian_features:
            return 0.5

        alpha = self.bayesian_features[feature]['alpha']
        beta = self.bayesian_features[feature]['beta']

        # Thompson sampling: Draw random sample from Beta distribution
        sample = np.random.beta(alpha, beta)

        return sample

    def _calculate_time_decay_weight(self, timestamp: datetime) -> float:
        """
        🎯 #8: Calculate exponential time decay weight for historical patterns.

        Weight = exp(-days_ago / half_life)

        - Recent patterns (1 day ago): ~97% weight
        - 30 days ago: 50% weight (half-life)
        - 60 days ago: 25% weight
        - 90 days ago: 12.5% weight
        """
        days_ago = (datetime.now() - timestamp).days
        weight = np.exp(-days_ago / self.time_decay_half_life_days)
        return weight

    def _is_pattern_statistically_significant(self, pattern: str) -> Tuple[bool, float, str]:
        """
        Test if a pattern's win rate is statistically significant using t-test.

        🎯 #8: Now with TIME-DECAY weighting - recent patterns matter more!

        Returns:
            Tuple of (is_significant, p_value, interpretation)
        """
        outcome_tuples = self.pattern_outcomes.get(pattern, [])  # List of (outcome, timestamp)

        # Need minimum sample size
        if len(outcome_tuples) < self.min_sample_size:
            return (False, 1.0, f"Insufficient data ({len(outcome_tuples)}/{self.min_sample_size} trades)")

        # 🎯 #8: Apply time-decay weighting
        weighted_outcomes = []
        total_weight = 0.0

        for outcome, timestamp in outcome_tuples:
            weight = self._calculate_time_decay_weight(timestamp)
            weighted_outcomes.append(outcome * weight)
            total_weight += weight

        # Calculate weighted win rate
        weighted_win_rate = sum(weighted_outcomes) / total_weight if total_weight > 0 else 0.5

        # For t-test, use weighted outcomes (normalized)
        normalized_outcomes = [outcome * self._calculate_time_decay_weight(ts) for outcome, ts in outcome_tuples]

        # H0: Win rate = 0.5 (random chance)
        # H1: Win rate ≠ 0.5 (pattern has predictive power)
        t_stat, p_value = stats.ttest_1samp(normalized_outcomes, 0.5)

        # Significance level: 0.05 (95% confidence)
        is_significant = p_value < 0.05

        if is_significant:
            if weighted_win_rate > 0.5:
                interpretation = f"✅ Reliable winning pattern ({weighted_win_rate:.1%} weighted WR, p={p_value:.4f})"
            else:
                interpretation = f"❌ Reliable losing pattern ({weighted_win_rate:.1%} weighted WR, p={p_value:.4f})"
        else:
            interpretation = f"⚠️ Not statistically significant ({weighted_win_rate:.1%} weighted WR, p={p_value:.4f})"

        return (is_significant, p_value, interpretation)

    def _is_feature_reliable(self, feature: str) -> Tuple[bool, float, str]:
        """
        Test if a feature's impact is statistically significant.

        🎯 #8: Now with TIME-DECAY weighting

        Returns:
            Tuple of (is_reliable, p_value, interpretation)
        """
        outcome_tuples = self.feature_outcomes.get(feature, [])  # List of (outcome, timestamp)

        # Need minimum sample size
        if len(outcome_tuples) < self.min_sample_size:
            return (False, 1.0, f"Insufficient data ({len(outcome_tuples)}/{self.min_sample_size})")

        # 🎯 #8: Apply time-decay weighting
        weighted_outcomes = []
        total_weight = 0.0

        for outcome, timestamp in outcome_tuples:
            weight = self._calculate_time_decay_weight(timestamp)
            weighted_outcomes.append(outcome * weight)
            total_weight += weight

        # Calculate weighted win rate
        weighted_win_rate = sum(weighted_outcomes) / total_weight if total_weight > 0 else 0.5

        # For t-test, use weighted outcomes
        normalized_outcomes = [outcome * self._calculate_time_decay_weight(ts) for outcome, ts in outcome_tuples]

        # T-test: Is win rate significantly different from 50% (random)?
        t_stat, p_value = stats.ttest_1samp(normalized_outcomes, 0.5)

        is_reliable = p_value < 0.05 and weighted_win_rate > 0.55  # At least 55% win rate + significant

        if is_reliable:
            interpretation = f"✅ Reliable feature ({weighted_win_rate:.1%} weighted WR, p={p_value:.4f})"
        elif p_value < 0.05 and weighted_win_rate < 0.45:
            interpretation = f"❌ Negative feature ({weighted_win_rate:.1%} weighted WR, p={p_value:.4f} - AVOID)"
        else:
            interpretation = f"⚠️ Unreliable feature ({weighted_win_rate:.1%} weighted WR, p={p_value:.4f})"

        return (is_reliable, p_value, interpretation)

    def get_confidence_interval(self, pattern: str, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for pattern win rate.

        🎯 #8: Now with TIME-DECAY weighting

        Returns:
            Tuple of (lower_bound, upper_bound) for win rate
        """
        outcome_tuples = self.pattern_outcomes.get(pattern, [])  # List of (outcome, timestamp)

        if len(outcome_tuples) < 10:
            return (0.0, 1.0)  # Too few samples, very wide interval

        # 🎯 #8: Apply time-decay weighting
        weighted_outcomes = []
        total_weight = 0.0

        for outcome, timestamp in outcome_tuples:
            weight = self._calculate_time_decay_weight(timestamp)
            weighted_outcomes.append(outcome * weight)
            total_weight += weight

        # Calculate weighted win rate
        weighted_win_rate = sum(weighted_outcomes) / total_weight if total_weight > 0 else 0.5

        # Effective sample size (accounting for weights)
        effective_n = total_weight

        # Standard error (using weighted variance)
        se = np.sqrt(weighted_win_rate * (1 - weighted_win_rate) / effective_n)

        # Z-score for confidence level (95% → 1.96)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        # Confidence interval
        lower = max(0.0, weighted_win_rate - z_score * se)
        upper = min(1.0, weighted_win_rate + z_score * se)

        return (lower, upper)

    def _optimize_confidence_threshold(self):
        """
        🎯 #4: Find optimal confidence threshold GLOBALLY and PER-SYMBOL

        Symbol-specific thresholds allow:
        - BTC/ETH: Lower threshold (more reliable)
        - Volatile altcoins: Higher threshold (more caution)
        - Adaptive learning per symbol characteristics
        """
        if not self.ai_accuracy_by_confidence:
            return

        # GLOBAL THRESHOLD: Find confidence level with highest accuracy and >20 samples
        best_threshold = 0.60
        best_accuracy = 0.0

        for confidence, stats in self.ai_accuracy_by_confidence.items():
            if stats['predictions'] >= 20 and stats['accuracy'] > best_accuracy:
                best_accuracy = stats['accuracy']
                best_threshold = confidence

        # Be conservative: only lower threshold if accuracy is very high
        if best_accuracy >= 0.65 and best_threshold < self.min_confidence_threshold:
            self.min_confidence_threshold = best_threshold

        logger.info(f"🎯 Global optimal threshold: {self.min_confidence_threshold:.1%} "
                   f"(Best observed accuracy: {best_accuracy:.1%})")

        # 🎯 #4: SYMBOL-SPECIFIC THRESHOLDS
        # Find optimal threshold per symbol based on historical win rate
        for symbol, perf in self.symbol_performance.items():
            total_trades = perf['wins'] + perf['losses']

            if total_trades < 20:  # Need at least 20 trades per symbol
                continue

            win_rate = perf['wins'] / total_trades
            avg_confidence = perf.get('avg_confidence', 0.0)

            # If symbol consistently wins at lower confidence, reduce threshold
            # If symbol needs higher confidence to win, increase threshold
            if win_rate >= 0.70 and avg_confidence < 0.70:
                # High win rate at low confidence → can trade lower
                optimal = max(0.55, avg_confidence - 0.05)
            elif win_rate >= 0.60 and avg_confidence < 0.75:
                # Good win rate → slight reduction
                optimal = max(0.60, avg_confidence - 0.03)
            elif win_rate < 0.45:
                # Poor win rate → need higher confidence
                optimal = min(0.80, avg_confidence + 0.10)
            else:
                # Average → use global or slightly adjusted
                optimal = self.min_confidence_threshold

            self.optimal_threshold_by_symbol[symbol] = optimal
            logger.info(f"   📊 {symbol}: optimal threshold {optimal:.1%} "
                       f"(WR: {win_rate:.1%}, avg conf: {avg_confidence:.1%})")

    def get_confidence_threshold(self, symbol: str) -> float:
        """
        🎯 #4: Get confidence threshold for a specific symbol
        Falls back to global threshold if symbol-specific not available
        """
        return self.optimal_threshold_by_symbol.get(symbol, self.min_confidence_threshold)


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

    # 🎯 #2: MULTI-MODEL ENSEMBLE METHODS

    def update_model_performance(
        self,
        model_name: str,
        symbol: str,
        market_regime: str,
        was_correct: bool
    ):
        """
        🎯 #2: Update model performance after trade closes.

        Args:
            model_name: 'qwen3-max' or 'deepseek'
            symbol: Trading symbol
            market_regime: 'trending_up', 'trending_down', 'ranging', 'volatile'
            was_correct: True if model prediction was correct
        """
        if model_name not in self.model_performance:
            logger.warning(f"Unknown model: {model_name}")
            return

        # Update per-symbol stats
        perf = self.model_performance[model_name][symbol]
        perf['total'] += 1
        if was_correct:
            perf['wins'] += 1
        else:
            perf['losses'] += 1

        # Update per-regime stats
        regime_perf = self.model_performance_by_regime[model_name][market_regime]
        regime_perf['total'] += 1
        if was_correct:
            regime_perf['wins'] += 1
        else:
            regime_perf['losses'] += 1

        # Update global reliability score (EMA with α=0.1)
        alpha = 0.1
        observation = 1.0 if was_correct else 0.0
        self.model_reliability_scores[model_name] = (
            alpha * observation + (1 - alpha) * self.model_reliability_scores[model_name]
        )

        logger.info(
            f"🎯 MODEL UPDATE: {model_name} on {symbol} ({market_regime}) - "
            f"{'✅ CORRECT' if was_correct else '❌ WRONG'} | "
            f"New reliability: {self.model_reliability_scores[model_name]:.1%}"
        )

    def get_model_weight(self, model_name: str, symbol: str, market_regime: str) -> float:
        """
        🎯 #2: Calculate dynamic weight for a model based on historical performance.

        Weighting factors:
        1. Symbol-specific win rate (40%)
        2. Regime-specific win rate (40%)
        3. Global reliability score (20%)

        Returns:
            Weight between 0.0 and 1.0
        """
        if model_name not in self.model_performance:
            return 0.5  # Neutral for unknown models

        # 1. Symbol-specific performance
        symbol_perf = self.model_performance[model_name][symbol]
        if symbol_perf['total'] >= 5:
            symbol_wr = symbol_perf['wins'] / symbol_perf['total']
        else:
            symbol_wr = 0.5  # Not enough data, use neutral

        # 2. Regime-specific performance
        regime_perf = self.model_performance_by_regime[model_name][market_regime]
        if regime_perf['total'] >= 5:
            regime_wr = regime_perf['wins'] / regime_perf['total']
        else:
            regime_wr = 0.5  # Not enough data, use neutral

        # 3. Global reliability
        global_reliability = self.model_reliability_scores[model_name]

        # Weighted combination
        weight = 0.40 * symbol_wr + 0.40 * regime_wr + 0.20 * global_reliability

        logger.debug(
            f"🎯 MODEL WEIGHT: {model_name} | Symbol WR: {symbol_wr:.1%} | "
            f"Regime WR: {regime_wr:.1%} | Global: {global_reliability:.1%} | "
            f"Final Weight: {weight:.1%}"
        )

        return weight

    def calculate_weighted_ensemble(
        self,
        analyses: List[Dict],
        symbol: str,
        market_regime: str,
        market_data: Dict[str, Any] = None
    ) -> Dict:
        """
        🎯 #2: Calculate weighted ensemble consensus from multiple AI models.

        Instead of simple 2/3 voting, uses performance-based weighted voting.

        Args:
            analyses: List of AI analyses from different models
            symbol: Trading symbol
            market_regime: Current market regime

        Returns:
            Weighted consensus with adjusted confidence
        """
        if not analyses:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'weighted_consensus': False,
                'reason': 'No model analyses available'
            }

        # Single model - return as-is (already ML-enhanced)
        if len(analyses) == 1:
            result = analyses[0].copy()
            result['weighted_consensus'] = True
            result['ensemble_method'] = 'single_model'

            # 🔥 CRITICAL FIX: Re-evaluate action after ML enhancement
            # If ML boosted confidence ≥ 50% but action is still "hold", fix it!
            final_confidence = result.get('confidence', 0)
            current_action = result.get('action', 'hold')
            original_side = result.get('side')

            logger.info(f"   🔍 ML DEBUG (ALWAYS): confidence={final_confidence:.1%}, action={current_action}, side={original_side}")

            if final_confidence >= 0.50 and current_action == 'hold':
                # ML enhanced confidence above threshold - should trade!
                logger.info(f"   🔍 ML DEBUG (IF TRUE): market_data_exists={market_data is not None}")

                # Determine direction from indicators or original AI reasoning
                if original_side:
                    # AI suggested a side but said "hold" - now trade it!
                    if original_side == 'LONG':
                        result['action'] = 'buy'
                    elif original_side == 'SHORT':
                        result['action'] = 'sell'
                    logger.info(
                        f"   🔧 ML FIX: Changed action from 'hold' to '{result['action']}' "
                        f"(confidence {final_confidence:.1%} ≥ 50% threshold, AI side hint)"
                    )
                else:
                    # No side hint - use market_data to determine direction
                    if market_data:
                        # Extract indicator values directly from market data
                        rsi_15m = market_data.get('rsi_15m', 50)
                        rsi_1h = market_data.get('rsi_1h', 50)
                        macd_hist = market_data.get('macd_hist_15m', 0)
                        order_flow = market_data.get('order_flow_imbalance', 0)

                        # Determine bullish/bearish based on indicators
                        rsi_bullish = rsi_15m < 40 or rsi_1h < 40  # Oversold
                        rsi_bearish = rsi_15m > 60 or rsi_1h > 60  # Overbought
                        macd_bullish = macd_hist > 0  # Positive MACD
                        macd_bearish = macd_hist < 0  # Negative MACD
                        flow_bullish = order_flow > 500  # Strong buy pressure
                        flow_bearish = order_flow < -500  # Strong sell pressure

                        # Count signals
                        bullish_count = sum([rsi_bullish, macd_bullish, flow_bullish])
                        bearish_count = sum([rsi_bearish, macd_bearish, flow_bearish])

                        if bullish_count > bearish_count:
                            result['action'] = 'buy'
                            result['side'] = 'LONG'
                            logger.info(
                                f"   🔧 ML FIX: Changed action 'hold' → 'buy' LONG "
                                f"(confidence {final_confidence:.1%}, {bullish_count} bullish signals: "
                                f"RSI={rsi_15m:.0f}, MACD={macd_hist:.4f}, Flow={order_flow:.0f})"
                            )
                        elif bearish_count > bullish_count:
                            result['action'] = 'sell'
                            result['side'] = 'SHORT'
                            logger.info(
                                f"   🔧 ML FIX: Changed action 'hold' → 'sell' SHORT "
                                f"(confidence {final_confidence:.1%}, {bearish_count} bearish signals: "
                                f"RSI={rsi_15m:.0f}, MACD={macd_hist:.4f}, Flow={order_flow:.0f})"
                            )
                        else:
                            # Truly uncertain - keep as hold
                            logger.warning(
                                f"   ⚠️ ML: Confidence {final_confidence:.1%} ≥ 50% but no clear direction "
                                f"(bullish={bullish_count}, bearish={bearish_count}). "
                                f"RSI={rsi_15m:.0f}, MACD={macd_hist:.4f}, Flow={order_flow:.0f}"
                            )
                    else:
                        # No market_data available - fallback to reasoning parse
                        reasoning = result.get('reasoning', '').lower()

                        # Parse indicators from reasoning
                        rsi_bullish = 'rsi' in reasoning and any(word in reasoning for word in ['bullish', 'oversold', 'rising'])
                        rsi_bearish = 'rsi' in reasoning and any(word in reasoning for word in ['bearish', 'overbought', 'falling'])
                        macd_bullish = 'macd' in reasoning and any(word in reasoning for word in ['bullish', 'positive', 'rising'])
                        macd_bearish = 'macd' in reasoning and any(word in reasoning for word in ['bearish', 'negative', 'falling'])
                        flow_bullish = any(word in reasoning for word in ['buying pressure', 'positive flow', 'accumulation'])
                        flow_bearish = any(word in reasoning for word in ['selling pressure', 'negative flow', 'distribution'])

                        # Count bullish vs bearish signals
                        bullish_count = sum([rsi_bullish, macd_bullish, flow_bullish])
                        bearish_count = sum([rsi_bearish, macd_bearish, flow_bearish])

                        if bullish_count > bearish_count:
                            result['action'] = 'buy'
                            result['side'] = 'LONG'
                            logger.info(
                                f"   🔧 ML FIX: Changed action 'hold' → 'buy' LONG "
                                f"(confidence {final_confidence:.1%}, {bullish_count} bullish signals from reasoning)"
                            )
                        elif bearish_count > bullish_count:
                            result['action'] = 'sell'
                            result['side'] = 'SHORT'
                            logger.info(
                                f"   🔧 ML FIX: Changed action 'hold' → 'sell' SHORT "
                                f"(confidence {final_confidence:.1%}, {bearish_count} bearish signals from reasoning)"
                            )
                        else:
                            logger.warning(
                                f"   ⚠️ ML: Confidence {final_confidence:.1%} ≥ 50% but no clear direction. "
                                f"No market_data and reasoning parse failed."
                            )

            return result

        # Multiple models - calculate weighted consensus
        total_weight = 0.0
        weighted_confidence = 0.0
        weighted_leverage = 0.0
        weighted_stop_loss = 0.0
        action_votes = defaultdict(float)  # action -> weight
        side_votes = defaultdict(float)  # side -> weight
        model_weights_used = {}

        for analysis in analyses:
            model_name = analysis.get('model_name', 'unknown')
            weight = self.get_model_weight(model_name, symbol, market_regime)

            # Accumulate weighted values
            total_weight += weight
            weighted_confidence += analysis.get('confidence', 0.0) * weight
            weighted_leverage += analysis.get('suggested_leverage', 3) * weight
            weighted_stop_loss += analysis.get('stop_loss_percent', 7.0) * weight

            # Vote on action and side
            action = analysis.get('action', 'hold')
            side = analysis.get('side')
            action_votes[action] += weight
            if side:
                side_votes[side] += weight

            model_weights_used[model_name] = weight

            logger.info(
                f"   🤖 {model_name.upper()}: {action} {side or ''} "
                f"(conf={analysis.get('confidence', 0):.1%}, weight={weight:.1%})"
            )

        # Normalize weighted values
        if total_weight > 0:
            weighted_confidence /= total_weight
            weighted_leverage /= total_weight
            weighted_stop_loss /= total_weight
        else:
            # Fallback if all weights are zero
            weighted_confidence = sum(a.get('confidence', 0) for a in analyses) / len(analyses)
            weighted_leverage = 3
            weighted_stop_loss = 10.0  # FIXED: Always 10% for consistent risk management

        # Determine winning action and side
        winning_action = max(action_votes.items(), key=lambda x: x[1])[0] if action_votes else 'hold'
        winning_side = max(side_votes.items(), key=lambda x: x[1])[0] if side_votes else None

        # Check if consensus is strong (winning action has >60% of total weight)
        winning_action_weight = action_votes.get(winning_action, 0)
        consensus_strength = winning_action_weight / total_weight if total_weight > 0 else 0

        # Build consensus result
        consensus = {
            'action': winning_action,
            'side': winning_side,
            'confidence': round(weighted_confidence, 2),
            'suggested_leverage': max(2, min(5, int(round(weighted_leverage)))),
            'stop_loss_percent': max(5.0, min(10.0, round(weighted_stop_loss, 1))),
            'weighted_consensus': consensus_strength > 0.60,
            'consensus_strength': round(consensus_strength, 2),
            'ensemble_method': 'weighted_voting',
            'model_weights': model_weights_used,
            'reasoning': f"Weighted ensemble from {len(analyses)} models (strength: {consensus_strength:.1%})",
            'individual_analyses': analyses
        }

        logger.info(
            f"🎯 ENSEMBLE RESULT: {winning_action} {winning_side or ''} "
            f"(confidence: {weighted_confidence:.1%}, strength: {consensus_strength:.1%})"
        )

        return consensus

    # 🎯 #6: CROSS-SYMBOL CORRELATION METHODS

    def update_price_for_correlation(self, symbol: str, price: float):
        """
        🎯 #6: Update price history for correlation tracking.

        Call this periodically (e.g., every 5 minutes) to build price history.

        Args:
            symbol: Trading symbol
            price: Current price
        """
        self.price_history[symbol].append(price)

        # Auto-update correlation matrix if enough time has passed
        if self.last_correlation_update is None:
            self.last_correlation_update = datetime.now()
        elif (datetime.now() - self.last_correlation_update).seconds >= self.correlation_update_interval_minutes * 60:
            self.calculate_correlation_matrix()
            self.last_correlation_update = datetime.now()

    def calculate_correlation_matrix(self):
        """
        🎯 #6: Calculate pairwise correlations between all tracked symbols.

        Uses Pearson correlation on price returns (not raw prices).
        Requires at least 30 data points per symbol.
        """
        # Get symbols with enough data
        valid_symbols = [s for s, prices in self.price_history.items() if len(prices) >= 30]

        if len(valid_symbols) < 2:
            logger.debug("Not enough symbols for correlation matrix (need at least 2 with 30+ prices)")
            return

        # Calculate returns for each symbol
        returns = {}
        for symbol in valid_symbols:
            prices = np.array(list(self.price_history[symbol]))
            # Calculate percentage returns
            price_returns = np.diff(prices) / prices[:-1]
            returns[symbol] = price_returns

        # Calculate pairwise correlations
        new_matrix = defaultdict(dict)

        for i, sym1 in enumerate(valid_symbols):
            for sym2 in valid_symbols[i+1:]:
                # Find common length (in case they differ)
                min_len = min(len(returns[sym1]), len(returns[sym2]))
                r1 = returns[sym1][-min_len:]
                r2 = returns[sym2][-min_len:]

                # Pearson correlation
                if len(r1) >= 30:  # Need enough samples
                    correlation = np.corrcoef(r1, r2)[0, 1]

                    # Store both directions
                    new_matrix[sym1][sym2] = correlation
                    new_matrix[sym2][sym1] = correlation

        self.correlation_matrix = dict(new_matrix)

        # Log high correlations
        high_corr_pairs = []
        for sym1, correlations in self.correlation_matrix.items():
            for sym2, corr in correlations.items():
                if abs(corr) > 0.80 and sym1 < sym2:  # Avoid duplicates
                    high_corr_pairs.append((sym1, sym2, corr))

        if high_corr_pairs:
            logger.info(f"🔗 High correlations detected:")
            for sym1, sym2, corr in high_corr_pairs[:5]:  # Show top 5
                logger.info(f"   {sym1} ↔ {sym2}: {corr:+.2f}")

    def check_correlation_risk(self, new_symbol: str, active_positions: List[Dict]) -> Dict[str, Any]:
        """
        🎯 #6: Check if opening a position in new_symbol would create over-correlation.

        Prevents portfolio concentration in highly correlated assets.

        Args:
            new_symbol: Symbol we want to trade
            active_positions: List of currently open positions

        Returns:
            Dict with 'safe': bool, 'reason': str, 'avg_correlation': float
        """
        if not active_positions:
            # No existing positions, safe to trade
            return {
                'safe': True,
                'reason': 'No existing positions',
                'avg_correlation': 0.0,
                'correlations': {}
            }

        if new_symbol not in self.correlation_matrix:
            # Not enough data to calculate correlation, allow but warn
            logger.warning(f"⚠️ No correlation data for {new_symbol}, allowing trade")
            return {
                'safe': True,
                'reason': f'Insufficient correlation data for {new_symbol}',
                'avg_correlation': 0.0,
                'correlations': {}
            }

        # Calculate correlation with each active position
        correlations = {}
        total_corr = 0.0
        count = 0

        for pos in active_positions:
            pos_symbol = pos['symbol']

            if pos_symbol in self.correlation_matrix.get(new_symbol, {}):
                corr = self.correlation_matrix[new_symbol][pos_symbol]
                correlations[pos_symbol] = corr
                total_corr += abs(corr)  # Use absolute correlation
                count += 1

                logger.debug(f"🔗 Correlation {new_symbol} ↔ {pos_symbol}: {corr:+.2f}")

        if count == 0:
            # No correlation data with existing positions
            return {
                'safe': True,
                'reason': 'No correlation data with existing positions',
                'avg_correlation': 0.0,
                'correlations': correlations
            }

        # Calculate average absolute correlation
        avg_correlation = total_corr / count

        # Check if over-correlated
        if avg_correlation > self.max_correlation_threshold:
            return {
                'safe': False,
                'reason': f'Average correlation {avg_correlation:.1%} > threshold {self.max_correlation_threshold:.1%}',
                'avg_correlation': avg_correlation,
                'correlations': correlations,
                'most_correlated': max(correlations.items(), key=lambda x: abs(x[1])) if correlations else None
            }

        # Safe to trade
        return {
            'safe': True,
            'reason': f'Average correlation {avg_correlation:.1%} within safe limits',
            'avg_correlation': avg_correlation,
            'correlations': correlations
        }


    async def analyze_opportunity(
        self,
        symbol: str,
        ai_analysis: Dict,
        market_data: Dict,
        market_sentiment: str = "NEUTRAL"  # 🎯 #7: Market sentiment parameter
    ) -> Dict:
        """
        Enhance AI analysis with ML insights

        Args:
            symbol: Trading symbol
            ai_analysis: AI analysis dict
            market_data: Market data dict
            market_sentiment: Market breadth sentiment (BULLISH_STRONG, BULLISH, NEUTRAL, BEARISH, BEARISH_STRONG)

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

            # 🔍 DEPLOYMENT VERIFICATION LOG
            logger.info(f"   🚀 ML ANALYZE_OPPORTUNITY START: symbol={symbol}, base_conf={base_confidence:.1%}, version=ATTEMPT#10")

            # 1. Check symbol historical performance
            perf = self.symbol_performance.get(symbol)
            symbol_modifier = 0.0

            if perf and (perf['wins'] + perf['losses']) >= 5:
                win_rate = perf['wins'] / (perf['wins'] + perf['losses'])
                if win_rate >= 0.60:
                    symbol_modifier = +0.05  # Boost good performers
                elif win_rate <= 0.35:
                    symbol_modifier = -0.10  # Penalize bad performers
            elif self.total_trades_analyzed < 10:
                # 🎯 COLD START BOOST: Give small optimistic bias for first trades
                # This helps bot take initial positions to start learning
                symbol_modifier = +0.03  # Small boost to encourage first trades
                logger.debug(f"🚀 Cold start: giving +3% boost (total trades: {self.total_trades_analyzed})")

            # 2. Check AI accuracy at this confidence level
            confidence_bucket = round(base_confidence * 20) / 20
            acc_stats = self.ai_accuracy_by_confidence.get(confidence_bucket)
            accuracy_modifier = 0.0

            if acc_stats and acc_stats['predictions'] >= 10:
                if acc_stats['accuracy'] >= 0.65:
                    accuracy_modifier = +0.03
                elif acc_stats['accuracy'] <= 0.45:
                    accuracy_modifier = -0.07
            elif self.total_trades_analyzed < 10:
                # 🎯 COLD START: Assume neutral-to-positive accuracy initially
                accuracy_modifier = +0.02  # Small boost for cold start
                logger.debug(f"🚀 Cold start: accuracy boost +2%")

            # 3. 🎯 #1: BAYESIAN FEATURE IMPORTANCE BOOST (Most Advanced!)
            reasoning = ai_analysis.get('reasoning', '')
            feature_boost = 0.0
            bayesian_features_used = []
            weak_features = []

            # Check if trade uses high-value features using Bayesian inference
            for feature in self.feature_scores.keys():
                feature_key = feature.replace('_', ' ')
                if feature_key in reasoning.lower():
                    # Get Bayesian win rate and confidence interval
                    win_rate = self.get_bayesian_win_rate(feature)
                    lower_ci, upper_ci = self.get_bayesian_confidence_interval(feature)

                    # Total observations
                    alpha = self.bayesian_features[feature]['alpha']
                    beta = self.bayesian_features[feature]['beta']
                    total_obs = alpha + beta - 2  # Subtract prior (α=1, β=1)

                    # Only use feature if:
                    # 1. Lower CI > 55% (confident it's better than random)
                    # 2. OR if new feature (< 10 obs), give it exploration bonus
                    if lower_ci > 0.55 and total_obs >= 10:
                        # PROVEN WINNER: Use Bayesian weight
                        # Boost proportional to win rate and confidence
                        confidence_width = upper_ci - lower_ci
                        certainty = 1.0 - confidence_width  # Narrower = more certain

                        boost = 0.10 * (win_rate - 0.5) * certainty  # Max ~+5% per feature
                        feature_boost += boost

                        bayesian_features_used.append((
                            feature, win_rate, (lower_ci, upper_ci), certainty, boost
                        ))

                        logger.info(
                            f"   ✅ BAYESIAN: '{feature}' WR={win_rate:.1%} "
                            f"(95% CI: [{lower_ci:.1%}, {upper_ci:.1%}]), "
                            f"certainty={certainty:.1%}, boost={boost:+.1%}"
                        )

                    elif total_obs < 10:
                        # NEW FEATURE: Give exploration bonus (Thompson Sampling)
                        # 🎯 AGGRESSIVE initial exploration (no trade history yet)
                        exploration_bonus = 0.05 if total_obs == 0 else 0.03  # Higher bonus for completely new features
                        feature_boost += exploration_bonus
                        logger.info(f"   🔍 EXPLORATION: '{feature}' (only {total_obs} obs, +{exploration_bonus:.1%} bonus)")

                    else:
                        # WEAK FEATURE: Lower CI < 55% or win rate < 55%
                        weak_features.append((feature, win_rate, (lower_ci, upper_ci)))
                        logger.warning(
                            f"   ⚠️ WEAK: '{feature}' WR={win_rate:.1%} "
                            f"(95% CI: [{lower_ci:.1%}, {upper_ci:.1%}]) - IGNORED"
                        )

            feature_boost = min(feature_boost, 0.15)  # Cap at +15% (higher than before!)

            if bayesian_features_used:
                logger.info(f"   📊 Applied {len(bayesian_features_used)} Bayesian-validated features")
            if weak_features:
                logger.info(f"   ⚠️ Ignored {len(weak_features)} weak features")

            # 4. Market regime adjustment + 🎯 #7: Sentiment-Aware ML
            regime_modifier = 0.0
            sentiment_modifier = 0.0
            regime_warning = ""

            # Existing regime logic
            if self.current_regime == "HIGH_VOLATILITY_UNFAVORABLE":
                regime_modifier = -0.10
                regime_warning = "⚠️ High volatility regime detected - reduced confidence"
            elif self.current_regime in ["BULL_TREND", "BEAR_TREND"]:
                side = ai_analysis.get('side', 'LONG')
                if (self.current_regime == "BULL_TREND" and side == "LONG") or \
                   (self.current_regime == "BEAR_TREND" and side == "SHORT"):
                    regime_modifier = +0.05
                    regime_warning = f"✅ Trade aligned with {self.current_regime}"

            # 🎯 #7: SENTIMENT-AWARE ML ENHANCEMENT
            # Trading with market sentiment = boost, against = penalty
            side = ai_analysis.get('side')
            if side:
                if market_sentiment == "BULLISH_STRONG" and side == "LONG":
                    sentiment_modifier = +0.08  # Strong tailwind
                    regime_warning += " | 🚀 STRONG BULLISH market momentum"
                elif market_sentiment == "BULLISH" and side == "LONG":
                    sentiment_modifier = +0.05  # Good tailwind
                    regime_warning += " | 📈 Bullish market momentum"
                elif market_sentiment == "BEARISH_STRONG" and side == "SHORT":
                    sentiment_modifier = +0.08  # Strong tailwind
                    regime_warning += " | 📉 STRONG BEARISH market momentum"
                elif market_sentiment == "BEARISH" and side == "SHORT":
                    sentiment_modifier = +0.05  # Good tailwind
                    regime_warning += " | 📊 Bearish market momentum"
                elif market_sentiment == "BULLISH_STRONG" and side == "SHORT":
                    sentiment_modifier = -0.15  # Counter-trend (very risky!)
                    regime_warning += " | ⚠️ WARNING: SHORT against STRONG BULLISH market!"
                elif market_sentiment == "BULLISH" and side == "SHORT":
                    sentiment_modifier = -0.08  # Counter-trend (risky)
                    regime_warning += " | ⚠️ SHORT against bullish market"
                elif market_sentiment == "BEARISH_STRONG" and side == "LONG":
                    sentiment_modifier = -0.15  # Counter-trend (very risky!)
                    regime_warning += " | ⚠️ WARNING: LONG against STRONG BEARISH market!"
                elif market_sentiment == "BEARISH" and side == "LONG":
                    sentiment_modifier = -0.08  # Counter-trend (risky)
                    regime_warning += " | ⚠️ LONG against bearish market"
                elif market_sentiment == "NEUTRAL":
                    sentiment_modifier = 0.0  # No adjustment
                    regime_warning += " | 😐 Neutral market sentiment"

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

            # 6. Calculate final ML-adjusted confidence (🎯 #7: Added sentiment_modifier)
            ml_adjustment = (
                symbol_modifier +
                accuracy_modifier +
                feature_boost +
                regime_modifier +
                sentiment_modifier +  # 🎯 #7: Sentiment-aware adjustment
                pattern_boost
            )

            adjusted_confidence = base_confidence + ml_adjustment
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))  # Clamp to [0, 1]

            # 🔧 CRITICAL FIX: Re-evaluate action if confidence crosses threshold
            # If DeepSeek returned "hold" but ML boosted confidence ≥ 50%, convert to buy/sell
            current_action = ai_analysis.get('action', 'hold')
            original_side = ai_analysis.get('side')

            logger.info(f"   🔍 ML FIX CHECK: confidence={adjusted_confidence:.1%}, action={current_action}, side={original_side}")

            if adjusted_confidence >= 0.50 and current_action == 'hold':
                logger.info(f"   🔧 ML FIX TRIGGERED: Converting 'hold' to tradeable action (confidence crossed 50% threshold)")

                # Use market data indicators to determine direction
                if market_data:
                    rsi_15m = market_data.get('rsi_15m', 50)
                    rsi_1h = market_data.get('rsi_1h', 50)
                    macd_hist = market_data.get('macd_hist_15m', 0)
                    order_flow = market_data.get('order_flow_imbalance', 0)

                    # Count bullish vs bearish signals
                    rsi_bullish = rsi_15m < 40 or rsi_1h < 40
                    rsi_bearish = rsi_15m > 60 or rsi_1h > 60
                    macd_bullish = macd_hist > 0
                    macd_bearish = macd_hist < 0
                    flow_bullish = order_flow > 500
                    flow_bearish = order_flow < -500

                    bullish_count = sum([rsi_bullish, macd_bullish, flow_bullish])
                    bearish_count = sum([rsi_bearish, macd_bearish, flow_bearish])

                    logger.info(f"   📊 Indicator signals: Bullish={bullish_count}, Bearish={bearish_count}")
                    logger.info(f"      RSI(15m)={rsi_15m:.1f}, RSI(1h)={rsi_1h:.1f}, MACD={macd_hist:.2f}, Flow={order_flow:.0f}")

                    # 🔥 AGGRESSIVE MODE: Dynamic leverage 10-20x with ULTRA TIGHT stops
                    # Tighter stops (3-5%) = liquidation stays far enough for 10-20x leverage
                    if adjusted_confidence >= 0.80:
                        leverage = 20  # Ultra high confidence: 20x
                        stop_loss = 3.0  # ULTRA TIGHT for 20x
                    elif adjusted_confidence >= 0.70:
                        leverage = 17  # High confidence: 17x
                        stop_loss = 3.5  # ULTRA TIGHT for 17x
                    elif adjusted_confidence >= 0.60:
                        leverage = 14  # Good confidence: 14x
                        stop_loss = 4.0  # TIGHT for 14x
                    else:  # 50-60%
                        leverage = 10  # Minimum: 10x
                        stop_loss = 5.0  # TIGHT for 10x

                    if bullish_count > bearish_count:
                        ai_analysis['action'] = 'buy'
                        ai_analysis['side'] = 'LONG'
                        ai_analysis['stop_loss_percent'] = stop_loss
                        ai_analysis['suggested_leverage'] = leverage
                        logger.info(f"   ✅ ML FIX: Converted to LONG (bullish signals dominate) | SL={stop_loss}% | {leverage}x")
                    elif bearish_count > bullish_count:
                        ai_analysis['action'] = 'sell'
                        ai_analysis['side'] = 'SHORT'
                        ai_analysis['stop_loss_percent'] = stop_loss
                        ai_analysis['suggested_leverage'] = leverage
                        logger.info(f"   ✅ ML FIX: Converted to SHORT (bearish signals dominate) | SL={stop_loss}% | {leverage}x")
                    else:
                        # Tie - use RSI as tiebreaker
                        avg_rsi = (rsi_15m + rsi_1h) / 2
                        if avg_rsi < 50:
                            ai_analysis['action'] = 'buy'
                            ai_analysis['side'] = 'LONG'
                            ai_analysis['stop_loss_percent'] = stop_loss
                            ai_analysis['suggested_leverage'] = leverage
                            logger.info(f"   ✅ ML FIX: Converted to LONG (RSI tiebreaker: {avg_rsi:.1f}) | SL={stop_loss}% | {leverage}x")
                        else:
                            ai_analysis['action'] = 'sell'
                            ai_analysis['side'] = 'SHORT'
                            ai_analysis['stop_loss_percent'] = stop_loss
                            ai_analysis['suggested_leverage'] = leverage
                            logger.info(f"   ✅ ML FIX: Converted to SHORT (RSI tiebreaker: {avg_rsi:.1f}) | SL={stop_loss}% | {leverage}x")
                else:
                    logger.warning(f"   ⚠️ ML FIX: No market_data available, cannot convert 'hold' to action")

            # 7. Decision (🎯 #4: Use symbol-specific threshold)
            symbol_threshold = self.get_confidence_threshold(symbol)
            should_trade = adjusted_confidence >= symbol_threshold

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
            if abs(sentiment_modifier) > 0.01:  # 🎯 #7: Log sentiment impact
                reasoning_parts.append(f"Market sentiment: {sentiment_modifier:+.1%}")
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
                'symbol_win_rate': perf['wins'] / (perf['wins'] + perf['losses']) if perf and (perf['wins'] + perf['losses']) > 0 else None,
                # 🔧 CRITICAL: Return modified action/side from ML FIX
                'action': ai_analysis.get('action'),
                'side': ai_analysis.get('side')
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

    async def _learn_from_snapshot(self, snapshot: Dict, is_winner: bool, timestamp: datetime):
        """
        🎯 SNAPSHOT-BASED LEARNING: Learn from numerical indicator values.

        This function analyzes actual indicator values (RSI, volume, support distance, etc.)
        to discover winning conditions like "RSI < 30 → 85% win rate" or
        "Volume ratio > 2.0 → 70% win rate".
        """
        try:
            # If snapshot is a JSON string (from database), parse it
            if isinstance(snapshot, str):
                import json
                try:
                    snapshot = json.loads(snapshot)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"Failed to parse snapshot JSON: {e}")
                    return

            if not snapshot or not isinstance(snapshot, dict):
                return

            if 'indicators' not in snapshot:
                return

            indicators = snapshot.get('indicators', {})
            volume = snapshot.get('volume', {})
            levels = snapshot.get('levels', {})
            order_book = snapshot.get('order_book')
            funding = snapshot.get('funding')
            market_regime = snapshot.get('market_regime', {})

            # Record indicator values with outcome
            indicator_data = {
                'rsi': indicators.get('rsi'),
                'rsi_category': indicators.get('rsi_category'),
                'volume_ratio': volume.get('volume_ratio'),
                'volume_category': volume.get('volume_category'),
                'support_distance_percent': levels.get('support_distance_percent'),
                'resistance_distance_percent': levels.get('resistance_distance_percent'),
                'macd_histogram': indicators.get('macd_histogram'),
                'bb_position': indicators.get('bb_position'),
                'bb_width_percent': indicators.get('bb_width_percent'),
                'adx': indicators.get('adx'),
                'adx_trend_strength': indicators.get('adx_trend_strength'),
                'order_book_imbalance': order_book.get('imbalance') if order_book else None,
                'funding_rate': funding.get('rate') if funding else None,
                'funding_category': funding.get('category') if funding else None,
                'market_regime': market_regime.get('combined'),
                'trend_direction': snapshot.get('trend', {}).get('overall_direction')
            }

            # Store this snapshot's indicators with outcome
            for indicator_name, value in indicator_data.items():
                if value is not None:
                    self.indicator_win_conditions[indicator_name].append((value, is_winner, timestamp))

            # Periodically update learned rules (every 10 trades)
            if self.total_trades_analyzed % 10 == 0:
                await self._update_learned_indicator_rules()

        except Exception as e:
            logger.debug(f"Failed to learn from snapshot: {e}")

    async def _update_learned_indicator_rules(self):
        """
        🎯 Analyze collected indicator data to discover winning patterns.

        Uses statistical analysis to find indicator ranges/values that correlate with wins.
        """
        try:
            logger.info("🧠 Updating learned indicator rules from snapshots...")

            # RSI Learning
            if 'rsi' in self.indicator_win_conditions:
                rsi_data = self.indicator_win_conditions['rsi']
                if len(rsi_data) >= 30:  # Need at least 30 samples
                    winning_rsis = [v for v, won, t in rsi_data if won]
                    losing_rsis = [v for v, won, t in rsi_data if not won]

                    if winning_rsis:
                        # Find winning RSI ranges
                        win_mean = np.mean(winning_rsis)
                        win_std = np.std(winning_rsis)
                        self.learned_indicator_rules['rsi']['winning_ranges'] = [
                            {'min': win_mean - win_std, 'max': win_mean + win_std, 'samples': len(winning_rsis)}
                        ]
                        logger.info(f"   📊 RSI winning range: {win_mean - win_std:.1f} - {win_mean + win_std:.1f}")

            # Volume Ratio Learning
            if 'volume_ratio' in self.indicator_win_conditions:
                vol_data = self.indicator_win_conditions['volume_ratio']
                if len(vol_data) >= 30:
                    winning_vols = [v for v, won, t in vol_data if won]
                    losing_vols = [v for v, won, t in vol_data if not won]

                    if winning_vols:
                        win_median = np.median(winning_vols)
                        logger.info(f"   📊 Volume ratio winning median: {win_median:.2f}x")

            # Support Distance Learning (how close to support level)
            if 'support_distance_percent' in self.indicator_win_conditions:
                support_data = [(v, won, t) for v, won, t in self.indicator_win_conditions['support_distance_percent'] if v is not None]
                if len(support_data) >= 20:
                    winning_distances = [v for v, won, t in support_data if won]
                    if winning_distances:
                        # Trades near support (<5% away) tend to win?
                        near_support_wins = len([d for d in winning_distances if d < 5.0])
                        near_support_total = len([d for d, w, t in support_data if d < 5.0])
                        if near_support_total > 0:
                            near_support_wr = near_support_wins / near_support_total
                            logger.info(f"   🎯 Near support (<5%) win rate: {near_support_wr:.1%}")

            # Bollinger Band Position Learning (categorical)
            if 'bb_position' in self.indicator_win_conditions:
                bb_data = self.indicator_win_conditions['bb_position']
                if len(bb_data) >= 20:
                    bb_win_counts = defaultdict(lambda: {'wins': 0, 'total': 0})
                    for position, won, t in bb_data:
                        bb_win_counts[position]['total'] += 1
                        if won:
                            bb_win_counts[position]['wins'] += 1

                    # Find best BB positions
                    for position, stats in bb_win_counts.items():
                        if stats['total'] >= 5:
                            wr = stats['wins'] / stats['total']
                            logger.info(f"   📊 BB {position}: {wr:.1%} WR ({stats['wins']}/{stats['total']})")

            # Order Book Imbalance Learning
            if 'order_book_imbalance' in self.indicator_win_conditions:
                ob_data = [(v, won, t) for v, won, t in self.indicator_win_conditions['order_book_imbalance'] if v is not None]
                if len(ob_data) >= 20:
                    winning_imbalances = [v for v, won, t in ob_data if won]
                    if winning_imbalances:
                        win_mean = np.mean(winning_imbalances)
                        logger.info(f"   📊 Order book imbalance winning mean: {win_mean:+.3f}")

            logger.info("✅ Indicator rules updated successfully")

        except Exception as e:
            logger.error(f"❌ Failed to update indicator rules: {e}", exc_info=True)

    def get_snapshot_based_prediction(self, snapshot: Dict) -> Dict[str, Any]:
        """
        🎯 Get ML prediction based on current snapshot indicators.

        Returns confidence boost/penalty based on learned indicator patterns.
        """
        try:
            if not snapshot or 'indicators' not in snapshot:
                return {'confidence_adjustment': 0.0, 'reasons': []}

            indicators = snapshot.get('indicators', {})
            volume = snapshot.get('volume', {})
            levels = snapshot.get('levels', {})

            confidence_adjustment = 0.0
            reasons = []

            # RSI-based adjustment
            rsi = indicators.get('rsi')
            if rsi and 'rsi' in self.learned_indicator_rules:
                winning_ranges = self.learned_indicator_rules['rsi'].get('winning_ranges', [])
                for range_data in winning_ranges:
                    if range_data['min'] <= rsi <= range_data['max']:
                        if range_data['samples'] >= 20:  # Only trust with enough samples
                            boost = 0.05
                            confidence_adjustment += boost
                            reasons.append(f"RSI {rsi:.0f} in winning range (+{boost:.1%})")
                            break

            # Volume-based adjustment
            volume_ratio = volume.get('volume_ratio')
            if volume_ratio:
                # High volume (>2x) often indicates strong moves
                if volume_ratio >= 2.0:
                    boost = 0.03
                    confidence_adjustment += boost
                    reasons.append(f"High volume {volume_ratio:.1f}x (+{boost:.1%})")

            # Support distance adjustment
            support_dist = levels.get('support_distance_percent')
            if support_dist and support_dist < 5.0:
                # Very close to support = potential bounce
                boost = 0.04
                confidence_adjustment += boost
                reasons.append(f"Near support {support_dist:.1f}% (+{boost:.1%})")

            # Resistance distance adjustment
            resistance_dist = levels.get('resistance_distance_percent')
            if resistance_dist and resistance_dist < 3.0:
                # Very close to resistance = potential rejection
                penalty = -0.04
                confidence_adjustment += penalty
                reasons.append(f"Near resistance {resistance_dist:.1f}% ({penalty:.1%})")

            # Bollinger Band position
            bb_position = indicators.get('bb_position')
            if bb_position == 'below_lower':
                # Oversold condition
                boost = 0.05
                confidence_adjustment += boost
                reasons.append(f"BB oversold (+{boost:.1%})")
            elif bb_position == 'above_upper':
                # Overbought condition
                penalty = -0.03
                confidence_adjustment += penalty
                reasons.append(f"BB overbought ({penalty:.1%})")

            return {
                'confidence_adjustment': round(confidence_adjustment, 3),
                'reasons': reasons,
                'total_snapshots_learned': sum(len(v) for v in self.indicator_win_conditions.values())
            }

        except Exception as e:
            logger.error(f"❌ Snapshot prediction failed: {e}")
            return {'confidence_adjustment': 0.0, 'reasons': []}


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
