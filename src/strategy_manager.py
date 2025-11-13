"""
Multi-Strategy Trading System

Implements multiple trading strategies and dynamically selects
the best strategy based on market regime.

Strategies:
1. Trend Following - Best in TRENDING markets
2. Mean Reversion - Best in RANGING markets
3. Breakout - Best in COMPRESSION/CONSOLIDATION
4. Momentum - Best in HIGH VOLATILITY

Expected Impact: +$2,000 improvement per 1680 trades
"""

from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from enum import Enum
import asyncio

from src.utils import setup_logging
from src.market_regime_detector import get_regime_detector, MarketRegime

logger = setup_logging()


class StrategyType(Enum):
    """Available trading strategies."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"


class TradingStrategy:
    """Base class for all trading strategies."""

    def __init__(self, name: str):
        self.name = name
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = Decimal("0.0")
        self.trade_count = 0

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market and generate trading signal.

        Returns:
            Dict with 'action', 'confidence', 'reasoning'
        """
        raise NotImplementedError("Strategy must implement analyze()")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        win_rate = (self.win_count / self.trade_count) if self.trade_count > 0 else 0.0
        avg_pnl = (self.total_pnl / self.trade_count) if self.trade_count > 0 else Decimal("0.0")

        return {
            'name': self.name,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': float(win_rate),
            'total_pnl': float(self.total_pnl),
            'avg_pnl': float(avg_pnl)
        }

    def record_trade(self, is_winner: bool, pnl: Decimal):
        """Record trade result for strategy performance tracking."""
        self.trade_count += 1
        if is_winner:
            self.win_count += 1
        else:
            self.loss_count += 1
        self.total_pnl += pnl


class TrendFollowingStrategy(TradingStrategy):
    """
    Trend Following Strategy

    Best for: TRENDING markets
    Logic: Ride strong trends, use moving average crossovers
    Entry: When price breaks above EMA50 (LONG) or below (SHORT)
    Exit: When trend weakens (MACD reversal)
    """

    def __init__(self):
        super().__init__("Trend Following")

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market for trend following opportunities."""
        try:
            indicators_1h = market_data.get('indicators', {}).get('1h', {})
            indicators_4h = market_data.get('indicators', {}).get('4h', {})

            rsi_1h = indicators_1h.get('rsi', 50)
            rsi_4h = indicators_4h.get('rsi', 50)
            macd_1h = indicators_1h.get('macd', 0)
            macd_signal_1h = indicators_1h.get('macd_signal', 0)
            trend_4h = indicators_4h.get('trend', 'neutral')

            current_price = market_data.get('current_price', 0)

            # Multi-timeframe analysis
            multi_tf = market_data.get('multi_timeframe', {})
            ema50_1h = multi_tf.get('ema50_analysis', {}).get('ema50_1h', current_price)
            ema50_4h = multi_tf.get('ema50_analysis', {}).get('ema50_4h', current_price)
            above_1h = current_price > ema50_1h
            above_4h = current_price > ema50_4h

            confluence_count = 0
            reasons = []

            # Bullish trend factors
            if trend_4h == 'uptrend':
                confluence_count += 2
                reasons.append("4h uptrend")

            if above_1h and above_4h:
                confluence_count += 2
                reasons.append("Above EMA50 on 1h & 4h")

            if rsi_4h > 50 and rsi_4h < 70:
                confluence_count += 1
                reasons.append(f"4h RSI healthy ({rsi_4h:.0f})")

            if macd_1h > macd_signal_1h:
                confluence_count += 1
                reasons.append("MACD bullish crossover")

            # Generate signal
            if confluence_count >= 4:
                return {
                    'action': 'buy',
                    'confidence': 0.80 + (confluence_count - 4) * 0.05,
                    'side': 'LONG',
                    'strategy': self.name,
                    'confluence_count': confluence_count,
                    'reasoning': f"TREND FOLLOWING: {' | '.join(reasons)}"
                }

            # Bearish trend factors
            confluence_count = 0
            reasons = []

            if trend_4h == 'downtrend':
                confluence_count += 2
                reasons.append("4h downtrend")

            if not above_1h and not above_4h:
                confluence_count += 2
                reasons.append("Below EMA50 on 1h & 4h")

            if rsi_4h < 50 and rsi_4h > 30:
                confluence_count += 1
                reasons.append(f"4h RSI healthy ({rsi_4h:.0f})")

            if macd_1h < macd_signal_1h:
                confluence_count += 1
                reasons.append("MACD bearish crossover")

            if confluence_count >= 4:
                return {
                    'action': 'sell',
                    'confidence': 0.80 + (confluence_count - 4) * 0.05,
                    'side': 'SHORT',
                    'strategy': self.name,
                    'confluence_count': confluence_count,
                    'reasoning': f"TREND FOLLOWING: {' | '.join(reasons)}"
                }

            # No clear trend
            return {
                'action': 'hold',
                'confidence': 0.0,
                'side': None,
                'strategy': self.name,
                'confluence_count': 0,
                'reasoning': "TREND FOLLOWING: No clear trend signal"
            }

        except Exception as e:
            logger.error(f"Trend following analysis error: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'side': None,
                'strategy': self.name,
                'reasoning': f"Error: {e}"
            }


class MeanReversionStrategy(TradingStrategy):
    """
    Mean Reversion Strategy

    Best for: RANGING markets
    Logic: Buy oversold, sell overbought within range
    Entry: RSI < 30 (LONG) or RSI > 70 (SHORT) near range boundaries
    Exit: Return to mean (RSI ~50)
    """

    def __init__(self):
        super().__init__("Mean Reversion")

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market for mean reversion opportunities."""
        try:
            indicators_15m = market_data.get('indicators', {}).get('15m', {})
            indicators_1h = market_data.get('indicators', {}).get('1h', {})

            rsi_15m = indicators_15m.get('rsi', 50)
            rsi_1h = indicators_1h.get('rsi', 50)

            bb_upper = indicators_15m.get('bb_upper', 0)
            bb_lower = indicators_15m.get('bb_lower', 0)
            current_price = market_data.get('current_price', 0)

            # Check if in ranging market
            market_regime = market_data.get('market_regime', 'UNKNOWN')

            if market_regime != 'RANGING':
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'side': None,
                    'strategy': self.name,
                    'reasoning': f"MEAN REVERSION: Not in ranging market ({market_regime})"
                }

            confluence_count = 0
            reasons = []

            # Oversold conditions (BUY)
            if rsi_15m < 30:
                confluence_count += 2
                reasons.append(f"15m RSI oversold ({rsi_15m:.0f})")

            if rsi_1h < 40:
                confluence_count += 1
                reasons.append(f"1h RSI oversold ({rsi_1h:.0f})")

            if current_price <= bb_lower:
                confluence_count += 2
                reasons.append("Price at BB lower")

            if confluence_count >= 3:
                return {
                    'action': 'buy',
                    'confidence': 0.75 + (confluence_count - 3) * 0.05,
                    'side': 'LONG',
                    'strategy': self.name,
                    'confluence_count': confluence_count,
                    'reasoning': f"MEAN REVERSION: {' | '.join(reasons)}"
                }

            # Overbought conditions (SELL)
            confluence_count = 0
            reasons = []

            if rsi_15m > 70:
                confluence_count += 2
                reasons.append(f"15m RSI overbought ({rsi_15m:.0f})")

            if rsi_1h > 60:
                confluence_count += 1
                reasons.append(f"1h RSI overbought ({rsi_1h:.0f})")

            if current_price >= bb_upper:
                confluence_count += 2
                reasons.append("Price at BB upper")

            if confluence_count >= 3:
                return {
                    'action': 'sell',
                    'confidence': 0.75 + (confluence_count - 3) * 0.05,
                    'side': 'SHORT',
                    'strategy': self.name,
                    'confluence_count': confluence_count,
                    'reasoning': f"MEAN REVERSION: {' | '.join(reasons)}"
                }

            return {
                'action': 'hold',
                'confidence': 0.0,
                'side': None,
                'strategy': self.name,
                'reasoning': "MEAN REVERSION: No extreme found"
            }

        except Exception as e:
            logger.error(f"Mean reversion analysis error: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'side': None,
                'strategy': self.name,
                'reasoning': f"Error: {e}"
            }


class BreakoutStrategy(TradingStrategy):
    """
    Breakout Strategy

    Best for: CONSOLIDATION/COMPRESSION markets
    Logic: Trade breakouts from consolidation zones
    Entry: Volume spike + price breaks resistance/support
    Exit: Momentum exhaustion
    """

    def __init__(self):
        super().__init__("Breakout")

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market for breakout opportunities."""
        try:
            indicators_15m = market_data.get('indicators', {}).get('15m', {})
            indicators_1h = market_data.get('indicators', {}).get('1h', {})

            volume_trend_15m = indicators_15m.get('volume_trend', 'neutral')
            volume_trend_1h = indicators_1h.get('volume_trend', 'neutral')

            # Volatility analysis
            volatility = market_data.get('volatility', {})
            breakout_detected = volatility.get('breakout_detected', False)
            volatility_level = volatility.get('volatility_level', 'NORMAL')

            # Support/Resistance
            support_resistance = market_data.get('support_resistance', {})
            near_resistance = support_resistance.get('resistance_distance_pct', 100) < 1.0
            near_support = support_resistance.get('support_distance_pct', 100) < 1.0

            current_price = market_data.get('current_price', 0)
            rsi_15m = indicators_15m.get('rsi', 50)

            confluence_count = 0
            reasons = []

            # Bullish breakout
            if breakout_detected:
                confluence_count += 3
                reasons.append("Volatility breakout detected")

            if volume_trend_15m == 'increasing' or volume_trend_1h == 'increasing':
                confluence_count += 2
                reasons.append("Volume spike")

            if near_resistance:
                confluence_count += 1
                reasons.append("Near resistance (breakout imminent)")

            if rsi_15m > 50 and rsi_15m < 70:
                confluence_count += 1
                reasons.append(f"RSI momentum ({rsi_15m:.0f})")

            if confluence_count >= 4:
                return {
                    'action': 'buy',
                    'confidence': 0.85 + (confluence_count - 4) * 0.03,
                    'side': 'LONG',
                    'strategy': self.name,
                    'confluence_count': confluence_count,
                    'reasoning': f"BREAKOUT: {' | '.join(reasons)}"
                }

            # Bearish breakdown
            confluence_count = 0
            reasons = []

            if breakout_detected:
                confluence_count += 3
                reasons.append("Volatility breakdown detected")

            if volume_trend_15m == 'increasing' or volume_trend_1h == 'increasing':
                confluence_count += 2
                reasons.append("Volume spike")

            if near_support:
                confluence_count += 1
                reasons.append("Near support (breakdown imminent)")

            if rsi_15m < 50 and rsi_15m > 30:
                confluence_count += 1
                reasons.append(f"RSI momentum ({rsi_15m:.0f})")

            if confluence_count >= 4:
                return {
                    'action': 'sell',
                    'confidence': 0.85 + (confluence_count - 4) * 0.03,
                    'side': 'SHORT',
                    'strategy': self.name,
                    'confluence_count': confluence_count,
                    'reasoning': f"BREAKOUT: {' | '.join(reasons)}"
                }

            return {
                'action': 'hold',
                'confidence': 0.0,
                'side': None,
                'strategy': self.name,
                'reasoning': "BREAKOUT: No breakout detected"
            }

        except Exception as e:
            logger.error(f"Breakout analysis error: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'side': None,
                'strategy': self.name,
                'reasoning': f"Error: {e}"
            }


class MomentumStrategy(TradingStrategy):
    """
    Momentum Strategy

    Best for: HIGH VOLATILITY markets
    Logic: Ride strong momentum moves
    Entry: Accelerating ROC + strong MACD
    Exit: Momentum deceleration
    """

    def __init__(self):
        super().__init__("Momentum")

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market for momentum opportunities."""
        try:
            # Momentum analysis
            momentum = market_data.get('momentum', {})
            momentum_direction = momentum.get('momentum_direction', 'neutral')
            momentum_strength = momentum.get('momentum_strength', 0)
            is_accelerating = momentum.get('is_accelerating', False)

            roc_1h = momentum.get('roc_1h', 0)
            roc_4h = momentum.get('roc_4h', 0)

            # MACD
            indicators_1h = market_data.get('indicators', {}).get('1h', {})
            macd = indicators_1h.get('macd', 0)
            macd_signal = indicators_1h.get('macd_signal', 0)

            confluence_count = 0
            reasons = []

            # Bullish momentum
            if momentum_direction == 'bullish' and is_accelerating:
                confluence_count += 3
                reasons.append(f"Accelerating bullish momentum ({momentum_strength:.0f})")

            if roc_1h > 2.0:
                confluence_count += 2
                reasons.append(f"Strong 1h ROC ({roc_1h:.1f}%)")

            if roc_4h > 3.0:
                confluence_count += 1
                reasons.append(f"Strong 4h ROC ({roc_4h:.1f}%)")

            if macd > macd_signal and macd > 0:
                confluence_count += 1
                reasons.append("MACD bullish")

            if confluence_count >= 4:
                return {
                    'action': 'buy',
                    'confidence': 0.82 + (confluence_count - 4) * 0.04,
                    'side': 'LONG',
                    'strategy': self.name,
                    'confluence_count': confluence_count,
                    'reasoning': f"MOMENTUM: {' | '.join(reasons)}"
                }

            # Bearish momentum
            confluence_count = 0
            reasons = []

            if momentum_direction == 'bearish' and is_accelerating:
                confluence_count += 3
                reasons.append(f"Accelerating bearish momentum ({momentum_strength:.0f})")

            if roc_1h < -2.0:
                confluence_count += 2
                reasons.append(f"Strong 1h ROC ({roc_1h:.1f}%)")

            if roc_4h < -3.0:
                confluence_count += 1
                reasons.append(f"Strong 4h ROC ({roc_4h:.1f}%)")

            if macd < macd_signal and macd < 0:
                confluence_count += 1
                reasons.append("MACD bearish")

            if confluence_count >= 4:
                return {
                    'action': 'sell',
                    'confidence': 0.82 + (confluence_count - 4) * 0.04,
                    'side': 'SHORT',
                    'strategy': self.name,
                    'confluence_count': confluence_count,
                    'reasoning': f"MOMENTUM: {' | '.join(reasons)}"
                }

            return {
                'action': 'hold',
                'confidence': 0.0,
                'side': None,
                'strategy': self.name,
                'reasoning': "MOMENTUM: No strong momentum detected"
            }

        except Exception as e:
            logger.error(f"Momentum analysis error: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'side': None,
                'strategy': self.name,
                'reasoning': f"Error: {e}"
            }


class StrategyManager:
    """
    Multi-Strategy Manager

    Dynamically selects best strategy based on market regime.
    Tracks performance of each strategy.
    """

    def __init__(self):
        # Initialize strategies
        self.strategies = {
            StrategyType.TREND_FOLLOWING: TrendFollowingStrategy(),
            StrategyType.MEAN_REVERSION: MeanReversionStrategy(),
            StrategyType.BREAKOUT: BreakoutStrategy(),
            StrategyType.MOMENTUM: MomentumStrategy()
        }

        # Strategy selection rules (regime -> strategy priority)
        # 🔧 FIXED: Use correct MarketRegime enum names from market_regime_detector.py
        self.regime_strategy_map = {
            MarketRegime.STRONG_BULLISH_TREND: [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM],
            MarketRegime.STRONG_BEARISH_TREND: [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM],
            MarketRegime.WEAK_TREND: [StrategyType.MEAN_REVERSION, StrategyType.TREND_FOLLOWING],
            MarketRegime.RANGING: [StrategyType.MEAN_REVERSION, StrategyType.BREAKOUT],
            MarketRegime.HIGH_VOLATILITY: [StrategyType.MOMENTUM, StrategyType.BREAKOUT]
        }

    def select_best_strategy(self, market_regime: str) -> List[StrategyType]:
        """
        Select best strategies for current market regime.

        Returns list of strategies in priority order.
        """
        try:
            # Map old indicator regime strings to new MarketRegime enum
            regime_mapping = {
                'TRENDING': MarketRegime.WEAK_TREND,  # Generic trend, use weak trend strategies
                'RANGING': MarketRegime.RANGING,
                'VOLATILE': MarketRegime.HIGH_VOLATILITY,
                'UNKNOWN': MarketRegime.WEAK_TREND,
                # New enum values (direct mapping)
                'strong_bullish_trend': MarketRegime.STRONG_BULLISH_TREND,
                'strong_bearish_trend': MarketRegime.STRONG_BEARISH_TREND,
                'weak_trend': MarketRegime.WEAK_TREND,
                'ranging': MarketRegime.RANGING,
                'high_volatility': MarketRegime.HIGH_VOLATILITY
            }

            # Try direct string mapping first
            regime_key = market_regime.upper() if market_regime.isupper() else market_regime.lower()
            if regime_key in regime_mapping:
                regime_enum = regime_mapping[regime_key]
            else:
                # Try enum parsing as fallback
                regime_enum = MarketRegime(market_regime.lower())

            return self.regime_strategy_map.get(
                regime_enum,
                [StrategyType.TREND_FOLLOWING]  # Default fallback
            )
        except (ValueError, AttributeError, KeyError):
            logger.warning(f"Unknown market regime: {market_regime}, using default")
            return [StrategyType.TREND_FOLLOWING]

    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market using best strategy for current regime.

        Returns:
            Combined analysis with best signal
        """
        try:
            # Detect market regime
            market_regime = market_data.get('market_regime', 'UNKNOWN')

            logger.debug(f"📊 Strategy selection for regime: {market_regime}")

            # Get strategies for this regime
            strategy_priorities = self.select_best_strategy(market_regime)

            # Run all priority strategies
            strategy_signals = []
            for strategy_type in strategy_priorities:
                strategy = self.strategies[strategy_type]
                signal = strategy.analyze(market_data)
                strategy_signals.append(signal)

                logger.debug(
                    f"  {strategy.name}: {signal['action']} "
                    f"({signal.get('confidence', 0):.0%})"
                )

            # Select best signal (highest confidence, not 'hold')
            valid_signals = [s for s in strategy_signals if s['action'] != 'hold']

            if not valid_signals:
                # All strategies say hold
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'side': None,
                    'strategy': 'multi_strategy',
                    'reasoning': 'All strategies recommend hold',
                    'regime': market_regime,
                    'strategies_evaluated': len(strategy_signals)
                }

            # Pick best signal
            best_signal = max(valid_signals, key=lambda x: x.get('confidence', 0))

            # Add regime context
            best_signal['regime'] = market_regime
            best_signal['strategies_evaluated'] = len(strategy_signals)

            logger.info(
                f"✅ Best strategy: {best_signal['strategy']} | "
                f"{best_signal['action']} @ {best_signal.get('confidence', 0):.0%}"
            )

            return best_signal

        except Exception as e:
            logger.error(f"Strategy manager analysis failed: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'side': None,
                'strategy': 'error',
                'reasoning': f'Error: {e}',
                'regime': 'UNKNOWN'
            }

    def get_all_strategy_stats(self) -> Dict[str, Any]:
        """Get performance stats for all strategies."""
        return {
            strategy_type.value: strategy.get_performance_stats()
            for strategy_type, strategy in self.strategies.items()
        }

    def record_trade_result(
        self,
        strategy_name: str,
        is_winner: bool,
        pnl: Decimal
    ):
        """Record trade result for strategy performance tracking."""
        for strategy in self.strategies.values():
            if strategy.name == strategy_name:
                strategy.record_trade(is_winner, pnl)
                break


# Singleton instance
_strategy_manager: Optional[StrategyManager] = None


def get_strategy_manager() -> StrategyManager:
    """Get or create strategy manager instance."""
    global _strategy_manager
    if _strategy_manager is None:
        _strategy_manager = StrategyManager()
    return _strategy_manager
