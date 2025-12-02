"""
ENHANCED TRADING SYSTEM - Professional Integration Layer

This module integrates professional trading components with the existing system:
- Volume Profile Analysis
- Confluence Scoring (60+ threshold)
- Dynamic Profit Targets
- Advanced Trailing Stops
- Dynamic Position Sizing

MINIMAL DISRUPTION PHILOSOPHY:
Instead of rewriting existing code, this module wraps and enhances it.
The existing system continues to work, but with professional-grade filters and calculations.

USAGE IN EXISTING CODE:
-----------------------
from src.enhanced_trading_system import EnhancedTradingSystem

enhanced = EnhancedTradingSystem()

# After PA analysis:
evaluation = await enhanced.evaluate_trading_opportunity(
    symbol=symbol,
    side='LONG',
    pa_analysis=pa_result,
    market_data=market_data,
    indicators=indicators
)

if evaluation['should_trade']:
    # Use enhanced position sizing
    position_size = evaluation['position_size_usd']
    profit_targets = evaluation['profit_targets']
    # Execute trade...


"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from decimal import Decimal

# Import professional modules
from src.volume_profile_analyzer import VolumeProfileAnalyzer, analyze_volume_profile
from src.confluence_scoring import ConfluenceScorer, score_trade_opportunity
from src.dynamic_profit_targets import DynamicProfitTargetCalculator, calculate_dynamic_targets
from src.advanced_trailing_stop import AdvancedTrailingStop, calculate_trailing_stop
from src.dynamic_position_sizer import DynamicPositionSizer, calculate_position_size
from src.market_regime_detector import get_regime_detector
# 🆕 v4.4.0: Import enhanced indicators for professional confluence scoring
# 🆕 v4.5.0: Import advanced indicators (VWAP, StochRSI, CMF, Fibonacci)
from src.indicators import calculate_enhanced_indicators, calculate_advanced_indicators

logger = logging.getLogger('trading_bot')


class EnhancedTradingSystem:
    """
    Professional trading system integration layer.

    Enhances existing trading logic with professional-grade components
    while maintaining compatibility with current codebase.
    """

    def __init__(self):
        # Initialize professional components
        self.volume_profile_analyzer = VolumeProfileAnalyzer(price_bins=50)
        self.confluence_scorer = ConfluenceScorer()
        self.profit_target_calculator = DynamicProfitTargetCalculator()
        self.trailing_stop_system = AdvancedTrailingStop()
        self.position_sizer = DynamicPositionSizer(base_risk_pct=0.02)
        self.regime_detector = get_regime_detector()

        # Configuration - Read from settings/environment!
        from src.config import get_settings
        settings = get_settings()

        # 🎯 v4.3.1: Use MIN_CONFLUENCE_SCORE from environment (default 75)
        self.min_confluence_score = settings.min_confluence_score  # From env!
        self.enable_volume_profile = True
        self.enable_confluence_filtering = True
        self.enable_dynamic_sizing = True

        logger.info("= Enhanced Trading System initialized")
        logger.info(f"   = Volume Profile: {'ENABLED' if self.enable_volume_profile else 'DISABLED'}")
        logger.info(f"   < Confluence Filtering (min {self.min_confluence_score}): {'ENABLED' if self.enable_confluence_filtering else 'DISABLED'}")
        logger.info(f"   = Dynamic Sizing: {'ENABLED' if self.enable_dynamic_sizing else 'DISABLED'}")

    async def evaluate_trading_opportunity(
        self,
        symbol: str,
        side: str,
        pa_analysis: Dict,
        market_data: Dict,
        indicators: Dict,
        account_balance: float,
        leverage: int,
        account_high_water_mark: Optional[float] = None,
        recent_win_rate: Optional[float] = None,
        recent_avg_rr: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Main evaluation method: Enhance PA analysis with professional components.

        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            pa_analysis: Price action analysis from existing system
            market_data: Market data dict with OHLCV
            indicators: Technical indicators
            account_balance: Current account balance
            leverage: Leverage multiplier
            account_high_water_mark: Peak balance (for drawdown calc)
            recent_win_rate: Recent win rate (for Kelly)
            recent_avg_rr: Recent avg R/R (for Kelly)

        Returns:
            Dict with:
                - should_trade: bool
                - confluence_score: float
                - quality: str
                - volume_profile_data: Dict
                - profit_targets: Dict
                - position_size_data: Dict
                - reasoning: str
        """
        try:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"= ENHANCED EVALUATION: {symbol} {side}")
            logger.info("=" * 80)

            # Step 1: Volume Profile Analysis
            volume_profile_data = await self._analyze_volume_profile(symbol, market_data)

            # Step 2: Market Regime Detection
            regime_data = await self._detect_market_regime(symbol, market_data, indicators)

            # Step 3: Multi-Timeframe Data (if available)
            mtf_data = market_data.get('mtf', None)

            # 🆕 v4.4.0: Get OHLCV data for enhanced indicators
            ohlcv_data = market_data.get('ohlcv_15m', [])

            # Step 4: Confluence Scoring with enhanced indicators
            confluence_result = self._score_confluence(
                symbol=symbol,
                side=side,
                pa_analysis=pa_analysis,
                indicators=indicators,
                volume_profile=volume_profile_data,
                market_regime=regime_data,
                mtf_data=mtf_data,
                ohlcv_data=ohlcv_data  # 🆕 v4.4.0: Pass OHLCV for enhanced indicators
            )

            # Step 5: Decision - Should we trade?
            should_trade = confluence_result['should_trade']
            confluence_score = confluence_result['total_score']
            quality = confluence_result['quality']

            if not should_trade:
                logger.warning(f"= {symbol} {side} - Confluence score {confluence_score:.1f} below minimum {self.min_confluence_score}")
                return {
                    'should_trade': False,
                    'confluence_score': confluence_score,
                    'quality': quality,
                    'reasoning': confluence_result['reasoning'],
                    'rejection_reason': f'Confluence score ({confluence_score:.1f}) below threshold ({self.min_confluence_score})'
                }

            logger.info(f" {symbol} {side} - Confluence score {confluence_score:.1f} PASSED (quality: {quality})")

            # Step 6: Dynamic Profit Targets
            entry_price = pa_analysis.get('entry_price', indicators.get('close', 0))
            stop_loss_price = pa_analysis.get('stop_loss', entry_price * 0.99)  # Fallback 1% SL
            atr = indicators.get('atr', entry_price * 0.02)  # Fallback 2% ATR

            # Get resistance/support for target calculation
            resistance_price = None
            support_price = None
            if side == 'LONG':
                resistance_level = pa_analysis.get('resistance_level', {})
                resistance_price = resistance_level.get('price', None)
            else:
                support_level = pa_analysis.get('support_level', {})
                support_price = support_level.get('price', None)

            profit_targets = self._calculate_profit_targets(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                side=side,
                atr=atr,
                leverage=leverage,
                position_value_usd=account_balance * 0.02,  # Base 2% position
                resistance_price=resistance_price,
                support_price=support_price,
                market_regime=regime_data.get('regime', None),
                volatility_level=regime_data.get('volatility_level', None)
            )

            # Step 7: Dynamic Position Sizing
            position_size_data = self._calculate_position_size(
                account_balance=account_balance,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                leverage=leverage,
                confluence_quality=quality,
                confluence_score=confluence_score,
                account_high_water_mark=account_high_water_mark,
                recent_win_rate=recent_win_rate,
                recent_avg_rr=recent_avg_rr
            )

            # Step 8: Compile final result
            result = {
                'should_trade': True,
                'symbol': symbol,
                'side': side,

                # Confluence data
                'confluence_score': confluence_score,
                'quality': quality,
                'confluence_reasoning': confluence_result['reasoning'],
                'component_scores': confluence_result['component_scores'],

                # Volume profile data
                'volume_profile_data': volume_profile_data,

                # Market regime data
                'market_regime_data': regime_data,

                # Profit targets
                'profit_targets': profit_targets,
                'tp1_price': profit_targets['tp1_price'],
                'tp2_price': profit_targets['tp2_price'],
                'tp3_price': profit_targets['tp3_price'],
                'expected_rr_ratio': profit_targets['rr_ratio'],

                # Position sizing
                'position_size_data': position_size_data,
                'position_size_usd': position_size_data['position_size_usd'],
                'position_size_coins': position_size_data['position_size_coins'],
                'risk_amount_usd': position_size_data['risk_amount_usd'],
                'risk_percentage': position_size_data['risk_percentage'],

                # Entry/Exit
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,

                # Summary reasoning
                'reasoning': self._generate_summary_reasoning(
                    confluence_result, volume_profile_data, regime_data, profit_targets, position_size_data
                )
            }

            self._log_evaluation_result(result)

            return result

        except Exception as e:
            logger.error(f"L Error in enhanced evaluation for {symbol}: {e}", exc_info=True)
            return {
                'should_trade': False,
                'confluence_score': 0,
                'quality': 'ERROR',
                'reasoning': f'Evaluation error: {str(e)}',
                'rejection_reason': f'System error: {str(e)}'
            }

    async def _analyze_volume_profile(self, symbol: str, market_data: Dict) -> Dict:
        """Analyze volume profile for the symbol."""
        if not self.enable_volume_profile:
            return {}

        try:
            # Get OHLCV data
            ohlcv = market_data.get('ohlcv_15m', [])
            if len(ohlcv) < 20:
                logger.warning(f" {symbol} - Insufficient data for volume profile")
                return {}

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Analyze volume profile
            volume_profile = self.volume_profile_analyzer.analyze_volume_profile(df)

            return volume_profile

        except Exception as e:
            logger.warning(f" {symbol} - Volume profile analysis error: {e}")
            return {}

    async def _detect_market_regime(self, symbol: str, market_data: Dict, indicators: Dict) -> Dict:
        """Detect market regime."""
        try:
            ohlcv = market_data.get('ohlcv_15m', [])
            if len(ohlcv) < 50:
                return {'regime': None, 'should_trade': True, 'confidence': 0.5}

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Detect regime
            regime_data = self.regime_detector.detect_regime(indicators, symbol)

            return regime_data

        except Exception as e:
            logger.warning(f" {symbol} - Regime detection error: {e}")
            return {'regime': None, 'should_trade': True, 'confidence': 0.5}

    def _score_confluence(
        self,
        symbol: str,
        side: str,
        pa_analysis: Dict,
        indicators: Dict,
        volume_profile: Dict,
        market_regime: Dict,
        mtf_data: Optional[Dict],
        ohlcv_data: Optional[list] = None  # 🆕 v4.4.0: OHLCV for enhanced indicators
    ) -> Dict:
        """Score confluence of all signals."""
        if not self.enable_confluence_filtering:
            # Bypass - assume all trades are good
            return {
                'should_trade': True,
                'total_score': 80,
                'quality': 'GOOD',
                'reasoning': 'Confluence filtering disabled',
                'component_scores': {}
            }

        try:
            # 🆕 v4.4.0: Calculate enhanced indicators if OHLCV data available
            enhanced_data = None
            advanced_data = None

            if ohlcv_data and len(ohlcv_data) >= 50:
                try:
                    enhanced_data = calculate_enhanced_indicators(ohlcv_data)
                    logger.info(f"🔬 {symbol}: Enhanced indicators calculated successfully")
                except Exception as e:
                    logger.warning(f"⚠️ {symbol}: Enhanced indicators failed: {e}")
                    enhanced_data = None

                # 🆕 v4.5.0: Calculate advanced indicators
                try:
                    advanced_data = calculate_advanced_indicators(ohlcv_data)
                    logger.info(f"🔬 {symbol}: Advanced indicators (VWAP, StochRSI, CMF, Fib) calculated")
                except Exception as e:
                    logger.warning(f"⚠️ {symbol}: Advanced indicators failed: {e}")
                    advanced_data = None

            # Score the opportunity with enhanced + advanced data
            score_result = self.confluence_scorer.score_opportunity(
                symbol=symbol,
                side=side,
                pa_analysis=pa_analysis,
                indicators=indicators,
                volume_profile=volume_profile,
                market_regime=market_regime,
                mtf_data=mtf_data,
                enhanced_data=enhanced_data,  # v4.4.0: Enhanced indicators
                advanced_data=advanced_data   # 🆕 v4.5.0: Advanced indicators
            )

            return score_result

        except Exception as e:
            logger.error(f"L Confluence scoring error for {symbol}: {e}", exc_info=True)
            return {
                'should_trade': False,
                'total_score': 0,
                'quality': 'ERROR',
                'reasoning': f'Scoring error: {str(e)}',
                'component_scores': {}
            }

    def _calculate_profit_targets(self, **kwargs) -> Dict:
        """Calculate dynamic profit targets."""
        try:
            targets = self.profit_target_calculator.calculate_profit_targets(**kwargs)
            return targets
        except Exception as e:
            logger.error(f"L Profit target calculation error: {e}", exc_info=True)
            # Fallback targets
            entry = kwargs.get('entry_price', 0)
            stop = kwargs.get('stop_loss_price', 0)
            side = kwargs.get('side', 'LONG')
            risk = abs(entry - stop)

            if side == 'LONG':
                return {
                    'tp1_price': entry + risk * 2,
                    'tp2_price': entry + risk * 3,
                    'tp3_price': entry + risk * 4,
                    'rr_ratio': 2.8
                }
            else:
                return {
                    'tp1_price': entry - risk * 2,
                    'tp2_price': entry - risk * 3,
                    'tp3_price': entry - risk * 4,
                    'rr_ratio': 2.8
                }

    def _calculate_position_size(self, **kwargs) -> Dict:
        """Calculate dynamic position size."""
        if not self.enable_dynamic_sizing:
            # Use static sizing (2% risk)
            account_balance = kwargs.get('account_balance', 100)
            return {
                'position_size_usd': account_balance * 0.02,
                'risk_amount_usd': account_balance * 0.02,
                'risk_percentage': 2.0,
                'reasoning': 'Dynamic sizing disabled - using static 2%'
            }

        try:
            size_data = self.position_sizer.calculate_position_size(**kwargs)
            return size_data
        except Exception as e:
            logger.error(f"L Position sizing error: {e}", exc_info=True)
            account_balance = kwargs.get('account_balance', 100)
            return {
                'position_size_usd': account_balance * 0.02,
                'risk_amount_usd': account_balance * 0.02,
                'risk_percentage': 2.0,
                'reasoning': f'Sizing error - using fallback 2%: {str(e)}'
            }

    def _generate_summary_reasoning(
        self,
        confluence_result: Dict,
        volume_profile: Dict,
        regime_data: Dict,
        profit_targets: Dict,
        position_size_data: Dict
    ) -> str:
        """Generate comprehensive summary reasoning."""
        lines = [
            "=" * 60,
            "TRADE EVALUATION SUMMARY",
            "=" * 60,
            "",
            f" Confluence Score: {confluence_result.get('total_score', 0):.1f}/100 ({confluence_result.get('quality', 'UNKNOWN')})",
            "",
            "Key Factors:",
        ]

        # Volume Profile
        if volume_profile:
            vpoc = volume_profile.get('vpoc', 0)
            lines.append(f"  = VPOC: ${vpoc:.2f} (volume-based support/resistance)")

        # Market Regime
        regime = regime_data.get('regime', None)
        if regime:
            regime_name = regime.value if hasattr(regime, 'value') else str(regime)
            lines.append(f"  < Market Regime: {regime_name}")

        # Profit Targets
        rr_ratio = profit_targets.get('rr_ratio', 0)
        lines.append(f"  = Expected R/R: {rr_ratio:.2f}:1")

        # Position Size
        risk_pct = position_size_data.get('risk_percentage', 0)
        lines.append(f"  = Position Risk: {risk_pct:.2f}%")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _log_evaluation_result(self, result: Dict):
        """Log evaluation result summary."""
        symbol = result['symbol']
        side = result['side']
        should_trade = result['should_trade']
        score = result['confluence_score']

        emoji = "" if should_trade else "="

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"{emoji} ENHANCED EVALUATION RESULT: {symbol} {side}")
        logger.info("=" * 80)
        logger.info(f"Decision: {'TRADE APPROVED' if should_trade else 'TRADE REJECTED'}")
        logger.info(f"Score: {score:.1f}/100")

        if should_trade:
            logger.info(f"Position Size: ${result['position_size_usd']:.2f}")
            logger.info(f"Risk: ${result['risk_amount_usd']:.2f} ({result['risk_percentage']:.2f}%)")
            logger.info(f"TP1: ${result['tp1_price']:.2f}")
            logger.info(f"TP2: ${result['tp2_price']:.2f}")
            logger.info(f"TP3: ${result['tp3_price']:.2f}")

        logger.info("=" * 80)
        logger.info("")


# Global instance
_enhanced_system = None


def get_enhanced_trading_system() -> EnhancedTradingSystem:
    """Get or create enhanced trading system instance."""
    global _enhanced_system
    if _enhanced_system is None:
        _enhanced_system = EnhancedTradingSystem()
    return _enhanced_system
